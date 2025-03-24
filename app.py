from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
from typing import List
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import redis.asyncio as redis
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API")

# Add CORS middleware to allow access from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# Initialize Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Load the saved model and scaler
model = tf.keras.models.load_model('fraud_detection_model.keras')
scaler = joblib.load('standard_scaler.pkl')

# Define the input data schema using Pydantic
class Transaction(BaseModel):
    step: float
    type: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

# Define numeric columns for scaling
numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Fraud Detection API. Use /docs for API documentation."}

# Batch prediction endpoint
@app.post("/predict/batch")
@limiter.limit("100/minute")  # Adjust rate limit as needed
async def predict_fraud_batch(request: Request, transactions: List[Transaction]):
    try:
        # Convert list of transactions to DataFrame
        input_data = [t.dict() for t in transactions]
        user_input = pd.DataFrame(input_data, columns=['step', 'type', 'amount', 'oldbalanceOrg', 
                                                       'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])

        # Scale numeric columns
        user_input[numeric_cols] = scaler.transform(user_input[numeric_cols])

        # Make batch prediction
        predictions = (model.predict(user_input) > 0.5).astype("int32").flatten()
        results = ["Fraud" if pred == 1 else "No Fraud" for pred in predictions]

        # Cache results for identical transactions
        for transaction, result in zip(input_data, results):
            transaction_key = str(transaction)  # Use transaction dict as cache key
            await redis_client.setex(transaction_key, 3600, result)  # Cache for 1 hour

        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Single prediction endpoint with caching
@app.post("/predict")
@limiter.limit("100/minute")
async def predict_fraud(request: Request, transaction: Transaction):
    try:
        # Check cache first
        transaction_key = str(transaction.dict())
        cached_result = await redis_client.get(transaction_key)
        if cached_result:
            return {"prediction": cached_result}

        # Convert input data to DataFrame
        input_data = transaction.dict()
        user_input = pd.DataFrame([input_data], columns=['step', 'type', 'amount', 'oldbalanceOrg', 
                                                         'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])

        # Scale numeric columns
        user_input[numeric_cols] = scaler.transform(user_input[numeric_cols])

        # Make prediction
        prediction = (model.predict(user_input) > 0.5).astype("int32")[0, 0]
        result = "Fraud" if prediction == 1 else "No Fraud"

        # Cache the result
        await redis_client.setex(transaction_key, 3600, result)  # Cache for 1 hour

        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Shutdown event to close Redis connection
@app.on_event("shutdown")
async def shutdown_event():
    await redis_client.close()

# Run the app with: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)