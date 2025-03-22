from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API")

# Add CORS middleware to allow access from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

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

# Define numeric columns for scaling (same as in your Colab)
numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Fraud Detection API. Use /docs for API documentation."}

# POST endpoint for fraud prediction
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        # Convert input data to dictionary and then to DataFrame
        input_data = transaction.dict()
        user_input = pd.DataFrame([input_data], columns=['step', 'type', 'amount', 'oldbalanceOrg', 
                                                         'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])

        # Scale numeric columns
        user_input[numeric_cols] = scaler.transform(user_input[numeric_cols])

        # Make prediction
        prediction = (model.predict(user_input) > 0.5).astype("int32")[0, 0]

        # Return result
        result = "Fraud" if prediction == 1 else "No Fraud"
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Run the app with: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
