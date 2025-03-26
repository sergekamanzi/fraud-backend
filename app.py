from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
import io
import os


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


# POST endpoint for retraining the model
@app.post("/retrain")
async def retrain_model(file: UploadFile = File(...)):
    try:
        # Read the uploaded CSV file
        contents = await file.read()
        new_data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Verify required columns are present
        required_columns = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                          'oldbalanceDest', 'newbalanceDest', 'isFraud']
        if not all(col in new_data.columns for col in required_columns):
            raise HTTPException(status_code=400, detail="Missing required columns in CSV file")

        # Preprocess the data similar to notebook
        # Drop unnecessary columns if they exist
        columns_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
        new_data = new_data.drop(columns=[col for col in columns_to_drop if col in new_data.columns])

        # Encode 'type' column if it's still categorical
        if new_data['type'].dtype == 'object':
            le = LabelEncoder()
            new_data['type'] = le.fit_transform(new_data['type'])

        # Prepare features and target
        X = new_data.drop('isFraud', axis=1)
        y = new_data['isFraud']

        # Scale numeric features
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        # Split the data (70% train, 15% val, 15% test)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Compute class weights
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        # Define Early Stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Retrain the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            class_weight=class_weight_dict,
            callbacks=[early_stopping],
            verbose=0
        )

        # Evaluate on test set
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        accuracy = float(np.mean(y_pred.flatten() == y_test.values))

        # Save the updated model and scaler
        model.save('fraud_detection_model.keras')
        joblib.dump(scaler, 'standard_scaler.pkl')

        return {
            "message": "Model retrained successfully",
            "test_accuracy": accuracy,
            "training_epochs": len(history.history['loss'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during retraining: {str(e)}")
    finally:
        await file.close()


# Run the app with: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    uvicorn.run(app, host="0.0.0.0", port=port)

