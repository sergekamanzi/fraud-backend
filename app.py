from fastapi import FastAPI, HTTPException, UploadFile, File
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
from sklearn.metrics import confusion_matrix
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Helper function to convert plot to base64 string
def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    logger.info("Generated base64 image string of length: %d", len(image_base64))
    return image_base64

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

        # Generate correlation heatmap (excluding target column)
        corr_matrix = new_data.drop(columns=['isFraud']).corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Heatmap (Without Target Columns)')
        correlation_heatmap = plot_to_base64()
        logger.info("Correlation heatmap generated")

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

        # Compute confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        confusion_matrix_plot = plot_to_base64()
        logger.info("Confusion matrix generated")

        # Format the accuracy to 6 decimal places
        formatted_accuracy = round(accuracy, 6)

        # Save the updated model and scaler
        model.save('fraud_detection_model.keras')
        joblib.dump(scaler, 'standard_scaler.pkl')

        # Log the response being sent
        response = {
            "message": "Model retrained successfully",
            "test_accuracy": formatted_accuracy,
            "training_epochs": len(history.history['loss']),
            "correlation_heatmap": f"data:image/png;base64,{correlation_heatmap}",
            "confusion_matrix": f"data:image/png;base64,{confusion_matrix_plot}"
        }
        logger.info("Sending response: %s", response.keys())
        return response
    except Exception as e:
        logger.error("Error during retraining: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error during retraining: {str(e)}")
    finally:
        await file.close()

# Run the app with: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
