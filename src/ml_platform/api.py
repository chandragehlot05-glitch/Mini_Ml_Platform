import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml_platform.prediction import load_saved_model, make_prediction


MODEL_PATH = "models/trained/loan_model.pkl"


# Load model safely
try:
    model, scaler = load_saved_model(MODEL_PATH)
    model_load_error = None

except Exception as e:
    model = None
    scaler = None
    model_load_error = str(e)


# Create FastAPI application
app = FastAPI(
    title="Mini ML Platform API",
    description="API for loan approval prediction",
    version="1.0.0"
)


# Request data validation
class CustomerData(BaseModel):

    age: float = Field(
        ...,
        ge=0,
        le=120,
        description="Customer age in years"
    )

    income: float = Field(
        ...,
        ge=0,
        description="Annual income"
    )

    experience: float = Field(
        ...,
        ge=0,
        description="Years of work experience"
    )


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Mini ML Platform API is running"
    }


# Health check endpoint
@app.get("/health")
def health():

    if model is None or scaler is None:
        return {
            "status": "unhealthy",
            "model_loaded": False
        }

    return {
        "status": "healthy",
        "model_loaded": True
    }


# Prediction endpoint
@app.post("/predict")
def predict_loan(customer: CustomerData):

    # Check whether model is available
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="ML model is not available"
        )

    # Business validation
    if customer.experience > customer.age:
        raise HTTPException(
            status_code=400,
            detail="Experience cannot be greater than age"
        )

    try:

        # Create input DataFrame
        data = pd.DataFrame(
            [[
                customer.age,
                customer.income,
                customer.experience
            ]],
            columns=[
                "age",
                "income",
                "experience"
            ]
        )

        # Generate prediction
        prediction = make_prediction(
            model,
            scaler,
            data
        )

        # Convert prediction into readable result
        result = (
            "Loan Approved"
            if prediction[0] == 1
            else "Loan Not Approved"
        )

        return {
            "prediction": int(prediction[0]),
            "result": result
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )