import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from ml_platform.prediction import load_saved_model, make_prediction


MODEL_PATH = "models/trained/loan_model.pkl"

model, scaler = load_saved_model(MODEL_PATH)


app = FastAPI(
    title="Mini ML Platform API",
    description="API for loan approval prediction",
    version="1.0.0"
)


class CustomerData(BaseModel):
    age: float
    income: float
    experience: float


@app.get("/")
def home():
    return {
        "message": "Mini ML Platform API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict_loan(customer: CustomerData):

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

    prediction = make_prediction(
        model,
        scaler,
        data
    )

    result = (
        "Loan Approved"
        if prediction[0] == 1
        else "Loan Not Approved"
    )

    return {
        "prediction": int(prediction[0]),
        "result": result
    }