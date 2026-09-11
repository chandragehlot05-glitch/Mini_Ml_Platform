
import sys
sys.path.insert(0, "src")

import pandas as pd

from ml_platform.prediction import load_saved_model, make_prediction

MODEL_PATH = "models/trained/loan_model.pkl"


def get_positive_number(message):
    while True:
        try:
            value = float(input(message))

            if value < 0:
                print("Please enter a positive value.")
                continue

            return value

        except ValueError:
            print("Invalid input. Please enter a number.")


print("=== ML Loan Prediction System ===")

age = get_positive_number("Enter age: ")
income = get_positive_number("Enter income: ")
experience = get_positive_number("Enter years of experience: ")

if age > 120:
    print("Invalid age. Please enter an age below 120.")
    sys.exit()

if experience > age:
    print("Invalid experience. Experience cannot be greater than age.")
    sys.exit()

new_customer = pd.DataFrame(
    [[age, income, experience]],
    columns=["age", "income", "experience"]
)

model, scaler = load_saved_model(MODEL_PATH)

prediction = make_prediction(
    model,
    scaler,
    new_customer
)

print("\nPrediction:", prediction[0])

if prediction[0] == 1:
    print("Result: Loan Approved")
else:
    print("Result: Loan Not Approved")

