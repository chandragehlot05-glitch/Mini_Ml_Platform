import sys

sys.path.insert(0, "src")

from ml_platform.training import train_and_save
from ml_platform.config import DATA_PATH, TARGET_COLUMN


MODEL_PATH = "models/trained/loan_model.pkl"


print("=== ML Model Training ===")

try:
    train_and_save(
        DATA_PATH,
        TARGET_COLUMN,
        MODEL_PATH
    )

    print("\nTraining completed successfully!")
    print("Model saved to:", MODEL_PATH)

except Exception as e:
    print("\nTraining failed.")
    print("Error:", e)