from ml_platform.data import load_data
from ml_platform.preprocessing import split_data, scale_data
from ml_platform.models import create_model, train_model, predict
from ml_platform.evaluation import evaluate_model
from ml_platform.config import DATA_PATH, TARGET_COLUMN

def run_pipeline():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(
        df, TARGET_COLUMN
    )

    X_train, X_test, scaler = scale_data(
        X_train, X_test
    )

    model = create_model()
    model = train_model(model, X_train, y_train)

    y_pred = predict(model, X_test)

    accuracy, matrix = evaluate_model(
        y_test, y_pred
    )

    print("ML PIPELINE COMPLETED")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(matrix)

    return model, scaler

if __name__ == "__main__":
    run_pipeline()
