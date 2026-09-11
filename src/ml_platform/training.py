import json
import joblib

from ml_platform.data import load_data
from ml_platform.preprocessing import split_data, scale_data
from ml_platform.models import create_model, train_model, predict
from ml_platform.evaluation import evaluate_model
from ml_platform.logger import get_logger


logger = get_logger("training")


def train_and_save(data_path, target_column, model_path):

    logger.info("Loading dataset")

    df = load_data(data_path)

    logger.info("Splitting data")

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column
    )

    logger.info("Scaling features")

    X_train, X_test, scaler = scale_data(
        X_train,
        X_test
    )

    logger.info("Training model")

    model = create_model()
    model = train_model(
        model,
        X_train,
        y_train
    )

    logger.info("Evaluating model")

    y_pred = predict(
        model,
        X_test
    )

    accuracy, matrix = evaluate_model(
        y_test,
        y_pred
    )

    logger.info("Model accuracy: %.2f", accuracy)

    logger.info("Saving model and scaler")

    joblib.dump(
        {
            "model": model,
            "scaler": scaler
        },
        model_path
    )

    metrics_path = "models/trained/metrics.json"

    metrics = {
        "accuracy": accuracy,
        "confusion_matrix": matrix.tolist()
    }

    with open(metrics_path, "w") as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    logger.info("Metrics saved to %s", metrics_path)
    logger.info("Training completed")

    return model, scaler, metrics