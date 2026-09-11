import joblib
from ml_platform.logger import get_logger

logger = get_logger("prediction")


def load_saved_model(path):
    try:
        saved = joblib.load(path)

        model = saved["model"]
        scaler = saved["scaler"]

        logger.info("Model loaded successfully")

        return model, scaler

    except FileNotFoundError:
        logger.error("Model file not found: %s", path)
        raise

    except Exception as e:
        logger.error("Error loading model: %s", e)
        raise


def make_prediction(model, scaler, data):
    try:
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)

        logger.info("Prediction generated successfully")

        return prediction

    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise