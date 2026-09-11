import sys

sys.path.insert(0, "src")

import pandas as pd

from ml_platform.data import load_data
from ml_platform.preprocessing import split_data, scale_data
from ml_platform.models import create_model, train_model, predict
from ml_platform.evaluation import evaluate_model


DATA_PATH = "data/raw/sample_data.csv"
TARGET_COLUMN = "loan_approved"


def test_data_loading():
    df = load_data(DATA_PATH)

    assert not df.empty
    assert TARGET_COLUMN in df.columns


def test_data_split():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(
        df,
        TARGET_COLUMN
    )

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_scaling():
    df = load_data(DATA_PATH)

    X_train, X_test, _, _ = split_data(
        df,
        TARGET_COLUMN
    )

    X_train_scaled, X_test_scaled, scaler = scale_data(
        X_train,
        X_test
    )

    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert scaler is not None


def test_model_training_and_prediction():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(
        df,
        TARGET_COLUMN
    )

    X_train, X_test, scaler = scale_data(
        X_train,
        X_test
    )

    model = create_model()

    model = train_model(
        model,
        X_train,
        y_train
    )

    predictions = predict(
        model,
        X_test
    )

    assert len(predictions) == len(y_test)


def test_model_evaluation():
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(
        df,
        TARGET_COLUMN
    )

    X_train, X_test, scaler = scale_data(
        X_train,
        X_test
    )

    model = create_model()

    model = train_model(
        model,
        X_train,
        y_train
    )

    predictions = predict(
        model,
        X_test
    )

    accuracy, matrix = evaluate_model(
        y_test,
        predictions
    )

    assert 0 <= accuracy <= 1
    assert matrix is not None