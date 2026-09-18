import pandas as pd
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split


DATA_FILE = "data/ml_dataset.csv"
MODEL_FILE = "models/execution_time_model.pkl"


def evaluate_model():

    dataframe = pd.read_csv(
        DATA_FILE
    )

    features = [
        "arrival_time",
        "deadline",
        "priority",
        "input_size"
    ]

    X = dataframe[features]

    y = dataframe["execution_time"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = joblib.load(
        MODEL_FILE
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n========== MODEL EVALUATION ==========\n")

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    print(
        "\n======================================"
    )


if __name__ == "__main__":
    evaluate_model()