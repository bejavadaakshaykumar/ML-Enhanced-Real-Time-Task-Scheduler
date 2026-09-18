import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split


DATA_FILE = "data/ml_dataset.csv"
MODEL_FILE = "models/execution_time_model.pkl"


def train_model():

    # ---------------------------------------------
    # Load dataset
    # ---------------------------------------------

    dataframe = pd.read_csv(DATA_FILE)

    # ---------------------------------------------
    # Features
    # ---------------------------------------------

    feature_columns = [
        "arrival_time",
        "deadline",
        "priority",
        "input_size"
    ]

    X = dataframe[feature_columns]

    # ---------------------------------------------
    # Target
    # ---------------------------------------------

    y = dataframe["execution_time"]

    # ---------------------------------------------
    # Train/test split
    # ---------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # ---------------------------------------------
    # Create model
    # ---------------------------------------------

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # ---------------------------------------------
    # Train
    # ---------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # ---------------------------------------------
    # Predict
    # ---------------------------------------------

    predictions = model.predict(X_test)

    # ---------------------------------------------
    # Evaluation
    # ---------------------------------------------

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

    # ---------------------------------------------
    # Save model
    # ---------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print("\n========== MODEL TRAINING ==========\n")

    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Testing samples  : {len(X_test)}"
    )

    print(
        f"MAE              : {mae:.4f}"
    )

    print(
        f"RMSE             : {rmse:.4f}"
    )

    print(
        f"R² Score         : {r2:.4f}"
    )

    print(
        f"\nModel saved to: {MODEL_FILE}"
    )

    print(
        "\n===================================="
    )

    return model


if __name__ == "__main__":
    train_model()