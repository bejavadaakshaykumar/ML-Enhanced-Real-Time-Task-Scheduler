import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


DATA_FILE = "data/ml_dataset.csv"
MODEL_FILE = "models/deadline_miss_model.pkl"


def create_deadline_labels(dataframe):
    """
    Create deadline-miss labels.

    0 = deadline met
    1 = deadline missed

    A task is considered to miss its deadline when:

        execution_time > deadline - arrival_time
    """

    dataframe = dataframe.copy()

    available_time = (
        dataframe["deadline"]
        - dataframe["arrival_time"]
    )

    dataframe["deadline_missed"] = (
        dataframe["execution_time"]
        > available_time
    ).astype(int)

    return dataframe


def train_classifier():

    # Load dataset
    dataframe = pd.read_csv(DATA_FILE)

    # Create labels
    dataframe = create_deadline_labels(
        dataframe
    )

    feature_columns = [
        "arrival_time",
        "deadline",
        "priority",
        "input_size"
    ]

    X = dataframe[feature_columns]
    y = dataframe["deadline_missed"]

    # Check that both classes exist
    unique_classes = sorted(
        y.unique().tolist()
    )

    print("\n========== DATASET CHECK ==========\n")

    print(
        f"Total samples : {len(dataframe)}"
    )

    print(
        f"Deadline met  : {(y == 0).sum()}"
    )

    print(
        f"Deadline miss : {(y == 1).sum()}"
    )

    print(
        f"Classes found : {unique_classes}"
    )

    print(
        "\n===================================\n"
    )

    if len(unique_classes) < 2:

        raise ValueError(
            "Classifier training requires both "
            "deadline-met (0) and deadline-missed (1) "
            "samples. Regenerate the dataset using "
            "the updated workload/generator.py."
        )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Predictions
    predictions = model.predict(
        X_test
    )

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # Confusion matrix
    matrix = confusion_matrix(
        y_test,
        predictions
    )

    # Classification report
    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    # Save model
    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        "\n========== CLASSIFIER TRAINING ==========\n"
    )

    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Testing samples  : {len(X_test)}"
    )

    print(
        f"Accuracy         : {accuracy:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        matrix
    )

    print(
        "\nClassification Report:"
    )

    print(
        report
    )

    print(
        f"Model saved to: {MODEL_FILE}"
    )

    print(
        "\n=========================================="
    )

    return model


if __name__ == "__main__":

    train_classifier()