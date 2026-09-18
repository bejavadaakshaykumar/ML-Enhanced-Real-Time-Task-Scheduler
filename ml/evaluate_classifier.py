import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split


DATA_FILE = "data/ml_dataset.csv"
MODEL_FILE = "models/deadline_miss_model.pkl"


def create_deadline_labels(dataframe):

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


def evaluate_classifier():

    # Load dataset
    dataframe = pd.read_csv(
        DATA_FILE
    )

    # Create labels
    dataframe = create_deadline_labels(
        dataframe
    )

    features = [
        "arrival_time",
        "deadline",
        "priority",
        "input_size"
    ]

    X = dataframe[features]
    y = dataframe["deadline_missed"]

    # Verify both classes
    unique_classes = sorted(
        y.unique().tolist()
    )

    if len(unique_classes) < 2:

        raise ValueError(
            "Evaluation requires both deadline-met "
            "and deadline-missed classes."
        )

    # Same split used during training
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Load trained model
    model = joblib.load(
        MODEL_FILE
    )

    # Predictions
    predictions = model.predict(
        X_test
    )

    # Metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print(
        "\n========== CLASSIFIER EVALUATION ==========\n"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
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
        "============================================"
    )


if __name__ == "__main__":

    evaluate_classifier()