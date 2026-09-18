import joblib
import pandas as pd

from scheduler.task import Task


MODEL_FILE = "models/deadline_miss_model.pkl"


class DeadlineMissPredictor:

    def __init__(
        self,
        model_file=MODEL_FILE
    ):

        self.model = joblib.load(
            model_file
        )

    def _create_features(self, task):

        return pd.DataFrame(
            [[
                task.arrival_time,
                task.deadline,
                task.priority,
                task.input_size
            ]],
            columns=[
                "arrival_time",
                "deadline",
                "priority",
                "input_size"
            ]
        )

    def predict_probability(
        self,
        task
    ):

        features = self._create_features(
            task
        )

        probabilities = self.model.predict_proba(
            features
        )[0]

        classes = list(
            self.model.classes_
        )

        # Find probability corresponding to class 1
        if 1 in classes:

            class_index = classes.index(1)

            miss_probability = probabilities[
                class_index
            ]

        else:

            # Model contains no deadline-miss class.
            # This should normally not happen after
            # fixing the dataset.
            miss_probability = 0.0

        return float(
            miss_probability
        )

    def predict(
        self,
        task
    ):

        probability = self.predict_probability(
            task
        )

        return int(
            probability >= 0.5
        )


if __name__ == "__main__":

    task = Task(
        task_id=100,
        arrival_time=5,
        execution_time=6,
        deadline=12,
        priority=2,
        input_size=50
    )

    predictor = DeadlineMissPredictor()

    probability = predictor.predict_probability(
        task
    )

    prediction = predictor.predict(
        task
    )

    print(
        "\n========== DEADLINE PREDICTION ==========\n"
    )

    print(
        f"Task ID                 : {task.task_id}"
    )

    print(
        f"Arrival Time            : {task.arrival_time}"
    )

    print(
        f"Execution Time          : {task.execution_time}"
    )

    print(
        f"Deadline                : {task.deadline}"
    )

    print(
        f"Priority                : {task.priority}"
    )

    print(
        f"Deadline-miss probability: "
        f"{probability:.2%}"
    )

    print(
        f"Predicted class         : {prediction}"
    )

    if prediction == 1:

        print(
            "Prediction              : "
            "Deadline likely to be missed"
        )

    else:

        print(
            "Prediction              : "
            "Deadline likely to be met"
        )

    print(
        "\n=========================================="
    )