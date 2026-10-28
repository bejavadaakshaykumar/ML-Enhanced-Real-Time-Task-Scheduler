import os

import joblib
import pandas as pd

from scheduler.task import Task


# Resolve model path relative to the project root,
# regardless of the current working directory.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "models",
    "execution_time_model.pkl"
)


class ExecutionTimePredictor:
    """
    Predicts the execution time of a task using a trained
    Random Forest Regressor model.

    Input features:
        arrival_time, deadline, priority, input_size

    Output:
        Predicted execution time (integer, minimum 1)
    """

    def __init__(
        self,
        model_file=MODEL_FILE
    ):

        self.model = joblib.load(
            model_file
        )

    def predict(self, task):
        """
        Predict execution time for a single task.

        Returns an integer >= 1.
        """

        features = pd.DataFrame(
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

        prediction = self.model.predict(
            features
        )[0]

        return max(
            1,
            round(prediction)
        )


if __name__ == "__main__":

    task = Task(
        task_id=100,
        arrival_time=5,
        execution_time=6,
        deadline=20,
        priority=2,
        input_size=50
    )

    predictor = ExecutionTimePredictor()

    prediction = predictor.predict(task)

    print(
        f"\nActual execution time : "
        f"{task.execution_time}"
    )

    print(
        f"Predicted execution time : "
        f"{prediction}"
    )