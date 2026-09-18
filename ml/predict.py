import joblib
import pandas as pd

from scheduler.task import Task


MODEL_FILE = "models/execution_time_model.pkl"


class ExecutionTimePredictor:

    def __init__(
        self,
        model_file=MODEL_FILE
    ):

        self.model = joblib.load(
            model_file
        )

    def predict(self, task):

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