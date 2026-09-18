import os
import sys

# Allow execution from the project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from scheduler.min_heap import MinHeap
from ml.predict import ExecutionTimePredictor
from ml.predict_deadline import DeadlineMissPredictor


class MLEDFScheduler:
    """
    Machine-Learning Enhanced Earliest Deadline First Scheduler.

    Scheduling decision uses:

    1. Deadline
    2. Predicted execution time
    3. Predicted deadline-miss probability
    4. Remaining slack

    Smaller score = higher scheduling priority.
    """

    def __init__(
        self,
        preemptive=False,
        risk_weight=10.0
    ):

        self.preemptive = preemptive
        self.risk_weight = risk_weight

        self.ready_queue = MinHeap(
            key_function=self.get_key
        )

        self.current_time = 0

        self.execution_predictor = (
            ExecutionTimePredictor()
        )

        self.deadline_predictor = (
            DeadlineMissPredictor()
        )

        # Store predictions for inspection
        self.predictions = {}

    def update_time(self, current_time):
        """
        Update the virtual scheduler time.
        """

        self.current_time = current_time

    def predict_task(self, task):
        """
        Generate ML predictions for a task.
        """

        predicted_execution_time = (
            self.execution_predictor.predict(task)
        )

        deadline_miss_probability = (
            self.deadline_predictor.predict_probability(task)
        )

        remaining_slack = (
            task.deadline
            - self.current_time
            - predicted_execution_time
        )

        prediction_data = {
            "predicted_execution_time":
                predicted_execution_time,

            "deadline_miss_probability":
                deadline_miss_probability,

            "remaining_slack":
                remaining_slack
        }

        self.predictions[
            task.task_id
        ] = prediction_data

        return prediction_data

    def get_key(self, task):
        """
        Calculate ML-EDF scheduling priority.

        Tasks with:
        - smaller slack
        - higher deadline-miss probability
        - earlier deadline

        receive higher priority.

        This is a project-specific risk-aware EDF
        policy rather than standard EDF.
        """

        prediction = self.predict_task(
            task
        )

        predicted_execution_time = (
            prediction[
                "predicted_execution_time"
            ]
        )

        miss_probability = (
            prediction[
                "deadline_miss_probability"
            ]
        )

        slack = (
            prediction[
                "remaining_slack"
            ]
        )

        # Risk-adjusted urgency
        risk_score = (
            miss_probability
            * self.risk_weight
        )

        # Lower slack means greater urgency.
        #
        # Negative slack means the predicted
        # execution time cannot fit before deadline.
        urgency_score = (
            slack
            - risk_score
        )

        return (
            urgency_score,
            task.deadline,
            miss_probability * -1,
            predicted_execution_time,
            task.priority,
            task.arrival_time,
            task.task_id
        )

    def add_task(self, task):

        self.ready_queue.push(
            task
        )

    def add_tasks(self, tasks):

        for task in tasks:
            self.add_task(task)

    def select_task(self):

        return self.ready_queue.pop()

    def peek_task(self):

        return self.ready_queue.peek()

    def is_empty(self):

        return self.ready_queue.is_empty()

    def size(self):

        return self.ready_queue.size()

    def clear(self):

        self.ready_queue.clear()

        self.predictions.clear()

    def get_prediction(self, task_id):

        return self.predictions.get(
            task_id
        )

    def print_predictions(self):

        print(
            "\n========== ML PREDICTIONS ==========\n"
        )

        for task_id, prediction in sorted(
            self.predictions.items()
        ):

            print(
                f"T{task_id}: "
                f"Predicted Exec = "
                f"{prediction['predicted_execution_time']}, "
                f"Miss Probability = "
                f"{prediction['deadline_miss_probability']:.2%}, "
                f"Slack = "
                f"{prediction['remaining_slack']}"
            )

        print(
            "\n====================================="
        )


if __name__ == "__main__":

    from scheduler.task import Task

    tasks = [
        Task(
            task_id=1,
            arrival_time=0,
            execution_time=8,
            deadline=20,
            priority=2,
            input_size=50
        ),

        Task(
            task_id=2,
            arrival_time=0,
            execution_time=3,
            deadline=10,
            priority=1,
            input_size=30
        ),

        Task(
            task_id=3,
            arrival_time=0,
            execution_time=5,
            deadline=15,
            priority=3,
            input_size=70
        )
    ]

    scheduler = MLEDFScheduler(
        preemptive=False
    )

    scheduler.update_time(0)

    scheduler.add_tasks(
        tasks
    )

    print(
        "\n========== ML-EDF TEST ==========\n"
    )

    while not scheduler.is_empty():

        task = scheduler.select_task()

        prediction = scheduler.get_prediction(
            task.task_id
        )

        print(
            f"Selected Task: T{task.task_id}"
        )

        print(
            f"  Deadline                : "
            f"{task.deadline}"
        )

        print(
            f"  Predicted Execution     : "
            f"{prediction['predicted_execution_time']}"
        )

        print(
            f"  Deadline-Miss Probability: "
            f"{prediction['deadline_miss_probability']:.2%}"
        )

        print(
            f"  Predicted Slack         : "
            f"{prediction['remaining_slack']}"
        )

        print()