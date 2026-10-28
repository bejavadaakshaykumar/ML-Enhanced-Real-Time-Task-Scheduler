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

    ML-EDF extends the classical EDF algorithm by incorporating
    two ML predictions into each scheduling decision:

    1. **Predicted Execution Time** — A Random Forest regression
       model estimates how long a task will take to execute.

    2. **Deadline-Miss Probability** — A Random Forest classifier
       estimates the probability that a task will miss its deadline.

    These predictions are combined into a risk-aware urgency score:

        remaining_slack = deadline - current_time - predicted_exec_time
        risk_score      = miss_probability × risk_weight
        urgency_score   = remaining_slack - risk_score

    A **smaller** urgency score means **higher** scheduling priority.

    When risk_weight = 0, ML-EDF degrades to a slack-based variant
    of EDF (tasks with less remaining slack are scheduled first).
    The deadline is used as a secondary tiebreaker so that the
    ordering remains consistent with classical EDF.

    Design Note
    -----------
    This is a project-specific risk-aware EDF policy rather than
    a standard textbook algorithm. Its purpose is to demonstrate
    how ML predictions can augment classical scheduling heuristics.
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

        # Store predictions for dashboard inspection
        self.predictions = {}

        # Track scheduling order for comparison
        self.scheduling_order = []

    def update_time(self, current_time):
        """
        Update the virtual scheduler time.

        Called by the simulator before each scheduling decision
        so that slack calculations use the correct current time.
        """

        self.current_time = current_time

    def predict_task(self, task):
        """
        Generate ML predictions for a task.

        Returns a dictionary containing:
        - predicted_execution_time: estimated execution duration
        - deadline_miss_probability: probability of missing deadline (0.0–1.0)
        - remaining_slack: deadline - current_time - predicted_exec_time
        - urgency_score: the final risk-adjusted scheduling score
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

        # Risk-adjusted urgency (lower = more urgent)
        risk_score = (
            deadline_miss_probability
            * self.risk_weight
        )

        urgency_score = (
            remaining_slack
            - risk_score
        )

        prediction_data = {
            "predicted_execution_time":
                predicted_execution_time,

            "deadline_miss_probability":
                deadline_miss_probability,

            "remaining_slack":
                remaining_slack,

            "risk_score":
                risk_score,

            "urgency_score":
                urgency_score
        }

        self.predictions[
            task.task_id
        ] = prediction_data

        return prediction_data

    def get_key(self, task):
        """
        Calculate ML-EDF scheduling priority.

        The key tuple determines heap ordering. Elements are
        compared left-to-right (Python tuple comparison):

        1. urgency_score   — primary: lower slack & higher risk → first
        2. deadline        — tiebreaker: earlier deadline → first
        3. -miss_prob      — tiebreaker: higher risk → first
        4. predicted_exec  — tiebreaker: shorter predicted job → first
        5. priority        — tiebreaker: lower priority value → first
        6. arrival_time    — tiebreaker: earlier arrival → first
        7. task_id         — final tiebreaker: deterministic ordering
        """

        prediction = self.predict_task(
            task
        )

        urgency_score = (
            prediction["urgency_score"]
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

        task = self.ready_queue.pop()

        if task is not None:
            self.scheduling_order.append(
                task.task_id
            )

        return task

    def peek_task(self):

        return self.ready_queue.peek()

    def is_empty(self):

        return self.ready_queue.is_empty()

    def size(self):

        return self.ready_queue.size()

    def clear(self):

        self.ready_queue.clear()

        self.predictions.clear()

        self.scheduling_order.clear()

    def get_prediction(self, task_id):

        return self.predictions.get(
            task_id
        )

    def get_all_predictions(self):
        """Return all predictions sorted by task_id."""

        return dict(
            sorted(self.predictions.items())
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
                f"{prediction['remaining_slack']}, "
                f"Urgency = "
                f"{prediction['urgency_score']:.2f}"
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

        print(
            f"  Urgency Score           : "
            f"{prediction['urgency_score']:.2f}"
        )

        print()