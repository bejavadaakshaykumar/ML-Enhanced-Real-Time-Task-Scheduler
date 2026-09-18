from dataclasses import dataclass


@dataclass
class Task:
    task_id: int
    arrival_time: int
    execution_time: int
    deadline: int
    priority: int = 0
    task_type: str = "CPU"
    input_size: int = 1

    # Runtime fields
    remaining_time: int = 0
    start_time: int | None = None
    completion_time: int | None = None

    def __post_init__(self):
        if self.execution_time <= 0:
            raise ValueError("execution_time must be greater than 0")

        if self.deadline < self.arrival_time:
            raise ValueError(
                "deadline must be greater than or equal to arrival_time"
            )

        if self.remaining_time == 0:
            self.remaining_time = self.execution_time

    def reset(self):
        """Reset runtime state before another simulation."""
        self.remaining_time = self.execution_time
        self.start_time = None
        self.completion_time = None

    @property
    def waiting_time(self):
        if self.completion_time is None:
            return None

        turnaround = self.completion_time - self.arrival_time
        return turnaround - self.execution_time

    @property
    def response_time(self):
        if self.start_time is None:
            return None

        return self.start_time - self.arrival_time

    @property
    def turnaround_time(self):
        if self.completion_time is None:
            return None

        return self.completion_time - self.arrival_time

    @property
    def tardiness(self):
        if self.completion_time is None:
            return None

        return max(0, self.completion_time - self.deadline)

    @property
    def deadline_missed(self):
        if self.completion_time is None:
            return False

        return self.completion_time > self.deadline