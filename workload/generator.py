import random

from scheduler.task import Task


TASK_TYPES = [
    "CPU",
    "IO",
    "PERIODIC",
    "APERIODIC"
]


def generate_workload(
    num_tasks=10,
    seed=42,
    max_arrival_time=20,
    min_execution_time=1,
    max_execution_time=10,
    min_deadline_offset=2,
    max_deadline_offset=12,
    num_priorities=5
):
    """
    Generate a reproducible task workload.

    The deadline is generated independently of execution time
    so that both deadline-met and deadline-missed tasks can occur.
    """

    rng = random.Random(seed)

    tasks = []

    for task_id in range(1, num_tasks + 1):

        arrival_time = rng.randint(
            0,
            max_arrival_time
        )

        execution_time = rng.randint(
            min_execution_time,
            max_execution_time
        )

        deadline_offset = rng.randint(
            min_deadline_offset,
            max_deadline_offset
        )

        # Important:
        # Deadline is NOT calculated using execution_time.
        deadline = arrival_time + deadline_offset

        priority = rng.randint(
            1,
            num_priorities
        )

        task_type = rng.choice(
            TASK_TYPES
        )

        input_size = rng.randint(
            1,
            100
        )

        task = Task(
            task_id=task_id,
            arrival_time=arrival_time,
            execution_time=execution_time,
            deadline=deadline,
            priority=priority,
            task_type=task_type,
            input_size=input_size
        )

        tasks.append(task)

    tasks.sort(
        key=lambda task: (
            task.arrival_time,
            task.task_id
        )
    )

    return tasks


def save_workload(tasks, filename):

    import csv

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "task_id",
            "arrival_time",
            "execution_time",
            "deadline",
            "priority",
            "task_type",
            "input_size"
        ])

        for task in tasks:

            writer.writerow([
                task.task_id,
                task.arrival_time,
                task.execution_time,
                task.deadline,
                task.priority,
                task.task_type,
                task.input_size
            ])