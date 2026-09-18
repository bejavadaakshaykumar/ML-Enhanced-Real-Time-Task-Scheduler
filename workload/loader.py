import csv
import json

from scheduler.task import Task


def load_csv(filename):

    tasks = []

    with open(
        filename,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            task = Task(
                task_id=int(row["task_id"]),
                arrival_time=int(row["arrival_time"]),
                execution_time=int(row["execution_time"]),
                deadline=int(row["deadline"]),
                priority=int(row.get("priority", 0)),
                task_type=row.get(
                    "task_type",
                    "CPU"
                ),
                input_size=int(
                    row.get("input_size", 1)
                )
            )

            tasks.append(task)

    return tasks


def save_json(tasks, filename):

    data = []

    for task in tasks:

        data.append({
            "task_id": task.task_id,
            "arrival_time": task.arrival_time,
            "execution_time": task.execution_time,
            "deadline": task.deadline,
            "priority": task.priority,
            "task_type": task.task_type,
            "input_size": task.input_size
        })

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


def load_json(filename):

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    tasks = []

    for item in data:

        task = Task(
            task_id=int(item["task_id"]),
            arrival_time=int(
                item["arrival_time"]
            ),
            execution_time=int(
                item["execution_time"]
            ),
            deadline=int(item["deadline"]),
            priority=int(
                item.get("priority", 0)
            ),
            task_type=item.get(
                "task_type",
                "CPU"
            ),
            input_size=int(
                item.get("input_size", 1)
            )
        )

        tasks.append(task)

    return tasks