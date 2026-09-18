def validate_task(task):

    errors = []

    if task.task_id < 0:
        errors.append(
            "task_id cannot be negative"
        )

    if task.arrival_time < 0:
        errors.append(
            "arrival_time cannot be negative"
        )

    if task.execution_time <= 0:
        errors.append(
            "execution_time must be greater than 0"
        )

    if task.deadline < task.arrival_time:
        errors.append(
            "deadline cannot be before arrival_time"
        )

    if task.priority < 0:
        errors.append(
            "priority cannot be negative"
        )

    if task.input_size < 0:
        errors.append(
            "input_size cannot be negative"
        )

    return errors


def validate_workload(tasks):

    errors = []

    if not tasks:
        errors.append(
            "Workload is empty"
        )
        return errors

    task_ids = set()

    for task in tasks:

        task_errors = validate_task(task)

        for error in task_errors:

            errors.append(
                f"Task {task.task_id}: {error}"
            )

        if task.task_id in task_ids:

            errors.append(
                f"Duplicate task_id: {task.task_id}"
            )

        task_ids.add(task.task_id)

    return errors


def is_valid_workload(tasks):

    return len(
        validate_workload(tasks)
    ) == 0