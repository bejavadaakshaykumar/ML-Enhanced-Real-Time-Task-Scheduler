def task_to_features(task):
    """
    Convert a Task object into ML input features.
    """

    return [
        task.arrival_time,
        task.deadline,
        task.priority,
        task.input_size
    ]


def tasks_to_features(tasks):
    """
    Convert multiple tasks into a feature matrix.
    """

    return [
        task_to_features(task)
        for task in tasks
    ]