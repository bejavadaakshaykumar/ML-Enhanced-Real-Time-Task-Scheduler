from workload.generator import (
    generate_workload
)

from workload.validator import (
    validate_workload,
    is_valid_workload
)


def test_workload_generation():

    tasks = generate_workload(
        num_tasks=10,
        seed=42
    )

    assert len(tasks) == 10

    assert is_valid_workload(tasks)


def test_reproducible_workload():

    tasks1 = generate_workload(
        num_tasks=10,
        seed=42
    )

    tasks2 = generate_workload(
        num_tasks=10,
        seed=42
    )

    for t1, t2 in zip(tasks1, tasks2):

        assert t1.task_id == t2.task_id
        assert t1.arrival_time == t2.arrival_time
        assert t1.execution_time == t2.execution_time
        assert t1.deadline == t2.deadline
        assert t1.priority == t2.priority


def test_different_seed():

    tasks1 = generate_workload(
        num_tasks=10,
        seed=42
    )

    tasks2 = generate_workload(
        num_tasks=10,
        seed=99
    )

    values1 = [
        task.execution_time
        for task in tasks1
    ]

    values2 = [
        task.execution_time
        for task in tasks2
    ]

    assert values1 != values2