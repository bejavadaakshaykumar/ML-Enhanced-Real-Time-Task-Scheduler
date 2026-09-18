from scheduler.task import Task
from scheduler.simulator import SchedulerSimulator
from ml.ml_edf import MLEDFScheduler


def test_ml_edf_simulation():

    tasks = [
        Task(
            task_id=1,
            arrival_time=0,
            execution_time=5,
            deadline=15,
            priority=2,
            input_size=40
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
            arrival_time=2,
            execution_time=4,
            deadline=12,
            priority=3,
            input_size=60
        )
    ]

    scheduler = MLEDFScheduler(
        preemptive=False
    )

    simulator = SchedulerSimulator(
        scheduler,
        tasks
    )

    results = simulator.run()

    assert results is not None

    assert len(
        results["tasks"]
    ) == 3

    for task in tasks:

        assert (
            task.completion_time
            is not None
        )

        assert (
            task.remaining_time
            == 0
        )

    assert (
        results["total_time"]
        > 0
    )

    assert len(
        scheduler.predictions
    ) > 0