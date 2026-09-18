from scheduler.task import Task
from scheduler.edf import EDFScheduler


def test_edf_order():

    t1 = Task(
        task_id=1,
        arrival_time=0,
        execution_time=5,
        deadline=20,
        priority=2
    )

    t2 = Task(
        task_id=2,
        arrival_time=0,
        execution_time=3,
        deadline=10,
        priority=1
    )

    t3 = Task(
        task_id=3,
        arrival_time=0,
        execution_time=2,
        deadline=15,
        priority=3
    )

    scheduler = EDFScheduler()

    scheduler.add_tasks([t1, t2, t3])

    assert scheduler.select_task().task_id == 2
    assert scheduler.select_task().task_id == 3
    assert scheduler.select_task().task_id == 1