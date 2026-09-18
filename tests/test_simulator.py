from scheduler.task import Task
from scheduler.fifo import FIFOScheduler
from scheduler.edf import EDFScheduler
from scheduler.simulator import SchedulerSimulator


def test_fifo_simulation():

    tasks = [
        Task(
            task_id=1,
            arrival_time=0,
            execution_time=5,
            deadline=20
        ),
        Task(
            task_id=2,
            arrival_time=0,
            execution_time=3,
            deadline=15
        ),
        Task(
            task_id=3,
            arrival_time=0,
            execution_time=2,
            deadline=10
        )
    ]

    scheduler = FIFOScheduler()

    simulator = SchedulerSimulator(
        scheduler,
        tasks
    )

    results = simulator.run()

    assert tasks[0].completion_time == 5
    assert tasks[1].completion_time == 8
    assert tasks[2].completion_time == 10

    assert results["total_time"] == 10


def test_edf_simulation():

    tasks = [
        Task(
            task_id=1,
            arrival_time=0,
            execution_time=5,
            deadline=20
        ),
        Task(
            task_id=2,
            arrival_time=0,
            execution_time=3,
            deadline=10
        ),
        Task(
            task_id=3,
            arrival_time=0,
            execution_time=2,
            deadline=15
        )
    ]

    scheduler = EDFScheduler(
        preemptive=False
    )

    simulator = SchedulerSimulator(
        scheduler,
        tasks
    )

    results = simulator.run()

    assert tasks[1].completion_time == 3
    assert tasks[2].completion_time == 5
    assert tasks[0].completion_time == 10


def test_preemptive_edf():

    tasks = [
        Task(
            task_id=1,
            arrival_time=0,
            execution_time=8,
            deadline=30
        ),
        Task(
            task_id=2,
            arrival_time=2,
            execution_time=2,
            deadline=5
        )
    ]

    scheduler = EDFScheduler(
        preemptive=True
    )

    simulator = SchedulerSimulator(
        scheduler,
        tasks
    )

    results = simulator.run()

    # T1 runs from 0 to 2
    # T2 arrives at 2
    # T2 has earlier deadline
    # T1 gets preempted
    # T2 runs from 2 to 4
    # T1 resumes from 4 to 10

    assert tasks[1].completion_time == 4
    assert tasks[0].completion_time == 10

    assert results["preemptions"] == 1


def test_preemption_trace():

    tasks = [
        Task(
            task_id=1,
            arrival_time=0,
            execution_time=6,
            deadline=20
        ),
        Task(
            task_id=2,
            arrival_time=2,
            execution_time=2,
            deadline=5
        )
    ]

    scheduler = EDFScheduler(
        preemptive=True
    )

    simulator = SchedulerSimulator(
        scheduler,
        tasks
    )

    results = simulator.run()

    trace = results["trace"]

    task_ids = [
        event.task_id
        for event in trace
        if event.event == "RUN"
    ]

    assert 1 in task_ids
    assert 2 in task_ids

    assert results["preemptions"] == 1


def test_context_switch():

    tasks = [
        Task(
            task_id=1,
            arrival_time=0,
            execution_time=3,
            deadline=10
        ),
        Task(
            task_id=2,
            arrival_time=0,
            execution_time=2,
            deadline=8
        )
    ]

    scheduler = EDFScheduler(
        preemptive=True
    )

    simulator = SchedulerSimulator(
        scheduler,
        tasks,
        context_switch_cost=1
    )

    results = simulator.run()

    assert results["context_switches"] >= 1


def test_idle_cpu():

    tasks = [
        Task(
            task_id=1,
            arrival_time=5,
            execution_time=2,
            deadline=10
        )
    ]

    scheduler = FIFOScheduler()

    simulator = SchedulerSimulator(
        scheduler,
        tasks
    )

    results = simulator.run()

    assert tasks[0].start_time == 5
    assert tasks[0].completion_time == 7

    idle_events = [
        event
        for event in results["trace"]
        if event.event == "IDLE"
    ]

    assert len(idle_events) == 1