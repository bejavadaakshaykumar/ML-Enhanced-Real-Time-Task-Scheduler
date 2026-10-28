"""
Tests for the ML-EDF Scheduler.

Verifies:
- ML-EDF simulation completes all tasks
- Predictions are generated for all tasks
- Prediction data contains required fields
- Scheduler handles edge cases
- Scheduling order tracking works
"""

from scheduler.task import Task
from scheduler.simulator import SchedulerSimulator
from ml.ml_edf import MLEDFScheduler


def test_ml_edf_simulation():
    """All tasks should complete in a basic simulation."""

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


def test_ml_edf_predictions_contain_required_fields():
    """Each prediction should have all required fields."""

    tasks = [
        Task(
            task_id=1,
            arrival_time=0,
            execution_time=5,
            deadline=20,
            priority=2,
            input_size=50
        ),
    ]

    scheduler = MLEDFScheduler(
        preemptive=False
    )

    scheduler.update_time(0)

    prediction = scheduler.predict_task(tasks[0])

    required_fields = [
        "predicted_execution_time",
        "deadline_miss_probability",
        "remaining_slack",
        "risk_score",
        "urgency_score"
    ]

    for field in required_fields:
        assert field in prediction, (
            f"Missing field: {field}"
        )

    assert prediction["predicted_execution_time"] >= 1
    assert 0.0 <= prediction["deadline_miss_probability"] <= 1.0


def test_ml_edf_urgency_score_formula():
    """
    Verify the urgency score formula:
    urgency = slack - risk = (deadline - time - pred_exec) - (miss_prob * weight)
    """

    task = Task(
        task_id=1,
        arrival_time=0,
        execution_time=5,
        deadline=20,
        priority=2,
        input_size=50
    )

    scheduler = MLEDFScheduler(
        preemptive=False,
        risk_weight=10.0
    )

    scheduler.update_time(5)

    prediction = scheduler.predict_task(task)

    expected_slack = (
        task.deadline
        - 5  # current_time
        - prediction["predicted_execution_time"]
    )

    expected_risk = (
        prediction["deadline_miss_probability"]
        * 10.0
    )

    expected_urgency = expected_slack - expected_risk

    assert prediction["remaining_slack"] == expected_slack
    assert abs(prediction["urgency_score"] - expected_urgency) < 0.01


def test_ml_edf_single_task():
    """ML-EDF should handle a single task correctly."""

    task = Task(
        task_id=1,
        arrival_time=0,
        execution_time=3,
        deadline=10,
        priority=1,
        input_size=20
    )

    scheduler = MLEDFScheduler(
        preemptive=False
    )

    simulator = SchedulerSimulator(
        scheduler,
        [task]
    )

    results = simulator.run()

    assert task.completion_time == 3
    assert results["total_time"] == 3


def test_ml_edf_scheduling_order_tracked():
    """The scheduling order should be recorded."""

    tasks = [
        Task(
            task_id=1,
            arrival_time=0,
            execution_time=3,
            deadline=20,
            priority=2,
            input_size=40
        ),
        Task(
            task_id=2,
            arrival_time=0,
            execution_time=2,
            deadline=10,
            priority=1,
            input_size=30
        ),
    ]

    scheduler = MLEDFScheduler(
        preemptive=False
    )

    simulator = SchedulerSimulator(
        scheduler,
        tasks
    )

    simulator.run()

    assert len(scheduler.scheduling_order) == 2
    assert set(scheduler.scheduling_order) == {1, 2}


def test_ml_edf_clear_resets_state():
    """Clear should reset predictions and scheduling order."""

    scheduler = MLEDFScheduler(
        preemptive=False
    )

    task = Task(
        task_id=1,
        arrival_time=0,
        execution_time=3,
        deadline=10,
        priority=1,
        input_size=20
    )

    scheduler.update_time(0)
    scheduler.add_task(task)

    # Force a prediction
    scheduler.predict_task(task)

    assert len(scheduler.predictions) > 0

    scheduler.clear()

    assert scheduler.is_empty()
    assert len(scheduler.predictions) == 0
    assert len(scheduler.scheduling_order) == 0


def test_ml_edf_delayed_arrival():
    """Tasks arriving later should still complete correctly."""

    tasks = [
        Task(
            task_id=1,
            arrival_time=0,
            execution_time=2,
            deadline=5,
            priority=1,
            input_size=20
        ),
        Task(
            task_id=2,
            arrival_time=10,
            execution_time=3,
            deadline=20,
            priority=2,
            input_size=50
        ),
    ]

    scheduler = MLEDFScheduler(
        preemptive=False
    )

    simulator = SchedulerSimulator(
        scheduler,
        tasks
    )

    results = simulator.run()

    assert tasks[0].completion_time == 2
    assert tasks[1].completion_time == 13

    # There should be an idle gap between tasks
    idle_events = [
        e for e in results["trace"]
        if e.event == "IDLE"
    ]

    assert len(idle_events) >= 1