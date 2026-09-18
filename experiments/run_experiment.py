# ============================================================
# ML-ENHANCED REAL-TIME TASK SCHEDULER
# SCHEDULER COMPARISON EXPERIMENT
# ============================================================

import json
import os

import pandas as pd

from scheduler.task import Task
from scheduler.fifo import FIFOScheduler
from scheduler.edf import EDFScheduler
from scheduler.simulator import SchedulerSimulator
from scheduler.metrics import calculate_summary

from ml.ml_edf import MLEDFScheduler

from workload.generator import (
    generate_workload,
    save_workload
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_TASKS = 100

SEED = 42

WORKLOAD_FILE = (
    "data/experiment_workload.csv"
)

RESULTS_CSV = (
    "results/scheduler_comparison.csv"
)

RESULTS_JSON = (
    "results/scheduler_comparison.json"
)


# ============================================================
# CLONE TASKS
# ============================================================

def clone_tasks(tasks):
    """
    Create independent copies of the workload.

    Every scheduler receives the same workload,
    but its own Task objects.
    """

    cloned_tasks = []

    for task in tasks:

        cloned_task = Task(

            task_id=task.task_id,

            arrival_time=task.arrival_time,

            execution_time=task.execution_time,

            deadline=task.deadline,

            priority=task.priority,

            task_type=task.task_type,

            input_size=task.input_size
        )

        cloned_tasks.append(
            cloned_task
        )

    return cloned_tasks


# ============================================================
# RUN ONE SCHEDULER
# ============================================================

def run_scheduler(
    scheduler_name,
    scheduler,
    tasks
):
    """
    Run one scheduler and calculate its metrics.
    """

    print(
        f"\nRunning {scheduler_name}..."
    )

    # --------------------------------------------------------
    # Create simulator
    # --------------------------------------------------------

    simulator = SchedulerSimulator(
        scheduler,
        tasks
    )

    # --------------------------------------------------------
    # Run simulation
    # --------------------------------------------------------

    results = simulator.run()

    # --------------------------------------------------------
    # Extract simulator results
    # --------------------------------------------------------

    total_time = (
        results["total_time"]
    )

    context_switches = (
        results["context_switches"]
    )

    preemptions = (
        results["preemptions"]
    )

    # --------------------------------------------------------
    # Calculate summary
    #
    # Correct function signature:
    #
    # calculate_summary(
    #     tasks,
    #     total_time,
    #     context_switches,
    #     preemptions
    # )
    # --------------------------------------------------------

    summary = calculate_summary(

        tasks,

        total_time,

        context_switches,

        preemptions
    )

    # --------------------------------------------------------
    # Add scheduler information
    # --------------------------------------------------------

    summary["scheduler"] = (
        scheduler_name
    )

    summary["total_time"] = (
        total_time
    )

    return summary, results


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print()

    print(
        "===================================================="
    )

    print(
        "       ML-ENHANCED REAL-TIME TASK SCHEDULER"
    )

    print(
        "              SCHEDULER COMPARISON"
    )

    print(
        "===================================================="
    )

    print()

    print(
        f"Number of tasks : {NUM_TASKS}"
    )

    print(
        f"Random seed     : {SEED}"
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    os.makedirs(
        "data",
        exist_ok=True
    )

    os.makedirs(
        "results",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Generate workload
    # --------------------------------------------------------

    print()

    print(
        "Generating workload..."
    )

    workload = generate_workload(

        num_tasks=NUM_TASKS,

        seed=SEED
    )

    # --------------------------------------------------------
    # Save workload
    # --------------------------------------------------------

    save_workload(

        workload,

        WORKLOAD_FILE
    )

    print(
        f"Workload saved to: "
        f"{WORKLOAD_FILE}"
    )

    # --------------------------------------------------------
    # Workload summary
    # --------------------------------------------------------

    print()

    print(
        "========== WORKLOAD SUMMARY =========="
    )

    print(
        f"Total Tasks : "
        f"{len(workload)}"
    )

    print(
        f"Arrival Range : "
        f"{min(task.arrival_time for task in workload)}"
        f" - "
        f"{max(task.arrival_time for task in workload)}"
    )

    print(
        f"Execution Range : "
        f"{min(task.execution_time for task in workload)}"
        f" - "
        f"{max(task.execution_time for task in workload)}"
    )

    print(
        f"Deadline Range : "
        f"{min(task.deadline for task in workload)}"
        f" - "
        f"{max(task.deadline for task in workload)}"
    )

    print(
        "======================================"
    )

    # --------------------------------------------------------
    # Scheduler list
    # --------------------------------------------------------

    schedulers = [

        (
            "FIFO",

            FIFOScheduler()
        ),

        (
            "EDF",

            EDFScheduler(
                preemptive=False
            )
        ),

        (
            "ML-EDF",

            MLEDFScheduler(

                preemptive=False,

                risk_weight=10.0
            )
        )

    ]

    # --------------------------------------------------------
    # Run all schedulers
    # --------------------------------------------------------

    summaries = []

    all_results = {}

    for scheduler_name, scheduler in schedulers:

        # Give each scheduler its own
        # copy of exactly the same workload.

        scheduler_tasks = clone_tasks(
            workload
        )

        summary, results = run_scheduler(

            scheduler_name,

            scheduler,

            scheduler_tasks
        )

        summaries.append(
            summary
        )

        all_results[
            scheduler_name
        ] = results

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    dataframe = pd.DataFrame(
        summaries
    )

    # --------------------------------------------------------
    # Arrange columns
    # --------------------------------------------------------

    preferred_columns = [

        "scheduler",

        "average_waiting_time",

        "average_response_time",

        "average_turnaround_time",

        "average_tardiness",

        "deadline_misses",

        "deadline_miss_rate",

        "cpu_utilization",

        "throughput",

        "context_switches",

        "preemptions",

        "total_time"
    ]

    existing_columns = [

        column

        for column in preferred_columns

        if column in dataframe.columns
    ]

    remaining_columns = [

        column

        for column in dataframe.columns

        if column not in existing_columns
    ]

    dataframe = dataframe[
        existing_columns
        +
        remaining_columns
    ]

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    dataframe.to_csv(

        RESULTS_CSV,

        index=False
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    json_data = {

        "experiment": {

            "num_tasks":
                NUM_TASKS,

            "seed":
                SEED
        },

        "workload_file":
            WORKLOAD_FILE,

        "results":
            summaries
    }

    with open(

        RESULTS_JSON,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            json_data,

            file,

            indent=4
        )

    # --------------------------------------------------------
    # Print comparison table
    # --------------------------------------------------------

    print()

    print(
        "===================================================="
    )

    print(
        "             SCHEDULER COMPARISON"
    )

    print(
        "===================================================="
    )

    print()

    print(
        dataframe.to_string(
            index=False
        )
    )

    print()

    print(
        "===================================================="
    )

    print(
        "Results saved:"
    )

    print()

    print(
        f"CSV  : {RESULTS_CSV}"
    )

    print(
        f"JSON : {RESULTS_JSON}"
    )

    print(
        "===================================================="
    )

    # --------------------------------------------------------
    # Detailed summaries
    # --------------------------------------------------------

    for summary in summaries:

        print()

        print(
            "========== "
            f"{summary['scheduler']} "
            "SUMMARY =========="
        )

        for key, value in summary.items():

            if isinstance(
                value,
                float
            ):

                print(
                    f"{key:25} : "
                    f"{value:.4f}"
                )

            else:

                print(
                    f"{key:25} : "
                    f"{value}"
                )

    # --------------------------------------------------------
    # ML-EDF information
    # --------------------------------------------------------

    ml_results = all_results.get(
        "ML-EDF"
    )

    if ml_results is not None:

        print()

        print(
            "========== ML-EDF SIMULATION =========="
        )

        print(
            f"Total Time       : "
            f"{ml_results['total_time']}"
        )

        print(
            f"Context Switches : "
            f"{ml_results['context_switches']}"
        )

        print(
            f"Preemptions      : "
            f"{ml_results['preemptions']}"
        )

        print(
            "========================================"
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()

    print(
        "Experiment completed successfully."
    )

    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()