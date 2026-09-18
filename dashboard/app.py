# ============================================================
# ML-ENHANCED REAL-TIME TASK SCHEDULER
# STREAMLIT DASHBOARD
# ============================================================

import os
import sys

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------

from scheduler.task import Task
from scheduler.fifo import FIFOScheduler
from scheduler.edf import EDFScheduler
from scheduler.simulator import SchedulerSimulator
from scheduler.metrics import calculate_summary

from ml.ml_edf import MLEDFScheduler

from workload.generator import generate_workload


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ML Task Scheduler",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "ML-Enhanced Real-Time Task Scheduler"
)

st.markdown(
    """
### Intelligent Scheduling using FIFO, EDF and ML-EDF

This dashboard simulates real-time task scheduling and
compares traditional scheduling algorithms with a
Machine-Learning Enhanced EDF scheduler.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Experiment Configuration"
)

num_tasks = st.sidebar.slider(
    "Number of Tasks",
    min_value=5,
    max_value=100,
    value=20,
    step=5
)

seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    value=42,
    step=1
)

risk_weight = st.sidebar.slider(
    "ML-EDF Risk Weight",
    min_value=0.0,
    max_value=20.0,
    value=10.0,
    step=0.5
)

scheduler_choice = st.sidebar.selectbox(
    "Scheduler",
    [
        "All",
        "FIFO",
        "EDF",
        "ML-EDF"
    ]
)

run_button = st.sidebar.button(
    "Run Simulation",
    type="primary"
)


# ============================================================
# SESSION STATE
# ============================================================

if "simulation_done" not in st.session_state:
    st.session_state.simulation_done = False

if "comparison" not in st.session_state:
    st.session_state.comparison = None

if "results" not in st.session_state:
    st.session_state.results = {}

if "workload" not in st.session_state:
    st.session_state.workload = None


# ============================================================
# CLONE TASKS
# ============================================================

def clone_tasks(tasks):

    cloned_tasks = []

    for task in tasks:

        cloned_tasks.append(
            Task(
                task_id=task.task_id,
                arrival_time=task.arrival_time,
                execution_time=task.execution_time,
                deadline=task.deadline,
                priority=task.priority,
                task_type=task.task_type,
                input_size=task.input_size
            )
        )

    return cloned_tasks


# ============================================================
# RUN SCHEDULER
# ============================================================

def run_scheduler(
    scheduler_name,
    scheduler,
    workload
):

    tasks = clone_tasks(
        workload
    )

    simulator = SchedulerSimulator(
        scheduler,
        tasks
    )

    results = simulator.run()

    summary = calculate_summary(
        tasks,
        results["total_time"],
        results["context_switches"],
        results["preemptions"]
    )

    summary["scheduler"] = scheduler_name

    summary["total_time"] = (
        results["total_time"]
    )

    return summary, results, tasks


# ============================================================
# RUN EXPERIMENT
# ============================================================

if run_button:

    with st.spinner(
        "Generating workload and running simulation..."
    ):

        workload = generate_workload(
            num_tasks=num_tasks,
            seed=int(seed)
        )

        scheduler_definitions = [

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
                    risk_weight=risk_weight
                )
            )

        ]

        summaries = []

        results = {}

        for name, scheduler in scheduler_definitions:

            summary, simulation, tasks = run_scheduler(
                name,
                scheduler,
                workload
            )

            summaries.append(
                summary
            )

            results[name] = {
                "simulation": simulation,
                "tasks": tasks
            }

        comparison = pd.DataFrame(
            summaries
        )

        st.session_state.workload = workload

        st.session_state.comparison = comparison

        st.session_state.results = results

        st.session_state.simulation_done = True


# ============================================================
# DASHBOARD
# ============================================================

if not st.session_state.simulation_done:

    st.info(
        "Configure the experiment from the sidebar "
        "and click 'Run Simulation'."
    )

else:

    comparison = (
        st.session_state.comparison
    )

    results = (
        st.session_state.results
    )

    workload = (
        st.session_state.workload
    )


    # ========================================================
    # WORKLOAD OVERVIEW
    # ========================================================

    st.header(
        "1. Workload Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Tasks",
            len(workload)
        )

    with col2:

        st.metric(
            "Arrival Range",
            f"{min(t.arrival_time for t in workload)} - "
            f"{max(t.arrival_time for t in workload)}"
        )

    with col3:

        st.metric(
            "Execution Range",
            f"{min(t.execution_time for t in workload)} - "
            f"{max(t.execution_time for t in workload)}"
        )

    with col4:

        st.metric(
            "Deadline Range",
            f"{min(t.deadline for t in workload)} - "
            f"{max(t.deadline for t in workload)}"
        )


    # ========================================================
    # WORKLOAD TABLE
    # ========================================================

    st.subheader(
        "Generated Task Workload"
    )

    workload_data = []

    for task in workload:

        workload_data.append(
            {
                "Task ID": task.task_id,
                "Arrival": task.arrival_time,
                "Execution": task.execution_time,
                "Deadline": task.deadline,
                "Priority": task.priority,
                "Type": task.task_type,
                "Input Size": task.input_size
            }
        )

    workload_df = pd.DataFrame(
        workload_data
    )

    st.dataframe(
        workload_df,
        use_container_width=True
    )


    # ========================================================
    # SCHEDULER COMPARISON
    # ========================================================

    st.header(
        "2. Scheduler Performance Comparison"
    )

    display_columns = [

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

    display_columns = [

        column

        for column in display_columns

        if column in comparison.columns
    ]

    display_df = comparison[
        display_columns
    ].copy()


    # --------------------------------------------------------
    # Rename columns for dashboard
    # --------------------------------------------------------

    display_df = display_df.rename(
        columns={
            "scheduler":
                "Scheduler",

            "average_waiting_time":
                "Avg Waiting Time",

            "average_response_time":
                "Avg Response Time",

            "average_turnaround_time":
                "Avg Turnaround Time",

            "average_tardiness":
                "Avg Tardiness",

            "deadline_misses":
                "Deadline Misses",

            "deadline_miss_rate":
                "Deadline Miss Rate (%)",

            "cpu_utilization":
                "CPU Utilization (%)",

            "throughput":
                "Throughput",

            "context_switches":
                "Context Switches",

            "preemptions":
                "Preemptions",

            "total_time":
                "Total Time"
        }
    )


    st.dataframe(
        display_df,
        use_container_width=True
    )


    # ========================================================
    # METRIC CARDS
    # ========================================================

    st.header(
        "3. Performance Metrics"
    )

    selected_scheduler = scheduler_choice

    if selected_scheduler == "All":

        metric_scheduler = "ML-EDF"

    else:

        metric_scheduler = selected_scheduler


    selected_row = comparison[
        comparison["scheduler"]
        == metric_scheduler
    ]

    if not selected_row.empty:

        row = selected_row.iloc[0]

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Avg Waiting Time",
                f"{row['average_waiting_time']:.2f}"
            )

        with c2:

            st.metric(
                "Avg Response Time",
                f"{row['average_response_time']:.2f}"
            )

        with c3:

            st.metric(
                "Deadline Misses",
                int(row["deadline_misses"])
            )

        with c4:

            st.metric(
                "CPU Utilization",
                f"{row['cpu_utilization']:.2f}%"
            )


    # ========================================================
    # CHARTS
    # ========================================================

    st.header(
        "4. Scheduler Comparison Charts"
    )


    # --------------------------------------------------------
    # Waiting Time
    # --------------------------------------------------------

    st.subheader(
        "Average Waiting Time"
    )

    fig1, ax1 = plt.subplots()

    ax1.bar(
        comparison["scheduler"],
        comparison["average_waiting_time"]
    )

    ax1.set_xlabel(
        "Scheduler"
    )

    ax1.set_ylabel(
        "Average Waiting Time"
    )

    ax1.set_title(
        "Waiting Time Comparison"
    )

    st.pyplot(
        fig1
    )

    plt.close(fig1)


    # --------------------------------------------------------
    # Deadline Misses
    # --------------------------------------------------------

    st.subheader(
        "Deadline Misses"
    )

    fig2, ax2 = plt.subplots()

    ax2.bar(
        comparison["scheduler"],
        comparison["deadline_misses"]
    )

    ax2.set_xlabel(
        "Scheduler"
    )

    ax2.set_ylabel(
        "Number of Deadline Misses"
    )

    ax2.set_title(
        "Deadline Miss Comparison"
    )

    st.pyplot(
        fig2
    )

    plt.close(fig2)


    # --------------------------------------------------------
    # CPU Utilization
    # --------------------------------------------------------

    st.subheader(
        "CPU Utilization"
    )

    fig3, ax3 = plt.subplots()

    ax3.bar(
        comparison["scheduler"],
        comparison["cpu_utilization"]
    )

    ax3.set_xlabel(
        "Scheduler"
    )

    ax3.set_ylabel(
        "CPU Utilization (%)"
    )

    ax3.set_title(
        "CPU Utilization Comparison"
    )

    st.pyplot(
        fig3
    )

    plt.close(fig3)


    # --------------------------------------------------------
    # Throughput
    # --------------------------------------------------------

    st.subheader(
        "Throughput"
    )

    fig4, ax4 = plt.subplots()

    ax4.bar(
        comparison["scheduler"],
        comparison["throughput"]
    )

    ax4.set_xlabel(
        "Scheduler"
    )

    ax4.set_ylabel(
        "Tasks / Time Unit"
    )

    ax4.set_title(
        "Throughput Comparison"
    )

    st.pyplot(
        fig4
    )

    plt.close(fig4)


    # ========================================================
    # EXECUTION TRACE
    # ========================================================

    st.header(
        "5. Execution Trace"
    )

    trace_scheduler = st.selectbox(
        "Select Scheduler Trace",
        [
            "FIFO",
            "EDF",
            "ML-EDF"
        ]
    )

    selected_results = results[
        trace_scheduler
    ]

    trace = selected_results[
        "simulation"
    ]["trace"]

    trace_data = []

    for event in trace:

        trace_data.append(
            {
                "Start": event.start_time,
                "End": event.end_time,
                "Task": (
                    f"T{event.task_id}"
                    if event.task_id is not None
                    else "-"
                ),
                "Event": event.event,
                "Duration": event.duration
            }
        )

    trace_df = pd.DataFrame(
        trace_data
    )

    st.dataframe(
        trace_df,
        use_container_width=True
    )


    # ========================================================
    # TRACE TIMELINE
    # ========================================================

    st.subheader(
        "Execution Timeline"
    )

    if not trace_df.empty:

        fig5, ax5 = plt.subplots()

        for index, event in enumerate(trace):

            duration = event.end_time - event.start_time

            label = (
                f"T{event.task_id}"
                if event.task_id is not None
                else event.event
            )

            ax5.barh(
                0,
                duration,
                left=event.start_time
            )

            if duration >= 1:

                ax5.text(
                    event.start_time + duration / 2,
                    0,
                    label,
                    ha="center",
                    va="center"
                )

        ax5.set_xlabel(
            "Virtual Time"
        )

        ax5.set_yticks([])

        ax5.set_title(
            f"{trace_scheduler} Execution Timeline"
        )

        st.pyplot(
            fig5
        )

        plt.close(fig5)


    # ========================================================
    # ML-EDF PREDICTIONS
    # ========================================================

    st.header(
        "6. ML-EDF Predictions"
    )

    ml_simulation = results[
        "ML-EDF"
    ]["simulation"]

    ml_scheduler = results[
        "ML-EDF"
    ]["simulation"]

    # --------------------------------------------------------
    # Extract predictions from the scheduler
    # --------------------------------------------------------

    ml_tasks = results[
        "ML-EDF"
    ]["tasks"]

    prediction_rows = []

    # Predictions are stored inside the scheduler object,
    # therefore obtain the scheduler through a fresh simulation
    # when necessary.

    prediction_scheduler = MLEDFScheduler(
        preemptive=False,
        risk_weight=risk_weight
    )

    prediction_scheduler.update_time(0)

    for task in ml_tasks:

        prediction = (
            prediction_scheduler.predict_task(
                task
            )
        )

        prediction_rows.append(
            {
                "Task ID":
                    task.task_id,

                "Deadline":
                    task.deadline,

                "Actual Execution":
                    task.execution_time,

                "Predicted Execution":
                    prediction[
                        "predicted_execution_time"
                    ],

                "Deadline-Miss Probability":
                    prediction[
                        "deadline_miss_probability"
                    ],

                "Predicted Slack":
                    prediction[
                        "remaining_slack"
                    ]
            }
        )

    prediction_df = pd.DataFrame(
        prediction_rows
    )

    if not prediction_df.empty:

        prediction_df[
            "Deadline-Miss Probability"
        ] = (
            prediction_df[
                "Deadline-Miss Probability"
            ] * 100
        ).round(2)

        st.dataframe(
            prediction_df,
            use_container_width=True
        )


    # ========================================================
    # SIMULATION INFORMATION
    # ========================================================

    st.header(
        "7. Simulation Information"
    )

    sim_data = []

    for name in [
        "FIFO",
        "EDF",
        "ML-EDF"
    ]:

        simulation = results[
            name
        ]["simulation"]

        sim_data.append(
            {
                "Scheduler":
                    name,

                "Total Time":
                    simulation[
                        "total_time"
                    ],

                "Context Switches":
                    simulation[
                        "context_switches"
                    ],

                "Preemptions":
                    simulation[
                        "preemptions"
                    ]
            }
        )

    simulation_df = pd.DataFrame(
        sim_data
    )

    st.dataframe(
        simulation_df,
        use_container_width=True
    )


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown("---")

    st.caption(
        "ML-Enhanced Real-Time Task Scheduler | "
        "FIFO + EDF + ML-EDF | "
        "Python + Machine Learning + Data Structures"
    )