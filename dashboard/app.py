# ============================================================
# ML-ENHANCED REAL-TIME TASK SCHEDULER
# STREAMLIT DASHBOARD
# ============================================================

import os
import sys

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

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
    page_title="ML-Enhanced Real-Time Task Scheduler",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS — Premium dark-mode styling
# ============================================================

st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Root variables */
    :root {
        --accent-blue: #4F8CFF;
        --accent-green: #34D399;
        --accent-red: #F87171;
        --accent-amber: #FBBF24;
        --accent-purple: #A78BFA;
        --bg-card: rgba(30, 33, 48, 0.6);
        --border-subtle: rgba(255, 255, 255, 0.06);
    }

    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Section headers */
    h1 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #4F8CFF 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem !important;
        margin-bottom: 0.2rem !important;
    }

    h2 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #E2E8F0 !important;
        border-bottom: 2px solid rgba(79, 140, 255, 0.3);
        padding-bottom: 0.5rem;
    }

    h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: #CBD5E1 !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 33, 48, 0.8) 0%, rgba(40, 45, 65, 0.6) 100%);
        border: 1px solid rgba(79, 140, 255, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8 !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #E2E8F0 !important;
    }

    /* Delta styling */
    [data-testid="stMetricDelta"] > div {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F1629 0%, #1A1F36 100%);
        border-right: 1px solid rgba(79, 140, 255, 0.1);
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #E2E8F0 !important;
        -webkit-text-fill-color: #E2E8F0 !important;
        background: none !important;
        font-size: 1.2rem !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        background: rgba(30, 33, 48, 0.5);
        border-radius: 8px;
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Info/Success/Warning boxes */
    .stAlert {
        border-radius: 10px;
        border-left-width: 4px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500;
        border-radius: 8px 8px 0 0;
    }

    /* Divider */
    hr {
        border-color: rgba(79, 140, 255, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE & PROJECT EXPLANATION
# ============================================================

st.title("🧠 ML-Enhanced Real-Time Task Scheduler")

st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(79, 140, 255, 0.08) 0%, rgba(167, 139, 250, 0.08) 100%);
    border: 1px solid rgba(79, 140, 255, 0.15);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 1.5rem;
">
    <p style="color: #CBD5E1; font-family: 'Inter', sans-serif; font-size: 0.95rem; line-height: 1.7; margin: 0;">
        This project compares <strong style="color: #4F8CFF;">three CPU scheduling algorithms</strong> on the same
        task workload. <strong>FIFO</strong> schedules tasks in arrival order.
        <strong>EDF</strong> (Earliest Deadline First) always picks the task with the earliest deadline.
        <strong>ML-EDF</strong> enhances EDF with Machine Learning — it predicts each task's execution time
        and deadline-miss probability, then uses a <em>risk-aware urgency score</em> to
        prioritize tasks that are most likely to miss their deadlines.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# HOW IT WORKS — Expandable
# ============================================================

with st.expander("📖 How It Works — Algorithm Details", expanded=False):

    col_algo1, col_algo2, col_algo3 = st.columns(3)

    with col_algo1:
        st.markdown("""
        ### 🟦 FIFO
        **First-In, First-Out**

        - Simplest scheduling algorithm
        - Tasks execute in arrival order
        - No priority or deadline awareness
        - Used as baseline comparison
        - **Key**: `arrival_order`
        """)

    with col_algo2:
        st.markdown("""
        ### 🟩 EDF
        **Earliest Deadline First**

        - Classical real-time scheduling
        - Always picks task with **nearest deadline**
        - Optimal for single-processor preemptive systems
        - Tiebreaker: priority → arrival → task_id
        - **Key**: `(deadline, priority, arrival, id)`
        """)

    with col_algo3:
        st.markdown("""
        ### 🟪 ML-EDF
        **ML-Enhanced EDF**

        - Extends EDF with ML predictions
        - Predicts **execution time** (regression)
        - Predicts **deadline-miss probability** (classifier)
        - Computes **risk-aware urgency score**:

        ```
        slack = deadline − time − pred_exec
        risk = miss_prob × risk_weight
        urgency = slack − risk
        ```
        - Lower urgency → scheduled first
        """)

    st.info(
        "💡 **Key Insight**: ML-EDF can reorder tasks compared to EDF because it considers "
        "predicted execution time and deadline-miss risk, not just the raw deadline. "
        "Tasks with tight slack or high miss probability get promoted."
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## ⚙️ Experiment Configuration")

st.sidebar.markdown("---")

num_tasks = st.sidebar.slider(
    "📋 Number of Tasks",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
    help="How many tasks to generate for the simulation."
)

seed = st.sidebar.number_input(
    "🎲 Random Seed",
    min_value=0,
    value=42,
    step=1,
    help="Controls reproducibility. Same seed = same workload."
)

risk_weight = st.sidebar.slider(
    "⚠️ ML-EDF Risk Weight",
    min_value=0.0,
    max_value=20.0,
    value=10.0,
    step=0.5,
    help=(
        "Controls how much deadline-miss probability affects "
        "ML-EDF scheduling. 0 = ignore risk (slack-only), "
        "20 = heavily penalize risky tasks."
    )
)

st.sidebar.markdown("---")

run_button = st.sidebar.button(
    "🚀 Run Simulation",
    type="primary",
    use_container_width=True
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style="
    background: rgba(79, 140, 255, 0.06);
    border: 1px solid rgba(79, 140, 255, 0.12);
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 0.78rem;
    color: #94A3B8;
    line-height: 1.5;
">
    <strong style="color: #CBD5E1;">Quick Start:</strong><br>
    1. Adjust task count & seed<br>
    2. Set risk weight for ML-EDF<br>
    3. Click <strong style="color: #4F8CFF;">Run Simulation</strong><br>
    4. Compare results below
</div>
""", unsafe_allow_html=True)


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

if "ml_scheduler_ref" not in st.session_state:
    st.session_state.ml_scheduler_ref = None

if "edf_task_order" not in st.session_state:
    st.session_state.edf_task_order = []


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
# COLOR PALETTE
# ============================================================

COLORS = {
    "FIFO": "#4F8CFF",
    "EDF": "#34D399",
    "ML-EDF": "#A78BFA"
}


# ============================================================
# RUN EXPERIMENT
# ============================================================

if run_button:

    with st.spinner(
        "⏳ Generating workload and running simulation..."
    ):

        workload = generate_workload(
            num_tasks=num_tasks,
            seed=int(seed)
        )

        ml_scheduler = MLEDFScheduler(
            preemptive=False,
            risk_weight=risk_weight
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
                ml_scheduler
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

        # Extract EDF execution order from trace
        edf_trace = results["EDF"]["simulation"]["trace"]
        edf_order = []
        seen = set()
        for event in edf_trace:
            if event.event == "RUN" and event.task_id is not None:
                if event.task_id not in seen:
                    edf_order.append(event.task_id)
                    seen.add(event.task_id)

        st.session_state.workload = workload
        st.session_state.comparison = comparison
        st.session_state.results = results
        st.session_state.simulation_done = True
        st.session_state.ml_scheduler_ref = ml_scheduler
        st.session_state.edf_task_order = edf_order


# ============================================================
# DASHBOARD — Waiting state
# ============================================================

if not st.session_state.simulation_done:

    st.markdown("---")

    st.markdown("""
    <div style="
        text-align: center;
        padding: 60px 20px;
        color: #64748B;
        font-family: 'Inter', sans-serif;
    ">
        <div style="font-size: 3rem; margin-bottom: 16px;">⚙️</div>
        <div style="font-size: 1.1rem; font-weight: 500; color: #94A3B8; margin-bottom: 8px;">
            No Simulation Running
        </div>
        <div style="font-size: 0.9rem;">
            Configure the experiment in the sidebar and click
            <strong style="color: #4F8CFF;">Run Simulation</strong> to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ============================================================
# DASHBOARD — Results
# ============================================================

comparison = st.session_state.comparison
results = st.session_state.results
workload = st.session_state.workload
ml_scheduler_ref = st.session_state.ml_scheduler_ref
edf_task_order = st.session_state.edf_task_order

st.success("✅ Simulation completed successfully!")

st.markdown("---")


# ========================================================
# SECTION 1: WORKLOAD OVERVIEW
# ========================================================

st.header("1️⃣ Workload Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Tasks",
        len(workload)
    )

with col2:
    st.metric(
        "Arrival Range",
        f"{min(t.arrival_time for t in workload)} – "
        f"{max(t.arrival_time for t in workload)}"
    )

with col3:
    st.metric(
        "Execution Range",
        f"{min(t.execution_time for t in workload)} – "
        f"{max(t.execution_time for t in workload)}"
    )

with col4:
    st.metric(
        "Deadline Range",
        f"{min(t.deadline for t in workload)} – "
        f"{max(t.deadline for t in workload)}"
    )


# --------------------------------------------------------
# Workload Table with Deadline Feasibility
# --------------------------------------------------------

st.subheader("📋 Generated Task Workload")

workload_data = []

for task in workload:

    available_time = task.deadline - task.arrival_time
    feasible = "✅ Feasible" if task.execution_time <= available_time else "⚠️ Tight"

    workload_data.append(
        {
            "Task ID": task.task_id,
            "Arrival": task.arrival_time,
            "Execution": task.execution_time,
            "Deadline": task.deadline,
            "Available Time": available_time,
            "Feasibility": feasible,
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
    use_container_width=True,
    hide_index=True
)

feasible_count = sum(1 for t in workload if t.execution_time <= t.deadline - t.arrival_time)
tight_count = len(workload) - feasible_count

st.caption(
    f"✅ {feasible_count} feasible tasks (execution ≤ available time)  •  "
    f"⚠️ {tight_count} tight tasks (execution > available time, may miss deadline)"
)


st.markdown("---")


# ========================================================
# SECTION 2: PERFORMANCE COMPARISON
# ========================================================

st.header("2️⃣ Scheduler Performance Comparison")

# --------------------------------------------------------
# Summary Metrics Cards
# --------------------------------------------------------

st.subheader("📊 Key Metrics at a Glance")

# Use ML-EDF as primary, show delta vs EDF
edf_row = comparison[comparison["scheduler"] == "EDF"].iloc[0]
ml_row = comparison[comparison["scheduler"] == "ML-EDF"].iloc[0]
fifo_row = comparison[comparison["scheduler"] == "FIFO"].iloc[0]

c1, c2, c3, c4 = st.columns(4)

with c1:
    delta_wait = ml_row["average_waiting_time"] - edf_row["average_waiting_time"]
    st.metric(
        "Avg Waiting Time (ML-EDF)",
        f"{ml_row['average_waiting_time']:.2f}",
        delta=f"{delta_wait:+.2f} vs EDF",
        delta_color="inverse"
    )

with c2:
    delta_resp = ml_row["average_response_time"] - edf_row["average_response_time"]
    st.metric(
        "Avg Response Time (ML-EDF)",
        f"{ml_row['average_response_time']:.2f}",
        delta=f"{delta_resp:+.2f} vs EDF",
        delta_color="inverse"
    )

with c3:
    delta_miss = int(ml_row["deadline_misses"]) - int(edf_row["deadline_misses"])
    st.metric(
        "Deadline Misses (ML-EDF)",
        int(ml_row["deadline_misses"]),
        delta=f"{delta_miss:+d} vs EDF",
        delta_color="inverse"
    )

with c4:
    st.metric(
        "CPU Utilization (ML-EDF)",
        f"{ml_row['cpu_utilization']:.1f}%"
    )


# --------------------------------------------------------
# Full Comparison Table
# --------------------------------------------------------

st.subheader("📈 Full Comparison Table")

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

display_df = display_df.rename(
    columns={
        "scheduler": "Scheduler",
        "average_waiting_time": "Avg Wait",
        "average_response_time": "Avg Response",
        "average_turnaround_time": "Avg Turnaround",
        "average_tardiness": "Avg Tardiness",
        "deadline_misses": "Deadline Misses",
        "deadline_miss_rate": "Miss Rate (%)",
        "cpu_utilization": "CPU Util (%)",
        "throughput": "Throughput",
        "context_switches": "Ctx Switches",
        "preemptions": "Preemptions",
        "total_time": "Total Time"
    }
)

# Round numeric columns
for col in display_df.columns:
    if col != "Scheduler":
        display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
        display_df[col] = display_df[col].round(3)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


st.markdown("---")


# ========================================================
# SECTION 3: COMPARISON CHARTS (Plotly)
# ========================================================

st.header("3️⃣ Comparison Charts")

chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
    "⏱️ Waiting Time",
    "❌ Deadline Misses",
    "💻 CPU Utilization",
    "📊 Throughput"
])

scheduler_names = comparison["scheduler"].tolist()
color_list = [COLORS[s] for s in scheduler_names]

with chart_tab1:
    fig = go.Figure(data=[
        go.Bar(
            x=scheduler_names,
            y=comparison["average_waiting_time"],
            marker_color=color_list,
            text=comparison["average_waiting_time"].round(2),
            textposition="outside",
            textfont=dict(size=14, color="#E2E8F0")
        )
    ])
    fig.update_layout(
        title="Average Waiting Time",
        yaxis_title="Time Units",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#CBD5E1"),
        height=400,
        margin=dict(t=50, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

with chart_tab2:
    fig = go.Figure(data=[
        go.Bar(
            x=scheduler_names,
            y=comparison["deadline_misses"],
            marker_color=color_list,
            text=comparison["deadline_misses"].astype(int),
            textposition="outside",
            textfont=dict(size=14, color="#E2E8F0")
        )
    ])
    fig.update_layout(
        title="Deadline Misses",
        yaxis_title="Number of Misses",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#CBD5E1"),
        height=400,
        margin=dict(t=50, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

with chart_tab3:
    fig = go.Figure(data=[
        go.Bar(
            x=scheduler_names,
            y=comparison["cpu_utilization"],
            marker_color=color_list,
            text=comparison["cpu_utilization"].round(1),
            textposition="outside",
            textfont=dict(size=14, color="#E2E8F0"),
            texttemplate="%{text}%"
        )
    ])
    fig.update_layout(
        title="CPU Utilization",
        yaxis_title="Utilization (%)",
        yaxis_range=[0, 105],
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#CBD5E1"),
        height=400,
        margin=dict(t=50, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

with chart_tab4:
    fig = go.Figure(data=[
        go.Bar(
            x=scheduler_names,
            y=comparison["throughput"],
            marker_color=color_list,
            text=comparison["throughput"].round(4),
            textposition="outside",
            textfont=dict(size=14, color="#E2E8F0")
        )
    ])
    fig.update_layout(
        title="Throughput (Tasks / Time Unit)",
        yaxis_title="Throughput",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#CBD5E1"),
        height=400,
        margin=dict(t=50, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)


st.markdown("---")


# ========================================================
# SECTION 4: EXECUTION TIMELINE (Gantt Chart)
# ========================================================

st.header("4️⃣ Execution Timeline")

trace_scheduler = st.selectbox(
    "Select Scheduler to Visualize",
    ["FIFO", "EDF", "ML-EDF"],
    index=2
)

selected_results = results[trace_scheduler]
trace = selected_results["simulation"]["trace"]
sim_tasks = selected_results["tasks"]

# --------------------------------------------------------
# Gantt chart using Plotly
# --------------------------------------------------------

if trace:

    gantt_data = []

    for event in trace:

        if event.task_id is not None:
            label = f"T{event.task_id}"

            # Check if this task missed deadline
            task_obj = None
            for t in sim_tasks:
                if t.task_id == event.task_id:
                    task_obj = t
                    break

            missed = False
            if task_obj and task_obj.completion_time is not None:
                missed = task_obj.completion_time > task_obj.deadline

            gantt_data.append({
                "Task": label,
                "Start": event.start_time,
                "End": event.end_time,
                "Duration": event.end_time - event.start_time,
                "Event": event.event,
                "Missed": missed
            })

        else:
            gantt_data.append({
                "Task": "IDLE",
                "Start": event.start_time,
                "End": event.end_time,
                "Duration": event.end_time - event.start_time,
                "Event": event.event,
                "Missed": False
            })

    if gantt_data:

        fig_gantt = go.Figure()

        # Color palette for tasks
        unique_tasks = sorted(set(d["Task"] for d in gantt_data if d["Task"] != "IDLE"))
        task_colors = {}
        palette = px.colors.qualitative.Set2
        for i, task in enumerate(unique_tasks):
            task_colors[task] = palette[i % len(palette)]

        for item in gantt_data:
            if item["Task"] == "IDLE":
                color = "rgba(100, 116, 139, 0.3)"
                border_color = "rgba(100, 116, 139, 0.5)"
            elif item["Missed"]:
                color = "rgba(248, 113, 113, 0.7)"
                border_color = "#F87171"
            else:
                color = task_colors.get(item["Task"], "#4F8CFF")
                border_color = color

            fig_gantt.add_trace(
                go.Bar(
                    x=[item["Duration"]],
                    y=[trace_scheduler],
                    base=item["Start"],
                    orientation="h",
                    name=item["Task"],
                    text=item["Task"] if item["Duration"] >= 1 else "",
                    textposition="inside",
                    textfont=dict(size=10, color="white"),
                    marker=dict(
                        color=color,
                        line=dict(width=1, color=border_color)
                    ),
                    hovertemplate=(
                        f"<b>{item['Task']}</b><br>"
                        f"Time: {item['Start']} → {item['End']}<br>"
                        f"Duration: {item['Duration']}<br>"
                        f"{'⚠️ Deadline Missed' if item['Missed'] else '✅ On Time'}"
                        "<extra></extra>"
                    ),
                    showlegend=False
                )
            )

        fig_gantt.update_layout(
            title=f"{trace_scheduler} — Execution Gantt Chart",
            xaxis_title="Virtual Time",
            yaxis_showticklabels=False,
            barmode="stack",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#CBD5E1"),
            height=180,
            margin=dict(t=50, b=40, l=20, r=20),
            xaxis=dict(dtick=5)
        )

        st.plotly_chart(fig_gantt, use_container_width=True)

    st.caption("🔴 Red bars = task that missed its deadline  •  Gray = CPU idle")


# --------------------------------------------------------
# Execution Trace Table
# --------------------------------------------------------

with st.expander("📜 Detailed Execution Trace", expanded=False):

    trace_data = []

    for event in trace:

        trace_data.append(
            {
                "Start": event.start_time,
                "End": event.end_time,
                "Task": (
                    f"T{event.task_id}"
                    if event.task_id is not None
                    else "—"
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
        use_container_width=True,
        hide_index=True
    )


st.markdown("---")


# ========================================================
# SECTION 5: PER-TASK RESULTS — Deadline Met / Missed
# ========================================================

st.header("5️⃣ Per-Task Deadline Results")

task_tab1, task_tab2, task_tab3 = st.tabs([
    "🟦 FIFO Results",
    "🟩 EDF Results",
    "🟪 ML-EDF Results"
])

def build_task_results_df(scheduler_tasks):
    """Build a per-task results dataframe with deadline status."""

    rows = []

    for task in sorted(scheduler_tasks, key=lambda t: t.task_id):

        if task.completion_time is None:
            continue

        missed = task.completion_time > task.deadline

        rows.append({
            "Task ID": task.task_id,
            "Arrival": task.arrival_time,
            "Execution": task.execution_time,
            "Deadline": task.deadline,
            "Start": task.start_time,
            "Completion": task.completion_time,
            "Waiting": task.waiting_time,
            "Response": task.response_time,
            "Turnaround": task.turnaround_time,
            "Tardiness": task.tardiness,
            "Status": "❌ Missed" if missed else "✅ Met"
        })

    return pd.DataFrame(rows)

with task_tab1:
    fifo_tasks_df = build_task_results_df(results["FIFO"]["tasks"])
    st.dataframe(fifo_tasks_df, use_container_width=True, hide_index=True)
    missed_count = (fifo_tasks_df["Status"] == "❌ Missed").sum()
    st.caption(f"FIFO: {len(fifo_tasks_df) - missed_count} met  •  {missed_count} missed")

with task_tab2:
    edf_tasks_df = build_task_results_df(results["EDF"]["tasks"])
    st.dataframe(edf_tasks_df, use_container_width=True, hide_index=True)
    missed_count = (edf_tasks_df["Status"] == "❌ Missed").sum()
    st.caption(f"EDF: {len(edf_tasks_df) - missed_count} met  •  {missed_count} missed")

with task_tab3:
    ml_tasks_df = build_task_results_df(results["ML-EDF"]["tasks"])
    st.dataframe(ml_tasks_df, use_container_width=True, hide_index=True)
    missed_count = (ml_tasks_df["Status"] == "❌ Missed").sum()
    st.caption(f"ML-EDF: {len(ml_tasks_df) - missed_count} met  •  {missed_count} missed")


st.markdown("---")


# ========================================================
# SECTION 6: ML-EDF PREDICTIONS & COMPARISON WITH EDF
# ========================================================

st.header("6️⃣ ML-EDF Predictions & EDF Comparison")

st.markdown("""
This section shows what ML-EDF "sees" for each task: the predicted execution time,
deadline-miss probability, remaining slack, and urgency score. Compare the
**ML-EDF scheduling order** with the **EDF scheduling order** to understand
how ML predictions change task prioritization.
""")

# --------------------------------------------------------
# Build prediction table from the actual scheduler
# --------------------------------------------------------

ml_tasks = results["ML-EDF"]["tasks"]

prediction_rows = []

# Re-run predictions at time=0 for display purposes
# (the scheduler already ran, but we want a clean view)
display_scheduler = MLEDFScheduler(
    preemptive=False,
    risk_weight=risk_weight
)
display_scheduler.update_time(0)

for task in sorted(ml_tasks, key=lambda t: t.task_id):

    prediction = display_scheduler.predict_task(task)

    available_time = task.deadline - task.arrival_time

    prediction_rows.append({
        "Task ID": task.task_id,
        "Deadline": task.deadline,
        "Actual Exec": task.execution_time,
        "Predicted Exec": prediction["predicted_execution_time"],
        "Exec Error": abs(prediction["predicted_execution_time"] - task.execution_time),
        "Miss Prob (%)": round(prediction["deadline_miss_probability"] * 100, 1),
        "Slack": prediction["remaining_slack"],
        "Urgency": round(prediction["urgency_score"], 2),
        "Feasible?": "✅" if task.execution_time <= available_time else "⚠️"
    })

prediction_df = pd.DataFrame(prediction_rows)

st.dataframe(
    prediction_df,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------------
# EDF vs ML-EDF Scheduling Order Comparison
# --------------------------------------------------------

st.subheader("🔄 Scheduling Order: EDF vs ML-EDF")

# Get ML-EDF order from trace
ml_trace = results["ML-EDF"]["simulation"]["trace"]
ml_order = []
ml_seen = set()
for event in ml_trace:
    if event.event == "RUN" and event.task_id is not None:
        if event.task_id not in ml_seen:
            ml_order.append(event.task_id)
            ml_seen.add(event.task_id)

col_edf, col_ml = st.columns(2)

with col_edf:
    st.markdown("**EDF Order** (by deadline)")
    order_str = " → ".join([f"T{tid}" for tid in edf_task_order])
    st.code(order_str, language=None)

with col_ml:
    st.markdown("**ML-EDF Order** (by urgency score)")
    order_str = " → ".join([f"T{tid}" for tid in ml_order])
    st.code(order_str, language=None)

# Find differences
if edf_task_order == ml_order:
    st.success(
        "✅ **Same ordering!** ML-EDF and EDF scheduled tasks in the same order "
        "for this workload. The ML predictions did not change the scheduling decisions."
    )
else:
    # Find which tasks were reordered
    reordered = []
    for i, (edf_id, ml_id) in enumerate(zip(edf_task_order, ml_order)):
        if edf_id != ml_id:
            reordered.append(i + 1)

    st.info(
        f"🔄 **Different ordering!** ML-EDF reordered tasks at position(s): "
        f"{', '.join(str(p) for p in reordered[:10])}. "
        f"This means the ML predictions (execution time + deadline-miss risk) "
        f"caused the scheduler to prioritize certain tasks differently than "
        f"pure deadline-first ordering."
    )


# --------------------------------------------------------
# Prediction Accuracy Chart
# --------------------------------------------------------

st.subheader("🎯 Execution Time: Actual vs Predicted")

fig_pred = go.Figure()

task_ids = prediction_df["Task ID"].tolist()

fig_pred.add_trace(go.Bar(
    name="Actual",
    x=[f"T{tid}" for tid in task_ids],
    y=prediction_df["Actual Exec"],
    marker_color="#34D399",
    opacity=0.8
))

fig_pred.add_trace(go.Bar(
    name="ML Predicted",
    x=[f"T{tid}" for tid in task_ids],
    y=prediction_df["Predicted Exec"],
    marker_color="#A78BFA",
    opacity=0.8
))

fig_pred.update_layout(
    barmode="group",
    title="Execution Time — Actual vs ML Prediction",
    xaxis_title="Task",
    yaxis_title="Execution Time",
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#CBD5E1"),
    height=400,
    margin=dict(t=50, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig_pred, use_container_width=True)

# Prediction accuracy summary
avg_error = prediction_df["Exec Error"].mean()
max_error = prediction_df["Exec Error"].max()

st.caption(
    f"Average prediction error: **{avg_error:.2f}** time units  •  "
    f"Max error: **{max_error}** time units"
)


st.markdown("---")


# ========================================================
# SECTION 7: WHY ML-EDF SCHEDULED DIFFERENTLY
# ========================================================

st.header("7️⃣ Why ML-EDF Makes Different Decisions")

st.markdown("""
ML-EDF uses a **risk-aware urgency score** instead of just the deadline.
Here's what drives the differences:
""")

# Build insight rows
insight_rows = []

for task in sorted(ml_tasks, key=lambda t: t.task_id):

    pred = display_scheduler.get_prediction(task.task_id)
    if pred is None:
        continue

    available = task.deadline - task.arrival_time

    # Determine classification
    if pred["deadline_miss_probability"] > 0.5:
        risk_label = "🔴 High Risk"
    elif pred["deadline_miss_probability"] > 0.2:
        risk_label = "🟡 Medium Risk"
    else:
        risk_label = "🟢 Low Risk"

    if pred["remaining_slack"] < 0:
        slack_label = "⚠️ Negative"
    elif pred["remaining_slack"] <= 2:
        slack_label = "🟡 Tight"
    else:
        slack_label = "🟢 Comfortable"

    insight_rows.append({
        "Task ID": task.task_id,
        "Deadline": task.deadline,
        "Predicted Slack": pred["remaining_slack"],
        "Slack Status": slack_label,
        "Miss Prob": f"{pred['deadline_miss_probability']:.1%}",
        "Risk Level": risk_label,
        "Urgency Score": round(pred["urgency_score"], 2),
        "Actual Missed?": "❌ Yes" if (task.completion_time and task.completion_time > task.deadline) else "✅ No"
    })

insight_df = pd.DataFrame(insight_rows)

st.dataframe(
    insight_df,
    use_container_width=True,
    hide_index=True
)

st.markdown(f"""
> **Risk Weight = {risk_weight}** — Each 10% increase in deadline-miss probability
> reduces the urgency score by **{risk_weight * 0.1:.1f}** units, effectively
> promoting risky tasks to be scheduled earlier.
""")


st.markdown("---")


# ========================================================
# SECTION 8: SIMULATION SUMMARY
# ========================================================

st.header("8️⃣ Simulation Summary")

sim_data = []

for name in ["FIFO", "EDF", "ML-EDF"]:

    simulation = results[name]["simulation"]

    sim_data.append(
        {
            "Scheduler": name,
            "Total Time": simulation["total_time"],
            "Context Switches": simulation["context_switches"],
            "Preemptions": simulation["preemptions"],
            "Tasks Completed": len([
                t for t in results[name]["tasks"]
                if t.completion_time is not None
            ])
        }
    )

simulation_df = pd.DataFrame(sim_data)

st.dataframe(
    simulation_df,
    use_container_width=True,
    hide_index=True
)


st.markdown("---")


# ========================================================
# FOOTER
# ========================================================

st.markdown("""
<div style="
    text-align: center;
    padding: 20px;
    color: #64748B;
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    border-top: 1px solid rgba(79, 140, 255, 0.1);
    margin-top: 2rem;
">
    <strong style="color: #94A3B8;">ML-Enhanced Real-Time Task Scheduler</strong><br>
    FIFO • EDF • ML-EDF &nbsp;|&nbsp;
    Python • Scikit-learn • Streamlit • Plotly<br>
    <em>Built by Bejavada Akshay Kumar — ANITS</em>
</div>
""", unsafe_allow_html=True)