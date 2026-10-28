# 🧠 ML-Enhanced Real-Time Task Scheduler

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-enhanced-real-time-task-scheduler-evvdqgmygygsopsey8eegz.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-20%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

An intelligent real-time task scheduling system that combines classical CPU scheduling algorithms with Machine Learning to improve task prioritization under execution-time uncertainty and deadline constraints.

## 🔍 Overview

Real-time systems execute tasks with strict deadlines. Traditional scheduling algorithms like FIFO and EDF use fixed rules and don't consider predicted execution behavior or deadline-miss risk.

This project implements, compares, and visualizes **three scheduling algorithms**:

| Algorithm | Strategy | Key Insight |
|-----------|----------|-------------|
| **FIFO** | First-In, First-Out | Baseline — tasks run in arrival order |
| **EDF** | Earliest Deadline First | Classical optimal — always picks nearest deadline |
| **ML-EDF** | ML-Enhanced EDF | Novel — uses ML predictions for risk-aware scheduling |

### Why Do EDF and ML-EDF Produce Different Results?

This is **by design**. EDF schedules purely by deadline. ML-EDF incorporates two additional signals:

1. **Predicted Execution Time** — estimates how long a task will take
2. **Deadline-Miss Probability** — estimates the risk of missing the deadline

These are combined into a **risk-aware urgency score**:

```
remaining_slack = deadline − current_time − predicted_execution_time
risk_score      = miss_probability × risk_weight
urgency_score   = remaining_slack − risk_score
```

A **lower urgency score** means **higher scheduling priority**. This means ML-EDF can reorder tasks compared to EDF — promoting tasks with tight slack or high miss probability, even if their raw deadline is later.

> **When risk_weight = 0**, ML-EDF uses only slack-based ordering (ignoring risk).
> **When risk_weight is high**, tasks with high miss probability get heavily promoted.

## ✨ Key Features

### Scheduling Engine
- FIFO, EDF, and ML-EDF schedulers with consistent interfaces
- Preemptive and non-preemptive scheduling modes
- Custom Min-Heap data structure for priority-based scheduling
- Virtual clock for deterministic simulation
- Context-switch and preemption tracking

### Machine Learning Pipeline
- **Random Forest Regressor** — predicts task execution time
- **Random Forest Classifier** — predicts deadline-miss probability
- Feature set: `arrival_time`, `deadline`, `priority`, `input_size`
- Model persistence with joblib (`.pkl` files)

### Interactive Dashboard
- Plotly-powered interactive charts (bar charts, Gantt timeline)
- EDF vs ML-EDF scheduling order comparison
- Per-task deadline met/missed status with color coding
- Actual vs predicted execution time visualization
- Risk analysis with urgency scores and miss probabilities
- Educational "How It Works" section explaining each algorithm
- Configurable workload size, random seed, and risk weight

### Simulation Metrics
- Waiting time, response time, turnaround time
- Deadline misses and miss rate
- CPU utilization and throughput
- Context switches and preemptions
- Tardiness (lateness beyond deadline)

## 🏗️ System Architecture

```
                    ┌──────────────────────┐
                    │   Workload Generator  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Task Model        │
                    │ arrival / execution   │
                    │ deadline / priority   │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 ▼             ▼             ▼
            ┌─────────┐  ┌─────────┐  ┌────────────┐
            │  FIFO   │  │   EDF   │  │   ML-EDF   │
            └────┬────┘  └────┬────┘  └─────┬──────┘
                 │             │             │
                 │             │       ┌─────┴──────┐
                 │             │       │ ML Models   │
                 │             │       │ Regression  │
                 │             │       │ Classifier  │
                 │             │       └─────┬──────┘
                 │             │             │
                 └─────────────┼─────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Simulation Engine   │
                    │ Virtual Clock         │
                    │ Context Switching     │
                    │ Preemption            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Metrics          │
                    │ Wait / Response Time  │
                    │ Deadline Misses       │
                    │ CPU Utilization       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Streamlit Dashboard  │
                    │  Plotly Visualizations │
                    └──────────────────────┘
```

## 📁 Project Structure

```
ML_Task_Scheduler/
│
├── scheduler/              # Core scheduling algorithms
│   ├── __init__.py
│   ├── task.py             # Task dataclass (arrival, exec, deadline, priority)
│   ├── virtual_clock.py    # Deterministic virtual clock
│   ├── fifo.py             # FIFO scheduler
│   ├── min_heap.py         # Custom min-heap with key functions
│   ├── edf.py              # Earliest Deadline First scheduler
│   ├── simulator.py        # Simulation engine (preemptive + non-preemptive)
│   └── metrics.py          # Performance metric calculations
│
├── workload/               # Workload generation and validation
│   ├── __init__.py
│   ├── generator.py        # Reproducible task workload generator
│   ├── loader.py           # CSV/JSON workload I/O
│   └── validator.py        # Task and workload validation
│
├── ml/                     # Machine learning pipeline
│   ├── __init__.py
│   ├── features.py         # Feature extraction
│   ├── dataset.py          # ML dataset generation
│   ├── train_regression.py # Execution time model training
│   ├── predict.py          # Execution time predictor
│   ├── evaluate.py         # Regression model evaluation
│   ├── train_classifier.py # Deadline-miss model training
│   ├── predict_deadline.py # Deadline-miss predictor
│   ├── evaluate_classifier.py # Classifier evaluation
│   └── ml_edf.py           # ML-Enhanced EDF scheduler
│
├── experiments/            # Experiment runner
│   ├── __init__.py
│   └── run_experiment.py   # CLI comparison of FIFO, EDF, ML-EDF
│
├── dashboard/              # Web interface
│   └── app.py              # Streamlit dashboard with Plotly charts
│
├── tests/                  # Test suite (20 tests)
│   ├── test_fifo.py        # FIFO ordering tests
│   ├── test_min_heap.py    # Heap operations tests
│   ├── test_edf.py         # EDF ordering tests
│   ├── test_simulator.py   # Simulation engine tests + edge cases
│   ├── test_workload.py    # Workload generation tests
│   └── test_ml_edf.py      # ML-EDF predictions, formula, edge cases
│
├── data/                   # Generated datasets
├── models/                 # Trained ML model files (.pkl)
├── results/                # Experiment output files
│
├── requirements.txt        # Python dependencies
├── config.yaml             # Project configuration
└── README.md
```

## 🛠️ Technologies

| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Core implementation |
| NumPy | Numerical operations |
| Pandas | Dataset and result processing |
| Scikit-learn | Random Forest models (regression + classification) |
| Plotly | Interactive charts and Gantt timelines |
| Streamlit | Web dashboard |
| Joblib | ML model serialization |
| Pytest | Automated testing (20 tests) |
| PyYAML | Configuration management |

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/bejavadaakshaykumar/ML-Enhanced-Real-Time-Task-Scheduler.git
cd ML-Enhanced-Real-Time-Task-Scheduler
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train ML Models (Optional — pre-trained models included)

```bash
python -m ml.dataset              # Generate training data
python -m ml.train_regression     # Train execution time predictor
python -m ml.train_classifier     # Train deadline-miss classifier
```

### 5. Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

### 6. Run Tests

```bash
python -m pytest tests/ -v
```

Expected output: **20 passed** ✅

### 7. Run CLI Experiment

```bash
python -m experiments.run_experiment
```

## 📊 Dashboard Sections

The Streamlit dashboard includes **8 sections**:

| # | Section | Description |
|---|---------|-------------|
| 1 | **Workload Overview** | Generated tasks with feasibility status |
| 2 | **Performance Comparison** | Key metrics with delta cards (ML-EDF vs EDF) |
| 3 | **Comparison Charts** | Interactive Plotly bar charts (wait time, misses, CPU, throughput) |
| 4 | **Execution Timeline** | Gantt chart showing task execution order with deadline status |
| 5 | **Per-Task Results** | Deadline met/missed per task for each scheduler |
| 6 | **ML Predictions** | Actual vs predicted execution time + EDF vs ML-EDF order comparison |
| 7 | **Why ML-EDF Differs** | Risk analysis with slack status, miss probability, urgency scores |
| 8 | **Simulation Summary** | Total time, context switches, preemptions per scheduler |

## 🧪 Test Coverage

| Test File | Tests | Covers |
|-----------|-------|--------|
| `test_fifo.py` | 1 | FIFO ordering |
| `test_min_heap.py` | 2 | Heap push/pop/peek operations |
| `test_edf.py` | 1 | EDF deadline-based ordering |
| `test_simulator.py` | 9 | Simulation engine, preemption, context switches, idle CPU, same-deadline tiebreaking, deadline miss detection |
| `test_ml_edf.py` | 7 | ML predictions, urgency formula verification, scheduling order tracking, state reset, delayed arrivals |
| `test_workload.py` | 3 | Workload generation, reproducibility, seed variation |
| **Total** | **20** | |

## 📈 Performance Metrics Explained

| Metric | Formula | Lower is Better? |
|--------|---------|:-:|
| Waiting Time | `start_time − arrival_time` | ✅ |
| Response Time | `first_run_time − arrival_time` | ✅ |
| Turnaround Time | `completion_time − arrival_time` | ✅ |
| Tardiness | `max(0, completion_time − deadline)` | ✅ |
| Deadline Misses | Count of tasks where `completion > deadline` | ✅ |
| CPU Utilization | `busy_time / total_time × 100` | ❌ (higher better) |
| Throughput | `completed_tasks / total_time` | ❌ (higher better) |

## 🔮 Future Enhancements

- Rate Monotonic (RM) and Deadline Monotonic (DM) schedulers
- Reinforcement learning-based scheduling
- Online model retraining during simulation
- Multi-core CPU simulation
- Energy-aware and resource-aware scheduling
- Real-time task arrival streams

## 👤 Author

**Bejavada Akshay Kumar**

B.Tech — Electronics and Communication Engineering, ANITS

## 📄 License

This project is intended for academic, learning, and demonstration purposes.
