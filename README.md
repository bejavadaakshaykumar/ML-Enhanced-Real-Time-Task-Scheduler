# ML-Enhanced Real-Time Task Scheduler

An intelligent real-time task scheduling system that combines classical CPU scheduling algorithms with Machine Learning to improve task prioritization under execution-time uncertainty and deadline constraints.

## Overview

Real-time systems execute tasks that often have strict deadlines. Traditional scheduling algorithms such as FIFO and Earliest Deadline First (EDF) use fixed scheduling rules and do not consider predicted execution behavior or deadline-miss risk.

This project implements and compares:

- FIFO (First-In, First-Out)
- EDF (Earliest Deadline First)
- ML-EDF (Machine Learning enhanced EDF)

The ML-EDF scheduler uses Machine Learning predictions to estimate execution time and deadline-miss probability, and combines these predictions with task deadline information to calculate a risk-aware scheduling priority.

The project also provides an interactive Streamlit dashboard for workload generation, scheduler comparison, performance visualization, and ML prediction analysis.

## Key Features

### Classical Scheduling

- FIFO scheduling
- Earliest Deadline First (EDF)
- Preemptive and non-preemptive scheduling support
- Custom Min-Heap implementation for priority-based scheduling
- Virtual clock for deterministic simulation
- Context-switch tracking
- Preemption tracking

### Machine Learning

Two Random Forest models are used:

1. **Execution-Time Regression** - Predicts the execution time of a task.
2. **Deadline-Miss Classification** - Predicts the probability that a task may miss its deadline.

ML-EDF combines these predictions with the current scheduling time to calculate remaining slack and risk-aware urgency.

### Simulation

The simulation engine supports:

- Task arrivals
- Ready-queue management
- Task execution
- Preemption
- Context switching
- Completion-time tracking
- Deadline-miss detection
- Waiting-time calculation
- Response-time calculation
- Turnaround-time calculation
- CPU utilization
- Throughput
- Tardiness

### Experimentation

The project compares FIFO, EDF, and ML-EDF using the same workload so that the scheduling algorithms can be evaluated under identical conditions.

Experiment results are saved as:

```text
results/scheduler_comparison.csv
results/scheduler_comparison.json
```

### Interactive Dashboard

The Streamlit dashboard provides:

- Workload configuration
- Task table
- Scheduler selection
- Scheduler performance comparison
- Waiting-time visualization
- Deadline-miss comparison
- CPU utilization
- Throughput
- Execution trace
- ML-EDF prediction information

## System Architecture

```text
                    +----------------------+
                    |   Workload Generator |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |      Task Model       |
                    | Arrival / Execution   |
                    | Deadline / Priority   |
                    +----------+-----------+
                               |
                 +-------------+-------------+
                 |             |             |
                 v             v             v
            +---------+   +---------+   +------------+
            |  FIFO   |   |   EDF   |   |   ML-EDF   |
            +----+----+   +----+----+   +-----+------+
                 |             |               |
                 |             |         +-----+------+
                 |             |         | ML Models  |
                 |             |         +-----+------+
                 |             |               |
                 +-------------+---------------+
                               |
                               v
                    +----------------------+
                    |  Simulation Engine   |
                    | Virtual Clock        |
                    | Context Switching    |
                    | Preemption           |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |       Metrics        |
                    | Waiting Time         |
                    | Response Time        |
                    | Deadline Misses      |
                    | CPU Utilization      |
                    | Throughput           |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Streamlit Dashboard  |
                    +----------------------+
```

## Project Structure

```text
ML_Task_Scheduler/
│
├── scheduler/
│   ├── __init__.py
│   ├── task.py
│   ├── virtual_clock.py
│   ├── fifo.py
│   ├── min_heap.py
│   ├── edf.py
│   ├── simulator.py
│   └── metrics.py
│
├── workload/
│   ├── __init__.py
│   ├── generator.py
│   ├── loader.py
│   └── validator.py
│
├── ml/
│   ├── __init__.py
│   ├── features.py
│   ├── dataset.py
│   ├── train_regression.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── train_classifier.py
│   ├── predict_deadline.py
│   ├── evaluate_classifier.py
│   └── ml_edf.py
│
├── experiments/
│   ├── __init__.py
│   └── run_experiment.py
│
├── dashboard/
│   └── app.py
│
├── tests/
│   ├── test_fifo.py
│   ├── test_min_heap.py
│   ├── test_edf.py
│   ├── test_simulator.py
│   └── test_ml_edf.py
│
├── data/
├── models/
├── results/
│
├── requirements.txt
├── config.yaml
└── README.md
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| NumPy | Numerical operations |
| Pandas | Dataset and result processing |
| Scikit-learn | Machine Learning models |
| Matplotlib | Visualization |
| Plotly | Interactive visualization |
| Streamlit | Web dashboard |
| Joblib | ML model serialization |
| PyYAML | Configuration management |
| Pytest | Automated testing |

## Scheduling Algorithms

### 1. FIFO

First-In, First-Out schedules tasks according to their arrival order.

```text
Task 1 -> Task 2 -> Task 3 -> Task 4
```

The task that enters the ready queue first is executed first.

### 2. EDF

Earliest Deadline First selects the task with the earliest deadline.

The custom Min-Heap uses the following ordering:

```text
Deadline
    |
    v
Priority
    |
    v
Arrival Time
    |
    v
Task ID
```

### 3. ML-EDF

ML-EDF extends EDF using Machine Learning predictions.

For each ready task, the system predicts:

```text
Predicted Execution Time
          +
Deadline-Miss Probability
          +
Remaining Slack
```

The scheduler calculates:

```text
Remaining Slack =
Deadline - Current Time - Predicted Execution Time
```

A risk-aware urgency score is calculated using:

```text
Risk Score =
Deadline-Miss Probability x Risk Weight

Urgency Score =
Remaining Slack - Risk Score
```

A smaller urgency score indicates higher scheduling urgency.

> ML-EDF is a project-specific ML-enhanced scheduling strategy rather than a standard scheduling algorithm.

## Machine Learning Pipeline

### Execution-Time Prediction

The regression model uses task information such as:

```text
Arrival Time
Deadline
Priority
Input Size
```

The target variable is:

```text
Execution Time
```

A Random Forest Regressor is trained and saved as:

```text
models/execution_time_model.pkl
```

### Deadline-Miss Prediction

The classification model predicts whether a task is expected to miss its deadline.

The model is saved as:

```text
models/deadline_miss_model.pkl
```

The ML-EDF scheduler uses the predicted probability rather than only a binary prediction.

## Workload Generation

The workload generator creates reproducible task sets using a random seed.

Each task contains information such as:

```text
Task ID
Arrival Time
Execution Time
Deadline
Priority
Task Type
Input Size
```

Task types include:

```text
CPU
IO
PERIODIC
APERIODIC
```

A fixed seed can be used to generate the same workload for repeatable experiments.

## Performance Metrics

The system evaluates scheduling performance using:

### Waiting Time

Time a task spends waiting before execution.

### Response Time

Time from task arrival until it first starts execution.

### Turnaround Time

```text
Completion Time - Arrival Time
```

### Tardiness

Amount of time by which a task completes after its deadline.

### Deadline Misses

Number of tasks that complete after their deadlines.

### CPU Utilization

Percentage of simulation time during which the CPU is executing tasks.

### Throughput

Number of completed tasks per unit of simulation time.

### Context Switches

Number of times the scheduler switches between tasks.

### Preemptions

Number of times a running task is interrupted so another task can execute.

## Running the Project Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ML_Task_Scheduler
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Train the Machine Learning Models

Generate the ML dataset:

```bash
python -m ml.dataset
```

Train the execution-time regression model:

```bash
python -m ml.train_regression
```

Evaluate the regression model:

```bash
python -m ml.evaluate
```

Train the deadline-miss classifier:

```bash
python -m ml.train_classifier
```

Evaluate the classifier:

```bash
python -m ml.evaluate_classifier
```

After training, the models should be available in:

```text
models/
├── execution_time_model.pkl
└── deadline_miss_model.pkl
```

## Run the Scheduler Experiment

Run the comparison experiment:

```bash
python -m experiments.run_experiment
```

The experiment compares:

```text
FIFO
EDF
ML-EDF
```

using the same workload.

Generated files include:

```text
data/experiment_workload.csv
results/scheduler_comparison.csv
results/scheduler_comparison.json
```

## Run Tests

Run the complete test suite:

```bash
python -m pytest -q
```

The tests cover:

- FIFO scheduler
- Min-Heap
- EDF scheduler
- Simulation engine
- ML-EDF scheduler

## Run the Streamlit Dashboard

Start the dashboard locally:

```bash
streamlit run dashboard/app.py
```

Streamlit will provide a local address such as:

```text
http://localhost:8501
```

Open that address in your browser.

## GitHub Deployment

The dashboard can be deployed using GitHub and Streamlit Community Cloud.

### Repository

Push the complete project to GitHub:

```bash
git init
git add .
git commit -m "Initial project commit"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

### Streamlit Configuration

In Streamlit Community Cloud, select:

```text
Repository: ML_Task_Scheduler
Branch: main
Main file path: dashboard/app.py
```

The platform installs dependencies from:

```text
requirements.txt
```

and runs:

```text
dashboard/app.py
```

## Reproducibility

Experiments use a fixed random seed when required.

For example:

```text
Seed = 42
```

Using the same workload allows FIFO, EDF, and ML-EDF to be compared under identical task conditions.

## Testing Strategy

The project follows unit-testing principles.

### Scheduler Tests

Verify that:

- Tasks are added correctly.
- Tasks are selected according to scheduler rules.
- Queues maintain the expected order.

### Heap Tests

Verify:

- Push operation
- Pop operation
- Peek operation
- Heap ordering
- Empty heap behavior

### Simulator Tests

Verify:

- Task execution
- Clock advancement
- Completion time
- Context switches
- Preemption
- Deadline handling

### ML-EDF Tests

Verify:

- ML predictions are generated.
- Tasks receive scheduling keys.
- The ML-EDF ready queue operates correctly.

## Example Workflow

```text
1. Generate workload
          |
          v
2. Create task dataset
          |
          v
3. Train ML models
          |
          v
4. Load trained models
          |
          v
5. Run FIFO
          |
          v
6. Run EDF
          |
          v
7. Run ML-EDF
          |
          v
8. Calculate metrics
          |
          v
9. Compare results
          |
          v
10. Visualize through Streamlit
```

## Project Goals

The main goals of this project are:

- Implement real-time scheduling algorithms.
- Understand CPU scheduling and deadline-based scheduling.
- Implement a custom Min-Heap data structure.
- Build a deterministic simulation environment.
- Apply Machine Learning to execution-time prediction.
- Apply Machine Learning to deadline-miss prediction.
- Develop an ML-enhanced scheduling strategy.
- Compare classical and ML-enhanced scheduling approaches.
- Demonstrate results through an interactive dashboard.

## Applications

Potential application areas include:

- Real-time operating systems
- Embedded systems
- IoT systems
- Industrial automation
- Robotics
- Edge computing
- Autonomous systems
- Real-time data processing
- CPU task scheduling research

## Future Enhancements

Possible future improvements include:

- Additional scheduling algorithms such as RM and DM
- Reinforcement-learning-based scheduling
- Online model retraining
- Dynamic task priorities
- Multi-core CPU simulation
- Energy-aware scheduling
- Resource-aware scheduling
- More advanced workload models
- Real-time task arrival streams
- Hardware/RTOS integration

## Author

**Bejavada Akshay Kumar**

B.Tech - Electronics and Communication Engineering

ANITS

## License

This project is intended for academic, learning, and demonstration purposes.
