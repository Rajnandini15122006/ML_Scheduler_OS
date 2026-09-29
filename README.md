# ML-Guided CPU Scheduler

An intelligent CPU scheduling simulation system that combines **classical CPU scheduling algorithms** with **Machine Learning-based workload classification** to investigate whether workload-aware scheduling can improve CPU scheduling performance and fairness.

The project simulates a realistic set of processes, classifies their workloads using a **Random Forest classifier**, applies an ML-guided dynamic scheduling policy, and compares its behavior against traditional scheduling algorithms such as:

- FCFS — First Come First Serve
- Round Robin
- Priority Scheduling
- Shortest Job First (SJF)
- ML-Guided Scheduling

The system evaluates scheduling performance using:

- Average Waiting Time
- Average Turnaround Time
- Average Response Time
- Jain's Fairness Index
- CPU Service Fairness

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Motivation](#motivation)
3. [Problem Statement](#problem-statement)
4. [Project Objectives](#project-objectives)
5. [Key Idea](#key-idea)
6. [System Architecture](#system-architecture)
7. [Project Workflow](#project-workflow)
8. [Workload Types](#workload-types)
9. [Dataset](#dataset)
10. [Dataset Features](#dataset-features)
11. [Machine Learning Model](#machine-learning-model)
12. [ML Features](#ml-features)
13. [Model Training](#model-training)
14. [Model Evaluation](#model-evaluation)
15. [Classical Scheduling Algorithms](#classical-scheduling-algorithms)
16. [ML-Guided Scheduler](#ml-guided-scheduler)
17. [Dynamic Scheduling Score](#dynamic-scheduling-score)
18. [Fairness Mechanism](#fairness-mechanism)
19. [Jain's Fairness Index](#jains-fairness-index)
20. [Performance Metrics](#performance-metrics)
21. [Project Structure](#project-structure)
22. [Technology Stack](#technology-stack)
23. [Environment Requirements](#environment-requirements)
24. [Installation](#installation)
25. [Running the Project](#running-the-project)
26. [Running Individual Components](#running-individual-components)
27. [Complete Execution Workflow](#complete-execution-workflow)
28. [Output Files](#output-files)
29. [Current Experimental Results](#current-experimental-results)
30. [Interpreting the Results](#interpreting-the-results)
31. [Visualization and Charts](#visualization-and-charts)
32. [Reproducing the Experiment](#reproducing-the-experiment)
33. [Troubleshooting](#troubleshooting)
34. [Common Errors](#common-errors)
35. [Limitations](#limitations)
36. [Future Improvements](#future-improvements)
37. [Possible Research Extensions](#possible-research-extensions)
38. [GitHub Setup](#github-setup)
39. [Contributing](#contributing)
40. [Conclusion](#conclusion)

---

# Project Overview

## ML-Guided CPU Scheduler

Traditional CPU scheduling algorithms generally use fixed rules.

For example:

- FCFS schedules according to arrival order.
- SJF prioritizes processes with shorter CPU bursts.
- Priority scheduling uses a predefined priority value.
- Round Robin uses a fixed time quantum.

However, real workloads can have different characteristics.

A process may be:

- CPU-intensive
- I/O-intensive
- Machine-learning related
- Mixed workload

A scheduler that understands the workload characteristics can potentially make more informed scheduling decisions.

This project explores that idea by introducing a **Machine Learning workload classification layer** before scheduling.

The ML model analyzes process characteristics and predicts the workload type:

```text
Process Information
        |
        v
Random Forest Classifier
        |
        v
Predicted Workload
        |
        +----------------+
        |                |
        v                v
CPU_BOUND          IO_BOUND
        |
        +----------------+
        |
        v
ML_GUIDED Scheduler
        |
        v
CPU Ready Queue
        |
        v
Execution
```

The resulting scheduler is then evaluated against traditional CPU scheduling algorithms.

---

# Motivation

Operating systems traditionally make scheduling decisions using predefined scheduling policies.

These policies work well for standard scenarios but do not explicitly understand the type of workload being executed.

For example:

A CPU-bound process may require long continuous CPU execution.

An I/O-bound process may benefit from receiving CPU time quickly and returning to I/O operations.

An ML-training workload may have different resource characteristics from a simple interactive process.

Therefore, the central idea of this project is:

> Can workload characteristics predicted using Machine Learning be incorporated into CPU scheduling decisions?

This project provides an experimental framework to investigate that question.

---

# Problem Statement

Design and implement a CPU scheduler that:

1. Generates or accepts realistic process workloads.
2. Extracts workload-related process features.
3. Classifies processes using Machine Learning.
4. Predicts workload categories.
5. Uses those predictions as part of a scheduling policy.
6. Simulates CPU scheduling.
7. Measures scheduling performance.
8. Measures fairness.
9. Compares ML-guided scheduling with traditional algorithms.

---

# Project Objectives

The project has the following objectives:

### 1. Generate realistic workloads

Create a dataset containing process-level characteristics such as:

- Arrival time
- CPU burst
- I/O burst
- Priority
- Previous runtime
- Waiting time
- Context switches
- Workload type

### 2. Train a workload classifier

Use a Random Forest classifier to classify processes into workload categories.

### 3. Implement traditional schedulers

Implement:

- FCFS
- Round Robin
- Priority
- SJF

### 4. Implement ML-guided scheduling

Use predicted workload type and other scheduling features to calculate a dynamic scheduling score.

### 5. Introduce fairness analysis

Use Jain's Fairness Index to evaluate fairness.

### 6. Compare all schedulers

Produce a common comparison table containing scheduling metrics.

---

# Key Idea

The main innovation explored by this project is the integration of:

```text
Machine Learning
       +
Operating System Scheduling
       +
Fairness Analysis
```

Instead of treating every process identically, the ML-guided scheduler first attempts to understand the workload.

For every process, the system predicts:

```text
CPU_BOUND
IO_BOUND
ML_TRAINING
MIXED
```

The prediction is accompanied by a confidence score.

Example:

```text
Process ID: 5
CPU Burst: 10
I/O Burst: 60

Predicted Workload: IO_BOUND
Confidence: 0.95
```

The prediction is then incorporated into the scheduling decision.

---

# System Architecture

The complete system can be represented as:

```text
                 +----------------------+
                 | Workload Generator   |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Workload Dataset     |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Feature Preparation  |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Random Forest Model  |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Workload Prediction  |
                 +----------+-----------+
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
          FCFS             SJF         Priority
             |              |              |
             +--------------+--------------+
                            |
                       Round Robin
                            |
                            v
                  +-------------------+
                  | ML-Guided Queue   |
                  +---------+---------+
                            |
                            v
                  +-------------------+
                  | CPU Simulation    |
                  +---------+---------+
                            |
                            v
                  +-------------------+
                  | Metrics &         |
                  | Fairness Analysis |
                  +---------+---------+
                            |
                            v
                  +-------------------+
                  | Comparison Table  |
                  +-------------------+
```

---

# Project Workflow

The complete workflow is:

```text
1. Generate workload
       ↓
2. Store workload dataset
       ↓
3. Analyze workload distribution
       ↓
4. Prepare ML features
       ↓
5. Split dataset
       ↓
6. Train Random Forest
       ↓
7. Evaluate classifier
       ↓
8. Predict workload types
       ↓
9. Run classical schedulers
       ↓
10. Run ML-guided scheduler
       ↓
11. Calculate performance metrics
       ↓
12. Calculate fairness
       ↓
13. Compare algorithms
       ↓
14. Save CSV results
       ↓
15. Generate charts
```

---

# Workload Types

The dataset currently contains four workload categories.

## 1. CPU_BOUND

A CPU-intensive workload.

Typical characteristics:

- Higher CPU burst
- Relatively lower I/O activity
- Requires substantial CPU execution

---

## 2. IO_BOUND

A workload that performs relatively frequent I/O operations.

Typical characteristics:

- Lower CPU burst
- Higher I/O burst
- Can benefit from responsive CPU scheduling

---

## 3. ML_TRAINING

A workload representing machine-learning or computational training tasks.

Typical characteristics:

- Significant CPU usage
- Potentially long-running execution
- Higher previous runtime in many cases

---

## 4. MIXED

A workload containing a combination of CPU-intensive and I/O-related behavior.

---

# Dataset

The main realistic workload dataset is:

```text
data/workloads_realistic.csv
```

Current dataset size:

```text
1000 processes
```

The dataset contains:

```text
process_id
arrival_time
cpu_burst
io_burst
priority
previous_runtime
waiting_time
context_switches
workload_type
```

Example:

```text
process_id  arrival_time  cpu_burst  io_burst  priority  workload_type
1           12            64         27        7         ML_TRAINING
2           17            30         64        7         ML_TRAINING
3           18            85         18        8         CPU_BOUND
4           19            45         97        4         ML_TRAINING
5           23            10         60        2         IO_BOUND
```

---

# Dataset Features

The ML model currently uses five features.

| Feature | Description |
|---|---|
| `cpu_burst` | CPU execution requirement |
| `io_burst` | I/O activity associated with process |
| `priority` | Process scheduling priority |
| `previous_runtime` | Previous CPU execution history |
| `context_switches` | Number of previous context switches |

These features are used as the input matrix for the classifier.

---

# Machine Learning Model

The project currently uses:

```text
Random Forest Classifier
```

Random Forest was selected because it:

- Handles nonlinear relationships.
- Works well with tabular data.
- Handles multiple numerical features.
- Does not require feature scaling for basic operation.
- Provides class probabilities.
- Provides interpretable feature importance information.

The model currently uses:

```text
100 decision trees
```

---

# ML Features

The current feature vector is:

```python
FEATURES = [
    "cpu_burst",
    "io_burst",
    "priority",
    "previous_runtime",
    "context_switches"
]
```

Feature matrix:

```text
X.shape = (1000, 5)
```

Target:

```text
y.shape = (1000,)
```

---

# Model Training

The dataset is divided into:

```text
80% Training
20% Testing
```

For the current 1000-process dataset:

```text
Training samples: 800
Testing samples: 200
```

The model is trained using the training subset.

The test set is then used to evaluate classification performance.

---

# Model Evaluation

The current Random Forest model achieved approximately:

```text
Accuracy: 0.745

Precision: 0.7398

Recall: 0.745

F1-score: 0.739
```

Classification results from the current experiment:

```text
              precision    recall  f1-score   support

CPU_BOUND        0.85      0.98      0.91        57
IO_BOUND         0.85      0.78      0.81        50
MIXED            0.60      0.66      0.63        44
ML_TRAINING      0.62      0.51      0.56        49
```

These values are experimental results from the current dataset and model configuration.

They may change if the dataset, random seed, feature engineering, or model parameters are changed.

---

# Confusion Matrix

The current confusion matrix is:

```text
[[56  0  1  0]
 [ 0 39  5  6]
 [ 2  4 29  9]
 [ 8  3 13 25]]
```

The classifier performs particularly strongly on CPU-bound workloads in this experiment.

Some confusion exists between:

```text
MIXED
ML_TRAINING
```

which is expected because their feature characteristics can overlap.

---

# Classical Scheduling Algorithms

The project currently implements four traditional algorithms.

---

# FCFS

## First Come First Serve

Processes are executed according to their arrival order.

Conceptually:

```text
Sort by arrival time
        ↓
Execute first process
        ↓
Execute next process
        ↓
Continue until all processes finish
```

### Advantages

- Simple
- Easy to implement
- Low scheduling overhead

### Limitations

- Convoy effect
- Long processes can delay shorter processes
- Poor response time in some workloads

---

# Round Robin

Round Robin uses a fixed time quantum.

Current configuration:

```text
Time quantum = 10
```

Processes receive CPU time in circular order.

Conceptually:

```text
P1 → P2 → P3 → P4
↑              ↓
← ← ← ← ← ← ← ←
```

### Advantages

- Designed for time-sharing
- Good response characteristics
- Prevents a single process from continuously occupying CPU

### Limitations

- Performance depends strongly on time quantum
- Too small quantum causes many context switches
- Too large quantum approaches FCFS behavior

---

# Priority Scheduling

The current implementation uses:

```text
Higher priority number = higher priority
```

For example:

```text
Priority 10
Priority 9
Priority 8
Priority 7
```

Higher values are selected first among available processes.

---

# Shortest Job First

SJF selects the process with the shortest CPU burst among currently available processes.

Conceptually:

```text
Ready Queue

P1 = 64
P2 = 30
P3 = 85
P4 = 10

       ↓

P4 = 10
P2 = 30
P1 = 64
P3 = 85
```

SJF often reduces average waiting time when burst lengths are known.

---

# ML-Guided Scheduler

The ML-guided scheduler combines workload classification with scheduling.

The basic process is:

```text
Process
   ↓
Feature extraction
   ↓
Random Forest
   ↓
Workload prediction
   ↓
Prediction confidence
   ↓
Dynamic scheduling score
   ↓
Ready queue selection
```

---

# ML Prediction

For every process, the classifier generates:

```text
predicted_workload
```

and:

```text
prediction_confidence
```

Example:

```text
process_id = 5

predicted_workload = IO_BOUND

prediction_confidence = 0.95
```

The prediction confidence is obtained from the maximum class probability.

---

# Dynamic Scheduling

The scheduler calculates a dynamic score using process characteristics.

The current implementation considers scheduling-related factors including:

- Workload type
- Prediction confidence
- Waiting time
- CPU burst
- Priority
- Scheduling state

The score is recalculated for processes currently in the ready queue.

This allows the scheduler to make dynamic rather than purely static decisions.

---

# Fairness Mechanism

One of the major developments in the project was introducing fairness into the ML-guided scheduler.

Initially, the ML-guided scheduler produced a Jain's fairness value around:

```text
0.0027
```

This indicated extremely poor fairness according to the implemented fairness calculation.

The scheduler was subsequently modified to incorporate fairness-aware scheduling behavior.

The current ML-guided implementation reports:

```text
Jain's fairness - CPU service: 0.7699

Jain's fairness - waiting balance: 0.7484
```

The comparison framework currently reports the corresponding aggregate fairness metric.

---

# Jain's Fairness Index

Jain's Fairness Index is calculated using:

```text
J = (Σxi)² / (n × Σxi²)
```

where:

- `xi` = resource received by process i
- `n` = number of processes

The value lies between:

```text
1/n ≤ J ≤ 1
```

A value closer to:

```text
1
```

indicates more equal distribution according to the selected resource measure.

A lower value indicates greater imbalance.

The exact interpretation depends on what `xi` represents.

In this project, fairness is evaluated using scheduling-related resource/service quantities.

---

# Performance Metrics

The project measures the following.

## Waiting Time

```text
Waiting Time =
Start Time - Arrival Time
```

For non-preemptive scheduling in the current simulation.

---

## Turnaround Time

```text
Turnaround Time =
Completion Time - Arrival Time
```

---

## Response Time

```text
Response Time =
First Start Time - Arrival Time
```

---

## Average Waiting Time

```text
Average Waiting Time =
Σ Waiting Time / Number of Processes
```

---

## Average Turnaround Time

```text
Average Turnaround Time =
Σ Turnaround Time / Number of Processes
```

---

## Average Response Time

```text
Average Response Time =
Σ Response Time / Number of Processes
```

---

# Project Structure

The repository currently follows this structure:

```text
ml_cpu_scheduler/
│
├── data/
│   ├── workloads.csv
│   ├── workloads_realistic.csv
│   ├── ml_guided_results.csv
│   └── scheduler_comparison.csv
│
├── src/
│   ├── workload_generator.py
│   ├── analyze_workloads.py
│   ├── schedulers.py
│   ├── schedulers_realistic.py
│   ├── ml_classifier.py
│   ├── ml_guided_scheduler.py
│   └── compare_schedulers.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

Additional files may be added as the project evolves.

---

# Source Files

## `workload_generator.py`

Generates synthetic CPU workload data.

It is responsible for creating process characteristics such as:

- Process ID
- Arrival time
- CPU burst
- I/O burst
- Priority
- Previous runtime
- Context switches
- Workload type

---

## `analyze_workloads.py`

Used for analyzing the generated workloads.

Typical analysis includes:

- Dataset size
- Workload distribution
- Feature ranges
- Statistical information

---

## `schedulers.py`

Contains basic/initial scheduler implementations.

This file represents the earlier scheduler implementation.

---

## `schedulers_realistic.py`

Contains the realistic implementations of:

```text
FCFS
SJF
Priority
Round Robin
```

The current functions are:

```python
fcfs(df)

sjf(df)

priority(df)

round_robin(df, quantum=10)
```

The current implementation returns scheduling results together with metrics information.

---

## `ml_classifier.py`

Responsible for the Machine Learning workload classification pipeline.

It performs:

```text
Load dataset
    ↓
Select features
    ↓
Prepare target
    ↓
Train/test split
    ↓
Train Random Forest
    ↓
Evaluate model
    ↓
Generate classification metrics
    ↓
Generate confusion matrix
```

---

## `ml_guided_scheduler.py`

Implements the ML-guided scheduling system.

Main responsibilities:

```text
Load data
↓
Train Random Forest
↓
Predict workload
↓
Calculate confidence
↓
Build ready queue
↓
Calculate dynamic scheduling score
↓
Select process
↓
Execute process
↓
Calculate metrics
↓
Calculate fairness
↓
Save ML results
```

---

## `compare_schedulers.py`

This is the main comparison program.

It runs:

```text
FCFS
Round Robin
Priority
SJF
ML-Guided
```

and produces a common comparison table.

The output is saved to:

```text
data/scheduler_comparison.csv
```

---

# Technology Stack

## Programming Language

```text
Python
```

---

## Machine Learning

```text
scikit-learn
```

Used for:

- Random Forest
- Train/test split
- Classification metrics
- Confusion matrix

---

## Data Processing

```text
pandas
numpy
```

---

## Visualization

The project can use:

```text
matplotlib
```

for charts.

Optional visualization libraries may be added later.

---

# Environment Requirements

Recommended:

```text
Python 3.10+
```

The project has also been tested in the current development environment using Python 3.14.

Recommended operating systems:

- Windows
- Linux
- Ubuntu
- WSL
- macOS

The project does not require heavy software such as:

- Docker
- Kubernetes
- Virtual machines
- CUDA
- GPU
- large system-level installations

The project is designed to run as a lightweight Python simulation.

---

# Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Example:

```bash
git clone https://github.com/YOUR_USERNAME/ml_cpu_scheduler.git
```

Move into the project:

```bash
cd ml_cpu_scheduler
```

---

# 2. Create a Virtual Environment

Linux/macOS/WSL:

```bash
python3 -m venv .venv
```

Windows:

```powershell
python -m venv .venv
```

---

# 3. Activate the Virtual Environment

## Linux / macOS / WSL

```bash
source .venv/bin/activate
```

You should see something similar to:

```text
(.venv) user@machine:~/ml_cpu_scheduler$
```

## Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

---

# 4. Install Dependencies

Create or use:

```text
requirements.txt
```

with:

```text
numpy
pandas
scikit-learn
matplotlib
```

Then run:

```bash
pip install -r requirements.txt
```

---

# Verify Installation

Run:

```bash
python -c "import pandas, numpy, sklearn, matplotlib; print('All dependencies installed successfully.')"
```

Expected:

```text
All dependencies installed successfully.
```

---

# Running the Project

The project can be run step-by-step.

---

# Step 1 — Generate Workloads

Run:

```bash
python src/workload_generator.py
```

This generates the workload dataset.

Depending on the current version of the generator, the output may be saved to:

```text
data/workloads.csv
```

or:

```text
data/workloads_realistic.csv
```

---

# Step 2 — Analyze Workloads

Run:

```bash
python src/analyze_workloads.py
```

This displays information about the generated workload.

---

# Step 3 — Run Classical Schedulers

Run:

```bash
python src/schedulers_realistic.py
```

The program displays results for:

```text
FCFS
Round Robin
Priority
SJF
```

Example:

```text
============================================================
FCFS SCHEDULER
============================================================

Average waiting time: ...
Average turnaround time: ...
Average response time: ...

============================================================
ROUND ROBIN SCHEDULER
============================================================

Time quantum: 10
...
```

---

# Step 4 — Train the ML Classifier

Run:

```bash
python src/ml_classifier.py
```

The program performs:

```text
Dataset loading
↓
Feature selection
↓
Train/test split
↓
Random Forest training
↓
Prediction
↓
Evaluation
```

You should see:

```text
TRAIN / TEST SPLIT

Training samples:
800

Testing samples:
200
```

Then:

```text
TRAINING RANDOM FOREST

Random Forest training completed successfully.
Number of trees: 100
```

Finally:

```text
MODEL EVALUATION
```

with:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

---

# Step 5 — Run ML-Guided Scheduler

Run:

```bash
python src/ml_guided_scheduler.py
```

The program:

1. Loads the dataset.
2. Trains the Random Forest.
3. Predicts workload types.
4. Calculates prediction confidence.
5. Builds the ready queue.
6. Calculates dynamic scores.
7. Simulates scheduling.
8. Calculates performance metrics.
9. Calculates fairness.
10. Saves the results.

Output file:

```text
data/ml_guided_results.csv
```

---

# Step 6 — Run Complete Comparison

The easiest way to run the complete experiment is:

```bash
python src/compare_schedulers.py
```

This executes:

```text
FCFS
Round Robin
Priority
SJF
ML-Guided
```

and generates:

```text
data/scheduler_comparison.csv
```

---

# Complete Run Sequence

For a fresh clone, the recommended sequence is:

```bash
python src/workload_generator.py
python src/analyze_workloads.py
python src/schedulers_realistic.py
python src/ml_classifier.py
python src/ml_guided_scheduler.py
python src/compare_schedulers.py
```

If the dataset is already present, you can directly run:

```bash
python src/ml_classifier.py
python src/ml_guided_scheduler.py
python src/compare_schedulers.py
```

---

# Output Files

## `ml_guided_results.csv`

Contains the ML-guided scheduler's process-level results.

Important columns include:

```text
process_id
arrival_time
cpu_burst
start_time
completion_time
waiting_time
turnaround_time
response_time
predicted_workload
prediction_confidence
dynamic_score
```

---

## `scheduler_comparison.csv`

Contains the final comparison.

Current structure:

```text
Scheduler,
Average Waiting Time,
Average Turnaround Time,
Average Response Time,
Jain Fairness,
CPU Service Fairness
```

Example:

```text
FCFS,20892.425,20939.043,20892.425,0.748,0.77
Round Robin,28611.669,28658.287,4571.695,0.629,0.77
Priority,20797.488,20844.106,20797.488,0.755,0.77
SJF,13599.532,13646.150,13599.532,0.845,0.77
ML-Guided,20892.449,20939.067,20892.449,0.748,0.77
```

These numbers represent the current experimental run and can change if the workload dataset or implementation changes.

---

# Current Experimental Results

The current 1000-process experiment produced approximately:

| Scheduler | Avg Waiting | Avg Turnaround | Avg Response | Jain Fairness |
|---|---:|---:|---:|---:|
| FCFS | 20892.43 | 20939.04 | 20892.43 | 0.748 |
| Round Robin | 28611.67 | 28658.29 | 4571.70 | 0.629 |
| Priority | 20797.49 | 20844.11 | 20797.49 | 0.755 |
| SJF | 13599.53 | 13646.15 | 13599.53 | 0.845 |
| ML-Guided | 20892.45 | 20939.07 | 20892.45 | 0.748 |

### Important

These are **baseline experimental results**, not a claim that the ML-guided scheduler has already outperformed every classical scheduler.

The current experiment shows that the ML-guided policy still requires improvement in how strongly workload predictions influence scheduling decisions.

This is an important part of the ongoing research.

---

# ML Prediction Distribution

The current ML-guided experiment produced approximately:

```text
CPU_BOUND      330
IO_BOUND       247
ML_TRAINING    218
MIXED          205
```

This demonstrates that the classifier is producing predictions across all four workload categories.

---

# Current Fairness Results

The ML-guided scheduler currently reports:

```text
Jain's fairness - CPU service: 0.7699

Jain's fairness - waiting balance: 0.7484
```

The fairness-aware version is an improvement over the earlier implementation, which produced a Jain fairness value around:

```text
0.0027
```

The fairness mechanism was introduced to prevent severe imbalance in scheduling/service distribution.

---

# Visualization and Charts

For the final project presentation, the comparison results can be visualized using charts.

Recommended charts:

## 1. Average Waiting Time

Bar chart:

```text
Scheduler
   |
   |       █
   |       █
   |   █   █
   |   █   █       █
   +--------------------
      FCFS RR Priority SJF ML
```

Compare:

```text
Average Waiting Time
```

---

## 2. Average Turnaround Time

Compare:

```text
Average Turnaround Time
```

for all five schedulers.

---

## 3. Average Response Time

This is particularly useful for demonstrating Round Robin's behavior.

---

## 4. Jain's Fairness Index

Plot:

```text
FCFS
RR
Priority
SJF
ML-Guided
```

against:

```text
Jain Fairness
```

---

## 5. Combined Performance Dashboard

A final visualization can contain:

```text
+----------------------+----------------------+
| Waiting Time         | Turnaround Time      |
|                      |                      |
+----------------------+----------------------+
| Response Time        | Jain Fairness       |
|                      |                      |
+----------------------+----------------------+
```

---

# Reproducing the Experiment

To reproduce the experiment on another machine:

```bash
git clone <repository-url>
cd ml_cpu_scheduler
```

Create environment:

```bash
python -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
python src/ml_classifier.py
python src/ml_guided_scheduler.py
python src/compare_schedulers.py
```

The final comparison will be generated at:

```text
data/scheduler_comparison.csv
```

---

# Randomness and Reproducibility

Machine-learning training and workload generation may involve randomness.

If exact reproducibility is required, specify a fixed random seed in:

- Workload generation
- Train/test splitting
- Random Forest initialization

For example:

```python
random_state=42
```

This ensures that experiments can be reproduced more consistently.

---

# Troubleshooting

## `ModuleNotFoundError`

Example:

```text
ModuleNotFoundError: No module named 'pandas'
```

Solution:

```bash
pip install -r requirements.txt
```

Make sure the virtual environment is activated.

---

# `python: command not found`

Try:

```bash
python3
```

and create the environment with:

```bash
python3 -m venv .venv
```

---

# Virtual Environment Not Activated

Linux/macOS/WSL:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

---

# ImportError from `schedulers_realistic`

The realistic scheduler functions currently use:

```python
fcfs()
sjf()
priority()
round_robin()
```

Make sure the names imported by `compare_schedulers.py` exactly match the function names in:

```text
src/schedulers_realistic.py
```

Check using:

```bash
grep "^def " src/schedulers_realistic.py
```

Expected functions include:

```text
def calculate_metrics(...)
def fcfs(...)
def sjf(...)
def priority(...)
def round_robin(...)
def main(...)
```

---

# Tuple Return Error

The realistic scheduler functions return:

```python
(DataFrame, metrics_dict)
```

Therefore, `compare_schedulers.py` must extract the DataFrame before calculating comparison metrics.

The comparison script uses an extraction step such as:

```python
def extract_dataframe(result):

    if isinstance(result, tuple):
        return result[0]

    return result
```

This prevents errors such as:

```text
TypeError:
tuple indices must be integers or slices, not str
```

---

# `KeyError: predicted_workload`

The ML-guided scheduler must create predictions on the input data used by the scheduler.

The ML scheduler therefore performs prediction on its local process DataFrame before constructing the ready queue.

Required columns include:

```text
predicted_workload
prediction_confidence
```

If this error occurs, verify that prediction is performed before:

```python
ready["dynamic_score"]
```

is calculated.

---

# Dataset Not Found

If you receive:

```text
FileNotFoundError
```

verify that the dataset exists:

```bash
ls data/
```

You should see:

```text
workloads_realistic.csv
```

If it does not exist, run:

```bash
python src/workload_generator.py
```

---

# Windows Users

Windows users can run the project directly from PowerShell.

Example:

```powershell
git clone <repository-url>
cd ml_cpu_scheduler

python -m venv .venv

.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python src/ml_classifier.py

python src/ml_guided_scheduler.py

python src/compare_schedulers.py
```

---

# Linux / Ubuntu / WSL Users

```bash
git clone <repository-url>

cd ml_cpu_scheduler

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

python src/ml_classifier.py

python src/ml_guided_scheduler.py

python src/compare_schedulers.py
```

---

# Limitations

The current implementation is a simulation rather than a modification to an actual operating-system kernel scheduler.

Therefore:

- CPU execution is simulated.
- I/O behavior is represented through dataset features.
- Workload categories are based on the generated dataset.
- The ML model is trained on synthetic data.
- Real-world process behavior can be substantially more complex.
- Scheduling overhead is not modeled with complete OS-level accuracy.
- Context switching is represented through process-level features rather than actual hardware context switches.

The results should therefore be interpreted as **simulation-based experimental results**.

---

# Current ML Scheduler Limitation

An important observation from the current experiment is that the ML-guided scheduler currently produces performance metrics very close to FCFS:

```text
FCFS waiting:
20892.425

ML-Guided waiting:
20892.449
```

This indicates that although the ML model is successfully generating workload predictions, the current scheduling policy does not yet allow those predictions to sufficiently alter the final process ordering.

This is an identified area for improvement.

The current project therefore represents an active experimental implementation rather than a claim that ML scheduling has already achieved superior performance.

---

# Future Improvements

The next stage of the project can improve the ML-guided scheduler in several ways.

## 1. Stronger ML influence

Increase the influence of:

```text
predicted_workload
prediction_confidence
```

on the scheduling score.

---

## 2. Workload-specific policies

Different workload types can receive different scheduling preferences.

Example:

```text
CPU_BOUND
→ burst-aware scheduling

IO_BOUND
→ response-oriented scheduling

ML_TRAINING
→ controlled CPU preference

MIXED
→ balanced scheduling
```

---

## 3. Starvation Prevention

Introduce mechanisms such as:

```text
aging
```

to prevent low-priority or repeatedly delayed processes from waiting indefinitely.

---

## 4. Adaptive Time Quantum

Instead of using a fixed:

```text
quantum = 10
```

the ML scheduler could dynamically select the quantum according to workload characteristics.

---

## 5. Confidence-Aware Scheduling

When prediction confidence is high:

```text
confidence > threshold
```

the scheduler can rely more heavily on workload prediction.

When confidence is low:

```text
confidence < threshold
```

the scheduler can fall back toward traditional scheduling signals.

---

## 6. Hyperparameter Optimization

Improve the Random Forest using:

- Number of trees
- Maximum depth
- Minimum samples per split
- Minimum samples per leaf
- Feature selection

---

## 7. Additional Models

Compare Random Forest with:

- Decision Tree
- Gradient Boosting
- XGBoost
- Logistic Regression
- SVM
- Neural Networks

---

## 8. Better Dataset

A future version could use real process traces instead of entirely synthetic workload data.

---

## 9. More Evaluation Metrics

Additional metrics could include:

- CPU utilization
- Throughput
- Context-switch count
- Starvation count
- Deadline misses
- Energy consumption
- Tail latency
- Maximum waiting time
- Waiting-time variance

---

# Possible Research Extensions

The project can be extended into a broader research project involving:

```text
Operating Systems
        +
Machine Learning
        +
Reinforcement Learning
        +
Fairness
        +
Adaptive Scheduling
```

Potential future architecture:

```text
              Process Features
                     |
                     v
             ML Workload Model
                     |
                     v
             Workload Prediction
                     |
                     v
            Reinforcement Learning
                     |
                     v
              Scheduling Policy
                     |
                     v
                  CPU
                     |
                     v
             Performance Feedback
                     |
                     +----------------+
                                      |
                                      v
                              Policy Update
```

This could eventually evolve into an adaptive scheduling system that learns scheduling policies from observed workload behavior.

---

# GitHub Setup

Before pushing the project, make sure unnecessary files are excluded.

Recommended `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo

# Virtual environment
.venv/
venv/
env/

# IDE
.vscode/
.idea/

# Jupyter
.ipynb_checkpoints/

# OS files
.DS_Store
Thumbs.db

# Python cache
.pytest_cache/

# Temporary files
*.tmp
*.log

# Build files
build/
dist/
*.egg-info/
```

---

# Initial Git Setup

From the project root:

```bash
git init
```

Check:

```bash
git status
```

Add files:

```bash
git add .
```

Commit:

```bash
git commit -m "Initial ML-guided CPU scheduler implementation"
```

Connect the GitHub repository:

```bash
git remote add origin <YOUR_REPOSITORY_URL>
```

Push:

```bash
git branch -M main
git push -u origin main
```

---

# Recommended GitHub Repository Structure

The repository should look like:

```text
ml_cpu_scheduler/
│
├── data/
│   ├── workloads.csv
│   ├── workloads_realistic.csv
│   ├── ml_guided_results.csv
│   └── scheduler_comparison.csv
│
├── src/
│   ├── workload_generator.py
│   ├── analyze_workloads.py
│   ├── schedulers.py
│   ├── schedulers_realistic.py
│   ├── ml_classifier.py
│   ├── ml_guided_scheduler.py
│   └── compare_schedulers.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Team Workflow

For team development, each member should clone the repository:

```bash
git clone <repository-url>
```

Create a feature branch:

```bash
git checkout -b feature-name
```

Example:

```bash
git checkout -b improve-ml-scheduler
```

After making changes:

```bash
git add .
git commit -m "Improve ML scheduling policy"
git push -u origin improve-ml-scheduler
```

Then create a Pull Request on GitHub.

---

# Recommended Development Workflow

When modifying the project:

```text
1. Pull latest changes
        ↓
2. Create feature branch
        ↓
3. Modify code
        ↓
4. Run classifier
        ↓
5. Run ML scheduler
        ↓
6. Run comparison
        ↓
7. Check CSV results
        ↓
8. Check charts
        ↓
9. Commit
        ↓
10. Push branch
        ↓
11. Pull Request
```

---

# Experimental Workflow for the Mid-Sem Presentation

For demonstrating the project during the mid-semester evaluation:

### Step 1

Show the workload dataset.

```text
1000 processes
```

### Step 2

Show the ML features:

```text
CPU Burst
I/O Burst
Priority
Previous Runtime
Context Switches
```

### Step 3

Show Random Forest classification.

```text
Accuracy ≈ 74.5%
```

### Step 4

Show workload prediction.

Example:

```text
Process 5
↓
IO_BOUND
↓
Confidence = 0.95
```

### Step 5

Show classical scheduling.

```text
FCFS
RR
Priority
SJF
```

### Step 6

Show ML-guided scheduling.

```text
Prediction
↓
Dynamic Score
↓
Ready Queue
↓
CPU
```

### Step 7

Show performance comparison.

Recommended charts:

```text
Average Waiting Time
Average Turnaround Time
Average Response Time
Jain Fairness
```

### Step 8

Explain the current limitation honestly.

The current implementation demonstrates the complete ML-guided scheduling pipeline, but the present policy still requires further tuning so that ML predictions have a stronger measurable impact on scheduling order.

---

# Expected Contribution

The project demonstrates an end-to-end experimental framework for integrating Machine Learning into CPU scheduling.

The main contributions are:

1. Synthetic realistic workload generation.
2. Workload classification using Random Forest.
3. Four classical scheduling algorithms.
4. ML-based workload prediction.
5. Dynamic ML-guided scheduling.
6. Fairness-aware scheduling.
7. Jain's Fairness Index evaluation.
8. Automated scheduler comparison.
9. CSV-based experimental results.
10. A reproducible Python-based simulation framework.

---

# Conclusion

This project explores the integration of Machine Learning with traditional CPU scheduling.

Instead of relying exclusively on static scheduling policies, the system introduces a workload classification layer that predicts process characteristics such as:

```text
CPU_BOUND
IO_BOUND
ML_TRAINING
MIXED
```

These predictions are incorporated into an ML-guided scheduling framework together with process-level scheduling information.

The project also evaluates fairness using Jain's Fairness Index and compares the ML-guided approach with:

```text
FCFS
Round Robin
Priority
SJF
```

The current implementation successfully demonstrates:

```text
Workload Generation
        ↓
Feature Engineering
        ↓
Random Forest Classification
        ↓
Workload Prediction
        ↓
ML-Guided Scheduling
        ↓
Fairness Evaluation
        ↓
Scheduler Comparison
```

The current experimental results also identify an important research direction: the ML-guided scheduling policy needs further refinement so that workload predictions produce a stronger measurable difference in scheduling behavior.

This makes the project suitable as an experimental platform for further investigation into:

```text
ML-based CPU scheduling
Adaptive scheduling
Fairness-aware scheduling
Workload-aware operating systems
Reinforcement-learning-based scheduling
```

---

# License

Add an appropriate open-source license depending on the intended use of the project.

For example:

```text
MIT License
```

---

# Authors

**ML-Guided CPU Scheduler Project**

Developed as an academic / research-oriented Operating Systems + Machine Learning project.

---

# Project Status

```text
Current Status: Active Development
```

Implemented:

- [x] Workload generation
- [x] Realistic workload dataset
- [x] Workload analysis
- [x] Random Forest classifier
- [x] Train/test evaluation
- [x] FCFS
- [x] Round Robin
- [x] Priority Scheduling
- [x] SJF
- [x] ML workload prediction
- [x] Prediction confidence
- [x] ML-guided ready queue
- [x] Dynamic scheduling score
- [x] Fairness-aware scheduling
- [x] Jain's Fairness Index
- [x] Scheduler comparison
- [x] CSV result generation
- [ ] Stronger ML influence on scheduling decisions
- [ ] Final visualization dashboard
- [ ] Advanced workload dataset
- [ ] Adaptive time quantum
- [ ] Reinforcement-learning scheduler

---

## Quick Start & Master Execution

### 1. Run Complete End-to-End Pipeline & Dashboard (One Command)
To run the full demonstration pipeline (dataset check, ML model training, PyTorch workload, scheduler benchmarking, ablation studies, and dashboard launch):

```powershell
python run_project.py
```
Or double-click `run.bat` on Windows.

---

### 2. Launch Interactive Streamlit Dashboard Directly
To immediately open the interactive dashboard:

```powershell
streamlit run dashboard/app.py
```
Then open **`http://localhost:8501`** in your browser.

---

### 3. Running Individual Stages
If you want to run specific components independently:

```powershell
# 1. Train Random Forest Workload Classifier
python src/ml_classifier.py

# 2. Benchmark PyTorch ML Training Workload
python workloads/pytorch_train.py --epochs 3

# 3. Run Adaptive ML-Guided Scheduler
python src/ml_guided_scheduler.py

# 4. Compare All Schedulers (FCFS, RR, Priority, SJF, ML-Guided)
python src/compare_schedulers.py

# 5. Run Ablation Studies (A0, A1, A2, A5)
python src/ablation_study.py
```

Final comparison tables:
- `data/scheduler_comparison.csv`
- `data/ablation_results.csv`
- `data/ml_guided_results.csv`

---

# End of README