"""
src/ml_guided_scheduler.py
MLTrain-Sched: ML-Guided CPU Scheduler Simulation
Implements:
  1. Random Forest Workload Classifier
  2. Multi-Level Dispatch Queues (DSQs: HIGH, MEDIUM, LOW)
  3. Adaptive Time-Slicing (Quantum) based on Workload Characteristics
  4. Confidence-Aware Rule-Based Fallback
  5. Starvation Safeguards via Continuous Aging
"""

import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET = "data/workloads_realistic.csv"

# ============================================================
# ML MODEL INITIALIZATION & TRAINING
# ============================================================

FEATURES = [
    "cpu_burst",
    "io_burst",
    "priority",
    "previous_runtime",
    "context_switches"
]
TARGET = "workload_type"

def train_workload_classifier(df):
    X = df[FEATURES]
    y = df[TARGET]
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model, encoder

# ============================================================
# POLICY RULES & QUANTUM ASSIGNMENT
# ============================================================

def rule_based_fallback(row):
    """Fallback when ML confidence is below threshold."""
    cpu = row["cpu_burst"]
    io = row["io_burst"]
    if cpu > 60 and io < 30:
        return "CPU_BOUND"
    elif io > 50 and cpu < 30:
        return "IO_BOUND"
    elif cpu > 40 and io > 40:
        return "MIXED"
    else:
        return "ML_TRAINING"

def get_adaptive_quantum(workload_class, min_q=4, max_q=20):
    """
    Adaptive time-slice policy:
    - CPU_BOUND: Long slice (20) to maintain cache locality & minimize context switches.
    - IO_BOUND: Short slice (5) to yield quickly and boost responsiveness.
    - ML_TRAINING: Long-medium slice (15) for compute loops.
    - MIXED: Standard slice (10).
    """
    if workload_class == "CPU_BOUND":
        q = 18
    elif workload_class == "IO_BOUND":
        q = 5
    elif workload_class == "ML_TRAINING":
        q = 14
    else:
        q = 10
    return int(np.clip(q, min_q, max_q))

def assign_dsq(workload_class, priority=5):
    """
    Assigns task to Dispatch Queue (DSQ_HIGH, DSQ_MED, DSQ_LOW) based on ML workload class & priority:
    - DSQ_HIGH: ML_TRAINING (GPU critical path) and IO_BOUND (latency sensitive)
    - DSQ_MEDIUM: MIXED (balanced) or high-priority CPU_BOUND (priority >= 8)
    - DSQ_LOW: CPU_BOUND batch contention workloads (priority < 8)
    """
    if workload_class in ["ML_TRAINING", "IO_BOUND"]:
        return "HIGH" if priority >= 3 else "MEDIUM"
    elif workload_class == "CPU_BOUND":
        return "MEDIUM" if priority >= 8 else "LOW"
    else:  # MIXED
        return "HIGH" if priority >= 8 else "MEDIUM"

# ============================================================
# MAIN SCHEDULER IMPLEMENTATION
# ============================================================

def ml_guided_scheduler(
    data,
    aging_weight=0.25,
    confidence_threshold=0.65,
    preemptive=True,
    min_quantum=4,
    max_quantum=20,
    starvation_limit=3000
):
    """
    Executes the MLTrain-Sched adaptive scheduling simulation.
    Supports preemptive adaptive quantum scheduling or non-preemptive dynamic queue dispatch.
    """
    processes = data.copy()

    # Train model on reference or input data
    model, encoder = train_workload_classifier(processes)

    # Inference
    probabilities = model.predict_proba(processes[FEATURES])
    raw_preds = encoder.inverse_transform(model.predict(processes[FEATURES]))
    confidences = probabilities.max(axis=1)

    predicted_classes = []
    quantums = []

    for i, (_, row) in enumerate(processes.iterrows()):
        conf = confidences[i]
        if conf >= confidence_threshold:
            cls = raw_preds[i]
        else:
            cls = rule_based_fallback(row)
        predicted_classes.append(cls)
        quantums.append(get_adaptive_quantum(cls, min_quantum, max_quantum))

    processes["predicted_workload"] = predicted_classes
    processes["prediction_confidence"] = confidences
    processes["target_quantum"] = quantums

    processes["assigned_dsq"] = processes.apply(
        lambda r: assign_dsq(r["predicted_workload"], r["priority"]),
        axis=1
    )

    processes = processes.sort_values(["arrival_time", "process_id"]).reset_index(drop=True)

    if not preemptive:
        # Non-preemptive priority scoring simulation
        completed = set()
        current_time = 0
        schedule = []
        n = len(processes)

        while len(completed) < n:
            ready = processes[
                (processes["arrival_time"] <= current_time) & (~processes["process_id"].isin(completed))
            ].copy()

            if ready.empty:
                remaining = processes[~processes["process_id"].isin(completed)]
                current_time = max(current_time, remaining["arrival_time"].min())
                continue

            ready["current_waiting"] = (current_time - ready["arrival_time"]).clip(lower=0)
            max_w = max(1, ready["current_waiting"].max())

            # Bounded dynamic priority score: ML preference + Short burst bonus + Aging
            def compute_score(r):
                base_w = {"ML_TRAINING": 0.35, "IO_BOUND": 0.30, "MIXED": 0.20, "CPU_BOUND": 0.15}[r["predicted_workload"]]
                burst_bonus = 0.25 * (1.0 / (1.0 + r["cpu_burst"]))
                priority_comp = 0.20 * (r["priority"] / 10.0)
                aging_comp = aging_weight * min(r["current_waiting"] / max_w, 1.0)
                return base_w + burst_bonus + priority_comp + aging_comp

            ready["dynamic_score"] = ready.apply(compute_score, axis=1)

            # Sort by DSQ (HIGH -> MEDIUM -> LOW), with aging escalation for starving tasks
            dsq_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            def get_effective_dsq_rank(r):
                if r["current_waiting"] > starvation_limit:
                    return 0  # Promoted to front due to starvation
                return dsq_order[r["assigned_dsq"]]

            ready["dsq_rank"] = ready.apply(get_effective_dsq_rank, axis=1)
            selected = ready.sort_values(by=["dsq_rank", "dynamic_score"], ascending=[True, False]).iloc[0]

            pid = int(selected["process_id"])
            start = max(current_time, selected["arrival_time"])
            completion = start + selected["cpu_burst"]

            schedule.append({
                **selected.to_dict(),
                "start_time": start,
                "completion_time": completion,
                "waiting_time": start - selected["arrival_time"],
                "turnaround_time": completion - selected["arrival_time"],
                "response_time": start - selected["arrival_time"]
            })

            current_time = completion
            completed.add(pid)

        result_df = pd.DataFrame(schedule)
    else:
        # Preemptive Adaptive Quantum Simulation (Matches sched_ext Time Slicing)
        remaining_burst = {int(r["process_id"]): int(r["cpu_burst"]) for _, r in processes.iterrows()}
        first_start = {}
        completion_times = {}
        final_scores = {}

        ready_queue = [] # list of pids
        p_dict = {int(r["process_id"]): r.to_dict() for _, r in processes.iterrows()}
        idx = 0
        current_time = 0
        total_p = len(processes)

        while len(completion_times) < total_p:
            while idx < total_p and processes.iloc[idx]["arrival_time"] <= current_time:
                ready_queue.append(int(processes.iloc[idx]["process_id"]))
                idx += 1

            if not ready_queue:
                if idx < total_p:
                    current_time = max(current_time, processes.iloc[idx]["arrival_time"])
                    continue

            # Prioritize queue using ML DSQ and Aging
            def get_task_sort_key(pid):
                info = p_dict[pid]
                wait = current_time - info["arrival_time"]
                base_dsq = info["assigned_dsq"]
                dsq_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[base_dsq]
                score = (info["priority"] * 0.3) + (aging_weight * (wait / 500.0))
                final_scores[pid] = score
                # Effective rank: starving tasks promoted to front
                effective_rank = 0 if wait > starvation_limit else dsq_rank
                return (effective_rank, -score)

            ready_queue.sort(key=get_task_sort_key)
            pid = ready_queue.pop(0)
            info = p_dict[pid]

            if pid not in first_start:
                first_start[pid] = current_time

            quantum = info["target_quantum"]
            execution = min(quantum, remaining_burst[pid])
            current_time += execution
            remaining_burst[pid] -= execution

            # Ingest newly arrived tasks during execution window
            while idx < total_p and processes.iloc[idx]["arrival_time"] <= current_time:
                ready_queue.append(int(processes.iloc[idx]["process_id"]))
                idx += 1

            if remaining_burst[pid] > 0:
                ready_queue.append(pid)
            else:
                completion_times[pid] = current_time

        rows = []
        for _, p in processes.iterrows():
            pid = int(p["process_id"])
            st = first_start[pid]
            ct = completion_times[pid]
            rows.append({
                **p.to_dict(),
                "assigned_dsq": p["assigned_dsq"],
                "dynamic_score": final_scores.get(pid, 0.5),
                "start_time": st,
                "completion_time": ct,
                "waiting_time": ct - p["arrival_time"] - p["cpu_burst"],
                "turnaround_time": ct - p["arrival_time"],
                "response_time": st - p["arrival_time"]
            })
        result_df = pd.DataFrame(rows)

    return result_df

if __name__ == "__main__":
    if os.path.exists(DATASET):
        df_in = pd.read_csv(DATASET)
        print("Running MLTrain-Sched Adaptive Simulation...")
        res = ml_guided_scheduler(df_in, preemptive=True)
        res.to_csv("data/ml_guided_results.csv", index=False)
        print("ML-Guided Results saved to data/ml_guided_results.csv")
        print(f"Avg Waiting Time: {res['waiting_time'].mean():.2f}")
        print(f"Avg Turnaround Time: {res['turnaround_time'].mean():.2f}")
        print(f"Avg Response Time: {res['response_time'].mean():.2f}")