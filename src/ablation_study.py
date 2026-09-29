"""
src/ablation_study.py
Executes ablation studies defined in Section 28 of the MLTrain-Sched blueprint:
  A0: No ML (Rule-based scheduler)
  A1: No Adaptive Quantum (Fixed quantum)
  A2: No Aging (No starvation prevention)
  A5: No Confidence Fallback
  Full: Complete ML-Guided Scheduler with all features
"""

import pandas as pd
import numpy as np
import os
from ml_guided_scheduler import ml_guided_scheduler

DATASET = "data/workloads_realistic.csv"

def jains_index(values):
    values = np.asarray(values, dtype=float)
    values = np.maximum(values, 1e-9)
    return (np.sum(values) ** 2) / (len(values) * np.sum(values ** 2))

def evaluate_run(result_df):
    avg_wait = result_df["waiting_time"].mean()
    avg_tat = result_df["turnaround_time"].mean()
    avg_resp = result_df["response_time"].mean()
    max_wait = result_df["waiting_time"].max()

    if max_wait == 0:
        waiting_fairness = np.ones(len(result_df))
    else:
        waiting_fairness = np.clip(1 - (result_df["waiting_time"] / max_wait), 0.01, None)

    fairness = jains_index(waiting_fairness)
    return {
        "Avg Waiting Time": round(avg_wait, 2),
        "Avg Turnaround Time": round(tat := avg_tat, 2),
        "Avg Response Time": round(avg_resp, 2),
        "Max Wait Time": round(max_wait, 2),
        "Jain Fairness": round(fairness, 3)
    }

def run_ablations():
    if not os.path.exists(DATASET):
        print(f"Dataset {DATASET} not found.")
        return

    df = pd.read_csv(DATASET)
    print("=" * 65)
    print("RUNNING MLTRAIN-SCHED ABLATION STUDIES (Section 28)")
    print("=" * 65)

    ablations = {}

    # 1. Full Policy
    print("Running Variant Full: Complete MLTrain-Sched Policy...")
    res_full = ml_guided_scheduler(df, aging_weight=0.25, confidence_threshold=0.65, min_quantum=4, max_quantum=20)
    ablations["Full ML-Guided Policy"] = evaluate_run(res_full)

    # 2. A0: No ML (Rule-based policy fallback always)
    print("Running Variant A0: No ML (Rule-Based Heuristic Only)...")
    res_a0 = ml_guided_scheduler(df, aging_weight=0.25, confidence_threshold=1.01) # Force fallback
    ablations["A0: No ML (Rule-Based)"] = evaluate_run(res_a0)

    # 3. A1: No Adaptive Quantum (Fixed quantum = 10)
    print("Running Variant A1: No Adaptive Quantum (Fixed Quantum = 10)...")
    res_a1 = ml_guided_scheduler(df, aging_weight=0.25, confidence_threshold=0.65, min_quantum=10, max_quantum=10)
    ablations["A1: No Adaptive Quantum"] = evaluate_run(res_a1)

    # 4. A2: No Aging (Aging weight = 0.0)
    print("Running Variant A2: No Aging (Starvation Risk)...")
    res_a2 = ml_guided_scheduler(df, aging_weight=0.0, confidence_threshold=0.65)
    ablations["A2: No Aging"] = evaluate_run(res_a2)

    # 5. A5: No Confidence Fallback (Trust ML even when confidence is 0)
    print("Running Variant A5: No Confidence Fallback...")
    res_a5 = ml_guided_scheduler(df, aging_weight=0.25, confidence_threshold=0.0)
    ablations["A5: No Confidence Fallback"] = evaluate_run(res_a5)

    ablation_df = pd.DataFrame(ablations).T
    os.makedirs("data", exist_ok=True)
    ablation_df.to_csv("data/ablation_results.csv")
    print("\n" + "=" * 65)
    print("ABLATION STUDY RESULTS:")
    print("=" * 65)
    print(ablation_df)
    print("\nSaved to data/ablation_results.csv")

if __name__ == "__main__":
    run_ablations()
