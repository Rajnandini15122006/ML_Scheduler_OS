"""
src/expanded_benchmarks.py
Executes Roadmap Priority 5 (Strengthen & Expand Benchmarking):
1. Evaluates 5 Schedulers: FCFS, Round Robin (Q=10), Priority, SJF, and MLTrain-Sched.
2. Across 3 Distinct Workload Profiles:
   - Profile A: ML Training Cluster (60% ML, 20% IO, 10% Mixed, 10% CPU)
   - Profile B: Interactive Dev Node (45% IO, 30% Mixed, 15% CPU, 10% ML)
   - Profile C: Contention Server (50% CPU noise, 20% ML, 15% IO, 15% Mixed)
3. Evaluated across 5 Independent Random Seeds (42, 101, 202, 303, 404).
4. Records Mean, P50, P90, P99, Max Wait (Starvation), Turnaround, Response, Jain Fairness,
   and ML/IO critical delays.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "data"
RESULTS_DIR = "results"
MODELS_DIR = "models"

SEEDS = [42, 101, 202, 303, 404]
N_TASKS = 500

PROFILES = {
    "ML_Cluster": {
        "desc": "ML Training Cluster (60% ML, 20% IO, 10% Mixed, 10% CPU)",
        "probs": [0.10, 0.20, 0.10, 0.60]  # [CPU, IO, MIXED, ML]
    },
    "Interactive_Dev": {
        "desc": "Interactive Development Workstation (45% IO, 30% Mixed, 15% CPU, 10% ML)",
        "probs": [0.15, 0.45, 0.30, 0.10]
    },
    "Contention_Server": {
        "desc": "Contention Server (50% CPU Noise, 20% ML, 15% IO, 15% Mixed)",
        "probs": [0.50, 0.15, 0.15, 0.20]
    }
}

CLASS_NAMES = ["CPU_BOUND", "IO_BOUND", "MIXED", "ML_TRAINING"]


def generate_scenario_workload(profile_name, seed, n=N_TASKS):
    np.random.seed(seed)
    probs = PROFILES[profile_name]["probs"]
    workload_types = np.random.choice(CLASS_NAMES, size=n, p=probs)

    records = []
    current_arrival = 0

    for pid in range(1, n + 1):
        current_arrival += np.random.randint(1, 8)
        wtype = workload_types[pid - 1]

        if wtype == "CPU_BOUND":
            cpu_burst = int(np.clip(np.random.normal(65, 18), 5, 120))
            io_burst = int(np.clip(np.random.normal(8, 6), 0, 30))
            previous_runtime = int(np.clip(np.random.normal(55, 20), 0, 120))
            context_switches = int(np.clip(np.random.normal(4, 2), 1, 15))
            mem_rss_mb = int(np.clip(np.random.normal(512, 128), 128, 1024))
            num_threads = int(np.clip(np.random.choice([1, 2, 4], p=[0.6, 0.3, 0.1]), 1, 8))
        elif wtype == "IO_BOUND":
            cpu_burst = int(np.clip(np.random.normal(18, 10), 3, 50))
            io_burst = int(np.clip(np.random.normal(65, 20), 10, 120))
            previous_runtime = int(np.clip(np.random.normal(18, 10), 0, 70))
            context_switches = int(np.clip(np.random.normal(13, 5), 1, 25))
            mem_rss_mb = int(np.clip(np.random.normal(256, 64), 64, 512))
            num_threads = int(np.clip(np.random.choice([1, 2], p=[0.7, 0.3]), 1, 4))
        elif wtype == "MIXED":
            cpu_burst = int(np.clip(np.random.normal(44, 20), 5, 110))
            io_burst = int(np.clip(np.random.normal(40, 20), 0, 100))
            previous_runtime = int(np.clip(np.random.normal(45, 20), 0, 110))
            context_switches = int(np.clip(np.random.normal(9, 4), 1, 20))
            mem_rss_mb = int(np.clip(np.random.normal(768, 256), 256, 2048))
            num_threads = int(np.clip(np.random.choice([2, 3, 4], p=[0.5, 0.3, 0.2]), 1, 6))
        else:  # ML_TRAINING
            cpu_burst = int(np.clip(np.random.normal(56, 22), 10, 115))
            io_burst = int(np.clip(np.random.normal(36, 22), 0, 100))
            previous_runtime = int(np.clip(np.random.normal(52, 22), 0, 120))
            context_switches = int(np.clip(np.random.normal(9, 4), 1, 20))
            mem_rss_mb = int(np.clip(np.random.normal(4096, 1024), 1024, 16384))
            num_threads = int(np.clip(np.random.choice([4, 6, 8, 12], p=[0.4, 0.3, 0.2, 0.1]), 2, 16))

        priority = int(np.clip(np.random.normal(5.5, 2.2), 1, 10))

        records.append({
            "process_id": pid,
            "arrival_time": current_arrival,
            "cpu_burst": cpu_burst,
            "io_burst": io_burst,
            "priority": priority,
            "previous_runtime": previous_runtime,
            "context_switches": context_switches,
            "mem_rss_mb": mem_rss_mb,
            "num_threads": num_threads,
            "workload_type": wtype
        })

    return pd.DataFrame(records)


def calculate_jain_fairness(times):
    arr = np.array(times, dtype=np.float64)
    if len(arr) == 0 or np.sum(arr**2) == 0:
        return 1.0
    return float((np.sum(arr)**2) / (len(arr) * np.sum(arr**2)))


# ==============================================================================
# SCHEDULER ALGORITHMS
# ==============================================================================

def sim_fcfs(df):
    processes = df.sort_values(["arrival_time", "process_id"]).copy()
    current_time = 0
    rows = []
    for _, p in processes.iterrows():
        start = max(current_time, p["arrival_time"])
        completion = start + p["cpu_burst"]
        rows.append({
            **p.to_dict(),
            "start_time": start,
            "completion_time": completion,
            "waiting_time": start - p["arrival_time"],
            "turnaround_time": completion - p["arrival_time"],
            "response_time": start - p["arrival_time"]
        })
        current_time = completion
    return pd.DataFrame(rows)


def sim_sjf(df):
    processes = df.copy()
    remaining = set(processes["process_id"])
    p_dict = {r["process_id"]: r.to_dict() for _, r in processes.iterrows()}
    current_time = 0
    rows = []

    while remaining:
        available = [p_dict[pid] for pid in remaining if p_dict[pid]["arrival_time"] <= current_time]
        if not available:
            min_arrival = min(p_dict[pid]["arrival_time"] for pid in remaining)
            current_time = max(current_time, min_arrival)
            continue

        selected = min(available, key=lambda x: (x["cpu_burst"], x["arrival_time"], x["process_id"]))
        start = current_time
        completion = start + selected["cpu_burst"]
        rows.append({
            **selected,
            "start_time": start,
            "completion_time": completion,
            "waiting_time": start - selected["arrival_time"],
            "turnaround_time": completion - selected["arrival_time"],
            "response_time": start - selected["arrival_time"]
        })
        current_time = completion
        remaining.remove(selected["process_id"])
    return pd.DataFrame(rows)


def sim_priority(df):
    processes = df.copy()
    remaining = set(processes["process_id"])
    p_dict = {r["process_id"]: r.to_dict() for _, r in processes.iterrows()}
    current_time = 0
    rows = []

    while remaining:
        available = [p_dict[pid] for pid in remaining if p_dict[pid]["arrival_time"] <= current_time]
        if not available:
            min_arrival = min(p_dict[pid]["arrival_time"] for pid in remaining)
            current_time = max(current_time, min_arrival)
            continue

        # Higher priority value runs first
        selected = max(available, key=lambda x: (x["priority"], -x["arrival_time"], -x["process_id"]))
        start = current_time
        completion = start + selected["cpu_burst"]
        rows.append({
            **selected,
            "start_time": start,
            "completion_time": completion,
            "waiting_time": start - selected["arrival_time"],
            "turnaround_time": completion - selected["arrival_time"],
            "response_time": start - selected["arrival_time"]
        })
        current_time = completion
        remaining.remove(selected["process_id"])
    return pd.DataFrame(rows)


def sim_round_robin(df, quantum=10):
    processes = df.sort_values(["arrival_time", "process_id"]).reset_index(drop=True)
    remaining_burst = {int(r["process_id"]): int(r["cpu_burst"]) for _, r in processes.iterrows()}
    first_start = {}
    completion_times = {}

    ready_queue = []
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

        pid = ready_queue.pop(0)
        if pid not in first_start:
            first_start[pid] = current_time

        execution = min(quantum, remaining_burst[pid])
        current_time += execution
        remaining_burst[pid] -= execution

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
            "start_time": st,
            "completion_time": ct,
            "waiting_time": ct - p["arrival_time"] - p["cpu_burst"],
            "turnaround_time": ct - p["arrival_time"],
            "response_time": st - p["arrival_time"]
        })
    return pd.DataFrame(rows)


def sim_mltrain_sched(df, tuned_artifacts, aging_weight=0.25, starvation_limit=15000):
    model = tuned_artifacts["model"]
    le = tuned_artifacts["encoder"]

    # Add engineered features
    df_feat = df.copy()
    df_feat["io_to_cpu_ratio"] = df_feat["io_burst"] / (df_feat["cpu_burst"] + 1.0)
    df_feat["cpu_fraction"] = df_feat["cpu_burst"] / (df_feat["cpu_burst"] + df_feat["io_burst"] + 1.0)
    df_feat["switch_intensity"] = df_feat["context_switches"] / (df_feat["previous_runtime"] + 1.0)
    
    feats = [
        "cpu_burst", "io_burst", "priority", "previous_runtime", "context_switches",
        "mem_rss_mb", "num_threads",
        "io_to_cpu_ratio", "cpu_fraction", "switch_intensity"
    ]

    preds = le.inverse_transform(model.predict(df_feat[feats]))
    df_feat["predicted_workload"] = preds

    def assign_dsq(wtype, prio):
        if wtype in ["ML_TRAINING", "IO_BOUND"]:
            return "HIGH" if prio >= 3 else "MEDIUM"
        elif wtype == "CPU_BOUND":
            return "MEDIUM" if prio >= 8 else "LOW"
        else:
            return "HIGH" if prio >= 8 else "MEDIUM"

    def get_quantum(wtype):
        return {"CPU_BOUND": 18, "IO_BOUND": 5, "ML_TRAINING": 14, "MIXED": 10}.get(wtype, 10)

    df_feat["assigned_dsq"] = [assign_dsq(w, p) for w, p in zip(preds, df_feat["priority"])]
    df_feat["target_quantum"] = [get_quantum(w) for w in preds]

    # Preemptive Multi-Queue Simulation
    processes = df_feat.sort_values(["arrival_time", "process_id"]).reset_index(drop=True)
    remaining_burst = {int(r["process_id"]): int(r["cpu_burst"]) for _, r in processes.iterrows()}
    first_start = {}
    completion_times = {}

    ready_queue = []
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

        def get_task_sort_key(pid):
            info = p_dict[pid]
            wait = current_time - info["arrival_time"]
            dsq = info["assigned_dsq"]
            dsq_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[dsq]
            score = (info["priority"] * 0.3) + (aging_weight * (wait / 500.0))
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
            "start_time": st,
            "completion_time": ct,
            "waiting_time": ct - p["arrival_time"] - p["cpu_burst"],
            "turnaround_time": ct - p["arrival_time"],
            "response_time": st - p["arrival_time"]
        })
    return pd.DataFrame(rows)


def extract_run_metrics(res_df, profile, seed, algorithm):
    w = res_df["waiting_time"]
    t = res_df["turnaround_time"]
    r = res_df["response_time"]

    ml_w = res_df[res_df["workload_type"] == "ML_TRAINING"]["waiting_time"]
    io_w = res_df[res_df["workload_type"] == "IO_BOUND"]["waiting_time"]

    return {
        "profile": profile,
        "seed": seed,
        "algorithm": algorithm,
        "mean_wait": float(w.mean()),
        "p50_wait": float(w.median()),
        "p90_wait": float(np.percentile(w, 90)),
        "p99_wait": float(np.percentile(w, 99)),
        "max_wait": float(w.max()),
        "mean_turnaround": float(t.mean()),
        "mean_response": float(r.mean()),
        "jain_fairness": calculate_jain_fairness(w),
        "ml_train_wait": float(ml_w.mean()) if len(ml_w) > 0 else 0.0,
        "io_bound_wait": float(io_w.mean()) if len(io_w) > 0 else 0.0
    }


def run_expanded_benchmarks():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("ROADMAP PRIORITY 5: EXPANDED MULTI-SEED & MULTI-PROFILE BENCHMARK")
    print("=" * 70)

    # Load tuned classifier artifacts
    model_path = os.path.join(MODELS_DIR, "best_tuned_classifier.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}. Run tune_and_evaluate_scheduler.py first.")

    with open(model_path, "rb") as f:
        tuned_artifacts = pickle.load(f)
    print(f"Loaded classifier engine: {tuned_artifacts.get('name', 'Tuned Model')}")

    algorithms = {
        "FCFS": sim_fcfs,
        "Round Robin (Q=10)": sim_round_robin,
        "Priority": sim_priority,
        "SJF": sim_sjf,
        "MLTrain-Sched": lambda df: sim_mltrain_sched(df, tuned_artifacts)
    }

    raw_results = []

    total_runs = len(PROFILES) * len(SEEDS) * len(algorithms)
    current_run = 0

    print(f"\nRunning {total_runs} simulation evaluations across 3 profiles and 5 seeds...")

    for prof_name in PROFILES:
        print(f"\n>>> Workload Profile: {prof_name} ({PROFILES[prof_name]['desc']})")
        for seed in SEEDS:
            df_workload = generate_scenario_workload(prof_name, seed)

            for algo_name, algo_fn in algorithms.items():
                current_run += 1
                res_df = algo_fn(df_workload)
                metrics = extract_run_metrics(res_df, prof_name, seed, algo_name)
                raw_results.append(metrics)

    df_raw = pd.DataFrame(raw_results)
    raw_path = os.path.join(DATA_DIR, "expanded_benchmark_results.csv")
    df_raw.to_csv(raw_path, index=False)
    print(f"\nSaved raw multi-seed benchmark results to: {raw_path}")

    # Compute Summary Statistics (Mean +/- Std across seeds)
    summary = df_raw.groupby(["profile", "algorithm"]).agg({
        "mean_wait": ["mean", "std"],
        "p50_wait": ["mean", "std"],
        "p90_wait": ["mean", "std"],
        "max_wait": ["mean", "std"],
        "mean_turnaround": ["mean", "std"],
        "mean_response": ["mean", "std"],
        "jain_fairness": ["mean", "std"],
        "ml_train_wait": ["mean", "std"],
        "io_bound_wait": ["mean", "std"]
    }).reset_index()

    # Flatten columns
    summary.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in summary.columns]
    summary_path = os.path.join(DATA_DIR, "expanded_benchmark_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved aggregated benchmark summary to: {summary_path}")

    print("\n" + "=" * 70)
    print("SUMMARY RESULTS TABLE (MEAN WAITING TIME ACROSS SEEDS)")
    print("=" * 70)
    display_cols = ["profile", "algorithm", "mean_wait_mean", "ml_train_wait_mean", "io_bound_wait_mean", "jain_fairness_mean"]
    print(summary[display_cols].to_string(index=False))

    # Generate Publication Visual
    plot_expanded_benchmarks(df_raw, summary)

    return summary


def plot_expanded_benchmarks(df_raw, summary):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#FFFFFF")

    profiles = list(PROFILES.keys())
    algos = ["FCFS", "Round Robin (Q=10)", "Priority", "SJF", "MLTrain-Sched"]
    colors = ["#64748B", "#F59E0B", "#8B5CF6", "#3B82F6", "#10B981"]

    # 1. Mean Waiting Time by Profile
    x = np.arange(len(profiles))
    width = 0.16

    for i, algo in enumerate(algos):
        means = []
        for p in profiles:
            val = summary[(summary["profile"] == p) & (summary["algorithm"] == algo)]["mean_wait_mean"].values[0]
            means.append(val)
        axes[0].bar(x + i*width - 2*width, means, width, label=algo, color=colors[i])

    axes[0].set_ylabel("Mean Waiting Time (ticks)", fontsize=10, fontweight="bold")
    axes[0].set_title("Overall Mean Waiting Time by Workload Profile", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["ML Cluster", "Interactive Dev", "Contention Server"], fontsize=10, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    # 2. Critical ML Training Delay (DataLoader Waiting Time)
    for i, algo in enumerate(algos):
        means = []
        for p in profiles:
            val = summary[(summary["profile"] == p) & (summary["algorithm"] == algo)]["ml_train_wait_mean"].values[0]
            means.append(val)
        axes[1].bar(x + i*width - 2*width, means, width, label=algo, color=colors[i])

    axes[1].set_ylabel("ML Training Delay (ticks)", fontsize=10, fontweight="bold")
    axes[1].set_title("ML Training Workload Queue Delay (GPU Unblocking)", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["ML Cluster", "Interactive Dev", "Contention Server"], fontsize=10, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    # 3. Interactive I/O Responsiveness
    for i, algo in enumerate(algos):
        means = []
        for p in profiles:
            val = summary[(summary["profile"] == p) & (summary["algorithm"] == algo)]["io_bound_wait_mean"].values[0]
            means.append(val)
        axes[2].bar(x + i*width - 2*width, means, width, label=algo, color=colors[i])

    axes[2].set_ylabel("I/O Waiting Time (ticks)", fontsize=10, fontweight="bold")
    axes[2].set_title("Interactive I/O Latency (Lower is Better)", fontsize=11, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["ML Cluster", "Interactive Dev", "Contention Server"], fontsize=10, fontweight="bold")
    axes[2].grid(axis="y", linestyle="--", alpha=0.3)

    plt.suptitle("MLTrain-Sched Expanded Multi-Seed Benchmark across 3 Workload Profiles (5 Seeds, N=500 per run)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_img = os.path.join(RESULTS_DIR, "expanded_benchmark_multi_profile.png")
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"Saved expanded multi-profile benchmark plot to: {out_img}")


if __name__ == "__main__":
    run_expanded_benchmarks()
