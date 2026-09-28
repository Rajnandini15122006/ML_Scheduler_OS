import pandas as pd
import numpy as np

from schedulers_realistic import (
    fcfs,
    round_robin,
    priority,
    sjf
)

from ml_guided_scheduler import ml_guided_scheduler


DATASET = "data/workloads_realistic.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATASET)

print("=" * 70)
print("CPU SCHEDULER COMPARISON")
print("=" * 70)

print(f"\nDataset: {DATASET}")
print(f"Processes: {len(df)}")


# ============================================================
# FAIRNESS FUNCTION
# ============================================================

def jains_index(values):

    values = np.asarray(values, dtype=float)

    values = np.maximum(values, 1e-9)

    numerator = np.sum(values) ** 2

    denominator = (
        len(values)
        *
        np.sum(values ** 2)
    )

    return numerator / denominator


def calculate_metrics(result):

    avg_waiting = result["waiting_time"].mean()

    avg_turnaround = result["turnaround_time"].mean()

    avg_response = result["response_time"].mean()

    # --------------------------------------------------------
    # Waiting fairness
    # --------------------------------------------------------

    max_wait = result["waiting_time"].max()

    if max_wait == 0:

        waiting_fairness = np.ones(
            len(result)
        )

    else:

        waiting_fairness = (
            1 -
            result["waiting_time"] /
            max_wait
        )

        waiting_fairness = np.clip(
            waiting_fairness,
            0.01,
            None
        )

    fairness = jains_index(
        waiting_fairness
    )

    # --------------------------------------------------------
    # CPU service fairness
    # --------------------------------------------------------

    cpu_service = result["cpu_burst"].values

    cpu_fairness = jains_index(
        cpu_service
    )

    return {

        "Average Waiting Time":
            avg_waiting,

        "Average Turnaround Time":
            avg_turnaround,

        "Average Response Time":
            avg_response,

        "Jain Fairness":
            fairness,

        "CPU Service Fairness":
            cpu_fairness
    }


def extract_dataframe(result):
    """
    schedulers_realistic functions return:
        (DataFrame, metrics_dict)

    This function extracts only the DataFrame.
    """
    if isinstance(result, tuple):
        return result[0]

    return result


# ============================================================
# RUN ALGORITHMS
# ============================================================

results = {}


print("\nRunning FCFS...")
fcfs_result = extract_dataframe(
    fcfs(df.copy())
)

results["FCFS"] = calculate_metrics(
    fcfs_result
)


print("\nRunning Round Robin...")
rr_result = extract_dataframe(
    round_robin(df.copy())
)

results["Round Robin"] = calculate_metrics(
    rr_result
)


print("\nRunning Priority...")
priority_result = extract_dataframe(
    priority(df.copy())
)

results["Priority"] = calculate_metrics(
    priority_result
)


print("\nRunning SJF...")
sjf_result = extract_dataframe(
    sjf(df.copy())
)

results["SJF"] = calculate_metrics(
    sjf_result
)


print("Running ML-Guided...")

ml_result = ml_guided_scheduler(
    df.copy()
)

results["ML-Guided"] = calculate_metrics(
    ml_result
)


# ============================================================
# COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(
    results
).T


comparison = comparison.round(3)


print("\n" + "=" * 70)
print("FINAL SCHEDULER COMPARISON")
print("=" * 70)

print(
    comparison.to_string()
)


# ============================================================
# SAVE TABLE
# ============================================================

comparison.to_csv(
    "data/scheduler_comparison.csv"
)

print(
    "\nComparison saved to:"
)

print(
    "data/scheduler_comparison.csv"
)