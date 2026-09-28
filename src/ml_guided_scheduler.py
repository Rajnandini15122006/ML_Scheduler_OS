import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATASET = "data/workloads_realistic.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATASET)

print("=" * 60)
print("ML-GUIDED CPU SCHEDULER")
print("=" * 60)

print("\nDataset loaded successfully.")
print(f"Number of processes: {len(df)}")


# ============================================================
# ML MODEL
# ============================================================

FEATURES = [
    "cpu_burst",
    "io_burst",
    "priority",
    "previous_runtime",
    "context_switches"
]

TARGET = "workload_type"

X = df[FEATURES]
y = df[TARGET]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("\nRandom Forest trained successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

df["predicted_workload_id"] = model.predict(df[FEATURES])
df["predicted_workload"] = encoder.inverse_transform(
    df["predicted_workload_id"]
)

probabilities = model.predict_proba(df[FEATURES])

df["prediction_confidence"] = probabilities.max(axis=1)


# ============================================================
# FAIRNESS-AWARE SCHEDULING
# ============================================================

def calculate_fairness_score(waiting_time, max_waiting):
    """
    Larger score = process deserves more attention.

    Waiting time is normalized so that processes waiting longer
    receive increasing scheduling priority.
    """

    if max_waiting <= 0:
        return 0.0

    return min(waiting_time / max_waiting, 1.0)


def workload_weight(workload):
    """
    ML workload-aware scheduling preference.

    These weights should NOT completely dominate fairness.
    """

    weights = {
        "CPU_BOUND": 0.45,
        "IO_BOUND": 0.55,
        "MIXED": 0.60,
        "ML_TRAINING": 0.65
    }

    return weights.get(workload, 0.50)


def calculate_dynamic_score(row, current_time, max_waiting):
    """
    Combined scheduling score.

    Components:

    30% priority
    20% short-job preference
    20% ML workload preference
    30% aging/fairness

    Aging prevents starvation.
    """

    priority_score = row["priority"] / 10.0

    burst_score = 1.0 / (1.0 + row["cpu_burst"])

    ml_score = workload_weight(row["predicted_workload"])

    waiting = max(0, current_time - row["arrival_time"])

    fairness_score = calculate_fairness_score(
        waiting,
        max_waiting
    )

    score = (
        0.30 * priority_score
        + 0.20 * burst_score
        + 0.20 * ml_score
        + 0.30 * fairness_score
    )

    return score


# ============================================================
# ML-GUIDED FAIR SCHEDULER
# ============================================================

def ml_guided_scheduler(data):

    processes = data.copy()

    # --------------------------------------------------------
    # ML PREDICTIONS FOR INPUT DATA
    # --------------------------------------------------------

    processes["predicted_workload_id"] = model.predict(
        processes[FEATURES]
    )

    processes["predicted_workload"] = encoder.inverse_transform(
        processes["predicted_workload_id"]
    )

    probabilities = model.predict_proba(
        processes[FEATURES]
    )

    processes["prediction_confidence"] = probabilities.max(
        axis=1
    )

    completed = set()

    current_time = 0

    schedule = []

    n = len(processes)

    while len(completed) < n:

        # ----------------------------------------------------
        # READY QUEUE
        # ----------------------------------------------------

        ready = processes[
            (processes["arrival_time"] <= current_time)
            & (~processes["process_id"].isin(completed))
        ].copy()

        # ----------------------------------------------------
        # CPU IDLE
        # ----------------------------------------------------

        if ready.empty:

            remaining = processes[
                ~processes["process_id"].isin(completed)
            ]

            next_arrival = remaining["arrival_time"].min()

            current_time = max(
                current_time,
                next_arrival
            )

            continue

        # ----------------------------------------------------
        # WAITING TIME
        # ----------------------------------------------------

        ready["current_waiting"] = (
            current_time - ready["arrival_time"]
        ).clip(lower=0)

        max_waiting = max(
            1,
            ready["current_waiting"].max()
        )

        # ----------------------------------------------------
        # CALCULATE SCORE
        # ----------------------------------------------------

        ready["dynamic_score"] = ready.apply(
            lambda row: calculate_dynamic_score(
                row,
                current_time,
                max_waiting
            ),
            axis=1
        )

        # ----------------------------------------------------
        # ANTI-STARVATION
        # ----------------------------------------------------

        # If a process has waited for a long time,
        # force it ahead of normal ML preferences.

        starvation_threshold = 250

        starving = ready[
            ready["current_waiting"] >= starvation_threshold
        ]

        if not starving.empty:

            selected = starving.sort_values(
                by=[
                    "current_waiting",
                    "priority"
                ],
                ascending=[
                    False,
                    False
                ]
            ).iloc[0]

        else:

            selected = ready.sort_values(
                by=[
                    "dynamic_score",
                    "current_waiting"
                ],
                ascending=[
                    False,
                    False
                ]
            ).iloc[0]

        # ----------------------------------------------------
        # EXECUTION
        # ----------------------------------------------------

        pid = selected["process_id"]

        arrival = selected["arrival_time"]

        burst = selected["cpu_burst"]

        start_time = max(
            current_time,
            arrival
        )

        completion_time = start_time + burst

        waiting_time = (
            start_time - arrival
        )

        turnaround_time = (
            completion_time - arrival
        )

        response_time = (
            start_time - arrival
        )

        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        schedule.append({

            "process_id": pid,

            "arrival_time": arrival,

            "cpu_burst": burst,

            "start_time": start_time,

            "completion_time": completion_time,

            "waiting_time": waiting_time,

            "turnaround_time": turnaround_time,

            "response_time": response_time,

            "predicted_workload":
                selected["predicted_workload"],

            "prediction_confidence":
                selected["prediction_confidence"],

            "dynamic_score":
                selected["dynamic_score"]

        })

        completed.add(pid)

        current_time = completion_time

    return pd.DataFrame(schedule)


# ============================================================
# RUN SCHEDULER
# ============================================================

result = ml_guided_scheduler(df)


# ============================================================
# METRICS
# ============================================================

avg_waiting = result["waiting_time"].mean()

avg_turnaround = result["turnaround_time"].mean()

avg_response = result["response_time"].mean()


print("\n" + "=" * 60)
print("ML-GUIDED FAIR SCHEDULER METRICS")
print("=" * 60)

print(
    f"\nAverage waiting time: "
    f"{avg_waiting:.2f}"
)

print(
    f"Average turnaround time: "
    f"{avg_turnaround:.2f}"
)

print(
    f"Average response time: "
    f"{avg_response:.2f}"
)


# ============================================================
# FAIRNESS METRIC
# ============================================================

# ------------------------------------------------------------
# Fairness based on normalized CPU service
# ------------------------------------------------------------

total_cpu = result["cpu_burst"].sum()

result["cpu_share"] = (
    result["cpu_burst"] / total_cpu
)

jain_cpu = (
    result["cpu_share"].sum() ** 2
    /
    (
        len(result)
        *
        (result["cpu_share"] ** 2).sum()
    )
)


# ------------------------------------------------------------
# Fairness based on waiting-time balance
# ------------------------------------------------------------

# Convert waiting time into a "service fairness" score.
# Larger waiting time => lower score.

max_wait = result["waiting_time"].max()

result["waiting_fairness"] = (
    1 -
    (
        result["waiting_time"] /
        max_wait
    )
)

result["waiting_fairness"] = (
    result["waiting_fairness"]
    .clip(lower=0.01)
)

jain_waiting = (
    result["waiting_fairness"].sum() ** 2
    /
    (
        len(result)
        *
        (
            result["waiting_fairness"] ** 2
        ).sum()
    )
)


print(
    f"\nJain's fairness - CPU service: "
    f"{jain_cpu:.4f}"
)

print(
    f"Jain's fairness - waiting balance: "
    f"{jain_waiting:.4f}"
)


# ============================================================
# WORKLOAD DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("PREDICTED WORKLOAD DISTRIBUTION")
print("=" * 60)

print(
    df["predicted_workload"].value_counts()
)


# ============================================================
# FIRST 15 PROCESSES
# ============================================================

print("\n" + "=" * 60)
print("FIRST 15 SCHEDULED PROCESSES")
print("=" * 60)

display_columns = [

    "process_id",
    "arrival_time",
    "cpu_burst",
    "start_time",
    "completion_time",
    "waiting_time",
    "turnaround_time",
    "response_time",
    "predicted_workload",
    "prediction_confidence",
    "dynamic_score"

]

print(
    result[
        display_columns
    ].head(15).to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

result.to_csv(
    "data/ml_guided_results.csv",
    index=False
)

print(
    "\nResults saved to "
    "data/ml_guided_results.csv"
)