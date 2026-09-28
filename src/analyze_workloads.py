import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

dataset = pd.read_csv("data/workloads.csv")


# ---------------------------------------------------------
# Basic dataset information
# ---------------------------------------------------------

print("=" * 60)
print("WORKLOAD DATASET ANALYSIS")
print("=" * 60)

print("\nDataset shape:")
print(dataset.shape)

print("\nColumns:")
print(list(dataset.columns))


# ---------------------------------------------------------
# Workload distribution
# ---------------------------------------------------------

print("\nWorkload distribution:")
print(dataset["workload_type"].value_counts())


# ---------------------------------------------------------
# Statistical summary
# ---------------------------------------------------------

print("\nStatistical summary:")
print(dataset.describe())


# ---------------------------------------------------------
# Average characteristics by workload type
# ---------------------------------------------------------

print("\nAverage characteristics by workload type:")

summary = dataset.groupby("workload_type")[
    [
        "cpu_burst",
        "io_burst",
        "priority",
        "previous_runtime",
        "context_switches"
    ]
].mean()

print(summary.round(2))


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

dataset["workload_type"].value_counts().plot(
    kind="bar"
)

plt.title("Synthetic Workload Distribution")
plt.xlabel("Workload Type")
plt.ylabel("Number of Processes")

plt.tight_layout()

plt.savefig("results/workload_distribution.png")

plt.show()