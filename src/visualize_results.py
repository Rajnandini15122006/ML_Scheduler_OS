import pandas as pd
import matplotlib.pyplot as plt


INPUT = "data/scheduler_comparison.csv"


df = pd.read_csv(
    INPUT,
    index_col=0
)


# ============================================================
# WAITING TIME
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    df.index,
    df["Average Waiting Time"]
)

plt.ylabel("Average Waiting Time")

plt.xlabel("Scheduling Algorithm")

plt.title(
    "Average Waiting Time Comparison"
)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "data/average_waiting_time.png",
    dpi=300
)

plt.show()


# ============================================================
# TURNAROUND TIME
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    df.index,
    df["Average Turnaround Time"]
)

plt.ylabel("Average Turnaround Time")

plt.xlabel("Scheduling Algorithm")

plt.title(
    "Average Turnaround Time Comparison"
)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "data/average_turnaround_time.png",
    dpi=300
)

plt.show()


# ============================================================
# RESPONSE TIME
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    df.index,
    df["Average Response Time"]
)

plt.ylabel("Average Response Time")

plt.xlabel("Scheduling Algorithm")

plt.title(
    "Average Response Time Comparison"
)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "data/average_response_time.png",
    dpi=300
)

plt.show()


# ============================================================
# JAIN FAIRNESS
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    df.index,
    df["Jain Fairness"]
)

plt.ylabel("Jain's Fairness Index")

plt.xlabel("Scheduling Algorithm")

plt.title(
    "Fairness Comparison"
)

plt.ylim(0, 1.05)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "data/jain_fairness.png",
    dpi=300
)

plt.show()


# ============================================================
# CPU SERVICE FAIRNESS
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    df.index,
    df["CPU Service Fairness"]
)

plt.ylabel("Jain's Fairness Index")

plt.xlabel("Scheduling Algorithm")

plt.title(
    "CPU Service Fairness Comparison"
)

plt.ylim(0, 1.05)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "data/cpu_service_fairness.png",
    dpi=300
)

plt.show()


print(
    "\nAll charts generated successfully."
)

print(
    "\nFiles:"
)

print("data/average_waiting_time.png")
print("data/average_turnaround_time.png")
print("data/average_response_time.png")
print("data/jain_fairness.png")
print("data/cpu_service_fairness.png")