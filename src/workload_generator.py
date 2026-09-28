import os
import numpy as np
import pandas as pd

np.random.seed(42)

N_PROCESSES = 1000
OUTPUT_FILE = "data/workloads_realistic.csv"


def generate_workload(n=N_PROCESSES):
    workload_types = np.random.choice(
        ["CPU_BOUND", "IO_BOUND", "MIXED", "ML_TRAINING"],
        size=n,
        p=[0.287, 0.248, 0.221, 0.244]
    )

    records = []

    current_arrival = 0

    for pid in range(1, n + 1):

        # Realistic irregular arrivals
        current_arrival += np.random.randint(1, 10)

        wtype = workload_types[pid - 1]

        if wtype == "CPU_BOUND":
            cpu_burst = int(np.clip(np.random.normal(65, 18), 5, 120))
            io_burst = int(np.clip(np.random.normal(8, 6), 0, 30))
            previous_runtime = int(np.clip(np.random.normal(55, 20), 0, 120))
            context_switches = int(np.clip(np.random.normal(4, 2), 1, 15))

        elif wtype == "IO_BOUND":
            cpu_burst = int(np.clip(np.random.normal(18, 10), 3, 50))
            io_burst = int(np.clip(np.random.normal(65, 20), 10, 120))
            previous_runtime = int(np.clip(np.random.normal(18, 10), 0, 70))
            context_switches = int(np.clip(np.random.normal(13, 5), 1, 25))

        elif wtype == "MIXED":
            cpu_burst = int(np.clip(np.random.normal(44, 20), 5, 110))
            io_burst = int(np.clip(np.random.normal(40, 20), 0, 100))
            previous_runtime = int(np.clip(np.random.normal(45, 20), 0, 110))
            context_switches = int(np.clip(np.random.normal(9, 4), 1, 20))

        else:  # ML_TRAINING
            cpu_burst = int(np.clip(np.random.normal(56, 22), 10, 115))
            io_burst = int(np.clip(np.random.normal(36, 22), 0, 100))
            previous_runtime = int(np.clip(np.random.normal(52, 22), 0, 120))
            context_switches = int(np.clip(np.random.normal(9, 4), 1, 20))

        priority = int(np.clip(np.random.normal(5.5, 2.2), 1, 10))

        records.append({
            "process_id": pid,
            "arrival_time": current_arrival,
            "cpu_burst": cpu_burst,
            "io_burst": io_burst,
            "priority": priority,
            "previous_runtime": previous_runtime,
            "waiting_time": 0,
            "context_switches": context_switches,
            "workload_type": wtype
        })

    return pd.DataFrame(records)


def main():

    os.makedirs("data", exist_ok=True)

    df = generate_workload()

    df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("REALISTIC WORKLOAD DATASET GENERATED")
    print("=" * 60)

    print("\nNumber of processes:")
    print(len(df))

    print("\nWorkload distribution:")
    print(df["workload_type"].value_counts())

    print("\nDataset preview:")
    print(df.head(10).to_string(index=False))

    print("\nAverage characteristics:")
    print(
        df.groupby("workload_type")[
            [
                "cpu_burst",
                "io_burst",
                "priority",
                "previous_runtime",
                "context_switches"
            ]
        ].mean().round(2)
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()