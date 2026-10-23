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
            "waiting_time": 0,
            "context_switches": context_switches,
            "mem_rss_mb": mem_rss_mb,
            "num_threads": num_threads,
            "workload_type": wtype
        })

    return pd.DataFrame(records)


def main():
    os.makedirs("data", exist_ok=True)
    df = generate_workload()
    df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("REALISTIC WORKLOAD DATASET GENERATED WITH MEMORY & THREADS")
    print("=" * 60)
    print("\nNumber of processes:")
    print(len(df))
    print("\nWorkload distribution:")
    print(df["workload_type"].value_counts())
    print("\nAverage characteristics by workload type:")
    print(
        df.groupby("workload_type")[
            [
                "cpu_burst",
                "io_burst",
                "priority",
                "previous_runtime",
                "context_switches",
                "mem_rss_mb",
                "num_threads"
            ]
        ].mean().round(2)
    )
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
