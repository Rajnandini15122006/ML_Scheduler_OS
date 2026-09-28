import pandas as pd

DATA_FILE = "data/workloads_realistic.csv"


def calculate_metrics(result):

    result["waiting_time"] = (
        result["completion_time"]
        - result["arrival_time"]
        - result["cpu_burst"]
    )

    result["turnaround_time"] = (
        result["completion_time"]
        - result["arrival_time"]
    )

    result["response_time"] = (
        result["start_time"]
        - result["arrival_time"]
    )

    metrics = {
        "waiting_time": result["waiting_time"].mean(),
        "turnaround_time": result["turnaround_time"].mean(),
        "response_time": result["response_time"].mean()
    }

    return result, metrics


def fcfs(df):

    processes = df.sort_values(
        ["arrival_time", "process_id"]
    ).copy()

    current_time = 0
    rows = []

    for _, p in processes.iterrows():

        current_time = max(
            current_time,
            p["arrival_time"]
        )

        start = current_time
        completion = start + p["cpu_burst"]

        rows.append({
            **p.to_dict(),
            "start_time": start,
            "completion_time": completion
        })

        current_time = completion

    result = pd.DataFrame(rows)

    return calculate_metrics(result)


def sjf(df):

    processes = df.copy()

    remaining = set(processes["process_id"])
    current_time = 0
    rows = []

    while remaining:

        available = processes[
            (processes["process_id"].isin(remaining))
            & (processes["arrival_time"] <= current_time)
        ]

        if available.empty:

            next_pid = processes[
                processes["process_id"].isin(remaining)
            ].sort_values(
                ["arrival_time", "process_id"]
            ).iloc[0]

            current_time = next_pid["arrival_time"]
            continue

        selected = available.sort_values(
            ["cpu_burst", "arrival_time", "process_id"]
        ).iloc[0]

        start = current_time
        completion = start + selected["cpu_burst"]

        rows.append({
            **selected.to_dict(),
            "start_time": start,
            "completion_time": completion
        })

        current_time = completion
        remaining.remove(selected["process_id"])

    result = pd.DataFrame(rows)

    return calculate_metrics(result)


def priority(df):

    processes = df.copy()

    remaining = set(processes["process_id"])
    current_time = 0
    rows = []

    while remaining:

        available = processes[
            (processes["process_id"].isin(remaining))
            & (processes["arrival_time"] <= current_time)
        ]

        if available.empty:

            next_pid = processes[
                processes["process_id"].isin(remaining)
            ].sort_values(
                ["arrival_time", "process_id"]
            ).iloc[0]

            current_time = next_pid["arrival_time"]
            continue

        selected = available.sort_values(
            ["priority", "arrival_time", "process_id"],
            ascending=[False, True, True]
        ).iloc[0]

        start = current_time
        completion = start + selected["cpu_burst"]

        rows.append({
            **selected.to_dict(),
            "start_time": start,
            "completion_time": completion
        })

        current_time = completion
        remaining.remove(selected["process_id"])

    result = pd.DataFrame(rows)

    return calculate_metrics(result)


def round_robin(df, quantum=10):

    processes = df.copy()

    processes = processes.sort_values(
        ["arrival_time", "process_id"]
    )

    remaining_burst = {
        int(row["process_id"]): int(row["cpu_burst"])
        for _, row in processes.iterrows()
    }

    first_start = {}
    completion_times = {}

    ready_queue = []

    process_ids = processes["process_id"].tolist()

    index = 0
    current_time = 0

    while len(completion_times) < len(processes):

        while (
            index < len(processes)
            and processes.iloc[index]["arrival_time"] <= current_time
        ):

            ready_queue.append(
                int(processes.iloc[index]["process_id"])
            )

            index += 1

        if not ready_queue:

            if index < len(processes):
                current_time = max(
                    current_time,
                    processes.iloc[index]["arrival_time"]
                )
                continue

        pid = ready_queue.pop(0)

        row = processes[
            processes["process_id"] == pid
        ].iloc[0]

        if pid not in first_start:
            first_start[pid] = current_time

        execution = min(
            quantum,
            remaining_burst[pid]
        )

        current_time += execution
        remaining_burst[pid] -= execution

        while (
            index < len(processes)
            and processes.iloc[index]["arrival_time"] <= current_time
        ):

            ready_queue.append(
                int(processes.iloc[index]["process_id"])
            )

            index += 1

        if remaining_burst[pid] > 0:
            ready_queue.append(pid)
        else:
            completion_times[pid] = current_time

    rows = []

    for _, p in processes.iterrows():

        pid = int(p["process_id"])

        rows.append({
            **p.to_dict(),
            "start_time": first_start[pid],
            "completion_time": completion_times[pid]
        })

    result = pd.DataFrame(rows)

    return calculate_metrics(result)


def main():

    df = pd.read_csv(DATA_FILE)

    print("=" * 60)
    print("CLASSICAL CPU SCHEDULERS")
    print("=" * 60)

    schedulers = {
        "FCFS": fcfs,
        "Round Robin": round_robin,
        "Priority": priority,
        "SJF": sjf
    }

    for name, scheduler in schedulers.items():

        result, metrics = scheduler(df)

        print("\n" + "=" * 60)
        print(name.upper())
        print("=" * 60)

        print(
            f"Average waiting time: "
            f"{metrics['waiting_time']:.2f}"
        )

        print(
            f"Average turnaround time: "
            f"{metrics['turnaround_time']:.2f}"
        )

        print(
            f"Average response time: "
            f"{metrics['response_time']:.2f}"
        )


if __name__ == "__main__":
    main()