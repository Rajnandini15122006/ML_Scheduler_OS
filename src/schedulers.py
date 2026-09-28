import pandas as pd


def fcfs_scheduler(dataset):
    """
    First Come First Serve CPU scheduling.

    FCFS is:
    - Non-preemptive
    - Processes are executed in order of arrival
    """

    # Sort processes by arrival time
    processes = dataset.sort_values(
        by=["arrival_time", "process_id"]
    ).copy()

    current_time = 0
    results = []

    for _, process in processes.iterrows():

        process_id = process["process_id"]
        arrival_time = process["arrival_time"]
        cpu_burst = process["cpu_burst"]

        # If CPU is idle, jump to the process arrival time
        if current_time < arrival_time:
            current_time = arrival_time

        # Process starts executing
        start_time = current_time

        # Waiting time
        waiting_time = start_time - arrival_time

        # Process finishes after CPU burst
        completion_time = start_time + cpu_burst

        # Turnaround time
        turnaround_time = completion_time - arrival_time

        # Response time
        response_time = start_time - arrival_time

        # Update CPU clock
        current_time = completion_time

        results.append({
            "process_id": process_id,
            "arrival_time": arrival_time,
            "cpu_burst": cpu_burst,
            "start_time": start_time,
            "completion_time": completion_time,
            "waiting_time": waiting_time,
            "turnaround_time": turnaround_time,
            "response_time": response_time,
            "workload_type": process["workload_type"]
        })

    return pd.DataFrame(results)



def round_robin_scheduler(dataset, quantum=10):
    """
    Round Robin CPU scheduling.

    Parameters:
        dataset: workload DataFrame
        quantum: maximum CPU time given to a process
                  during one turn.
    """

    # Sort by arrival time
    processes = dataset.sort_values(
        by=["arrival_time", "process_id"]
    ).copy()

    # Convert required columns into dictionaries
    process_list = processes.to_dict("records")

    ready_queue = []

    remaining_burst = {
        process["process_id"]: process["cpu_burst"]
        for process in process_list
    }

    start_times = {}
    completion_times = {}

    current_time = 0
    next_process_index = 0

    # Continue until every process has completed
    while next_process_index < len(process_list) or ready_queue:

        # Add newly arrived processes
        while (
            next_process_index < len(process_list)
            and process_list[next_process_index]["arrival_time"] <= current_time
        ):
            ready_queue.append(process_list[next_process_index])
            next_process_index += 1

        # If no process is ready, jump to next arrival
        if not ready_queue:

            current_time = process_list[next_process_index]["arrival_time"]

            continue

        # Select process from front of queue
        process = ready_queue.pop(0)

        process_id = process["process_id"]

        # Record first time process gets CPU
        if process_id not in start_times:
            start_times[process_id] = current_time

        # Execute for one quantum or until completion
        execution_time = min(
            quantum,
            remaining_burst[process_id]
        )

        current_time += execution_time

        remaining_burst[process_id] -= execution_time

        # Add processes that arrived during execution
        while (
            next_process_index < len(process_list)
            and process_list[next_process_index]["arrival_time"] <= current_time
        ):
            ready_queue.append(process_list[next_process_index])
            next_process_index += 1

        # Process completed
        if remaining_burst[process_id] == 0:

            completion_times[process_id] = current_time

        else:

            # Process still needs CPU time
            ready_queue.append(process)

    # Build result table
    results = []

    for process in process_list:

        process_id = process["process_id"]

        arrival_time = process["arrival_time"]

        cpu_burst = process["cpu_burst"]

        start_time = start_times[process_id]

        completion_time = completion_times[process_id]

        turnaround_time = completion_time - arrival_time

        waiting_time = turnaround_time - cpu_burst

        response_time = start_time - arrival_time

        results.append({
            "process_id": process_id,
            "arrival_time": arrival_time,
            "cpu_burst": cpu_burst,
            "start_time": start_time,
            "completion_time": completion_time,
            "waiting_time": waiting_time,
            "turnaround_time": turnaround_time,
            "response_time": response_time,
            "workload_type": process["workload_type"]
        })

    return pd.DataFrame(results)


def priority_scheduler(dataset):
    """
    Non-preemptive Priority Scheduling.

    Higher numerical priority = higher scheduling priority.
    """

    processes = dataset.copy()

    # Track processes that have not yet executed
    remaining_processes = processes.to_dict("records")

    current_time = 0
    results = []

    while remaining_processes:

        # Find processes that have already arrived
        available = [
            process
            for process in remaining_processes
            if process["arrival_time"] <= current_time
        ]

        # If no process has arrived yet, jump to the next arrival
        if not available:

            current_time = min(
                process["arrival_time"]
                for process in remaining_processes
            )

            continue

        # Select:
        # 1. Highest priority
        # 2. Earlier arrival time if priorities are equal
        # 3. Smaller process ID if still tied
        selected = max(
            available,
            key=lambda process: (
                process["priority"],
                -process["arrival_time"],
                -process["process_id"]
            )
        )

        process_id = selected["process_id"]
        arrival_time = selected["arrival_time"]
        cpu_burst = selected["cpu_burst"]

        # Process starts
        start_time = current_time

        # Calculate waiting time
        waiting_time = start_time - arrival_time

        # Execute entire CPU burst
        completion_time = start_time + cpu_burst

        # Calculate metrics
        turnaround_time = completion_time - arrival_time
        response_time = start_time - arrival_time

        # Move CPU clock forward
        current_time = completion_time

        results.append({
            "process_id": process_id,
            "arrival_time": arrival_time,
            "cpu_burst": cpu_burst,
            "priority": selected["priority"],
            "start_time": start_time,
            "completion_time": completion_time,
            "waiting_time": waiting_time,
            "turnaround_time": turnaround_time,
            "response_time": response_time,
            "workload_type": selected["workload_type"]
        })

        # Remove completed process
        remaining_processes.remove(selected)

    return pd.DataFrame(results)

if __name__ == "__main__":

    # Load workload dataset
    dataset = pd.read_csv("data/workloads.csv")

    # -----------------------------------------------------
    # FCFS
    # -----------------------------------------------------

    fcfs_results = fcfs_scheduler(dataset)

    print("=" * 60)
    print("FCFS SCHEDULER")
    print("=" * 60)

    print(
        "Average waiting time:",
        round(fcfs_results["waiting_time"].mean(), 2)
    )

    print(
        "Average turnaround time:",
        round(fcfs_results["turnaround_time"].mean(), 2)
    )

    print(
        "Average response time:",
        round(fcfs_results["response_time"].mean(), 2)
    )

    # -----------------------------------------------------
    # Round Robin
    # -----------------------------------------------------

    rr_results = round_robin_scheduler(
        dataset,
        quantum=10
    )

    print()
    print("=" * 60)
    print("ROUND ROBIN SCHEDULER")
    print("=" * 60)

    print(
        "Time quantum: 10"
    )

    print(
        "Average waiting time:",
        round(rr_results["waiting_time"].mean(), 2)
    )

    print(
        "Average turnaround time:",
        round(rr_results["turnaround_time"].mean(), 2)
    )

    print(
        "Average response time:",
        round(rr_results["response_time"].mean(), 2)
    )

        # -----------------------------------------------------
    # Priority Scheduling
    # -----------------------------------------------------

    priority_results = priority_scheduler(dataset)

    print()
    print("=" * 60)
    print("PRIORITY SCHEDULER")
    print("=" * 60)

    print(
        "Priority rule: Higher number = higher priority"
    )

    print(
        "Average waiting time:",
        round(priority_results["waiting_time"].mean(), 2)
    )

    print(
        "Average turnaround time:",
        round(priority_results["turnaround_time"].mean(), 2)
    )

    print(
        "Average response time:",
        round(priority_results["response_time"].mean(), 2)
    )