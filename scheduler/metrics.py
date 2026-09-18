# ============================================================
# SCHEDULER METRICS
# ============================================================


def calculate_task_metrics(tasks):
    """
    Calculate performance metrics for every completed task.
    """

    results = []

    for task in tasks:

        # Ignore tasks that were not completed
        if task.completion_time is None:
            continue

        waiting_time = (
            task.completion_time
            - task.arrival_time
            - task.execution_time
        )

        response_time = (
            task.start_time
            - task.arrival_time
        )

        turnaround_time = (
            task.completion_time
            - task.arrival_time
        )

        tardiness = max(
            0,
            task.completion_time - task.deadline
        )

        deadline_missed = (
            task.completion_time > task.deadline
        )

        results.append(
            {
                "task_id": task.task_id,

                "arrival_time": task.arrival_time,

                "execution_time": task.execution_time,

                "deadline": task.deadline,

                "priority": task.priority,

                "start_time": task.start_time,

                "completion_time": task.completion_time,

                "waiting_time": waiting_time,

                "response_time": response_time,

                "turnaround_time": turnaround_time,

                "tardiness": tardiness,

                "deadline_missed": deadline_missed
            }
        )

    return results


# ============================================================
# SUMMARY METRICS
# ============================================================

def calculate_summary(
    tasks,
    total_time,
    context_switches=0,
    preemptions=0
):
    """
    Calculate overall scheduler performance.

    Parameters
    ----------
    tasks:
        List of completed Task objects.

    total_time:
        Total simulation time.

    context_switches:
        Number of context switches.

    preemptions:
        Number of preemptions.

    Returns
    -------
    dict
        Scheduler performance summary.
    """

    metrics = calculate_task_metrics(tasks)

    # --------------------------------------------------------
    # No completed tasks
    # --------------------------------------------------------

    if not metrics:

        return {
            "average_waiting_time": 0,

            "average_response_time": 0,

            "average_turnaround_time": 0,

            "average_tardiness": 0,

            "deadline_misses": 0,

            "deadline_miss_rate": 0,

            "cpu_utilization": 0,

            "throughput": 0,

            "context_switches": context_switches,

            "preemptions": preemptions
        }

    # --------------------------------------------------------
    # Number of completed tasks
    # --------------------------------------------------------

    n = len(metrics)

    # --------------------------------------------------------
    # Average waiting time
    # --------------------------------------------------------

    average_waiting = (
        sum(
            item["waiting_time"]
            for item in metrics
        ) / n
    )

    # --------------------------------------------------------
    # Average response time
    # --------------------------------------------------------

    average_response = (
        sum(
            item["response_time"]
            for item in metrics
        ) / n
    )

    # --------------------------------------------------------
    # Average turnaround time
    # --------------------------------------------------------

    average_turnaround = (
        sum(
            item["turnaround_time"]
            for item in metrics
        ) / n
    )

    # --------------------------------------------------------
    # Average tardiness
    # --------------------------------------------------------

    average_tardiness = (
        sum(
            item["tardiness"]
            for item in metrics
        ) / n
    )

    # --------------------------------------------------------
    # Deadline misses
    # --------------------------------------------------------

    deadline_misses = sum(
        item["deadline_missed"]
        for item in metrics
    )

    # --------------------------------------------------------
    # Total execution time
    # --------------------------------------------------------

    total_execution = sum(
        item["execution_time"]
        for item in metrics
    )

    # --------------------------------------------------------
    # CPU utilization
    # --------------------------------------------------------

    cpu_utilization = 0

    if total_time > 0:

        cpu_utilization = (
            total_execution / total_time
        ) * 100

    # --------------------------------------------------------
    # Throughput
    # --------------------------------------------------------

    throughput = 0

    if total_time > 0:

        throughput = (
            n / total_time
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    return {

        "average_waiting_time":
            average_waiting,

        "average_response_time":
            average_response,

        "average_turnaround_time":
            average_turnaround,

        "average_tardiness":
            average_tardiness,

        "deadline_misses":
            deadline_misses,

        "deadline_miss_rate":
            (
                deadline_misses / n
            ) * 100,

        "cpu_utilization":
            cpu_utilization,

        "throughput":
            throughput,

        "context_switches":
            context_switches,

        "preemptions":
            preemptions
    }