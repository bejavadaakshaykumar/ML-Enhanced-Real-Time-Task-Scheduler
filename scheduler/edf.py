from .min_heap import MinHeap


class EDFScheduler:
    """
    Earliest Deadline First (EDF) Scheduler.

    Selects the task with the earliest (smallest) deadline
    from the ready queue. This is the optimal preemptive
    uniprocessor scheduling algorithm for real-time systems
    when all tasks have deadlines.

    Tie-breaking order:
        1. deadline     — earliest deadline first
        2. priority     — lower priority value first
        3. arrival_time — earlier arrival first
        4. task_id      — deterministic ordering
    """

    def __init__(self, preemptive=False):
        self.preemptive = preemptive

        self.ready_queue = MinHeap(
            key_function=self.get_key
        )

    def get_key(self, task):
        return (
            task.deadline,
            task.priority,
            task.arrival_time,
            task.task_id
        )

    def add_task(self, task):
        self.ready_queue.push(task)

    def add_tasks(self, tasks):
        for task in tasks:
            self.add_task(task)

    def select_task(self):
        return self.ready_queue.pop()

    def peek_task(self):
        return self.ready_queue.peek()

    def is_empty(self):
        return self.ready_queue.is_empty()

    def size(self):
        return self.ready_queue.size()

    def clear(self):
        self.ready_queue.clear()