from collections import deque


class FIFOScheduler:
    """
    First-In, First-Out (FIFO) Scheduler.

    Tasks are executed in the order they arrive in the ready queue.
    This is the simplest scheduling algorithm — no priority or
    deadline information is used.

    Used as a baseline comparison against EDF and ML-EDF.
    """

    def __init__(self):
        self.queue = deque()

    def add_task(self, task):
        self.queue.append(task)

    def add_tasks(self, tasks):
        for task in tasks:
            self.add_task(task)

    def select_task(self):
        if not self.queue:
            return None

        return self.queue.popleft()

    def peek_task(self):
        """Return the next task without removing it."""
        if not self.queue:
            return None

        return self.queue[0]

    def is_empty(self):
        return len(self.queue) == 0

    def size(self):
        return len(self.queue)

    def clear(self):
        self.queue.clear()