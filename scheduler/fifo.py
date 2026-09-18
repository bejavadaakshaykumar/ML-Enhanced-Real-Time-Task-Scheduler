from collections import deque


class FIFOScheduler:

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

    def is_empty(self):
        return len(self.queue) == 0

    def size(self):
        return len(self.queue)

    def clear(self):
        self.queue.clear()