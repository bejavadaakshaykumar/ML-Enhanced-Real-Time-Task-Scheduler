class VirtualClock:
    def __init__(self, start_time=0):
        self.time = start_time

    def now(self):
        return self.time

    def tick(self, amount=1):
        if amount < 0:
            raise ValueError("Clock cannot move backwards")

        self.time += amount

    def advance_to(self, target_time):
        if target_time < self.time:
            raise ValueError("Cannot move clock backwards")

        self.time = target_time

    def reset(self, start_time=0):
        self.time = start_time