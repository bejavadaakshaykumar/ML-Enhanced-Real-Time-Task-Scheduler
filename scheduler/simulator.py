from dataclasses import dataclass

from .virtual_clock import VirtualClock


@dataclass
class TraceEvent:
    start_time: int
    end_time: int
    task_id: int | None
    event: str

    @property
    def duration(self):
        return self.end_time - self.start_time


class SchedulerSimulator:

    def __init__(
        self,
        scheduler,
        tasks,
        context_switch_cost=0
    ):
        self.scheduler = scheduler

        self.tasks = sorted(
            tasks,
            key=lambda task: (
                task.arrival_time,
                task.task_id
            )
        )

        self.context_switch_cost = (
            context_switch_cost
        )

        self.clock = VirtualClock()

        self.trace = []

        self.context_switches = 0

        self.preemptions = 0

        self.current_task = None

        self.previous_task_id = None

        self.next_arrival_index = 0

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------

    def reset(self):

        self.clock.reset()

        self.trace = []

        self.context_switches = 0

        self.preemptions = 0

        self.current_task = None

        self.previous_task_id = None

        self.next_arrival_index = 0

        self.scheduler.clear()

        for task in self.tasks:
            task.reset()

    # --------------------------------------------------
    # UPDATE SCHEDULER TIME
    # --------------------------------------------------

    def update_scheduler_time(self):

        """
        Update scheduler time when supported.

        This is especially important for ML-EDF,
        because its priority calculation uses:

            deadline
            predicted execution time
            predicted miss probability
            current time
            predicted slack
        """

        if hasattr(
            self.scheduler,
            "update_time"
        ):

            self.scheduler.update_time(
                self.clock.now()
            )

    # --------------------------------------------------
    # ADD ARRIVED TASKS
    # --------------------------------------------------

    def add_arrived_tasks(self):

        current_time = self.clock.now()

        while (
            self.next_arrival_index
            < len(self.tasks)
            and
            self.tasks[
                self.next_arrival_index
            ].arrival_time
            <= current_time
        ):

            task = self.tasks[
                self.next_arrival_index
            ]

            self.scheduler.add_task(
                task
            )

            self.next_arrival_index += 1

    # --------------------------------------------------
    # TRACE
    # --------------------------------------------------

    def record_event(
        self,
        start_time,
        end_time,
        task_id,
        event
    ):

        if end_time <= start_time:
            return

        self.trace.append(
            TraceEvent(
                start_time=start_time,
                end_time=end_time,
                task_id=task_id,
                event=event
            )
        )

    # --------------------------------------------------
    # CONTEXT SWITCH
    # --------------------------------------------------

    def perform_context_switch(
        self,
        new_task
    ):

        if new_task is None:
            return

        if self.previous_task_id is None:
            return

        if (
            self.previous_task_id
            == new_task.task_id
        ):
            return

        self.context_switches += 1

        start_time = self.clock.now()

        end_time = (
            start_time
            + self.context_switch_cost
        )

        self.record_event(
            start_time,
            end_time,
            None,
            "CONTEXT_SWITCH"
        )

        self.clock.tick(
            self.context_switch_cost
        )

        self.update_scheduler_time()

    # --------------------------------------------------
    # SELECT NEXT TASK
    # --------------------------------------------------

    def select_next_task(self):

        self.update_scheduler_time()

        task = self.scheduler.select_task()

        if task is None:
            return None

        self.perform_context_switch(
            task
        )

        self.current_task = task

        return task

    # --------------------------------------------------
    # EXECUTE ONE UNIT
    # --------------------------------------------------

    def execute_one_unit(
        self,
        task
    ):

        current_time = self.clock.now()

        if task.start_time is None:

            task.start_time = (
                current_time
            )

        task.remaining_time -= 1

        self.record_event(
            current_time,
            current_time + 1,
            task.task_id,
            "RUN"
        )

        self.clock.tick(1)

        self.update_scheduler_time()

        if task.remaining_time == 0:

            task.completion_time = (
                self.clock.now()
            )

            self.previous_task_id = (
                task.task_id
            )

            self.current_task = None

            return True

        return False

    # --------------------------------------------------
    # PREEMPTIVE SIMULATION
    # --------------------------------------------------

    def run_preemptive(self):

        self.reset()

        while True:

            # Update ML scheduler time
            self.update_scheduler_time()

            # Add tasks that have arrived
            self.add_arrived_tasks()

            # ------------------------------------------
            # No current task
            # ------------------------------------------

            if self.current_task is None:

                if self.scheduler.is_empty():

                    # Future task exists
                    if (
                        self.next_arrival_index
                        < len(self.tasks)
                    ):

                        next_arrival = (
                            self.tasks[
                                self.next_arrival_index
                            ].arrival_time
                        )

                        current_time = (
                            self.clock.now()
                        )

                        if (
                            next_arrival
                            > current_time
                        ):

                            self.record_event(
                                current_time,
                                next_arrival,
                                None,
                                "IDLE"
                            )

                            self.clock.advance_to(
                                next_arrival
                            )

                            self.update_scheduler_time()

                            continue

                    # No tasks remaining
                    else:
                        break

                self.select_next_task()

            # ------------------------------------------
            # Check preemption
            # ------------------------------------------

            if (
                self.current_task is not None
                and
                not self.scheduler.is_empty()
            ):

                current_task = (
                    self.current_task
                )

                next_task = (
                    self.scheduler.peek_task()
                )

                if next_task is not None:

                    # ML-EDF
                    #
                    # If scheduler provides get_key(),
                    # use its scheduling policy.
                    #
                    # This allows standard EDF and
                    # ML-EDF to work with the same
                    # simulator.

                    if hasattr(
                        self.scheduler,
                        "get_key"
                    ):

                        current_key = (
                            self.scheduler.get_key(
                                current_task
                            )
                        )

                        next_key = (
                            self.scheduler.get_key(
                                next_task
                            )
                        )

                        if next_key < current_key:

                            self.preemptions += 1

                            self.scheduler.add_task(
                                self.current_task
                            )

                            self.current_task = None

                            self.select_next_task()

            # ------------------------------------------
            # Execute one unit
            # ------------------------------------------

            if self.current_task is not None:

                completed = (
                    self.execute_one_unit(
                        self.current_task
                    )
                )

                if completed:

                    self.current_task = None

        return self.get_results()

    # --------------------------------------------------
    # NON-PREEMPTIVE SIMULATION
    # --------------------------------------------------

    def run_non_preemptive(self):

        self.reset()

        while True:

            self.update_scheduler_time()

            self.add_arrived_tasks()

            # ------------------------------------------
            # CPU idle
            # ------------------------------------------

            if self.scheduler.is_empty():

                if (
                    self.next_arrival_index
                    < len(self.tasks)
                ):

                    next_arrival = (
                        self.tasks[
                            self.next_arrival_index
                        ].arrival_time
                    )

                    current_time = (
                        self.clock.now()
                    )

                    if (
                        next_arrival
                        > current_time
                    ):

                        self.record_event(
                            current_time,
                            next_arrival,
                            None,
                            "IDLE"
                        )

                        self.clock.advance_to(
                            next_arrival
                        )

                        self.update_scheduler_time()

                        continue

                else:
                    break

            # ------------------------------------------
            # Select task
            # ------------------------------------------

            self.update_scheduler_time()

            task = (
                self.scheduler.select_task()
            )

            if task is None:
                continue

            self.perform_context_switch(
                task
            )

            self.current_task = task

            if task.start_time is None:

                task.start_time = (
                    self.clock.now()
                )

            start_time = (
                self.clock.now()
            )

            execution_time = (
                task.remaining_time
            )

            task.remaining_time = 0

            self.clock.tick(
                execution_time
            )

            self.update_scheduler_time()

            task.completion_time = (
                self.clock.now()
            )

            self.record_event(
                start_time,
                self.clock.now(),
                task.task_id,
                "RUN"
            )

            self.previous_task_id = (
                task.task_id
            )

            self.current_task = None

        return self.get_results()

    # --------------------------------------------------
    # MAIN RUN
    # --------------------------------------------------

    def run(self):

        if getattr(
            self.scheduler,
            "preemptive",
            False
        ):

            return self.run_preemptive()

        return self.run_non_preemptive()

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    def get_results(self):

        return {
            "tasks": self.tasks,
            "trace": self.trace,
            "context_switches":
                self.context_switches,
            "preemptions":
                self.preemptions,
            "total_time":
                self.clock.now()
        }

    # --------------------------------------------------
    # PRINT TRACE
    # --------------------------------------------------

    def print_trace(self):

        print()

        print(
            "========== EXECUTION TRACE =========="
        )

        print()

        for event in self.trace:

            if event.task_id is None:

                print(
                    f"[{event.start_time:>3} -> "
                    f"{event.end_time:>3}] "
                    f"{event.event}"
                )

            else:

                print(
                    f"[{event.start_time:>3} -> "
                    f"{event.end_time:>3}] "
                    f"T{event.task_id} "
                    f"{event.event}"
                )

        print()

        print(
            "====================================="
        )

        print(
            f"Context Switches : "
            f"{self.context_switches}"
        )

        print(
            f"Preemptions      : "
            f"{self.preemptions}"
        )

        print(
            f"Total Time       : "
            f"{self.clock.now()}"
        )