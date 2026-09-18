class MinHeap:

    def __init__(self, key_function):
        self.heap = []
        self.key_function = key_function

    def size(self):
        return len(self.heap)

    def is_empty(self):
        return len(self.heap) == 0

    def peek(self):
        if self.is_empty():
            return None

        return self.heap[0]

    def push(self, item):
        self.heap.append(item)
        self._heapify_up(len(self.heap) - 1)

    def pop(self):
        if self.is_empty():
            return None

        if len(self.heap) == 1:
            return self.heap.pop()

        minimum = self.heap[0]

        self.heap[0] = self.heap.pop()

        self._heapify_down(0)

        return minimum

    def clear(self):
        self.heap.clear()

    def _heapify_up(self, index):

        while index > 0:

            parent = (index - 1) // 2

            if self._less(self.heap[index], self.heap[parent]):
                self.heap[index], self.heap[parent] = (
                    self.heap[parent],
                    self.heap[index]
                )

                index = parent

            else:
                break

    def _heapify_down(self, index):

        size = len(self.heap)

        while True:

            left = 2 * index + 1
            right = 2 * index + 2

            smallest = index

            if (
                left < size
                and self._less(self.heap[left], self.heap[smallest])
            ):
                smallest = left

            if (
                right < size
                and self._less(self.heap[right], self.heap[smallest])
            ):
                smallest = right

            if smallest == index:
                break

            self.heap[index], self.heap[smallest] = (
                self.heap[smallest],
                self.heap[index]
            )

            index = smallest

    def _less(self, a, b):
        return self.key_function(a) < self.key_function(b)

    def heapify(self, items):

        self.heap = list(items)

        if len(self.heap) <= 1:
            return

        start = (len(self.heap) // 2) - 1

        for index in range(start, -1, -1):
            self._heapify_down(index)

    def to_list(self):
        return list(self.heap)