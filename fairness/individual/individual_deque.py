import heapq
from collections import deque


class IndividualDeque(object):
    """A deque for maintaining individual fairness information, required by individual fairness notions

    Attributes:
        window: (Optional) The window size to consider
        max_n: The maximum number of nearest neighbours to consider for nearest neighbours computations
        deque: The deque storing all comparisons for the individual
        n_smallest: The nearest neighbours of the individual based on the distance
    """
    def __init__(self, max_n, window=None):
        self.window = window
        self.max_n = max_n
        self.deque = deque(maxlen=self.window)
        self.n_smallest = []
        self._len_deque = 0
        self._min_n = 0

    def __len__(self):
        return self._len_deque

    def append(self, element):
        """Add an element to the deque"""
        recompute_n_smallest = False
        # Deque is full, element gets removed
        if self._len_deque == self.window and self.deque[0] in self.n_smallest:
            self.n_smallest.remove(self.deque[0])
            recompute_n_smallest = True
        else:
            self._len_deque += 1
        self.deque.append(element)

        if self._len_deque < self.max_n:
            recompute_n_smallest = True
            self._min_n = self._len_deque
        else:
            self._min_n = self._len_deque
            for idx, el in enumerate(self.n_smallest):
                if element[0] < el[0]:
                    self.n_smallest.insert(idx, element)
                    if len(self.n_smallest) > self.max_n:
                        self.n_smallest = self.n_smallest[:self.max_n]
                    recompute_n_smallest = False
                    break
        if recompute_n_smallest or len(self.n_smallest) < self.max_n:
            self.n_smallest = heapq.nsmallest(self._min_n, self.deque)

    def popleft(self):
        """Remove the oldest instance of a comparison from the deque"""
        first = self.deque.popleft()
        if first in self.n_smallest:
            self.n_smallest = heapq.nsmallest(self._min_n, self.deque)
