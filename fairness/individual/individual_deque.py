import heapq
from collections import deque


class IndividualDeque(object):
    """A deque for maintaining individual fairness information, required by individual fairness notions"""
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
        first = self.deque.popleft()
        if first in self.n_smallest:
            self.n_smallest = heapq.nsmallest(self._min_n, self.deque)

#          17416315 function calls (16912449 primitive calls) in 21.403 seconds
#
#    Ordered by: cumulative time
#
#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#         1    0.017    0.017   21.936   21.936 main_pcn_core.py:292(train_fair)
#       984    0.021    0.000   21.696    0.022 fairness_framework.py:138(step)
#      2952    0.003    0.000   19.669    0.007 fairness_framework.py:92(get_individual_notion)
#      2952    0.050    0.000   19.665    0.007 individual_fairness.py:137(get_notion)
#       984    1.272    0.001   12.535    0.013 individual_fairness.py:142(individual_fairness)
#       984    0.116    0.000    8.686    0.009 individual_fairness.py:178(<listcomp>)
#    699187    0.505    0.000    8.681    0.000 fairness_framework.py:111(<lambda>)
#    483636    0.340    0.000    8.570    0.000 individual_fairness.py:59(_pool_individual_fairness)
#    699187    5.419    0.000    8.176    0.000 __init__.py:291(H_OM_distance)
#       984    0.031    0.000    6.303    0.006 individual_fairness.py:402(consistency_score_complement_inn)
#    483636    0.149    0.000    6.028    0.000 __init__.py:274(similarity_metric)
#    483834    1.927    0.000    5.201    0.000 optimised_swinn.py:226(_search)
#       983    0.002    0.000    4.250    0.004 optimised_swinn.py:145(get_nn_for_all)
#       926    0.201    0.000    4.234    0.005 optimised_swinn.py:161(<listcomp>)
#    215551    0.081    0.000    2.884    0.000 base.py:31(__call__)
#   1401317    2.759    0.000    2.759    0.000 {built-in method builtins.sum}
#    483636    2.202    0.000    2.202    0.000 individual_fairness.py:43(hellinger)
#    967272    1.843    0.000    2.171    0.000 individual_deque.py:16(append)

#          14993981 function calls (14973751 primitive calls) in 20.330 seconds
#
#    Ordered by: cumulative time
#
#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#         1    0.015    0.015   20.989   20.989 main_pcn_core.py:292(train_fair)
#       984    0.019    0.000   20.759    0.021 fairness_framework.py:138(step)
#      2952    0.003    0.000   18.825    0.006 fairness_framework.py:92(get_individual_notion)
#      2952    0.049    0.000   18.822    0.006 individual_fairness.py:137(get_notion)
#       984    1.268    0.001   11.781    0.012 individual_fairness.py:142(individual_fairness)
#       984    0.119    0.000    8.488    0.009 individual_fairness.py:178(<listcomp>)
#    483636    0.357    0.000    8.369    0.000 individual_fairness.py:59(_pool_individual_fairness)
#    699187    0.484    0.000    8.367    0.000 fairness_framework.py:111(<lambda>)
#    699187    5.238    0.000    7.884    0.000 __init__.py:291(H_OM_distance)
#       984    0.029    0.000    6.255    0.006 individual_fairness.py:402(consistency_score_complement_inn)
#    483636    0.140    0.000    5.801    0.000 __init__.py:274(similarity_metric)
#    483834    1.866    0.000    5.032    0.000 optimised_swinn.py:226(_search)
#       983    0.002    0.000    4.124    0.004 optimised_swinn.py:145(get_nn_for_all)
#       926    0.196    0.000    4.109    0.004 optimised_swinn.py:161(<listcomp>)
#    215551    0.080    0.000    2.786    0.000 base.py:31(__call__)
#   1401317    2.647    0.000    2.647    0.000 {built-in method builtins.sum}
#    483636    2.212    0.000    2.212    0.000 individual_fairness.py:43(hellinger)
#    967272    1.530    0.000    1.707    0.000 individual_deque.py:17(append)
