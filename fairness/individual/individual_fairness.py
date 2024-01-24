from __future__ import annotations

import heapq
import math
import time
import typing
from collections import deque
from enum import Enum
from itertools import groupby
from multiprocessing import Pool
from pstats import SortKey

import numpy as np
import scipy.spatial.distance
from river import neighbors
from river.neighbors.ann.nn_vertex import Vertex
from river.neighbors.base import FunctionWrapper, DistanceFunc

from fairness.history import History, DiscountedHistory
from fairness.individual import IndividualNotion, IndividualFairnessBase
from scenario import CombinedState


def dict_to_array(d):
    a = []
    for v in d.values():
        if isinstance(v, Enum):
            a.append(v.value)
        else:
            a.append(v)
    return np.array(a, dtype=float)


def key_state(s):
    return str(dict_to_array(s))


SQRT2 = np.sqrt(2)


def hellinger(p, q, num_actions):
    # Slightly faster computation time
    if num_actions == 2:
        sqrt_0_1_2 = np.sqrt(p[0]) - np.sqrt(q[0])
        sqrt_1_1_2 = np.sqrt(p[1]) - np.sqrt(q[1])
        total = (sqrt_0_1_2 * sqrt_0_1_2) + (sqrt_1_1_2 * sqrt_1_1_2)
        h_dist = np.sqrt(total) / SQRT2
    else:
        total = 0
        for i in range(num_actions):
            sqrt1_2 = np.sqrt(p[i]) - np.sqrt(q[i])
            total += (sqrt1_2 * sqrt1_2)
        h_dist = np.sqrt(total) / SQRT2
    return h_dist


def _pool_individual_fairness(args):
    i, j, state_i, state_j, score_i, score_j, similarity_metric, alpha, distance_metric, num_actions = args
    d = similarity_metric(state_i, state_j, alpha=alpha, distance=distance_metric)
    D = hellinger(score_i, score_j, num_actions=num_actions)
    # print(score_i, score_j, D, d, d - D)
    # i, j, Fair, difference, D, d
    return i, j, D <= d, d - D, D, d


def _pool_weakly_meritocratic(args):
    i, state, action, true_action, score, reward, actions, alpha, probs = args
    q_action = score[action]
    is_fair = True
    max_diff = 0
    for a in actions:
        # a = a'
        if a == action:
            continue
        q_a = score[a.value]
        if q_action > (q_a + alpha):
            # Fair
            if probs[action] >= probs[a.value]:
                continue
            # Unfair
            else:
                is_fair = False
                max_diff = q_action - (q_a + alpha)

    return i, is_fair, max_diff


def _pool_weakly_meritocratic_knn(args):
    i, states, actions, q_values, nearest_neighours = args
    is_fair = True
    max_diff = 0

    # TODO:
    for cluster in ...:
        pass

    return i, is_fair, max_diff


class IndividualFairness(IndividualFairnessBase):
    """A collection of fairness notions w.r.t. individuals.

        Attributes:
            actions: A list of enumerations, representing the actions to check fairness for.
        """

    def __init__(self, actions, ind_distance_metrics, csc_distance_metrics, inn_sensitive_features=None, seed=None,
                 steps=None):
        # Super call
        super(IndividualFairness, self).__init__(actions)
        # Mapping from enumeration to fairness method
        self._map = {
            IndividualNotion.IndividualFairness: self.individual_fairness,
            IndividualNotion.WeaklyMeritocratic: self.weakly_meritocratic,
            IndividualNotion.ConsistencyScoreComplement: self.consistency_score_complement,
            #
            IndividualNotion.ConsistencyScoreComplement_INN: self.consistency_score_complement_inn,
        }
        self.ind_distance_metrics = ind_distance_metrics
        self.csc_distance_metrics = csc_distance_metrics
        all_metrics = set(ind_distance_metrics).union(csc_distance_metrics)

        # Don't recalculate individuals who have been compared already, they haven't changed
        self._individual_comparisons = {d: {} for d in all_metrics}
        self._individual_last_window = {d: None for d in all_metrics}
        self._individual_total = {d: 0.0 for d in all_metrics}
        self._last_ind = {d: [] for d in all_metrics}
        #
        # self._csc_nbrs = {d: None for d in self.csc_distance_metrics}
        self._neighbours = None
        self.inn_sensitive_features = inn_sensitive_features
        self.seed = seed
        self.steps = steps

    def get_notion(self, notion: IndividualNotion, history: History, threshold=None,
                   similarity_metric=None, alpha=None, distance_metric=None):
        # noinspection PyArgumentList
        return self._map[notion](history, threshold, similarity_metric, alpha, distance_metric)

    def individual_fairness(self, history: History, threshold=None, similarity_metric=None, alpha=1.0,
                            distance_metric=("braycurtis", "braycurtis")):
        """Let i and j be two individuals represented by their attributes values vectors v_i and v_j.
        Let d(v_i,v_j) represent the similarity distance between individuals i and j.
        Let D be a distance metric between probability distributions M(v_i) and M(v_j).
        Fairness through awareness is achieved iff, for any pair of individuals i and j

        D(M(v_i), M(v_j)) ≤ d(v_i, v_j)
        """
        distance_metric, metric = distance_metric
        states, actions, true_actions, scores, rewards = history.get_history()
        n = len(states)
        num_actions = len(self.actions)
        lowest_n = 0
        exact = True

        with_window = history.window is not None
        is_discounted = isinstance(history, DiscountedHistory)

        # Keep track of the differences to discard once the window passes
        if (with_window or is_discounted) and self._individual_last_window.get(distance_metric) is None:
            self._individual_last_window[distance_metric] = deque(maxlen=history.window)

        # Can only compare as many individuals as present in self._individual_last_window + 1,
        #   others are dropped due to threshold
        if is_discounted:
            lowest_n = n - (len(self._individual_last_window[distance_metric]) + 1)

        # If given n interactions/individuals, under the assumption that all interactions until n have been compared in
        #   the previous timestep, only individual n should be compared to 0 until n - 1
        map_i_j = []
        i = n - 1
        for j in range((n - 1) - 1, lowest_n - 1, -1):  # Run range(lowest_n, n - 1) backwards for is_discounted
            map_i_j.append((i, j, states[i], states[j], scores[i], scores[j],
                            similarity_metric, alpha, metric, num_actions))
        unsatisfied_pairs = 0
        results = [_pool_individual_fairness(ij) for ij in map_i_j]

        if with_window or is_discounted:
            # Store data at individual with lowest index: comparison gets removed with individual when moving out
            #   of the sliding window. Remove earliest individual from deque if outside window.
            if len(self._individual_last_window[distance_metric]) == history.window:
                last, _ = self._individual_last_window[distance_metric][0]
                self._individual_total[distance_metric] -= np.nansum(last)
            self._individual_last_window[distance_metric].append(
                ([], deque(maxlen=history.window)))  # diffs, deque/heap

        # Windowed + full history
        if not is_discounted:
            if self._individual_total.get(distance_metric) is None:
                self._individual_total[distance_metric] = 0.0
            total = self._individual_total[distance_metric]
            total_comparisons = n * (n - 1) // 2

            for i, j, fair, diff, D, d in results:
                if not np.isnan(diff):
                    total += diff
                if with_window:
                    # j is a previously encountered individual, i is current
                    self._individual_last_window[distance_metric][j][0].append(diff)
                    #
                    if len(self._individual_last_window[distance_metric][j][1]) == history.window:
                        self._individual_last_window[distance_metric][j][1].popleft()
                    self._individual_last_window[distance_metric][j][1].append((d, i, diff, actions[i], actions[j]))
                    self._individual_last_window[distance_metric][i][1].append((d, j, diff, actions[j], actions[i]))

                # Exact fair
                if fair:
                    continue
                exact = False
                # Not approx fair either
                if not threshold or diff < -threshold:
                    unsatisfied_pairs += 1
        # Discounted
        else:
            # n is shifted based on if _individual_last_window has been reduced in previous iterations
            shifted_n = n - len(self._individual_last_window[distance_metric])

            for i, j, fair, diff, D, d in results:
                # j is a previously encountered individual, i is current
                shifted_j = j - shifted_n
                self._individual_last_window[distance_metric][shifted_j][0].append(diff)
                #
                if len(self._individual_last_window[distance_metric][shifted_j][1]) == history.window:
                    self._individual_last_window[distance_metric][shifted_j][1].popleft()
                self._individual_last_window[distance_metric][shifted_j][1].append(
                    (d, i, diff, actions[i], actions[shifted_j]))
                self._individual_last_window[distance_metric][i][1].append(
                    (d, shifted_j, diff, actions[shifted_j], actions[i]))
                # Exact fair
                if fair:
                    continue
                exact = False
                # Not approx fair either
                if not threshold or diff < -threshold:
                    unsatisfied_pairs += 1

            # Start from newest timestep and go backwards
            m = len(self._individual_last_window[distance_metric])
            total = 0
            total_comparisons = 0
            t = 0
            remove = 0
            for j in range(m - 1 - 1, -1, -1):
                diffs_j = self._individual_last_window[distance_metric][j][0]
                new_total = total + np.nansum(diffs_j) * (history.discount_factor ** t)
                new_total_comparisons = total_comparisons + len(diffs_j) * (history.discount_factor ** t)
                t += 1

                # Check if difference is large enough
                # noinspection PyUnresolvedReferences
                if abs(total / max(1,
                                   total_comparisons) - new_total / new_total_comparisons) < history.discount_threshold:
                    remove += 1
                    # Wait for comparisons of at least 5 consecutive individuals in the history
                    # to not pass the threshold
                    if remove >= 5:
                        # Remove all individuals before the given range
                        print("discarding", j - 1, "/", m)

                        for k in range(j - 1):
                            self._individual_last_window[distance_metric].popleft()
                            # TODO: How/When to remove corresponding interactions in history? (Check usage of other notions)
                        # Stop considering older encounters of individuals
                        break

                total = new_total
                total_comparisons = new_total_comparisons

        if exact:
            approx = True
        else:
            approx = unsatisfied_pairs == 0

        # < 2 individuals encountered, assume it's fair
        if n < 2:
            diff = 0.0
        else:
            diff = total / max(1, total_comparisons)
            diff = diff - 1.0  # Maximise diff (negative is unfair) & shift from [0, 1] to [-1, 0] like the group notions

        self._individual_total[distance_metric] = total

        # removed unsatisfied_pairs and difference_per_pair for performance
        return (exact, approx), diff, ([], [], [])

    def weakly_meritocratic(self, history: History, threshold=None, similarity_metric=None, alpha=0,
                            distance_metric=("braycurtis", "braycurtis")):  # TODO: update with new history, with distance metric=> KNN?
        """Never prefer one action over another if the long-term (discounted) reward of
        choosing the latter action is higher
        """
        distance_metric, metric = distance_metric
        unsatisfied = []
        difference_per_ind = []
        exact = True
        combos = 0

        states, actions, true_actions, scores, rewards = history.get_history()
        new_history = list(zip(states, actions, true_actions, scores, rewards))
        n = len(history.get_history()[0])

        new_history = sorted(new_history, key=lambda h: key_state(h[0]))
        new_history = groupby(new_history, key=lambda h: key_state(h[0]))

        if threshold is None:
            threshold = 0

        # For each state
        for _, state_histories in new_history:
            new_state_histories = list(state_histories)
            # Only one individual/state => not enough info, assume fairness (TODO)
            if len(new_state_histories) <= 1:
                continue

            counts = np.zeros(len(self.actions))
            for state, action, true_action, score, reward in new_state_histories:
                counts[action] += 1

            probs = counts / np.sum(counts)

            # TODO: pool
            # num_threads = max(os.cpu_count(), 32)
            num_threads = 2

            map_i = [(i, state, action, true_action, score, reward, self.actions, alpha, probs)
                     for i, (state, action, true_action, score, reward) in enumerate(new_state_histories)]
            if len(map_i) > 500:
                # TODO: this seems to be a bottleneck and works better single-threaded now that previous
                #   comparisons are stored. Leaving it for frameworks with no window if it helps speed up the process there
                with Pool(processes=num_threads) as pool:
                    results = pool.map(_pool_weakly_meritocratic, map_i)
            else:
                results = [_pool_weakly_meritocratic(i) for i in map_i]

            combos += len(results)
            for i, fair, diff in results:
                # Exact fair
                if fair:
                    continue
                exact = False
                # Not approx fair either
                if not threshold or abs(diff) > threshold:
                    individual_i = (states[i], actions[i], true_actions[i], scores[i], rewards[i])
                    unsatisfied.append(individual_i)
                    difference_per_ind.append(diff)

        u = len(unsatisfied)
        if exact:
            approx = True
        else:
            approx = u == 0
        diff = 0 if combos == 0 else u / combos
        # TODO: what is more important, all pairs being satisfied under threshold
        #  OR threshold pairs max that don't satisfy?

        # print((exact, approx), diff, (unsatisfied, [], difference_per_ind))
        # removed unsatisfied and difference_per_ind for performance
        return (exact, approx), diff, ([], [], [])

    def consistency_score_complement(self, history: History, threshold=None, similarity_metric=None,
                                     alpha=None, distance_metric=("braycurtis", "braycurtis")):
        """Individual fairness metric from [1] that measures how similar the labels are for similar instances.

        1 - \frac{1}{n}\sum_{i=1}^n |\hat{y}_i - \frac{1}{\text{n_neighbors}} \sum_{j\in\mathcal{N}_{\text{n_neighbors}}(x_i)} \hat{y}_j|

        [1]	R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork, “Learning Fair Representations,”
            International Conference on Machine Learning, 2013.
        """
        distance_metric, metric = distance_metric
        states, actions, _, _, _ = history.get_history()

        # If individual fairness was not run yet for given distance metric, run it to compute the distances
        if distance_metric not in self.ind_distance_metrics:
            self.individual_fairness(history, threshold, similarity_metric, alpha=1.0,
                                     distance_metric=(distance_metric, metric))

        n = len(actions)
        if n < 2:
            CON = 0.0
        else:
            # Use distances already calculated and stored from individual fairness notion
            #   (d, j, diff, actions[i], actions[j])
            nearest = [heapq.nsmallest(min(n, 5), deq) for _, deq in self._individual_last_window[distance_metric]]
            n_actions = [np.mean([n[-2] for n in nn]) for nn in nearest]
            actions = [[n[-1] for n in nn] for nn in nearest]
            n_actions = np.array(n_actions)
            actions = np.array([a[0] for a in actions])

            # compute consistency score
            CON = - abs(actions - n_actions).mean()

        diff = CON

        exact = diff == 0
        approx = diff > -threshold if threshold else exact

        unsatisfied = []
        difference_per_ind = []

        return (exact, approx), diff, (unsatisfied, [], difference_per_ind)

    def consistency_score_complement_inn(self, history: History, threshold=None, similarity_metric=None,
                                         alpha=None, distance_metric=("braycurtis", "braycurtis")):
        """Individual fairness metric from [1] that measures how similar the labels are for similar instances.

        1 - \frac{1}{n}\sum_{i=1}^n |\hat{y}_i - \frac{1}{\text{n_neighbors}} \sum_{j\in\mathcal{N}_{\text{n_neighbors}}(x_i)} \hat{y}_j|

        [1]	R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork, “Learning Fair Representations,”
            International Conference on Machine Learning, 2013.
        """
        distance_metric, metric = distance_metric
        states, actions, _, _, _ = history.get_history()

        state = states[-1]
        action = actions[-1]
        sensitive_features = tuple(state[self.inn_sensitive_features])
        # print(sensitive_features)

        # TODO:
        if isinstance(distance_metric, str) and distance_metric == "braycurtis":
            metric = scipy.spatial.distance.braycurtis

        def _init_swinn():
            window = self.steps if history.window is None else history.window
            return SWINN(graph_k=10, dist_func=FunctionWrapper(metric), maxlen=window,
                         # warm_up=499,  # TODO: 500
                         warm_up=99 if self.inn_sensitive_features else 499,  # TODO: 500
                         max_candidates=None, delta=0.0001, prune_prob=0.0, n_iters=10, seed=self.seed)

        # Initialisation
        if self._neighbours is None:
            # All together
            if self.inn_sensitive_features is None:
                self._neighbours = _init_swinn()
            else:
                self._neighbours = {sensitive_features: _init_swinn()}
        #
        if self.inn_sensitive_features is not None and self._neighbours.get(sensitive_features) is None:
            self._neighbours[sensitive_features] = _init_swinn()

        nbrs = self._neighbours if self.inn_sensitive_features is None else self._neighbours[sensitive_features]
        # Add new individual to graph
        nbrs.append((state, action))

        n = len(actions)
        if n < 2:
            CON = 0.0
        else:
            if self.inn_sensitive_features is None:
                # Retrieve nearest neighbours (item, nn, dists)
                nearest = nbrs.get_nn_for_all(k=min(n, 5), epsilon=0.1)
                n_actions = np.array([np.mean([n[1] for n in nn[1]]) for nn in nearest])
                actions = np.array([n[0][1] for n in nearest])
            else:
                n_actions = []
                actions = []
                for sf, nbrs in self._neighbours.items():
                    nearest = nbrs.get_nn_for_all(k=min(n, 5), epsilon=0.1)
                    na = np.array([np.mean([n[1] for n in nn[1]]) for nn in nearest])
                    a = np.array([n[0][1] for n in nearest])
                    n_actions.append(na)
                    actions.append(a)
                n_actions = np.hstack(n_actions)
                actions = np.hstack(actions)

            if history.t % 500 == 0:
                import pickle
                if self.inn_sensitive_features:
                    for sf, nbrs in self._neighbours.items():
                        print(sf, len(nbrs._data))
                        graph = nbrs.get_graph()
                        with open(f"SWINN_graph_{sf}_{distance_metric}_{history.t}.pickle", mode="wb") as file:
                            pickle.dump(graph, file)
                else:
                    graph = nbrs.get_graph()
                    with open(f"SWINN_graph_{distance_metric}_{history.t}.pickle", mode="wb") as file:
                        pickle.dump(graph, file)
                    exit()

            # compute consistency score
            CON = - abs(actions - n_actions).mean()

        diff = CON

        exact = diff == 0
        approx = diff > -threshold if threshold else exact

        unsatisfied = []
        difference_per_ind = []

        return (exact, approx), diff, (unsatisfied, [], difference_per_ind)


class SWINN(neighbors.SWINN):
    def get_nn_for_all(self, k, epsilon: float = 0.1):
        """Get the nearest neighbours for all individuals in the graph"""
        return [(p.item, *self.search(p.item, k, epsilon)) for p in self]
        # equivalent to
        # return [(self[item_id].item, [self[item_id].item] + [p[-1] for p in heapq.nsmallest(k - 1, self.individual_heaps[item_id])])
        #         for item_id in range(len(self))]

    def _linear_scan(self, item, k):
        # Lazy search while the warm-up period is not finished
        return super(SWINN, self)._linear_scan(item, k)

    def _search(self, item, k, epsilon: float = 0.1, seed=None, exclude=None) -> tuple[list, list]:
        return super(SWINN, self)._search(item, k, epsilon, seed, exclude)

    def get_graph(self):
        """Get the graph structure"""
        nodes = []
        edges = []
        for i, node in enumerate(self._data):
            # print(node.uuid)
            ind, action = node.item
            # Get node id for display
            # TODO: abstract getters to work for other environments
            age, gender, degree, experience, married, extra_degree, nationality, dutch, french, english, german = ind
            nat = "belgian" if nationality == 0.0 else "foreign"
            gen = "man" if gender == 0.0 else "woman"
            selector_id = f"{nat}_{gen}"
            if married:
                selector_id += f"_married"
            if degree:
                selector_id += f"_degree"
            if extra_degree:
                selector_id += f"_extra-degree"
            if dutch:
                selector_id += f"_dutch"
            if french:
                selector_id += f"_french"
            if english:
                selector_id += f"_english"
            if german:
                selector_id += f"_german"
            hired = "rejected" if action == 0 else "hired"
            selector_id += f"_{hired}"

            label = f"(Node {node.uuid}) Age {int(age * (65 - 18) + 18)}, Exp. {int(experience * (65 - 18))}"

            # Add node
            nodes.append({"classes": ['node', "individual", selector_id],
                          "data": {"id": node.uuid, "label": label,
                                   "individual": ind,
                                   "action": action,
                                   }})

            # Add neighbours
            # print(node.neighbors())
            for n, distance in zip(*node.neighbors()):
                # Round for visibility
                dist = round(distance, 5)

                edge = {'classes': ['edge', "nn"],
                        'data': {'id': f"{node.uuid}_{n.uuid}", 'source': node.uuid, 'target': n.uuid,
                                 'weight': dist,  # 'opacity': max(dist, 0.25),
                                 'arrow_weight': 1.5}}
                edges.append(edge)

        output = {
            'id': 'cytoscape-neighbours',
            'elements': nodes + edges,
        }
        print(len(nodes), "nodes,", len(edges), "edges")
        return output


class OptimisedVertex(Vertex):
    """Optimised SWINN Vertex"""

    def __init__(self, item, uuid: int) -> None:
        # Super call
        super(OptimisedVertex, self).__init__(item, uuid)
        #
        self.neighbours: set[OptimisedVertex] = set()
        self.search_up_to_date = False
        self._search_neighbours = None
        self._search_distributions = None

    def __hash__(self) -> int:
        return self.uuid

    def __eq__(self, other) -> bool:
        return self.uuid == other.uuid

    def __lt__(self, other) -> bool:
        return self.uuid < other.uuid

    def not_up_to_date(self):
        self.search_up_to_date = False
        self._search_neighbours = None
        self._search_distributions = None

    def fill(self, neighbors: list[OptimisedVertex], dists: list[float]):
        for n, dist in zip(neighbors, dists):
            self.edges[n] = dist
            self.flags.add(n)
            n.r_edges[self] = dist
            #
            self.neighbours.add(n)
            n.neighbours.add(self)
            #
            n.not_up_to_date()

        # Neighbors are ordered by distance
        self.worst_edge = n
        self.not_up_to_date()

    def farewell(self):
        # Super call
        super(OptimisedVertex, self).farewell()
        #
        self.neighbours = None

    def add_edge(self, vertex: OptimisedVertex, dist):
        # Super call
        super(OptimisedVertex, self).add_edge(vertex, dist)
        #
        self.neighbours.add(vertex)
        vertex.neighbours.add(self)
        self.not_up_to_date()
        vertex.not_up_to_date()

    def rem_edge(self, vertex: OptimisedVertex):
        # Super call
        super(OptimisedVertex, self).rem_edge(vertex)
        #
        vertex.neighbours.discard(self)
        self.neighbours.discard(vertex)
        self.not_up_to_date()
        vertex.not_up_to_date()

    def is_neighbor(self, vertex):
        return vertex in self.neighbours

    def all_neighbors(self):
        return self.neighbours


class OptimisedSWINN(neighbors.SWINN):
    """Optimised version of SWINN

    TODO: ASSUMPTION: for some methods, the item requesting a search query is in the graph itself:
        retrieve its nearest neighbours. For items that are explicitly mentioned to (possibly) NOT be in the graph,
        the default implementations are used.
    """

    def __init__(self, graph_k: int = 20,
                 dist_func: DistanceFunc | FunctionWrapper | None = None,
                 maxlen: int = 1000,
                 warm_up: int = 500,
                 max_candidates: int = None,
                 delta: float = 0.0001,
                 prune_prob: float = 0.0,
                 n_iters: int = 10,
                 seed: int = None, ):
        # Super call
        super(OptimisedSWINN, self).__init__(graph_k, dist_func, maxlen, warm_up, max_candidates,
                                             delta, prune_prob, n_iters, seed)
        # Store heap of sorted neighbours for linear scan during warm-up period
        self.individual_heaps = {}
        self.individual_comparisons = {}
        self._t = 0

    def _init_graph(self):
        """Create a nearest neighbor graph from stored info. Original creates a random first.
        This starts the graph from already computed distances/neighbours and continues the
        standard refinement process."""
        for vertex in self._data:
            # Based on Vertex.fill(...) method
            for (dist, t, n) in self.individual_heaps[vertex.uuid][:self.graph_k]:
                vertex.edges[n] = dist
                vertex.flags.add(n)
                n.r_edges[vertex] = dist
                #
                vertex.neighbours.add(n)
                n.neighbours.add(vertex)
                #
                n.not_up_to_date()
            vertex.not_up_to_date()
            # Neighbors are ordered by distance
            vertex.worst_edge = n
        self.individual_heaps = {}

    def append(self, item: typing.Any, **kwargs):
        node = OptimisedVertex(item, next(self._uuid))
        if not self._index:
            self._data.append(node)

            # Add individual to heaps
            self.individual_heaps[node.uuid] = []
            for i, neighbour in enumerate(self._data):
                # Skip the item itself
                if i == len(self._data) - 1:
                    break
                # Add this neighbour to the node
                dist = self.dist_func(node.item, neighbour.item)
                heapq.heappush(self.individual_heaps[node.uuid],
                               (dist, self._t + i, neighbour))  # uuid influences tie-breakers
                # Add node to the neighbour
                heapq.heappush(self.individual_heaps[neighbour.uuid],
                               (dist, self._t + i, node))  # uuid influences tie-breakers
                try:
                    self.individual_comparisons[node][neighbour] = dist
                except KeyError:
                    self.individual_comparisons[node] = {neighbour: dist}
            self._t += 1

            if len(self) >= self.warm_up:
                self._init_graph()
                self._refine()
                self._index = True
            return

        # A slot will be replaced, so let's update the search graph first
        if len(self) == self.maxlen:
            self._safe_node_removal()

        # Assign the closest neighbors to the new item
        neighbors, dists = self._search(node.item, self.graph_k)

        # Add the new element to the buffer
        self._data.append(node)
        node.fill(neighbors, dists)
        # The current neighbours have been updated
        node.search_up_to_date = True
        node._search_neighbours = neighbors
        node._search_distributions = dists

    def get_nn_for_all(self, k, epsilon: float = 0.1, return_distances=False):
        """Get the nearest neighbours for all individuals in the graph"""
        if len(self) < self.warm_up:
            # Linear scan
            if return_distances:
                neighbours_p = [heapq.nsmallest(k, self.individual_heaps[item_id]) for item_id in range(len(self))]
                nn = [[p[-1].item for p in np] for np in neighbours_p]
                dists = [[p[0] for p in np] for np in neighbours_p]
                return [(self[item_id].item, (nn[item_id], dists[item_id]))
                        for item_id in range(len(self))]
            else:
                return [(self[item_id].item, [p[-1].item for p in heapq.nsmallest(k, self.individual_heaps[item_id])])
                        for item_id in range(len(self))]
        else:
            return [(p.item, self._search(p.item, k, epsilon, seed=p, return_dists=return_distances)) for p in self]

    def _linear_scan(self, item, k):
        # Lazy search while the warm-up period is not finished
        return super(OptimisedSWINN, self)._linear_scan(item, k)

    def _search(self, item, k, epsilon: float = 0.1, seed=None, exclude=None, return_dists=True) -> tuple[list, list]:
        # Search has already been performed and the seeds neighbours didn't change
        if seed is not None and seed.search_up_to_date:
            if return_dists:
                return seed._search_neighbours[:k], seed._search_distributions[:k]
            else:
                return seed._search_neighbours[:k]

        # Limiter for the distance bound
        distance_scale = 1 + epsilon
        # Distance threshold for early stops
        distance_bound = math.inf

        if exclude is None:
            exclude = set()
        else:
            exclude = {exclude.uuid}

        if seed is None:
            # Make sure the starting point for the search is valid
            while True:
                # Random seed point to start the search
                seed = self[self._rng.randint(0, len(self) - 1)]
                if not seed.is_isolated() and seed.uuid not in exclude:
                    break

        # To avoid computing distances more than once for a given node
        visited = {seed.uuid}
        visited |= exclude

        # Search pool is a minimum heap
        pool = []

        # Results are stored in a maximum heap
        result = []

        # c_dist, c_n = heapq.heappop(pool)
        c_dist, c_n = 0, seed
        while c_dist < distance_bound:
            for n in c_n.all_neighbors():
                if n.uuid in visited:
                    continue

                # TODO: added for individual_comparisons
                try:
                    comp = self.individual_comparisons[n]
                except KeyError:
                    comp = None

                if c_n == seed:
                    _, _, dist = seed.get_edge(n)
                # TODO: assumption that seed.item is item in most searches, or a comparison with current node
                #  has been made in the past for given neighbour
                elif seed.item is item and seed.is_neighbor(n):
                    _, _, dist = seed.get_edge(n)
                elif comp is not None:
                    try:
                        dist = self.individual_comparisons[n][seed]
                    except KeyError:
                        dist = self.dist_func(item, n.item)
                        self.individual_comparisons[n][seed] = dist
                else:
                    dist = self.dist_func(item, n.item)
                    self.individual_comparisons[n] = {seed: dist}

                if len(result) < k:
                    heapq.heappush(result, (-dist, n))
                    heapq.heappush(pool, (dist, n))
                    distance_bound = distance_scale * -result[0][0]
                elif dist < -result[0][0]:
                    heapq.heapreplace(result, (-dist, n))
                    heapq.heappush(pool, (dist, n))
                    distance_bound = distance_scale * -result[0][0]
                visited.add(n.uuid)
            if len(pool) == 0:
                break
            c_dist, c_n = heapq.heappop(pool)

        result.sort(reverse=True)
        if return_dists:
            neighbors, dists = map(list, zip(*((r[1], -r[0]) for r in result)))
            # The current neighbours have been updated
            if seed is not None:
                seed.search_up_to_date = True
                seed._search_neighbours = neighbors
                seed._search_distributions = dists
            return neighbors, dists
        else:
            neighbors = [r[1] for r in result]
            # The current neighbours have been updated
            if seed is not None:
                seed.search_up_to_date = True
                seed._search_neighbours = neighbors
                seed._search_distributions = None
            return neighbors

    def _safe_node_removal(self):
        """Remove the oldest data point from the search graph.

        Make sure nodes are accessible from any given starting point after removing the oldest
        node in the search graph. New traversal paths will be added in case the removed node was
        the only bridge between its neighbors.

        """
        node = self._data.popleft()
        # Get previous neighborhood info
        rns = node.r_neighbors()[0]
        ns = node.neighbors()[0]
        node.farewell()
        node.not_up_to_date()

        ########### added
        # Delete all comparisons of this node with others
        for n in self.individual_comparisons[node]:
            del self.individual_comparisons[n][node]
        # Delete the nodes comparisons
        del self.individual_comparisons[node]
        ##########

        # Nodes whose only direct neighbor was the removed node
        rns = {rn for rn in rns if not rn.has_neighbors()}
        # Nodes whose only reverse neighbor was the removed node
        ns = {n for n in ns if not n.has_rneighbors()}

        affected = list(rns | ns)
        isolated = rns.intersection(ns)

        # First we handle the unreachable nodes
        for al in isolated:
            neighbors, dists = self._search(al.item, self.graph_k)
            al.fill(neighbors, dists)
            al.search_up_to_date = True
            al._search_neighbours = neighbors
            al._search_distributions = dists

        rns -= isolated
        ns -= isolated
        ns = tuple(ns)

        # Nodes with no direct neighbors
        for rn in rns:
            seed = None
            # Check the group of nodes without reverse neighborhood for seeds
            # Thus we can join two separate groups
            if len(ns) > 0:
                seed = self._rng.choice(ns)

            # Use the search index to create new connections
            neighbors, dists = self._search(rn.item, self.graph_k, seed=seed, exclude=rn)
            rn.fill(neighbors, dists)
            rn.search_up_to_date = True
            rn._search_neighbours = neighbors
            rn._search_distributions = dists

        self._refine(affected)  # TODO: with search


if __name__ == '__main__':
    from scenario.job_hiring.features import ApplicantGenerator

    sseed = 0
    n = 500#0
    n_max = 10000

    # rng = np.random.default_rng(seed=sseed)
    # vertex_neighbours = {nn: 0 for nn in rng.choice(range(n_max), size=n)}
    # vertex_neighbours2 = {nn: 0 for nn in rng.choice(range(n_max), size=n)}
    #
    # import cProfile
    # with cProfile.Profile() as pr:
    #     t0 = time.time()
    #     for vertex in range(n_max):
    #         is_neighbour = vertex in vertex_neighbours or vertex in vertex_neighbours2
    #
    #     t1 = time.time()
    #     print("time old", t1 - t0)
    #
    #     pr.print_stats(SortKey.CUMULATIVE)
    #
    # with cProfile.Profile() as pr:
    #     t0 = time.time()
    #     for vertex in range(n_max):
    #         is_neighbour = vertex_neighbours.get(vertex) or vertex_neighbours2.get(vertex)
    #
    #     t1 = time.time()
    #     print("time new", t1 - t0)
    #
    #     pr.print_stats(SortKey.CUMULATIVE)
    #
    # exit()
    # TODO

    warm_up = 100  # 500
    graph_k = 10
    k = 5
    epsilon = 0.1

    print(f"seed={sseed}, num_samples={n}, warm-up={warm_up}, graph_k={graph_k}, k={k}, epsilon={epsilon}")

    ag = ApplicantGenerator(seed=sseed, csv="../../scenario/job_hiring/data/belgian_population.csv")
    population = ag.sample(n=n)

    # nearest_neighbours_orig = SWINN(graph_k=graph_k, dist_func=FunctionWrapper(scipy.spatial.distance.braycurtis),
    #                                 warm_up=warm_up, seed=sseed)
    # import cProfile
    # with cProfile.Profile() as pr:
    #     t0 = time.time()
    #     for t, p in enumerate(population):
    #         individual = CombinedState(sample_context={}, sample_individual=p).to_array(individual_only=True)
    #         action = int(individual[1])
    #         nearest_neighbours_orig.append((individual, action))
    #         #
    #         nearest = nearest_neighbours_orig.get_nn_for_all(k=k, epsilon=epsilon)
    #
    #     t1 = time.time()
    #     d1 = t1 - t0
    #     print("time orig", d1)
    #
    #     pr.print_stats(SortKey.CUMULATIVE)

    nearest_neighbours = OptimisedSWINN(graph_k=graph_k, dist_func=FunctionWrapper(scipy.spatial.distance.braycurtis),
                                        warm_up=warm_up, seed=sseed)

    import cProfile
    with cProfile.Profile() as pr:
        t0 = time.time()
        for t, p in enumerate(population):
            individual = CombinedState(sample_context={}, sample_individual=p).to_array(individual_only=True)
            action = int(individual[1])
            nearest_neighbours.append((individual, action))
            #
            nearest = nearest_neighbours.get_nn_for_all(k=k, epsilon=epsilon)

        # old: -0.43920000000000003 new: -0.44839999999999997
        # for n in nearest[:5]:
        #     print(n)
        #     break
        n_actions = np.array([np.mean([n.item[1] for n in nn[1]]) for nn in nearest])  # TODO: extract item in method
        actions = np.array([n[0][1] for n in nearest])
        CON = - abs(actions - n_actions).mean()
        print(CON)

        t1 = time.time()
        print("time new", t1 - t0)

        pr.print_stats(SortKey.CUMULATIVE)

