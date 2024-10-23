from __future__ import annotations

from collections import deque
from enum import Enum
from itertools import groupby
from multiprocessing import Pool

import numpy as np
import scipy.spatial.distance
from river.neighbors.base import FunctionWrapper

from fairness.history import History, DiscountedHistory, HistoryTimestep
from fairness.individual import IndividualNotion, IndividualFairnessBase
from fairness.individual.individual_deque import IndividualDeque
from fairness.individual.swinn.optimised_swinn import OptimisedSWINN


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
            IndividualNotion.IndividualFairness_t: self.individual_fairness,
            IndividualNotion.WeaklyMeritocratic: self.weakly_meritocratic,
            # IndividualNotion.WeaklyMeritocratic_t: self.weakly_meritocratic,
            IndividualNotion.ConsistencyScoreComplement: self.consistency_score_complement,
            # IndividualNotion.ConsistencyScoreComplement_t: self.consistency_score_complement,
            #
            IndividualNotion.ConsistencyScoreComplement_INN: self.consistency_score_complement_inn,
            # IndividualNotion.ConsistencyScoreComplement_INN_t: self.consistency_score_complement_inn,
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
        is_t = isinstance(history, HistoryTimestep)

        # Keep track of the differences to discard once the window passes
        if (with_window or is_discounted) and self._individual_last_window.get(distance_metric) is None and not is_t:
            self._individual_last_window[distance_metric] = deque(maxlen=history.window)

        # Can only compare as many individuals as present in self._individual_last_window + 1,
        #   others are dropped due to threshold
        if is_discounted:
            lowest_n = n - (len(self._individual_last_window[distance_metric]) + 1)

        # If given n interactions/individuals, under the assumption that all interactions until n have been compared in
        #   the previous timestep, only individual n should be compared to 0 until n - 1
        if history.newly_added == 1:
            i = n - 1
            map_i_j = [(i, j, states[i], states[j], scores[i], scores[j], similarity_metric, alpha, metric, num_actions)
                       for j in range((n - 1) - 1, lowest_n - 1, -1)]
        # Multiple individuals were added after last interaction: add all of them to all previous AND each other
        else:
            map_i_j = [(i, j, states[i], states[j], scores[i], scores[j], similarity_metric, alpha, metric, num_actions)
                       for i in range(n - history.newly_added, n)
                       for j in range(i - 1, lowest_n - 1, -1)
                       ]
        results = [_pool_individual_fairness(ij) for ij in map_i_j]
        unsatisfied_pairs = 0

        if with_window or is_discounted:
            # Store data at individual with lowest index: comparison gets removed with individual when moving out
            #   of the sliding window. Remove earliest individual from deque if outside window.
            if len(self._individual_last_window[distance_metric]) == history.window:
                last, _ = self._individual_last_window[distance_metric][0]
                self._individual_total[distance_metric] -= np.nansum(last)
            # diffs, deque/heap
            for _ in range(history.newly_added):
                self._individual_last_window[distance_metric].append(([], IndividualDeque(max_n=5, window=history.window)))

        # Windowed + full history
        if not is_discounted:
            if self._individual_total.get(distance_metric) is None:
                self._individual_total[distance_metric] = 0.0
            if is_t:
                total = 0
                total_comparisons = len(results)
            else:
                total = self._individual_total[distance_metric]
                total_comparisons = n * (n - 1) // 2

            # Timestep has different indices
            for i, j, fair, diff, D, d in results:
                if not np.isnan(diff):
                    total += diff
                if with_window and not is_t:
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
        if len(results) < 2:
            diff = 0.0
        else:
            diff = total / max(1, total_comparisons)
            diff = diff - 1.0  # Maximise diff (negative is unfair) & shift from [0, 1] to [-1, 0] like the group notions

        self._individual_total[distance_metric] = total

        # removed unsatisfied_pairs and difference_per_pair for performance
        return (exact, approx), diff, ([], [], [])

    def weakly_meritocratic(self, history: History, threshold=None, similarity_metric=None, alpha=0,
                            distance_metric=(
                            "braycurtis", "braycurtis")):  # TODO: update with new history, with distance metric=> KNN?
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
        is_t = isinstance(history, HistoryTimestep)
        # TODO: nearest neighbours at single timestep currently interpreted as nearest neighbours of current timesteps
        #  individuals with entire sliding/discounted history

        # If individual fairness was not run yet for given distance metric, run it to compute the distances
        if distance_metric not in self.ind_distance_metrics:
            self.individual_fairness(history, threshold, similarity_metric, alpha=1.0,
                                     distance_metric=(distance_metric, metric))

        n = len(history.actions)
        if n < history.newly_added + 1:
            CON = 0.0
        else:
            # Use distances already calculated and stored from individual fairness notion
            #   (d, j, diff, actions[i], actions[j])
            if is_t:
                nearest = [deq.n_smallest for _, deq in self._individual_last_window[distance_metric][n-history.newly_added:n]]
            else:
                nearest = [deq.n_smallest for _, deq in self._individual_last_window[distance_metric]]
            try:
                n_actions = np.mean([[n[-2] for n in nn] for nn in nearest], axis=1)
            except ValueError:
                n_actions = [np.mean([n[-2] for n in nn]) for nn in nearest]
            actions = np.array([nn[0][-1] for nn in nearest])

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
            return OptimisedSWINN(graph_k=10, dist_func=FunctionWrapper(metric), maxlen=window,
                                  warm_up=59, # if self.inn_sensitive_features else 499,  # TODO: 500
                                  max_candidates=None,
                                  delta=0.0001, prune_prob=0.5,  # prune_prob=0.0
                                  n_iters=10, seed=self.seed)

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
                nearest = nbrs.get_nn_for_all(k=min(n, 5), epsilon=0.1, return_distances=False, return_as_array=True)
                n_actions = np.mean([[n[1] for n in nn[1]] for nn in nearest], axis=1)
                # n_actions = [np.nanmean([n[1] for n in nn[1]]) for nn in nearest]
                actions = np.array([n[0][1] for n in nearest])

            else:
                n_actions = []
                actions = []
                for sf, nbrs in self._neighbours.items():
                    nearest = nbrs.get_nn_for_all(k=min(n, 5), epsilon=0.1)
                    # na = np.mean([[n[1] for n in nn[1]] for nn in nearest], axis=1)
                    na = [np.nanmean([n[1] for n in nn[1]]) for nn in nearest]
                    a = np.array([n[0][1] for n in nearest])
                    n_actions.append(na)
                    actions.append(a)
                n_actions = np.hstack(n_actions)
                actions = np.hstack(actions)
            # compute consistency score
            CON = - np.nanmean(abs(actions - n_actions))

        diff = CON

        exact = diff == 0
        approx = diff > -threshold if threshold else exact

        unsatisfied = []
        difference_per_ind = []

        return (exact, approx), diff, (unsatisfied, [], difference_per_ind)
