from collections import deque
from enum import Enum
from itertools import groupby
from multiprocessing import Pool

import numpy as np
from aif360.sklearn.metrics import consistency_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_X_y

from fairness.individual import IndividualNotion, IndividualFairnessBase
from fairness.history import History, DiscountedHistory
from scenario import CombinedState


def dict_to_array(d):
    a = []
    for v in d.values():
        if isinstance(v, Enum):
            a.append(v.value)
        else:
            a.append(v)
    return np.array(a, dtype=float)


def key_state(s, get_individual):
    return str(dict_to_array(s))


def hellinger(p, q):
    return np.sum([(np.sqrt(t1) - np.sqrt(t2)) * (np.sqrt(t1) - np.sqrt(t2))
                   for t1, t2 in zip(p, q)]) / np.sqrt(2)


def _pool_individual_fairness(args):
    i, j, state_i, state_j, score_i, score_j, similarity_metric, alpha, distance_metric = args
    d = similarity_metric(state_i, state_j, alpha=alpha, distance=distance_metric)
    D = hellinger(score_i, score_j)
    # print(score_i, score_j, D, d, d - D)
    # i, j Fair, difference, D, d
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


class IndividualFairness(IndividualFairnessBase):
    """A collection of fairness notions w.r.t. individuals.

        Attributes:
            actions: A list of enumerations, representing the actions to check fairness for.
        """

    def __init__(self, actions):
        # Super call
        super(IndividualFairness, self).__init__(actions)
        # Mapping from enumeration to fairness method
        self._map = {
            IndividualNotion.IndividualFairness: self.individual_fairness,
            IndividualNotion.WeaklyMeritocratic: self.weakly_meritocratic,
            IndividualNotion.ConsistencyScoreComplement: self.consistency_score_complement,
        }

        # Don't recalculate individuals who have been compared already, they haven't changed
        self._individual_comparisons = {}
        self._individual_last_window = None
        self._individual_total = 0.0

    def get_notion(self, notion: IndividualNotion, history: History, get_individual=lambda state: state, threshold=None,
                   similarity_metric=None, alpha=None, distance_metric=None):
        # noinspection PyArgumentList
        # print("get_notion", notion)
        return self._map[notion](history, get_individual, threshold, similarity_metric, alpha, distance_metric)

    def individual_fairness(self, history: History, get_individual, threshold=None, similarity_metric=None, alpha=1.0,
                            distance_metric="minkowski"):
        """Let i and j be two individuals represented by their attributes values vectors v_i and v_j.
        Let d(v_i,v_j) represent the similarity distance between individuals i and j.
        Let D be a distance metric between probability distributions M(v_i) and M(v_j).
        Fairness through awareness is achieved iff, for any pair of individuals i and j

        D(M(v_i), M(v_j)) ≤ d(v_i, v_j)
        """
        unsatisfied_pairs = []
        difference_per_pair = []
        states, actions, true_actions, scores, rewards = history.get_history()
        ids = history.ids
        n = len(states)
        lowest_n = 0
        combos = 0
        exact = True

        # num_threads = max(os.cpu_count(), 32)
        num_threads = 4

        with_window = history.window is not None
        is_discounted = isinstance(history, DiscountedHistory)

        # Keep track of the differences to discard once the window passes
        if (with_window or is_discounted) and self._individual_last_window is None:
            self._individual_last_window = deque(maxlen=history.window)

        # Can only compare as many individuals as present in self._individual_last_window + 1,
        #   others are dropped due to threshold
        if is_discounted:
            lowest_n = n - (len(self._individual_last_window) + 1)

        # If given n interactions/individuals, under the assumption that all interactions until n have been compared in
        #   the previous timestep, only individual n should be compared to 0 until n - 1
        map_i_j = []
        i = n - 1
        for j in range((n - 1) - 1, lowest_n - 1, -1):  # Run range(lowest_n, n - 1) backwards for is_discounted
            map_i_j.append((i, j, states[i], states[j], scores[i], scores[j],
                            similarity_metric, alpha, distance_metric))
        results = [_pool_individual_fairness(ij) for ij in map_i_j]
        unsatisfied_pairs = 0

        if with_window or is_discounted:
            # Store data at individual with lowest index: comparison gets removed with individual when moving out
            #   of the sliding window. Remove earliest individual from deque if outside window.
            if len(self._individual_last_window) == history.window:
                last = self._individual_last_window[0]
                self._individual_total -= np.nansum(last)
            self._individual_last_window.append([])

        # Windowed + full history
        if not is_discounted:
            total = self._individual_total
            total_comparisons = n * (n - 1) // 2

            for i, j, fair, diff, D, d in results:
                if not np.isnan(diff):
                    total += diff
                if with_window:
                    # j is a previously encountered individual, i is current
                    self._individual_last_window[j].append(diff)

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
            shifted_n = n - len(self._individual_last_window)

            for i, j, fair, diff, D, d in results:
                # j is a previously encountered individual, i is current
                shifted_j = j - shifted_n
                self._individual_last_window[shifted_j].append(diff)
                # Exact fair
                if fair:
                    continue
                exact = False
                # Not approx fair either
                if not threshold or diff < -threshold:
                    unsatisfied_pairs += 1

            # Start from newest timestep and go backwards
            m = len(self._individual_last_window)
            total = 0
            total_comparisons = 0
            t = 0
            remove = 0
            for j in range(m - 1 - 1, -1, -1):
                diffs_j = self._individual_last_window[j]
                new_total = total + np.nansum(diffs_j) * (history.discount_factor ** t)
                new_total_comparisons = total_comparisons + len(diffs_j) * (history.discount_factor ** t)
                t += 1

                # Check if difference is large enough
                # noinspection PyUnresolvedReferences
                if abs(total / max(1, total_comparisons) - new_total / new_total_comparisons) < history.discount_threshold:
                    remove += 1
                    # Wait for comparisons of at least 5 consecutive individuals in the history
                    # to not pass the threshold
                    if remove >= 5:
                        # Remove all individuals before the given range
                        print("discarding", j-1, "/", m)

                        for k in range(j - 1):
                            self._individual_last_window.popleft()
                            # TODO: How/When to remove corresponding interactions in history? (Check usage of other notions)
                        # Stop considering older encounters of individuals
                        break

                total = new_total
                total_comparisons = new_total_comparisons
            # total_comparisons = t * (t - 1) // 2
            # print("ind_fairness", t, total_comparisons)

        if exact:
            approx = True
        else:
            approx = unsatisfied_pairs == 0

        diff = total / max(1, total_comparisons)
        diff = diff - 1.0  # Maximise diff (negative is unfair) & shift from [0, 1] to [-1, 0] like the group notions

        self._individual_total = total

        # removed unsatisfied_pairs and difference_per_pair for performance
        return (exact, approx), diff, ([], [], [])

    def weakly_meritocratic(self, history: History, get_individual, threshold=None, similarity_metric=None, alpha=0,
                            distance_metric="minkowski"):  # TODO: update with new history
        """Never prefer one action over another if the long-term (discounted) reward of
        choosing the latter action is higher
        """
        unsatisfied = []
        difference_per_ind = []
        exact = True
        combos = 0

        states, actions, true_actions, scores, rewards = history.get_history()
        individual_states = map(get_individual, states)
        new_history = list(zip(individual_states, actions, true_actions, scores, rewards))
        n = len(history.get_history()[0])

        new_history = sorted(new_history, key=lambda h: key_state(h[0], get_individual))
        new_history = groupby(new_history, key=lambda h: key_state(h[0], get_individual))

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

    def consistency_score_complement(self, history: History, get_individual, threshold=None, similarity_metric=None,
                                     alpha=None, distance_metric="minkowski"):
        """Individual fairness metric from [1] that measures how similar the labels are for similar instances.

        1 - \frac{1}{n}\sum_{i=1}^n |\hat{y}_i - \frac{1}{\text{n_neighbors}} \sum_{j\in\mathcal{N}_{\text{n_neighbors}}(x_i)} \hat{y}_j|

        [1]	R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork, “Learning Fair Representations,”
            International Conference on Machine Learning, 2013.
        """
        states, actions, _, _, _ = history.get_history()
        # individual_states = list(map(lambda state: state.to_array(individual_only=True), states))
        if isinstance(states[0], CombinedState):
            individual_states = list(map(lambda state: get_individual(state, normalise=True), states))
        else:
            individual_states = states

        n = len(actions)

        if n < 2:
            CON = -1.0
        else:
            # TODO: abstract n_neighbors
            CON = consistency_score_metric(individual_states, actions, n_neighbors=min(n, 5),
                                           distance_metric=distance_metric)
        diff = CON

        exact = diff == 0
        approx = diff > -threshold if threshold else exact

        unsatisfied = []
        difference_per_ind = []

        return (exact, approx), diff, (unsatisfied, [], difference_per_ind)


def consistency_score_metric(X, y, n_neighbors=5, distance_metric="minkowski"):
    """Compute the consistency score, based on aif360.sklearn.metrics.consistency_score
    with optional distance metric. Default for algorithm='ball_tree' is metric='minkowski'
    """
    # print(distance_metric)
    # cast as ndarrays
    X, y = check_X_y(X, y)
    # learn a KNN on the features
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric=distance_metric)
    nbrs.fit(X)
    indices = nbrs.kneighbors(X, return_distance=False)

    # compute consistency score
    return - abs(y - y[indices].mean(axis=1)).mean()
