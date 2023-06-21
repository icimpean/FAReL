from enum import Enum
from itertools import groupby
from multiprocessing import Pool

import numpy as np
from aif360.sklearn.metrics import consistency_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_X_y

from fairness.individual import IndividualNotion, IndividualFairnessBase
from fairness.history import History


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
        combos = 0
        exact = True

        # num_threads = max(os.cpu_count(), 32)
        num_threads = 4

        map_i_j = []
        previous_results = []
        for i in range(n):
            for j in range(i + 1, n):
                id_i, id_j = ids[i], ids[j]
                comp_id = id_i+id_j
                previous_comp = self._individual_comparisons.get(comp_id)
                if previous_comp is None:
                    map_i_j.append((i, j, states[i], states[j], scores[i], scores[j],
                                    similarity_metric, alpha, distance_metric))
                else:
                    # print("Individuals\n\t", states[i], "\nand\n\t", states[j], "\nwere compared before (", comp_id, "), moving on...")
                    previous_results.append(previous_comp)
        # TODO: numba compiler
        if False:#(map_i_j) > 500:
            # TODO: this seems to be a bottleneck and works better single-threaded now that previous
            #   comparisons are stored. Leaving it for frameworks with no window should it help speed up there
            with Pool(processes=num_threads) as pool:
                # map_i_j = [(i, j, states[i], states[j], scores[i], scores[j], similarity_metric, None)
                #            for i in range(n) for j in range(i + 1, n)]
                results = pool.map(_pool_individual_fairness, map_i_j)
        else:
            results = [_pool_individual_fairness(ij) for ij in map_i_j]
        results.extend(previous_results)

        combos = len(results)
        # print(results[:10])
        # print(combos, "results, of which", len(previous_results), "were previously calculated")
        total = []
        new_individual_comparisons = {}
        for i, j, fair, diff, D, d in results:
            id_i, id_j = ids[i], ids[j]
            comp_id = id_i + id_j
            new_individual_comparisons[comp_id] = (i, j, fair, diff, D, d)
            total.append(diff)

            # Exact fair
            if fair:
                continue
            exact = False
            # Not approx fair either
            if not threshold or diff < -threshold:
                individual_i = (states[i], actions[i], true_actions[i], scores[i], rewards[i])
                individual_j = (states[j], actions[j], true_actions[j], scores[j], rewards[j])
                unsatisfied_pairs.append((individual_i, individual_j))
                difference_per_pair.append(diff)

        u = len(unsatisfied_pairs)
        if exact:
            approx = True
        else:
            approx = u == 0
        total = np.array(total)
        diff = np.nansum(total) / max(1, len(total))
        diff = diff - 1.0  # Maximise diff (negative is unfair) & shift from [0, 1] to [-1, 0] like the group notions
        # if len(total) > 0:
        #     tot = np.array(total)
        #     print(diff, tot.mean(), tot.min(), tot.max())

        # print(u, "/", combos, "unsatisfied_pairs")
        # print((exact, approx), diff, ([], unsatisfied_pairs, difference_per_pair))
        self._individual_comparisons = new_individual_comparisons

        return (exact, approx), diff, ([], unsatisfied_pairs, difference_per_pair)

    def weakly_meritocratic(self, history: History, get_individual, threshold=None, similarity_metric=None, alpha=0,
                            distance_metric="minkowski"):
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
        return (exact, approx), diff, (unsatisfied, [], difference_per_ind)

    def consistency_score_complement(self, history: History, get_individual, threshold=None, similarity_metric=None,
                                     alpha=None, distance_metric="minkowski"):
        """Individual fairness metric from [1] that measures how similar the labels are for similar instances.

        1 - \frac{1}{n}\sum_{i=1}^n |\hat{y}_i - \frac{1}{\text{n_neighbors}} \sum_{j\in\mathcal{N}_{\text{n_neighbors}}(x_i)} \hat{y}_j|

        [1]	R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork, “Learning Fair Representations,”
            International Conference on Machine Learning, 2013.
        """
        states, actions, _, _, _ = history.get_history()
        # individual_states = list(map(lambda state: state.to_array(individual_only=True), states))
        individual_states = list(map(lambda state: get_individual(state, normalise=True), states))
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
