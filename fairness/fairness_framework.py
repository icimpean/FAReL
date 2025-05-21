from typing import Union, List, Iterable
from functools import partial

import gymnasium as gym

from fairness import SensitiveAttribute
from fairness.group import GroupNotion, TIMESTEP_GROUP_NOTIONS
from fairness.group.group_fairness import GroupFairness
from fairness.history import History, SlidingWindowHistory, DiscountedHistory, HistoryTimestep
from fairness.individual import IndividualNotion, TIMESTEP_INDIVIDUAL_NOTIONS
from fairness.individual.individual_fairness import IndividualFairness
from scenario import CombinedState


class FairnessFramework(object):
    """A fairness framework.

    Attributes:
        actions: The possible actions for the agent-environment interaction.
        sensitive_attributes: The attributes for which to check fairness.
        threshold: The threshold for defining approximate fairness.
        group_notions: The group fairness notions considered.
            If None, all implemented group fairness notions are considered.
        individual_notions: The individual fairness notions considered.
            If None, all implemented individual fairness notions are considered.

        history: The collection of state-action-score-reward tuples encountered by an agent
    """
    def __init__(self, actions, sensitive_attributes: Union[SensitiveAttribute, List[SensitiveAttribute]],
                 threshold=None, similarity_metric=None,
                 distance_metrics=[], alpha=None,
                 group_notions=None, individual_notions=None, window=None,
                 store_interactions=True, has_individual_fairness=True,
                 discount_factor=None, discount_threshold=None, discount_delay=None, min_window=None,
                 nearest_neighbours=None,
                 inn_sensitive_features=None, seed=None, steps=None):
        self.actions = actions
        self.window = window
        self.store_interactions = store_interactions
        self.has_individual_fairness = has_individual_fairness
        self.discount_factor = discount_factor
        self.discount_threshold = discount_threshold
        self.discount_delay = discount_delay
        self.min_window = min_window
        self.nearest_neighbours = nearest_neighbours
        # Use a discounted history
        if discount_factor is not None:
            self.history = DiscountedHistory(actions,
                                             self.discount_factor, self.discount_threshold, self.discount_delay,
                                             self.min_window, store_interactions=self.store_interactions,
                                             has_individual_fairness=self.has_individual_fairness,
                                             nearest_neighbours=self.nearest_neighbours)
        # Use a sliding window history
        else:
            self.history = SlidingWindowHistory(actions, self.window, store_interactions=self.store_interactions,
                                                has_individual_fairness=self.has_individual_fairness,
                                                nearest_neighbours=self.nearest_neighbours)
        self.history_t = HistoryTimestep(actions, has_individual_fairness=self.has_individual_fairness,
                                         nearest_neighbours=self.nearest_neighbours)
        #
        self.sensitive_attributes = [sensitive_attributes] \
            if isinstance(sensitive_attributes, SensitiveAttribute) else sensitive_attributes
        #
        self.similarity_metric = similarity_metric
        self.alpha = alpha
        #
        self.threshold = threshold
        #
        self.group_notions = group_notions
        self.group_fairness = GroupFairness(actions)
        #
        self.individual_notions = individual_notions
        if not self.has_individual_fairness:
            self.individual_notions = []
        assert (len(distance_metrics) == len(self.individual_notions)) or (len(distance_metrics) == 1), \
            f"The number of distance_metrics given must be either 1 or equal to the number of individual_notions. " \
            f"Found distance_metrics: {distance_metrics}, individual_notions: {self.individual_notions}"
        self.distance_metrics = distance_metrics if len(distance_metrics) != 1 \
            else distance_metrics * len(self.individual_notions)

        ind_metrics = [d for n, d in zip(self.individual_notions, self.distance_metrics)
                       if n is IndividualNotion.IndividualFairness]
        csc_metrics = [d for n, d in zip(self.individual_notions, self.distance_metrics)
                       if n is IndividualNotion.ConsistencyScoreComplement]
        self.individual_fairness = IndividualFairness(actions, ind_metrics, csc_metrics, inn_sensitive_features, seed,
                                                      steps)

        self.all_notions = self.group_notions + self.individual_notions

    def update_history(self, episode, t, entities):
        """Update the framework with a new observed tuple

        Args:
            episode: The episode where the interaction took place
            t: The timestep of the interaction
            state: The observed state
            action: The action taken in that state
            true_action: The correct action according to the ground truth of the problem
            score: The score assigned by the agent for the given state, or state-action pair
            reward: The reward received for the given action
        """
        self.history_t.update_t(episode, t, entities, self.sensitive_attributes)
        self.history.update(episode, t, entities, self.sensitive_attributes)

    def get_group_notion(self, group_notion: GroupNotion, sensitive_attribute: SensitiveAttribute, threshold=None):
        """Get the given group notion"""
        history = self.history_t if group_notion in TIMESTEP_GROUP_NOTIONS else self.history
        return self.group_fairness.get_notion(group_notion, history, sensitive_attribute, threshold)

    def get_individual_notion(self, individual_notion: IndividualNotion,
                              threshold=None, similarity_metric=None, alpha=None,
                              distance_metric=("braycurtis", "braycurtis")):
        """Get the given individual notion"""
        history = self.history_t if individual_notion in TIMESTEP_INDIVIDUAL_NOTIONS else self.history
        return self.individual_fairness.get_notion(individual_notion, history, threshold,
                                                   similarity_metric, alpha, distance_metric)


class ExtendedfMDP(gym.Env):
    """An extended job hiring fMDP, with a fairness framework"""
    def __init__(self, env, fairness_framework: FairnessFramework):
        # Super call
        super(ExtendedfMDP, self).__init__()
        #
        self.env = env
        self.fairness_framework = fairness_framework
        if not self.fairness_framework.store_interactions and self.fairness_framework.has_individual_fairness:
            self.fairness_framework.history.store_state_array = env.state_to_array
        self.H_OM_distance = {
            # distance_metric: lambda state1, state2: self.env.H_OM_distance(state1, state2, distance_metric,
            #                                                                self.fairness_framework.alpha, exp=True)
            distance_metric: partial(self.env.H_OM_distance, distance=distance_metric,
                                     alpha=self.fairness_framework.alpha, exp=True)
            for distance_metric in ["HEOM", "HMOM"]
        }

        #
        self._t = -1
        self._episode = -1

        # Objective names
        self.obj_names = ["reward"]
        if len(self.fairness_framework.sensitive_attributes) <= 1:
            for notion in self.fairness_framework.group_notions:
                self.obj_names.append(notion.name)
        else:
            for sensitive_attribute in self.fairness_framework.sensitive_attributes:
                for notion in self.fairness_framework.group_notions:
                    self.obj_names.append(f"{notion.name} {str(sensitive_attribute)}")
        for notion in self.fairness_framework.individual_notions:
            self.obj_names.append(notion.name)

    def reset(self, seed=None, options=None):
        self._t += 1
        self._episode += 1
        self.fairness_framework.history.features = self.env._features_i
        return self.env.reset()

    def step(self, action, scores=None):
        next_state, reward, done, info = self.env.step(action)

        true_action = info.get("true_action")
        if true_action is None:
            true_action = -1
        entities = self.env.get_all_entities_in_state(self.env.previous_state, action, true_action, scores, reward)
        self.fairness_framework.update_history(self._episode, self._t, entities)

        # Add fairness notions as additional rewards
        reward = reward if isinstance(reward, Iterable) else [reward]
        # Group notions: For each sensitive attribute
        for sensitive_attribute in self.fairness_framework.sensitive_attributes:
            for notion in self.fairness_framework.group_notions:
                (exact, approx), diff, (prob_sensitive, prob_other) = \
                    self.fairness_framework.get_group_notion(notion, sensitive_attribute,
                                                             self.fairness_framework.threshold)
                reward.append(diff)
        # Individual notions:
        for notion, distance_metric in zip(self.fairness_framework.individual_notions,
                                           self.fairness_framework.distance_metrics):
            if distance_metric.startswith("H") and distance_metric.endswith("OM"):
                metric = self.H_OM_distance[distance_metric]
            elif distance_metric == "braycurtis":
                metric = self.env.braycurtis_metric  # TODO: w
            else:
                metric = distance_metric
            (exact, approx), diff, (u_ind, u_pairs, U_diff) = \
                self.fairness_framework.get_individual_notion(notion,
                                                              self.fairness_framework.threshold,
                                                              self.fairness_framework.similarity_metric,
                                                              self.fairness_framework.alpha,
                                                              (distance_metric, metric))
            reward.append(diff)
        self._t += 1

        return next_state, reward, done, info

    def normalise_state(self, state: CombinedState):
        return self.env.normalise_state(state)
