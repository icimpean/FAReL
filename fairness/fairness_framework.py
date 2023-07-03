from typing import Union, List

from fairness import SensitiveAttribute
from fairness.group import GroupNotion, ALL_GROUP_NOTIONS
from fairness.group.group_fairness import GroupFairness
from fairness.history import History
from fairness.individual import ALL_INDIVIDUAL_NOTIONS, IndividualNotion
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
                 threshold=None, get_individual=lambda state: state, similarity_metric=None,
                 distance_metric="minkowski", alpha=None,
                 group_notions=None, individual_notions=None, window=None,
                 store_interactions=True, has_individual_fairness=True,
                 visualise=False, visualise_cm=False, visualise_notions=False,
                 visualise_reward=False, visualise_hist=False):
        self.actions = actions
        self.window = window
        self.store_interactions = store_interactions
        self.has_individual_fairness = has_individual_fairness
        self.history = History(actions, self.window, store_interactions=self.store_interactions,
                               has_individual_fairness=self.has_individual_fairness)
        self.visualise = visualise
        self.visualise_cm = visualise_cm
        self.visualise_notions = visualise_notions
        self.visualise_reward = visualise_reward
        self.visualise_hist = visualise_hist
        #
        self.sensitive_attributes = [sensitive_attributes] \
            if isinstance(sensitive_attributes, SensitiveAttribute) else sensitive_attributes
        #
        self.get_individual = get_individual
        self.distance_metric = distance_metric
        self.similarity_metric = similarity_metric
        self.alpha = alpha
        #
        self.threshold = threshold
        #
        self.group_notions = group_notions if group_notions is not None else ALL_GROUP_NOTIONS
        self.group_fairness = GroupFairness(actions)
        #
        self.individual_notions = individual_notions if individual_notions is not None else ALL_INDIVIDUAL_NOTIONS
        if not self.has_individual_fairness:
            self.individual_notions = []
        self.individual_fairness = IndividualFairness(actions)

        self.all_notions = self.group_notions + self.individual_notions

    def update_history(self, episode, t, state, action, true_action, score, reward):
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
        self.history.update(episode, t, state, action, true_action, score, reward, self.sensitive_attributes)

    def get_group_notion(self, group_notion: GroupNotion, sensitive_attribute: SensitiveAttribute, threshold=None):
        """Get the given group notion"""
        return self.group_fairness.get_notion(group_notion, self.history, sensitive_attribute, threshold)

    def get_individual_notion(self, individual_notion: IndividualNotion, get_individual=lambda state: state,
                              threshold=None, similarity_metric=None, alpha=None, distance_metric="minkowski"):
        """Get the given individual notion"""
        return self.individual_fairness.get_notion(individual_notion, self.history, get_individual, threshold,
                                                   similarity_metric, alpha, distance_metric)


class ExtendedfMDP(object):
    """An extended job hiring fMDP, with a fairness framework"""
    def __init__(self, env, fairness_framework: FairnessFramework):
        # Super call
        super(ExtendedfMDP, self).__init__()
        #
        self.env = env
        self.fairness_framework = fairness_framework
        if not self.fairness_framework.store_interactions and self.fairness_framework.has_individual_fairness:
            self.fairness_framework.history.store_state_array = env.state_to_array

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

    def reset(self):
        self._t += 1
        self._episode += 1
        return self.env.reset()

    def step(self, action, scores=None):
        next_state, reward, done, info = self.env.step(action)

        true_action = info.get("true_action")
        if true_action is None:
            true_action = -1
        self.fairness_framework.update_history(self._episode, self._t,
                                               self.env.previous_state, action, true_action, scores, reward)

        # Add fairness notions as additional rewards
        reward = [reward]
        # Group notions: For each sensitive attribute
        for sensitive_attribute in self.fairness_framework.sensitive_attributes:
            for notion in self.fairness_framework.group_notions:
                (exact, approx), diff, (prob_sensitive, prob_other) = \
                    self.fairness_framework.get_group_notion(notion, sensitive_attribute,
                                                             self.fairness_framework.threshold)
                reward.append(diff)
        # Individual notions:
        for notion in self.fairness_framework.individual_notions:
            (exact, approx), diff, (u_ind, u_pairs, U_diff) = \
                self.fairness_framework.get_individual_notion(notion, self.fairness_framework.get_individual,
                                                              self.fairness_framework.threshold,
                                                              self.fairness_framework.similarity_metric,
                                                              self.fairness_framework.alpha,
                                                              self.fairness_framework.distance_metric)
            reward.append(diff)

        self._t += 1

        return next_state, reward, done, info

    def normalise_state(self, state: CombinedState):
        return self.env.normalise_state(state)
