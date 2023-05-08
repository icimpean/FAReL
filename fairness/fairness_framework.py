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
        env: The environment
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
                 threshold=None, get_individual=lambda state: state, similarity_metric=None, alpha=None,
                 group_notions=None, individual_notions=None, window=None,
                 visualise=False, visualise_cm=False, visualise_notions=False,
                 visualise_reward=False, visualise_hist=False):
        self.actions = actions
        self.window = window
        self.history = History(self.window)
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
        self.similarity_metric = similarity_metric
        self.alpha = alpha
        #
        self.threshold = threshold
        #
        self.group_notions = group_notions if group_notions is not None else ALL_GROUP_NOTIONS
        self.group_fairness = GroupFairness(actions)
        #
        self.individual_notions = individual_notions if individual_notions is not None else ALL_INDIVIDUAL_NOTIONS
        self.individual_fairness = IndividualFairness(actions)

        self.all_notions = self.group_notions + self.individual_notions
        self.exact_fairness = self._create_dictionaries()
        self.approx_fairness = self._create_dictionaries()
        self.reward_fairness = self._create_dictionaries()

    def _create_dictionaries(self):
        new_dictionary = {}
        for notion in self.all_notions:
            # Individual notions don't focus on a specific feature
            if isinstance(notion, IndividualNotion):
                new_dictionary[notion] = []
            else:
                new_dictionary[notion] = {str(attr): [] for attr in self.sensitive_attributes}
        return new_dictionary

    def update_history(self, state, action, true_action, score, reward):
        """Update the framework with a new observed tuple

        Args:
            state: The observed state
            action: The action taken in that state
            true_action: The correct action according to the ground truth of the problem
            score: The score assigned by the agent for the given state, or state-action pair
            reward: The reward received for the given action
        """
        self.history.update(state, action, true_action, score, reward)
        # Group notions
        for notion in self.group_notions:
            for sensitive_attribute in self.sensitive_attributes:
                (exact, approx), diff, (prob_sensitive, prob_other) = \
                    self.get_group_notion(notion, sensitive_attribute, self.threshold)
                notion_reward = -diff
                attr_id = str(sensitive_attribute)
                self.exact_fairness[notion][attr_id].append(exact)
                self.approx_fairness[notion][attr_id].append(approx)
                self.reward_fairness[notion][attr_id].append(notion_reward)

        # Individual notions
        for notion in self.individual_notions:
            (exact, approx), diff, (unsatisfied_individuals, unsatisfied_pairs, _) = \
                self.get_individual_notion(notion, self.get_individual, self.threshold,
                                           self.similarity_metric, self.alpha)
            notion_reward = -diff
            self.exact_fairness[notion].append(exact)
            self.approx_fairness[notion].append(approx)
            self.reward_fairness[notion].append(notion_reward)

    def get_group_notion(self, group_notion: GroupNotion, sensitive_attribute: SensitiveAttribute, threshold=None):
        """Get the given group notion"""
        return self.group_fairness.get_notion(group_notion, self.history, sensitive_attribute.feature,
                                              sensitive_attribute.sensitive_values, sensitive_attribute.other_values,
                                              threshold)

    def get_individual_notion(self, individual_notion: IndividualNotion, get_individual=lambda state: state,
                              threshold=None, similarity_metric=None, alpha=None):
        """Get the given individual notion"""
        return self.individual_fairness.get_notion(individual_notion, self.history, get_individual, threshold,
                                                   similarity_metric, alpha)


class ExtendedfMDP(object):
    """An extended job hiring fMDP, with a fairness framework"""
    def __init__(self, job_hiring_env, fairness_framework: FairnessFramework):
        # Super call
        super(ExtendedfMDP, self).__init__()
        #
        self.job_hiring_env = job_hiring_env
        self.fairness_framework = fairness_framework

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
        return self.job_hiring_env.reset()

    def step(self, action, scores=None):
        next_state, reward, done, info = self.job_hiring_env.step(action)

        true_action = info.get("true_action")
        if true_action is None:
            true_action = -1
        self.fairness_framework.update_history(self.job_hiring_env.previous_state, action, true_action, scores, reward)

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
                                                              self.fairness_framework.alpha)
            reward.append(diff)

        return next_state, reward, done, info

    def normalise_state(self, state: CombinedState):
        return self.job_hiring_env.normalise_state(state)
