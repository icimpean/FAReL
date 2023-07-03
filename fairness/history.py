from collections import deque
from enum import Enum
from typing import List

import numpy as np

from fairness import SensitiveAttribute, ConfusionMatrix


class History(object):
    """A history of encountered states and actions

    Attributes:
        env_actions: The actions taken in environment.
        window: (Optional) Use a sliding window for the stored history.
        store_interactions: (Optional) Store the full interactions instead of only the required information for
            fairness notions. Default: True.
        has_individual_fairness: (Optional) Is used to compute individual fairness notions. Default: True.
    """
    def __init__(self, env_actions, window=None, store_interactions=True, has_individual_fairness=True,
                 store_state_array=lambda state: state):
        self.env_actions = env_actions
        self.window = window
        self.store_interactions = store_interactions
        self.has_individual_fairness = has_individual_fairness
        self.store_state_array = store_state_array
        self.CM = ConfusionMatrix(self.env_actions)
        self.confusion_matrices = {}
        self.t = 0
        #
        if self.store_interactions or self.has_individual_fairness:
            self.states = deque(maxlen=self.window)
            self.actions = deque(maxlen=self.window)
            self.true_actions = deque(maxlen=self.window)
            self.scores = deque(maxlen=self.window)
            self.rewards = deque(maxlen=self.window)
            self.ids = deque(maxlen=self.window)
            self.feature_values = {}

    def update(self, episode, t, state, action, true_action, score, reward,
               sensitive_attributes: List[SensitiveAttribute]):
        """Update the history with a newly observed tuple

        Args:
            episode: The episode where the interaction took place
            t: The timestep of the interaction
            state: The observed state
            action: The action taken in that state
            true_action: The correct action according to the ground truth of the problem
            score: The score assigned by the agent for the given state, or state-action pair
            reward: The reward received for the given action.
            sensitive_attributes: The sensitive attributes for which to store computations.
        """
        self.t = t
        if self.store_interactions:
            self.states.append(state)
            self.actions.append(action)
            self.true_actions.append(true_action)
            self.scores.append(score)
            self.rewards.append(reward)
            self.ids.append(f"E{episode}T{t}")

            features = state.get_state_features(get_name=False, no_hist=True, individual_only=True)

            if len(self.feature_values) == 0:
                for feature in features:
                    self.feature_values[feature] = deque(maxlen=self.window)

            values = state.get_features(features)
            for feature, value in zip(features, values):
                if isinstance(value, Enum):
                    value = value.value
                self.feature_values[feature].append(value)

        else:
            if len(self.confusion_matrices) == 0:
                for sensitive_attribute in sensitive_attributes:
                    if self.window is None:
                        # Need 2 confusion matrices (4 values each) for sensitive & other values
                        #   => 8 possibilities for each interaction:
                        #       * 0 - 3: sensitive TN, FP, FN, TP
                        #       * 4 - 7: other TN, FP, FN, TP
                        self.confusion_matrices[sensitive_attribute] = [0 for _ in range(8)]
                    else:
                        self.confusion_matrices[sensitive_attribute] = deque(maxlen=self.window)

            # Add information to corresponding confusion matrices
            self._add_cm_value(state, action, true_action, score, reward, sensitive_attributes)

            if self.has_individual_fairness:
                # Store state array and other required info only
                self.states.append(self.store_state_array(state))
                self.actions.append(action)
                # self.true_actions.append(true_action)
                self.scores.append(score)
                # self.rewards.append(reward)
                self.ids.append(f"E{episode}T{t}")

    def get_history(self):
        """Get history"""
        return self.states, self.actions, self.true_actions, self.scores, self.rewards

    def _add_cm_value(self, state, action, true_action, score, reward, sensitive_attributes: List[SensitiveAttribute]):
        for sensitive_attribute in sensitive_attributes:
            # Group fairness:
            #   => 8 possibilities for each interaction:
            #       * 0 - 3: sensitive TN, FP, FN, TP
            #       * 4 - 7: other TN, FP, FN, TP
            feature_value = state[sensitive_attribute.feature]
            is_sensitive = sensitive_attribute.is_sensitive(feature_value)

            # TN TP
            if action == true_action:
                idx = 0 if action == 0 else 3
            # FP FN
            else:
                idx = 1 if action == 1 else 2
            # Other value
            if not is_sensitive:
                idx += 4

            if self.window is None:
                self.confusion_matrices[sensitive_attribute][idx] += 1
            else:
                self.confusion_matrices[sensitive_attribute].append(idx)

    def get_confusion_matrices(self, sensitive_attribute: SensitiveAttribute):
        """Get the confusion matrices for the given sensitive attribute"""
        if self.store_interactions:
            cm_sensitive = self.CM.confusion_matrix(self.states, self.actions, self.true_actions,
                                                    sensitive_attribute.feature, sensitive_attribute.sensitive_values)
            if sensitive_attribute.other_values is None:
                value = sensitive_attribute.sensitive_values
                excluded = True
            else:
                value = sensitive_attribute.other_values
                excluded = False
            cm_other = self.CM.confusion_matrix(self.states, self.actions, self.true_actions,
                                                sensitive_attribute.feature, value, excluded=excluded)
        else:
            if self.window is None:
                cm_sensitive = self.confusion_matrices[sensitive_attribute][:4]
                cm_other = self.confusion_matrices[sensitive_attribute][4:]
            else:
                unique, counts = np.unique(self.confusion_matrices[sensitive_attribute], return_counts=True)
                cm = [0 for _ in range(8)]
                for u, c in zip(unique, counts):
                    cm[u] = c
                cm_sensitive = cm[:4]
                cm_other = cm[4:]
            cm_sensitive = np.array(cm_sensitive).reshape((2, 2))
            cm_other = np.array(cm_other).reshape((2, 2))

        return cm_sensitive, cm_other
