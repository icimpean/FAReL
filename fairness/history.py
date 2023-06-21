from collections import deque
from enum import Enum

import numpy as np


class History(object):
    """A history of encountered states and actions"""
    def __init__(self, window=None):
        self.window = window
        self.states = deque(maxlen=self.window)
        self.actions = deque(maxlen=self.window)
        self.true_actions = deque(maxlen=self.window)
        self.scores = deque(maxlen=self.window)
        self.rewards = deque(maxlen=self.window)
        self.ids = deque(maxlen=self.window)
        self.feature_values = {}
        # self.histograms = {}

    def update(self, episode, t, state, action, true_action, score, reward):
        """Update the history with a newly observed tuple

        Args:
            episode: The episode where the interaction took place
            t: The timestep of the interaction
            state: The observed state
            action: The action taken in that state
            true_action: The correct action according to the ground truth of the problem
            score: The score assigned by the agent for the given state, or state-action pair
            reward: The reward received for the given action
        """
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

        # if len(self.states) > 1:
        #     for feature in features:
        #         values = self.feature_values[feature]
        #         uniques = len(np.unique(values))
        #         if uniques < 5:
        #             bins = uniques
        #         else:
        #             bins = "fd"
        #         print("bins:", bins)
        #         counts, bin_edges = np.histogram(values, bins=bins, density=True)
        #         counts = counts * np.diff(bin_edges)
        #         self.histograms[feature] = (counts, bin_edges)

    def get_history(self):
        """Get history"""
        return self.states, self.actions, self.true_actions, self.scores, self.rewards
