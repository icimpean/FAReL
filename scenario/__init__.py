from enum import Enum
from typing import List, Iterable
import types

import numpy as np
import pandas as pd
from numpy.random import Generator


class Feature(Enum):
    """The feature for a scenario"""
    pass


class State(object):
    """A sample from a scenario"""
    def __init__(self, sample):
        self.sample_dict = sample

    def __str__(self):
        features = [key.name if isinstance(key, Enum) else key for key in self.sample_dict.keys()]
        vals = self.to_array()
        lst = [f"{f}: {v}" for f, v in zip(features, vals)]
        s = ", ".join(lst)
        return f"<{s}>"

    def __getitem__(self, feature: Feature):
        """Get the value of a given feature"""
        return self.sample_dict[feature]

    def to_array(self, return_features=False):
        """Return the state as a numpy array of the values"""
        a = []
        features = []
        for k, v in self.sample_dict.items():
            val = v.value if isinstance(v, Enum) else v
            a.append(val)
            features.append(k)
        if return_features:
            return np.array(a, dtype=float), features
        else:
            return np.array(a, dtype=float)

    def to_vector_dict(self):
        """Return the state as a dictionary"""
        d = {}
        for k, v in self.sample_dict.items():
            if isinstance(v, Enum):
                d[k.name] = v.value
            elif isinstance(k, Enum):
                d[k.name] = v
            else:
                d[k] = v
        return d

    def get_state_features(self, get_name=False, no_hist=False):
        """Return the names of the features"""
        features = []
        for feature in self.sample_dict.keys():
            f = feature.name if get_name and isinstance(feature, Feature) else feature
            if no_hist and isinstance(feature, Feature):
                features.append(f)
            elif not no_hist:
                features.append(f)
        return features

    def get_features(self, features: List[Feature]):
        """Get the values of the requested features"""
        values = [self[feature] for feature in features]
        return values


class CombinedState(State):
    """The state of an individual and additional information on the context from the environment"""
    def __init__(self, sample_context, sample_individual):
        # Combined sample
        sample = sample_context.copy()
        sample.update(sample_individual)
        # Super call
        super(CombinedState, self).__init__(sample)
        #
        self.sample_context = sample_context
        self.sample_individual = sample_individual

    @staticmethod
    def from_array(array, context_features: List[Feature], individual_features: List[Feature]):
        """Return a state from an array, given the enumeration over the features"""
        sample_context = {f: array[f.value] for f in context_features}
        sample_individual = {f: array[f.value] for f in individual_features}
        return CombinedState(sample_context, sample_individual)

    def _get_sample(self, context_only=False, individual_only=False):
        get_all = context_only is False and individual_only is False
        assert get_all or context_only != individual_only
        if get_all:
            return self.sample_dict
        elif context_only:
            return self.sample_context
        else:
            return self.sample_individual

    def to_array(self, return_features=False, context_only=False, individual_only=False):
        """Return the state as a numpy array of the values"""
        a = []
        features = []
        for k, v in self._get_sample(context_only, individual_only).items():
            val = v.value if isinstance(v, Enum) else v
            a.append(val)
            features.append(k)
        # return np.array(a, dtype=object)
        if return_features:
            return np.array(a, dtype=float), features
        else:
            return np.array(a, dtype=float)

    def to_vector_dict(self, context_only=False, individual_only=False):
        """Return the state as a dictionary"""
        d = {}
        for k, v in self._get_sample(context_only, individual_only).items():
            if isinstance(v, Enum):
                d[k.name] = v.value
            elif isinstance(k, Enum):
                d[k.name] = v
            else:
                d[k] = v
        return d

    def get_state_features(self, get_name=False, no_hist=False, context_only=False, individual_only=False):
        """Return the names of the features"""
        features = []
        for feature in self._get_sample(context_only, individual_only).keys():
            f = feature.name if get_name and isinstance(feature, Feature) else feature
            if no_hist and isinstance(feature, Feature):
                features.append(f)
            elif not no_hist:
                features.append(f)
        return features


class FeatureBias(object):
    """Bias on goodness score for a given feature"""
    def __init__(self, features, feature_values, bias):
        self.features = features
        self.feature_values = feature_values
        self.bias = bias

    def get_bias(self, state: State):
        """Get the amount of bias to add to the goodness score for the given state"""
        # features is a single feature
        if not isinstance(self.features, Iterable):
            self.features = [self.features]
            self.feature_values = [self.feature_values]

        additions = []
        for feature, feature_value in zip(self.features, self.feature_values):
            # feature_value is a list of allowed values
            if isinstance(feature_value, Iterable) and not isinstance(feature_value, str):
                add_bias = lambda v: v in feature_value
            # The feature_value is a function
            elif isinstance(feature_value, types.FunctionType):
                add_bias = feature_value
            # Only if equal to the given feature value
            else:
                add_bias = lambda v: v == feature_value
            additions.append(add_bias(state[feature]))

        # Add bias only if all conditions are met
        if all(additions):
            return self.bias
        else:
            return 0


class Scenario(object):
    """A scenario for generating data for a given setting"""
    def __init__(self, seed=None):
        # The random generator for the scenario
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

    def generate_sample(self):
        """Generate a sample"""
        raise NotImplementedError

    def calc_goodness(self, sample: State):
        """Calculate the goodness score for a given sample"""
        raise NotImplementedError

    def calculate_rewards(self, sample: State, goodness):
        """Calculate the rewards for taking different actions in the current state, given the goodness score"""
        raise NotImplementedError

    def step(self, action):
        """Sample a state and return the rewards for corresponding actions in the scenario"""
        state = self.generate_sample()
        goodness = self.calc_goodness(state)
        rewards = self.calculate_rewards(state, goodness)
        return state, rewards

    def create_dataset(self, num_samples, show_goodness=False, show_rewards=False, rounding=None):
        """Generate a dataset with the given number of samples."""
        dataset = []
        features = None
        for t in range(num_samples):
            sample = self.generate_sample()
            entry = list(sample.to_array())
            if features is None:
                features = sample.get_state_features()
                if show_goodness:
                    features.append("goodness")
                if show_rewards:
                    features.append("rewards")
            if show_goodness or show_rewards:
                goodness = self.calc_goodness(sample)
                if show_goodness:
                    entry.append(goodness)
                if show_rewards:
                    rewards = self.calculate_rewards(sample, goodness)
                    new_rewards = {k.name: (v if rounding is None else round(v, rounding)) for k, v in rewards.items()}
                    entry.append(new_rewards)
            dataset.append(np.array(entry, dtype=object))
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 120)
        dataset = pd.DataFrame(np.array(dataset), columns=features)
        return dataset
