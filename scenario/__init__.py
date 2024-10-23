from enum import Enum
from typing import List, Iterable, Union
import types

import numpy as np
import pandas as pd
from numpy.random import Generator
from scipy.linalg import norm


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

    def __getitem__(self, feature: Union[Feature, List[Feature]]):
        """Get the value of a given feature"""
        if isinstance(feature, Feature):
            return self.sample_dict[feature]
        else:
            return [self.sample_dict[f] for f in feature]

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

    def get_features(self, features: List[Feature], as_array=False):
        """Get the values of the requested features"""
        values = [self[feature] for feature in features]
        if as_array:
            values = [v.value if isinstance(v, Enum) else v for v in values]
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
        self._features = None

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
    def __init__(self, features, nominal_features=(), numerical_features=(), exclude_from_distance=(), seed=None):
        # The random generator for the scenario
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        #
        self.features = features
        self.nominal_features = nominal_features
        self.numerical_features = numerical_features
        self.exclude_from_distance = exclude_from_distance
        #
        self.previous_state = None
        #
        self._goodness = None
        self._rewards = None
        #
        self._features = None
        self._nom_indices = None
        self._num_indices = None
        #
        self._features_i = None
        self._nom_indices_i = None
        self._num_indices_i = None
        self.indices_i = None

    def generate_sample(self):
        """Generate a sample"""
        raise NotImplementedError

    def calc_goodness(self, sample: State):
        """Calculate the goodness score for a given sample"""
        raise NotImplementedError

    def calculate_rewards(self, sample: State, goodness):
        """Calculate the rewards for taking different actions in the current state, given the goodness score"""
        raise NotImplementedError

    def init_features(self, state):
        if self._features is None:
            # Full state
            self._features = state.sample_dict.keys()
            self._nom_indices = [i for i, f in enumerate(self._features)
                                 if (f in self.nominal_features) and (f not in self.exclude_from_distance)]
            self._num_indices = [i for i, f in enumerate(self._features)
                                 if (f in self.numerical_features) and (f not in self.exclude_from_distance)]
            # Individual state
            self._features_i = state.sample_individual.keys()
            self._nom_indices_i = [i for i, f in enumerate(self._features_i)
                                   if (f in self.nominal_features) and (f not in self.exclude_from_distance)]
            self._num_indices_i = [i for i, f in enumerate(self._features_i)
                                   if (f in self.numerical_features) and (f not in self.exclude_from_distance)]
            self.indices_i = [i for i, f in enumerate(self._features_i)
                              if f not in self.exclude_from_distance]

    def step(self, action):
        """Sample a state and return the rewards for corresponding actions in the scenario"""
        self.previous_state = self.generate_sample()
        self._goodness = self.calc_goodness(self.previous_state)
        self._rewards = self.calculate_rewards(self.previous_state, self._goodness)

        return self.previous_state, self._rewards

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

    def similarity_metric(self, state1: Union[CombinedState, np.ndarray], state2: Union[CombinedState, np.ndarray],
                          distance="HMOM", alpha=1.0, exp=True):
        try:
            return distance(state1, state2)
        except TypeError:
            if distance.startswith("H") and distance.endswith("OM"):
                return self.H_OM_distance(state1, state2, distance, alpha, exp)
            # Minkowski distance between two 1-D arrays (minkowski)
            elif distance == "minkowski":
                d = self.minkowski_metric(state1, state2, p=2, w=None)  # TODO: absract p, w together with consistency score
                return d
            elif distance == "braycurtis":
                d = self.braycurtis_metric(state1, state2, w=None)  # TODO: absract w together with consistency score
                return d
            else:
                raise ValueError(f"Expected one of [HEOM, HMOM, minkowski, braycurtis]. Got: {distance}")

    def H_OM_distance(self, state1: Union[CombinedState, np.ndarray], state2: Union[CombinedState, np.ndarray],
                      distance="HMOM", alpha=1.0, exp=True):
        try:
            # Extract features corresponding to given indices
            num1 = state1[self._num_indices_i]
            nom1 = state1[self._nom_indices_i]
            num2 = state2[self._num_indices_i]
            nom2 = state2[self._nom_indices_i]
        except KeyError:
            norm_state1 = self._normalise_features(state1)
            norm_state2 = self._normalise_features(state2)
            num1 = norm_state1[self._num_indices]
            nom1 = norm_state1[self._nom_indices]
            num2 = norm_state2[self._num_indices]
            nom2 = norm_state2[self._nom_indices]

        # Heterogeneous Euclidean-Overlap Metric (HEOM)
        if distance == 'HEOM':
            # d = np.sum(np.abs(num1 - num2)) + np.sum(nom1 != nom2)
            d = sum(np.abs(num1 - num2)) + sum(nom1 != nom2)
        # Heterogeneous Manhattan-Overlap Metric (HMOM)
        elif distance == 'HMOM':
            # d = np.sum((num1 - num2) ** 2) + np.sum(nom1 != nom2)
            diff = (num1 - num2)
            d = sum(diff * diff) + sum(nom1 != nom2)
        else:
            raise ValueError(f"Expected distance: HEOM or HMOM. Got: {distance}")
        return np.exp(-alpha * d) if exp else d

    def minkowski_metric(self, state1: Union[CombinedState, np.ndarray], state2: Union[CombinedState, np.ndarray],
                         p=2, w=None):
        # Normalise if it hasn't happened yet
        # if isinstance(state1, CombinedState):
        #     norm1 = self._normalise_features(state1, indices=self.indices_i)
        #     norm2 = self._normalise_features(state2, indices=self.indices_i)
        # else:
        #     norm1 = state1[self.indices_i]
        #     norm2 = state2[self.indices_i]
        # return minkowski(norm1, norm2, p=p, w=w)
        try:
            norm1 = state1[self.indices_i]
            norm2 = state2[self.indices_i]
        except KeyError:
            norm1 = self._normalise_features(state1, indices=self.indices_i)
            norm2 = self._normalise_features(state2, indices=self.indices_i)
        # Based on from scipy.spatial.distance.minkowski
        if p <= 0:
            raise ValueError("p must be greater than 0")
        u_v = norm1 - norm2
        if w is not None:
            if p == 1:
                root_w = w
            elif p == 2:
                # better precision and speed
                root_w = np.sqrt(w)
            elif p == np.inf:
                root_w = (w != 0)
            else:
                root_w = np.power(w, 1 / p)
            u_v = root_w * u_v
        dist = norm(u_v, ord=p)
        return dist

    def braycurtis_metric(self, state1: Union[CombinedState, np.ndarray], state2: Union[CombinedState, np.ndarray],
                          w=None):
        # Normalise if it hasn't happened yet
        # if isinstance(state1, CombinedState):
        #     norm1 = self._normalise_features(state1, indices=self.indices_i)
        #     norm2 = self._normalise_features(state2, indices=self.indices_i)
        # else:
        #     norm1 = state1[self.indices_i]
        #     norm2 = state2[self.indices_i]
        try:
            norm1 = state1[self.indices_i]
            norm2 = state2[self.indices_i]
        except KeyError:
            norm1 = self._normalise_features(state1, indices=self.indices_i)
            norm2 = self._normalise_features(state2, indices=self.indices_i)

        # Based on from scipy.spatial.distance.braycurtis
        l1_diff = abs(norm1 - norm2)
        l1_sum = abs(norm1 + norm2)
        if w is not None:
            l1_diff = w * l1_diff
            l1_sum = w * l1_sum

        return sum(l1_diff) / sum(l1_sum)

    def _normalise_features(self, state: Union[CombinedState, np.ndarray], features: List[Feature] = None,
                            indices=None):
        raise NotImplementedError

    def state_to_array(self, state: Union[CombinedState, np.ndarray]):
        """Used by history to store individuals"""
        # If state is an array, assume it is preprocessed as needed
        s = state
        if isinstance(state, CombinedState):
            s = self._normalise_features(state, self._features_i)
        return s

    def normalise_state(self, state: CombinedState):
        """Used by the RL agent interacting with the environment"""
        raise NotImplementedError

    def get_all_entities_in_state(self, state: CombinedState, action, true_action, score, reward):
        """Returns a list of combined states, indicating all the entities encountered at timestep t"""
        raise NotImplementedError
