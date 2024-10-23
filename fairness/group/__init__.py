from enum import Enum, auto
import numpy as np

from fairness import ConfusionMatrix
from fairness.history import History


class GroupNotion(Enum):
    """Enumeration for group fairness notions"""
    StatisticalParity = auto()
    EqualOpportunity = auto()
    OverallAccuracyEquality = auto()
    PredictiveParity = auto()
    PredictiveEquality = auto()
    EqualizedOdds = auto()
    ConditionalUseAccuracyEquality = auto()
    TreatmentEquality = auto()
    #
    StatisticalParity_t = auto()
    EqualOpportunity_t = auto()
    OverallAccuracyEquality_t = auto()
    PredictiveParity_t = auto()
    PredictiveEquality_t = auto()
    EqualizedOdds_t = auto()
    ConditionalUseAccuracyEquality_t = auto()
    TreatmentEquality_t = auto()


ALL_GROUP_NOTIONS = list(GroupNotion)
TIMESTEP_GROUP_NOTIONS = [g for g in GroupNotion if g.name.endswith("_t")]


class GroupFairnessBase(object):
    """Base class with helping functions for group fairness."""
    def __init__(self, actions):
        self.actions = actions
        self.action_order = [a.value for a in self.actions]
        self.labels = [a.name for a in self.actions]

    @staticmethod
    def _is_value(x, feature, value):
        """x's feature must be equal to value or be one of the values in the list value"""
        if isinstance(value, list):
            return x[feature] in value
        else:
            return x[feature] == value

    def _is_other_value(self, x, feature, sensitive_value, other_value):
        """x's feature must be equal to other_value via _is_value or must be anything but sensitive_value if None."""
        if other_value is None:
            return not self._is_value(x, feature, sensitive_value)
        else:
            return self._is_value(x, feature, other_value)

    @staticmethod
    def _is_fair(prob_0, prob_1, threshold):
        """Return whether or not there is exact and approximate fairness"""
        exact = False
        approx = False
        diff = 0
        # Multiple probabilities given
        if isinstance(prob_0, tuple) and isinstance(prob_1, tuple):
            is_exact = [p0 == p1 for p0, p1 in zip(prob_0, prob_1)]
            exact = all(is_exact)
            diffs = [abs(p0 - p1) for p0, p1 in zip(prob_0, prob_1)]
            diff = max(diffs)
            if threshold is not None:
                is_approx = [d <= threshold for d in diffs]
                approx = all(is_approx)
        else:
            exact = prob_0 == prob_1
            diff = abs(prob_0 - prob_1)
            if threshold is not None:
                approx = diff <= threshold

        # Return exact and approximate fairness
        return exact, approx, diff

    def _fairness_notion(self, history: History, calc_function, sensitive_attribute, threshold):
        """Calculate the given fairness notion"""
        # TODO:
        import warnings
        warnings.filterwarnings("ignore")

        # Confusion matrices
        cm_sensitive, cm_other = history.get_confusion_matrices(sensitive_attribute)

        # Calculate probabilities/ratios of the two groups
        prob_sensitive = calc_function(cm_sensitive)
        prob_other = calc_function(cm_other)

        exact, approx, diff = self._is_fair(prob_sensitive, prob_other, threshold)

        # Set NaN values (missing groups so fairness cannot be calculated) to unfair
        if (np.sum(cm_sensitive) < 1) and (np.sum(cm_other) < 1):
            diff = 1.0
        diff = np.nan_to_num(-diff)

        # Return fairness
        return (exact, approx), diff, (prob_sensitive, prob_other)
