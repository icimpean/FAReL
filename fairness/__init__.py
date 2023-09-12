import types
from enum import Enum
from typing import Iterable

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix


class ConfusionMatrix(object):
    """The class for calculating the confusion matrices for given data"""
    def __init__(self, actions):
        self.actions = actions
        self.action_order = [a.value for a in self.actions]
        self.labels = [a.name for a in self.actions]

    @staticmethod
    def _filter(X, y_pred, y_true, feature, values, excluded=False):
        """Filter the given observations and their true and predicted values over the given feature value(s).

        If values is a single value, only samples with that value as a feature are kept.
        If values is a list of values, samples with any one of the given values are kept.
        If excluded, all data not containing the given values is kept.
        """
        new_X = []
        new_y_pred = []
        new_y_true = []

        # Keep all samples with value in feature_val if feature_val is a list
        if isinstance(values, Iterable) and not isinstance(values, str):
            keep = lambda v: v in values
        # The values is a function
        elif isinstance(values, types.FunctionType):
            keep = values
        # Only keep samples with the given feature value
        else:
            keep = lambda v: v == values

        # Gather relevant samples
        for x, yp, yt in zip(X, y_pred, y_true):
            has_value = keep(x[feature])
            if excluded:
                has_value = not has_value
            if has_value:
                new_X.append(x)
                new_y_pred.append(yp)
                new_y_true.append(yt)

        return new_X, new_y_pred, new_y_true

    def confusion_matrix(self, X, y_pred, y_true, feature=None, feature_val=None, action_order=None, normalize="all",
                         excluded=False):
        """Create a confusion matrix for the given observations and predictions.

        Args:
            X: The observations.
            y_pred: The predictions of a model.
            y_true: The ground truth values.
            feature: (Optional) The feature to filter the observations X on.
                If supplied, requires feature_val to also be given.
            feature_val: (Optional) The value(s) to indicate the observations to consider.
                If supplied, requires feature to also be given.
            action_order: (Optional) The order of actions for the matrix.
            normalize: (Optional) Whether or not to normalise the confusion matrix.
            excluded: (Optional) Keep samples where the feature doesn't have (any of) the given feature_val.

        Returns:
            A confusion matrix.
        """

        # Sort the confusion matrix actions by the given labels or the instance's labels
        if action_order is None:
            action_order = self.action_order

        # Confusion matrix over all samples
        if feature is None:
            pass
        # Confusion matrix over samples with given feature value
        elif feature is not None and feature_val is not None:
            X, y_pred, y_true = self._filter(X, y_pred, y_true, feature, feature_val, excluded)
        # Wrong argument combination
        else:
            raise ValueError(f"Expecting both a feature index and value if feature isn't None. "
                             f" feature: {feature}, feature_val: {feature_val}")

        # Create the confusion matrix
        cm = confusion_matrix(y_true, y_pred, normalize=normalize, labels=action_order)
        return cm

    def multilabel_confusion_matrix(self, X, y_pred, y_true, feature=None, feature_val=None,
                                    action_order=None, normalize="all"):
        """Create an action-wise confusion matrix for the given observations and predictions.

        From sklearn.metrics.multilabel_confusion_matrix:
            "Multiclass data will be treated as if binarized under a one-vs-rest transformation."

        Args:
            X: The observations.
            y_pred: The predictions of a model.
            y_true: The ground truth values.
            feature: (Optional) The feature to filter the observations X on.
                If supplied, requires feature_val to also be given.
            feature_val: (Optional) The value(s) to indicate the observations to consider.
                If supplied, requires feature to also be given.
            action_order: (Optional) The order of actions for the matrix.
            normalize: (Optional) Whether or not to normalise the confusion matrix.

        Returns:
            A list of confusion matrices, binarized per action under a one-vs-rest transformation.
        """
        # Sort the confusion matrix actions by the given labels or the instance's labels
        if action_order is None:
            action_order = self.action_order

        # Confusion matrix over all samples
        if feature is None:
            pass
        # Confusion matrix over samples with given feature value
        elif feature is not None and feature_val is not None:
            X, y_pred, y_true = self._filter(X, y_pred, y_true, feature, feature_val)
        # Wrong argument combination
        else:
            raise ValueError(f"Expecting both a feature index and value if feature isn't None. "
                             f" feature: {feature}, feature_val: {feature_val}")

        # Create the confusion matrices
        cm = multilabel_confusion_matrix(y_true, y_pred, labels=action_order)
        # Normalize (not done by multilabel_confusion_matrix function)
        if normalize == "all":
            num_samples = len(X)
            cm /= num_samples
        return cm


def ppv(cm):
    """Positive Predictive Value (Precision)
    PPV = TP / (TP + FP)"""
    tn, fp, fn, tp = cm.ravel()
    return tp / (tp + fp)


def npv(cm):
    """Negative Predictive Value.
    NPV = TN / (TN + FN)"""
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fn)


def tpr(cm):
    """True Positive Rate (Sensitivity, Recall)
     TPR = TP / (TP + FN)"""
    tn, fp, fn, tp = cm.ravel()
    return tp / (tp + fn)


def fpr(cm):
    """False Positive Rate (Model Error)
     FPR = FP / (FP + TN)"""
    tn, fp, fn, tp = cm.ravel()
    return fp / (fp + tn)


class SensitiveAttribute(object):
    """The class defining a sensitive attribute and specifying the sensitive value.

    Attributes:
        feature: The sensitive attribute to check fairness for.
        sensitive_values: The sensitive values for that attribute, indicating minorities or instances for which there
            might be a negative bias.
        other_values: (Optional) The other values to compare the sensitive values to. If None, this contains all other
            possible values not specified in sensitive_values.
    """
    def __init__(self, feature, sensitive_values, other_values=None):
        self.feature = feature
        self.sensitive_values = sensitive_values
        self.other_values = other_values

    def __str__(self):
        # other = "" if self.other_values is None else f" =/= {self.other_values}"
        if isinstance(self.sensitive_values, Enum):
            values = self.sensitive_values.name
        elif isinstance(self.sensitive_values, Iterable) and not isinstance(self.sensitive_values, str):
            values = ", ".join([v.name if isinstance(v, Enum) else v for v in self.sensitive_values])
        elif isinstance(self.sensitive_values, types.FunctionType):
            values = self.sensitive_values.__name__
        else:
            values = self.sensitive_values

        return f"Feature<{self.feature.name} = {values}>"

    def is_sensitive(self, value):
        """Is the given feature value a sensitive one."""
        if isinstance(self.sensitive_values, Iterable) and not isinstance(self.sensitive_values, str):
            return value in self.sensitive_values
        elif isinstance(self.sensitive_values, types.FunctionType):
            return self.sensitive_values(value)
        else:
            return value == self.sensitive_values


class CombinedSensitiveAttribute(SensitiveAttribute):
    """The class defining a combination of multiple sensitive attributes, for minorities or subgroup fairness"""
    def __init__(self, features, sensitive_values, other_values=None):
        # Super call
        super(CombinedSensitiveAttribute, self).__init__(features, sensitive_values, other_values)

    def __str__(self):
        string = ""
        for feature, sensitive_values, other_values in zip(self.feature, self.sensitive_values, self.other_values):
            # other = "" if self.other_values is None else f" =/= {self.other_values}"
            if isinstance(sensitive_values, Enum):
                values = sensitive_values.name
            elif isinstance(sensitive_values, Iterable) and not isinstance(sensitive_values, str):
                values = ", ".join([v.name if isinstance(v, Enum) else v for v in sensitive_values])
            elif isinstance(sensitive_values, types.FunctionType):
                values = sensitive_values.__name__
            else:
                values = sensitive_values
            string += "~AND~" + f"Feature<{feature.name} = {values}>"

        return string

    def is_sensitive(self, values):
        """Is the given feature value a sensitive one."""
        sensitive = []
        for value, sensitive_values in zip(values, self.sensitive_values):
            if isinstance(sensitive_values, Iterable) and not isinstance(sensitive_values, str):
                sensitive.append(value in sensitive_values)
            elif isinstance(sensitive_values, types.FunctionType):
                sensitive.append(sensitive_values(value))
            else:
                sensitive.append(value == sensitive_values)

        return all(sensitive)
