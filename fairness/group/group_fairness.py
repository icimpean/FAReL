from fairness import SensitiveAttribute, ppv, npv, fpr, tpr
from fairness.group import GroupFairnessBase, GroupNotion
from fairness.history import History


class GroupFairness(GroupFairnessBase):
    """A collection of fairness notions w.r.t. protected and unprotected groups.

        Attributes:
            actions: A list of enumerations, representing the actions to check fairness for.

        # TODO: unimplemented (group) fairness notions:
            #   Conditional Statistical Parity: requires explainable features e
            #   Balance for Positive Class: requires score s
            #   Balance for Negative Class: requires score s
            #   Calibration: requires score s
            #   Well-calibration: requires score s

        # TODO: multiple actions instead of just binary
        """

    def __init__(self, actions):
        # Super call
        super(GroupFairness, self).__init__(actions)
        # Mapping from enumeration to fairness method
        self._map = {
            GroupNotion.StatisticalParity: self.statistical_parity,
            GroupNotion.StatisticalParity_t: self.statistical_parity,
            GroupNotion.EqualOpportunity: self.equal_opportunity,
            GroupNotion.EqualOpportunity_t: self.equal_opportunity,
            GroupNotion.PredictiveEquality: self.predictive_equality,
            GroupNotion.PredictiveEquality_t: self.predictive_equality,
            GroupNotion.EqualizedOdds: self.equalized_odds,
            GroupNotion.EqualizedOdds_t: self.equalized_odds,
            GroupNotion.OverallAccuracyEquality: self.overall_accuracy_equality,
            GroupNotion.OverallAccuracyEquality_t: self.overall_accuracy_equality,
            GroupNotion.PredictiveParity: self.predictive_parity,
            GroupNotion.PredictiveParity_t: self.predictive_parity,
            GroupNotion.ConditionalUseAccuracyEquality: self.conditional_use_accuracy_equality,
            GroupNotion.ConditionalUseAccuracyEquality_t: self.conditional_use_accuracy_equality,
            GroupNotion.TreatmentEquality: self.treatment_equality,
            GroupNotion.TreatmentEquality_t: self.treatment_equality,
        }

    def get_notion(self, group_notion: GroupNotion, history: History,
                   sensitive_attribute: SensitiveAttribute, threshold=None):
        # noinspection PyArgumentList
        return self._map[group_notion](history, sensitive_attribute, threshold)

    def statistical_parity(self, history: History, sensitive_attribute: SensitiveAttribute, threshold=None):
        """Predicted acceptance rates for both protected and unprotected groups should be equal.

        P(y_pred = 1 | feature = sensitive_value)
                ==
        P(y_pred = 1 | feature = other_value)

        Using confusion matrix:
        (TP + FP) / (TP + FP + FN + TN) should be equal for both groups.
        """
        def _calc(confusion_matrix):
            TN, FP, FN, TP = confusion_matrix.ravel()
            return (TP + FP) / (TP + FP + FN + TN)

        return self._fairness_notion(history, _calc, sensitive_attribute, threshold)

    def equal_opportunity(self, history: History, sensitive_attribute: SensitiveAttribute, threshold=None):
        """Select equal proportions of individuals from the qualified fraction of each group.

        P(y_pred = 1 | y_true = 1, feature = sensitive_value)
                ==
        P(y_pred = 1 | y_true = 1, feature = other_value)

        Using confusion matrix:
        TPR = TP / (TP + FN) should be equal for both groups.
        """

        return self._fairness_notion(history, tpr, sensitive_attribute, threshold)

    def predictive_equality(self, history: History, sensitive_attribute: SensitiveAttribute, threshold=None):
        """Select equal proportions of individuals from the unqualified fraction of each group.

        P(y_pred = 1 | y_true = 0, feature = sensitive_value)
                ==
        P(y_pred = 1 | y_true = 0, feature = other_value)

        Using confusion matrix:
        FPR = FP / (FP + TN) should be equal for both groups.
        """

        return self._fairness_notion(history, fpr, sensitive_attribute, threshold)

    def equalized_odds(self, history: History, sensitive_attribute: SensitiveAttribute, threshold=None):
        """Prediction is conditionally independent of the sensitive attribute, given the actual outcome.

        P(y_pred = 1 | y_true = y, feature = sensitive_value)
                ==
        P(y_pred = 1 | y_true = y, feature = other_value)
                for y in {0, 1}

        Using confusion matrix:
        Both groups should have an equal TPR and an equal FPR.
        """
        # equalized_odds == equal_opportunity AND predictive_equality
        def _calc(confusion_matrix):
            return fpr(confusion_matrix), tpr(confusion_matrix)  # (y = reject, y = hire)

        return self._fairness_notion(history, _calc, sensitive_attribute, threshold)

    def overall_accuracy_equality(self, history: History, sensitive_attribute: SensitiveAttribute, threshold=None):
        """Overall accuracy is the same for both groups.

        P(y_pred = y_true | feature = sensitive_value)
                ==
        P(y_pred = y_true | feature = other_value)

        Using confusion matrix:
        (TP + TN) / (TP + FN + FP + TN) should be equal for both groups.
        """
        def _calc(confusion_matrix):
            TN, FP, FN, TP = confusion_matrix.ravel()
            return (TP + TN) / (TP + FN + FP + TN)

        return self._fairness_notion(history, _calc, sensitive_attribute, threshold)

    def predictive_parity(self, history: History, sensitive_attribute: SensitiveAttribute, threshold=None):
        """Select equal proportions of individuals from the qualified fraction of each group.

        P(y_pred = 1 | y_true = 1, feature = sensitive_value)
                ==
        P(y_pred = 1 | y_true = 1, feature = other_value)

        Using confusion matrix:
        PPV = TP / (TP + FP) should be equal for both groups.
        """

        return self._fairness_notion(history, ppv, sensitive_attribute, threshold)

    def conditional_use_accuracy_equality(self, history: History, sensitive_attribute: SensitiveAttribute,
                                          threshold=None):
        """The probability of subjects with positive predictive value to truly belong to the positive class
        and the probability of subjects with negative predictive value to truly belong to the negative class
        should be the same for both groups.

        P(y_pred = y | y_true = 1, feature = sensitive_value)
                ==
        P(y_pred = 1 | y_true = 1, feature = other_value)

        Using confusion matrix:
        PPV and NPV should be equal for both groups.
        """
        def _calc(confusion_matrix):
            return npv(confusion_matrix), ppv(confusion_matrix)  # (y = reject, y = hire)

        return self._fairness_notion(history, _calc, sensitive_attribute, threshold)

    def treatment_equality(self, history: History, sensitive_attribute: SensitiveAttribute, threshold=None):
        """Ratio of FNs and FPs is the same for both groups.

        Using confusion matrix:
        FN / FP should be equal for both groups.
        """
        def _calc(confusion_matrix):
            TN, FP, FN, TP = confusion_matrix.ravel()
            return FN / FP

        return self._fairness_notion(history, _calc, sensitive_attribute, threshold)
