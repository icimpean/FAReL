from enum import Enum, auto


class IndividualNotion(Enum):
    """Enumeration for group fairness notions"""
    IndividualFairness = auto()
    WeaklyMeritocratic = auto()
    ConsistencyScoreComplement = auto()
    # Incremental nearest neighbour-based implementations
    # IndividualFairness_INN = auto()
    # WeaklyMeritocratic_INN = auto()
    ConsistencyScoreComplement_INN = auto()
    #
    IndividualFairness_t = auto()
    # WeaklyMeritocratic_t = auto()
    # ConsistencyScoreComplement_t = auto()
    # ConsistencyScoreComplement_INN_t = auto()


ALL_INDIVIDUAL_NOTIONS = list(IndividualNotion)
TIMESTEP_INDIVIDUAL_NOTIONS = [i for i in IndividualNotion if i.name.endswith("_t")]


class IndividualFairnessBase(object):
    """Base class with helping functions for group fairness."""
    def __init__(self, actions):
        self.actions = actions
        self.action_order = [a.value for a in self.actions]
        self.labels = [a.name for a in self.actions]

