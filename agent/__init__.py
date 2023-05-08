

class Agent(object):
    """An agent capable of interacting with an environment"""
    def __init__(self):
        pass

    def step(self, t, state, rewards=None, use_best_arm=False):
        pass

    def init_train(self, t, states, actions, true_actions, rewards):
        pass
