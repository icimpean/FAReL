import numpy as np


##########
# Policy #
##########
class Policy(object):
    """The abstract class for an action selection policy.

    Attributes:
        name: The name of the policy.
        value_name: (Optional) The name of the value the policy uses,
            or None if not applicable.
    """
    def __init__(self, name, value_name=None):
        self.name = name
        self.value_name = value_name

    def select_action(self, q_values, step, episode):
        """The action selection method for the current policy.

        Args:
            q_values: The q-values to select an action with.
            step: The time step at which the action is selected.
            episode: The episode at which the action is selected.

        Returns:
            action, an action as selected from the q-values.
        """
        raise NotImplementedError

    def get_distribution(self, q_values, step, episode):
        """The method to get a probability distribution from the given q-values.

        Args:
            q_values: The q-values to select an action with.
            step: The time step at which the action is selected.
            episode: The episode at which the action is selected.

        Returns:
            distribution, a probability distribution for the q-values.
        """
        raise NotImplementedError

    @staticmethod
    def _check_q_values_dimension(q_values):
        # Expecting q-values to be 1-dimensional: not an array/list of multiple q-values
        assert q_values.ndim == 1, f"Expected 1-dimensional q-values. Found {q_values.ndim} dimensions: {q_values}"

    def __str__(self):
        return f"{self.name} Policy"


class GreedyPolicy(Policy):
    """A greedy policy.

    Attributes:
        None.
    """
    def __init__(self):
        # Super call
        super(GreedyPolicy, self).__init__(name="Greedy")

    @staticmethod
    def select(q_values):
        # Greedy selection
        return np.argmax(q_values)

    def select_action(self, q_values, step, episode):
        # Check q-values dimension
        self._check_q_values_dimension(q_values)
        # Greedy selection
        return self.select(q_values)

    def get_distribution(self, q_values, step, episode):
        # The maximum q-values have probability 1
        new_q_values = np.zeros_like(q_values)
        greedy_actions = np.argmax(q_values, axis=1)
        xs = np.arange(new_q_values.shape[0])
        new_q_values[xs, greedy_actions] = 1
        return new_q_values


class RandomPolicy(Policy):
    """A random policy.

    Attributes:
        None.
    """
    def __init__(self):
        # Super call
        super(RandomPolicy, self).__init__(name="Random")

    @staticmethod
    def select(q_values):
        # Random selection
        return np.random.randint(0, q_values.shape[0])

    def select_action(self, q_values, step, episode):
        # Check q-values dimension
        self._check_q_values_dimension(q_values)
        # Random selection
        return self.select(q_values)

    def get_distribution(self, q_values, step, episode):
        # All values have a uniform probability
        new_q_values = np.full_like(q_values, fill_value=q_values.shape[-1])
        return new_q_values


###########################
# Epsilon-Greedy Policies #
###########################
class EpsilonGreedyPolicy(Policy):
    """An epsilon greedy policy.

    Attributes:
        epsilon: The epsilon value in [0, 1] to select an action with.
    """
    def __init__(self, epsilon):
        # Super call
        super(EpsilonGreedyPolicy, self).__init__(name=u"\u03B5-greedy", value_name=u"\u03B5")
        # Epsilon must be between 0 and 1
        assert 0 <= epsilon <= 1, f"Value of epsilon should be in [0, 1]. Given: {epsilon}."
        # Store the arguments
        self.epsilon = epsilon

    def select_action(self, q_values, step, episode):
        # Check q-values dimension
        self._check_q_values_dimension(q_values)
        # Explore
        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, q_values.shape[0])
        # Greedy selection
        else:
            return np.argmax(q_values)

    def get_distribution(self, q_values, step, episode):
        # The maximum q-values have probability 1
        new_q_values = np.zeros_like(q_values)
        greedy_actions = np.argmax(q_values, axis=1)
        xs = np.arange(new_q_values.shape[0])
        new_q_values[xs, greedy_actions] = 1
        return new_q_values

    def __str__(self):
        val = round(self.epsilon, 5)
        return f"{self.name} Policy ({self.value_name} = {val})"


class DecayingEpsilonGreedyPolicy(EpsilonGreedyPolicy):
    """An epsilon greedy policy, with a decaying epsilon over a number of steps/episodes.

    Attributes:
        epsilon_max: The starting value of the epsilon.
        epsilon_min: The ending value of the epsilon.
        steps: (Optional) The number of steps over which to decay the epsilon value.
        episodes: (Optional) The number of episodes over which to decay the epsilon value.
        decay: (Optional) The decaying factor.
        decay_per_step: (Optional) Whether or not to decay per step or per episode.
    """
    def __init__(self, epsilon_max, epsilon_min, warm_up_steps=0,
                 steps=None, episodes=None, decay=False, decay_per_step=False):
        # Super call
        super(DecayingEpsilonGreedyPolicy, self).__init__(epsilon=epsilon_max)
        self.name = u"Decaying \u03B5-greedy"
        # Epsilon values must be between 0 and 1
        assert 0 <= epsilon_max <= 1, f"Value of epsilon_max should be in [0, 1]. Given: {epsilon_max}."
        assert 0 <= epsilon_min <= 1, f"Value of epsilon_min should be in [0, 1]. Given: {epsilon_min}."
        assert epsilon_min < epsilon_max, f"Expected epsilon_min < epsilon_max. Found: {epsilon_min} ≥ {epsilon_max}."
        # Either steps or episode must be supplied
        steps_supplied = steps is not None
        episodes_supplied = episodes is not None
        decay_supplied = decay is not False
        supply_count = 0
        if steps_supplied:
            supply_count += 1
        if episodes_supplied:
            supply_count += 1
        if decay_supplied:
            supply_count += 1
        # Expecting exactly 1 argument
        if supply_count == 3:
            raise ValueError(f"All of 'steps', 'episodes' and 'decay' have been supplied. Please only supply one.")
        elif supply_count == 2:
            val1 = 'steps' if steps_supplied else 'episodes'
            val2 = 'decay' if decay_supplied else 'episodes'
            raise ValueError(f"Both '{val1}' and '{val2}' have been supplied. Please only supply one.")
        elif supply_count == 0:
            raise AttributeError(f"No 'steps', 'episodes' or 'decay' have been supplied. Please supply one.")
        # Decaying each time step
        elif steps_supplied:
            assert 0 < steps == int(steps), f"Value of steps must be > 0 and an integer. Given: {steps}."
            self.update_per_step = True
        # Decaying each episode
        elif episodes_supplied:
            assert 0 < episodes == int(episodes), f"Value of episodes must be > 0 and an integer. Given: {episodes}."
            self.update_per_step = False
        # Decaying with a factor
        else:
            pass
        # Store the arguments
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.steps = steps
        self.episodes = episodes
        self.warm_up_steps = warm_up_steps
        self.decay = decay
        self.decay_per_step = decay_per_step
        self._previous_episode = 0

    def _update_epsilon_value(self, step, episode):
        # Decay with factor
        if self.decay and (self.decay_per_step or (self._previous_episode != episode)):
            new_value = max(self.epsilon_min, self.epsilon * self.decay)
            # Update the current epsilon
            self.epsilon = new_value
            self._previous_episode = episode
        # Calculate the new value if decaying per step or the episode changed
        elif not self.decay and (self.update_per_step or (self._previous_episode != episode)):
            # Linearly anneal
            a = - float(self.epsilon_max - self.epsilon_min)
            a /= float(self.steps) if self.update_per_step else float(self.episodes)
            new_value = a * (step if self.update_per_step else episode) + self.epsilon_max
            new_value = max(self.epsilon_min, new_value)
            # Update the current epsilon
            self.epsilon = new_value
            self._previous_episode = episode

    def select_action(self, q_values, step, episode):
        if np.any(np.isnan(q_values)):
            raise RuntimeError("ENCOUNTERED NaN")
            # q_values = np.full_like(q_values, fill_value=1/len(q_values))
        # Update the epsilon value only after the warm-up steps
        if step > self.warm_up_steps:
            self._update_epsilon_value(step, episode)
        # Select an action using epsilon-greedy
        return super(DecayingEpsilonGreedyPolicy, self).select_action(q_values, step, episode)


##################
# Softmax Policy #
##################
class SoftmaxPolicy(Policy):
    """A Softmax policy.

    Attributes:
        None.
    """
    def __init__(self):
        # Super call
        super(SoftmaxPolicy, self).__init__(name="Softmax")

    def select_action(self, q_values, step, episode):
        # Check q-values dimension
        self._check_q_values_dimension(q_values)
        # Make sure the q-values are positive for probability distribution
        min_q = np.min(q_values)
        if min_q < 0:
            q_values += abs(min_q)
        # Calculate action probabilities based on q-values
        probabilities = q_values / np.sum(q_values)
        # Select action based on probability
        action = np.random.choice(range(q_values.shape[0]), p=probabilities)
        return action

    def get_distribution(self, q_values, step, episode):
        # Make sure the q-values are positive for probability distribution
        min_q = np.min(q_values)
        if min_q < 0:
            q_values += abs(min_q)
        # Calculate action probabilities based on q-values
        probabilities = q_values / np.sum(q_values)
        return probabilities


####################
# Boltzmann Policy #
####################
class BoltzmannPolicy(Policy):
    """A Boltzmann policy.

    Attributes:
        tau: The temperature value for the Boltzmann distribution.
        clip: The clipping value used for the Boltzmann distribution, with values in [-clip, clip].
    """
    def __init__(self, tau, clip=500.0):
        # Super call
        super(BoltzmannPolicy, self).__init__(name="Boltzmann", value_name=u"\u03C4")
        # Tau must be between 0 and 1
        assert 0 < tau, f"Value of tau should be strictly positive. Given: {tau}."
        assert 0 < clip, f"Clipping value must be strictly positive. Given: {clip}."
        # Store the arguments
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values, step, episode):
        # Check q-values dimension
        self._check_q_values_dimension(q_values)
        # Make sure the q-values are positive for probability distribution
        q_values = q_values.astype(np.float64)
        min_q = np.min(q_values)
        if min_q < 0:
            q_values += abs(min_q)
        if np.any(np.isnan(q_values)):
            raise RuntimeError("ENCOUNTERED NaN")
            # q_values = np.full_like(q_values, fill_value=1/len(q_values))
        # Compute the Boltzmann distribution
        exp_values = np.exp(np.clip(q_values / self.tau, -self.clip, self.clip))
        probabilities = exp_values / np.sum(exp_values)
        # Select action based on Boltzmann distribution
        action = np.random.choice(range(q_values.shape[0]), p=probabilities)
        return action

    def get_distribution(self, q_values, step, episode):
        # Make sure the q-values are positive for probability distribution
        q_values = np.array(q_values, dtype=np.float64)
        min_q = np.min(q_values)
        if min_q < 0:
            q_values += abs(min_q)
        # Compute the Boltzmann distribution
        exp_values = np.exp(np.clip(q_values / self.tau, -self.clip, self.clip))
        probabilities = exp_values / np.sum(exp_values, axis=1)[:, None]
        return probabilities.astype(np.float32)

    def __str__(self):
        val = round(self.tau, 5)
        return f"{self.name} Policy ({self.value_name} = {val})"
