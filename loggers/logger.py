from loggers import LogEntry


class AgentLogger(LogEntry):
    """A log entry containing data per timestep for an agent."""
    def __init__(self, path_experience, path_train, path_eval, path_eval_axes):
        # Super call
        super(AgentLogger, self).__init__()
        # The fields of the entry type
        self.episode = "episode"
        self.t = "t"
        self.state = "state"
        self.action = "action"
        self.reward = "reward"
        self.done = "done"
        self.info = "info"
        self.time = "time"
        self.status = "status"
        # All the entry fields
        self.entry_fields = [self.episode, self.t, self.state, self.action, self.reward, self.done, self.info,
                             self.time, self.status]
        #
        self.path_experience = path_experience
        self.path_train = path_train
        self.path_eval = path_eval
        self.path_eval_axes = path_eval_axes

    def create_entry(self, episode, t, state, action, reward, done, info, time, status):
        """Method to create an entry for the log."""
        return {
            self.episode: episode,
            self.t: t,
            self.state: state,
            self.action: action,
            self.reward: reward,
            self.done: done,
            self.info: info,
            self.time: time,
            self.status: status
        }


class LeavesLogger(LogEntry):
    """A log entry for the experience replay buffer"""
    def __init__(self, objective_names):
        # Super call
        super(LeavesLogger, self).__init__()
        # The fields of the entry type
        self.objectives = objective_names
        self.episode = "episode"
        self.t = "t"
        # All the entry fields
        self.entry_fields = [self.episode, self.t] + self.objectives

    def create_entry(self, episode, t, e_returns):
        """Method to create an entry for the log."""
        entry = {
            self.episode: episode,
            self.t: t,
        }
        for obj, ret in zip(self.objectives, e_returns):
            entry[obj] = ret
        return entry


class TrainingPCNLogger(LogEntry):
    """A log entry containing data per timestep for an agent."""
    def __init__(self, objectives):
        # Super call
        super(TrainingPCNLogger, self).__init__()
        # The fields of the entry type
        self.episode = "episode"
        self.t = "t"
        self.loss = "loss"
        self.entropy = "entropy"
        self.desired_horizon = "desired_horizon"
        self.horizon_distance = "horizon_distance"
        self.episode_steps = "episode_steps"
        self.hypervolume = "hypervolume"
        self.coverage_set = "coverage_set"
        self.nd_coverage_set = "nd_coverage_set"

        self.objectives = objectives
        # All the entry fields
        self.entry_fields = [self.episode, self.t, self.loss, self.entropy, self.desired_horizon, self.horizon_distance,
                             self.episode_steps, self.hypervolume, self.coverage_set, self.nd_coverage_set] + \
                            [f"return_{o}_value" for o in self.objectives] + \
                            [f"return_{o}_desired" for o in self.objectives] + \
                            [f"return_{o}_distance" for o in self.objectives]

    def create_entry(self, episode, t, loss, entropy, desired_horizon, horizon_distance, episode_steps, hypervolume,
                     coverage_set, nd_coverage_set, return_values, desired_returns, return_distances):
        """Method to create an entry for the log."""
        entry = {
            self.episode: episode,
            self.t: t,
            self.loss: loss,
            self.entropy: entropy,
            self.desired_horizon: desired_horizon,
            self.horizon_distance: horizon_distance,
            self.episode_steps: episode_steps,
            self.hypervolume: hypervolume,
            self.coverage_set: coverage_set,
            self.nd_coverage_set: nd_coverage_set
        }
        for i, o in enumerate(self.objectives):
            entry[f"return_{o}_value"] = return_values[o]
            entry[f"return_{o}_desired"] = desired_returns[o]
            entry[f"return_{o}_distance"] = return_distances[o]
        return entry


class EvalLogger(LogEntry):
    """A log entry for the evaluation"""
    def __init__(self, objectives):
        # Super call
        super(EvalLogger, self).__init__()
        #
        self.episode = "episode"
        self.t = "t"
        self.epsilon_max = "epsilon_max"
        self.epsilon_mean = "epsilon_mean"
        self.eval_type = "eval_type"

        self.objectives = objectives
        # All the entry fields
        self.entry_fields = [self.episode, self.t, self.epsilon_max, self.epsilon_mean] + \
                            [f"desired_{o}" for o in self.objectives] + \
                            [f"return_{o}" for o in self.objectives] + [self.eval_type]

    def create_entry(self, episode, t, epsilon_max, epsilon_mean, desired, returns, eval_type):
        """Method to create an entry for the log."""
        entry = {
            self.episode: episode,
            self.t: t,
            self.epsilon_max: epsilon_max,
            self.epsilon_mean: epsilon_mean,
            self.eval_type: eval_type
        }
        for i, o in enumerate(self.objectives):
            entry[f"desired_{o}"] = desired[o]
            entry[f"return_{o}"] = returns[o]
        return entry
