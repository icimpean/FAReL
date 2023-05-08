from copy import deepcopy
from typing import List

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

from agent import Agent
from agent.dqn.experience_replay import Experience, ReplayMemory
from agent.dqn.policy import RandomPolicy, EpsilonGreedyPolicy, DecayingEpsilonGreedyPolicy, GreedyPolicy


class QNeuralNetwork(nn.Module):
    """A Q-learning neural network"""
    def __init__(self, input_shape, h1, output_shape, learning_rate, seed):
        # Super call
        super(QNeuralNetwork, self).__init__()
        # Create the internal layers of the neural network
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.l1 = nn.Linear(input_shape, h1)
        self.l2 = nn.Linear(h1, output_shape)
        self.to(device=torch.device('cpu'))
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.seed = seed

    def forward(self, x):
        """The forward pass through the model."""
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

    def get_q_values(self, x, is_single_sample=False):
        """Get the q-values of a given input."""
        state = self.to_variable(x.reshape(-1, self.input_shape))
        q_values = self(state)
        return q_values[0] if is_single_sample else q_values

    @staticmethod
    def to_variable(x):
        """Convert input to a torch tensor."""
        # noinspection PyArgumentList
        return Variable(torch.Tensor(x))


class DQNAgent(Agent):
    """The DQN agent."""
    def __init__(self, input_shape, h1, actions, warm_up=0, policy=EpsilonGreedyPolicy(0.1), memory_size=3000, batch_size=64, learning_rate=0.001, training=True, seed=0):
        # Super Call
        super(DQNAgent, self).__init__()
        # Store some parameters for easier access
        self.seed = seed
        self.input_shape = input_shape
        self.actions = actions
        self.nr_actions = len(self.actions)
        self.gamma = 0.99
        self.double_dqn = False
        self.warm_up_steps = warm_up
        self.policy = policy
        self.test_policy = GreedyPolicy()
        # Create the neural network and the target network
        self.model = QNeuralNetwork(input_shape, h1, self.nr_actions, learning_rate, self.seed)
        self.target_model = deepcopy(self.model) if self.double_dqn else None
        self.target_update = 250
        self.target_tau = 1
        if self.double_dqn:
            self.target_model.train(mode=False)
        # Create the requested memory type
        self.memory = ReplayMemory(memory_size=memory_size, batch_size=batch_size)
        # Are we training the agent?
        self.training = training
        self.model.train(mode=self.training)

    def set_training(self, train=True):
        """Specify if the agent is training or not"""
        self.training = train
        self.model.train(mode=self.training)

    def predict_q_values(self, state):
        """Return the predicted q-values for a state.

        Args:
            state: a state to predict the q-values for.
        """
        with torch.no_grad():
            q_values = self.model.get_q_values(state, is_single_sample=True).data.numpy()
            # print("predict_q_values", q_values)
            # if np.any(np.isnan(q_values)):
            #     print("was nan for:", state)
        return q_values

    def select_action(self, state, step, episode=0, return_q_values=False):
        """Select an action with the current policy.

        Args:
            state: The state observed.
            step: The current time step.
            episode: The current episode.
            return_q_values: (Optional) Whether or not to also return the q_values.

        Returns:
            The action chosen by the agent's policy. (Optional) The q-values returned by the agent.
        """
        state = state.to_array()
        # Get the q-values predicted for the state
        q_values = self.predict_q_values(state)
        # If training
        if self.training:
            # Use a random policy for the warm up steps
            if step < self.warm_up_steps:
                action = RandomPolicy.select(q_values)
            # Use the policy
            else:
                action = self.policy.select_action(q_values, step, episode)
        # If testing
        else:
            action = self.test_policy.select_action(q_values, step, episode)
        # Return the action, along with the q_values if requested
        if return_q_values:
            return action, q_values
        else:
            return action

    def store_experience(self, experience: Experience):
        """Store an experience in the agent's replay memory."""
        self.memory.store_experience(experience, self)

    def train(self, step):
        """Train the DQN agent and store the latest experience in memory.

        Args:
            step: The current time step.

        Returns:
            loss: The training loss of the agent.
        """
        # Train on a batch of experiences if the number of warm up steps are done
        if step >= self.warm_up_steps:
            experiences = self.memory.get_experience_batch()
            # Train the underlying Q model on the batch of experiences
            return self._train_on_batch(experiences, step)

    def _train_on_batch(self, experiences: List[Experience], step):
        """Train the underlying Q network and update the target network."""
        # Preprocess the data
        samples = len(experiences)
        states = np.empty(shape=(samples, self.input_shape))
        next_states = np.empty_like(states)
        actions = np.empty(shape=samples, dtype=int)
        rewards = np.empty(shape=samples)
        terminals = np.empty_like(rewards, bool)

        for idx, experience in enumerate(experiences):
            state, action, reward, terminate, next_state, _ = experience
            states[idx] = state.to_array()
            next_states[idx] = next_state.to_array()
            actions[idx] = action
            rewards[idx] = reward
            terminals[idx] = terminate
        # Get the predicted q-values for each state
        q_values = self.model.get_q_values(states)
        # print("_train_on_batch q_values", q_values)
        # The target q_value: reward + gamma * future_reward
        target_q_values = q_values.clone().data.numpy()
        # print("_train_on_batch target_q_values", q_values)
        # If using double Q-learning, predict with the target network
        if self.double_dqn:
            next_q_values = self.target_model.get_q_values(next_states).data.numpy()
        else:
            next_q_values = self.model.get_q_values(next_states).data.numpy()
        # The future reward is max Q(next_state) or 0 if next_state is terminal
        future_rewards = np.max(next_q_values, axis=1) * ~terminals
        target_q_values[np.arange(len(target_q_values)), actions] = rewards + self.gamma * future_rewards
        target_q_values = self.model.to_variable(target_q_values)

        # Reset gradients
        self.model.optimizer.zero_grad()
        # Compute the loss
        loss = self.model.loss(q_values, target_q_values)
        # print("loss", loss)
        # Optimize the model
        loss.backward()
        self.model.optimizer.step()

        # Double DQN
        if self.double_dqn:
            # Hard update every target_update steps
            if self.target_update >= 1 and step % self.target_update == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            # Soft update every target_update steps
            elif step % self.target_update == 0:
                for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                    # noinspection PyUnresolvedReferences
                    target_param.data.copy_(target_param.data * (1.0 - self.target_tau) + param.data * self.target_tau)

        # Return the loss
        return loss.item()
