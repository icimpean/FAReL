import random
from collections import namedtuple


##############
# Experience #
##############
# The named tuple representing an experience
Experience = namedtuple("Experience", "state, action, reward, done, next_state, q_values")


################################
# The Experience Replay Memory #
################################
class ReplayMemory(object):
    """The class representing an experience replay memory for an agent.

    Attributes:
        memory_size: The maximum number of experiences in the replay memory.
        batch_size: The batch_size used for sampling the replay memory.
    """
    def __init__(self, memory_size, batch_size):
        # Store the arguments
        self.memory_size = memory_size
        self.batch_size = batch_size
        # The replay memory is implemented as a circular list for faster sampling (faster than deque sampling)
        self._memory = [None] * memory_size
        self._index = 0
        self._size = 0

        self.last_counts = []

    def __len__(self):
        """Get the length of the internal memory."""
        return self._size

    def _count_experiences(self, experiences):
        if len(experiences) == 0:
            self.last_counts = []
        else:
            # Store the counts per action
            counts = [0 for _ in range(len(experiences[0].q_values))]
            for x in experiences:
                counts[x.action] += 1
            self.last_counts = counts

    def store_experience(self, experience, agent=None):
        """Store an experience in the memory."""
        # Replace the oldest memory in the circularly iterated list
        self._memory[self._index] = experience
        self._size = min(self._size + 1, self.memory_size)
        self._index = (self._index + 1) % self.memory_size

    def get_experience_batch(self):
        """Get a random batch of experiences from the memory."""
        # You can only get as many experiences as present in the memory, with a maximum of batch_size
        number_of_experiences = min(self._size, self.batch_size)
        # Get the random experiences from the memory
        if self._size < self.memory_size:
            random_experiences = random.sample(self._memory[:self._size], number_of_experiences)
        else:
            random_experiences = random.sample(self._memory, number_of_experiences)
        # Count the number of sampled experiences per action
        self._count_experiences(random_experiences)
        return random_experiences
