from collections import deque
from typing import List, Tuple

import random
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Replay buffer used in DDPG implementation.

    Stores tuples like: (obs, actions, rewards, next_obs, dones)
    """

    def __init__(self, size: int, sample_size: int, num_agents=2):
        self.size = size
        self.deque = deque(maxlen=self.size)
        self.sample_size = sample_size
        self.num_agents = num_agents

    def push(self, transition: Tuple[torch.Tensor]):
        """Pushes into the buffer."""
        # Sanity checks on dimensionality.
        assert len(transition) == 5               # (obs, actions, rewards, next_obs, dones)
        for t in transition:
            assert len(t.shape) > 1
            assert t.shape[0] == self.num_agents  # First dimension is the agent.
        self.deque.append(transition)             # Push into the buffer!

    def sample(self) -> Tuple[torch.Tensor]:
        """Samples from the buffer. Returns a tuple of tensors for convenience.

        The tensors shapes are: (num_agents, batch_size, ...)
        """
        samples = random.sample(self.deque, self.sample_size) # list of tuples of tensors

        obs      = torch.stack([x[0] for x in samples]).to(device)
        actions  = torch.stack([x[1] for x in samples]).to(device)
        rewards  = torch.stack([x[2] for x in samples]).to(device)
        next_obs = torch.stack([x[3] for x in samples]).to(device)
        dones    = torch.stack([x[4] for x in samples]).to(device)

        t_samples = (obs, actions, rewards, next_obs, dones)  # (batch_size, num_agents, ...)
        return tuple(x.transpose(0, 1) for x in t_samples)    # (num_agents, batch_size, ...)

    def __len__(self):
        return len(self.deque)
