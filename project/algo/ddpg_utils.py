import torch.nn.functional as F
from torch import nn
from collections import namedtuple
import numpy as np
import torch

# from torch.distributions import Categorical
from torch.distributions import Normal, Independent

import pickle, os, random, torch

from collections import defaultdict
import pandas as pd 
import gymnasium as gym
import matplotlib.pyplot as plt

Batch = namedtuple('Batch', ['state', 'action', 'next_state', 'reward', 'not_done', 'extra'])

def soft_update_params(m, m_target, tau):
    """
    Update slow-moving average of online network (target network) at rate tau.
    """
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)

# Actor-critic agent
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )
        

    def forward(self, state):
        return self.max_action * torch.tanh(self.actor(state))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # True

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.value(x)  # output shape [batch, 1]

class ReplayBuffer(object):
    def __init__(self, state_shape: tuple, action_dim: int, max_size=int(1e6)):
        # Convert max_size to int
        if isinstance(max_size, str):
            max_size = int(float(max_size))

        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        dtype = torch.float32
        
        # Ensure state_shape is a tuple
        if isinstance(state_shape, int):
            state_shape = (state_shape,)  # Convert int to tuple

        observation_dim = state_shape[0]  # Extract the first element

        self.state = torch.zeros(size=(max_size, observation_dim), dtype=dtype)
        self.action = torch.zeros(size=(max_size, action_dim), dtype=dtype)
        self.next_state = torch.zeros(size=(max_size, observation_dim), dtype=dtype)
        self.reward = torch.zeros(size=(max_size, 1), dtype=dtype)
        self.not_done = torch.zeros(size=(max_size, 1), dtype=dtype)
        self.extra = {}
        
    def _to_tensor(self, data, dtype=torch.float32):
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype)
        return torch.tensor(data, dtype=dtype)

    def add(self, state, action, next_state, reward, done, extra: dict = None):
        self.state[self.ptr] = self._to_tensor(state, dtype=self.state.dtype)
        self.action[self.ptr] = self._to_tensor(action)
        self.next_state[self.ptr] = self._to_tensor(next_state, dtype=self.state.dtype)
        self.reward[self.ptr] = self._to_tensor(reward)
        self.not_done[self.ptr] = self._to_tensor(1.0 - done)

        if extra is not None:
            for key, value in extra.items():
                if key not in self.extra:  # init buffer
                    self.extra[key] = torch.zeros(
                        (self.max_size, *value.shape), dtype=torch.float32
                    )
                self.extra[key][self.ptr] = self._to_tensor(value)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device="cpu"):
        ind = np.random.randint(0, self.size, size=batch_size)

        if self.extra:
            extra = {key: value[ind].to(device) for key, value in self.extra.items()}
        else:
            extra = {}

        batch = Batch(
            state=self.state[ind].to(device),
            action=self.action[ind].to(device),
            next_state=self.next_state[ind].to(device),
            reward=self.reward[ind].to(device),
            not_done=self.not_done[ind].to(device),
            extra=extra,
        )
        return batch

    def get_all(self, device="cpu"):
        if self.extra:
            extra = {
                key: value[: self.size].to(device) for key, value in self.extra.items()
            }
        else:
            extra = {}

        batch = Batch(
            state=self.state[: self.size].to(device),
            action=self.action[: self.size].to(device),
            next_state=self.next_state[: self.size].to(device),
            reward=self.reward[: self.size].to(device),
            not_done=self.not_done[: self.size].to(device),
            extra=extra,
        )
        return batch
        # True
