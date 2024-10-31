import numpy as np
import torch


class Buffer:
    """replay buffer for each agent"""

    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = capacity

        self.obs = np.zeros((capacity, obs_dim))
        self.action = np.zeros((capacity, act_dim))
        self.reward = np.zeros(capacity)
        self.next_obs = np.zeros((capacity, obs_dim))
        self.done = np.zeros(capacity, dtype=bool)

        self._index = 0
        self._size = 0

        self.device = device

    def add(self, obs, action, reward, next_obs, done):
        """ add an experience to the memory """
        self.obs[self._index] = obs
        self.action[self._index] = action
        self.reward[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.done[self._index] = done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        # retrieve data, Note that the data stored is ndarray
        obs = self.obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        next_obs = self.next_obs[indices]
        done = self.done[indices]

        # NOTE that `obs`, `action`, `next_obs` will be passed to network(nn.Module),
        # so the first dimension should be `batch_size`
        obs = torch.from_numpy(obs).float().to(self.device)  # torch.Size([batch_size, state_dim])
        action = torch.from_numpy(action).float().to(self.device)  # torch.Size([batch_size, action_dim])
        reward = torch.from_numpy(reward).float().to(self.device)  # just a tensor with length: batch_size
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)  # Size([batch_size, state_dim])
        done = torch.from_numpy(done).float().to(self.device)  # just a tensor with length: batch_size

        return obs, action, reward, next_obs, done

    def __len__(self):
        return self._size


class Buffer_op:
    """replay buffer for each agent"""

    def __init__(self, capacity, obs_dim, act_dim, option_dim, device):
        self.capacity = capacity

        self.obs = np.zeros((capacity, obs_dim))
        self.action = np.zeros((capacity, act_dim))
        self.options = np.zeros((capacity, option_dim))
        self.reward = np.zeros(capacity)
        self.next_obs = np.zeros((capacity, obs_dim))
        self.done = np.zeros(capacity, dtype=bool)

        self._index = 0
        self._size = 0

        self.device = device

    def add(self, obs, action, reward, next_obs, done, options):
        """ add an experience to the memory """
        self.obs[self._index] = obs
        self.action[self._index] = action
        self.options[self._index] = options.detach()
        self.reward[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.done[self._index] = done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        # retrieve data, Note that the data stored is ndarray
        obs = self.obs[indices]
        action = self.action[indices]
        options = self.options[indices]
        reward = self.reward[indices]
        next_obs = self.next_obs[indices]
        done = self.done[indices]

        # NOTE that `obs`, `action`, `next_obs` will be passed to network(nn.Module),
        # so the first dimension should be `batch_size`
        obs = torch.from_numpy(obs).float().to(self.device)  # torch.Size([batch_size, state_dim])
        action = torch.from_numpy(action).float().to(self.device)  # torch.Size([batch_size, action_dim])
        options = torch.from_numpy(options).float().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)  # just a tensor with length: batch_size
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)  # Size([batch_size, state_dim])
        done = torch.from_numpy(done).float().to(self.device)  # just a tensor with length: batch_size

        return obs, action, reward, next_obs, done, options

    def __len__(self):
        return self._size

class Option_Buffer:
    """option replay buffer for each agent"""

    def __init__(self, capacity, option_dim, obs_dim, device):
        self.capacity = capacity

        self.obs = np.zeros((capacity, obs_dim))
        self.options = np.zeros((capacity, option_dim))
        self.jump_obs = np.zeros((capacity, obs_dim))

        self._index = 0
        self._size = 0

        self.device = device

    def add(self, options, obs, jump_obs):
        """ add an experience to the memory """
        self.obs[self._index] = obs
        self.options[self._index] = options
        self.jump_obs[self._index] = jump_obs

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        # retrieve data, Note that the data stored is ndarray
        obs = self.obs[indices]
        options = self.options[indices]
        jump_obs = self.jump_obs[indices]

        # NOTE that `obs`, `action`, `next_obs` will be passed to network(nn.Module),
        # so the first dimension should be `batch_size`
        obs = torch.from_numpy(obs).float().to(self.device)  # torch.Size([batch_size, state_dim])
        options = torch.from_numpy(options).float().to(self.device)
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        jump_obs = torch.from_numpy(jump_obs).float().to(self.device)  # Size([batch_size, state_dim])

        return obs, jump_obs, options

    def __len__(self):
        return self._size
