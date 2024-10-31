import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class OptionModel:
    def __init__(self, num_agents, obs_dim0, obs_dim1, option_dim, agent_list, device):
        agent_list0 = []
        agent_list1 = []
        for agent in agent_list:
            if agent.startswith("agent"):
                agent_list1.append(agent)
            else:
                agent_list0.append(agent)
        #print(f"###############\n{obs_dim0} {obs_dim1}\n####################")
        #print(f"###############\n{agent_list0}\n{agent_list1}\n####################")
        self.option0 = OptionAE(num_agents, obs_dim0, option_dim, agent_list0, device)
        self.option1 = OptionAE(num_agents, obs_dim1, option_dim, agent_list1, device)

    def train_step(self, buffer0, buffer1):
        self.option0.train_step(buffer0)
        self.option1.train_step(buffer1)

    def generate_options_for_all_agents(self, obs_all):
        options0 = self.option0.generate_options_for_all_agents(obs_all)
        options1 = self.option1.generate_options_for_all_agents(obs_all)
        return torch.cat((options0, options1), dim = 0)



class OptionAE(nn.Module):
    def __init__(self, num_agents, obs_dim, option_dim, agent_list, device):
        super(OptionAE, self).__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.device = device
        self.agent_list = agent_list
        # First encoder to process all agents' observations with a GAP layer
        self.encoder1 = nn.Sequential(
            nn.Linear(obs_dim, obs_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(obs_dim)  # GAP layer to ensure permutation invariance
        )

        # Second encoder to process the combined observation
        self.encoder2 = nn.Sequential(
            nn.Linear(2 * obs_dim, obs_dim),
            nn.ReLU(),
            nn.Linear(obs_dim, option_dim)  # Outputs option_k
        )

        # Decoder to predict jump_observation
        self.decoder = nn.Sequential(
            nn.Linear(obs_dim + option_dim, obs_dim),
            nn.ReLU(),
            nn.Linear(obs_dim, obs_dim)  # Outputs predicted jump_observation
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, obs_all, obs_k):
        """
        Forward pass through encoders to get the option for a specific agent.
        obs_all: torch tensor of shape (num_agents, obs_dim)
        obs_k: torch tensor of shape (obs_dim,)
        """
        obs_k = torch.tensor(obs_k)
        obs_all = torch.tensor([obs_all[k] for k in self.agent_list])
        # First encoder with GAP layer
        all_obs_encoded = self.encoder1(obs_all).mean(dim=0)  # shape: (obs_dim,)

        # Concatenate with agent_k's observation and pass through second encoder
        combined_obs = torch.cat((all_obs_encoded, obs_k), dim=0)
        option_k = self.encoder2(combined_obs)
        return option_k

    def generate_option_for_agent(self, obs_all, agent_k):
        """
        Generate option_k for a specific agent (agent_k).
        obs_all: torch tensor of shape (num_agents, obs_dim)
        agent_k: index of the specific agent
        """
        obs_k = obs_all[agent_k]  # Get the observation of the specific agent
        option_k = self.forward(obs_all, obs_k)
        return option_k

    def generate_options_for_all_agents(self, obs_all):
        """
        Generate options for all agents.
        obs_all: torch tensor of shape (num_agents, obs_dim)
        Returns: torch tensor of shape (num_agents, option_dim)
        """
        options = torch.stack([self.generate_option_for_agent(obs_all, k) for k in self.agent_list])
        return options

    def train_step(self, buffer):
        """
        Train the model using data from the buffer.
        buffer: Option_Buffer instance
        """
        # Sample a batch of experiences from the buffer
        batch_size = 64
        indices = np.random.choice(len(buffer), batch_size, replace=False)
        obs, jump_obs, options = buffer.sample(indices)

        # Generate predictions
        all_options_pred = torch.stack([self.generate_option_for_agent(obs, k) for k in range(self.num_agents)], dim=0)
        obs_with_options = torch.cat((obs, all_options_pred), dim=-1)  # Concatenate obs and options

        # Use decoder to predict jump_obs
        predicted_jump_obs = self.decoder(obs_with_options)

        # Compute L2 loss and backpropagate
        loss = nn.MSELoss()(predicted_jump_obs, jump_obs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
