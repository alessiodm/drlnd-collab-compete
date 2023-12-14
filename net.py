import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def layer_init(layer, weight_mul=1.0):
    """Layer initialization for the neural-netowork linear layers."""
    torch.nn.init.xavier_uniform_(layer.weight, gain=0.4)
    layer.weight.data.mul_(weight_mul)
    return layer


class ActorNet(nn.Module):
    """Feed-forward neural network for DDPG actor (continous action space)."""
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_size, 256)),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            layer_init(nn.Linear(256, 128)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(64, action_size), weight_mul=1e-3),
            nn.Tanh(),
        )

    def forward(self, x) -> torch.Tensor:
        return self.net(x)


class CriticNet(nn.Module):
    """Feed-forward neural network for DDPG critic."""
    def __init__(self, critic_input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(critic_input_size, 256)),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            layer_init(nn.Linear(256, 128)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(64, 1), weight_mul=1e-3),
        )

    def forward(self, obs_all_agents, action_all_agents) -> torch.Tensor:
        obs_all_agents    = torch.cat(obs_all_agents, dim=1)
        action_all_agents = torch.cat(action_all_agents, dim=1)
        critic_input      = torch.cat((obs_all_agents, action_all_agents), dim=1).to(device)
        return self.net(critic_input)
