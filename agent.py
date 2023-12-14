import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import List, Tuple

from buffer import ReplayBuffer
from net import ActorNet, CriticNet
from ou_noise import OUNoise


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    """Implementation of a single DDPG agent for multi-agent DDPG learning.

    This class encapsulates the actor / critic (and corresponding target) networks for a single
    agent. The bulk of the learning logic is contained in the MultiAgent class.
    """
    def __init__(self, state_size, action_size, num_agents=2):
        self.actor = ActorNet(state_size, action_size).to(device)
        self.target_actor = ActorNet(state_size, action_size).to(device)
        critic_input_size = (state_size + action_size) * num_agents
        self.critic = CriticNet(critic_input_size).to(device)
        self.target_critic = CriticNet(critic_input_size).to(device)
        # Initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        # Noise for actions
        self.ou_noise = OUNoise(action_size, scale=1.0)
        # Define optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

    def eval_act(self, obs, noise=0.0):
        """Select an action (via the regular actor network) during evaluation phase."""
        self.actor.eval()
        obs = torch.Tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs) + noise * self.ou_noise.noise()
        self.actor.train()
        action = action.squeeze().cpu().detach()
        return np.clip(action, -1, 1)  # Clip action in the admissible interval

    def target_act(self, obs: torch.Tensor, noise=0.0) -> torch.Tensor:
        """Select an action via the target actor network (used in training)."""
        return self.target_actor(obs) + noise * self.ou_noise.noise()


class MultiAgent:
    def __init__(self, agents: List[Agent], gamma=0.995, tau=1e-3,
                 buffer_len=int(1e5), batch_size=256, update_every=2,
                 n_updates=4):
        self.agents = agents
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(buffer_len, sample_size=batch_size)
        self.batch_size = batch_size
        self.update_every = update_every
        self.n_updates = n_updates
        self.t_step = 0

    def eval_act(self, obs_all_agents, noise=0.0) -> List[torch.Tensor]:
        """Get actions from all agents in the MultiAgent object for evaluation."""
        obs_all_agents = list(obs_all_agents)
        return [agent.eval_act(obs, noise) for agent, obs in zip(self.agents, obs_all_agents)]

    def target_act(self, obs_all_agents, noise=0.0) -> List[torch.Tensor]:
        """Get target network actions from all the agents in the MultiAgent object for training."""
        return [agent.target_act(obs, noise) for agent, obs in zip(self.agents, obs_all_agents)]

    def step(self, obs, actions, next_obs, rewards, dones):
        """Multi-agent step: tracks the experience, and updates the agents."""
        obs      = torch.Tensor(obs)                    # (n_agents, state_size)
        actions  = torch.stack(actions)                 # (n_agents, action_size)
        next_obs = torch.Tensor(next_obs)               # (n_agents, state_size)
        rewards  = torch.Tensor(rewards).unsqueeze(1)   # (n_agents, 1)
        dones    = torch.Tensor(dones).unsqueeze(1)     # (n_agents, 1)

        transition = (obs, actions, rewards, next_obs, dones)
        self.replay_buffer.push(transition)

        self.t_step += 1

        # Update once after every n timesteps.
        if len(self.replay_buffer) > self.batch_size and self.t_step % self.update_every == 0:
            for _ in range(self.n_updates):  # Run multiple agents update iterations.
                for a_i in range(len(self.agents)):
                    samples = self.replay_buffer.sample()
                    self.update(samples, a_i)
                # Soft update the target network towards the actual networks. NOTE: we are updating
                # all the target networks _after_ all the updates (which use target predictions
                # themselves).
                self.update_targets()

    def update(self, samples: Tuple[torch.Tensor], agent_number: int):
        """Update the critics and actors of all the agents in the MultiAgent object."""
        # Turning these into lists of tensors (for each agent) for convenience, to keep the
        # torch.cat operations and logic here and in the MultiAgent methods mostly unchanged.
        obs, action, reward, next_obs, done = tuple(list(x) for x in samples)
        agent = self.agents[agent_number]

        # ---------------------------- update critic ---------------------------- #
        agent.critic_optimizer.zero_grad()
        # Get predicted next-state actions and Q values from target models
        actions_next = self.target_act(next_obs)  # list of action tensors for each agent
        with torch.no_grad():
            Q_targets_next = agent.target_critic(next_obs, actions_next)

        Q_expected = agent.critic(obs, action)
        # Compute Q targets for current states (y_i)
        Q_targets = reward[agent_number] + self.gamma * Q_targets_next * (1 - done[agent_number])

        # Calculate the critic loss (batch mean of (y - Q(s,a) from target network) ^ 2)
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        agent.actor_optimizer.zero_grad()
        # Make input to agent, detach the other agents to save computation.
        actions_pred = [ self.agents[i].actor(ob) \
                           if i == agent_number else self.agents[i].actor(ob).detach() \
                           for i, ob in enumerate(obs) ]
        actor_loss = -agent.critic(obs, actions_pred).mean()

        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
        agent.actor_optimizer.step()

    def update_targets(self):
        """Soft update actor / critic target networks."""
        for agent in self.agents:
            soft_update(agent.actor, agent.target_actor, self.tau)
            soft_update(agent.critic, agent.target_critic, self.tau)

    def save(self, filename='multiagent.pt'):
        """Saves the multi-agent into a file."""
        d = {}
        for i, a in enumerate(self.agents):
            d[f'actor-{i}']  = a.actor.state_dict()
            d[f'critic-{i}'] = a.critic.state_dict()
        torch.save(d, filename)

    def load(self, filename: str):
        """Loads a multi-agent from a file."""
        d = torch.load(filename)
        for k, v in d.items():
            t = k.split('-')[0]  # actor or critic
            i = int(k.split('-')[1])
            agent = self.agents[i]
            if t == 'actor':
                agent.actor.load_state_dict(v)
            elif t == 'critic':
                agent.critic.load_state_dict(v)
            else:
                raise Exception('Unexpected format.')


def soft_update(local_model: nn.Module, target_model: nn.Module, tau: float):
    """Soft update model parameters.
    θ_target = τ * θ_local + (1 - τ) * θ_target

    Params
    ======
        local_model: weights will be copied from
        target_model: weights will be copied to
        tau: interpolation parameter (0 < tau < 1)
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def hard_update(target: nn.Module, source: nn.Module):
    """
    Copy network parameters from source to target.

    Params
    ======
        source: whose parameters to copy
        target: to copy parameters to
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
