import numpy as np
import torch

from collections import deque
from unityagents import UnityEnvironment

from agent import MultiAgent


class MA_DDPG:
    """Multi-Agent DDPG training loop implementation for the Tennis environment."""

    def __init__(self, env: UnityEnvironment, multi_agent: MultiAgent):
        self.env             = env
        self.multi_agent     = multi_agent
        self.brain_name: str = env.brain_names[0]
        self.brain           = self.env.brains[self.brain_name]
        self.action_size     = self.brain.vector_action_space_size
        env_info             = self.env.reset(train_mode=False)[self.brain_name]
        self.n_agents        = len(env_info.agents)
        states               = env_info.vector_observations
        self.state_size      = states.shape[1]

    def train(self, max_episodes=1500, max_t=1000, solved=0.5,
              noise=1., noise_reduction=0.9999, noise_min=0.005, persist=False):
        """DDPG training: loop over N episodes, with max timesteps per episode."""
        episodes_all_scores = []
        episodes_100_scores = deque(maxlen=100)
        best_100_average    = 0.

        for n_episode in range(1, max_episodes + 1):
            scores = torch.zeros((self.n_agents,))
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            obs = env_info.vector_observations

            for _ in range(max_t):
                actions = self.multi_agent.eval_act(obs, noise=max(noise, noise_min))
                # Noise decay: high exploration only for a certain number of episodes.
                noise *= noise_reduction

                # Step forward one frame / timestep.
                env_info = self.env.step([a.numpy() for a in actions])[self.brain_name]
                next_obs = env_info.vector_observations
                rewards  = env_info.rewards
                dones    = env_info.local_done

                self.multi_agent.step(obs, actions, next_obs, rewards, dones)

                obs = next_obs
                scores += np.array(rewards)

                if np.any(dones):
                    break

            episode_score = scores.max()
            episodes_all_scores.append(episode_score)
            episodes_100_scores.append(episode_score)
            avg = np.mean(episodes_100_scores)

            if best_100_average < solved and avg >= solved:
                print()
                print(f'Environment solved at episode n.{n_episode} with average score: {avg:.3f}')
                print()
                best_100_average = avg
                if persist:
                    self.multi_agent.save('multiagent_solved.pt')

            if best_100_average >= solved and avg > best_100_average + 0.02 and persist:
                print(f'Saving new best model at episode n.{n_episode} with avg score: {avg:.3f}')
                best_100_average = avg
                self.multi_agent.save('multiagent_best.pt')

            if n_episode % 50 == 0:
                print(f'Episode n.{n_episode} completed. ' +
                      f'Average score: {np.mean(episodes_100_scores):.3f}, ' +
                      f'Noise: {noise:.5f}, Timestep: {self.multi_agent.t_step}')

        # Save the scores on disk if instructed to do so.
        if persist:
            np.savetxt("scores.csv", np.asarray(episodes_all_scores), delimiter=",")

        return episodes_all_scores
