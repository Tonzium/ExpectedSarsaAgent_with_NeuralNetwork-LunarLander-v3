#!/usr/bin/env python

"""RandomWalk environment class for RL-Glue-py.
"""

from src.environments.base_environment import BaseEnvironment
import gymnasium as gym

class LunarLanderEnvironment(BaseEnvironment):
    def env_init(self, env_info={}):
        """
        Setup for the environment called when the experiment first starts.
        """
        self.env = gym.make("LunarLander-v3")
        seed = env_info.get("seed", 0)
        # Note: In Gymnasium, seed is passed to reset() or when making the env.
        self.seed = seed

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """        
        
        reward = 0.0
        observation, info = self.env.reset(seed=self.seed)
        is_terminal = False
                
        self.reward_obs_term = (reward, observation, is_terminal)
        
        # return first state observation from the environment
        return self.reward_obs_term[1]
        
    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        last_state = self.reward_obs_term[1]
        # Gymnasium step returns: observation, reward, terminated, truncated, info
        current_state, reward, terminated, truncated, info = self.env.step(action)
        
        is_terminal = terminated or truncated
        self.reward_obs_term = (reward, current_state, is_terminal)
        
        return self.reward_obs_term