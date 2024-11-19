import gymnasium
import numpy as np

class VelocityEnvWrapper(gymnasium.Wrapper):
    """
    A custom wrapper for the velocity environment.

    This wrapper allows you to modify or log observations, rewards,
    or interactions with the environment.
    """
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        """
        Reset the environment and modify the initial observation if needed.
        """
        obs, info = self.env.reset(**kwargs)
        # Modify the observation here if needed
        return obs, info

    def step(self, action):
        """
        Execute a step in the environment and modify the results if needed.
        """
        obs, reward, done, truncated, info = self.env.step(action)
        # Example: Modify the reward (e.g., scale or add penalties)
        reward = self.modify_reward(reward)
        return obs, reward, done, truncated, info

    def modify_reward(self, reward):
        """
        Modify the reward before returning it.
        Example: Apply a scaling factor.
        """
        return reward * 0.9  # Scale the reward by 0.9

