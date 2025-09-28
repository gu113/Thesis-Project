import gymnasium as gym
import numpy as np
from gymnasium import Wrapper

class BreakoutRewardWrapper(Wrapper):
    def __init__(self, env):
        super(BreakoutRewardWrapper, self).__init__(env)
        self.last_lives = 5 # Initial lives for Breakout
        self.original_score = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_lives = self.env.unwrapped.ale.lives()
        self.original_score = 0
        
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.original_score += reward

        shaped_reward = 0

        # Get current lives from the info dictionary
        current_lives = self.env.unwrapped.ale.lives()

        # Ball hit brick (environment gives +1 to +5)
        if reward > 0:
            shaped_reward += 1

        # Life lost (missed ball)
        if current_lives < self.last_lives:
            shaped_reward -= 1

        # Small time-alive bonus (encourages longer play)
        shaped_reward += 0.0001

        # Update last_lives for the next step's comparison
        self.last_lives = current_lives

        info['original_score'] = self.original_score

        return obs, shaped_reward, terminated, truncated, info