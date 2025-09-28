import gymnasium as gym
import ale_py
import numpy as np
from gymnasium.utils.play import play

import sys
sys.path.append('./')

# Import Custom Wrappers
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from wrappers.Breakout.rewards import BreakoutRewardWrapper

# Initialize Environment
env = gym.make('ALE/Breakout-v5', render_mode='rgb_array', frameskip=1)
env = AtariPreprocessing(env, frame_skip=2, grayscale_obs=True, scale_obs=False)
env = FrameStackObservation(env, stack_size=4)
#env = BreakoutRewardWrapper(env)

# Key Mapping
controls = {
    "s": 0,  # Noop (Do nothing)
    "w": 1,  # Start
    "d": 2,  # Right
    "a": 3,  # Left
}

# Score tracking variables
episode_score = [0]  # Use list to allow modification inside callback

def callback(obs_t, obs_tp1, action, reward, terminated, truncated, info):
    episode_score[0] += reward
    if terminated or truncated:
        print(f"Score: {episode_score[0]:.2f}")
        episode_score[0] = 0  # Reset score for next episode
    return [reward]  # You can return this for plotter if needed

# Play the environment interactively
play(env, fps=30, keys_to_action=controls, noop=0, wait_on_player=False, callback=callback)

