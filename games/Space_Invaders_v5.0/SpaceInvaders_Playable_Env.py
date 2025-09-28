import gymnasium as gym
import ale_py
import numpy as np
from gymnasium.utils.play import play

import sys
sys.path.append('./')

# Import Custom Reward Modifier Wrapper
from wrappers.SpaceInvaders.rewards import RewardModifierWrapper

# Initialize Environment
env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array', frameskip=1)
env = RewardModifierWrapper(env)

# Print action space to check available actions
print("Action Space:", env.action_space)

# Key Mapping
controls = {
    "s": 0,  # Noop (Do nothing)
    "w": 1,  # Fire
    "d": 2,  # Right
    "a": 3,  # Left
    "wd": 4,  # Right + Fire
    "wa": 5,  # Left + Fire
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
play(env, fps=60, keys_to_action=controls, noop=0, wait_on_player=False, callback=callback)

