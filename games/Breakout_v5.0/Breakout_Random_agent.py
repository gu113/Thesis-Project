import gymnasium as gym
import ale_py
import torch
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt

import sys
sys.path.append('./')

# Import Utils
from utils.save_load import save_agent, load_agent
from utils.plots import plot_scores

# Import Custom Wrappers
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from wrappers.Breakout.rewards import BreakoutRewardWrapper
from wrappers.Breakout.fire_reset import FireOnLifeLoss

# Initialize Environment
env = gym.make('ALE/Breakout-v5', frameskip=1) # Average Human Score: Around 31 points (based on reports from DeepMind and OpenAI Gym).
env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
env = BreakoutRewardWrapper(env)
env = FrameStackObservation(env, stack_size=4)
env = FireOnLifeLoss(env)

# Set up Device
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(n_episodes):

    # Start Timing
    start_time = time.time()

    # Score Tracking
    scores = []
    best_score = 0
    original_best_score = 0
    episode_rewards = []

    # Score Windows for Averages
    scores_window = deque(maxlen=100)
    original_score_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):

        state = env.reset()[0]

        score = 0
        original_score = 0

        while True:
            
            action = env.action_space.sample()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = next_state
            score += reward
            original_score = info['original_score']

            if done:
                break

        # Append Scores & Windows
        scores.append(score)
        episode_rewards.append(original_score)
        
        scores_window.append(score)
        original_score_window.append(original_score)

        # Print Scores
        print(f"Episode {i_episode:5d} | Score: {score:.2f}     | Original Score: {original_score:.2f}")

        # Track High Scores
        if score > best_score:
            best_score = score
            original_best_score = original_score
            print(f"New Highscore: {best_score} (Game Score: {original_best_score}) on episode {i_episode}")
        
        # Print Progress
        if i_episode % 10 == 0:
            print(f"Episode {i_episode:5d} | Avg Score: {np.mean(scores_window):.2f} | Avg Original Score: {np.mean(original_score_window):.2f}")
            
            # Plot Progress
            if i_episode % n_episodes == 0:
                plot_scores(episode_rewards[-1000:], i_episode)

    # End Timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Highest Score Achieved: {best_score}")
    print(f"Highest Original Score Achieved: {original_best_score}")

    return scores

# Run Training
train(n_episodes=1000)

# Close Environment
env.close()