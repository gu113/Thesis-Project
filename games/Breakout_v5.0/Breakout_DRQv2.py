import gymnasium as gym
import ale_py
import torch
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt

import sys
sys.path.append('./')

# Import Agents
from agents.drqv2_agent import DrQv2Agent

# Import Utils
from utils.plots import plot_scores

# Import Custom Wrappers
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from wrappers.Breakout.rewards import BreakoutRewardWrapper
from wrappers.Breakout.fire_reset import FireOnLifeLoss

# Initialize Environment
env = gym.make('ALE/Breakout-v5', frameskip=1)
env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
env = BreakoutRewardWrapper(env)
env = FrameStackObservation(env, stack_size=4)
env = FireOnLifeLoss(env)

# Set up Device
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
GAMMA = 0.99
BUFFER_SIZE = 100000 # DrQ-v2 typically uses 500000 or 1000000
BATCH_SIZE = 128     # DrQ-v2 typically uses 128 or 256
LR = 0.0001
TAU = 0.01
UPDATE_EVERY = 4
NUM_SEED_STEPS = 5000
STDDEV_SCHEDULE = 'linear(1.0,0.1,100000)'

# Initialize Agents
agent = DrQv2Agent(state_size=INPUT_SHAPE, action_size=ACTION_SIZE, device=device, lr=LR, gamma=GAMMA, tau=TAU, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE, update_every_steps=UPDATE_EVERY, num_seed_steps=NUM_SEED_STEPS, stddev_schedule=STDDEV_SCHEDULE)
#agent.load('trained_models/Breakout_DrQv2.pth')

# Training Function
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

        if torch.cuda.is_available():
            ##state = torch.from_numpy(env.reset()[0]).to(torch.float32)              # FULL PRECISION (FP32)
            ##state = torch.from_numpy(env.reset()[0]).to(torch.float16)              # HALF PRECISION (FP16)
            ##state = torch.tensor(env.reset()[0], dtype=torch.float32)               # TENSORS - FULL PRECISION (FP32)
            state = torch.tensor(env.reset()[0], dtype=torch.float16)                 # TENSORS - HALF PRECISION (FP16)
        else:
            ##state = torch.from_numpy(env.reset()[0]).to(torch.float32)              # FULL PRECISION (FP32)
            state = torch.from_numpy(env.reset()[0]).to(torch.float16)                # HALF PRECISION (FP16)

        score = 0
        original_score = 0

        while True:

            with torch.amp.autocast("cuda"):
                action = agent.act(state, training=True)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if torch.cuda.is_available():
                ##next_state =torch.from_numpy(next_state).to(torch.float32)            # FULL PRECISION (FP32)
                ##next_state = torch.from_numpy(next_state).to(torch.float16)           # HALF PRECISION (FP16)
                ##next_state = torch.tensor(next_state, dtype=torch.float32)            # TENSORS - FULL PRECISION (FP32)
                next_state = torch.tensor(next_state, dtype=torch.float16)              # TENSORS - HALF PRECISION (FP16)
            else:
                ##next_state = torch.from_numpy(next_state).to(torch.float32)           # FULL PRECISION (FP32)
                next_state = torch.tensor(next_state, dtype=torch.float16)              # HALF PRECISION (FP16)

            agent.step(state, action, reward, next_state, done)  

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
            print(f"New Highscore: {best_score} (Original Score: {original_best_score}) on episode {i_episode}")

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
agent.save('trained_models/Breakout_DrQv2.pth')     # Open AI DDQN: 418 Score in 50k Episodes

# Close Environment
env.close()
