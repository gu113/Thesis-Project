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
from agents.dqn_agent import DQNAgent

# Import Models
from models.dqn_cnn import DQNCnn

# Import Utils
from utils.save_load import save_agent, load_agent
from utils.plots import plot_scores

# Import Custom Wrappers
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from wrappers.SpaceInvaders.rewards import ComplexRewardModifierWrapper
from wrappers.SpaceInvaders.noop_reset import NoopResetEnv

# Initialize Environment
env = gym.make('ALE/SpaceInvaders-v5', frameskip=1)
env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
env = FrameStackObservation(env, stack_size=4)
env = ComplexRewardModifierWrapper(env)
#env = NoopResetEnv(env, noop_max=30)

# Set up Device
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
GAMMA = 0.99
BUFFER_SIZE = 25000
BATCH_SIZE = 64
LR = 0.00025
TAU = 0.001
UPDATE_EVERY = 4
EPS_START = 0.99
EPS_END = 0.01
EPS_DECAY = 500

# Initialize Agents
agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, 0, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, 10000, DQNCnn)
load_agent(agent, device, 'trained_models/SpaceInvaders_DQN_Embedded.pth')

# Linear Epsilon Decay Function
def epsilon_by_episode_linear(episode):
    return max(EPS_END, EPS_START - (episode / EPS_DECAY))

# Exponential Epsilon Decay Function
def epsilon_by_episode_exponential(episode):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-episode / EPS_DECAY)

# Training Function
def train(n_episodes):

    ## Start Timing
    start_time = time.time()

    ## Score Tracking
    scores = []
    best_score = 0
    original_best_score = 0
    episode_rewards = []

    # Score Windows for Averages
    scores_window = deque(maxlen=100)
    original_score_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):

        if torch.cuda.is_available():
            ##state = stack_frames(None, env.reset()[0], True).to(torch.float32)    # FULL PRECISION (FP32)
            ##state = stack_frames(None, env.reset()[0], True).to(torch.float16)    # HALF PRECISION (FP16)
            ##state = torch.tensor(env.reset()[0], dtype=torch.float32)              # TENSORS - FULL PRECISION (FP32)
            state = torch.tensor(env.reset()[0], dtype=torch.float16)               # TENSORS - FULL PRECISION (FP16)
        else:
            ##state = torch.from_numpy(env.reset()[0]).to(torch.float32)                      # FULL PRECISION (FP32)
            state = torch.from_numpy(env.reset()[0]).to(torch.float16)     # HALF PRECISION (FP16)

        score = 0
        original_score = 0

        #eps = epsilon_by_episode_linear(i_episode)
        eps = epsilon_by_episode_exponential(i_episode)
        
        while True:

            with torch.amp.autocast("cuda"):  # AUTOCAST - HALF PRECISION (FP16)
                action = agent.act(state, eps)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if torch.cuda.is_available():
                ##next_state =torch.from_numpy(next_state).to(torch.float32)     # FULL PRECISION (FP32)
                #next_state =torch.from_numpy(next_state).to(torch.float16)      # HALF PRECISION (FP16)
                #next_state = torch.tensor(next_state, dtype=torch.float32)                 # TENSORS - FULL PRECISION (FP32)
                next_state = torch.tensor(next_state, dtype=torch.float16)                  # TENSORS - HALF PRECISION (FP16)

            else:
                ##next_state =torch.from_numpy(next_state).to(torch.float32) # FULL PRECISION (FP32)
                next_state =torch.from_numpy(next_state).to(torch.float16) # HALF PRECISION (FP16)

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
        #print(f"Episode {i_episode:5d} | Epsilon: {eps:.4f}     | Score: {score:.2f}     | Original Score: {original_score:.2f}")

        # Track High Scores
        if original_score > best_score:
            best_score = original_score
            print(f"New Highscore: {best_score} on episode {i_episode}")
        
        # Print Progress
        if i_episode % 100 == 0 or i_episode % 250 == 0:
            print(f"Episode {i_episode:5d} | Avg Score: {np.mean(scores_window):.2f} | Avg Original Score: {np.mean(original_score_window):.2f}")

    # End Timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Highest Score Achieved: {best_score}")

    # Plot Progress
    plot_scores(episode_rewards[-2500:], i_episode)

    return scores

# Run Training
train(n_episodes=2500)
save_agent(agent, 'trained_models/SpaceInvaders_DQN_Embedded.pth') # Trained for: 10k Episodes           Average Human Score: 1670        Open AI DDQN: 2628 Score in 50k Episodes

# Close Environment
env.close()
