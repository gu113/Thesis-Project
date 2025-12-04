import gymnasium as gym
import ale_py
import torch
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt
import random

import sys
sys.path.append('./')

# Import Rainbow Agent
from agents.rdqn_agent import RainbowDQNAgent, ComplexRainbowDQNAgent

# Import Utils
from utils.save_load import save_agent_rdqn, load_agent_rdqn
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
INPUT_SHAPE = (4, 84, 84)           # Output shape
ACTION_SIZE = env.action_space.n
GAMMA = 0.99                        # Discount factor
BUFFER_SIZE = 200000                # Rainbow paper default: 1M (for PER)
BATCH_SIZE = 32                     # Rainbow paper default: 32
LR = 0.0000625                      # Rainbow paper default: 6.25e-5

# Rainbow Specific Hyperparameters
TARGET_UPDATE_FREQ = 2000           # How often to update target network (in agent steps)
N_STEP_RETURNS = 3                  # N-step bootstrapping
ALPHA = 0.5                         # PER alpha exponent
BETA_START = 0.4                    # PER beta exponent start
BETA_ANNEAL_FRAMES = 5000000       # Frames over which beta anneals

NUM_ATOMS = 51                      # C51: number of atoms in value distribution
V_MIN = -10                         # C51: minimum value for distribution support
V_MAX = 10                          # C51: maximum value for distribution support

LEARNING_STARTS = 7500              # Number of random steps before learning begins
MAX_FRAMES = 5000000               # Total frames to train for

# Initialize Agent
agent = ComplexRainbowDQNAgent(INPUT_SHAPE, ACTION_SIZE, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TARGET_UPDATE_FREQ, N_STEP_RETURNS, ALPHA, BETA_START, BETA_ANNEAL_FRAMES, NUM_ATOMS, V_MIN, V_MAX, LEARNING_STARTS)
#load_agent_rdqn(agent, device, 'trained_models/SpaceInvaders_RDQN.pth')

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
        
        # Reset env and get first observation as a normalized NumPy array
        observation_np, _ = env.reset()
        observation_normalized = observation_np.astype(np.float32) / 255.0
        
        observation_tensor = torch.from_numpy(observation_normalized).to(device)

        score = 0
        original_score = 0
        done = False
        truncated = False
        
        agent.q_net.reset_noise() 

        while not done and not truncated:

            # Choose action from the GPU-based tensor
            with torch.amp.autocast(device_type="cuda"):
                action = agent.choose_action(observation_tensor)
            
            # Step the environment
            next_observation_np, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Process the new observation
            next_observation_normalized = next_observation_np.astype(np.float32) / 255.0
            
            # Add the transition to the buffer
            agent.step(observation_normalized, action, reward, next_observation_normalized, done) 

            if total_frames > agent.learning_starts:
                loss_value = None
                with torch.amp.autocast(device_type="cuda"):
                    loss_value = agent.learn()
                
                if loss_value is not None:
                    learn_steps += 1
            
            # Update for next loop iteration
            observation_normalized = next_observation_normalized

            # Create a new tensor for the next action choice
            observation_tensor = torch.from_numpy(observation_normalized).to(device)
            
            total_frames += 4
            
            score += reward
            original_score = info['original_score']
            
            if total_frames >= MAX_FRAMES:
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
#save_agent_rdqn(agent, 'trained_models/SpaceInvaders_RDQN.pth') # Trained for: 10k Episodes           Average Human Score: 1670        Open AI DDQN: 2628 Score in 50k Episodes

# Close Environment
env.close()