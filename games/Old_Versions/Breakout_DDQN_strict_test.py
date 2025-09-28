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
from agents.ddqn_agent import DDQNAgent, DDQNAgentPER, ComplexDDQNAgentPER

# Import Models
from models.ddqn_cnn import DDQNCnn, ComplexDDQNCnn

# Import Utils
from utils.stack_frame import preprocess_frame, stack_frame
from utils.save_load import save_agent, load_agent, save_features, load_features

# Import Custom Wrappers
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from wrappers.Breakout.rewards import BreakoutRewardWrapper
from wrappers.Breakout.fire_reset import FireOnLifeLoss

# Initialize Environment
env = gym.make('ALE/Breakout-v5', frameskip=1) # Average Human Score: Around 31 points (based on reports from DeepMind and OpenAI Gym).
env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
env = FrameStackObservation(env, stack_size=4)
env = FireOnLifeLoss(env)
#env = BreakoutRewardWrapper(env)

# Set up Device
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to stack frames for the agent input
def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (8, -12, -12, 4), 84)
    
    if torch.cuda.is_available():
        ##frame = torch.from_numpy(frame).to(torch.float32).to(device) / 255.0 # float32 with GPU
        frame = torch.from_numpy(frame).to(torch.float16).to(device) / 255.0 # float16 with GPU
    else:
        ##frame = frame.astype(np.float32) / 255.0  # Use float32 for better precision
        frame = frame.astype(np.float16) / 255.0  # Use float16 for better speed
    
    frames = stack_frame(frames, frame, is_new)

    return frames

"""
def plot_scores(scores, episode_num):
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title(f'Training Progress - Episodes {episode_num-len(scores)+1} to {episode_num}')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    plt.savefig(f'training_progress_{episode_num}.png')
    plt.close()
"""
# Hyperparameters
INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
GAMMA = 0.99
BUFFER_SIZE = 100000 #100k - 7GB RAM, 200k - 12+GB RAM
BATCH_SIZE = 64
LR = 0.00001
TAU = 0.001
UPDATE_EVERY = 4
EPS_START = 0.99
EPS_END = 0.01
EPS_DECAY = 500

# Initialize Agents
agent = DDQNAgentPER(INPUT_SHAPE, ACTION_SIZE, 0, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, 10000, DDQNCnn)
load_features(agent, device, 'trained_models/Pong_strict_test_features.pth')
print(f"Loaded features {list(agent.policy_net.features[0].parameters())}")

# Linear Epsilon Decay Function
def epsilon_by_episode_linear(episode):
    return max(EPS_END, EPS_START - (episode / EPS_DECAY))

# Exponential Epsilon Decay Function
def epsilon_by_episode_exponential(episode):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-episode / EPS_DECAY)

# Training Function
def train(n_episodes):

    # Start Timing
    start_time = time.time()

    # Score Tracking
    scores = []
    best_score = 0
    ##episode_rewards = []

    # Score Windows for Averages
    scores_window = deque(maxlen=10)


    for i_episode in range(1, n_episodes + 1):

        if torch.cuda.is_available():
            ##state = stack_frames(None, env.reset()[0], True).to(torch.float32)    # FULL PRECISION (FP32)
            ##state = stack_frames(None, env.reset()[0], True).to(torch.float16)    # HALF PRECISION (FP16)
            #state = torch.tensor(env.reset()[0], dtype=torch.float32)              # TENSORS - FULL PRECISION (FP32)
            state = torch.tensor(env.reset()[0], dtype=torch.float16)               # TENSORS - FULL PRECISION (FP16)
        else:
            ##state = stack_frames(None, env.reset()[0], True)                      # FULL PRECISION (FP32)
            state = stack_frames(None, env.reset()[0], True).astype(np.float16)     # HALF PRECISION (FP16)

        score = 0
        eps = epsilon_by_episode_exponential(i_episode)
        
        while True:

            with torch.amp.autocast("cuda"):  # AUTOCAST - HALF PRECISION (FP16)
                action = agent.act(state, eps)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if torch.cuda.is_available():
                ##next_state = stack_frames(state, next_state, False).to(torch.float32)     # FULL PRECISION (FP32)
                #next_state = stack_frames(state, next_state, False).to(torch.float16)      # HALF PRECISION (FP16)
                #next_state = torch.tensor(next_state, dtype=torch.float32)                 # TENSORS - FULL PRECISION (FP32)
                next_state = torch.tensor(next_state, dtype=torch.float16)                  # TENSORS - HALF PRECISION (FP16)

            else:
                ##next_state = stack_frames(state, next_state, False) # FULL PRECISION (FP32)
                next_state = stack_frames(state, next_state, False).astype(np.float16) # HALF PRECISION (FP16)

            agent.step(state, action, reward, next_state, done)  

            state = next_state
            score += reward

            if done:
                break

        # Append Scores & Windows
        scores.append(score)
        ##episode_rewards.append(score)
        
        scores_window.append(score)

        # Print Scores
        print(f"Episode {i_episode:5d} | Score: {score:.2f}")

        # Track High Scores
        if score > best_score:
            best_score = score
            print(f"New Highscore: {best_score} on episode {i_episode}")

        # Print Progress
        if i_episode % 10 == 0:
            print(f"Episode {i_episode:5d} | Avg Score: {np.mean(scores_window):.2f} | Epsilon: {eps:.2f}")

            """
            # Plot Progress
            if i_episode % n_episodes == 0:
                plot_scores(episode_rewards[-1000:], i_episode)
            """

    # End Timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Highest Score Achieved: {best_score}")

    return scores

# Run Training
train(n_episodes=100)
#save_agent(agent, 'trained_models/Breakout_strict_test.pth') # Trained for: 10k Episodes         Open AI DDQN: 418 Score in 50k Episodes

# Close Environment
env.close()
