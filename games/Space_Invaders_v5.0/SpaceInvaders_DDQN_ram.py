import gymnasium as gym
import ale_py
import torch
import numpy as np
from collections import deque
import time

import sys
sys.path.append('./')

# Import Agents
from agents.ddqn_agent import DDQNAgent

# Import Models
from models.ddqn_ram import DDQNMLP

# Import Custom Reward Modifier Wrapper
from rewards.SpaceInvaders.SpaceInvaders_rewards import RewardModifierWrapper, ComplexRewardModifierWrapper

# Impot Utils
from utils.save_load import save_agent, load_agent

# Initialize Environment
env = gym.make('ALE/SpaceInvaders-ram-v5', frameskip=4)
#env = ComplexRewardModifierWrapper(env)

# Set up Device
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
INPUT_SHAPE = (128,)  # RAM observation shape
ACTION_SIZE = env.action_space.n
GAMMA = 0.99
BUFFER_SIZE = 100000
BATCH_SIZE = 64
LR = 0.0001
TAU = 0.005 # 0.001
UPDATE_EVERY = 8
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 1000

# Initialize Agents
agent = DDQNAgent(INPUT_SHAPE, ACTION_SIZE, 0, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, 5000, DDQNMLP)
#load_agent(agent, device, 'trained_models/SpaceInvaders_DDQN_ram.pth')

# Simplified epsilon function
def epsilon_by_episode(episode):
    return max(EPS_END, EPS_START - (episode / EPS_DECAY))

# Training function (with AMP for FP16)
def train(n_episodes):

    start_time = time.time()  # Start Timer
    scores = []
    scores_window = deque(maxlen=10)

    for i_episode in range(1, n_episodes + 1):

        # state = env.reset()[0].astype(np.float16)  # FP16 - HALF PRECISION
        state = env.reset()[0].astype(np.float32)
        score = 0
        eps = epsilon_by_episode(i_episode)
        
        while True:

            action = agent.act(state, eps)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            #next_state = next_state.astype(np.float16) # FP16 - HALF PRECISION
            next_state = next_state.astype(np.float32)

            done = terminated or truncated
            
            #reward = np.float16(reward)  # FP16 - HALF PRECISION
            reward = np.float32(reward)
            
            agent.step(state, action, reward, next_state, done)  

            state = next_state
            score += reward

            if done:
                break

        scores_window.append(score)
        scores.append(score)

        # Print Progress
        if i_episode % 10 == 0:
            print(f"Episode {i_episode} - Avg Score: {np.mean(scores_window):.2f} - Epsilon: {eps:.2f}")

    end_time = time.time()  # End Timer
    elapsed_time = end_time - start_time
    print(f"\nTotal Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    return scores

# Run Training
train(n_episodes=1000)
save_agent(agent, 'trained_models/SpaceInvaders_DDQN_ram_256.pth')
env.close()
