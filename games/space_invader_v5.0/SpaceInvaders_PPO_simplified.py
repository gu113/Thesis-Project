import gymnasium as gym
import ale_py
import torch
import numpy as np
from collections import deque
import time

import sys
sys.path.append('./')

# Import Agents
from agents.ppo_agent import PPOAgent

# Import Models
from models.actor_critic_cnn import ActorCnn, CriticCnn

# Import Utils
from utils.stack_frame import preprocess_frame, stack_frame

# Import Custom Reward Modifier Wrapper
from rewards.SpaceInvaders.SpaceInvaders_rewards import RewardModifierWrapper

# Initialize Environment
env = gym.make('ALE/SpaceInvaders-v5', frameskip=4)
env = RewardModifierWrapper(env)
#env.unwrapped.seed(0)

# Set up Device
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (8, -12, -12, 4), 84)
    #frame = frame.astype(np.float16) / 255.0  # Convert to FP16 and normalize
    frames = stack_frame(frames, frame, is_new)

    return frames

# Hyperparameters
INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
GAMMA = 0.99
ALPHA = 0.0001 # Learning rate for the actor
BETA = 0.001 # Learning rate for the critic
TAU = 0.95
BATCH_SIZE = 128
PPO_EPOCH = 5
CLIP_PARAM = 0.2
UPDATE_EVERY = 50


# Initialize Agents
agent = PPOAgent(INPUT_SHAPE, ACTION_SIZE, 0, device, GAMMA, ALPHA, BETA, TAU, UPDATE_EVERY, BATCH_SIZE, PPO_EPOCH, CLIP_PARAM, ActorCnn, CriticCnn)

# Training function (with AMP for FP16)
def train(n_episodes):

    start_time = time.time()  # Start timing
    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        #state = stack_frames(None, env.reset()[0], True) # Original
        state = stack_frames(None, env.reset()[0], True).astype(np.float16) # FP16 - HALF PRECISION
        score = 0
        
        while True:

            # Use FP16 for model training
            with torch.amp.autocast("cuda"):  # Enable FP16
                action, log_prob, value = agent.act(state)
            
            # No FP16 needed for env interaction
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            #next_state = stack_frames(state, next_state, False) # Original
            next_state = stack_frames(state, next_state, False).astype(np.float16) # FP16 - HALF PRECISION

            reward = np.float16(reward)  # FP16 - HALF PRECISION
            agent.step(state, action, value, log_prob, reward, done, next_state) 

            state = next_state
            score += reward

            if done:
                break

        scores_window.append(score)
        scores.append(score)

        # Print progress instead of plotting
        if i_episode % 10 == 0:
            print(f"Episode {i_episode} - Avg Score: {np.mean(scores_window):.2f}")

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    print(f"\nTotal Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    return scores

# Run Training
train(n_episodes=1000)

env.close()
