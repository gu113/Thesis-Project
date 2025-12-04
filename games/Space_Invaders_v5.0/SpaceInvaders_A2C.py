import gymnasium as gym
import ale_py
import torch
import numpy as np
from collections import deque
import time

import sys
sys.path.append('./')

# Import Agents
from agents.a2c_agent import A2CAgent, ComplexA2CAgent

# Import Models
from models.actor_critic_cnn import ActorCnn, CriticCnn

# Import Utils
from utils.save_load import save_agent_a2c, load_agent_a2c
from utils.plots import plot_scores

# Import Custom Reward Modifier Wrapper
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
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
GAMMA = 0.99
ALPHA = 0.0001 # Learning rate for the actor
BETA = 0.001 # Learning rate for the critic
UPDATE_EVERY = 50

# Initialize Agents
agent = ComplexA2CAgent(INPUT_SHAPE, ACTION_SIZE, 0, device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)
#load_agent_a2c(agent, device, 'trained_models/SpaceInvaders_A2C.pth')

# Training function (with AMP for FP16)
def train(n_episodes):

    # Start timing
    start_time = time.time()

    # Score Tracking
    scores = []
    best_score = 0
    episode_rewards = []

    # Score Windows for Averages
    scores_window = deque(maxlen=100)
    original_score_window = deque(maxlen=100)
   
    for i_episode in range(1, n_episodes + 1):
        ##state = torch.tensor(np.array(env.reset()[0]), dtype=torch.float32).to(device)        # TENSORS - FULL PRECISION (FP32)
        state = torch.tensor(np.array(env.reset()[0]), dtype=torch.float16).to(device)          # TENSORS - HALF PRECISION (FP16)

        score = 0
        original_score = 0
        
        while True:
            if torch.cuda.is_available():
                with torch.amp.autocast(device_type="cuda"): 
                    action, log_prob, value = agent.act(state)
            else:
                action, log_prob, value = agent.act(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ##next_state = torch.tensor(np.array(next_state), dtype=torch.float32).to(device)   # TENSORS - FULL PRECISION (FP32)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float16).to(device)     # TENSORS - HALF PRECISION (FP16)

            agent.step(state, action, reward, next_state, done, log_prob, value)

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
        #print(f"Episode {i_episode:5d} | Score: {score:.2f}    | Original Score: {original_score:.2f}")

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
train(n_episodes=1000)
#save_agent_a2c(agent, device, 'trained_models/SpaceInvaders_A2C.pth')

# Close Environment
env.close()
