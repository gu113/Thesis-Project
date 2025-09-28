import gymnasium as gym
import ale_py
import torch
import numpy as np
from collections import deque
import time

import sys
sys.path.append('./')

# Import Agents
from agents.reinforce_agent import REINFORCEAgent, ComplexREINFORCEAgent

# Import Models
from models.actor_critic_cnn import REINFORCEActorCnn

# Import Utils
from utils.save_load import save_agent_REINFORCE, load_agent_REINFORCE
from utils.plots import plot_scores

# Import Custom Wrappers
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from wrappers.Breakout.rewards import BreakoutRewardWrapper
from wrappers.Breakout.fire_reset import FireOnLifeLoss

# Initialize Environment
env = gym.make('ALE/Breakout-v5', frameskip=1)
env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
env = BreakoutRewardWrapper(env)
env = FireOnLifeLoss(env)
env = FrameStackObservation(env, stack_size=4)

# Set up Device
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
GAMMA = 0.99
LR_ACTOR = 0.0001  # 0.00025 (Learning Collapes by Episode 300)

# Initialize Agents
agent = ComplexREINFORCEAgent(INPUT_SHAPE, ACTION_SIZE, 0, device, GAMMA, LR_ACTOR, REINFORCEActorCnn)
#load_agent_REINFORCE(agent, device, 'trained_models/Breakout_REINFORCE.pth')

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

        state_raw, info = env.reset()
        state_raw = state_raw.astype(np.float32) / 255.0
        
        if torch.cuda.is_available():
            ##state = torch.from_numpy(state_raw).to(torch.float32).to(device)      # FULL PRECISION (FP32)
            ##state = torch.from_numpy(state_raw).to(torch.float16).to(device)      # HALF PRECISION (FP16)
            ##state = torch.tensor(state_raw, dtype=torch.float32).to(device)      # TENSORS - FULL PRECISION (FP32)
            state = torch.tensor(state_raw, dtype=torch.float16).to(device)        # TENSORS - HALF PRECISION (FP16)
        else:
            ##state = torch.from_numpy(state_raw).to(torch.float32)                # FULL PRECISION (FP32)
            state = torch.from_numpy(state_raw).to(torch.float16)                 # HALF PRECISION (FP16)

        if len(state.shape) == 3:
            state = state.unsqueeze(0)

        score = 0
        original_score = 0
        
        while True:
            
            if torch.cuda.is_available():
                with torch.amp.autocast(device_type="cuda"): 
                    action, log_prob = agent.act(state) 
            else:
                action, log_prob = agent.act(state) 
            
            next_state_raw, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state_raw = next_state_raw.astype(np.float32) / 255.0

            if torch.cuda.is_available():
                ##next_state = torch.from_numpy(next_state_raw).to(torch.float32).to(device)           # FULL PRECISION (FP32)
                ##next_state = torch.from_numpy(next_state_raw).to(torch.float16).to(device)           # HALF PRECISION (FP16)
                ##next_state = torch.tensor(next_state_raw, dtype=torch.float32).to(device)           # TENSORS - FULL PRECISION (FP32)
                next_state = torch.tensor(next_state_raw, dtype=torch.float16).to(device)            # TENSORS - HALF PRECISION (FP16)
            else:
                ##next_state = torch.from_numpy(next_state_raw).to(torch.float32)                      # FULL PRECISION (FP32)
                next_state = torch.from_numpy(next_state_raw).to(torch.float16)                      # HALF PRECISION (FP16)

            if len(next_state.shape) == 3:
                next_state = next_state.unsqueeze(0)

            agent.step(log_prob, reward, done) 

            state = next_state
            score += reward
            original_score = info['original_score']

            if done:
                agent.learn()
                break

        # Append Scores & Windows
        scores.append(score)
        episode_rewards.append(original_score)

        scores_window.append(score)
        original_score_window.append(original_score)

        # Print Scores
        print(f"Episode {i_episode:5d} | Score: {score:.2f}    | Original Score: {original_score:.2f}")

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
#save_agent_REINFORCE(agent, 'trained_models/Breakout_REINFORCE.pth')      # Open AI DDQN: 418 Score in 50k Episodes

# Close Environment
env.close()