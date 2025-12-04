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
from agents.reinforce_agent import REINFORCEAgent, ComplexREINFORCEAgent

# Import Models
from models.actor_critic_cnn import REINFORCEActorCnn

# Import Utils
from utils.save_load import save_agent_REINFORCE, load_agent_REINFORCE
from utils.plots import plot_scores_pong

# Import Custom Wrappers
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from wrappers.Pong.rewards import PongScoreWrapper
from wrappers.Pong.fire_reset import FireResetEnv

PONG_MAX_SCORE = 21

# Initialize Environment
env = gym.make('ALE/Pong-v5', frameskip=1)
env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
env = FrameStackObservation(env, stack_size=4)
env = FireResetEnv(env)
env = PongScoreWrapper(env)

# Set up Device
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
GAMMA = 0.99
LR_ACTOR = 0.00025

# Initialize Agents
agent = ComplexREINFORCEAgent(INPUT_SHAPE, ACTION_SIZE, 0, device, GAMMA, LR_ACTOR, REINFORCEActorCnn)
#load_agent_REINFORCE(agent, device, 'trained_models/Pong_REINFORCE.pth')

# Training Function
def train(n_episodes):

    # Start Timing
    start_time = time.time()

    # Score Tracking
    scores = []
    agent_scores = []
    opponent_scores = []
    best_score = 0
    worst_opponent_score = PONG_MAX_SCORE

    # Score Windows for Averages
    score_window = deque(maxlen=10)
    agent_score_window = deque(maxlen=10)
    opponent_score_window = deque(maxlen=10)
    

    for i_episode in range(1, n_episodes + 1):
        
        state_raw = env.reset()[0].astype(np.float32) / 255.0
        
        if torch.cuda.is_available():
            ##state = torch.from_numpy(state_raw).to(torch.float32).to(device)     # FULL PRECISION (FP32)
            ##state = torch.from_numpy(state_raw).to(torch.float16).to(device)     # HALF PRECISION (FP16)
            ##state = torch.tensor(state_raw, dtype=torch.float32).to(device)      # TENSORS - FULL PRECISION (FP32)
            state = torch.tensor(state_raw, dtype=torch.float16).to(device)        # TENSORS - HALF PRECISION (FP16)
        else:
            ##state = torch.from_numpy(state_raw).to(torch.float32)                # FULL PRECISION (FP32)
            state = torch.from_numpy(state_raw).to(torch.float16)                  # HALF PRECISION (FP16)

        if len(state.shape) == 3:
            state = state.unsqueeze(0)

        score = 0
        
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
                ##next_state = torch.tensor(next_state_raw, dtype=torch.float32).to(device)            # TENSORS - FULL PRECISION (FP32)
                next_state = torch.tensor(next_state_raw, dtype=torch.float16).to(device)              # TENSORS - HALF PRECISION (FP16)
            else:
                ##next_state = torch.from_numpy(next_state_raw).to(torch.float32)                      # FULL PRECISION (FP32)
                next_state = torch.from_numpy(next_state_raw).to(torch.float16)                        # HALF PRECISION (FP16)

            if len(next_state.shape) == 3:
                next_state = next_state.unsqueeze(0)

            agent.step(log_prob, reward, done) 

            state = next_state
            score += reward

            if done:
                agent.learn()
                break

        # Get Game Scores from Wrapper
        agent_score, opponent_score = env.get_scores()

        # Append Scores & Windows
        scores.append(score)
        agent_scores.append(agent_score)
        opponent_scores.append(opponent_score)

        score_window.append(score)
        agent_score_window.append(agent_score)
        opponent_score_window.append(opponent_score)

        # Print Scores
        print(f"Episode {i_episode:5d} | Agent - {agent_score} x {opponent_score} - Opponent")

        # Track High Scores
        if agent_score > best_score or opponent_score < worst_opponent_score:
            best_score = agent_score
            worst_opponent_score = opponent_score
            print(f"New Highscore | Agent - {agent_score} x {opponent_score} - Opponent on episode {i_episode}")

        # Print Progress
        if i_episode % 100 == 0 or i_episode % 250 == 0:
            print(f"Episode {i_episode:5d} | Avg Agent Score: {np.mean(agent_score_window):.2f} x Avg Opponent Score: {np.mean(opponent_score_window):.2f}")

    # End Timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Highest Score Achieved: Agent - {best_score} x {worst_opponent_score} - Opponent")

    # Plot Progress
    plot_scores_pong(agent_scores[-2500:], opponent_scores[-2500:], i_episode)

    return scores

# Run Training
train(n_episodes=1000)
#save_agent_REINFORCE(agent, 'trained_models/Pong_REINFORCE.pth')     #Target Score: Agent 21 x 0 Opponent

# Close Environment
env.close()