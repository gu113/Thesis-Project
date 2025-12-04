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
from utils.save_load import save_agent, load_agent
from utils.plots import plot_scores_pong

# Import Custom Wrappers
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from wrappers.Pong.rewards import PongScoreWrapper
from wrappers.Pong.fire_reset import FireResetEnv

# Initialize Environment
env = gym.make('ALE/Pong-v5', frameskip=1)
env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
env = FrameStackObservation(env, stack_size=4)
env = FireResetEnv(env)
env = PongScoreWrapper(env)

PONG_MAX_SCORE = 11

# Set up Device
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
GAMMA = 0.99
BUFFER_SIZE = 200000 # DrQ-v2 typically uses 500000 or 1000000
BATCH_SIZE = 64     # DrQ-v2 typically uses 128 or 256
LR = 0.0001
TAU = 0.01
UPDATE_EVERY = 4
NUM_SEED_STEPS = 5000
STDDEV_SCHEDULE = 'linear(1.0,0.1,100000)'

# Initialize Agents
agent = DrQv2Agent(state_size=INPUT_SHAPE, action_size=ACTION_SIZE, device=device, lr=LR, gamma=GAMMA, tau=TAU, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE, update_every_steps=UPDATE_EVERY, num_seed_steps=NUM_SEED_STEPS, stddev_schedule=STDDEV_SCHEDULE)
#load_agent(agent, device, 'trained_models/Pong_DrQv2.pth')

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
    score_window = deque(maxlen=100)
    agent_score_window = deque(maxlen=100)
    opponent_score_window = deque(maxlen=100)


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

            if done:
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
train(n_episodes=2000)
#save_agent(agent, 'trained_models/Pong_DrQv2.pth')     #Target Score: Agent 21 x 0 Opponent

# Close Environment
env.close()
