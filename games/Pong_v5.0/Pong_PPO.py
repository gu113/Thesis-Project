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
from agents.ppo_agent import PPOAgent, ComplexPPOAgent

# Import Models
from models.actor_critic_cnn import PPOActorCnn, PPOCriticCnn

# Import Utils
from utils.save_load import save_agent_ppo, load_agent_ppo
from utils.plots import plot_scores_pong

# Import Custom Wrappers
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from wrappers.Pong.rewards import PongScoreWrapper
from wrappers.Pong.fire_reset import FireResetEnv

PONG_MAX_SCORE = 11

# Initialize Environment
env = gym.make('ALE/Pong-v5', frameskip=1)
env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
env = FrameStackObservation(env, stack_size=4)
env = FireResetEnv(env)
env = PongScoreWrapper(env, PONG_MAX_SCORE)

# Set up Device
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
GAMMA = 0.99
LR_ACTOR = 0.00025      # Actor learning rate
LR_CRITIC = 0.00025     # Critic learning rate
GAE_LAMBDA = 0.95       # General Advantage Estimation Lambda
PPO_EPOCHS = 10         # Number of PPO epochs to run per policy update
CLIP_PARAM = 0.2        # PPO clipping parameter (epsilon)
BATCH_SIZE = 128        # Batch size for PPO updates
UPDATE_EVERY = 2048     # Common value, usually a power of 2

# Initialize Agents
agent = ComplexPPOAgent(INPUT_SHAPE, ACTION_SIZE, 0, device, GAMMA, LR_ACTOR, LR_CRITIC, GAE_LAMBDA, UPDATE_EVERY, BATCH_SIZE, PPO_EPOCHS, CLIP_PARAM, PPOActorCnn, PPOCriticCnn)
load_agent_ppo(agent, device, 'trained_models/Pong_PPO_Embedded.pth')

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
        
        ##state = torch.tensor(np.array(env.reset()[0]), dtype=torch.float32).to(device)        # TENSORS - FULL PRECISION (FP32)
        state = torch.tensor(np.array(env.reset()[0]), dtype=torch.float16).to(device)          # TENSORS - HALF PRECISION (FP16)

        score = 0
        
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
        #print(f"Episode {i_episode:5d} | Agent - {agent_score} x {opponent_score} - Opponent")

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
train(n_episodes=2500)
save_agent_ppo(agent, 'trained_models/Pong_PPO_Embedded.pth')     #Target Score: Agent 21 x 0 Opponent

# Close Environment
env.close()
