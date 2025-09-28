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
from agents.rdqn_agent import RainbowDQNAgent, ComplexRainbowDQNAgent

# Import Utils
from utils.save_load import save_agent, load_agent

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

# Set up Device
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
INPUT_SHAPE = (4, 84, 84)           # Output shape from FrameStackObservation
ACTION_SIZE = env.action_space.n
GAMMA = 0.99                        # Discount factor
BUFFER_SIZE = 200000                # Rainbow paper default: 1M (for PER)
BATCH_SIZE = 32                     # Rainbow paper default: 32
LR = 0.0001                         # Rainbow paper default: 6.25e-5

# Rainbow Specific Hyperparameters
TARGET_UPDATE_FREQ = 2000           # How often to update target network (in agent steps)
N_STEP_RETURNS = 3                  # N-step bootstrapping
ALPHA = 0.5                         # PER alpha exponent
BETA_START = 0.4                    # PER beta exponent start
BETA_ANNEAL_FRAMES = 50000000       # Frames over which beta anneals

NUM_ATOMS = 51                      # C51: number of atoms in value distribution
V_MIN = -10                         # C51: minimum value for distribution support
V_MAX = 10                          # C51: maximum value for distribution support

LEARNING_STARTS = 7500              # Number of random steps before learning begins
MAX_FRAMES = 50000000               # Total frames to train for

# Initialize Agent
agent = ComplexRainbowDQNAgent(INPUT_SHAPE, ACTION_SIZE, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TARGET_UPDATE_FREQ, N_STEP_RETURNS, ALPHA, BETA_START, BETA_ANNEAL_FRAMES, NUM_ATOMS, V_MIN, V_MAX, LEARNING_STARTS)
# load_agent(agent, device, 'trained_models/Pong_RDQN.pth')

# Training Function
def train(n_episodes):

    # Start Timing
    start_time = time.time()

    # Score Tracking
    scores = []
    agent_scores = []
    opponent_scores = []
    best_score = 0

    # Score Windows for Averages
    score_window = deque(maxlen=100)
    agent_score_window = deque(maxlen=10)
    opponent_score_window = deque(maxlen=10)

    total_frames = 0
    learn_steps = 0

    for i_episode in range(1, n_episodes + 1):
        
        observation_np, _ = env.reset()
        observation_normalized = observation_np.astype(np.float32) / 255.0
        observation_tensor = torch.from_numpy(observation_normalized).to(device)
        
        score = 0
        done = False
        truncated = False
        
        agent.q_net.reset_noise() 

        while not done and not truncated:
            if device.type == 'cuda':
                with torch.amp.autocast(device_type="cuda"):
                    action = agent.choose_action(observation_tensor)
            else:
                action = agent.choose_action(observation_tensor)
            
            next_observation_np, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            next_observation_normalized = next_observation_np.astype(np.float32) / 255.0
            next_observation_tensor = torch.from_numpy(next_observation_normalized).to(device)

            agent.step(observation_normalized, action, reward, next_observation_normalized, done) 

            observation_np = next_observation_np
            observation_normalized = next_observation_normalized
            observation_tensor = next_observation_tensor
            total_frames += 1

            score += reward

            loss_value = None
            if device.type == 'cuda':
                with torch.amp.autocast(device_type="cuda"):
                    loss_value = agent.learn()
            else:
                loss_value = agent.learn()
             
            if loss_value is not None:
                learn_steps += 1
             
            if total_frames >= MAX_FRAMES:
                break

        agent_score, opponent_score = env.get_scores()

        scores.append(score)
        agent_scores.append(agent_score)
        opponent_scores.append(opponent_score)

        score_window.append(score)
        agent_score_window.append(agent_score)
        opponent_score_window.append(opponent_score)

        print(f"Episode {i_episode:5d} | Agent - {agent_score} x {opponent_score} - Opponent")

        if agent_score > best_score:
            best_score = agent_score
            print(f"New Highscore | Agent - {agent_score} x {opponent_score} - Opponent on episode {i_episode}")
            #save_agent(agent, f'trained_models/Pong_RainbowDQN_best_game_score.pth')

        if i_episode % 10 == 0:
            avg_raw_reward_100 = np.mean(list(score_window)[-100:]) if len(score_window) >= 100 else np.mean(score_window)
            print(f"Episode {i_episode:5d} | Avg Agent Score: {np.mean(agent_score_window):.2f} x Avg Opponent Score: {np.mean(opponent_score_window):.2f}")
            
        if total_frames >= MAX_FRAMES:
            print(f"\nReached max frames ({MAX_FRAMES}). Stopping training.")
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    save_agent(agent, 'trained_models/Pong_RDQN.pth')

    return scores

train(n_episodes=1000)

env.close()