import gymnasium as gym
import ale_py
import torch
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt

import sys
sys.path.append('./')

# Import the new A2C Agent
from agents.a2c_agent import A2CAgent, ComplexA2CAgent 

# Import the new A2C CNN Models
from models.actor_critic_cnn import A2CActorCnn, A2CCriticCnn 

# Import Utils
from utils.stack_frame import preprocess_frame, stack_frame
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
INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
GAMMA = 0.99
LR_ACTOR = 0.00025       
LR_CRITIC = 0.0005     
VALUE_LOSS_COEF = 0.5    
ENTROPY_COEF = 0.01      
UPDATE_EVERY = 4

# Initialize Agents
agent = ComplexA2CAgent(INPUT_SHAPE, ACTION_SIZE, 0, device, GAMMA, LR_ACTOR, LR_CRITIC, UPDATE_EVERY, A2CActorCnn, A2CCriticCnn, VALUE_LOSS_COEF, ENTROPY_COEF)
#load_agent(agent, device, 'trained_models/Pong_A2C.pth')

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
    score_window = deque(maxlen=10)
    agent_score_window = deque(maxlen=10)
    opponent_score_window = deque(maxlen=10)

    for i_episode in range(1, n_episodes + 1):

        state = env.reset()[0].astype(np.float32) / 255.0
        
        score = 0
        
        while True:

            action, log_prob, entropy = agent.act(state) 
            
            next_state_raw, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state = next_state_raw.astype(np.float32) / 255.0

            agent.step(state, log_prob, entropy, reward, done, next_state)

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
        if agent_score > best_score:
            best_score = agent_score
            print(f"New Highscore | Agent - {agent_score} x {opponent_score} - Opponent on episode {i_episode}")

        # Print Progress
        if i_episode % 10 == 0:
            print(f"Episode {i_episode:5d} | Avg Agent Score: {np.mean(agent_score_window):.2f} x Avg Opponent Score: {np.mean(opponent_score_window):.2f}")

            """
            # Plot Progress
            if i_episode % n_episodes == 0:
                plot_scores(episode_rewards[-1000:], i_episode)
            """

    # End Timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    return scores

# Run Training
train(n_episodes=1000)
#save_agent(agent, 'trained_models/Pong_A2C.pth')     #Target Score: Agent 21 x 0 Opponent

# Close Environment
env.close()