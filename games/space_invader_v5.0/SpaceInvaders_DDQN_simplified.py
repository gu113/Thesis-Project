import gymnasium as gym
import ale_py
import torch
import numpy as np
from collections import deque
import time

import sys
sys.path.append('./')

# Import Agents
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.reinforce_agent import ReinforceAgent
from agents.ddqn_agent import DDQNAgent
from agents.ddqn_agent_PER import DDQNAgentPER

# Import Models
from models import DDQNCnn
from models import DQNCnn

# Import Utils
from utils.stack_frame import preprocess_frame, stack_frame

# Custom Reward Modifier Wrapper
class RewardModifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        # Simple reward shaping
        if reward > 0:
            reward += 5  # Encourage hitting enemies
        if terminated:
            reward -= 10  # Penalize game over
        
        return state, reward, terminated, truncated, info

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
BUFFER_SIZE = 50000
BATCH_SIZE = 128
LR = 0.005
TAU = 0.005 # 0.001 #0.1
UPDATE_EVERY = 50
EPS_START = 0.99
EPS_END = 0.01
EPS_DECAY = 1000
alpha = 0.6 # PER prioritization parameter

# Initialize Agents
agent = DDQNAgent(INPUT_SHAPE, ACTION_SIZE, 0, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, 5000, DDQNCnn)

# Initialize PER Agents
#agent = DDQNAgentPER(INPUT_SHAPE, ACTION_SIZE, 0, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, 5000, DDQNCnn, alpha)


# Simplified epsilon function
def epsilon_by_episode(episode):
    return max(EPS_END, EPS_START - (episode / EPS_DECAY))

# Training function (with AMP for FP16)
def train(n_episodes):

    start_time = time.time()  # Start timing
    scores = []
    scores_window = deque(maxlen=100)
    #scaler = torch.amp.GradScaler("cuda")  # Helps with gradient stability

    for i_episode in range(1, n_episodes + 1):
        #state = stack_frames(None, env.reset()[0], True) # Original
        state = stack_frames(None, env.reset()[0], True).astype(np.float16) # FP16 - HALF PRECISION
        score = 0
        eps = epsilon_by_episode(i_episode)
        
        while True:

            #action = agent.act(state, eps)
            
            # Use FP16 for model training
            with torch.amp.autocast("cuda"):  # Enable FP16
                action = agent.act(state, eps)
            
            # No FP16 needed for env interaction
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            #next_state = stack_frames(state, next_state, False)
            next_state = stack_frames(state, next_state, False).astype(np.float16) # FP16 - HALF PRECISION

            reward = np.float16(reward)  # FP16 - HALF PRECISION
            agent.step(state, action, reward, next_state, done)  

            state = next_state
            score += reward

            if done:
                break

        scores_window.append(score)
        scores.append(score)

        # Print progress instead of plotting
        if i_episode % 10 == 0:
            print(f"Episode {i_episode} - Avg Score: {np.mean(scores_window):.2f} - Epsilon: {eps:.2f}")

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    print(f"\nTotal Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    return scores

# Run Training
train(n_episodes=1000)

env.close()
