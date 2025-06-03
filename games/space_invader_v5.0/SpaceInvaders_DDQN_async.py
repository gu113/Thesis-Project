import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import ale_py
import torch
import numpy as np
from collections import deque
import time
import sys

sys.path.append('./')

# Import Agents
from agents.ddqn_agent import AsyncDDQNAgent

# Import Models
from models import DDQNCnn

# Import Utils
from utils.stack_frame import preprocess_frame, stack_frame

# Import Custom Reward Modifier Wrapper
from wrappers.SpaceInvaders.rewards import RewardModifierWrapper

NUM_ENVS = 4

# Initialize Environment Factory
def make_env():
    def thunk():
        env = gym.make('ALE/SpaceInvaders-v5', frameskip=4)
        env = RewardModifierWrapper(env)
        return env
    return thunk

def stack_frame(stacked_frames, new_frame, is_new_episode):
    if is_new_episode or stacked_frames is None:
        # Initialize with 4 copies of the new frame
        stacked_frames = [new_frame for _ in range(NUM_ENVS)]
    else:
        # Remove oldest frame and append new one
        stacked_frames.append(new_frame)
        stacked_frames.pop(0)

    return np.stack(stacked_frames, axis=0)

# Stack Frames for Parallel Environments
def stack_frames_batch(prev_frames, states, is_new_batch):
    stacked = []
    for i, state in enumerate(states):
        frame = preprocess_frame(state, (8, -12, -12, 4), 84)
        frames = stack_frame(prev_frames[i] if prev_frames else None, frame, is_new_batch[i])
        stacked.append(frames)
    return np.stack(stacked)

# Hyperparameters
INPUT_SHAPE = (4, 84, 84)
GAMMA = 0.99
BUFFER_SIZE = 50000
BATCH_SIZE = 128
LR = 0.005
TAU = 0.005
UPDATE_EVERY = 50
EPS_START = 0.99
EPS_END = 0.01
EPS_DECAY = 1000

# Simplified epsilon function
def epsilon_by_episode(episode):
    return max(EPS_END, EPS_START - (episode / EPS_DECAY))

# Training function (with AMP for FP16)
def train(n_episodes, envs, agent):
    start_time = time.time()
    scores = []
    scores_window = deque(maxlen=100)

    stacked_frames = [None for _ in range(NUM_ENVS)]
    obs, _ = envs.reset()
    obs = stack_frames_batch(stacked_frames, obs, [True] * NUM_ENVS).astype(np.float16)
    episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)

    for episode in range(1, n_episodes + 1):
        eps = epsilon_by_episode(episode)

        while True:
            with torch.amp.autocast("cuda"):
                actions = agent.act(obs, eps)

            next_obs, rewards, terminated, truncated, _ = envs.step(actions)
            dones = np.logical_or(terminated, truncated)
            next_obs = stack_frames_batch(stacked_frames, next_obs, dones).astype(np.float16)

            for i in range(NUM_ENVS):
                agent.step(obs[i], actions[i], rewards[i], next_obs[i], dones[i])
                episode_rewards[i] += rewards[i]

            obs = next_obs

            for i in range(NUM_ENVS):
                if dones[i]:
                    scores.append(episode_rewards[i])
                    scores_window.append(episode_rewards[i])
                    episode_rewards[i] = 0

            if all(dones):
                obs, _ = envs.reset()
                obs = stack_frames_batch(stacked_frames, obs, [True] * NUM_ENVS).astype(np.float16)
                break

        if episode % 10 == 0:
            print(f"Episode {episode} - Avg Score: {np.mean(scores_window):.2f} - Epsilon: {eps:.2f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    return scores

# MAIN ENTRY POINT
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    # Set up Device
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize environments
    envs = AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)], shared_memory=True)

    ACTION_SIZE = envs.single_action_space.n

    # Initialize Agent
    agent = AsyncDDQNAgent(INPUT_SHAPE, ACTION_SIZE, 0, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, 5000, DDQNCnn)

    # Run Training
    train(n_episodes=1000, envs=envs, agent=agent)

    envs.close()
