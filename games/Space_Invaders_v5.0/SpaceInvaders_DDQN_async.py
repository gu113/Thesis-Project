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
from utils.save_load import save_agent, load_agent

# Import Custom Reward Modifier Wrapper
from wrappers.SpaceInvaders.rewards import ComplexRewardModifierWrapper
from wrappers.SpaceInvaders.noop_reset import NoopResetEnv

NUM_ENVS = 4

# Initialize Environment Factory
def make_env():
    def thunk():
        env = gym.make('ALE/SpaceInvaders-v5', frameskip=4)
        env = ComplexRewardModifierWrapper(env)
        env = NoopResetEnv(env, noop_max=30)
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
GAMMA = 0.99
BUFFER_SIZE = 100000
BATCH_SIZE = 64
LR = 0.0001
TAU = 0.001
UPDATE_EVERY = 4
EPS_START = 0.99
EPS_END = 0.01
EPS_DECAY = 1000

# Linear Epsilon Decay Function
def epsilon_by_episode_linear(episode):
    return max(EPS_END, EPS_START - (episode / EPS_DECAY))

# Exponential Epsilon Decay Function
def epsilon_by_episode_exponential(episode):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-episode / EPS_DECAY)

# Training function (with AMP for FP16)
def train(n_episodes, envs, agent):
    start_time = time.time()
    scores = []
    scores_window = deque(maxlen=100)
    original_score_window = deque(maxlen=10)
    best_score = 0
    original_best_score = 0

    stacked_frames = [None for _ in range(NUM_ENVS)]
    obs, _ = envs.reset()
    obs = stack_frames_batch(stacked_frames, obs, [True] * NUM_ENVS).astype(np.float16)
    episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)

    for i_episode in range(1, n_episodes + 1):
        score = 0
        original_score = 0
        eps = epsilon_by_episode_exponential(i_episode)

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

        if i_episode % 10 == 0:
            print(f"Episode {i_episode:5d} | Avg Score: {np.mean(scores_window):.2f} | Avg Real Score: {np.mean(original_score_window):.2f} | Epsilon: {eps:.2f}")
            
            """
            # Plot Progress
            if i_episode % n_episodes == 0:
                plot_scores(episode_rewards[-1000:], i_episode)
            """

    end_time = time.time() # End timing
    elapsed_time = end_time - start_time
    print(f"\nTotal Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Highest Score Achieved: {best_score}")
    print(f"Highest Original Score Achieved: {original_best_score}")

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
