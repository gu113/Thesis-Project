import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.vector import SyncVectorEnv
import ale_py
import torch
import numpy as np
from collections import deque
import time

import sys
sys.path.append('./')

# Import Agents
from agents.ddqn_agent_ram_Async import DDQNAgent

# Import Models
from models.ddqn_ram_async import DDQNMLP

# Import Custom Reward Modifier Wrapper
from rewards.SpaceInvaders.SpaceInvaders_rewards import RewardModifierWrapper, ComplexRewardModifierWrapper

# Impot Utils
from utils.save_load import save_agent, load_agent

NUM_ENVS = 2

# Create Env Factory
def make_env(i):
    def thunk():
        env = gym.make('ALE/SpaceInvaders-ram-v5', frameskip=4)
        # env = RewardModifierWrapper(env)
        return env
    return thunk

# Hyperparameters
INPUT_SHAPE = (128,)  # RAM observation shape
GAMMA = 0.99
BUFFER_SIZE = 100000
BATCH_SIZE = 32
LR = 0.0001
TAU = 0.005
UPDATE_EVERY = 4
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 1000

# Epsilon decay function
def epsilon_by_episode(episode):
    return max(EPS_END, EPS_START - (episode / EPS_DECAY))

# Async RAM Training
def train(n_episodes, envs, agent):
    start_time = time.time()
    scores = []
    scores_window = deque(maxlen=100)

    obs, _ = envs.reset()
    obs = obs.astype(np.float32)
    episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)

    for episode in range(1, n_episodes + 1):
        eps = epsilon_by_episode(episode)

        while True:
            actions = agent.act(obs, eps)
            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy()

            next_obs, rewards, terminated, truncated, _ = envs.step(actions)
            dones = np.logical_or(terminated, truncated)
            next_obs = np.array(next_obs, dtype=np.float32)

            for i in range(NUM_ENVS):
                agent.step(obs[i], actions[i], rewards[i], next_obs[i], dones[i])
                #agent.soft_update(agent.policy_net, agent.target_net, agent.tau)
                episode_rewards[i] += rewards[i]

            obs = next_obs

            for i in range(NUM_ENVS):
                if dones[i]:
                    scores.append(episode_rewards[i])
                    scores_window.append(episode_rewards[i])
                    episode_rewards[i] = 0

            if all(dones):
                obs, _ = envs.reset()
                obs = np.array(obs, dtype=np.float32)
                break

        if episode % 1 == 0:
            print(f"Episode {episode} - Avg Score: {np.mean(scores_window):.2f} - Epsilon: {eps:.2f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal Training Time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

    return scores

# MAIN
if __name__ == '__main__':
    import multiprocessing
    #multiprocessing.set_start_method('spawn', force=True)
    #multiprocessing.freeze_support()

    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    envs = AsyncVectorEnv([make_env(i) for i in range(NUM_ENVS)], shared_memory=True)
    ACTION_SIZE = envs.single_action_space.n

    # Initialize Agent with RAM MLP
    agent = DDQNAgent(INPUT_SHAPE, ACTION_SIZE, 0, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, 5000, DDQNMLP)

    # Run Training
    train(n_episodes=1000, envs=envs, agent=agent)

    # Cleanup
    envs.close()