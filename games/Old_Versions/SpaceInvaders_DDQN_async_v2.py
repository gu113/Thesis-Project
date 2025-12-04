import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import ale_py
import torch
import numpy as np
from collections import deque
import time
import sys
import gc

sys.path.append('./')

# Import Agents
from agents.ddqn_agent_Async_v2 import DDQNAgent

# Import Models
from models.ddqn_cnn_async import DDQNCnn

# Import Utils
from utils.stack_frame import preprocess_frame, stack_frame

# Import Custom Reward Modifier Wrapper
#from rewards.SpaceInvaders.SpaceInvaders_rewards import RewardModifierWrapper

torch.cuda.empty_cache()

# Set up Device
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def stack_frames(frames, state, is_new=False):
    if isinstance(state, np.ndarray) and state.ndim == 4:
        # Batch of states
        batch_size = state.shape[0]
        stacked_batch = []

        for i in range(batch_size):
            frame = preprocess_frame(state[i], (8, -12, -12, 4), 84)
            if is_new or frames is None:
                f = deque([frame] * 4, maxlen=4)
            else:
                f = frames[i]
                f.append(frame)
            stacked = np.stack(f, axis=0)  # Shape: (4, 84, 84)
            stacked_batch.append(stacked)
        
        return stacked_batch  # return list of stacked frames (shape [B, 4, 84, 84])

    else:
        # Single state
        frame = preprocess_frame(state, (8, -12, -12, 4), 84)
        if is_new or frames is None:
            frames = deque([frame] * 4, maxlen=4)
        else:
            frames.append(frame)
        stacked = np.stack(frames, axis=0)
        return frames, stacked

# Stack Frames for Parallel Environments
def stack_frames_batch(prev_frames, states, is_new_batch):
    stacked = []
    new_frame_buffers = []

    for i, state in enumerate(states):
        frames, stacked_state = stack_frames(
            prev_frames[i] if prev_frames else None,
            state,
            is_new_batch[i]
        )
        stacked.append(stacked_state)
        new_frame_buffers.append(frames)

    return new_frame_buffers, np.array(stacked, dtype=np.float32)  # Shape: [B, 4, 84, 84]


# Hyperparameters
INPUT_SHAPE = (4, 84, 84)
GAMMA = 0.99
BUFFER_SIZE = 50000
BATCH_SIZE = 4
LR = 0.005
TAU = 0.005 # 0.001 #0.1
UPDATE_EVERY = 50
EPS_START = 0.99
EPS_END = 0.01
EPS_DECAY = 1000

# Simplified epsilon function
def epsilon_by_episode(episode):
    return max(EPS_END, EPS_START - (episode / EPS_DECAY))

# Training function (with AMP for FP16)
def train(n_episodes):
    start_time = time.time()
    scores = np.zeros(NUM_ENVS)
    episode_rewards = [[] for _ in range(NUM_ENVS)]
    episode_count = 0
    max_episodes = n_episodes
    all_scores = []

    # Initialize frame buffers and initial state
    raw_state = envs.reset()[0]  # Shape: (NUM_ENVS, H, W, C)
    frame_buffers, state = stack_frames_batch(None, raw_state, [True] * NUM_ENVS)

    while episode_count < max_episodes:
        eps = epsilon_by_episode(episode_count)

        with torch.amp.autocast("cuda"):
            actions = agent.act(state, eps)

        next_states_raw, rewards, terminateds, truncateds, infos = envs.step(actions)
        dones = np.logical_or(terminateds, truncateds)

        # Stack next states using frame buffers
        frame_buffers, next_states = stack_frames_batch(frame_buffers, next_states_raw, dones)

        # FP16 conversions
        rewards = rewards.astype(np.float16)
        state_fp16 = torch.from_numpy(state).to(device=device, dtype=torch.float16)
        next_states_fp16 = torch.from_numpy(next_states).to(device=device, dtype=torch.float16)
        rewards = torch.from_numpy(rewards).to(device=device, dtype=torch.float16)
        dones = torch.from_numpy(dones).to(device=device, dtype=torch.bool)

        agent.step(state_fp16, actions, rewards, next_states_fp16, dones)

        for i in range(NUM_ENVS):
            episode_rewards[i].append(rewards[i])
            if dones[i]:
                episode_count += 1
                total_reward = sum(episode_rewards[i])
                all_scores.append(total_reward)
                print(f"Episode {episode_count} - Score: {total_reward:.2f} - Epsilon: {eps:.2f}")
                episode_rewards[i] = []

                # Print average score every 10 episodes
                if episode_count % 10 == 0:
                    avg_score = np.mean(torch.tensor(all_scores[-10:]).cpu().numpy())
                    print(f"Episode {episode_count} - Score Average (last 10 episodes): {avg_score:.2f}")
                    gc.collect()
                    torch.cuda.empty_cache()


        state = next_states  # Move to next state

    envs.close()
    print(f"Total Training Time: {(time.time() - start_time)/60:.2f} minutes")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Optional but helpful on Windows

    # Initialize Environment Vector
    NUM_ENVS = 3
    def make_env():
        def thunk():
            env = gym.make('ALE/SpaceInvaders-v5', frameskip=4)
            #env = RewardModifierWrapper(env)
            return env
        return thunk

    env_fns = [make_env() for _ in range(NUM_ENVS)]
    envs = AsyncVectorEnv(env_fns)

    ACTION_SIZE = envs.single_action_space.n

    # Initialize Agents
    agent = DDQNAgent(INPUT_SHAPE, ACTION_SIZE, 0, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, 5000, DDQNCnn)

    # Run Training
    train(n_episodes=1000)

    envs.close()