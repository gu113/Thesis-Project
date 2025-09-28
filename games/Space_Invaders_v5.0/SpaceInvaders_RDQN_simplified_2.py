import gymnasium as gym
import ale_py
import torch
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt
import random

import sys
sys.path.append('./')

# Import Rainbow Agent
from agents.rainbow3 import RainbowDQNAgent

# Import Utils
from utils.stack_frame import preprocess_frame, stack_frame
from utils.save_load import save_agent, load_agent

# Import Custom Wrappers
from wrappers.SpaceInvaders.rewards import ComplexRewardModifierWrapper
from wrappers.SpaceInvaders.noop_reset import NoopResetEnv

# Initialize Environment
env = gym.make('ALE/SpaceInvaders-v5', frameskip=4)
#env = ComplexRewardModifierWrapper(env)
env = NoopResetEnv(env, noop_max=30)

# Set up Device
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def stack_frames(frames, state, is_new=False):
    """Stack frames for the agent input"""
    frame = preprocess_frame(state, (8, -12, -12, 4), 84)
    
    if torch.cuda.is_available():
        frame = torch.from_numpy(frame).to(torch.float32).to(device) / 255.0
    else:
        frame = frame.astype(np.float32) / 255.0
    
    frames = stack_frame(frames, frame, is_new)
    return frames

"""
def plot_scores(scores, episode_num):
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title(f'Rainbow DQN Training Progress - Episodes {episode_num-len(scores)+1} to {episode_num}')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    plt.savefig(f'rainbow_training_progress_{episode_num}.png')
    plt.close()
"""
# Hyperparameters
INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
GAMMA = 0.99
BUFFER_SIZE = 50000
BATCH_SIZE = 16
LR = 0.0001
TAU = 0.005
UPDATE_EVERY = 4
NUM_ATOMS = 51
V_MIN = -10
V_MAX = 10
N_STEP = 3

# Initialize Rainbow Agent
agent = RainbowDQNAgent(
    state_size=INPUT_SHAPE,
    action_size=ACTION_SIZE,
    device=device,
    lr=LR,
    gamma=GAMMA,
    tau=TAU,
    update_every=UPDATE_EVERY,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    num_atoms=NUM_ATOMS,
    v_min=V_MIN,
    v_max=V_MAX,
    n_step=N_STEP
)

# Load pretrained model if available
# load_agent(agent, device, 'trained_models/rainbow_spaceinvaders.pth')

def train(n_episodes):
    """Training function for Rainbow DQN"""
    start_time = time.time()
    scores = []
    scores_window = deque(maxlen=100)
    #original_score_window = deque(maxlen=100)
    best_score = 0
    #original_best_score = 0
    episode_rewards = []

    for i_episode in range(1, n_episodes + 1):
        
        # Reset environment and initialize state
        if torch.cuda.is_available():
            state = stack_frames(None, env.reset()[0], True).to(torch.float32)
        else:
            state = stack_frames(None, env.reset()[0], True)

        score = 0
        #original_score = 0
        
        while True:
            # Rainbow DQN doesn't need epsilon - noisy layers handle exploration
            action = agent.act(state)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if torch.cuda.is_available():
                next_state = stack_frames(state, next_state, False).to(torch.float32)
            else:
                next_state = stack_frames(state, next_state, False)

            # Convert tensors to numpy for storage in replay buffer
            if isinstance(state, torch.Tensor):
                state_np = state.squeeze(0).cpu().numpy() if state.dim() == 4 else state.cpu().numpy()
            else:
                state_np = state
                
            if isinstance(next_state, torch.Tensor):
                next_state_np = next_state.squeeze(0).cpu().numpy() if next_state.dim() == 4 else next_state.cpu().numpy()
            else:
                next_state_np = next_state

            agent.step(state_np, action, reward, next_state_np, done)

            state = next_state
            score += reward
            #original_score = info['original_score']

            if done:
                break

        scores_window.append(score)
        #original_score_window.append(original_score)
        scores.append(score)
        episode_rewards.append(score)

        # Track High Scores
        if score > best_score:
            best_score = score
            #original_best_score = original_score
            print(f"New Highscore: {best_score:.2f} on episode {i_episode}")

        # Print Progress
        if i_episode % 10 == 0:
            avg_score = np.mean(scores_window)
            #avg_original = np.mean(original_score_window)
            print(f"Episode {i_episode:5d} | Avg Score: {avg_score:.2f}")

        # Plot Progress
        #if i_episode % 100 == 0:
            #plot_scores(episode_rewards[-1000:], i_episode)

        # Save model periodically
        #if i_episode % 500 == 0:
            #save_agent(agent, f'trained_models/rainbow_spaceinvaders_{i_episode}.pth')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Highest Score Achieved: {best_score:.2f}")
    #print(f"Highest Original Score Achieved: {original_best_score}")

    return scores

def test_agent(n_episodes=10):
    """Test the trained agent"""
    print("Testing Rainbow DQN Agent...")
    
    scores = []
    for i_episode in range(1, n_episodes + 1):
        if torch.cuda.is_available():
            state = stack_frames(None, env.reset()[0], True).to(torch.float32)
        else:
            state = stack_frames(None, env.reset()[0], True)
            
        score = 0
        
        while True:
            action = agent.act(state)  # No exploration during testing
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if torch.cuda.is_available():
                next_state = stack_frames(state, next_state, False).to(torch.float32)
            else:
                next_state = stack_frames(state, next_state, False)
                
            state = next_state
            score += reward
            
            if done:
                break
                
        scores.append(score)
        print(f"Test Episode {i_episode}: Score = {score:.2f}, Original Score = {info['original_score']}")
    
    print(f"Average Test Score: {np.mean(scores):.2f}")
    return scores

if __name__ == "__main__":
    print("Starting Rainbow DQN Training for Space Invaders...")
    
    # Run Training
    scores = train(n_episodes=1000)
    
    # Save final model
    #save_agent(agent, 'trained_models/rainbow_spaceinvaders_final.pth')
    
    # Test the trained agent
    test_scores = test_agent(n_episodes=10)
    
    # Plot final results
    #plot_scores(scores, len(scores))
    
    env.close()
    print("Training completed!")