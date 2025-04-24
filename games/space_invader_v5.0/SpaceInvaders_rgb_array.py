import time
import gymnasium as gym
from gymnasium import Wrapper
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
import math 

import sys
sys.path.append('./')

from agents import DDQNAgent
from models import DDQNCnn
from utils.stack_frame import preprocess_frame, stack_frame

# Custom Reward Modifier Wrapper
class RewardModifierWrapper(Wrapper):
    def __init__(self, env):
        super(RewardModifierWrapper, self).__init__(env)

    def step(self, action):
        # Take a step in the environment
        # observation, reward, done, info = self.env.step(action)
        state, reward, terminated, truncated, info = self.env.step(action)
        if reward > 0:
            print(reward)
        #print(info)
        
        # Modify the reward based on custom logic
        # Example: Penalize more for losing a life
        #lives_remaining = info.get('lives', 3)  # Default to 3 lives if info doesn't provide it
        #if lives_remaining < 3:
        #    reward -= 20  # Apply penalty if a life is lost

        return state, reward, terminated, truncated, info
    
env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')
#env.seed(0)

env = RewardModifierWrapper(env)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# View Environment
print("The size of frame is: ", env.observation_space.shape)
print("No. of Actions: ", env.action_space.n)
env.reset()
plt.figure()
plt.imshow(env.reset()[0])
plt.title('Original Frame')
plt.show()

def random_play():
    score = 0
    env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        #state, reward, done, _ = env.step(action)
        score += reward
        if terminated or truncated:
            env.close()
            print("Your Score at end of game is: ", score)
            break
random_play()


env.reset()
plt.figure()
plt.imshow(preprocess_frame(env.reset()[0], (8, -12, -12, 4), 84), cmap="gray")
plt.title('Pre Processed image')
plt.show()


def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (8, -12, -12, 4), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames


INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
SEED = 0
GAMMA = 0.99           # discount factor
BUFFER_SIZE = 100000   # replay buffer size
BATCH_SIZE = 32        # Update batch size
LR = 0.0001            # learning rate 
TAU = 0.001            # for soft update of target parameters
UPDATE_EVERY = 100     # how often to update the network
UPDATE_TARGET = 10000  # After which thershold replay to be started 
EPS_START = 0.99       # starting value of epsilon
EPS_END = 0.01         # Ending value of epsilon
EPS_DECAY = 100        # Rate by which epsilon to be decayed

agent = DDQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DDQNCnn)


# Watch an untrained agent
state = stack_frames(None, env.reset()[0], True)  # Ensure that the initial state is set correctly

for j in range(200):
    env.render()
    action = agent.act(state, .9)

    # Get the result from env.step()
    step_result = env.step(action)

    # Unpack the result from env.step()
    if len(step_result) == 5:  # If it returns five values
        next_state, reward, done, truncated, info = step_result
    elif len(step_result) == 4:  # If it returns four values
        next_state, reward, done, info = step_result
        truncated = False  # Set truncated to False if not provided
    elif len(step_result) == 3:  # If it returns three values
        next_state, reward, done = step_result
        info = {}  # Create an empty dictionary for info if not provided
        truncated = False  # Set truncated to False if not provided

    # Only stack frames if next_state is properly defined
    state = stack_frames(state, next_state, False)
    
    if done:
        break 

env.close()


start_epoch = 0
scores = []
scores_window = deque(maxlen=20)




epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)

plt.plot([epsilon_by_epsiode(i) for i in range(1000)])



def train(n_episodes=100, IterationsBetweenGraph=10):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
        IterationsBetweenGraph (int): how often to update the plot
    """
    for i_episode in range(start_epoch + 1, n_episodes + 1):
        state = stack_frames(None, env.reset()[0], True)
        score = 0
        eps = epsilon_by_epsiode(i_episode)
        
        while True:
            action = agent.act(state, eps)
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            
            # Custom reward shaping
            if reward > 0:
                reward += 5  # Extra reward for hitting enemies
            if terminated:  
                reward -= 20  # Penalty for losing the game
            else:
                reward += 0.1  # Small reward for staying alive
            
            next_state = stack_frames(state, observation, False)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            if done:
                break
        
        scores_window.append(score)  # Save most recent score
        scores.append(score)         # Save most recent score
        
        # Only update and display the graph every `IterationsBetweenGraph` episodes
        if i_episode % IterationsBetweenGraph == 0 or i_episode == n_episodes:
            clear_output(True)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(np.arange(len(scores)), scores)
            plt.ylabel('Score')
            plt.xlabel('Episode #')
            plt.show()
        
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.2f}', end="")

    return scores

# Run training for 100 episodes
scores = train(n_episodes=1000, IterationsBetweenGraph=1000)

# Final plot after training
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
