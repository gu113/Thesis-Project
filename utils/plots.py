import numpy as np
import matplotlib.pyplot as plt

def plot_scores_simple(scores, episode_num):
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title(f'Training Progress - Episodes {episode_num-len(scores)+1} to {episode_num}')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    plt.savefig(f'training_progress_{episode_num}.png')
    plt.close()


def plot_scores(scores, episode_num):
    window = 100
    moving_average = np.convolve(scores, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(12, 8))
    
    plt.plot(scores, color='lightblue', alpha=0.6, label='Scores')
    
    plt.plot(np.arange(len(moving_average)) + window, moving_average, color='darkblue', label=f'Moving Average ({window} episodes)')

    plt.title(f'Training Progress - Episodes {episode_num-len(scores)+1} to {episode_num}', fontsize=16)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=12)
    
    plt.savefig(f'training_progress_{episode_num}.png')
    plt.show()
    plt.close()


def plot_scores_pong(scores, enemy_scores, episode_num):
    window = 100
    moving_average = np.convolve(scores, np.ones(window)/window, mode='valid')
    enemy_moving_average = np.convolve(enemy_scores, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(12, 8))
    
    plt.plot(scores, color='lightblue', alpha=0.6, label='Scores')
    
    plt.plot(np.arange(len(moving_average)) + window, moving_average, color='darkblue', label=f'Moving Average ({window} episodes)')
    plt.plot(np.arange(len(enemy_moving_average)) + window, enemy_moving_average, color='red', label=f'Moving Average ({window} episodes)')

    plt.title(f'Training Progress - Episodes {episode_num-len(scores)+1} to {episode_num}', fontsize=16)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=12)
    
    plt.savefig(f'training_progress_{episode_num}.png')
    plt.show()
    plt.close()