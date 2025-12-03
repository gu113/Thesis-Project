import numpy as np
import matplotlib.pyplot as plt

def plot_scores_simple(scores, episode_num):
    """Simple plotting function to visualize scores over episodes"""
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title(f'Training Progress - Episodes {episode_num-len(scores)+1} to {episode_num}')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    plt.savefig(f'training_progress_{episode_num}.png')
    plt.close()


def plot_scores(scores, episode_num):
    """Plotting function to visualize scores with moving average over episodes"""
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
    """Plotting function to visualize scores and enemy scores with moving averages over episodes"""
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


def plot_multiple_curves(all_scores, episode_num, hyperparameter):
    """Plotting function to compare multiple score curves with moving averages over episodes"""
    window = 100
    plt.figure(figsize=(12, 8))

    for label, scores in all_scores:
        scores_array = np.array(scores)

        moving_average = np.array([np.mean(scores_array[max(0, i-window+1):i+1]) if i >= 9 else np.nan for i in range(len(scores_array))])
        #moving_average = np.convolve(scores_array, np.ones(window)/window, mode='same')

        x_axis = np.arange(1, len(moving_average) + 1)
        plt.plot(x_axis, moving_average, label=f'{hyperparameter} = {label}', linewidth=2)

    plt.title(f'{hyperparameter} Comparison - Episodes {episode_num-len(scores)+1} to {episode_num}', fontsize=16)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=12)
    
    plt.savefig(f'{hyperparameter}_comparison_{episode_num}.png')
    plt.show()
    plt.close()