import os
import torch
import torch.nn as nn

def save_agent(agent, filename):
    """Saves the entire agent's state (policy, target, optimizer) to a file"""
    checkpoint = {
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Agent saved to {filename}")


def load_agent(agent, map_location, filename):
    """Loads the entire agent's state from a file"""
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded agent from {filename}")
    else:
        print("No saved agent found, starting fresh.")


def save_agent_a2c(agent, filename):
    """Saves the state of an A2C agent (actor, critic, and their optimizers)"""
    state = {
        'actor_net_state_dict': agent.actor_net.state_dict(),
        'critic_net_state_dict': agent.critic_net.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
    }
    torch.save(state, filename)
    print(f"A2C agent saved to {filename}")


def load_agent_a2c(agent, map_location, filename):
    """Loads the state of an A2C agent from a file"""
    if os.path.isfile(filename):
        print(f"Loading A2C agent from {filename}...")
        checkpoint = torch.load(filename, map_location)
        
        if 'actor_net_state_dict' in checkpoint:
            agent.actor_net.load_state_dict(checkpoint['actor_net_state_dict'])
        
        if 'critic_net_state_dict' in checkpoint:
            agent.critic_net.load_state_dict(checkpoint['critic_net_state_dict'])

        if 'actor_optimizer_state_dict' in checkpoint:
            agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

        if 'critic_optimizer_state_dict' in checkpoint:
            agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
        print(f"A2C agent loaded successfully.")
    else:
        print("No saved A2C agent found, starting fresh.")


def save_agent_a3c(agent, filename):
    """Saves the state of an A3C agent (actor, critic, and their optimizers)"""
    state = {
        'actor_net_state_dict': agent.actor_net.state_dict(),
        'critic_net_state_dict': agent.critic_net.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
    }
    torch.save(state, filename)
    print(f"A3C agent saved to {filename}")

def load_agent_a3c(agent, map_location, filename):
    """Loads the state of an A3C agent from a file"""
    if os.path.isfile(filename):
        print(f"Loading A3C agent from {filename}...")
        checkpoint = torch.load(filename, map_location)
        if 'actor_net_state_dict' in checkpoint:
            agent.actor_net.load_state_dict(checkpoint['actor_net_state_dict'])
        if 'critic_net_state_dict' in checkpoint:
            agent.critic_net.load_state_dict(checkpoint['critic_net_state_dict'])
        if 'actor_optimizer_state_dict' in checkpoint:
            agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        if 'critic_optimizer_state_dict' in checkpoint:
            agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"A3C agent loaded successfully.")
    else:
        print("No saved A3C agent found, starting fresh.")

        
def save_agent_REINFORCE(agent, filename):
    """Saves the state of a REINFORCE agent (policy and optimizer)"""
    state = {
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }
    torch.save(state, filename)
    print(f"REINFORCE agent saved to {filename}")


def load_agent_REINFORCE(agent, map_location, filename):
    """Loads the state of a REINFORCE agent from a file"""
    if os.path.isfile(filename):
        print(f"Loading REINFORCE agent from {filename}...")
        checkpoint = torch.load(filename, map_location)
        
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        print(f"REINFORCE agent loaded successfully.")
    else:
        print("No saved REINFORCE agent found, starting fresh.")


def save_agent_ppo(agent, filename):
    """Saves the state of a PPO agent (actor, critic, and their optimizers)"""
    state = {
        'actor_net_state_dict': agent.actor_net.state_dict(),
        'critic_net_state_dict': agent.critic_net.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
    }
    torch.save(state, filename)
    print(f"PPO agent saved to {filename}")


def load_agent_ppo(agent, map_location, filename):
    """Loads the state of a PPO agent from a file"""
    if os.path.isfile(filename):
        print(f"Loading PPO agent from {filename}...")
        checkpoint = torch.load(filename, map_location)
        
        agent.actor_net.load_state_dict(checkpoint['actor_net_state_dict'])
        agent.critic_net.load_state_dict(checkpoint['critic_net_state_dict'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
        print(f"PPO agent loaded successfully.")
    else:
        print("No saved PPO agent found, starting fresh.")


def save_agent_trpo(agent, filename):
    """Saves the state of a TRPO agent (actor and critic)"""
    checkpoint = {
        'actor_net_state_dict': agent.actor_net.state_dict(),
        'critic_net_state_dict': agent.critic_net.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"TRPO agent saved to {filename}")


def load_agent_trpo(agent, map_location, filename):
    """Loads the state of a TRPO agent from a file"""
    if os.path.isfile(filename):
        print(f"Loading TRPO agent from {filename}...")
        checkpoint = torch.load(filename, map_location)
        
        agent.actor_net.load_state_dict(checkpoint['actor_net_state_dict'])
        agent.critic_net.load_state_dict(checkpoint['critic_net_state_dict'])
        print(f"TRPO agent loaded from {filename}")
    else:
        print("No saved TRPO agent found, starting fresh.")


def save_agent_sac(agent, filename):
    """Saves the SAC agent's networks and optimizers to a file"""
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    state = {
        'policy_state_dict': agent.policy_net.state_dict(),
        'q1_state_dict': agent.q1_net.state_dict(),
        'q2_state_dict': agent.q2_net.state_dict(),
        'target_q1_state_dict': agent.target_q1_net.state_dict(),
        'target_q2_state_dict': agent.target_q2_net.state_dict(),
        'policy_optimizer_state_dict': agent.policy_optimizer.state_dict(),
        'q1_optimizer_state_dict': agent.q1_optimizer.state_dict(),
        'q2_optimizer_state_dict': agent.q2_optimizer.state_dict(),
        'log_alpha': agent.log_alpha,
        'alpha_optimizer_state_dict': agent.alpha_optimizer.state_dict(),
    }
    torch.save(state, filename)
    print(f"Agent state saved to {filename}")

def load_agent_sac(agent, filename, map_location):
    """Loads a SAC agent's state from a file"""
    if os.path.isfile(filename):
        print(f"Loading SAC agent from {filename}...")
        checkpoint = torch.load(filename, map_location)
        agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        agent.q1_net.load_state_dict(checkpoint['q1_state_dict'])
        agent.q2_net.load_state_dict(checkpoint['q2_state_dict'])
        agent.target_q1_net.load_state_dict(checkpoint['target_q1_state_dict'])
        agent.target_q2_net.load_state_dict(checkpoint['target_q2_state_dict'])
        agent.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        agent.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        agent.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        agent.log_alpha = checkpoint['log_alpha']
        agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        print(f"Agent state loaded from {filename}")
    else:
        print(f"No saved agent found at {filename}, starting fresh.")


def save_agent_rdqn(agent, filename):
    """Saves the Rainbow Agent's state to a file"""
    checkpoint = {
        'q_net_state_dict': agent.q_net.state_dict(),
        'target_q_net_state_dict': agent.target_q_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'steps_done': agent.steps_done,
        'scaler_state_dict': agent.scaler.state_dict() if agent.scaler else None,
    }
    torch.save(checkpoint, filename)
    print(f"Agent state saved to {filename}")

def load_agent_rdqn(agent, map_location, filename):
    """Loads the Rainbow Agent's state from a file"""
    if os.path.isfile(filename):
        print(f"Loading agent from {filename}...")
        checkpoint = torch.load(filename, map_location=map_location, weights_only=False)

        agent.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        agent.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.steps_done = checkpoint['steps_done']
        if agent.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            agent.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        agent.target_q_net.eval()
        
        print(f"Successfully loaded agent from {filename}")
    else:
        print("No saved agent found, starting fresh.")


def save_features(agent, filename):
    """Saves the state_dict of the feature extraction (convolutional) layers of a given model to a file"""
    if not hasattr(agent.policy_net, 'features'):
        raise AttributeError("Model must have a 'features' attribute")
    
    torch.save(agent.policy_net.features.state_dict(), filename)
    print(f"Features saved to {filename}")


def load_features(agent, map_location, filename):
    """Loads the state_dict of feature extraction (convolutional) layers from a file into the 'features' attribute of the given model"""
    if not hasattr(agent.policy_net, 'features'):
        raise AttributeError("Model must have a 'features' attribute")

    if os.path.isfile(filename):
        feature_state_dict = torch.load(filename, map_location)

        try:
            agent.policy_net.features.load_state_dict(feature_state_dict, strict=False)
            print(f"Loaded features into model from {filename}")
        except RuntimeError as e:
            print(f"Error loading features: {e}. Ensure feature extractor architectures match exactly.")
    else:
        print(f"No features file found at {filename}, starting feature extractor fresh.")
