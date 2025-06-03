import os
import torch

def save_agent(agent, filename):
    checkpoint = {
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Agent saved to {filename}")

def load_agent(agent, map_location, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location)
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded agent from {filename}")
    else:
        print("No saved agent found, starting fresh.")
