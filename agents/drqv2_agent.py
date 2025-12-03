import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import kornia.augmentation as T
from utils.replay_buffer import DrQv2Buffer
from models.drqv2_cnn import DrQv2CNN

class DrQAgent:
    def __init__(self, state_size, action_size, device='cuda', lr=1e-4, gamma=0.99, tau=0.01, batch_size=128, buffer_size=100000, target_update_freq=2, feature_dim=50, std_schedule=0.1, num_aug=2):
        """Initialize a Simple DrQ-v2 Agent
        Params
        ======
            state_size (tuple): Dimension of each state
            action_size (int): Dimension of each action
            device (str): Device to use ('cuda' or 'cpu')
            lr (float): Learning rate
            gamma (float): Discount factor
            tau (float): Soft update parameter
            batch_size (int): Mini-batch size
            buffer_size (int): Replay buffer size
            target_update_freq (int): Frequency of target network updates
            feature_dim (int): Dimension of feature representation
            std_schedule (float): Standard deviation for exploration noise
            num_aug (int): Number of augmentations for target Q-value estimation
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.std_schedule = std_schedule
        self.num_aug = num_aug
        
        # Q-Networks
        self.q_network = DrQv2CNN(state_size, action_size, feature_dim).to(device)
        self.q_target = DrQv2CNN(state_size, action_size, feature_dim).to(device)
        
        # Copy weights to target
        self.q_target.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = DrQv2Buffer(buffer_size, state_size, device)
        
        # Exploration noise
        self.std = std_schedule
        
        # Counters
        self.total_steps = 0
        self.updates = 0
        
    def act(self, state, eps=0.0):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        
        self.q_network.eval()

        # Select action with no exploration noise
        with torch.no_grad():
            q1, q2 = self.q_network(state)
            q_values = torch.min(q1, q2)  # Conservative Q-learning
            action = q_values.argmax(dim=1).item()
        self.q_network.train()
        
        # Epsilon-greedy exploration
        if random.random() < eps:
            return random.randrange(self.action_size)
        
        return action
    
    def step(self, state, action, reward, next_state, done):

        # Store transition and update if ready
        self.memory.add(state, action, reward, next_state, done)
        self.total_steps += 1
        
        # Update every step (when buffer is large enough)
        if len(self.memory) > self.batch_size:
            self.learn()
    
    def learn(self):

        # Sample a batch from memory
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return
            
        states, actions, rewards, next_states, dones = batch
        
        # Current Q-values
        current_q1, current_q2 = self.q_network(states, augment=True)
        current_q1 = current_q1.gather(1, actions.unsqueeze(1)).squeeze(1)
        current_q2 = current_q2.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values with augmentation
        with torch.no_grad():
            # Average over multiple augmentations
            target_q1_list = []
            target_q2_list = []
            
            for _ in range(self.num_aug):
                next_q1, next_q2 = self.q_network(next_states, augment=True)
                next_actions = torch.min(next_q1, next_q2).argmax(dim=1)
                
                target_q1_aug, target_q2_aug = self.q_target(next_states, augment=True)
                target_q1_list.append(target_q1_aug.gather(1, next_actions.unsqueeze(1)).squeeze(1))
                target_q2_list.append(target_q2_aug.gather(1, next_actions.unsqueeze(1)).squeeze(1))
            
            # Average target Q-values
            target_q1 = torch.stack(target_q1_list).mean(dim=0)
            target_q2 = torch.stack(target_q2_list).mean(dim=0)
            target_q = torch.min(target_q1, target_q2)
            
            target_q = rewards + (self.gamma * target_q * (~dones))
        
        # Compute losses
        loss_q1 = F.mse_loss(current_q1, target_q)
        loss_q2 = F.mse_loss(current_q2, target_q)
        loss = loss_q1 + loss_q2
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        # Update target network
        self.updates += 1
        if self.updates % self.target_update_freq == 0:
            self.soft_update()
    
    def soft_update(self):
        # Soft update model parameters
        for target_param, param in zip(self.q_target.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


class DrQv2Agent:
    def __init__(self, state_size, action_size, device='cuda', lr=1e-4, gamma=0.99, tau=0.01, batch_size=128, buffer_size=500000, feature_dim=50, update_every_steps=2, num_seed_steps=5000, stddev_schedule='linear(1.0,0.1,100000)'):
        """Initialize a DrQ-v2 Agent
        Params
        ======
            state_size (tuple): Dimension of each state
            action_size (int): Dimension of each action
            device (str): Device to use ('cuda' or 'cpu')
            lr (float): Learning rate
            gamma (float): Discount factor
            tau (float): Soft update parameter
            batch_size (int): Mini-batch size
            buffer_size (int): Replay buffer size
            feature_dim (int): Dimension of feature representation
            update_every_steps (int): Frequency of network updates
            num_seed_steps (int): Number of initial random steps
            stddev_schedule (str): Schedule for exploration noise standard deviation
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every_steps = update_every_steps
        self.num_seed_steps = num_seed_steps
        
        self.stddev_schedule = self.parse_schedule(stddev_schedule)
        
        # Q-Networks
        self.q_network = DrQv2CNN(state_size, action_size, feature_dim).to(device)
        self.q_target = DrQv2CNN(state_size, action_size, feature_dim).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, betas=(0.9, 0.999))
        
        # Replay buffer
        self.memory = DrQv2Buffer(buffer_size, state_size, device)
        
        self.step_count = 0
        self.episode_count = 0
        
    def parse_schedule(self, schedule_str):
        # Simple parser for linear schedule strings
        if schedule_str.startswith('linear'):
            params = schedule_str[7:-1].split(',')
            start_val = float(params[0])
            end_val = float(params[1])
            num_steps = int(params[2])
            return lambda step: max(end_val, start_val - (start_val - end_val) * step / num_steps)
        else:
            return lambda step: float(schedule_str)
    
    def act(self, state_tensor, training=True):

        # Ensure state_tensor is float32 for model input
        if state_tensor.dtype != torch.float32:
            state_tensor = (state_tensor.float() / 255.0) # Normalize to 0-1 and convert to float32
        
        if len(state_tensor.shape) == 3:
            state_tensor = state_tensor.unsqueeze(0) # Add batch dimension if missing
        
        if training and self.step_count < self.num_seed_steps:
            return random.randrange(self.action_size)
        
        # Select action
        self.q_network.eval()
        with torch.no_grad():
            q1, q2 = self.q_network(state_tensor.to(self.device))
            q_values = torch.min(q1, q2)
            
            if training:
                stddev = self.stddev_schedule(self.step_count)
                noise = torch.randn_like(q_values) * stddev
                q_values = q_values + noise
            
            action = q_values.argmax(dim=1).item()
        self.q_network.train()
        
        return action
    
    def step(self, state, action, reward, next_state, done):
        self.step_count += 1

        # Normalize states to float32 [0,1] for storage
        state_np_normalized = (state.cpu().numpy().astype(np.float32) / 255.0)
        next_state_np_normalized = (next_state.cpu().numpy().astype(np.float32) / 255.0)

        # Store transition
        self.memory.add(state_np_normalized, action, reward, next_state_np_normalized, done)
        
        if done:
            self.episode_count += 1
        
        if (self.step_count >= self.num_seed_steps and 
            self.step_count % self.update_every_steps == 0 and
            len(self.memory) > self.batch_size):
            self.learn()
    
    def learn(self):
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return
            
        states, actions, rewards, next_states, dones = batch
        
        # Compute current Q-values
        with torch.amp.autocast("cuda"): 
            current_q1, current_q2 = self.q_network(states, augment=True)
            current_q1 = current_q1.gather(1, actions.unsqueeze(1)).squeeze(1)
            current_q2 = current_q2.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute target Q-values
            with torch.no_grad():
                next_q1, next_q2 = self.q_network(next_states, augment=True)
                next_q = torch.min(next_q1, next_q2)
                next_actions = next_q.argmax(dim=1) # Double Q-learning: uses online network for action selection
                
                target_q1, target_q2 = self.q_target(next_states, augment=True)
                target_q1_selected = target_q1.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q2_selected = target_q2.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q = torch.min(target_q1_selected, target_q2_selected) # Min over two target Qs
                
                target_q = rewards + (self.gamma * target_q * (~dones))
            
            # Compute losses
            loss_q1 = F.mse_loss(current_q1, target_q)
            loss_q2 = F.mse_loss(current_q2, target_q)
            total_loss = loss_q1 + loss_q2
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        self.soft_update()
    
    def soft_update(self):
        for target_param, param in zip(self.q_target.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    # Default Save Method for DrQ-v2 Agent
    def save(self, filepath):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'q_target_state_dict': self.q_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
        }, filepath)
    
    # Default Load Method for DrQ-v2 Agent
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.q_target.load_state_dict(checkpoint['q_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint.get('step_count', 0)
        self.episode_count = checkpoint.get('episode_count', 0)