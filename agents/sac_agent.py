import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from torch.distributions import Categorical
from utils.replay_buffer import SACReplayBuffer

class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent for discrete action spaces.
    """
    def __init__(self, input_shape, action_size, device, gamma, lr_actor, lr_critic, alpha, tau, buffer_size, batch_size, actor_m, critic_m):
        self.input_shape = input_shape
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer = SACReplayBuffer(buffer_size)
        
        # Networks
        self.policy_net = actor_m(input_shape, action_size).to(device)
        self.q1_net = critic_m(input_shape, action_size).to(device)
        self.q2_net = critic_m(input_shape, action_size).to(device)
        self.target_q1_net = critic_m(input_shape, action_size).to(device)
        self.target_q2_net = critic_m(input_shape, action_size).to(device)

        # Copy initial weights
        self.target_q1_net.load_state_dict(self.q1_net.state_dict())
        self.target_q2_net.load_state_dict(self.q2_net.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_actor)
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=lr_critic)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=lr_critic)

        # Entropy temperature parameter
        self.target_entropy = -np.log(1.0 / self.action_size)
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)
    
    def act(self, state, evaluate=False):
        """
        Returns an action to take in the environment.
        """
        state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            dist = Categorical(logits=logits)
            
            if evaluate:
                action = torch.argmax(logits, dim=1).item()
            else:
                action = dist.sample().item()
        
        return action

    def step(self, state, action, reward, next_state, done):
        """
        Performs a single update step on the networks.
        """
        self.buffer.add(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update Q-networks
        with torch.no_grad():
            next_logits = self.policy_net(next_states)
            next_probs = F.softmax(next_logits, dim=1)
            next_log_probs = F.log_softmax(next_logits, dim=1)
            
            target_q1_values = self.target_q1_net(next_states)
            target_q2_values = self.target_q2_net(next_states)
            min_target_q = torch.min(target_q1_values, target_q2_values)
            
            next_v_value = (next_probs * (min_target_q - self.log_alpha.exp() * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = rewards.unsqueeze(1) + (self.gamma * (1 - dones.unsqueeze(1)) * next_v_value)
            
        current_q1 = self.q1_net(states).gather(1, actions.unsqueeze(-1))
        current_q2 = self.q2_net(states).gather(1, actions.unsqueeze(-1))

        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update Policy network and Alpha
        for _ in range(1):
            logits = self.policy_net(states)
            log_probs = F.log_softmax(logits, dim=1)
            probs = F.softmax(logits, dim=1)
            
            with torch.no_grad():
                q1_val = self.q1_net(states)
                q2_val = self.q2_net(states)
                min_q = torch.min(q1_val, q2_val)

            # Fixed policy loss
            policy_loss = (probs * (self.log_alpha.exp() * log_probs - min_q)).sum(dim=1).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            # Fixed alpha loss
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Update target networks
        for target_param, local_param in zip(self.target_q1_net.parameters(), self.q1_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        for target_param, local_param in zip(self.target_q2_net.parameters(), self.q2_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

