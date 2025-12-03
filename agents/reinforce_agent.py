import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random

class REINFORCEAgent():
    def __init__(self, input_shape, action_size, seed, device, gamma, lr, policy):
        """Initialize a REINFORCE Agent
        Params
        ======
            input_shape (tuple): dimension of each state (C, H, W)
            action_size (int): dimension of each action
            seed (int): random seed
            device(string): Use Gpu or CPU
            gamma (float): discount factor
            lr (float): Learning rate
            policy(Model): Pytorch Policy Model
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.lr = lr
        self.gamma = gamma

        # Actor-Network
        self.policy_net = policy(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Memory
        self.log_probs = []
        self.rewards   = []
        self.masks     = []

    def step(self, log_prob, reward, done):

        # Save experience in  memory
        self.log_probs.append(log_prob)
        self.rewards.append(torch.from_numpy(np.array([reward])).to(self.device))
        self.masks.append(torch.from_numpy(np.array([1 - done])).to(self.device))

                
    def act(self, state):

        # Ensure state is a float32 tensor on the correct device
        if state.ndim == 3:
            state = state.unsqueeze(0)

        action_probs = self.policy_net(state)

        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)

        return action.item(), log_prob
        
    def learn(self):

        returns = self.compute_returns(0, self.gamma)

        log_probs = torch.cat(self.log_probs)
        returns   = torch.cat(returns).detach()

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss  = -(log_probs * returns).mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.reset_memory()

    def reset_memory(self):
        del self.log_probs[:]
        del self.rewards[:]
        del self.masks[:]

    def compute_returns(self, next_value, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + gamma * R * self.masks[step]
            returns.insert(0, R)
        return returns
    

class ComplexREINFORCEAgent():
    def __init__(self, input_shape, action_size, seed, device, gamma, lr, policy_model):
        """Initialize a more Complex REINFORCE Agent with mixed precision training support
        Params
        ======
            input_shape (tuple): dimension of each state (C, H, W)
            action_size (int): dimension of each action
            seed (int): random seed
            device(string): Use Gpu or CPU
            gamma (float): discount factor
            lr (float): Learning rate
            policy_model(Model): Pytorch Policy Model
        """
        self.input_shape = input_shape
        self.action_size = action_size
        random.seed(seed)
        torch.manual_seed(seed)
        self.device = device
        self.lr = lr
        self.gamma = gamma

        # Policy Network
        self.policy_net = policy_model(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None

        # Memory
        self.log_probs = []
        self.rewards = []
        self.masks = []

    def step(self, log_prob, reward, done):
        # Save experience in memory
        self.log_probs.append(log_prob)
        self.rewards.append(torch.tensor([reward], dtype=torch.float32).to(self.device))
        self.masks.append(torch.tensor([1 - done], dtype=torch.float32).to(self.device))

    def act(self, state):
        self.policy_net.eval() 
        action_logits, _ = self.policy_net(state) 
        
        # Create a categorical distribution over the action logits
        m = Categorical(logits=action_logits) 
        
        action = m.sample()
        log_prob = m.log_prob(action) 
        
        return action.item(), log_prob

    def learn(self):
        self.policy_net.train()

        # Compute returns
        returns = self.compute_returns(next_value=0, gamma=self.gamma) 

        log_probs = torch.cat(self.log_probs)
        returns = torch.cat(returns).detach()

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        policy_loss = -(log_probs * returns).mean()

        self.optimizer.zero_grad()
        
        # Backward pass and optimization step with mixed precision support
        if self.scaler:
            self.scaler.scale(policy_loss).backward() 
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            policy_loss.backward()
            self.optimizer.step()

        self.reset_memory() 

    def reset_memory(self):
        # Clear the memory buffers
        del self.log_probs[:]
        del self.rewards[:]
        del self.masks[:]

    def compute_returns(self, next_value, gamma):
        R = next_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + gamma * R * self.masks[step]
            returns.insert(0, R)
        return returns