import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random

# Import autocast for mixed precision
from torch.cuda.amp import autocast # ADDED/CONFIRMED

class REINFORCEAgent(): # This class is not currently used by Pong_REINFORCEy.py, but corrected for completeness
    def __init__(self, input_shape, action_size, seed, device, gamma, lr, policy):
        """Initialize an Agent object.
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
        """Returns action, log_prob for given state as per current policy."""
        
        # Ensure state is a float32 tensor on the correct device with batch dimension
        # Assuming state is already a PyTorch tensor on the correct device when passed to act
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

        # Normalize returns (ADDED)
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
        self.input_shape = input_shape
        self.action_size = action_size
        random.seed(seed)
        torch.manual_seed(seed)
        self.device = device
        self.lr = lr
        self.gamma = gamma

        self.policy_net = policy_model(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None

        self.log_probs = []
        self.rewards = []
        self.masks = []

    def step(self, log_prob, reward, done):
        self.log_probs.append(log_prob)
        self.rewards.append(torch.tensor([reward], dtype=torch.float32).to(self.device))
        self.masks.append(torch.tensor([1 - done], dtype=torch.float32).to(self.device))

    def act(self, state):
        self.policy_net.eval() 
        action_logits, _ = self.policy_net(state) 
        
        m = Categorical(logits=action_logits) 
        
        action = m.sample()
        log_prob = m.log_prob(action) 
        
        return action.item(), log_prob

    def learn(self):
        self.policy_net.train()

        returns = self.compute_returns(next_value=0, gamma=self.gamma) 

        log_probs = torch.cat(self.log_probs)
        returns = torch.cat(returns).detach()

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        policy_loss = -(log_probs * returns).mean()

        self.optimizer.zero_grad()
        
        if self.scaler:
            self.scaler.scale(policy_loss).backward() 
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            policy_loss.backward()
            self.optimizer.step()

        self.reset_memory() 

    def reset_memory(self):
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