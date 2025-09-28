import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random


class A2CAgent():
    def __init__(self, input_shape, action_size, seed, device, gamma, alpha, beta, update_every, actor_m, critic_m):
        """Initialize an Agent object.
        Params
        ======
            input_shape (tuple): dimension of each state (C, H, W)
            action_size (int): dimension of each action
            seed (int): random seed
            device(string): Use Gpu or CPU
            gamma (float): discount factor
            alpha (float): Actor learning rate
            beta (float): Critic learning rate 
            update_every (int): how often to update the network
            actor_m(Model): Pytorch Actor Model
            critic_m(Model): PyTorch Critic Model
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.update_every = update_every

        # Actor-Network
        self.actor_net = actor_m(input_shape, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha)

        # Critic-Network
        self.critic_net = critic_m(input_shape).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.beta)

        # Memory
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.masks     = []
        self.entropies = []

        self.t_step = 0

    def step(self, state, log_prob, entropy, reward, done, next_state):

        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device) # Added float() to fix type mismatch
        
        value = self.critic_net(state)
        
        # Save experience in  memory
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(torch.from_numpy(np.array([reward])).to(self.device))
        self.masks.append(torch.from_numpy(np.array([1 - done])).to(self.device))
        self.entropies.append(entropy)

        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
           self.learn(next_state)
           self.reset_memory()
                
    def act(self, state):
        """Returns action, log_prob, entropy for given state as per current policy."""
        
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device) # Added float() to fix type mismatch
        action_probs = self.actor_net(state)

        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        entropy = action_probs.entropy().mean()

        return action.item(), log_prob, entropy

        
        
    def learn(self, next_state):
        next_state = torch.from_numpy(next_state).unsqueeze(0).float().to(self.device) # Added float() to fix type mismatch
        next_value = self.critic_net(next_state)

        returns = self.compute_returns(next_value, self.gamma)

        log_probs = torch.cat(self.log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(self.values)

        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * sum(self.entropies)

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def reset_memory(self):
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]
        del self.masks[:]
        del self.entropies[:]

    def compute_returns(self, next_value, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + gamma * R * self.masks[step]
            returns.insert(0, R)
        return returns
    

class ComplexA2CAgent:
    def __init__(self, input_shape, action_size, seed, device, gamma, alpha, beta, update_every, actor_m, critic_m, value_loss_coef=0.5, entropy_coef=0.01, gae_lambda=0.95):
        """Initialize an A2C Agent object."""
        self.input_shape = input_shape
        self.action_size = action_size
        random.seed(seed)
        torch.manual_seed(seed)
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.update_every = update_every
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda

        # Separate Actor and Critic networks
        self.actor_net = actor_m(input_shape, action_size).to(self.device)
        self.critic_net = critic_m(input_shape).to(self.device)

        # Separate Optimizers
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.beta)

        # Memory for the current batch of trajectory
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.masks = []
        self.entropies = []

        self.t_step = 0

    def step(self, state, log_prob, entropy, reward, done):
        """
        Stores experience for the current batch of steps.
        """
        # Save experience in memory
        self.states.append(torch.from_numpy(state).float().to(self.device))
        self.log_probs.append(log_prob)
        self.rewards.append(torch.tensor([reward], dtype=torch.float32).to(self.device))
        self.masks.append(torch.tensor([1 - done], dtype=torch.float32).to(self.device))
        self.entropies.append(entropy)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            self.learn(self.states[-1])  # Pass the last state of the batch as the next state
            self.reset_memory()

    def act(self, state):
        """Returns action, log_prob, and entropy for given state as per current policy."""
        self.actor_net.eval()

        state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            action_logits = self.actor_net(state_tensor)

        m = Categorical(logits=action_logits)

        action = m.sample()
        log_prob = m.log_prob(action)
        entropy = m.entropy().mean()

        return action.item(), log_prob, entropy

    def learn(self, next_state):
        """
        Performs policy and value updates using the collected batch of trajectory.
        """
        self.actor_net.train()
        self.critic_net.train()

        # Concatenate lists of tensors into single tensors
        states_batch = torch.stack(self.states)
        log_probs_batch = torch.stack(self.log_probs)
        rewards_batch = torch.cat(self.rewards).squeeze()
        masks_batch = torch.cat(self.masks).squeeze()
        entropies_batch = torch.stack(self.entropies)
        
        # CRITICAL FIX: Get values for the entire batch and the next state
        values_batch = self.critic_net(states_batch).squeeze(-1)
        
        with torch.no_grad():
            next_value = self.critic_net(next_state.unsqueeze(0)).squeeze()

        # Compute GAE advantages and returns
        gae = 0
        advantages = []
        
        for step in reversed(range(len(rewards_batch))):
            if step == len(rewards_batch) - 1:
                next_val = next_value * masks_batch[step]
            else:
                next_val = values_batch[step + 1] * masks_batch[step]
                
            delta = rewards_batch[step] + self.gamma * next_val - values_batch[step]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        advantages = torch.stack(advantages).to(self.device)
        returns = advantages + values_batch

        # Normalize advantages (optional but beneficial for stability)
        if len(advantages) > 1 and advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # Calculate Actor Loss (Policy Loss)
        actor_loss = -(log_probs_batch * advantages.detach()).mean()

        # Calculate Critic Loss (Value Loss)
        critic_loss = F.mse_loss(values_batch, returns.detach())

        # Calculate Entropy for exploration regularization
        entropy_loss = entropies_batch.mean()

        # Total Loss for A2C
        total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy_loss

        # Optimization Step
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()

        # Clip gradients to prevent them from exploding
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def reset_memory(self):
        """Clears the collected states, log probabilities, rewards, masks, and entropies."""
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.masks[:]
        del self.entropies[:]