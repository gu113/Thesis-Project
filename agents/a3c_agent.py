import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class A3CAgent():
    def __init__(self, input_shape, action_size, seed, device, gamma, lr_actor, lr_critic, update_every, actor_m, critic_m, value_loss_coef=0.5, entropy_coef=0.01):
        self.input_shape = input_shape
        self.action_size = action_size
        random.seed(seed)
        torch.manual_seed(seed)
        self.device = device
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.update_every = update_every
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.actor_net = actor_m(input_shape, action_size).to(self.device)
        self.critic_net = critic_m(input_shape).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr_critic)

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.entropies = []
        self.t_step = 0

    def act(self, state):
        self.actor_net.eval()
        state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            action_logits = self.actor_net(state_tensor)
        m = Categorical(logits=action_logits)
        action = m.sample()
        log_prob = m.log_prob(action)
        entropy = m.entropy().mean()
        return action.item(), log_prob, entropy

    def step(self, state, log_prob, entropy, reward, done, next_state_raw):
        state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        value = self.critic_net(state_tensor)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(torch.tensor([reward], dtype=torch.float32).to(self.device))
        self.masks.append(torch.tensor([1 - done], dtype=torch.float32).to(self.device))
        self.entropies.append(entropy)
        self.t_step += 1
        
        if self.t_step % self.update_every == 0 or done:
            next_state_tensor = torch.from_numpy(next_state_raw).unsqueeze(0).float().to(self.device)
            self.learn(next_state_tensor)
            self.reset_memory()

    def learn(self, next_state):
        self.actor_net.train()
        self.critic_net.train()

        with torch.no_grad():
            next_value = self.critic_net(next_state)
        returns = self.compute_returns(next_value, self.gamma)
        log_probs_batch = torch.stack(self.log_probs)
        values_batch = torch.cat(self.values)
        entropies_batch = torch.stack(self.entropies)
        returns = returns.detach()
        
        advantage = returns - values_batch
        if len(advantage) > 1 and advantage.std() > 1e-6:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

        actor_loss = -(log_probs_batch * advantage.squeeze(-1).detach()).mean()
        critic_loss = F.mse_loss(values_batch.squeeze(-1), returns.squeeze(-1))
        entropy_loss = entropies_batch.mean()
        total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy_loss
        
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def reset_memory(self):
        del self.log_probs[:]
        del self.rewards[:]
        del self.masks[:]
        del self.values[:]
        del self.entropies[:]
        self.t_step = 0
        
    def compute_returns(self, next_value, gamma):
        R = next_value.squeeze(-1)
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + gamma * R * self.masks[step]
            returns.insert(0, R)
        return torch.stack(returns).to(self.device)