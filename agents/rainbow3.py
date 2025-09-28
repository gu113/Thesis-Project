import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math
import gc

class NoisyLinear(nn.Module):
    """Noisy Linear Layer for parameter space exploration"""
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, input):
        if self.training:
            return F.linear(input, 
                          self.weight_mu + self.weight_sigma * self.weight_epsilon,
                          self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class RainbowCNN(nn.Module):
    """CNN backbone for Rainbow DQN with distributional output"""
    def __init__(self, input_shape, num_actions, num_atoms=51, v_min=-10, v_max=10):
        super(RainbowCNN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Efficient CNN layers
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        
        # Calculate feature size
        self.feature_size = self._get_conv_out(input_shape)
        
        # Dueling architecture with noisy layers
        self.advantage = nn.Sequential(
            NoisyLinear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            NoisyLinear(512, num_actions * num_atoms)
        )
        
        self.value = nn.Sequential(
            NoisyLinear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            NoisyLinear(512, num_atoms)
        )
        
        # Support for distributional RL
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
        
    def _get_conv_out(self, shape):
        o = self.features(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
        
    def forward(self, x, log=False):
        batch_size = x.size(0)
        
        # Feature extraction
        features = self.features(x)
        features = features.view(batch_size, -1)
        
        # Dueling streams
        advantage = self.advantage(features).view(batch_size, self.num_actions, self.num_atoms)
        value = self.value(features).view(batch_size, 1, self.num_atoms)
        
        # Dueling formula
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply softmax
        if log:
            return F.log_softmax(q_atoms, dim=-1)
        else:
            return F.softmax(q_atoms, dim=-1)
    
    def reset_noise(self):
        """Reset noise for all noisy layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class HybridReplayBuffer:
    """Hybrid replay buffer with smart memory management"""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=1e-4, device='cuda', 
                 gpu_buffer_size=10000, cleanup_freq=1000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.device = device
        self.epsilon = 1e-6
        self.gpu_buffer_size = min(gpu_buffer_size, capacity)  # Keep recent experiences on GPU
        self.cleanup_freq = cleanup_freq
        self.cleanup_counter = 0
        
        # GPU buffer for recent experiences (fast access)
        self.gpu_states = torch.empty((self.gpu_buffer_size, 4, 84, 84), dtype=torch.uint8, device=device)
        self.gpu_actions = torch.empty(self.gpu_buffer_size, dtype=torch.long, device=device)
        self.gpu_rewards = torch.empty(self.gpu_buffer_size, dtype=torch.float32, device=device)
        self.gpu_next_states = torch.empty((self.gpu_buffer_size, 4, 84, 84), dtype=torch.uint8, device=device)
        self.gpu_dones = torch.empty(self.gpu_buffer_size, dtype=torch.bool, device=device)
        
        # CPU buffer for older experiences (memory efficient)
        self.cpu_states = []
        self.cpu_actions = []
        self.cpu_rewards = []
        self.cpu_next_states = []
        self.cpu_dones = []
        
        # Priorities for all experiences
        self.priorities = np.ones(capacity, dtype=np.float32)
        
        self.pos = 0
        self.size = 0
        self.gpu_pos = 0
        self.gpu_size = 0
        
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        # Convert to tensors if needed
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state)
        
        # Add to GPU buffer first
        self.gpu_states[self.gpu_pos] = (state * 255).byte().to(self.device)
        self.gpu_actions[self.gpu_pos] = action
        self.gpu_rewards[self.gpu_pos] = reward
        self.gpu_next_states[self.gpu_pos] = (next_state * 255).byte().to(self.device)
        self.gpu_dones[self.gpu_pos] = done
        
        # Set priority
        if self.size > 0:
            max_priority = self.priorities[:self.size].max()
        else:
            max_priority = 1.0
        self.priorities[self.pos] = max_priority
        
        # Update positions
        self.gpu_pos = (self.gpu_pos + 1) % self.gpu_buffer_size
        self.gpu_size = min(self.gpu_size + 1, self.gpu_buffer_size)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        # Move old GPU data to CPU if GPU buffer is full
        if self.gpu_size == self.gpu_buffer_size and len(self.cpu_states) < (self.capacity - self.gpu_buffer_size):
            oldest_gpu_idx = self.gpu_pos  # This is the oldest after wrapping
            
            # Move to CPU (compressed)
            self.cpu_states.append(self.gpu_states[oldest_gpu_idx].cpu().numpy())
            self.cpu_actions.append(self.gpu_actions[oldest_gpu_idx].cpu().item())
            self.cpu_rewards.append(self.gpu_rewards[oldest_gpu_idx].cpu().item())
            self.cpu_next_states.append(self.gpu_next_states[oldest_gpu_idx].cpu().numpy())
            self.cpu_dones.append(self.gpu_dones[oldest_gpu_idx].cpu().item())
            
            # Limit CPU buffer size
            if len(self.cpu_states) > (self.capacity - self.gpu_buffer_size):
                self.cpu_states.pop(0)
                self.cpu_actions.pop(0)
                self.cpu_rewards.pop(0)
                self.cpu_next_states.pop(0)
                self.cpu_dones.pop(0)
        
        # Periodic cleanup
        self.cleanup_counter += 1
        if self.cleanup_counter >= self.cleanup_freq:
            self._cleanup_memory()
            self.cleanup_counter = 0
    
    def _cleanup_memory(self):
        """Cleanup memory periodically"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def sample(self, batch_size):
        """Sample batch with prioritized sampling"""
        total_size = min(self.size, self.gpu_size + len(self.cpu_states))
        if total_size < batch_size:
            return None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:total_size] ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(total_size, batch_size, p=probs, replace=True)
        
        # Separate GPU and CPU indices
        gpu_indices = indices[indices < self.gpu_size]
        cpu_indices = indices[indices >= self.gpu_size] - self.gpu_size
        
        # Prepare batch tensors
        states = torch.empty((batch_size, 4, 84, 84), dtype=torch.float32, device=self.device)
        actions = torch.empty(batch_size, dtype=torch.long, device=self.device)
        rewards = torch.empty(batch_size, dtype=torch.float32, device=self.device)
        next_states = torch.empty((batch_size, 4, 84, 84), dtype=torch.float32, device=self.device)
        dones = torch.empty(batch_size, dtype=torch.bool, device=self.device)
        
        # Fill from GPU buffer
        if len(gpu_indices) > 0:
            gpu_mask = indices < self.gpu_size
            states[gpu_mask] = self.gpu_states[gpu_indices].float() / 255.0
            actions[gpu_mask] = self.gpu_actions[gpu_indices]
            rewards[gpu_mask] = self.gpu_rewards[gpu_indices]
            next_states[gpu_mask] = self.gpu_next_states[gpu_indices].float() / 255.0
            dones[gpu_mask] = self.gpu_dones[gpu_indices]
        
        # Fill from CPU buffer
        if len(cpu_indices) > 0:
            cpu_mask = indices >= self.gpu_size
            for i, cpu_idx in enumerate(cpu_indices):
                if cpu_idx < len(self.cpu_states):
                    batch_idx = np.where(cpu_mask)[0][i]
                    states[batch_idx] = torch.from_numpy(self.cpu_states[cpu_idx]).float().to(self.device) / 255.0
                    actions[batch_idx] = self.cpu_actions[cpu_idx]
                    rewards[batch_idx] = self.cpu_rewards[cpu_idx]
                    next_states[batch_idx] = torch.from_numpy(self.cpu_next_states[cpu_idx]).float().to(self.device) / 255.0
                    dones[batch_idx] = self.cpu_dones[cpu_idx]
        
        # Calculate importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (total_size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        weights = torch.from_numpy(weights).float().to(self.device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences"""
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()
        self.priorities[indices] = (np.abs(priorities) + self.epsilon) ** self.alpha
    
    def __len__(self):
        return min(self.size, self.gpu_size + len(self.cpu_states))

class RainbowDQNAgent:
    """Rainbow DQN Agent with hybrid memory management"""
    def __init__(self, state_size, action_size, device='cuda', lr=6.25e-5, gamma=0.99,
                 tau=1.0, update_every=4, buffer_size=50000, batch_size=32,
                 num_atoms=51, v_min=-10, v_max=10, n_step=3, target_update_freq=8000,
                 gpu_buffer_size=8000):  # Reduced default buffer size
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.n_step = n_step
        self.target_update_freq = target_update_freq
        
        # Networks
        self.q_network = RainbowCNN(state_size, action_size, num_atoms, v_min, v_max).to(device)
        self.q_target = RainbowCNN(state_size, action_size, num_atoms, v_min, v_max).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, eps=1.5e-4)
        
        # Copy weights to target
        self.q_target.load_state_dict(self.q_network.state_dict())
        
        # Hybrid replay buffer
        self.memory = HybridReplayBuffer(buffer_size, device=device, gpu_buffer_size=gpu_buffer_size)
        
        # N-step buffer
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Support for distributional RL
        self.support = torch.linspace(v_min, v_max, num_atoms, device=device)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Counters
        self.t_step = 0
        self.updates = 0
        
        # Memory management
        self.memory_cleanup_freq = 1000
        self.memory_cleanup_counter = 0
        
    def step(self, state, action, reward, next_state, done):
        """Store experience and learn if ready"""
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) == self.n_step:
            # Calculate n-step return
            n_step_return = 0
            gamma_n = 1
            for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_return += gamma_n * r
                gamma_n *= self.gamma
                if d:
                    break
            
            # Get the appropriate states
            first_state = self.n_step_buffer[0][0]
            first_action = self.n_step_buffer[0][1]
            
            # Find the actual next state (accounting for early termination)
            next_state_idx = min(len(self.n_step_buffer) - 1, self.n_step - 1)
            final_next_state = self.n_step_buffer[next_state_idx][3]
            final_done = any(exp[4] for exp in self.n_step_buffer)
            
            # Store in replay buffer
            self.memory.add(first_state, first_action, n_step_return, final_next_state, final_done)
        
        # Learn every update_every steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            self.learn()
            
        # Periodic memory cleanup
        self.memory_cleanup_counter += 1
        if self.memory_cleanup_counter >= self.memory_cleanup_freq:
            self._cleanup_memory()
            self.memory_cleanup_counter = 0
    
    def _cleanup_memory(self):
        """Clean up memory periodically"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def act(self, state, eps=0.0):
        """Choose action using the policy network"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        
        self.q_network.eval()
        with torch.no_grad():
            # Get distributional Q-values
            q_dist = self.q_network(state)
            # Convert to expected Q-values
            q_values = (q_dist * self.support).sum(dim=2)
            action = q_values.argmax(dim=1).item()
        self.q_network.train()
        
        # Epsilon-greedy exploration (though noisy nets handle most exploration)
        if random.random() < eps:
            return random.randrange(self.action_size)
        return action
    
    def learn(self):
        """Update value parameters using sampled batch"""
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return
            
        states, actions, rewards, next_states, dones, indices, weights = batch
        
        # Current Q-value distributions
        current_q_dist = self.q_network(states, log=True)
        current_q_dist = current_q_dist[range(self.batch_size), actions]
        
        with torch.no_grad():
            # Double DQN: use online network to select actions
            next_q_dist = self.q_network(next_states)
            next_q_values = (next_q_dist * self.support).sum(dim=2)
            next_actions = next_q_values.argmax(dim=1)
            
            # Use target network to evaluate actions
            next_q_dist_target = self.q_target(next_states)
            next_q_dist_target = next_q_dist_target[range(self.batch_size), next_actions]
            
            # Compute target support
            target_support = rewards.unsqueeze(1) + (self.gamma ** self.n_step) * self.support.unsqueeze(0) * (~dones).unsqueeze(1)
            target_support = target_support.clamp(self.v_min, self.v_max)
            
            # Distribute probability
            b = (target_support - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Handle edge case where l == u
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1
            
            # Distribute probabilities
            target_dist = torch.zeros_like(next_q_dist_target)
            offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size, 
                                  dtype=torch.long, device=self.device).unsqueeze(1).expand_as(l)
            
            target_dist.view(-1).index_add_(0, (l + offset).view(-1), 
                                          (next_q_dist_target * (u.float() - b)).view(-1))
            target_dist.view(-1).index_add_(0, (u + offset).view(-1), 
                                          (next_q_dist_target * (b - l.float())).view(-1))
        
        # Compute loss
        loss_per_sample = -(target_dist * current_q_dist).sum(dim=1)
        loss = (loss_per_sample * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        # Update priorities
        priorities = loss_per_sample.detach()
        self.memory.update_priorities(indices, priorities)
        
        # Reset noise
        self.q_network.reset_noise()
        self.q_target.reset_noise()
        
        # Update target network
        self.updates += 1
        if self.updates % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.q_target.load_state_dict(checkpoint['q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])