import numpy as np
import random
from collections import namedtuple, deque
import torch

from utils.sum_tree import SumTree, RainbowSumTree, RainbowSumTreeGPU

class ReplayBuffer:
    """Buffer to store experience tuples"""

    def __init__(self, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object"""

        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        #first_state = experiences[0].state
        #if isinstance(first_state, torch.Tensor): # GPU Version
        states = torch.stack([e.state for e in experiences]).float().to(self.device)
        actions = torch.tensor([e.action for e in experiences], device=self.device, dtype=torch.long)
        rewards = torch.tensor([e.reward for e in experiences], device=self.device, dtype=torch.float32)
        next_states = torch.stack([e.next_state for e in experiences]).float().to(self.device)
        dones = torch.tensor([e.done for e in experiences], device=self.device, dtype=torch.bool)

        """
        else: # CPU Version (Original)
            states = torch.from_numpy(np.array([e.state for e in experiences])).float().to(self.device)
            actions = torch.from_numpy(np.array([e.action for e in experiences])).long().to(self.device)
            rewards = torch.from_numpy(np.array([e.reward for e in experiences])).float().to(self.device)
            next_states = torch.from_numpy(np.array([e.next_state for e in experiences])).float().to(self.device)
            dones = torch.from_numpy(np.array([e.done for e in experiences])).bool().to(self.device)
        """
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)


class OldPERBuffer:
    """Old Implementation of a Prioritized Experience Replay (PER) buffer"""

    def __init__(self, buffer_size, batch_size, seed, device, alpha=0.6):
        """Initialize a ReplayBufferPER object"""
        self.memory = deque(maxlen=buffer_size)
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.position = 0
    
    def add(self, state, action, reward, next_state, done, td_error=None):
        """Add a new experience to memory with max priority"""
        ##max_priority = self.priorities.max() if self.memory else 1.0
        experience = self.experience(state, action, reward, next_state, done)
        
        if len(self.memory) < self.buffer_size:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience

        if td_error is not None:
            priority = abs(td_error) + 1e-6
        else:
            priority = self.priorities.max() if self.memory else 1.0
        
        ##self.priorities[self.position] = max_priority
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.memory.maxlen
    
    def sample(self, beta=0.4):
        """Sample a batch of experiences based on priority"""
        if len(self.memory) == 0:
            return [], [], [], [], []

        # Ensure priorities are non-zero and bounded
        priorities = self.priorities[: len(self.memory)] ** self.alpha
        priorities = np.maximum(priorities, 1e-6)  # Avoid zero priorities

        # Normalize probabilities
        probs = priorities / priorities.sum()
        
        # Check for NaN or Inf in the probabilities
        if np.isnan(probs).any() or np.isinf(probs).any():
            print("Warning: NaN or Inf found in probabilities")
            probs = np.ones_like(probs) / len(probs)  # Fallback to uniform distribution

        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        experiences = [self.memory[i] for i in indices]

        # Calculate Importance Sampling Weights
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)

        # Prevent extreme weight values (overflow)
        weights = np.clip(weights, a_min=1e-6, a_max=10.0)  # Clip to avoid extreme weights

        # Normalize weights
        weights /= weights.max()

        states = torch.from_numpy(np.array([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.array([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.array([e.done for e in experiences], dtype=np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones, indices, torch.tensor(weights, dtype=torch.float32).to(self.device)

    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD-error"""
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()

        td_errors = np.nan_to_num(td_errors, nan=1.0, posinf=1.0, neginf=1.0) # Replace NaNs/Infs with 1.0
        self.priorities[indices] = np.abs(td_errors) + 1e-6  # Small offset to avoid zero priority

    
    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)


class AsyncReplayBuffer:
    """Async Version of the Replay Buffer"""

    def __init__(self, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object"""

        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)
        first_state = experiences[0].state

        # Old CPU and GPU Versions (Without if statement)
        """
        states = torch.stack([e.state for e in experiences if e is not None]).float()
        states = states.cpu().numpy()  # Move to CPU before converting to numpy
        states = torch.from_numpy(states).float().to(self.device)  # Move it back to the device

        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).long().to(self.device)

        rewards = torch.stack([e.reward for e in experiences if e is not None]).float()
        rewards = rewards.cpu().numpy()  # Move to CPU before converting to numpy
        rewards = torch.from_numpy(rewards).float().to(self.device)  # Move it back to the device

        next_states = torch.stack([e.next_state for e in experiences if e is not None]).float()
        next_states = next_states.cpu().numpy()  # Move to CPU before converting to numpy
        next_states = torch.from_numpy(next_states).float().to(self.device)  # Move it back to the device

        dones = torch.stack([e.done for e in experiences if e is not None]).float()
        dones = dones.cpu().numpy()  # Move to CPU before converting to numpy
        dones = torch.from_numpy(dones).float().to(self.device)  # Move it back to the device

        
        states = torch.tensor(([e.state for e in experiences]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(([e.next_state for e in experiences]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32, device=self.device)
        
        states = torch.tensor(np.stack([e.state.cpu().numpy() for e in experiences]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([e.action for e in experiences]), dtype=torch.int64, device=self.device)
        rewards = torch.tensor(np.stack([e.reward.cpu().numpy() if torch.is_tensor(e.reward) else np.array(e.reward) for e in experiences]), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([e.next_state.cpu().numpy() for e in experiences]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.stack([e.done.cpu().numpy() if torch.is_tensor(e.done) else np.array(e.done) for e in experiences]), dtype=torch.float32, device=self.device)
        """

        if isinstance(first_state, torch.Tensor): # GPU Version
            states = torch.stack([e.state for e in experiences]).to(self.device, non_blocking=True)
            actions = torch.tensor([e.action for e in experiences], device=self.device, dtype=torch.long)
            rewards = torch.tensor([e.reward for e in experiences], device=self.device, dtype=torch.float32)
            next_states = torch.stack([e.next_state for e in experiences]).to(self.device, non_blocking=True)
            dones = torch.tensor([e.done for e in experiences], device=self.device, dtype=torch.float32)

        else: # CPU Version (Original)
            states = torch.from_numpy(np.array([e.state for e in experiences])).float().to(self.device)
            actions = torch.from_numpy(np.array([e.action for e in experiences])).long().to(self.device)
            rewards = torch.from_numpy(np.array([e.reward for e in experiences])).float().to(self.device)
            next_states = torch.from_numpy(np.array([e.next_state for e in experiences])).float().to(self.device)
            dones = torch.from_numpy(np.array([e.done for e in experiences])).float().to(self.device)

    
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class AsyncRamReplayBuffer:
    """Async + RAM Version of the Replay Buffer"""

    def __init__(self, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object"""
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.tensor(np.stack([e.state for e in experiences]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([e.next_state for e in experiences]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32, device=self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)


class PERBuffer:
    """Prioritized Experience Replay (PER) Buffer"""

    def __init__(self, buffer_size, batch_size, seed, device, alpha=0.6, beta=0.4, beta_increment=0.001):
        """Initialize a ReplayBufferPER object"""
        self.tree = SumTree(buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
        
        # PER parameters
        self.alpha = alpha  # How much prioritization to use (0 = uniform, 1 = full prioritization)
        self.beta = beta    # Importance sampling compensation (0 = no compensation, 1 = full compensation)
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # Small value to prevent zero priorities
        self.max_priority = 1.0 # Maximum priority for new experiences
        
    def add(self, max_priority, state, action, reward, next_state, done):
        """Add a new experience to memory with max priority"""
        self.max_priority = max_priority

        state = torch.tensor(state, dtype=torch.float32) if not isinstance(state, torch.Tensor) else state.detach()
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32) if not isinstance(next_state, torch.Tensor) else next_state.detach()
        done = torch.tensor(done, dtype=torch.float32)

        experience = self.experience(state, action, reward, next_state, done)
        priority = max_priority  # New experiences get max priority
        self.max_priority = max(self.max_priority, priority)
        self.tree.add(priority ** self.alpha, experience)
    
    def sample(self):
        """Sample a batch of experiences based on priority"""
        experiences = []
        indices = []
        priorities = []
        
        # Calculate segment size for stratified sampling
        segment = self.tree.total() / self.batch_size
        
        # Increase beta towards 1.0 over time
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(self.batch_size):
            # Stratified sampling: divide total priority into batch_size segments
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, experience = self.tree.get(s)
            
            if experience is not None:
                experiences.append(experience)
                indices.append(idx)
                priorities.append(priority)
        
        # Old Tensor Conversion Version
        """
        states = torch.from_numpy(np.array([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.array([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.array([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        
        states = torch.stack([e.state for e in experiences]).float().to(self.device)
        actions = torch.stack([e.action for e in experiences]).long().to(self.device)
        rewards = torch.stack([e.reward for e in experiences]).float().to(self.device)
        next_states = torch.stack([e.next_state for e in experiences]).float().to(self.device)
        dones = torch.stack([e.done for e in experiences]).float().to(self.device)
        
        """

        total_priority = self.tree.total()

        # Calculate importance sampling weights
        if len(priorities) > 0:
            sampling_probabilities = np.array(priorities) / total_priority
            is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
            is_weights = is_weights / is_weights.max()  # Normalize
            is_weights = torch.from_numpy(is_weights).float().to(self.device)
        else:
            is_weights = torch.ones(len(experiences)).float().to(self.device)

        
        ##return (states, actions, rewards, next_states, dones, indices, is_weights)
        return(experiences, indices, is_weights)
    
    def update_priorities(self, indices, priorities):
        """Update priorities based on TD errors"""
        for idx, priority in zip(indices, priorities):
            priority = float(abs(priority)) + self.epsilon  # Ensure non-zero priority
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority ** self.alpha)
    
    def __len__(self):
        return self.tree.n_entries

class ComplexPERBuffer:
    """Complex Prioritized Experience Replay (PER) Buffer with dual buffers"""

    def __init__(self, buffer_size, batch_size, seed, device, reward_threshold=100, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """Initialize a ReplayBufferPER object"""
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        random.seed(seed)
        np.random.seed(seed)

        self.reward_threshold = reward_threshold
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.high_reward_buffer = []
        self.high_reward_priorities = []

        self.normal_buffer = []
        self.normal_priorities = []

        self.experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done", "episode_reward"])

    def add(self, state, action, reward, next_state, done, episode_reward=0):
        """Add a new experience to memory"""
        experience = self.experience(state, action, reward, next_state, done, episode_reward)
        max_priority_high = max(self.high_reward_priorities) if self.high_reward_priorities else 1.0
        max_priority_normal = max(self.normal_priorities) if self.normal_priorities else 1.0

        if episode_reward > self.reward_threshold or reward > 10:
            self.high_reward_buffer.append(experience)
            self.high_reward_priorities.append(max_priority_high)
            if len(self.high_reward_buffer) > self.buffer_size // 4:
                self.high_reward_buffer.pop(0)
                self.high_reward_priorities.pop(0)
        else:
            self.normal_buffer.append(experience)
            self.normal_priorities.append(max_priority_normal)
            if len(self.normal_buffer) > (3 * self.buffer_size // 4):
                self.normal_buffer.pop(0)
                self.normal_priorities.pop(0)

    def _sample_prioritized(self, buffer, priorities, sample_size):
        """Sample experiences from a given buffer based on priorities"""
        if len(buffer) == 0:
            return [], [], []

        scaled_priorities = np.array(priorities) ** self.alpha
        prob_dist = scaled_priorities / scaled_priorities.sum()

        indices = np.random.choice(len(buffer), sample_size, p=prob_dist)
        experiences = [buffer[idx] for idx in indices]

        total = len(buffer)
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        weights = (total * prob_dist[indices]) ** (-beta)
        weights = weights / weights.max()

        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        return experiences, indices, weights

    def sample(self):
        """Sample a batch of experiences from both buffers"""
        high_sample_size = min(int(self.batch_size * 0.6), len(self.high_reward_buffer))
        normal_sample_size = self.batch_size - high_sample_size

        high_exp, high_indices, high_weights = self._sample_prioritized(self.high_reward_buffer, self.high_reward_priorities, high_sample_size)
        normal_exp, normal_indices, normal_weights = self._sample_prioritized(self.normal_buffer, self.normal_priorities, normal_sample_size)

        experiences = high_exp + normal_exp
        indices = list(high_indices) + [len(self.high_reward_buffer) + i for i in normal_indices]
        
        if isinstance(high_weights, list):
            high_weights = torch.tensor(high_weights, dtype=torch.float32).to(self.device)
        if isinstance(normal_weights, list):
            normal_weights = torch.tensor(normal_weights, dtype=torch.float32).to(self.device)

        if high_weights.numel() > 0 and normal_weights.numel() > 0:
            weights = torch.cat([high_weights, normal_weights])
        elif high_weights.numel() > 0:
            weights = high_weights
        else:
            weights = normal_weights

        if len(experiences) == 0:
            return None

        if isinstance(experiences[0].state, torch.Tensor):
            states = torch.stack([e.state for e in experiences]).float().to(self.device)
            actions = torch.tensor([e.action for e in experiences], dtype=torch.long).to(self.device)
            rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float).to(self.device)
            next_states = torch.stack([e.next_state for e in experiences]).float().to(self.device)
            dones = torch.tensor([e.done for e in experiences], dtype=torch.float).to(self.device)
        else:
            states = torch.from_numpy(np.array([e.state for e in experiences])).float().to(self.device)
            actions = torch.from_numpy(np.array([e.action for e in experiences])).long().to(self.device)
            rewards = torch.from_numpy(np.array([e.reward for e in experiences])).float().to(self.device)
            next_states = torch.from_numpy(np.array([e.next_state for e in experiences])).float().to(self.device)
            dones = torch.from_numpy(np.array([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, indices, priorities):
        """Update priorities based on TD errors"""
        high_len = len(self.high_reward_buffer)
        for idx, prio in zip(indices, priorities):
            if idx < high_len:
                self.high_reward_priorities[idx] = prio
            else:
                normal_idx = idx - high_len
                if normal_idx < len(self.normal_priorities):
                    self.normal_priorities[normal_idx] = prio

    def __len__(self):
        return len(self.high_reward_buffer) + len(self.normal_buffer)
    

class SACReplayBuffer:
    """Replay buffer to store and sample experiences for SAC algorithm"""

    def __init__(self, capacity):
        """Initialize a SACReplayBuffer object"""
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)

        # Convert the batch of NumPy arrays to a batch of PyTorch tensors
        state, action, reward, next_state, done = zip(*batch)

        # Stack the individual arrays and convert them to tensors
        state_tensor = torch.from_numpy(np.stack(state)).float().cuda()
        action_tensor = torch.LongTensor(action).cuda()
        reward_tensor = torch.FloatTensor(reward).unsqueeze(1).cuda()
        next_state_tensor = torch.from_numpy(np.stack(next_state)).float().cuda()
        done_tensor = torch.FloatTensor(done).unsqueeze(1).cuda()

        return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor

    def __len__(self):
        return len(self.buffer)
    

class RainbowPERBuffer:
    """Rainbow Prioritized Experience Replay (PER) Buffer"""

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """Initialize a RainbowPERBuffer object"""
        self.tree = RainbowSumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start
        self.frame = 0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory with max priority"""
        transition = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size):
        """Sample a batch of experiences based on priority"""
        assert self.tree.n_entries >= batch_size
        
        batch_indices = np.zeros(batch_size, dtype=np.int32)
        batch_experiences = []
        batch_weights = np.zeros(batch_size, dtype=np.float32)

        segment = self.tree.total_priority() / batch_size
        self.beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            
            prob = priority / self.tree.total_priority()
            weight = (prob * self.tree.n_entries)**(-self.beta)
            batch_weights[i] = weight

            batch_indices[i] = idx
            batch_experiences.append(data)
        
        batch_weights /= batch_weights.max()
        
        states, actions, rewards, next_states, dones = zip(*batch_experiences)

        states = torch.stack(list(states))
        actions = torch.stack(list(actions))
        rewards = torch.stack(list(rewards))
        next_states = torch.stack(list(next_states))
        dones = torch.stack(list(dones))
        weights = torch.tensor(batch_weights, dtype=torch.float32)

        return batch_indices, states, actions, rewards, next_states, dones, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-5)**self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries
    

class DrQv2Buffer:
    """DrQv2 Replay Buffer"""

    def __init__(self, capacity, state_shape=(4, 84, 84), device='cuda'):
        """Initialize a DrQv2Buffer object"""
        self.capacity = capacity
        self.device = device
        self.state_shape = state_shape
        
        self.states = torch.empty((capacity, *state_shape), dtype=torch.uint8, device=device)
        self.actions = torch.empty(capacity, dtype=torch.long, device=device)
        self.rewards = torch.empty(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.empty((capacity, *state_shape), dtype=torch.uint8, device=device)
        self.dones = torch.empty(capacity, dtype=torch.bool, device=device)
        
        self.pos = 0
        self.size = 0
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        self.states[self.pos].copy_(torch.from_numpy((state * 255).astype(np.uint8)).to(self.device))
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos].copy_(torch.from_numpy((next_state * 255).astype(np.uint8)).to(self.device))
        self.dones[self.pos] = done
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        if self.size < batch_size:
            return None
            
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        states = self.states[indices].float() / 255.0
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices].float() / 255.0
        dones = self.dones[indices]
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return self.size
    

class RainbowPERBufferGPU:
    """Rainbow Prioritized Experience Replay (PER) Buffer with GPU acceleration"""

    def __init__(self, capacity, device, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.device = device
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start
        self.frame = 0
        self.max_priority = 1.0
        
        self.tree = RainbowSumTreeGPU(capacity, device)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory with max priority"""
        transition = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size):
        """Sample a batch of experiences based on priority"""
        assert self.tree.n_entries >= batch_size
        
        # Calculate current beta and total priority
        self.beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        total_priority = self.tree.total_priority()

        # Generate random numbers for each segment
        segment = total_priority / batch_size
        s_values = torch.linspace(0, batch_size - 1, batch_size, device=self.device) * segment
        s_values += torch.rand(batch_size, device=self.device) * segment

        # Loop through the tree to retrieve indices
        batch_indices = []
        batch_priorities = []
        batch_experiences = []
        
        for s in s_values.cpu().tolist():
            idx, priority, data = self.tree.get(s)
            batch_indices.append(idx)
            batch_priorities.append(priority)
            batch_experiences.append(data)

        # Convert everything to tensors
        batch_indices_tensor = torch.tensor(batch_indices, dtype=torch.long, device=self.device)
        batch_priorities_tensor = torch.tensor(batch_priorities, dtype=torch.float32, device=self.device)

        # Vectorized calculation of weights
        probs = batch_priorities_tensor / total_priority
        weights = (self.tree.n_entries * probs)**(-self.beta)
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*batch_experiences)

        states = torch.stack(list(states), dim=0).to(self.device)
        actions = torch.stack(list(actions), dim=0).to(self.device)
        rewards = torch.stack(list(rewards), dim=0).to(self.device)
        next_states = torch.stack(list(next_states), dim=0).to(self.device)
        dones = torch.stack(list(dones), dim=0).to(self.device)
        
        return batch_indices_tensor, states, actions, rewards, next_states, dones, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities of sampled transitions"""

        indices_cpu = indices.cpu().tolist()
        
        # Calculate new priorities
        new_priorities = (td_errors.abs() + 1e-5)**self.alpha
        
        # Loop over items and update tree
        for idx, new_p in zip(indices_cpu, new_priorities.cpu().tolist()):
            self.tree.update(idx, new_p)

    def __len__(self):
        return self.tree.n_entries