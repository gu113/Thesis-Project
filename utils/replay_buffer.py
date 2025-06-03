import numpy as np
import random
from collections import namedtuple, deque
import torch

from utils.sum_tree import SumTree

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (string): GPU or CPU
        """

        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        first_state = experiences[0].state

        if isinstance(first_state, torch.Tensor): # GPU Version
            states = torch.stack([e.state for e in experiences]).to(self.device).float()
            actions = torch.tensor([e.action for e in experiences], device=self.device, dtype=torch.long)
            rewards = torch.tensor([e.reward for e in experiences], device=self.device, dtype=torch.float32)
            next_states = torch.stack([e.next_state for e in experiences]).to(self.device).float()
            dones = torch.tensor([e.done for e in experiences], device=self.device, dtype=torch.float32)

        else: # CPU Version (Original)
            states = torch.from_numpy(np.array([e.state for e in experiences])).float().to(self.device)
            actions = torch.from_numpy(np.array([e.action for e in experiences])).long().to(self.device)
            rewards = torch.from_numpy(np.array([e.reward for e in experiences])).float().to(self.device)
            next_states = torch.from_numpy(np.array([e.next_state for e in experiences])).float().to(self.device)
            dones = torch.from_numpy(np.array([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OldPERBuffer:
    """Prioritized Experience Replay (PER) buffer."""

    def __init__(self, buffer_size, batch_size, seed, device, alpha=0.6):
        """Initialize a ReplayBufferPER object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (string): GPU or CPU
            alpha (float): prioritization exponent (0 = uniform, 1 = full prioritization)
        """
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
        """Add a new experience to memory with max priority."""
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
        """Sample a batch of experiences based on priority."""
        if len(self.memory) == 0:
            return [], [], [], [], []

        # Ensure priorities are non-zero and bounded
        priorities = self.priorities[: len(self.memory)] ** self.alpha
        priorities = np.maximum(priorities, 1e-6)  # Avoid zero priorities

        # Normalize probabilities
        probs = priorities / priorities.sum()
        
        # Check for NaN or Inf in the probabilities
        if np.isnan(probs).any() or np.isinf(probs).any():
            print("Warning: NaN or Inf found in probabilities!")
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
        """Update priorities based on TD-error."""
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()

        td_errors = np.nan_to_num(td_errors, nan=1.0, posinf=1.0, neginf=1.0) # Replace NaNs/Infs with 1.0
        self.priorities[indices] = np.abs(td_errors) + 1e-6  # Small offset to avoid zero priority

    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class AsyncReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (string): GPU or CPU
        """

        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

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
        """

        
        states = torch.tensor(np.stack([e.state for e in experiences]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([e.next_state for e in experiences]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32, device=self.device)
        """
        states = torch.tensor(np.stack([e.state.cpu().numpy() for e in experiences]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([e.action for e in experiences]), dtype=torch.int64, device=self.device)
        rewards = torch.tensor(np.stack([e.reward.cpu().numpy() if torch.is_tensor(e.reward) else np.array(e.reward) for e in experiences]), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([e.next_state.cpu().numpy() for e in experiences]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.stack([e.done.cpu().numpy() if torch.is_tensor(e.done) else np.array(e.done) for e in experiences]), dtype=torch.float32, device=self.device)
        """

    
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class AsyncRamReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (string): GPU or CPU
        """

        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.tensor(np.stack([e.state for e in experiences]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([e.next_state for e in experiences]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32, device=self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PERBuffer:
    """Prioritized Experience Replay Buffer"""
    def __init__(self, buffer_size, batch_size, seed, device, alpha=0.6, beta=0.4, beta_increment=0.001):
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
        
    def add(self, max_priority, state, action, reward, next_state, done):

        self.max_priority = max_priority

        state = state.clone().detach().float()
        action = torch.tensor(action).long()
        reward = torch.tensor(reward).float()
        next_state = next_state.clone().detach().float()
        done = torch.tensor(done).float()


        experience = self.experience(state, action, reward, next_state, done)
        priority = max_priority  # New experiences get max priority
        self.tree.add(priority ** self.alpha, experience)
    
    def sample(self):
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
        
        # Convert to tensors
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
        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights = is_weights / is_weights.max()  # Normalize
        is_weights = torch.from_numpy(is_weights).float().to(self.device)


        
        ##return (states, actions, rewards, next_states, dones, indices, is_weights)
        return(experiences, indices, is_weights)
    
    def update_priorities(self, indices, priorities):
        """Update priorities based on TD errors"""
        for idx, priority in zip(indices, priorities):
            priority = abs(priority) + self.epsilon  # Ensure non-zero priority
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority ** self.alpha)
    
    def __len__(self):
        return self.tree.n_entries

class RewardShapingPERBuffer:
    """Enhanced replay buffer that focuses on high-reward experiences"""
    def __init__(self, buffer_size, batch_size, seed, device, reward_threshold=100):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.seed = random.seed(seed)
        self.reward_threshold = reward_threshold
        
        # Separate buffers for different quality experiences
        self.high_reward_buffer = []  # Experiences with high rewards
        self.normal_buffer = []       # Regular experiences
        
        self.experience = namedtuple("Experience", 
                                   field_names=["state", "action", "reward", "next_state", "done", "episode_reward"])
        
    def add(self, state, action, reward, next_state, done, episode_reward=0):
        experience = self.experience(state, action, reward, next_state, done, episode_reward)
        
        # Prioritize experiences from high-performing episodes
        if episode_reward > self.reward_threshold or reward > 10:  # Positive reward or good episode
            self.high_reward_buffer.append(experience)
            if len(self.high_reward_buffer) > self.buffer_size // 4:  # Keep 25% for high-reward
                self.high_reward_buffer.pop(0)
        else:
            self.normal_buffer.append(experience)
            if len(self.normal_buffer) > (3 * self.buffer_size // 4):  # 75% for normal
                self.normal_buffer.pop(0)
    
    def sample(self):
        # Sample 60% from high-reward buffer, 40% from normal buffer
        high_reward_samples = min(int(self.batch_size * 0.6), len(self.high_reward_buffer))
        normal_samples = self.batch_size - high_reward_samples
        
        experiences = []
        
        # Sample from high-reward buffer
        if high_reward_samples > 0 and len(self.high_reward_buffer) > 0:
            experiences.extend(random.sample(self.high_reward_buffer, high_reward_samples))
        
        # Sample from normal buffer
        if normal_samples > 0 and len(self.normal_buffer) > 0:
            experiences.extend(random.sample(self.normal_buffer, 
                                           min(normal_samples, len(self.normal_buffer))))
        
        # If we don't have enough experiences, fill from whatever we have
        while len(experiences) < self.batch_size:
            if len(self.high_reward_buffer) > 0:
                experiences.append(random.choice(self.high_reward_buffer))
            elif len(self.normal_buffer) > 0:
                experiences.append(random.choice(self.normal_buffer))
            else:
                break
        
        if len(experiences) == 0:
            return None
            
        # Convert to tensors
        states = torch.from_numpy(np.array([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.array([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.array([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.high_reward_buffer) + len(self.normal_buffer)