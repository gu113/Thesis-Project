import numpy as np
import random
from collections import namedtuple, deque
import torch

class ReplayBufferPER:
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
