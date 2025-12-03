import numpy as np
import gc
import weakref
import torch

class SumTree:
    """Sum Tree data structure"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        ###self.data = np.zeros(capacity, dtype=object)
        self.data = [None] * capacity 
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])
    
    def max(self):
        if self.n_entries == 0:
            return 0
        leaf_start = self.capacity - 1
        return np.max(self.tree[leaf_start : leaf_start + self.n_entries])



class SumTreewithTensorManagement:
    """Sum Tree data structure with tensor management to prevent memory leaks"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.states = [None] * capacity
        self.actions = [None] * capacity  
        self.rewards = [None] * capacity
        self.next_states = [None] * capacity
        self.dones = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, experience):
        idx = self.write + self.capacity - 1
        
        # Clear old tensors
        self._clear_position(self.write)
        
        # Store experience components
        self.states[self.write] = experience.state
        self.actions[self.write] = experience.action
        self.rewards[self.write] = experience.reward
        self.next_states[self.write] = experience.next_state
        self.dones[self.write] = experience.done
        
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def _clear_position(self, pos):
        if self.states[pos] is not None:
            del self.states[pos]
        if self.actions[pos] is not None:
            del self.actions[pos]
        if self.rewards[pos] is not None:
            del self.rewards[pos]
        if self.next_states[pos] is not None:
            del self.next_states[pos]
        if self.dones[pos] is not None:
            del self.dones[pos]
            
        self.states[pos] = None
        self.actions[pos] = None
        self.rewards[pos] = None
        self.next_states[pos] = None
        self.dones[pos] = None

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        
        # Reconstruct experience from components
        from collections import namedtuple
        Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])
        
        experience = Experience(
            self.states[dataIdx],
            self.actions[dataIdx], 
            self.rewards[dataIdx],
            self.next_states[dataIdx],
            self.dones[dataIdx]
        )
        
        return (idx, self.tree[idx], experience)
    
    def max(self):
        if self.n_entries == 0:
            return 0
        leaf_start = self.capacity - 1
        return np.max(self.tree[leaf_start : leaf_start + self.n_entries])
    
    def clear(self):
        for i in range(self.capacity):
            self._clear_position(i)
        self.tree.fill(0)
        self.write = 0
        self.n_entries = 0
        gc.collect() # Collect garbage to free up memory


class RainbowSumTree:
    """Sum Tree data structure for the Rainbow algorithm"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_ptr = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        # Iterative propagation
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _retrieve(self, idx, s):
        # Iterative retrieval
        while idx < self.capacity - 1:
            left_child = 2 * idx + 1

            if s <= self.tree[left_child]:
                idx = left_child
            else:
                s -= self.tree[left_child]
                idx = left_child + 1
        return idx

    def total_priority(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.data_ptr + self.capacity - 1
        self.data[self.data_ptr] = data
        self.update(idx, priority)
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority_val): 
        change = priority_val - self.tree[idx]
        self.tree[idx] = priority_val
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - (self.capacity - 1)
        return (idx, self.tree[idx], self.data[data_idx])
    

class RainbowSumTreeGPU:
    """Sum Tree data structure for the Rainbow algorithm on GPU"""
    def __init__(self, capacity, device):
        self.device = device
        self.capacity = capacity
        self.tree = torch.zeros(2 * capacity - 1, dtype=torch.float32, device=device)
        self.data_store = [None] * capacity  # Store Python objects on CPU
        self.data_ptr = 0
        self.n_entries = 0
        
    def _propagate(self, idx, change):
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _retrieve(self, idx, s):
        while idx < self.capacity - 1:
            left_child = 2 * idx + 1
            if s <= self.tree[left_child]:
                idx = left_child
            else:
                s -= self.tree[left_child]
                idx = left_child + 1
        return idx

    def total_priority(self):
        return self.tree[0].item()

    def add(self, priority, data):
        idx = self.data_ptr + self.capacity - 1
        self.data_store[self.data_ptr] = data
        self.update(idx, priority)
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority_val):
        change = priority_val - self.tree[idx].item()
        self.tree[idx] = priority_val
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx].item(), self.data_store[data_idx]