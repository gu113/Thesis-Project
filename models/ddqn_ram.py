import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Double DQN CNN Model for RAM version of Atari games
class RamDDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(RamDDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        input_size = input_shape[0]

        # Shared feature extractor
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)

        # Dueling DQN streams
        self.advantage = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        advantage = self.advantage(x)
        value = self.value(x)

        # Combine value and advantage streams
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# Async Double DQN CNN Model for RAM version of Atari games
class AsyncRamDDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(AsyncRamDDQN, self).__init__()
        input_size = input_shape[0]

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)

        # Dueling architecture: Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        # Dueling architecture: Value stream
        self.value = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        adv = self.advantage(x)
        val = self.value(x)

        # Combine value and advantage streams
        return val + adv - adv.mean(dim=1, keepdim=True)
