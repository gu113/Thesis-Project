import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


# Actor-Critic CNN Models (Actor)
class ActorCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Define the shared convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        dist = Categorical(x) # Create a categorical distribution over actions
        return dist
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

# Actor-Critic CNN Models (Critic)
class CriticCnn(nn.Module):
    def __init__(self, input_shape):
        super(CriticCnn, self).__init__()
        self.input_shape = input_shape
        
        # Define the shared convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    

# PPO Actor-Critic CNN Models (Actor)
class PPOActorCnn(nn.Module):
    def __init__(self, input_shape, action_size):
        super(PPOActorCnn, self).__init__()
        self.input_shape = input_shape
        self.action_size = action_size

        # Define the shared convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Define the fully connected layers
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        # Extract features
        conv_out = self.conv(x).view(x.size(0), -1)
        logits = self.fc(conv_out)
        return Categorical(logits=logits)

# PPO Actor-Critic CNN Models (Critic)
class PPOCriticCnn(nn.Module):
    def __init__(self, input_shape):
        super(PPOCriticCnn, self).__init__()
        self.input_shape = input_shape

        # Define the shared convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Define the fully connected layers
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        # Extract features
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)
    

# REINFORCE Actor-Critic CNN Model
class REINFORCEActorCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(REINFORCEActorCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        # Define the shared convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self._conv_out_size = self._get_conv_out(input_shape)

        # Define the actor head
        self.actor_head = nn.Sequential(
            nn.Linear(self._conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        # Define the critic head
        self.critic_head = nn.Sequential(
            nn.Linear(self._conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size(0), -1)
        action_logits = self.actor_head(conv_out)
        state_value = self.critic_head(conv_out)
        return action_logits, state_value
    

# A2C Actor-Critic CNN Models (Actor)
class A2CActorCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(A2CActorCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Define the shared convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
        )
        
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits
    
    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

# A2C Actor-Critic CNN Models (Critic)
class A2CCriticCnn(nn.Module):
    def __init__(self, input_shape):
        super(A2CCriticCnn, self).__init__()
        self.input_shape = input_shape
        
        # Define the shared convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1) # Output a single value for state-value
        )
        
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits
    
    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
    

# A3C Actor-Critic CNN Models (Actor)
class A3CActorCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(A3CActorCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Define the shared convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
        )
        
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        x = x / 255.0
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits
    
    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

# A3C Actor-Critic CNN Models (Critic)
class A3CCriticCnn(nn.Module):
    def __init__(self, input_shape):
        super(A3CCriticCnn, self).__init__()
        self.input_shape = input_shape
        
        # Define the shared convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        x = x / 255.0
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits
    
    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)


# TRPO Actor-Critic CNN Models (Actor)
class TRPOActorCnn(nn.Module):
    def __init__(self, input_shape, action_size):
        super(TRPOActorCnn, self).__init__()
        self.input_shape = input_shape
        self.action_size = action_size

        # Define the shared convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Define the fully connected layers
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        conv_out = self.conv(x).view(x.size(0), -1)
        logits = self.fc(conv_out)
        return Categorical(logits=logits)

# TRPO Actor-Critic CNN Models (Critic)
class TRPOCriticCnn(nn.Module):
    def __init__(self, input_shape):
        super(TRPOCriticCnn, self).__init__()
        self.input_shape = input_shape

        # Define the shared convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Define the fully connected layers
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)
    

# Soft Actor-Critic CNN Models (Actor)
class SoftActorCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(SoftActorCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Define the shared convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self._get_feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

    def _get_feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

# Soft Actor-Critic CNN Models (Critic)
class SoftCriticCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(SoftCriticCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Define the shared convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self._get_feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        q_values = self.fc(x)
        return q_values

    def _get_feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)