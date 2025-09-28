import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class ActorCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
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
        dist = Categorical(x)
        return dist
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

class CriticCnn(nn.Module):
    def __init__(self, input_shape):
        super(CriticCnn, self).__init__()
        self.input_shape = input_shape
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
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
    

class PPOActorCnn(nn.Module):
    def __init__(self, input_shape, action_size):
        super(PPOActorCnn, self).__init__()
        self.input_shape = input_shape
        self.action_size = action_size

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

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

class PPOCriticCnn(nn.Module):
    def __init__(self, input_shape):
        super(PPOCriticCnn, self).__init__()
        self.input_shape = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

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
    

class REINFORCEActorCnn(nn.Module):

    def __init__(self, input_shape, num_actions):
        super(REINFORCEActorCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self._conv_out_size = self._get_conv_out(input_shape)

        self.actor_head = nn.Sequential(
            nn.Linear(self._conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

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
    

class A2CActorCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(A2CActorCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
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

class A2CCriticCnn(nn.Module):
    def __init__(self, input_shape):
        super(A2CCriticCnn, self).__init__()
        self.input_shape = input_shape
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
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
    

class A3CActorCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(A3CActorCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
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

class A3CCriticCnn(nn.Module):
    def __init__(self, input_shape):
        super(A3CCriticCnn, self).__init__()
        self.input_shape = input_shape
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
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


class TRPOActorCnn(nn.Module):
    def __init__(self, input_shape, action_size):
        super(TRPOActorCnn, self).__init__()
        self.input_shape = input_shape
        self.action_size = action_size

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

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

class TRPOCriticCnn(nn.Module):
    def __init__(self, input_shape):
        super(TRPOCriticCnn, self).__init__()
        self.input_shape = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

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
    

class SoftActorCnn(nn.Module):
    """
    The Actor (policy network) for the Soft Actor-Critic algorithm.
    It takes an observation and outputs the logits for a categorical distribution
    over the discrete action space.
    """
    def __init__(self, input_shape, num_actions):
        super(SoftActorCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
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


class SoftCriticCnn(nn.Module):
    """
    The Critic (Q-network) for the Soft Actor-Critic algorithm.
    It takes an observation and outputs the Q-values for each possible action.
    SAC uses two of these networks to prevent Q-value overestimation.
    """
    def __init__(self, input_shape, num_actions):
        super(SoftCriticCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
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