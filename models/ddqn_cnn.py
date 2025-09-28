import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F
import numpy as np


class DDQNCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DDQNCnn, self).__init__()
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
        
        self.advantage = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value = self.value(x)
        ###return value + advantage  - advantage.mean()
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
  

class ComplexDDQNCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ComplexDDQNCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Improved feature extraction with batch normalization and dropout
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Additional convolutional layer for better feature extraction
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Calculate the flattened feature size
        self.feature_dim = self._get_feature_dim()
        
        # Advantage stream with improved architecture
        self.advantage = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_actions)
        )
        
        # Value stream with improved architecture
        self.value = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _get_feature_dim(self):
        """Calculate the dimension of flattened features"""
        with torch.no_grad():
            sample_input = torch.zeros(1, *self.input_shape)
            sample_output = self.features(sample_input)
            return int(np.prod(sample_output.size()[1:]))
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Ensure input is in the correct range [0, 1]
        if x.max() > 1.0:
            x = x / 255.0
            
        # Extract features
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        # Calculate advantage and value
        advantage = self.advantage(features)
        value = self.value(features)
        
        # Combine using dueling architecture
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class SimpleDDQNCnn(nn.Module):
    """A simpler but more stable version for comparison"""
    def __init__(self, input_shape, num_actions):
        super(SimpleDDQNCnn, self).__init__()
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
        
        # Calculate feature size
        self.feature_size = self._calculate_feature_size()
        
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def _calculate_feature_size(self):
        with torch.no_grad():
            sample = torch.zeros(1, *self.input_shape)
            features = self.features(sample)
            return int(np.prod(features.size()[1:]))
    
    def forward(self, x):
        if x.max() > 1.0:
            x = x / 255.0
            
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        advantage = self.advantage(features)
        value = self.value(features)
        
        # Dueling DQN combination
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    

class DuelingDQNCnn(nn.Module):
    """
    A Dueling Deep Q-Learning network architecture.
    """
    def __init__(self, input_shape, num_actions):
        super(DuelingDQNCnn, self).__init__()
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
        
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(self._get_feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(self._get_feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def _get_feature_size(self):
        """
        Calculates the size of the flattened feature vector.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)
            return self.features(dummy_input).view(1, -1).size(1)
        
    def forward(self, x):
        # Pass input through shared feature layers
        features = self.features(x)
        # Flatten the features
        features = features.view(features.size(0), -1)
        
        # Get advantage and value from their respective streams
        advantage = self.advantage(features)
        value = self.value(features)
        
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
    

class ComplexDuelingDQNCnn(nn.Module):
    """
    A more complex Dueling Deep Q-Learning network architecture
    with Batch Normalization, Dropout, and Kaiming initialization.
    """
    def __init__(self, input_shape, num_actions):
        super(ComplexDuelingDQNCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Improved feature extraction with batch normalization and dropout
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Calculate the flattened feature size
        self.feature_dim = self._get_feature_dim()
        
        # Advantage stream with improved architecture
        self.advantage = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_actions)
        )
        
        # Value stream with improved architecture
        self.value = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _get_feature_dim(self):
        """Calculate the dimension of flattened features"""
        with torch.no_grad():
            sample_input = torch.zeros(1, *self.input_shape)
            sample_output = self.features(sample_input)
            return int(np.prod(sample_output.size()[1:]))
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Normalize input if needed
        if x.max() > 1.0:
            x = x / 255.0
            
        # Extract features
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        # Calculate advantage and value
        advantage = self.advantage(features)
        value = self.value(features)
        
        # Combine using dueling architecture
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    

class AsyncDDQNCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(AsyncDDQNCnn, self).__init__()
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
        
        self.advantage = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        # Check if x has an extra dimension (5D) and reshape it to 4D
        if len(x.shape) == 5:  # if input has shape [batch_size, 2, 4, height, width]
            x = x.view(-1, x.size(2), x.size(3), x.size(4))  # Reshape to [batch_size * 2, height, width]

        # Pass through the feature extraction layers
        x = self.features(x)
        
        # Flatten the output from the convolutional layers to a 2D tensor
        x = x.view(x.size(0), -1)
        
        # Calculate advantage and value streams
        advantage = self.advantage(x)
        value = self.value(x)
        
        # Return the combined value and advantage (Double DQN)
        return value + advantage - advantage.mean()

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)