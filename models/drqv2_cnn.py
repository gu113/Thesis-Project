import torch
import torch.nn as nn
import kornia.augmentation as T
import numpy as np

# DrQv2 CNN Model (Based on the Original DrQv2 Architecture)
class DrQv2CNN(nn.Module):
    def __init__(self, input_shape, num_actions, feature_dim=50):
        super(DrQv2CNN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        
        # Define the CNN layers
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        
        self.encoder_out_size = self._get_conv_out(input_shape)
        
        # Define the fully connected layers
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder_out_size, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )
        
        # Define the output layers for Q-values
        self.Q1 = nn.Linear(feature_dim, num_actions)
        self.Q2 = nn.Linear(feature_dim, num_actions)
        
        self.aug = nn.Sequential(
            # T.RandomRGBShift(4, p=1.0), # Only for colored environments
            T.RandomBrightness(brightness=(0.8, 1.2), p=1.0),
            T.RandomGaussianNoise(0.0, 0.05, p=1.0) 
        )
        
    def _get_conv_out(self, shape):
        o = self.encoder(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x, augment=False):
        if augment and self.training:
            x = self.aug(x) 
        
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        
        h = self.trunk(h)
        
        q1 = self.Q1(h)
        q2 = self.Q2(h)
        
        return q1, q2