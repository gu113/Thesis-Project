import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma_init / math.sqrt(fan_in))
        nn.init.constant_(self.bias_sigma, self.sigma_init / math.sqrt(fan_in))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


class CategoricalQNetwork(nn.Module):
    # Dueling Q-Network with Categorical output (C51)
    def __init__(self, input_shape, n_actions, num_atoms, V_min, V_max):
        super(CategoricalQNetwork, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.num_atoms = num_atoms
        self.V_min = V_min
        self.V_max = V_max
        # Supports should be on the same device as the network when created
        # No need to move it to device in get_q_values if it's already here.
        self.register_buffer('supports', torch.linspace(V_min, V_max, num_atoms)) 

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)

        # NoisyLinear layers for Dueling branches
        self.fc_adv = NoisyLinear(conv_out_size, 512)
        self.adv_atom = NoisyLinear(512, n_actions * num_atoms)

        self.fc_val = NoisyLinear(conv_out_size, 512)
        self.val_atom = NoisyLinear(512, num_atoms)

    def _get_conv_out(self, shape):
        # Create a dummy tensor of float32 for shape inference, as inputs are now float32
        o = self.conv(torch.zeros(1, *shape, dtype=torch.float32)) 
        return int(torch.prod(torch.tensor(o.size())))

    def forward(self, x):
        # x is assumed to be float32 and normalized (0-1) from the main loop now
        # No need for x = x.float() here if inputs are already float32.
        # If using autocast, inputs will be automatically handled.
        
        conv_out = self.conv(x).view(x.size(0), -1)

        val_out = F.relu(self.fc_val(conv_out))
        val_atom = self.val_atom(val_out)

        adv_features = F.relu(self.fc_adv(conv_out))
        adv_out = self.adv_atom(adv_features) 
        adv_atom = adv_out.view(-1, self.n_actions, self.num_atoms)
        
        q_atoms = val_atom.unsqueeze(1) + (adv_atom - adv_atom.mean(dim=1, keepdim=True))
        q_probs = F.softmax(q_atoms, dim=-1).clamp(min=1e-5) # Clamp for stability
        
        return q_probs

    def reset_noise(self):
        for name, module in self.named_modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def get_q_values(self, q_probs):
        # Calculate expected Q-values from distributions for action selection
        # self.supports is now a buffer, so it's always on the correct device.
        return torch.sum(q_probs * self.supports, dim=-1)