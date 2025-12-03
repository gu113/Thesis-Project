import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

import sys
sys.path.append('./')

from models.rdqn_cnn import CategoricalQNetwork
from utils.replay_buffer import RainbowPERBuffer, RainbowPERBufferGPU

class RainbowDQNAgent:
    def __init__(self, input_shape, n_actions, device, buffer_capacity=1_000_000, batch_size=32, gamma=0.99, lr=1e-4, target_update_freq=10000, n_step=3, alpha=0.6, beta_start=0.4, beta_frames=100000, num_atoms=51, V_min=-10, V_max=10, learning_starts=20000):
        """Initialize a simpler version of the Rainbow DQN Agent (Custom Implementation)
        Params
        ======
            input_shape (tuple): Shape of the input state (C, H, W)
            n_actions (int): Number of possible actions
            device (torch.device): Device to run the model on (CPU or GPU)
            buffer_capacity (int): Capacity of the replay buffer
            batch_size (int): Mini-batch size for training
            gamma (float): Discount factor
            lr (float): Learning rate for the optimizer
            target_update_freq (int): Frequency of target network updates (in steps)
            n_step (int): Number of steps for N-step returns
            alpha (float): PER alpha parameter
            beta_start (float): Initial PER beta parameter
            beta_frames (int): Number of frames over which beta is annealed
            num_atoms (int): Number of atoms for C51
            V_min (float): Minimum value for C51
            V_max (float): Maximum value for C51
            learning_starts (int): Number of steps before learning starts
            """
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_step = n_step
        self.learning_starts = learning_starts
        self.num_atoms = num_atoms
        self.V_min = V_min
        self.V_max = V_max
        self.delta_z = (V_max - V_min) / (num_atoms - 1)
        self.supports = torch.linspace(V_min, V_max, num_atoms).to(device)

        # Q-Networks: CategoricalQNetwork includes Dueling and NoisyNets
        self.q_net = CategoricalQNetwork(input_shape, n_actions, num_atoms, V_min, V_max).to(device)
        self.target_q_net = CategoricalQNetwork(input_shape, n_actions, num_atoms, V_min, V_max).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay Buffer
        self.replay_buffer = RainbowPERBuffer(
            buffer_capacity, alpha=alpha, beta_start=beta_start, beta_frames=beta_frames
        )
        
        self.steps_done = 0
        self.n_step_buffer = []

    def choose_action(self, state):
        # Choose action based on current policy
        with torch.no_grad():
            state = state.unsqueeze(0).float().to(self.device)
            q_probs = self.q_net(state)
            q_values = self.q_net.get_q_values(q_probs)
            action = q_values.argmax(dim=1).item()
        return action

    def step(self, state, action, reward, next_state, done):
        # Store transition in N-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) == self.n_step or done:
            cumulative_reward = 0
            n_step_done = False
            for i in range(len(self.n_step_buffer)):
                r = self.n_step_buffer[i][2]
                d = self.n_step_buffer[i][4]
                cumulative_reward += r * (self.gamma ** i)
                if d:
                    n_step_done = True
                    break
            
            # Store N-step transition in replay buffer
            initial_state, initial_action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
            final_next_state, final_done = self.n_step_buffer[i][3], n_step_done

            # Convert to PyTorch tensors for the buffer
            initial_state_tensor = torch.from_numpy(initial_state)
            final_next_state_tensor = torch.from_numpy(final_next_state)
            
            # Convert other components to tensors
            initial_action_tensor = torch.tensor(initial_action, dtype=torch.long)
            cumulative_reward_tensor = torch.tensor(cumulative_reward, dtype=torch.float32)
            final_done_tensor = torch.tensor(final_done, dtype=torch.bool)

            # Add to replay buffer
            self.replay_buffer.add(
                initial_state_tensor, 
                initial_action_tensor, 
                cumulative_reward_tensor, 
                final_next_state_tensor, 
                final_done_tensor
            )
            self.n_step_buffer = []

        self.steps_done += 1
        
        if self.steps_done % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            self.q_net.reset_noise()
            self.target_q_net.reset_noise()

    def learn(self):
        if self.steps_done < self.learning_starts:
            return None

        if len(self.replay_buffer) < self.batch_size:
            return None

        indices, states, actions, rewards, next_states, dones, weights = \
            self.replay_buffer.sample(self.batch_size)

        states = states.float()
        next_states = next_states.float()

        # If using GPU, pin memory for faster transfer
        if self.device.type == 'cuda':
            states = states.pin_memory()
            actions = actions.pin_memory()
            rewards = rewards.pin_memory()
            next_states = next_states.pin_memory()
            dones = dones.pin_memory()
            weights = weights.pin_memory()
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        with torch.no_grad():
            # Target Q-value calculation (Double DQN + C51 Projection)
            next_q_values_main_net = self.q_net.get_q_values(self.q_net(next_states))
            next_actions = next_q_values_main_net.argmax(dim=1).unsqueeze(1)

            # Target network evaluation
            next_q_target_output = self.target_q_net(next_states)
            next_actions_expanded = next_actions.unsqueeze(2).repeat(1, 1, self.num_atoms) 

            # Get the probabilities of the selected actions
            next_q_probs_target = next_q_target_output.gather(
                1, next_actions_expanded
            ).squeeze(1)
            
            # Compute the projected distribution
            Tz = rewards.unsqueeze(1) + (self.gamma ** self.n_step) * self.supports.unsqueeze(0) * (~dones).unsqueeze(1)
            Tz = Tz.clamp(self.V_min, self.V_max)
            
            b_j = (Tz - self.V_min) / self.delta_z
            l_j = b_j.floor().long()
            u_j = b_j.ceil().long()

            l_j = l_j.clamp(0, self.num_atoms - 1)
            u_j = u_j.clamp(0, self.num_atoms - 1)
            
            # Construct the projected distribution
            m = torch.zeros(self.batch_size, self.num_atoms, device=self.device)

            # Distribute probability mass to neighboring atoms
            m.scatter_add_(1, l_j, next_q_probs_target * (u_j.float() - b_j))
            m.scatter_add_(1, u_j, next_q_probs_target * (b_j - l_j.float()))

            done_mask = dones.bool()
            if done_mask.any():
                # Handle terminal states
                reward_vals_done = rewards[done_mask]
                reward_proj_idx_done = (reward_vals_done - self.V_min) / self.delta_z
                
                # Project rewards for terminal states
                l_done = reward_proj_idx_done.floor().long()
                u_done = reward_proj_idx_done.ceil().long()

                l_done = l_done.clamp(0, self.num_atoms - 1)
                u_done = u_done.clamp(0, self.num_atoms - 1)

                m_done_temp = torch.zeros(len(reward_vals_done), self.num_atoms, device=self.device)
                
                exact_match_mask = (l_done == u_done)
                if exact_match_mask.any():
                    m_done_temp.scatter_(1, l_done[exact_match_mask].unsqueeze(1), 1.0)

                non_exact_match_mask = ~exact_match_mask
                if non_exact_match_mask.any():
                    l_ne = l_done[non_exact_match_mask]
                    u_ne = u_done[non_exact_match_mask]
                    b_ne = reward_proj_idx_done[non_exact_match_mask]

                    m_done_temp[non_exact_match_mask].scatter_add_(1, l_ne.unsqueeze(1), (u_ne.float() - b_ne).unsqueeze(1))
                    m_done_temp[non_exact_match_mask].scatter_add_(1, u_ne.unsqueeze(1), (b_ne - l_ne.float()).unsqueeze(1))
                
                m[done_mask] = m_done_temp

        m = m.clamp(min=1e-5)

        # Calculate loss
        current_q_net_output = self.q_net(states)
        actions_expanded = actions.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.num_atoms)

        current_q_probs = current_q_net_output.gather(
            1, actions_expanded
        ).squeeze(1)

        loss = -torch.sum(m * torch.log(current_q_probs), dim=1)
        loss = (loss * weights).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            # Calculate TD-errors for priority updates using KL-divergence
            priorities_q_net_output = self.q_net(states)
            actions_expanded_for_priorities = actions.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.num_atoms)
            
            current_q_probs_for_priorities = priorities_q_net_output.gather(
                1, actions_expanded_for_priorities
            ).squeeze(1)
            
            td_errors_for_priorities = F.kl_div(
                torch.log(current_q_probs_for_priorities.clamp(min=1e-5)),
                m.clamp(min=1e-5),
                reduction='none'
            ).sum(dim=1).cpu().numpy()

        self.replay_buffer.update_priorities(indices, td_errors_for_priorities)
        
        return loss.item()
    

class ComplexRainbowDQNAgent:
    def __init__(self, input_shape, n_actions, device, buffer_capacity=1_000_000, batch_size=32, gamma=0.99, lr=1e-4, target_update_freq=10000, n_step=3, alpha=0.6, beta_start=0.4, beta_frames=100000, num_atoms=51, V_min=-10, V_max=10, learning_starts=20000):
        """Initialize a more complex version of the Rainbow DQN Agent (Implementation Based on Official Paper)
        Params
        ======
            input_shape (tuple): Shape of the input state (C, H, W)
            n_actions (int): Number of possible actions
            device (torch.device): Device to run the model on (CPU or GPU)
            buffer_capacity (int): Capacity of the replay buffer
            batch_size (int): Mini-batch size for training
            gamma (float): Discount factor
            lr (float): Learning rate for the optimizer
            target_update_freq (int): Frequency of target network updates (in steps)
            n_step (int): Number of steps for N-step returns
            alpha (float): PER alpha parameter
            beta_start (float): Initial PER beta parameter
            beta_frames (int): Number of frames over which beta is annealed
            num_atoms (int): Number of atoms for C51
            V_min (float): Minimum value for C51
            V_max (float): Maximum value for C51
            learning_starts (int): Number of steps before learning starts
            """
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_step = n_step
        self.learning_starts = learning_starts

        self.num_atoms = num_atoms
        self.V_min = V_min
        self.V_max = V_max
        self.delta_z = (V_max - V_min) / (num_atoms - 1)
        self.supports = torch.linspace(V_min, V_max, num_atoms).to(device)

        # Q-Networks: CategoricalQNetwork includes Dueling and NoisyNets
        self.q_net = CategoricalQNetwork(input_shape, n_actions, num_atoms, V_min, V_max).to(device)
        self.target_q_net = CategoricalQNetwork(input_shape, n_actions, num_atoms, V_min, V_max).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay Buffer
        self.replay_buffer = RainbowPERBuffer(
            buffer_capacity, alpha=alpha, beta_start=beta_start, beta_frames=beta_frames
        )
        
        self.steps_done = 0
        self.n_step_buffer = []

        if self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda') 
        else:
            self.scaler = None

    def choose_action(self, state):
        with torch.no_grad():
            state = state.unsqueeze(0)
            q_probs = self.q_net(state)
            q_values = self.q_net.get_q_values(q_probs)
            action = q_values.argmax(dim=1).item()
        return action

    def step(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) == self.n_step or done:
            cumulative_reward = 0
            n_step_done = False
            for i in range(len(self.n_step_buffer)):
                r = self.n_step_buffer[i][2]
                d = self.n_step_buffer[i][4]
                cumulative_reward += r * (self.gamma ** i)
                if d:
                    n_step_done = True
                    break
            
            initial_state, initial_action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
            final_next_state, final_done = self.n_step_buffer[i][3], n_step_done

            # Convert to PyTorch tensors for the buffer
            initial_state_tensor = torch.from_numpy(initial_state) 
            final_next_state_tensor = torch.from_numpy(final_next_state)
            
            initial_action_tensor = torch.tensor(initial_action, dtype=torch.long)
            cumulative_reward_tensor = torch.tensor(cumulative_reward, dtype=torch.float32)
            final_done_tensor = torch.tensor(final_done, dtype=torch.bool)

            # Add to replay buffer
            self.replay_buffer.add(
                initial_state_tensor, 
                initial_action_tensor, 
                cumulative_reward_tensor, 
                final_next_state_tensor, 
                final_done_tensor
            )
            self.n_step_buffer = []

        self.steps_done += 1
        
        # Update target network periodically
        if self.steps_done % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            self.target_q_net.eval()
            self.q_net.reset_noise()
            self.target_q_net.reset_noise()

    def learn(self):
        if self.steps_done < self.learning_starts:
            return None

        if len(self.replay_buffer) < self.batch_size:
            return None

        # Update beta for PER
        self.replay_buffer.beta = min(1.0, self.replay_buffer.beta_start +  self.steps_done * (1.0 - self.replay_buffer.beta_start) / self.replay_buffer.beta_frames)

        indices, states, actions, rewards, next_states, dones, weights = \
            self.replay_buffer.sample(self.batch_size)

        # If using GPU, pin memory for faster transfer
        if self.device.type == 'cuda':
            states = states.pin_memory().to(self.device, non_blocking=True)
            actions = actions.pin_memory().to(self.device, non_blocking=True)
            rewards = rewards.pin_memory().to(self.device, non_blocking=True)
            next_states = next_states.pin_memory().to(self.device, non_blocking=True)
            dones = dones.pin_memory().to(self.device, non_blocking=True)
            weights = weights.pin_memory().to(self.device, non_blocking=True)
        else:
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            weights = weights.to(self.device)
        
        with torch.amp.autocast(device_type="cuda") if self.scaler else torch.no_grad():

            # Target Q-value calculation (Double DQN + C51 Projection)
            next_q_values_main_net = self.q_net.get_q_values(self.q_net(next_states))
            next_actions = next_q_values_main_net.argmax(dim=1).unsqueeze(1)

            # Target network evaluation
            next_q_target_output = self.target_q_net(next_states)
            next_actions_expanded = next_actions.unsqueeze(2).repeat(1, 1, self.num_atoms) 

            next_q_probs_target = next_q_target_output.gather(
                1, next_actions_expanded
            ).squeeze(1)
            
            # Compute the projected distribution
            Tz = rewards.unsqueeze(1) + (self.gamma ** self.n_step) * self.supports.unsqueeze(0) * (~dones).unsqueeze(1)
            Tz = Tz.clamp(self.V_min, self.V_max)
            
            b_j = (Tz - self.V_min) / self.delta_z
            l_j = b_j.floor().long()
            u_j = b_j.ceil().long()

            l_j = l_j.clamp(0, self.num_atoms - 1)
            u_j = u_j.clamp(0, self.num_atoms - 1)
            
            m = torch.zeros(self.batch_size, self.num_atoms, device=self.device)

            m.scatter_add_(1, l_j, next_q_probs_target * (u_j.float() - b_j))
            m.scatter_add_(1, u_j, next_q_probs_target * (b_j - l_j.float()))

            # Handle terminal states
            done_mask = dones.bool()
            if done_mask.any():
                # Project rewards for terminal states
                reward_vals_done = rewards[done_mask]
                reward_proj_idx_done = (reward_vals_done - self.V_min) / self.delta_z 
                
                l_done = reward_proj_idx_done.floor().long()
                u_done = reward_proj_idx_done.ceil().long()

                l_done = l_done.clamp(0, self.num_atoms - 1)
                u_done = u_done.clamp(0, self.num_atoms - 1)

                m_done_temp = torch.zeros(len(reward_vals_done), self.num_atoms, device=self.device)
                
                exact_match_mask = (l_done == u_done)
                if exact_match_mask.any():
                    m_done_temp[exact_match_mask].scatter_(1, l_done[exact_match_mask].unsqueeze(1), 1.0)

                non_exact_match_mask = ~exact_match_mask
                if non_exact_match_mask.any():
                    l_ne = l_done[non_exact_match_mask]
                    u_ne = u_done[non_exact_match_mask]
                    b_ne = reward_proj_idx_done[non_exact_match_mask]

                    m_done_temp[non_exact_match_mask].scatter_add_(1, l_ne.unsqueeze(1), (u_ne.float() - b_ne).unsqueeze(1))
                    m_done_temp[non_exact_match_mask].scatter_add_(1, u_ne.unsqueeze(1), (b_ne - l_ne.float()).unsqueeze(1))
                
                m[done_mask] = m_done_temp

            m = m.clamp(min=1e-5)

        # Calculate loss
        current_q_net_output = self.q_net(states)
        actions_expanded = actions.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.num_atoms)

        # Get the probabilities of the taken actions
        current_q_probs = current_q_net_output.gather(1, actions_expanded).squeeze(1)

        # Calculate loss
        loss = -torch.sum(m * torch.log(current_q_probs), dim=1) # KL-Divergence
        loss = (loss * weights).mean() # Apply PER importance sampling weights
        
        # Optimize the model
        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        # Update priorities after optimization
        with torch.no_grad():
            priorities_q_net_output = self.q_net(states)
            actions_expanded_for_priorities = actions.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.num_atoms)
            
            current_q_probs_for_priorities = priorities_q_net_output.gather(1, actions_expanded_for_priorities).squeeze(1)
            
            # Use KL-divergence as TD-error for C51
            td_errors_for_priorities = F.kl_div(
                torch.log(current_q_probs_for_priorities.clamp(min=1e-5)),
                m.clamp(min=1e-5),
                reduction='none'
            ).sum(dim=1).cpu().numpy()

        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_errors_for_priorities)
        
        return loss.item()


class ComplexRainbowDQNAgentGPU:
    def __init__(self, input_shape, n_actions, device, buffer_capacity=1_000_000, batch_size=32, gamma=0.99, lr=1e-4, target_update_freq=10000, n_step=3, alpha=0.6, beta_start=0.4, beta_frames=100000, num_atoms=51, V_min=-10, V_max=10, learning_starts=20000):
        """Initialize a more complex version of the Rainbow DQN Agent with GPU-optimized Replay Buffer (Implementation Based on Unofficial Papers)
        Params
        ======
            input_shape (tuple): Shape of the input state (C, H, W)
            n_actions (int): Number of possible actions
            device (torch.device): Device to run the model on (CPU or GPU)
            buffer_capacity (int): Capacity of the replay buffer
            batch_size (int): Mini-batch size for training
            gamma (float): Discount factor
            lr (float): Learning rate for the optimizer
            target_update_freq (int): Frequency of target network updates (in steps)
            n_step (int): Number of steps for N-step returns
            alpha (float): PER alpha parameter
            beta_start (float): Initial PER beta parameter
            beta_frames (int): Number of frames over which beta is annealed
            num_atoms (int): Number of atoms for C51
            V_min (float): Minimum value for C51
            V_max (float): Maximum value for C51
            learning_starts (int): Number of steps before learning starts
        """
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_step = n_step
        self.learning_starts = learning_starts

        self.num_atoms = num_atoms
        self.V_min = V_min
        self.V_max = V_max
        self.delta_z = (V_max - V_min) / (num_atoms - 1)
        self.supports = torch.linspace(V_min, V_max, num_atoms).to(device)

        # Q-Networks: CategoricalQNetwork includes Dueling and NoisyNets
        self.q_net = CategoricalQNetwork(input_shape, n_actions, num_atoms, V_min, V_max).to(device)
        self.target_q_net = CategoricalQNetwork(input_shape, n_actions, num_atoms, V_min, V_max).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # GPU-optimized Replay Buffer
        self.replay_buffer = RainbowPERBufferGPU(
            buffer_capacity, device, alpha=alpha, beta_start=beta_start, beta_frames=beta_frames
        )
        
        self.steps_done = 0
        self.n_step_buffer = []

        if self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda') 
        else:
            self.scaler = None

    def choose_action(self, state):
        with torch.no_grad():
            state_tensor = state.unsqueeze(0).to(self.device)
            q_probs = self.q_net(state_tensor)
            q_values = self.q_net.get_q_values(q_probs)
            action = q_values.argmax(dim=1).item()
        return action

    def step(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # Store N-step transition in replay buffer
        if len(self.n_step_buffer) == self.n_step or done:
            cumulative_reward = 0
            n_step_done = False
            for i in range(len(self.n_step_buffer)):
                r = self.n_step_buffer[i][2]
                d = self.n_step_buffer[i][4]
                cumulative_reward += r * (self.gamma ** i)
                if d:
                    n_step_done = True
                    break
            
            # Prepare tensors for replay buffer
            initial_state, initial_action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
            final_next_state, final_done = self.n_step_buffer[i][3], n_step_done

            initial_state_tensor = torch.from_numpy(initial_state) 
            final_next_state_tensor = torch.from_numpy(final_next_state)
            
            initial_action_tensor = torch.tensor(initial_action, dtype=torch.long)
            cumulative_reward_tensor = torch.tensor(cumulative_reward, dtype=torch.float32)
            final_done_tensor = torch.tensor(final_done, dtype=torch.bool)

            # Add to replay buffer
            self.replay_buffer.add(
                initial_state_tensor, 
                initial_action_tensor, 
                cumulative_reward_tensor, 
                final_next_state_tensor, 
                final_done_tensor
            )
            self.n_step_buffer = []

        self.steps_done += 1
        
        # Update target network periodically
        if self.steps_done % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            self.target_q_net.eval()
            self.q_net.reset_noise()
            self.target_q_net.reset_noise()

    def learn(self):
        if self.steps_done < self.learning_starts:
            return None

        if len(self.replay_buffer) < self.batch_size:
            return None

        # Update beta for PER
        self.replay_buffer.beta = min(1.0, self.replay_buffer.beta_start + self.steps_done * (1.0 - self.replay_buffer.beta_start) / self.replay_buffer.beta_frames)

        indices, states, actions, rewards, next_states, dones, weights = \
            self.replay_buffer.sample(self.batch_size)
        
        with torch.amp.autocast(device_type="cuda") if self.scaler else torch.no_grad():
            # Target Q-value calculation (Double DQN + C51 Projection)
            next_q_values_main_net = self.q_net.get_q_values(self.q_net(next_states))
            next_actions = next_q_values_main_net.argmax(dim=1).unsqueeze(1)

            # Target network evaluation
            next_q_target_output = self.target_q_net(next_states)
            next_actions_expanded = next_actions.unsqueeze(2).repeat(1, 1, self.num_atoms) 

            next_q_probs_target = next_q_target_output.gather(
                1, next_actions_expanded
            ).squeeze(1)
            
            # Compute the projected distribution
            Tz = rewards.unsqueeze(1) + (self.gamma ** self.n_step) * self.supports.unsqueeze(0) * (~dones).unsqueeze(1)
            Tz = Tz.clamp(self.V_min, self.V_max)
            
            b_j = (Tz - self.V_min) / self.delta_z
            l_j = b_j.floor().long()
            u_j = b_j.ceil().long()

            l_j = l_j.clamp(0, self.num_atoms - 1)
            u_j = u_j.clamp(0, self.num_atoms - 1)
            
            m = torch.zeros(self.batch_size, self.num_atoms, device=self.device)

            m.scatter_add_(1, l_j, next_q_probs_target * (u_j.float() - b_j))
            m.scatter_add_(1, u_j, next_q_probs_target * (b_j - l_j.float()))

            done_mask = dones.bool()
            if done_mask.any():
                # Project rewards for terminal states
                reward_vals_done = rewards[done_mask]
                reward_proj_idx_done = (reward_vals_done - self.V_min) / self.delta_z 
                
                l_done = reward_proj_idx_done.floor().long()
                u_done = reward_proj_idx_done.ceil().long()

                l_done = l_done.clamp(0, self.num_atoms - 1)
                u_done = u_done.clamp(0, self.num_atoms - 1)

                m_done_temp = torch.zeros(len(reward_vals_done), self.num_atoms, device=self.device)
                
                exact_match_mask = (l_done == u_done)
                if exact_match_mask.any():
                    m_done_temp[exact_match_mask].scatter_(1, l_done[exact_match_mask].unsqueeze(1), 1.0)

                non_exact_match_mask = ~exact_match_mask
                if non_exact_match_mask.any():
                    l_ne = l_done[non_exact_match_mask]
                    u_ne = u_done[non_exact_match_mask]
                    b_ne = reward_proj_idx_done[non_exact_match_mask]

                    m_done_temp[non_exact_match_mask].scatter_add_(1, l_ne.unsqueeze(1), (u_ne.float() - b_ne).unsqueeze(1))
                    m_done_temp[non_exact_match_mask].scatter_add_(1, u_ne.unsqueeze(1), (b_ne - l_ne.float()).unsqueeze(1))
                
                m[done_mask] = m_done_temp

            m = m.clamp(min=1e-5)

        # Calculate loss
        current_q_net_output = self.q_net(states)
        actions_expanded = actions.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.num_atoms)

        # Get the probabilities of the taken actions
        current_q_probs = current_q_net_output.gather(1, actions_expanded).squeeze(1)

        # Calculate loss
        loss = -torch.sum(m * torch.log(current_q_probs), dim=1)
        loss = (loss * weights).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            # Calculate TD-errors for priority updates using KL-divergence
            priorities_q_net_output = self.q_net(states)
            actions_expanded_for_priorities = actions.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.num_atoms)
            
            current_q_probs_for_priorities = priorities_q_net_output.gather(1, actions_expanded_for_priorities).squeeze(1)
            
            # Use KL-divergence as TD-error for C51
            td_errors_for_priorities = F.kl_div(
                torch.log(current_q_probs_for_priorities.clamp(min=1e-5)),
                m.clamp(min=1e-5),
                reduction='none'
            ).sum(dim=1)

        # Update priorities in GPU-optimized replay buffer
        self.replay_buffer.update_priorities(indices, td_errors_for_priorities)
        
        return loss.item()