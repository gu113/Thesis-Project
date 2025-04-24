import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from utils.replay_buffer_PER import ReplayBufferPER
from torch.cuda.amp import autocast, GradScaler


class DDQNAgentPER():
    def __init__(self, input_shape, action_size, seed, device, buffer_size, batch_size, gamma, lr, tau, update_every, replay_after, model, alpha):
        """Initialize an Agent object.
        
        Params
        ======
            input_shape (tuple): dimension of each state (C, H, W)
            action_size (int): dimension of each action
            seed (int): random seed
            device(string): Use Gpu or CPU
            buffer_size (int): replay buffer size
            batch_size (int):  minibatch size
            gamma (float): discount factor
            lr (float): learning rate 
            update_every (int): how often to update the network
            replay_after (int): After which replay to be started
            model(Model): Pytorch Model
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_every = update_every
        self.replay_after = replay_after
        self.DQN = model
        self.tau = tau
        self.alpha = alpha

        
        # Q-Network
        # self.policy_net = self.DQN(input_shape, action_size).to(self.device)
        # self.target_net = self.DQN(input_shape, action_size).to(self.device)

        # Q-Network with HALF PERCISION
        self.policy_net = self.DQN(input_shape, action_size).to(self.device)
        self.target_net = self.DQN(input_shape, action_size).to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Replay memory
        #self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.seed, self.device)
        self.memory = ReplayBufferPER(self.buffer_size, self.batch_size, self.seed, self.device, self.alpha)
        
        self.t_step = 0

    
    def step(self, state, action, reward, next_state, done):

        # Save experience in replay memory
        #self.memory.add(state, action, reward, next_state, done)
    	
        # Use PER
        with torch.no_grad():
            ###state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device).half()
            state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
            ###next_state_tensor = torch.from_numpy(next_state).unsqueeze(0).to(self.device).half()
            next_state_tensor = torch.from_numpy(next_state).unsqueeze(0).float().to(self.device)
            
            # Compute TD error
            current_q = self.policy_net(state_tensor)[0, action]
            next_q = self.target_net(next_state_tensor).max(1)[0].item()
            target_q = reward + (self.gamma * next_q * (1 - done))
            
            # Detach values before computing error
            ###td_error = abs(current_q.detach() - target_q).cpu().numpy()
            td_error = abs(current_q.float().detach() - torch.tensor(target_q).float()).cpu().numpy()

        # Store in PER buffer
        self.memory.add(state, action, reward, next_state, done, td_error)
        
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        """
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.replay_after:
                experiences = self.memory.sample()
                self.learn(experiences)
        """

        if self.t_step == 0 and len(self.memory) > self.replay_after:
            experiences = self.memory.sample(beta=min(1.0, 0.4 + self.t_step * (1.0 - 0.4) / 10000))
            self.learn(experiences)
                
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy."""
        
        #state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        state = torch.from_numpy(state).unsqueeze(0).to(self.device).half() # FP16 - HALF PRECISION

        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences):

        self.scaler = torch.amp.GradScaler("cuda")

        # Unpack experiences
        states, actions, rewards, next_states, dones, indices, weights = experiences
        #experiences, indices, weights = experiences

        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Convert PER weights to tensor
        #weights = torch.from_numpy(weights).to(self.device).half()
        ###weights = weights.to(self.device).half()


        # Default
        # states, actions, rewards, next_states, dones = experiences

        # Convert to FP16 - HALF PRECISION
        ###states = states.half()
        ###next_states = next_states.half()

        # Get expected Q values from policy model
        #Q_expected_current = self.policy_net(states)
        #Q_expected = Q_expected_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            # Get expected Q values from policy model (PER)
            Q_expected = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.target_net(next_states).detach().max(1)[0]
            
            # Compute Q targets for current states 
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
            # Compute loss
            #loss = F.mse_loss(Q_expected, Q_targets)

            # Compute loss with PER weights
            loss = (weights * (Q_expected - Q_targets) ** 2).mean()

        # Scale the loss for FP16
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Minimize the loss
        ###self.optimizer.zero_grad()
        ###loss.backward()
        ###self.optimizer.step()

        # Update priorities in the PER buffer
        td_errors = abs(Q_expected - Q_targets).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        self.soft_update(self.policy_net, self.target_net, self.tau)

    
    # θ'=θ×τ+θ'×(1−τ)
    def soft_update(self, policy_model, target_model, tau):
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)