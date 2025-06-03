import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from utils.replay_buffer import ReplayBuffer, OldPERBuffer, AsyncReplayBuffer, AsyncRamReplayBuffer, PERBuffer, RewardShapingPERBuffer

class DDQNAgent():
    def __init__(self, input_shape, action_size, seed, device, buffer_size, batch_size, gamma, lr, tau, update_every, replay_after, model):
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

        
        # Q-Network
        self.policy_net = self.DQN(input_shape, action_size).to(self.device)
        self.target_net = self.DQN(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.seed, self.device)
        
        self.t_step = 0

    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.replay_after:
                experiences = self.memory.sample()
                self.learn(experiences)
                
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy."""
        
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        else:
            state = state.unsqueeze(0).to(self.device)

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
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from policy model
        Q_expected_current = self.policy_net(states)
        Q_expected = Q_expected_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_net(next_states).detach().max(1)[0]
        
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.policy_net, self.target_net, self.tau)

    
    # θ'=θ×τ+θ'×(1−τ)
    @torch.no_grad()
    def soft_update(self, policy_model, target_model, tau):
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)



class OldAsyncDDQNAgent():
    def __init__(self, input_shape, action_size, seed, device, buffer_size, batch_size, gamma, lr, tau, update_every, replay_after, model):
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

        
        # Q-Network
        self.policy_net = self.DQN(input_shape, action_size).to(self.device)
        self.target_net = self.DQN(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.seed, self.device)
        
        self.t_step = 0

    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.replay_after:
                experiences = self.memory.sample()
                self.learn(experiences)
                
    def act(self, state_batch, eps=0.0):
        self.policy_net.eval()
        
        # Ensure input is a NumPy array, and has correct shape
        if isinstance(state_batch, np.ndarray):
            state_batch = torch.from_numpy(state_batch).float().to(self.device)
        elif isinstance(state_batch, torch.Tensor):
            state_batch = state_batch.float().to(self.device)
        else:
            raise TypeError("state_batch must be np.ndarray or torch.Tensor")

        with torch.no_grad():
            q_values = self.policy_net(state_batch)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        self.policy_net.train()

        # Apply epsilon-greedy strategy
        for i in range(len(actions)):
            if np.random.rand() < eps:
                actions[i] = np.random.randint(0, self.action_size)

        return actions

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from policy model
        Q_expected_current = self.policy_net(states)
        Q_expected = Q_expected_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_net(next_states).detach().max(1)[0]
        
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.policy_net, self.target_net, self.tau)

    
    # θ'=θ×τ+θ'×(1−τ)
    def soft_update(self, policy_model, target_model, tau):
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)


class OldDDQNAgentPER():
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
        self.scaler = torch.amp.GradScaler("cuda")
        self.learn_step = 0

        
        # Q-Network
        # self.policy_net = self.DQN(input_shape, action_size).to(self.device)
        # self.target_net = self.DQN(input_shape, action_size).to(self.device)

        # Q-Network with HALF PERCISION
        self.policy_net = self.DQN(input_shape, action_size).to(self.device)
        self.target_net = self.DQN(input_shape, action_size).to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Replay memory
        #self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.seed, self.device)
        self.memory = OldPERBuffer(self.buffer_size, self.batch_size, self.seed, self.device, self.alpha)
        
        self.t_step = 0

    
    def step(self, state, action, reward, next_state, done):

        # Save experience in replay memory
        #self.memory.add(state, action, reward, next_state, done)
    	
        # Use PER
        with torch.no_grad():

            """
            ###state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device).half()
            state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
            ###next_state_tensor = torch.from_numpy(next_state).unsqueeze(0).to(self.device).half()
            next_state_tensor = torch.from_numpy(next_state).unsqueeze(0).float().to(self.device)
            
            # Compute TD error
            current_q = self.policy_net(state_tensor)[0, action]
            next_q = self.target_net(next_state_tensor).max(1)[0].item()
            target_q = reward + (self.gamma * next_q * (1 - done))
            """

            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device).float()#.half()
            next_state_tensor = torch.from_numpy(next_state).unsqueeze(0).to(self.device).float()#.half()
            
            #self.policy_net#.half()
            #self.target_net#.half()
            
            current_q = self.policy_net(state_tensor)[0, action]
            next_q = self.target_net(next_state_tensor).max(1)[0].item()
            target_q = reward + (self.gamma * next_q * (1 - done))
            
            # Detach values before computing error
            ###td_error = abs(current_q.detach() - target_q).cpu().numpy()
            td_error = abs(current_q.float().detach() - torch.tensor(target_q).float()).item()

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

        """
        if self.t_step == 0 and len(self.memory) > self.replay_after:
            experiences = self.memory.sample(beta=min(1.0, 0.4 + self.t_step * (1.0 - 0.4) / 10000))
            self.learn(experiences)
        """

        if self.t_step == 0 and len(self.memory) > self.replay_after:
            beta = min(1.0, 0.4 + self.learn_step * (1.0 - 0.4) / 100000)
            experiences = self.memory.sample(beta=beta)
            self.learn(experiences)
            self.learn_step += 1
                
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy."""
        
        #state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        state = torch.from_numpy(state).unsqueeze(0).to(self.device).float()#.half() # FP16 - HALF PRECISION

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
        td_errors = (Q_expected - Q_targets).detach().abs().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        self.soft_update(self.policy_net, self.target_net, self.tau)

    
    # θ'=θ×τ+θ'×(1−τ)
    def soft_update(self, policy_model, target_model, tau):
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)


class AsyncDDQNAgent():
    def __init__(self, input_shape, action_size, seed, device, buffer_size, batch_size, gamma, lr, tau, update_every, replay_after, model):
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

        
        # Q-Network
        self.policy_net = self.DQN(input_shape, action_size).to(self.device)
        self.target_net = self.DQN(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Replay memory
        self.memory = AsyncReplayBuffer(self.buffer_size, self.batch_size, self.seed, self.device)
        
        self.t_step = 0

    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.replay_after:
                experiences = self.memory.sample()
                self.learn(experiences)
                
    def act(self, states, eps=0.0):
        states = torch.tensor(states, dtype=torch.float32, device=self.device) # Uses fp16, OR torch.float32 for normal version
        with torch.no_grad():
            q_values = self.policy_net(states)
        if np.random.rand() > eps:
            return torch.argmax(q_values, dim=1).cpu().numpy()
        else:
            return np.random.randint(self.action_size, size=(states.shape[0],))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        
        # Get Q-values for current states
        Q_expected_current = self.policy_net(states)
        
        # Get Q-values for next states (target network)
        Q_expected_next = self.target_net(next_states)
        
        # Ensure Q_targets_next has the shape [batch_size] (max Q-value per sample)
        Q_targets_next = Q_expected_next.max(1)[0]  # This will have shape [batch_size]
        
        # Gather Q-values based on actions
        Q_expected = Q_expected_current.gather(1, actions.view(-1, 1)).squeeze(1)
        
        # Ensure rewards and dones are 1D tensors with shape [batch_size]
        rewards = rewards.view(-1)
        dones = dones.view(-1)
        
        # Compute target Q-values (Double DQN target)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))  # Shape [batch_size]
        
        # Compute loss and perform optimization step
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    # θ'=θ×τ+θ'×(1−τ)
    def soft_update(self, policy_model, target_model, tau):
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)


class AsyncRamDDQNAgent():
    def __init__(self, input_shape, action_size, seed, device, buffer_size, batch_size, gamma, lr, tau, update_every, replay_after, model):
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

        
        # Q-Network
        self.policy_net = self.DQN(input_shape, action_size).to(self.device)
        self.target_net = self.DQN(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Replay memory
        self.memory = AsyncRamReplayBuffer(self.buffer_size, self.batch_size, self.seed, self.device)
        
        self.t_step = 0
        self.update_counter = 0
        self.target_update_every = 1000

    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.replay_after:
                experiences = self.memory.sample()
                self.learn(experiences)
                
    def act(self, states, eps=0.0):
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device) # Uses fp16, OR torch.float32 for normal version
        with torch.no_grad():
            q_values = self.policy_net(states)
        if np.random.rand() > eps:
            return torch.argmax(q_values, dim=1).detach().cpu().numpy()
        else:
            return np.random.randint(self.action_size, size=(states.shape[0],))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get Q-values from policy network for current states
        Q_expected_current = self.policy_net(states)
        Q_expected = Q_expected_current.gather(1, actions.view(-1, 1)).squeeze(1)

        # Double DQN:
        # Select best actions using policy_net
        next_actions = self.policy_net(next_states).argmax(1, keepdim=True)

        # Evaluate selected actions using target_net
        Q_targets_next = self.target_net(next_states).gather(1, next_actions).squeeze(1)

        # Compute target Q-values
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        # Soft update target network
        ##self.soft_update(self.policy_net, self.target_net, self.tau)
        self.update_counter += 1
        if self.update_counter % self.target_update_every == 0:
            self.soft_update(self.policy_net, self.target_net, self.tau)


    @torch.no_grad()
    def soft_update(self, policy_net, target_net, tau):
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(
                tau * policy_param.data.to(target_param.device) +
                (1.0 - tau) * target_param.data
            )


class DDQNAgentPER():
    def __init__(self, input_shape, action_size, seed, device, buffer_size, batch_size, gamma, lr, tau, update_every, replay_after, model):
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

        self.policy_net = self.DQN(input_shape, action_size).to(self.device)
        self.target_net = self.DQN(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.memory = PERBuffer(buffer_size, batch_size, seed, device)
        self.frame_idx = 0

        # PER beta annealing parameters
        self.per_beta_start = 0.4
        self.per_beta_frames = 100000

        self.t_step = 0

    def _beta_by_frame(self, frame_idx):
        return min(1.0, self.per_beta_start + frame_idx * (1.0 - self.per_beta_start) / self.per_beta_frames)

    def step(self, state, action, reward, next_state, done):
        max_priority = self.memory.tree.max() if self.memory.tree.n_entries > 0 else 1.0
        # Add experience with max priority, note 'error' param for add() here is max_priority for new samples
        self.memory.add(max_priority, state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if self.memory.tree.n_entries > self.replay_after:
                beta = self._beta_by_frame(self.frame_idx)
                experiences, idxs, is_weights = self.memory.sample()
                # is_weights already accounts for beta internally in your buffer
                self.learn(experiences, idxs, is_weights)
                self.frame_idx += 1

    def act(self, state, eps=0.):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).unsqueeze(0).to(self.device).float()
        else:
            state = state.unsqueeze(0).to(self.device).float()

        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, idxs, is_weights):
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.stack(states).to(self.device).float()
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device).float()
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        ##is_weights = is_weights.clone().detach().float().to(self.device)
        is_weights = is_weights.float()

        Q_expected_all = self.policy_net(states)
        Q_expected = Q_expected_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        Q_targets_next = self.target_net(next_states).detach().max(1)[0]

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        td_errors = Q_expected - Q_targets

        loss = (is_weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.policy_net, self.target_net, self.tau)

        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(idxs, new_priorities)

    @torch.no_grad()
    def soft_update(self, policy_model, target_model, tau):
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)


class ComplexDDQNAgentPER():
    def __init__(self, input_shape, action_size, seed, device, buffer_size, batch_size, gamma, lr, tau, update_every, replay_after, model, use_per=True):
        """Enhanced DDQN Agent with Prioritized Experience Replay and reward focusing"""
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
        self.use_per = use_per

        # Q-Networks
        self.policy_net = self.DQN(input_shape, action_size).to(self.device)
        self.target_net = self.DQN(input_shape, action_size).to(self.device)
        
        # Enhanced optimizer with learning rate scheduling
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-5
        )
        
        # More aggressive learning rate schedule for faster convergence
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',  # Reduce when performance plateaus
            factor=0.8,
            patience=5000,
        )
        
        # Initialize target network
        self.hard_update(self.target_net, self.policy_net)
        
        # Choose replay buffer type
        if use_per:
            self.memory = PERBuffer(
                self.buffer_size, self.batch_size, self.seed, self.device,
                alpha=0.6, beta=0.4, beta_increment=0.0001
            )
        else:
            self.memory = RewardShapingPERBuffer(
                self.buffer_size, self.batch_size, self.seed, self.device,
                reward_threshold=300  # Focus on episodes with 300+ score
            )
        
        # Counters
        self.t_step = 0
        self.learn_step = 0
        self.current_episode_reward = 0
        
        # Performance tracking
        self.loss_history = []
        self.q_values_history = []
        self.td_errors_history = []

    def start_episode(self):
        """Call this at the start of each episode"""
        self.current_episode_reward = 0

    def step(self, state, action, reward, next_state, done):
        # Track episode reward
        self.current_episode_reward += reward
        
        # Save experience in replay memory
        if self.use_per:
            max_priority = self.memory.tree.max() if self.memory.tree.n_entries > 0 else 1.0
            self.memory.add(max_priority, state, action, reward, next_state, done)
        else:
            episode_reward = self.current_episode_reward if done else 0
            self.memory.add(state, action, reward, next_state, done, episode_reward)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            if len(self.memory) > self.replay_after:
                if self.use_per:
                    experiences = self.memory.sample()
                    if experiences is not None:
                        loss, td_errors = self.learn_per(experiences)
                        self.loss_history.append(loss)
                else:
                    experiences = self.memory.sample()
                    if experiences is not None:
                        loss = self.learn(experiences)
                        self.loss_history.append(loss)
                
    def act(self, state, eps=0.):
        """Enhanced action selection with better exploration"""
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        else:
            state = state.float().unsqueeze(0).to(self.device)
            
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
            
        self.policy_net.train()
        
        # Store Q-values for monitoring
        max_q = action_values.cpu().numpy().max()
        self.q_values_history.append(max_q)
        
        # Enhanced epsilon-greedy with Boltzmann exploration for better actions
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Occasionally use Boltzmann exploration instead of random
            if eps > 0.1 and random.random() < 0.3:  # 30% of exploration uses Boltzmann
                temperature = eps * 2  # Higher temperature = more exploration
                probs = F.softmax(action_values / temperature, dim=1)
                action = torch.multinomial(probs, 1).item()
                return action
            else:
                return random.choice(np.arange(self.action_size))
        
    def learn_per(self, experiences):
        """Learning with Prioritized Experience Replay"""
        states, actions, rewards, next_states, dones, indices, is_weights = experiences

        # Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            Q_targets_next = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate TD errors for priority updates
        td_errors = Q_targets - Q_expected
        
        # Weighted loss using importance sampling
        loss = (is_weights * F.smooth_l1_loss(Q_expected, Q_targets, reduction='none')).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update priorities in replay buffer
        priorities = abs(td_errors.detach().cpu().numpy()) + 1e-6
        self.memory.update_priorities(indices, priorities)
        
        # Update target network
        self.soft_update(self.policy_net, self.target_net, self.tau)
        
        self.learn_step += 1
        self.td_errors_history.extend(abs(td_errors.detach().cpu().numpy()))
        
        return loss.item(), td_errors.detach().cpu().numpy()
        
    def learn(self, experiences):
        """Standard learning for reward shaping buffer"""
        states, actions, rewards, next_states, dones = experiences

        # Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            Q_targets_next = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Huber loss
        loss = F.smooth_l1_loss(Q_expected, Q_targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network
        self.soft_update(self.policy_net, self.target_net, self.tau)
        
        self.learn_step += 1
        
        return loss.item()
    
    def update_learning_rate(self, avg_score):
        """Update learning rate based on performance"""
        self.scheduler.step(avg_score)
    
    def soft_update(self, policy_model, target_model, tau):
        """Soft update model parameters"""
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
    
    def hard_update(self, target_model, policy_model):
        """Hard update: copy weights from policy to target network"""
        target_model.load_state_dict(policy_model.state_dict())
    
    def get_stats(self):
        """Return detailed training statistics"""
        stats = {
            'avg_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            'avg_q_value': np.mean(self.q_values_history[-100:]) if self.q_values_history else 0,
            'avg_td_error': np.mean(self.td_errors_history[-100:]) if self.td_errors_history else 0,
            'buffer_size': len(self.memory),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'learn_steps': self.learn_step
        }
        return stats