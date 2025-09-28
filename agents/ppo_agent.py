import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

class PPOAgent():
    def __init__(self, input_shape, action_size, seed, device, gamma, alpha, beta, tau, update_every, batch_size, ppo_epoch, clip_param, actor_m, critic_m):
        """Initialize an Agent object.
        Params
        ======
            input_shape (tuple): dimension of each state (C, H, W)
            action_size (int): dimension of each action
            seed (int): random seed
            device(string): Use Gpu or CPU
            gamma (float): discount factor
            alpha (float): Actor learning rate
            beta (float): Critic learning rate 
            tau (float): Tau Value
            update_every: How often to update network
            batch_size (int): Mini Batch size to be used every epoch 
            ppo_epoch(int): Total No epoch for ppo
            clip_param(float): Clip Paramter
            actor_m(Model): Pytorch Actor Model
            critic_m(Model): PyTorch Critic Model
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.ppo_epoch = ppo_epoch
        self.clip_param = clip_param

        # Actor-Network
        self.actor_net = actor_m(input_shape, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha)

        # Critic-Network
        self.critic_net = critic_m(input_shape).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.beta)

        # Memory
        self.log_probs = []
        self.values    = []
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.masks     = []
        self.entropies = []

        self.t_step = 0

    def step(self, state, action, value, log_prob, reward, done, next_state):
        
        # Save experience in  memory
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.states.append(torch.from_numpy(state).unsqueeze(0).to(self.device))
        self.rewards.append(torch.from_numpy(np.array([reward])).to(self.device))
        self.actions.append(torch.from_numpy(np.array([action])).to(self.device))
        self.masks.append(torch.from_numpy(np.array([1 - done])).to(self.device))

        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
           self.learn(next_state)
           self.reset_memory()
                
    def act(self, state):
        """Returns action, log_prob, value for given state as per current policy."""
        
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device) # Added float() to fix type mismatch
        action_probs = self.actor_net(state)
        value = self.critic_net(state)

        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)

        return action.item(), log_prob, value
        
    def learn(self, next_state):
        next_state = torch.from_numpy(next_state).unsqueeze(0).float().to(self.device) # Added float() to fix type mismatch
        next_value = self.critic_net(next_state)

        returns        = torch.cat(self.compute_gae(next_value)).detach()
        self.log_probs = torch.cat(self.log_probs).detach()
        self.values    = torch.cat(self.values).detach()
        self.states    = torch.cat(self.states)
        self.actions   = torch.cat(self.actions)
        advantages     = returns - self.values

        for _ in range(self.ppo_epoch):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(returns, advantages):

                dist = self.actor_net(state)
                value = self.critic_net(state)

                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()
                
                loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

                # Minimize the loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.reset_memory()

    
    def ppo_iter(self, returns, advantage):
        memory_size = self.states.size(0)
        for _ in range(memory_size // self.batch_size):
            rand_ids = np.random.randint(0, memory_size, self.batch_size)
            yield self.states[rand_ids, :], self.actions[rand_ids], self.log_probs[rand_ids], returns[rand_ids, :], advantage[rand_ids, :]

    def reset_memory(self):
        self.log_probs = []
        self.values    = []
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.masks     = []
        self.entropies = []

    def compute_gae(self, next_value):
        gae = 0
        returns = []
        values = self.values + [next_value]
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * self.masks[step] - values[step]
            gae = delta + self.gamma * self.tau * self.masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns
    

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, state_tensor, action_tensor, reward, done, log_prob, value):
        self.states.append(state_tensor)
        self.actions.append(action_tensor)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def get_full_trajectory(self):
        # Using torch.stack to correctly create a batch dimension
        states = torch.stack(self.states).float() 
        actions = torch.cat(self.actions) 
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.bool)
        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values).squeeze(-1)

        return states, actions, rewards, dones, log_probs, values


class ComplexPPOAgent():
    def __init__(self, input_shape, action_size, seed, device, gamma, alpha, beta, gae_lambda, update_every_steps, batch_size, ppo_epoch, clip_param, actor_m, critic_m):
        self.input_shape = input_shape
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.gae_lambda = gae_lambda
        self.update_every_steps = update_every_steps
        self.batch_size = batch_size
        self.ppo_epoch = ppo_epoch
        self.clip_param = clip_param

        # Actor-Network
        self.actor_net = actor_m(input_shape, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha)

        # Critic-Network
        self.critic_net = critic_m(input_shape).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.beta)

        # Memory for collecting trajectory segments
        self.memory = PPOMemory()
        
        self.t_step = 0

        self.scaler = torch.amp.GradScaler("cuda") if self.device.type == 'cuda' else None

    def step(self, state_tensor, action, reward, next_state_tensor, done, log_prob, value):
        self.memory.add(
            state_tensor, 
            torch.tensor([action], dtype=torch.long).to(self.device), 
            reward,
            done,
            log_prob.unsqueeze(0).detach(), 
            value
        )

        self.t_step += 1

        if self.t_step % self.update_every_steps == 0:
            self.learn()
            self.memory.clear()
            self.t_step = 0

    def act(self, state_tensor):
        state_input = state_tensor.unsqueeze(0) 
        
        self.actor_net.eval()
        self.critic_net.eval()

        with torch.no_grad():
            action_dist = self.actor_net(state_input)
            value = self.critic_net(state_input)

        action = action_dist.sample()
        log_prob = action_dist.log_prob(action) 

        return action.item(), log_prob, value

    def learn(self):
        self.actor_net.train()
        self.critic_net.train()

        states, actions, rewards, dones, old_log_probs, values = self.memory.get_full_trajectory()

        returns = []
        advantages = []
        gae = 0

        with torch.no_grad():
            last_value = self.critic_net(states[-1].unsqueeze(0)).squeeze(-1) 
            last_value = last_value.item() * (1 - dones[-1].item()) 

            for i in reversed(range(len(rewards))):
                current_reward = rewards[i].item()
                current_value = values[i].item() 
                current_done = dones[i].item()

                if i == len(rewards) - 1:
                    next_value_for_delta = last_value
                else:
                    next_value_for_delta = values[i+1].item()

                delta = current_reward + self.gamma * next_value_for_delta * (1 - current_done) - current_value
                gae = delta + self.gamma * self.gae_lambda * (1 - current_done) * gae
                
                advantages.insert(0, gae)
                returns.insert(0, gae + current_value)

            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
            
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(len(states))
        
        for _ in range(self.ppo_epoch):
            np.random.shuffle(indices)
            for i in range(0, len(states), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]

                batch_states = states[batch_indices] 
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                if self.scaler:
                    with torch.amp.autocast(device_type="cuda"):
                        action_dist = self.actor_net(batch_states)
                        new_log_probs = action_dist.log_prob(batch_actions)
                        
                        batch_old_log_probs_cast = batch_old_log_probs.to(new_log_probs.dtype) 
                        batch_advantages_cast = batch_advantages.to(new_log_probs.dtype)

                        ratio = (new_log_probs - batch_old_log_probs_cast).exp()
                        surr1 = ratio * batch_advantages_cast
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages_cast
                        
                        actor_loss = -torch.min(surr1, surr2).mean()
                        
                        value_pred = self.critic_net(batch_states).squeeze(-1)
                        
                        batch_returns_cast = batch_returns.to(value_pred.dtype)

                        critic_loss = F.mse_loss(value_pred, batch_returns_cast)

                        entropy_loss = action_dist.entropy().mean()
                        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss 
                else: 
                    action_dist = self.actor_net(batch_states)
                    new_log_probs = action_dist.log_prob(batch_actions)
                    
                    ratio = (new_log_probs - batch_old_log_probs).exp()
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    value_pred = self.critic_net(batch_states).squeeze(-1)
                    
                    critic_loss = F.mse_loss(value_pred, batch_returns)

                    entropy_loss = action_dist.entropy().mean()
                    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss 

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.actor_optimizer)
                    self.scaler.unscale_(self.critic_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=1.0)
                    self.scaler.step(self.actor_optimizer)
                    self.scaler.step(self.critic_optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=1.0)
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()