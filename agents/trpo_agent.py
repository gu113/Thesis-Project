import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import random
from models.actor_critic_cnn import TRPOActorCnn, TRPOCriticCnn


class TRPOMemory:
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


class TRPOAgent():
    def __init__(self, input_shape, action_size, seed, device, gamma, alpha, beta, gae_lambda, update_every_steps, kl_constraint, cg_steps, ls_steps, entropy_beta, actor_m, critic_m):
        self.input_shape = input_shape
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.gae_lambda = gae_lambda
        self.update_every_steps = update_every_steps
        self.kl_constraint = kl_constraint
        self.cg_steps = cg_steps
        self.ls_steps = ls_steps
        self.entropy_beta = entropy_beta

        # Actor-Network
        self.actor_net = actor_m(input_shape, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha) # Not used in TRPO update, only for critic

        # Critic-Network
        self.critic_net = critic_m(input_shape).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.beta)

        # Memory for collecting trajectory segments
        self.memory = TRPOMemory()
        
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
        
        # Squeeze the extra dimension from old_log_probs to make it a 1D tensor
        old_log_probs = old_log_probs.squeeze(-1)

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

        # TRPO Policy Update
        action_dist = self.actor_net(states)
        new_log_probs = action_dist.log_prob(actions)
        ratio = (new_log_probs - old_log_probs).exp()
        
        # Calculate entropy of the new policy for the entropy bonus
        entropy = action_dist.entropy().mean()

        # Calculate gradients of L on actor parameters, including the entropy bonus
        loss = -(ratio * advantages).mean() - self.entropy_beta * entropy
        grad = torch.autograd.grad(loss, self.actor_net.parameters(), retain_graph=True)
        grad_flat = torch.cat([g.reshape(-1) for g in grad])

        # Conjugate Gradient to find inverse Hessian-vector product
        Hv_closure = lambda x: self._hessian_vector_product(x, states, action_dist.logits.detach())
        step_dir = self._conjugate_gradient(Hv_closure, grad_flat)

        # Calculate a scale for the step direction
        shs = 0.5 * step_dir.dot(Hv_closure(step_dir))
        scale = torch.sqrt(self.kl_constraint / (shs + 1e-8))
        
        full_step = scale * step_dir
        
        # Line Search
        with torch.no_grad():
            old_loss = -(ratio * advantages).mean() - self.entropy_beta * entropy
        
        for i in range(self.ls_steps):
            new_params = [p + full_step[start:start+p.numel()].reshape_as(p) for p, (start, end) in zip(self.actor_net.parameters(), self._param_indices)]
            self._set_actor_params(new_params)

            with torch.no_grad():
                new_action_dist = self.actor_net(states)
                kl_div = torch.distributions.kl_divergence(action_dist, new_action_dist).mean()
                
                # Recalculate new ratio and loss for comparison
                new_ratio = (new_action_dist.log_prob(actions) - old_log_probs).exp()
                new_entropy = new_action_dist.entropy().mean()
                new_loss = -(new_ratio * advantages).mean() - self.entropy_beta * new_entropy
                
            if kl_div <= self.kl_constraint * 1.5 and new_loss <= old_loss:
                break
            else:
                full_step *= 0.5
        
        # Critic Update
        for _ in range(self.cg_steps):
            value_pred = self.critic_net(states).squeeze(-1)
            critic_loss = F.mse_loss(value_pred, returns)
            
            self.critic_optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(critic_loss).backward()
                self.scaler.step(self.critic_optimizer)
                self.scaler.update()
            else:
                critic_loss.backward()
                self.critic_optimizer.step()

    def _hessian_vector_product(self, vector, states, old_logits):
        # Calculates H_theta(old_theta) * v, where H is the Hessian of KL divergence
        kl = torch.distributions.kl_divergence(self.actor_net(states), Categorical(logits=old_logits)).mean()
        grad_kl = torch.autograd.grad(kl, self.actor_net.parameters(), create_graph=True)
        grad_kl_flat = torch.cat([g.reshape(-1) for g in grad_kl])

        grad_product = (grad_kl_flat * vector.detach()).sum()
        hessian_vector_product = torch.autograd.grad(grad_product, self.actor_net.parameters())
        
        return torch.cat([h.reshape(-1) for h in hessian_vector_product]).detach()

    def _conjugate_gradient(self, hvp, b, max_iter=10):
        # Solves Ax = b for x where A is the Hessian
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        
        for i in range(max_iter):
            alpha = (r @ r) / (p @ hvp(p) + 1e-8)
            x_new = x + alpha * p
            r_new = r - alpha * hvp(p)
            beta = (r_new @ r_new) / (r @ r + 1e-8)
            p_new = r_new + beta * p
            
            x = x_new
            r = r_new
            p = p_new

        return x

    def _set_actor_params(self, params):
        # Helper function to set actor network parameters
        # params is a list of tensors
        for p, new_p in zip(self.actor_net.parameters(), params):
            p.data.copy_(new_p)

    @property
    def _param_indices(self):
        # Helper to get the start and end indices of each parameter in a flattened tensor
        indices = []
        offset = 0
        for p in self.actor_net.parameters():
            indices.append((offset, offset + p.numel()))
            offset += p.numel()
        return indices
