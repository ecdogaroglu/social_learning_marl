import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from collections import deque, namedtuple
import random

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experience replay buffer
Experience = namedtuple('Experience', 
                        ('observation', 'latent_strategies', 'action', 'other_actions', 
                         'reward', 'next_observation', 'next_latent_strategies'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, *args):
        self.buffer.append(Experience(*args))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Belief state representation using GRU
class BeliefStateNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim):
        super(BeliefStateNetwork, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.input_dim = observation_dim + action_dim
        self.gru = nn.GRU(self.input_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim
        self.hidden_state = None
        
    def initialize_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)
        
    def forward(self, observation, action, hidden=None):
        if hidden is None and self.hidden_state is None:
            # Initialize hidden state based on batch size
            batch_size = observation.shape[0] if observation.dim() > 1 else 1
            self.hidden_state = self.initialize_hidden(batch_size)
        elif hidden is not None:
            self.hidden_state = hidden
        
        # Ensure both tensors are properly shaped for GRU
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)  # Add batch dimension
        if action.dim() == 1:
            action = action.unsqueeze(0)  # Add batch dimension
            
        # Handle case where action has extra dimensions (e.g., from one-hot encoding)
        if action.dim() == 3:
            action = action.squeeze(1)
            
        # Check if batch sizes match
        if observation.shape[0] != action.shape[0]:
            # Adjust smaller tensor to match larger one
            if observation.shape[0] < action.shape[0]:
                observation = observation.expand(action.shape[0], -1)
            else:
                action = action.expand(observation.shape[0], -1)
        
        # Check if dimensions match expected dimensions
        if observation.shape[-1] != self.observation_dim:
            # Reshape or pad/truncate to match expected size
            if observation.shape[-1] > self.observation_dim:
                observation = observation[..., :self.observation_dim]
            else:
                # Pad with zeros
                padding = torch.zeros(observation.shape[0], self.observation_dim - observation.shape[-1], device=observation.device)
                observation = torch.cat([observation, padding], dim=-1)
                
        if action.shape[-1] != self.action_dim:
            # Reshape or pad/truncate to match expected size
            if action.shape[-1] > self.action_dim:
                action = action[..., :self.action_dim]
            else:
                # Pad with zeros
                padding = torch.zeros(action.shape[0], self.action_dim - action.shape[-1], device=action.device)
                action = torch.cat([action, padding], dim=-1)
                
        # Combine observation and action
        x = torch.cat([observation, action], dim=-1).unsqueeze(1)  # Add sequence dimension
        
        # Ensure hidden state batch size matches input batch size
        # Check if we need to resize hidden state
        if self.hidden_state.shape[1] != x.shape[0]:
            # Create a new hidden state with the right batch size instead of trying to expand
            # This avoids dimension mismatch errors when batch sizes don't match
            new_hidden = torch.zeros(1, x.shape[0], self.hidden_dim, device=device)
            # Copy values where possible
            min_batch = min(self.hidden_state.shape[1], x.shape[0])
            new_hidden[0, :min_batch] = self.hidden_state[0, :min_batch]
            self.hidden_state = new_hidden
        
        # Update belief state
        output, self.hidden_state = self.gru(x, self.hidden_state)
        
        return output.squeeze(1), self.hidden_state

# Variational inference components for policy inference
class EncoderNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, latent_dim, hidden_dim):
        super(EncoderNetwork, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.total_input_dim = observation_dim + action_dim * 2
        self.fc1 = nn.Linear(self.total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, observation, action, other_action):
        # Ensure all inputs have correct dimensions
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if other_action.dim() == 1:
            other_action = other_action.unsqueeze(0)
            
        # Check and fix input dimensions
        if observation.shape[-1] != self.observation_dim:
            # Reshape or pad/truncate to match expected size
            if observation.shape[-1] > self.observation_dim:
                observation = observation[..., :self.observation_dim]
            else:
                padding = torch.zeros(observation.shape[0], self.observation_dim - observation.shape[-1], device=observation.device)
                observation = torch.cat([observation, padding], dim=-1)
                
        if action.shape[-1] != self.action_dim:
            # Reshape or pad/truncate to match expected size
            if action.shape[-1] > self.action_dim:
                action = action[..., :self.action_dim]
            else:
                padding = torch.zeros(action.shape[0], self.action_dim - action.shape[-1], device=action.device)
                action = torch.cat([action, padding], dim=-1)
                
        if other_action.shape[-1] != self.action_dim * (self.total_input_dim - self.observation_dim - self.action_dim):
            # For other_action, we need to handle multi-agent scenarios
            # Resize to match expected input size
            needed_size = self.total_input_dim - self.observation_dim - self.action_dim
            if other_action.shape[-1] > needed_size:
                other_action = other_action[..., :needed_size]
            else:
                padding = torch.zeros(other_action.shape[0], needed_size - other_action.shape[-1], device=other_action.device)
                other_action = torch.cat([other_action, padding], dim=-1)
        
        x = torch.cat([observation, action, other_action], dim=-1)
        
        # Ensure the concatenated tensor has the right shape for fc1
        if x.shape[-1] != self.total_input_dim:
            # Resize to match expected input
            if x.shape[-1] > self.total_input_dim:
                x = x[..., :self.total_input_dim]
            else:
                padding = torch.zeros(x.shape[0], self.total_input_dim - x.shape[-1], device=x.device)
                x = torch.cat([x, padding], dim=-1)
                
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar
    
    def sample(self, observation, action, other_action):
        mean, logvar = self.forward(observation, action, other_action)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

class DecoderNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, latent_dim, hidden_dim, discrete_actions=True):
        super(DecoderNetwork, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.input_dim = observation_dim + latent_dim
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.discrete_actions = discrete_actions
        
        if discrete_actions:
            self.action_probs = nn.Linear(hidden_dim, action_dim)
        else:
            self.action_mean = nn.Linear(hidden_dim, action_dim)
            self.action_logstd = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, observation, latent):
        # Ensure tensors have correct dimensions
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
            
        # Check and fix input dimensions
        if observation.shape[-1] != self.observation_dim:
            # Reshape or pad/truncate to match expected size
            if observation.shape[-1] > self.observation_dim:
                observation = observation[..., :self.observation_dim]
            else:
                padding = torch.zeros(observation.shape[0], self.observation_dim - observation.shape[-1], device=observation.device)
                observation = torch.cat([observation, padding], dim=-1)
                
        if latent.shape[-1] != self.latent_dim:
            # Reshape or pad/truncate to match expected size
            if latent.shape[-1] > self.latent_dim:
                latent = latent[..., :self.latent_dim]
            else:
                padding = torch.zeros(latent.shape[0], self.latent_dim - latent.shape[-1], device=latent.device)
                latent = torch.cat([latent, padding], dim=-1)
        
        x = torch.cat([observation, latent], dim=-1)
        
        # Ensure input size matches expected size
        if x.shape[-1] != self.input_dim:
            if x.shape[-1] > self.input_dim:
                x = x[..., :self.input_dim]
            else:
                padding = torch.zeros(x.shape[0], self.input_dim - x.shape[-1], device=x.device)
                x = torch.cat([x, padding], dim=-1)
                
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        if self.discrete_actions:
            action_probs = F.softmax(self.action_probs(x), dim=-1)
            return action_probs
        else:
            action_mean = self.action_mean(x)
            action_logstd = self.action_logstd(x)
            return action_mean, action_logstd

# Policy network that takes belief states as inputs
class PolicyNetwork(nn.Module):
    def __init__(self, belief_dim, action_dim, latent_dim, hidden_dim, discrete_actions=True):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(belief_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.discrete_actions = discrete_actions
        
        if discrete_actions:
            self.action_probs = nn.Linear(hidden_dim, action_dim)
        else:
            self.action_mean = nn.Linear(hidden_dim, action_dim)
            self.action_logstd = nn.Linear(hidden_dim, action_dim)
            
    def forward(self, belief, latent):
        # Ensure tensors have the right dimensions
        if belief.dim() == 1:
            belief = belief.unsqueeze(0)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
            
        x = torch.cat([belief, latent], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        if self.discrete_actions:
            action_probs = F.softmax(self.action_probs(x), dim=-1)
            return action_probs
        else:
            action_mean = self.action_mean(x)
            action_logstd = self.action_logstd(x)
            action_logstd = torch.clamp(action_logstd, -20, 2)  # Prevent numerical instability
            return action_mean, action_logstd
    
    def sample_action(self, belief, latent):
        # Ensure tensors have the right dimensions
        if belief.dim() == 1:
            belief = belief.unsqueeze(0)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
            
        if self.discrete_actions:
            action_probs = self.forward(belief, latent)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob
        else:
            action_mean, action_logstd = self.forward(belief, latent)
            action_std = torch.exp(action_logstd)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob

# Q-network for action-value estimation
class QNetwork(nn.Module):
    def __init__(self, belief_dim, action_dim, latent_dim, hidden_dim, discrete_actions=True):
        super(QNetwork, self).__init__()
        self.discrete_actions = discrete_actions
        self.action_dim = action_dim
        
        if discrete_actions:
            # For discrete actions, we output Q-values for each action
            self.fc1 = nn.Linear(belief_dim + latent_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.q_values = nn.Linear(hidden_dim, action_dim)
        else:
            # For continuous actions, we take action as input
            self.fc1 = nn.Linear(belief_dim + latent_dim + action_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.q_value = nn.Linear(hidden_dim, 1)
            
    def forward(self, belief, latent, action=None):
        # Ensure tensors have the right dimensions
        if belief.dim() == 1:
            belief = belief.unsqueeze(0)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
            
        if self.discrete_actions:
            x = torch.cat([belief, latent], dim=-1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            q_values = self.q_values(x)
            return q_values
        else:
            # For continuous actions, ensure action has right dimensions
            if action.dim() == 1:
                action = action.unsqueeze(0)
                
            x = torch.cat([belief, latent, action], dim=-1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            q_value = self.q_value(x)
            return q_value

# FURTHER algorithm for partially observable settings
class PartiallyObservableFURTHER:
    def __init__(self, 
                 observation_dim, 
                 action_dim, 
                 latent_dim=5, 
                 hidden_dim=128, 
                 learning_rate=0.001, 
                 gamma=0.99, 
                 tau=0.005, 
                 alpha=0.2, 
                 discrete_actions=True,
                 buffer_size=10000,
                 batch_size=64):
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.discrete_actions = discrete_actions
        self.batch_size = batch_size
        
        # Initialize networks
        self.belief_network = BeliefStateNetwork(observation_dim, action_dim, hidden_dim).to(device)
        
        self.encoder = EncoderNetwork(observation_dim, action_dim, latent_dim, hidden_dim).to(device)
        self.decoder = DecoderNetwork(observation_dim, action_dim, latent_dim, hidden_dim, discrete_actions).to(device)
        
        self.policy = PolicyNetwork(hidden_dim, action_dim, latent_dim, hidden_dim, discrete_actions).to(device)
        
        self.q1 = QNetwork(hidden_dim, action_dim, latent_dim, hidden_dim, discrete_actions).to(device)
        self.q2 = QNetwork(hidden_dim, action_dim, latent_dim, hidden_dim, discrete_actions).to(device)
        self.target_q1 = QNetwork(hidden_dim, action_dim, latent_dim, hidden_dim, discrete_actions).to(device)
        self.target_q2 = QNetwork(hidden_dim, action_dim, latent_dim, hidden_dim, discrete_actions).to(device)
        
        # Copy parameters to target networks
        self.hard_update(self.q1, self.target_q1)
        self.hard_update(self.q2, self.target_q2)
        
        # Average reward estimate
        self.log_rho = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.q1_optimizer = optim.Adam(list(self.q1.parameters()) + [self.log_rho], lr=learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=learning_rate)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)
        self.belief_optimizer = optim.Adam(self.belief_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # For metrics
        self.policy_losses = []
        self.q_losses = []
        self.elbo_losses = []
        
    def select_action(self, observation, latent_strategies, evaluate=False):
        """Select an action given the current observation and inferred latent strategies"""
        with torch.no_grad():
            # Convert to tensors
            observation = torch.FloatTensor(observation).to(device)
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)
            
            latent_strategies = torch.FloatTensor(latent_strategies).to(device)
            if latent_strategies.dim() == 1:
                latent_strategies = latent_strategies.unsqueeze(0)
            
            # Get zero action for initial belief update
            if self.discrete_actions:
                dummy_action = torch.zeros(1, self.action_dim, device=device)
            else:
                dummy_action = torch.zeros(1, self.action_dim, device=device)
            
            # Update belief state
            belief, _ = self.belief_network(observation, dummy_action)
            
            if evaluate:
                # Select action with highest probability/mean
                if self.discrete_actions:
                    action_probs = self.policy(belief, latent_strategies)
                    action = torch.argmax(action_probs, dim=-1).item()
                else:
                    action_mean, _ = self.policy(belief, latent_strategies)
                    action = action_mean.cpu().numpy().flatten()
            else:
                # Sample action
                action, _ = self.policy.sample_action(belief, latent_strategies)
                if self.discrete_actions:
                    action = action.item()
                else:
                    action = action.cpu().numpy().flatten()
                    
            return action, belief.cpu().numpy().flatten()
    
    def infer_latent_strategies(self, observation, action, other_action):
        """Infer the latent strategies of other agents"""
        with torch.no_grad():
            # Convert to tensors
            observation = torch.FloatTensor(observation).to(device)
            if observation.dim() == 1:
                observation = observation.unsqueeze(0)
                
            action = torch.FloatTensor(action).to(device)
            if action.dim() == 1:
                action = action.unsqueeze(0)
                
            other_action = torch.FloatTensor(other_action).to(device)
            if other_action.dim() == 1:
                other_action = other_action.unsqueeze(0)
            
            # Infer latent strategies
            latent_strategies = self.encoder.sample(observation, action, other_action)
            
            return latent_strategies.cpu().numpy().flatten()
    
    def update_belief(self, observation, action, hidden=None):
        """Update the belief state given new observation and action"""
        with torch.no_grad():
            # Convert to tensors
            observation = torch.FloatTensor(observation).to(device)
            action = torch.FloatTensor(action).to(device)
            
            # Update belief state
            belief, hidden_state = self.belief_network(observation, action, hidden)
            
            return belief.cpu().numpy().flatten(), hidden_state
    
    def add_experience(self, observation, latent_strategies, action, other_actions, reward, next_observation, next_latent_strategies):
        """Add experience to replay buffer"""
        self.replay_buffer.push(observation, latent_strategies, action, other_actions, reward, next_observation, next_latent_strategies)
    
    def compute_elbo_loss(self, observations, actions, other_actions):
        """Compute the ELBO loss for policy inference"""
        # Get latent variables from encoder
        latent_mean, latent_logvar = self.encoder(observations, actions, other_actions)
        latent_std = torch.exp(0.5 * latent_logvar)
        latent_dist = Normal(latent_mean, latent_std)
        
        # Sample latent variables
        latent = latent_dist.rsample()
        
        # Get predicted action probabilities/parameters
        if self.discrete_actions:
            pred_action_probs = self.decoder(observations, latent)
            
            # Handle case where other_actions are indices (not one-hot)
            if other_actions.dim() == 1 or (other_actions.dim() == 2 and other_actions.shape[1] == 1):
                # Convert other_actions to indices if needed
                if other_actions.dim() == 2:
                    other_actions = other_actions.squeeze(1)
                    
                # Use gather to select the predicted probability for the actual action
                log_prob = torch.log(torch.gather(pred_action_probs, 1, other_actions.long().unsqueeze(1)).squeeze(1) + 1e-10)
            else:
                # If other_actions is already one-hot or a multi-dimensional tensor
                # We need to make sure dimensions match
                if other_actions.shape[1] != self.action_dim:
                    # Resize other_actions to match expected action dim
                    if other_actions.shape[1] > self.action_dim:
                        other_actions = other_actions[:, :self.action_dim]
                    else:
                        padding = torch.zeros(other_actions.shape[0], self.action_dim - other_actions.shape[1], device=other_actions.device)
                        other_actions = torch.cat([other_actions, padding], dim=1)
                
                # Compute log probability using dot product with one-hot encoding
                log_prob = torch.log(torch.sum(pred_action_probs * other_actions, dim=1) + 1e-10)
        else:
            pred_action_mean, pred_action_logstd = self.decoder(observations, latent)
            pred_action_std = torch.exp(pred_action_logstd)
            pred_action_dist = Normal(pred_action_mean, pred_action_std)
            log_prob = pred_action_dist.log_prob(other_actions).sum(dim=-1)
        
        # KL divergence
        # We use standard normal as prior
        kl_divergence = -0.5 * torch.sum(1 + latent_logvar - latent_mean.pow(2) - latent_std.pow(2), dim=-1)
        
        # ELBO loss
        elbo_loss = -log_prob + 0.01 * kl_divergence  # Weighted KL term
        
        return elbo_loss.mean()
    
    def compute_critic_loss(self, beliefs, latent_strategies, actions, rewards, next_beliefs, next_latent_strategies):
        """Compute the critic loss"""
        rho = self.log_rho.exp()  # Average reward estimate
        
        # Get Q-values for current state-action pairs
        if self.discrete_actions:
            current_q1 = self.q1(beliefs, latent_strategies)
            current_q2 = self.q2(beliefs, latent_strategies)
            
            # Gather Q-values for the actions taken
            # Make sure actions is the right shape for gather
            if actions.dim() == 1:
                actions_idx = actions.unsqueeze(1)
            else:
                # If somehow actions has unexpected shape, try to fix it
                actions_idx = actions.view(-1, 1)
                
            current_q1 = current_q1.gather(1, actions_idx).squeeze(1)
            current_q2 = current_q2.gather(1, actions_idx).squeeze(1)
        else:
            current_q1 = self.q1(beliefs, latent_strategies, actions).squeeze(1)
            current_q2 = self.q2(beliefs, latent_strategies, actions).squeeze(1)
        
        # Get target Q-values
        with torch.no_grad():
            if self.discrete_actions:
                # For discrete actions, get probabilities
                next_action_probs = self.policy(next_beliefs, next_latent_strategies)
                target_q1 = self.target_q1(next_beliefs, next_latent_strategies)
                target_q2 = self.target_q2(next_beliefs, next_latent_strategies)
                
                # Take minimum of Q-values
                target_q = torch.min(target_q1, target_q2)
                
                # Compute expected Q-value
                expected_q = (next_action_probs * target_q).sum(dim=1)
                
                # Add entropy term
                entropy = -torch.sum(next_action_probs * torch.log(next_action_probs + 1e-10), dim=1)
                expected_q = expected_q + self.alpha * entropy
            else:
                # For continuous actions, sample next actions
                next_actions, next_log_probs = self.policy.sample_action(next_beliefs, next_latent_strategies)
                
                # Get Q-values for next state-action pairs
                target_q1 = self.target_q1(next_beliefs, next_latent_strategies, next_actions).squeeze(1)
                target_q2 = self.target_q2(next_beliefs, next_latent_strategies, next_actions).squeeze(1)
                
                # Take minimum of Q-values
                target_q = torch.min(target_q1, target_q2)
                
                # Subtract log probability for entropy regularization
                expected_q = target_q - self.alpha * next_log_probs
            
            # Compute target using differential returns
            targets = rewards - rho + expected_q
        
        # Compute losses
        q1_loss = F.mse_loss(current_q1, targets)
        q2_loss = F.mse_loss(current_q2, targets)
        
        return q1_loss, q2_loss
    
    def compute_policy_loss(self, beliefs, latent_strategies):
        """Compute the policy loss"""
        if self.discrete_actions:
            # Get action probabilities
            action_probs = self.policy(beliefs, latent_strategies)
            
            # Get Q-values
            q1_values = self.q1(beliefs, latent_strategies)
            q2_values = self.q2(beliefs, latent_strategies)
            q_values = torch.min(q1_values, q2_values)
            
            # Compute entropy
            log_probs = torch.log(action_probs + 1e-10)
            entropy = -torch.sum(action_probs * log_probs, dim=1)
            
            # Compute expected Q-value
            expected_q = torch.sum(action_probs * q_values, dim=1)
            
            # Compute policy loss
            policy_loss = -(expected_q + self.alpha * entropy).mean()
        else:
            # Sample actions and get log probabilities
            actions, log_probs = self.policy.sample_action(beliefs, latent_strategies)
            
            # Get Q-values
            q1_values = self.q1(beliefs, latent_strategies, actions).squeeze(1)
            q2_values = self.q2(beliefs, latent_strategies, actions).squeeze(1)
            q_values = torch.min(q1_values, q2_values)
            
            # Compute policy loss
            policy_loss = -(q_values - self.alpha * log_probs).mean()
            
        return policy_loss
    
    def update_parameters(self):
        """Update all network parameters"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        observations = torch.FloatTensor(np.vstack([e.observation for e in batch])).to(device)
        latent_strategies = torch.FloatTensor(np.vstack([e.latent_strategies for e in batch])).to(device)
        
        # Handle actions based on discrete vs continuous
        if self.discrete_actions:
            # For discrete actions, ensure they're LongTensor and properly shaped
            actions = torch.LongTensor(np.array([e.action for e in batch])).to(device)
            # Make sure actions are proper indices for gather operation
            if actions.dim() > 1 and actions.shape[1] == self.action_dim:
                # If one-hot encoded, convert to indices
                actions = actions.argmax(dim=1)
        else:
            # For continuous actions, ensure they're FloatTensor
            actions = torch.FloatTensor(np.vstack([e.action for e in batch])).to(device)
            
        other_actions = torch.FloatTensor(np.vstack([e.other_actions for e in batch])).to(device)
        rewards = torch.FloatTensor(np.array([e.reward for e in batch])).to(device)
        next_observations = torch.FloatTensor(np.vstack([e.next_observation for e in batch])).to(device)
        next_latent_strategies = torch.FloatTensor(np.vstack([e.next_latent_strategies for e in batch])).to(device)
        
        # Update belief representations
        # Create properly formatted action tensors for belief updates
        if self.discrete_actions:
            # For discrete actions, create one-hot encoded tensor
            zero_actions = torch.zeros(self.batch_size, self.action_dim, device=device)
            # We'll use this as a placeholder for initial belief update
        else:
            zero_actions = torch.zeros(self.batch_size, self.action_dim, device=device)
        
        # Compute beliefs with detached gradients to avoid in-place modification issues
        with torch.no_grad():
            beliefs, _ = self.belief_network(observations, zero_actions)
            
            # For next observations, create action tensor from current actions
            if self.discrete_actions:
                # Convert discrete actions to one-hot for belief update
                action_tensor = F.one_hot(actions, self.action_dim).float()
            else:
                action_tensor = actions
            
            next_beliefs, _ = self.belief_network(next_observations, action_tensor)
            
        # Detach completely for use in separate network updates
        beliefs_detached = beliefs.detach()
        next_beliefs_detached = next_beliefs.detach()
        
        # 1. Update inference model
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        
        elbo_loss = self.compute_elbo_loss(observations, action_tensor, other_actions)
        elbo_loss.backward()
        
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        
        # 2. Update critics
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        
        q1_loss, q2_loss = self.compute_critic_loss(
            beliefs_detached.clone(), 
            latent_strategies, 
            actions, 
            rewards, 
            next_beliefs_detached.clone(), 
            next_latent_strategies
        )
        
        q1_loss.backward()
        self.q1_optimizer.step()
        
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # 3. Update policy
        self.policy_optimizer.zero_grad()
        
        policy_loss = self.compute_policy_loss(beliefs_detached.clone(), latent_strategies)
        policy_loss.backward()
        
        self.policy_optimizer.step()
        
        # 4. Update target networks
        self.soft_update(self.q1, self.target_q1)
        self.soft_update(self.q2, self.target_q2)
        
        # 5. Update belief network
        # For belief network, we'll use a fresh forward pass to avoid in-place issues
        self.belief_optimizer.zero_grad()
        
        # Do a fresh belief computation for gradient flow
        belief_outputs, _ = self.belief_network(observations, zero_actions)
        
        # Use a simplified loss for belief updates: MSE to make the belief network 
        # predict consistent beliefs (matching the Q and policy networks' expectations)
        belief_loss = F.mse_loss(belief_outputs, beliefs_detached)
        belief_loss.backward()
        
        self.belief_optimizer.step()
        
        # Store metrics
        self.policy_losses.append(policy_loss.item())
        self.q_losses.append((q1_loss.item() + q2_loss.item()) / 2)
        self.elbo_losses.append(elbo_loss.item())
        
    def soft_update(self, source, target):
        """Soft update of target network parameters"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
            
    def hard_update(self, source, target):
        """Hard update of target network parameters"""
        target.load_state_dict(source.state_dict())

    def save_models(self, path):
        """Save all model parameters"""
        torch.save({
            'belief_network': self.belief_network.state_dict(),
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'target_q1': self.target_q1.state_dict(),
            'target_q2': self.target_q2.state_dict(),
            'log_rho': self.log_rho
        }, path)
        
    def load_models(self, path):
        """Load all model parameters"""
        checkpoint = torch.load(path)
        self.belief_network.load_state_dict(checkpoint['belief_network'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.target_q1.load_state_dict(checkpoint['target_q1'])
        self.target_q2.load_state_dict(checkpoint['target_q2'])
        self.log_rho = checkpoint['log_rho']