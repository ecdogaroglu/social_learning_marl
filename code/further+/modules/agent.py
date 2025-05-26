import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.networks import EncoderNetwork, DecoderNetwork, PolicyNetwork, QNetwork, TransformerBeliefProcessor, TemporalGNN
from modules.replay_buffer import ReplayBuffer
from modules.utils import get_best_device, encode_observation

class POLARISAgent:
    """POLARIS agent for social learning with additional advantage-based Transformer training."""
    def __init__(
        self,
        agent_id,
        num_agents,
        num_states,
        observation_dim,
        action_dim,
        hidden_dim=64,
        belief_dim=64,
        latent_dim=64,
        learning_rate=1e-3,
        discount_factor=0.99,
        entropy_weight=0.01,
        kl_weight=0.01,
        target_update_rate=0.005,
        device=None,
        buffer_capacity=1000,
        max_trajectory_length=50,
        use_gnn=True,
        use_ewc=False,
        ewc_lambda=100.0,
        ewc_online=False,
        ewc_gamma=0.95
    ):
        
        # Use the best available device if none is specified
        if device is None:
            device = get_best_device()
        print(f"Using device: {device}")
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.num_states = num_states
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = device
        self.discount_factor = discount_factor
        self.entropy_weight = entropy_weight
        self.kl_weight = kl_weight
        self.target_update_rate = target_update_rate
        self.max_trajectory_length = max_trajectory_length
        self.latent_dim = latent_dim
        self.use_gnn = use_gnn
        
        # EWC parameters
        self.use_ewc = use_ewc
        self.ewc_lambda = ewc_lambda
        self.ewc_online = ewc_online
        self.ewc_gamma = ewc_gamma
        
        # EWC Fisher information and parameter storage
        self.fisher_belief = None  # Fisher for belief processor
        self.fisher_policy = None  # Fisher for policy network
        self.optimal_belief_params = None  # Optimal parameters for belief processor
        self.optimal_policy_params = None  # Optimal parameters for policy network
        self.observed_true_states = set()  # Set of observed true states
        
        # Global variables for action logits and neighbor action logits
        self.action_logits = None
        self.neighbor_action_logits = None
        
        # Initialize replay buffer with our enhanced version
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            observation_dim=observation_dim,
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            device=device,
            sequence_length=max_trajectory_length
        )
        
        # Initialize all networks
        self.belief_processor = TransformerBeliefProcessor(
            hidden_dim=belief_dim,
            action_dim=action_dim,
            device=device,
            num_belief_states=num_states,
            nhead=4,  # Number of attention heads
            num_layers=2,  # Number of transformer layers
            dropout=0.1  # Dropout rate
        ).to(device)
        
        # Initialize either the GNN or the traditional encoder-decoder
        if self.use_gnn:
            # Use the new TemporalGNN for inference learning
            self.inference_module = TemporalGNN(
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                latent_dim=latent_dim,
                num_agents=num_agents,
                device=device,
                num_belief_states=num_states,
                num_gnn_layers=2,  # Default value, will be updated later if needed
                num_attn_heads=4,  # Default value, will be updated later if needed
                dropout=0.1,
                temporal_window_size=5  # Default value, will be updated later if needed
            ).to(device)
        else:
            # Use the traditional encoder-decoder approach
            self.encoder = EncoderNetwork(
                action_dim=action_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_agents=num_agents,
                device=device,
                num_belief_states=num_states
            ).to(device)
            
            self.decoder = DecoderNetwork(
                action_dim=action_dim,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_agents=num_agents,
                num_belief_states=num_states,
                device=device
            ).to(device)
        
        self.policy = PolicyNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device
        ).to(device)
        
        self.q_network1 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        
        self.q_network2 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        
        # Create target networks
        self.target_q_network1 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        
        self.target_q_network2 = QNetwork(
            belief_dim=belief_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            device=device
        ).to(device)
        
        # Copy parameters to target networks
        self.target_q_network1.load_state_dict(self.q_network1.state_dict())
        self.target_q_network2.load_state_dict(self.q_network2.state_dict())
        
        # Average reward estimate (for average reward formulation)
        self.gain_parameter = nn.Parameter(torch.tensor(0.0, device=device))
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate
        )
        
        # Separate Transformer optimizer - only for belief processor
        self.transformer_optimizer = torch.optim.Adam(
            self.belief_processor.parameters(),
            lr=learning_rate
        )
        
        self.q_optimizer = torch.optim.Adam(
            list(self.q_network1.parameters()) + 
            list(self.q_network2.parameters()),
            lr=learning_rate
        )
        
        # Set up inference optimizer based on which inference module we're using
        if self.use_gnn:
            self.inference_optimizer = torch.optim.Adam(
                self.inference_module.parameters(),
                lr=learning_rate
            )
        else:
            self.inference_optimizer = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                lr=learning_rate
            )
            
        self.gain_optimizer = torch.optim.Adam([self.gain_parameter], lr=learning_rate)
        
        # Initialize belief and latent states with correct shapes
        self.current_belief = torch.ones(1, 1, belief_dim, device=device) / self.belief_processor.hidden_dim  # [1, batch_size=1, hidden_dim]
        self.current_latent = torch.ones(1, latent_dim, device=device) / latent_dim  # [1, latent_dim]
        self.current_mean = torch.zeros(1, latent_dim, device=device)
        self.current_logvar = torch.zeros(1, latent_dim, device=device)
        
        # Initialize belief distribution
        self.current_belief_distribution = torch.ones(1, self.belief_processor.num_belief_states, device=device) / self.belief_processor.num_belief_states
        
        # Initialize opponent belief distribution
        self.current_opponent_belief_distribution = torch.ones(1, self.num_agents, device=device) / self.num_agents

        # For tracking learning metrics
        self.action_probs_history = []
        
        # Episode tracking
        self.episode_step = 0
        
        # Track seen states for EWC
        self.seen_states = set()
        self.fisher = {}
        self.optpar = {}
    
    def observe(self, signal, neighbor_actions):
        """Update belief state based on new observation."""
        # Check if this is the first observation of the episode
        is_first_obs = (self.episode_step == 0)
        self.episode_step += 1

        # Pass observation and belief to the belief processor
        belief, belief_distribution = self.belief_processor(
            signal,
            neighbor_actions, 
            self.current_belief
        )
        
        # Store belief state with consistent shape [1, batch_size=1, hidden_dim]
        self.current_belief = belief
        
        # Store the belief distribution
        self.current_belief_distribution = belief_distribution
        
        return self.current_belief, self.current_belief_distribution
        
    def store_transition(self, observation, belief, latent, action, reward, 
                         next_observation, next_belief, next_latent, mean=None, logvar=None, neighbor_actions=None):
        """Store a transition in the replay buffer."""
        # Ensure belief states have consistent shape before storing
        belief = self.belief_processor.standardize_belief_state(belief)
        next_belief = self.belief_processor.standardize_belief_state(next_belief)
        
        self.replay_buffer.push(
            observation, belief, latent, action, reward,
            next_observation, next_belief, next_latent, mean, logvar, neighbor_actions
        )
        
    def set_train_mode(self):
        """Set all networks to training mode."""
        self.belief_processor.train()
        if self.use_gnn:
            self.inference_module.train()
        else:
            self.encoder.train()
            self.decoder.train()
        self.policy.train()
        self.q_network1.train()
        self.q_network2.train()
        self.target_q_network1.train()
        self.target_q_network2.train()
        
    def set_eval_mode(self):
        """Set all networks to evaluation mode."""
        self.belief_processor.eval()
        if self.use_gnn:
            self.inference_module.eval()
        else:
            self.encoder.eval()
            self.decoder.eval()
        self.policy.eval()
        self.q_network1.eval()
        self.q_network2.eval()
        self.target_q_network1.eval()
        self.target_q_network2.eval()
        
    def reset_internal_state(self):
        """Reset the agent's internal state (belief and latent variables)."""
        # Use zeros for a complete reset with correct shapes
        self.current_belief = torch.zeros(1, 1, self.belief_processor.hidden_dim, device=self.device)  # [1, batch_size=1, hidden_dim]
        self.current_latent = torch.zeros(1, self.latent_dim, device=self.device)
        self.current_mean = torch.zeros(1, self.latent_dim, device=self.device)
        self.current_logvar = torch.zeros(1, self.latent_dim, device=self.device)
        
        # Reset belief distribution 
        self.current_belief_distribution = torch.ones(1, self.belief_processor.num_belief_states, device=self.device) / self.belief_processor.num_belief_states
        
        # Detach all tensors to ensure no gradient flow between episodes
        self.current_belief = self.current_belief.detach()
        self.current_latent = self.current_latent.detach()
        self.current_mean = self.current_mean.detach()
        self.current_logvar = self.current_logvar.detach()
        if self.current_belief_distribution is not None:
            self.current_belief_distribution = self.current_belief_distribution.detach()
        if hasattr(self, 'current_opponent_belief_distribution') and self.current_opponent_belief_distribution is not None:
            self.current_opponent_belief_distribution = self.current_opponent_belief_distribution.detach()
        
        # If using GNN, reset its temporal memory
        if self.use_gnn:
            self.inference_module.reset_memory()
    
    def infer_latent(self, signal, neighbor_actions, reward, next_signal):
        """Infer latent state of neighbors based on our observations which already contain neighbor actions."""

        # Convert reward to tensor
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32).to(self.device).squeeze(1)

        if self.use_gnn:
            # Use the GNN for inference
            mean, logvar, opponent_belief_distribution = self.inference_module(
                signal,
                neighbor_actions,
                reward_tensor,
                next_signal,
                self.current_latent
            )
        else:
            # Use the traditional encoder
            mean, logvar, opponent_belief_distribution = self.encoder(
                signal,
                neighbor_actions,
                reward_tensor,
                next_signal,
                self.current_latent
            )
        
        # Sample based on reparameterized distribution 
        # Ref: https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
        # Add numerical stability safeguards
        # First, clamp logvar to prevent extreme values
        logvar = torch.clamp(logvar, min=-20.0, max=2.0)
        
        # Then calculate variance with better safety measures
        var = torch.exp(0.5 * logvar)
        epsilon = 1e-6
        var = torch.clamp(var, min=epsilon, max=1e6)  # Also add maximum bound
        distribution = torch.distributions.Normal(mean, var)
        new_latent = distribution.rsample()
            
        # Store the current latent, mean, logvar, and opponent belief distribution
        self.current_latent = new_latent.unsqueeze(0)
        self.current_mean = mean
        self.current_logvar = logvar
        self.current_opponent_belief_distribution = opponent_belief_distribution
        
        return new_latent
    
    def select_action(self):
        """Select action based on current belief and latent."""

        # Calculate fresh action logits for action selection
        action_logits = self.policy(self.current_belief, self.current_latent)

        # Store a detached copy for caching
        self.action_logits = action_logits.detach()

        # Convert to probabilities
        action_probs = F.softmax(action_logits, dim=-1)

        # Store probability of incorrect action for learning rate calculation
        self.action_probs_history.append(action_probs.squeeze(0).detach().cpu().numpy())
        
        # Sample action from distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample().item()
        
        # Alternatively, use argmax for deterministic action selection
        #action = action_probs.argmax(dim=-1).item()

        
        return action, action_probs.squeeze(0).detach().cpu().numpy()
    
    def train(self, batch_size=32, sequence_length=32):
        """Train the agent using sequential data from the replay buffer."""
        # Sample sequential data from the replay buffer
        batch_sequences = self.replay_buffer.sample(batch_size, sequence_length, mode="sequence")
        
        # Update networks using sequential data
        return self.update(batch_sequences)
    
    def update(self, batch):
        """Update the agent's model parameters based on experiences."""
        # Extract data from batch
        observations = batch['observations']
        neighbor_actions = batch['neighbor_actions']
        beliefs = batch['beliefs']
        latents = batch['latents']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        next_beliefs = batch['next_beliefs']
        next_latents = batch['next_latents']
        
        # Forward pass through policy network to get action distribution
        action_logits = self.policy(beliefs, latents)
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        
        # Compute log probability of actions taken
        action_log_probs = action_dist.log_prob(actions)
        
        # Compute entropy for exploration
        entropy = action_dist.entropy().mean()
        
        # Calculate policy loss using REINFORCE with rewards
        policy_loss = -(action_log_probs * rewards).mean()
        
        # Compute value loss if we have a value network
        value_loss = torch.zeros(1, device=self.device)
        if hasattr(self, 'value_network') and self.value_network is not None:
            predicted_values = self.value_network(beliefs, latents).squeeze()
            value_loss = F.mse_loss(predicted_values, rewards)
        
        # Compute reconstruction loss for latent space
        kl_loss = torch.zeros(1, device=self.device)
        if hasattr(self, 'inference_module') and self.inference_module is not None:
            means = batch.get('means', None)
            logvars = batch.get('logvars', None)
            if means is not None and logvars is not None:
                # KL divergence to standard normal
                kl_loss = -0.5 * torch.mean(1 + logvars - means.pow(2) - logvars.exp())
        
        # Entropy bonus to encourage exploration
        entropy_loss = -self.entropy_weight * entropy
        
        # Calculate total loss
        total_loss = policy_loss + value_loss + entropy_loss + self.kl_weight * kl_loss
        
        # Add EWC penalty if enabled
        ewc_loss = torch.tensor(0.0, device=self.device)
        if self.use_ewc and hasattr(self, 'fisher') and self.fisher and hasattr(self, 'optpar') and self.optpar:
            for name, param in self.named_parameters():
                if name in self.fisher and name in self.optpar and param.requires_grad:
                    # Calculate the EWC penalty for this parameter
                    ewc_loss += torch.sum(self.fisher[name] * (param - self.optpar[name]) ** 2)
            
            # Scale the EWC loss by lambda/2
            ewc_loss = 0.5 * self.ewc_lambda * ewc_loss
            
            # Add to total loss
            total_loss += ewc_loss
        
        # Update parameters
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)  # Gradient clipping
        self.policy_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'kl_loss': kl_loss.item(),
            'ewc_loss': ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else 0.0,
            'total_loss': total_loss.item(),
        }
    
    def _update_inference(self, signals, neighbor_actions, next_signals, next_latents, means, logvars):
        """Update inference module with FURTHER-style temporal KL."""
        
        if self.use_gnn:
            # For GNN inference module
            # We need dummy rewards for the GNN forward pass
            batch_size = signals.size(0)
            dummy_rewards = torch.zeros(batch_size, 1, device=self.device)
            
            # Forward pass through GNN to get new distribution parameters
            # Note: we detach next_latents to avoid gradients flowing back through the target network
            new_means, new_logvars, _ = self.inference_module(
                signals, 
                neighbor_actions, 
                dummy_rewards, 
                next_signals
            )
            
            # Generate action predictions using the current batch
            batch_neighbor_logits = self.inference_module.predict_actions(signals, next_latents.detach())
            
            # Reshape batch_neighbor_logits if needed for cross entropy
            if batch_neighbor_logits.dim() == 3:
                batch_size, seq_len, action_dim = batch_neighbor_logits.shape
                batch_neighbor_logits = batch_neighbor_logits.view(batch_size * seq_len, action_dim)
                neighbor_actions_reshaped = neighbor_actions.view(-1)
            else:
                neighbor_actions_reshaped = neighbor_actions
                
            # Calculate reconstruction loss
            recon_loss = F.cross_entropy(batch_neighbor_logits, neighbor_actions_reshaped)
            
            # Calculate temporal KL divergence with numerical stability
            kl_loss = self._calculate_temporal_kl_divergence(new_means, new_logvars)
        else:
            # For traditional encoder-decoder
            # Generate fresh neighbor action logits for the batch
            # Use the decoder directly on the batch
            batch_neighbor_logits = self.decoder(signals, next_latents)
        
            # Reshape if needed for cross entropy
            if batch_neighbor_logits.dim() == 3:
                batch_size, seq_len, action_dim = batch_neighbor_logits.shape
                batch_neighbor_logits = batch_neighbor_logits.view(batch_size * seq_len, action_dim)
                neighbor_actions_reshaped = neighbor_actions.view(-1)
            else:
                neighbor_actions_reshaped = neighbor_actions
            
            # Calculate reconstruction loss
            recon_loss = F.cross_entropy(batch_neighbor_logits, neighbor_actions_reshaped)
            
            # Calculate temporal KL divergence (FURTHER-style)
            kl_loss = self._calculate_temporal_kl_divergence(means, logvars)
        
        # Total loss
        inference_loss = recon_loss + kl_loss
        
        # Update networks
        self.inference_optimizer.zero_grad()
        inference_loss.backward()
        
        if self.use_gnn:
            torch.nn.utils.clip_grad_norm_(
                self.inference_module.parameters(), 
                max_norm=1.0
            )
        else:
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()), 
                max_norm=1.0
            )
            
        self.inference_optimizer.step()
        
        return inference_loss.item()
    
    def _calculate_temporal_kl_divergence(self, means_seq, logvars_seq):
        """Calculate KL divergence between sequential latent states (temporal smoothing)."""

        # KL(N(mu,E), N(m, S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m - mu)^T S^-1 (m - mu)))
        # Ref: https://github.com/lmzintgraf/varibad/blob/master/vae.py
        # Ref: https://github.com/dkkim93/further/blob/main/algorithm/further/agent.py

        kl_first_term = torch.sum(logvars_seq[:-1, :], dim=-1) - torch.sum(logvars_seq[1:, :], dim=-1)
        kl_second_term = self.latent_dim
        kl_third_term = torch.sum(1. / torch.exp(logvars_seq[:-1, :]) * torch.exp(logvars_seq[1:, :]), dim=-1)
        kl_fourth_term = (means_seq[:-1, :] - means_seq[1:, :]) / torch.exp(logvars_seq[:-1, :]) * (means_seq[:-1, :] - means_seq[1:, :])
        kl_fourth_term = kl_fourth_term.sum(dim=-1)
        
        kl = 0.5 * (kl_first_term - kl_second_term + kl_third_term + kl_fourth_term)

        return self.kl_weight * torch.mean(kl)
    
    def _update_critics(self, signals, neighbor_actions, beliefs, latents, actions, next_neighbor_actions, rewards, next_signals, next_beliefs, next_latents):
        """Update Q-networks."""
        # Get current Q-values
        q1 = self.q_network1(beliefs, latents, neighbor_actions).gather(1, actions.unsqueeze(1))
        q2 = self.q_network2(beliefs, latents, neighbor_actions).gather(1, actions.unsqueeze(1))
        
        # Compute next action probabilities
        with torch.no_grad():
            # Calculate fresh action logits for critic update
            next_action_logits = self.policy(next_beliefs, next_latents)
            next_action_probs = F.softmax(next_action_logits, dim=1)
            next_log_probs = F.log_softmax(next_action_logits, dim=1)
            entropy = -torch.sum(next_action_probs * next_log_probs, dim=1, keepdim=True)
            
            # Compute Q-values with predicted next neighbor actions
            next_q1 = self.target_q_network1(next_beliefs, next_latents, next_neighbor_actions)
            next_q2 = self.target_q_network2(next_beliefs, next_latents, next_neighbor_actions)
            
            # Take minimum
            next_q = torch.min(next_q1, next_q2)
            
            # Expected Q-value
            expected_q = (next_action_probs * next_q).sum(dim=1, keepdim=True)
            
            # Add entropy
            expected_q = expected_q + self.entropy_weight * entropy
            
            # Compute target
            if self.discount_factor > 0:  # Discounted return
                target_q = rewards + self.discount_factor * expected_q
            else:  # Average reward
                target_q = rewards - self.gain_parameter + expected_q
        
        # Compute loss
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss
        
        # Update networks
        self.q_optimizer.zero_grad()
        if self.discount_factor == 0:  # Only update gain parameter for average reward
            self.gain_optimizer.zero_grad()
        
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q_network1.parameters()) + list(self.q_network2.parameters()),
            max_norm=1.0
        )
        self.q_optimizer.step()
        
        if self.discount_factor == 0:
            self.gain_optimizer.step()
        
        return q_loss.item()
    
    def _update_transformer(self, signals, neighbor_actions, beliefs, 
                next_signals):
        """
        Update belief processor (Transformer) by optimizing for next state prediction.
        
        Args:
            signals: Current signals/observations
            neighbor_actions: Current neighbor actions
            beliefs: Current belief states
            latents: Current latent states
            actions: Actions taken
            next_signals: Next signals/observations
        
        Returns:
            float: Loss value
        """
        # Process current signals and neighbor actions through Transformer to get next belief
        _, belief_distributions = self.belief_processor(
            signals, neighbor_actions, beliefs
        )
        
        # Calculate log likelihood loss of next signal given current belief
        transformer_loss = F.binary_cross_entropy(
            belief_distributions, next_signals, reduction='none'
        ).sum(dim=1).mean()
        
        # Calculate EWC loss for belief processor if enabled
        if self.use_ewc and self.fisher_belief is not None:
            belief_ewc_loss, _ = self.calculate_fisher_matrices(self.replay_buffer)
            transformer_loss += self.ewc_lambda * belief_ewc_loss
                
        # Update Transformer parameters
        self.transformer_optimizer.zero_grad()
        transformer_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.belief_processor.parameters(), max_norm=1.0)
        self.transformer_optimizer.step()
        
        return transformer_loss.item()
    
    def _update_policy(self, beliefs, latents, actions, neighbor_actions):
        """Update policy network and calculate advantage for Transformer training.
        
        Returns:
            Tuple of (policy_loss_value, advantage)
        """
        # Generate fresh action logits for the batch
        action_logits = self.policy(beliefs, latents)
        
        # Calculate probabilities from the logits
        action_probs = F.softmax(action_logits, dim=1)
        log_probs = F.log_softmax(action_logits, dim=1)
        
        # Compute entropy
        entropy = -torch.sum(action_probs * log_probs, dim=1, keepdim=True)
        
        # Get Q-values
        with torch.no_grad():
            q1 = self.q_network1(beliefs, latents, neighbor_actions)
            q2 = self.q_network2(beliefs, latents, neighbor_actions)
            q = torch.min(q1, q2)
        
        # Compute expected Q-value
        expected_q = torch.sum(action_probs * q, dim=1, keepdim=True)
        
        # Policy loss is negative of expected Q-value plus entropy
        policy_loss = -(expected_q + self.entropy_weight * entropy).mean()
        
        # Calculate advantage for Transformer training
        # Advantage is Q-value of taken action minus expected Q-value (baseline)
        q_actions = q.gather(1, actions.unsqueeze(1))
        advantage = q_actions - expected_q.detach()
        
        # Calculate EWC loss for policy network if enabled
        if self.use_ewc and self.fisher_policy is not None:
            _, policy_ewc_loss = self.calculate_fisher_matrices(self.replay_buffer)
            policy_loss += self.ewc_lambda * policy_ewc_loss
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        
        return policy_loss.item(), advantage
    
    def _update_targets(self):
        """Update target networks."""
        for target_param, param in zip(self.target_q_network1.parameters(), self.q_network1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.target_update_rate) + 
                param.data * self.target_update_rate
            )
        
        for target_param, param in zip(self.target_q_network2.parameters(), self.q_network2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.target_update_rate) + 
                param.data * self.target_update_rate
            )
    
    def save(self, path):
        """Save the agent's networks to a file."""
        checkpoint = {
            'belief_processor': self.belief_processor.state_dict(),
            'policy': self.policy.state_dict(),
            'q_network1': self.q_network1.state_dict(),
            'q_network2': self.q_network2.state_dict(),
            'target_q_network1': self.target_q_network1.state_dict(),
            'target_q_network2': self.target_q_network2.state_dict(),
            'gain_parameter': self.gain_parameter,
            'use_gnn': self.use_gnn,
            'use_ewc': self.use_ewc
        }
        
        # Save the appropriate inference module
        if self.use_gnn:
            checkpoint['inference_module'] = self.inference_module.state_dict()
        else:
            checkpoint['encoder'] = self.encoder.state_dict()
            checkpoint['decoder'] = self.decoder.state_dict()
        
        # Save EWC data if enabled
        if self.use_ewc:
            checkpoint['fisher_belief'] = self.fisher_belief
            checkpoint['fisher_policy'] = self.fisher_policy
            checkpoint['optimal_belief_params'] = self.optimal_belief_params
            checkpoint['optimal_policy_params'] = self.optimal_policy_params
            checkpoint['observed_true_states'] = list(self.observed_true_states)
        
        torch.save(checkpoint, path)
    
    def load(self, path, evaluation_mode=False):
        """
        Load agent model.
        
        Args:
            path: Path to the saved model
            evaluation_mode: If True, sets the model to evaluation mode after loading
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Check if we're loading a GRU model into a Transformer model
        is_gru_to_transformer = False
        try:
            # Try to load belief processor (may fail if architecture changed from GRU to Transformer)
            self.belief_processor.load_state_dict(checkpoint['belief_processor'])
        except RuntimeError as e:
            print(f"Warning: Could not load belief processor due to architecture change: {e}")
            print("Using the new Transformer belief processor with initialized weights.")
            print("Attempting to transfer knowledge from GRU to Transformer...")
            is_gru_to_transformer = True
            
            # Try to transfer knowledge from GRU to Transformer
            self._transfer_gru_to_transformer_knowledge(checkpoint)
            
        # Load other components that should be compatible
        try:
            if self.use_gnn:
                self.inference_module.load_state_dict(checkpoint['inference_module'])
            else:
                self.encoder.load_state_dict(checkpoint['encoder'])
                self.decoder.load_state_dict(checkpoint['decoder'])
            self.policy.load_state_dict(checkpoint['policy'])
            self.q_network1.load_state_dict(checkpoint['q_network1'])
            self.q_network2.load_state_dict(checkpoint['q_network2'])
            self.target_q_network1.load_state_dict(checkpoint['target_q_network1'])
            self.target_q_network2.load_state_dict(checkpoint['target_q_network2'])
            self.gain_parameter.data = checkpoint['gain_parameter']
        except RuntimeError as e:
            print(f"Warning: Could not load some components due to compatibility issues: {e}")
            
        # Load EWC data if present and enabled
        if self.use_ewc and 'fisher_belief' in checkpoint and 'fisher_policy' in checkpoint:
            self.fisher_belief = checkpoint['fisher_belief']
            self.fisher_policy = checkpoint['fisher_policy']
            self.optimal_belief_params = checkpoint['optimal_belief_params']
            self.optimal_policy_params = checkpoint['optimal_policy_params']
            
            # Load observed true states
            if 'observed_true_states' in checkpoint:
                self.observed_true_states = set(checkpoint['observed_true_states'])
            
            print(f"Loaded EWC data with {len(self.observed_true_states)} observed true states")
            
        # Set to evaluation mode if requested
        if evaluation_mode:
            self.set_eval_mode()
        
        print(f"Model loaded from {path}")
        
        return is_gru_to_transformer

    def _transfer_gru_to_transformer_knowledge(self, checkpoint):
        """
        Transfer knowledge from a GRU model to a Transformer model.
        This helps preserve some of the learned knowledge when switching architectures.
        """
        try:
            # The most important part to transfer is the belief head weights
            # which map from hidden state to belief distribution
            if 'belief_processor' in checkpoint:
                gru_state_dict = checkpoint['belief_processor']
                
                # Transfer belief head weights if they have the same dimensions
                if 'belief_head.weight' in gru_state_dict and gru_state_dict['belief_head.weight'].size() == self.belief_processor.belief_head.weight.size():
                    self.belief_processor.belief_head.weight.data.copy_(gru_state_dict['belief_head.weight'])
                    self.belief_processor.belief_head.bias.data.copy_(gru_state_dict['belief_head.bias'])
                    print("Successfully transferred belief head weights from GRU to Transformer.")
                    
                # We can also try to initialize the input projection with GRU input weights
                if 'gru.weight_ih_l0' in gru_state_dict:
                    # The input weights of GRU can be used to initialize part of the input projection
                    gru_input_weights = gru_state_dict['gru.weight_ih_l0']
                    input_dim = min(gru_input_weights.size(1), self.belief_processor.input_projection.weight.size(1))
                    output_dim = min(gru_input_weights.size(0) // 3, self.belief_processor.input_projection.weight.size(0))
                    
                    # Copy the reset gate weights (first third of GRU weights)
                    self.belief_processor.input_projection.weight.data[:output_dim, :input_dim].copy_(
                        gru_input_weights[:output_dim, :input_dim]
                    )
                    print("Partially initialized Transformer input projection with GRU weights.")
                    
        except Exception as e:
            print(f"Error during knowledge transfer: {e}")
            print("Continuing with randomly initialized Transformer.")

    def get_belief_state(self):
        """Return the current belief state.
        
        Returns:
            belief: Current belief state tensor with shape [1, batch_size=1, hidden_dim]
        """
        return self.current_belief
    
    def get_latent_state(self):
        """Return the current latent state.
        
        Returns:
            latent: Current latent state tensor
        """
        return self.current_latent
    
    def get_belief_distribution(self):
        """Return the current belief distribution.
        
        Returns:
            belief_distribution: Current belief distribution tensor or None if not available
        """
        return self.current_belief_distribution
    
    def get_latent_distribution_params(self):
        """Return the current latent distribution parameters (mean and logvar).
        
        Returns:
            mean: Current mean of the latent distribution
            logvar: Current log variance of the latent distribution
        """
        return self.current_mean, self.current_logvar
    
    def get_opponent_belief_distribution(self):
        """Return the current opponent belief distribution.
        
        Returns:
            opponent_belief_distribution: Current opponent belief distribution tensor or None if not available
        """
        return self.current_opponent_belief_distribution if hasattr(self, 'current_opponent_belief_distribution') else None
        
    def end_episode(self):
        """
        Backward compatibility method - does nothing in the continuous version.
        The internal state is maintained across what would have been episode boundaries.
        """
        pass
    
    def observe_new_true_state(self, true_state):
        """
        Check if this is a new true state and record it for EWC.
        Returns True if this is a new state, False otherwise.
        """
        if not self.use_ewc:
            return False
        
        # Convert true_state to hashable type if needed
        if isinstance(true_state, np.ndarray):
            state_key = tuple(true_state.flatten())
        else:
            state_key = true_state
        
        # Check if we've seen this state before
        if state_key in self.seen_states:
            return False
        else:
            # Record the new state
            self.seen_states.add(state_key)
            return True
    
    def calculate_fisher_matrices(self, replay_buffer):
        """
        Calculate Fisher Information matrices for EWC.
        This should be called after learning a task to prepare for the next one.
        
        Args:
            replay_buffer: The replay buffer containing transitions
        """
        if not self.use_ewc:
            return
        
        # Put all networks in evaluation mode temporarily
        self.set_eval_mode()
        
        # Initialize/reset Fisher matrices
        self.fisher = {}
        
        # Sample a batch from the replay buffer if there's enough data
        batch_size = min(256, len(replay_buffer))
        if batch_size == 0:
            print(f"Agent {self.agent_id}: No samples in replay buffer for Fisher calculation")
            return
        
        batch = replay_buffer.sample(batch_size)
        
        # Get relevant data from batch
        observations = batch['observations']
        action_observations = batch['neighbor_actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        
        # Initialize Fisher matrices for each parameter
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param.data)
        
        # Save current parameters
        self.optpar = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.optpar[name] = param.data.clone()
        
        # Compute Fisher matrices
        for i in range(batch_size):
            # Get single sample
            obs = observations[i:i+1]
            acts = action_observations[i:i+1]
            rew = rewards[i:i+1]
            next_obs = next_observations[i:i+1]
            
            # Forward pass
            self.zero_grad()
            
            # Calculate belief update
            next_belief, next_dist = self.observe(obs, acts)
            
            # Calculate log probabilities of belief distribution
            log_probs = torch.log(next_dist + 1e-10)
            
            # Compute loss for Fisher calculation (negative log likelihood)
            loss = -torch.sum(next_dist * log_probs)
            
            # Backward pass
            loss.backward()
            
            # Accumulate squared gradients in Fisher matrices
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2 / batch_size
        
        # For online EWC, we need to update the existing Fisher matrices
        if self.ewc_online and hasattr(self, 'prev_fisher'):
            for name in self.fisher.keys():
                if name in self.prev_fisher:
                    self.fisher[name] = self.ewc_gamma * self.prev_fisher[name] + (1 - self.ewc_gamma) * self.fisher[name]
        
        # Save current Fisher for online EWC
        if self.ewc_online:
            self.prev_fisher = {name: self.fisher[name].clone() for name in self.fisher}
        
        # Put networks back in training mode if we were training
        if self.training:
            self.set_train_mode()

    def calculate_ewc_loss(self):
        """Calculate the EWC loss based on saved Fisher matrices and optimal parameters."""
        if not self.use_ewc:
            return torch.tensor(0.0, device=self.device)
        
        if not hasattr(self, 'fisher') or not self.fisher or not hasattr(self, 'optpar') or not self.optpar:
            return torch.tensor(0.0, device=self.device)
        
        ewc_loss = torch.tensor(0.0, device=self.device)
        
        # Calculate the EWC penalty for all parameters
        for name, param in self.named_parameters():
            if name in self.fisher and name in self.optpar and param.requires_grad:
                # Calculate squared distance weighted by Fisher information
                ewc_loss += torch.sum(self.fisher[name] * (param - self.optpar[name]) ** 2)
        
        # Scale by lambda/2
        ewc_loss = 0.5 * self.ewc_lambda * ewc_loss
        
        return ewc_loss
    