import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Tuple

from environment import MultiAgentEnvironment
from nn import MultiAgentRNN
from metrics import MultiAgentMetricsTracker

class DecentralizedActor(nn.Module):
    """Decentralized actor network for each agent."""
    def __init__(self, num_agents: int, hidden_size: int, action_size: int = 2):
        super(DecentralizedActor, self).__init__()
        self.rnn = MultiAgentRNN(num_agents, hidden_size=hidden_size)
        self.policy = nn.Linear(hidden_size, action_size)
    
    def forward(self, signal: torch.Tensor, actions: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.rnn(signal, actions, hidden)
        action_probs = F.softmax(self.policy(hidden), dim=1)
        return action_probs, hidden
    
    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(1, self.rnn.hidden_size)

class CentralizedCritic(nn.Module):
    """Centralized critic network with per-agent value predictions."""
    def __init__(self, num_agents: int, hidden_size: int):
        super(CentralizedCritic, self).__init__()
        
        # Input: Concatenated hidden states from all agents
        self.input_size = hidden_size * num_agents
        self.num_agents = num_agents
        
        # Larger network to process joint state
        self.fc1 = nn.Linear(self.input_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        # Output separate value for each agent
        self.value = nn.Linear(hidden_size, num_agents)
        
    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # Concatenate all agents' hidden states
        x = torch.cat([h.detach() for h in hidden_states], dim=1)
        
        # Forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        state_values = self.value(x)  # Returns value for each agent
        
        return state_values

class MultiAgentTrainer:
    """CTDE trainer for multi-agent social learning."""
    def __init__(self, num_agents: int, hidden_size: int = 64, 
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3):
        self.num_agents = num_agents
        
        # Initialize actors and critic
        self.actors = [DecentralizedActor(num_agents, hidden_size) for _ in range(num_agents)]
        self.critic = CentralizedCritic(num_agents, hidden_size)
        
        # Initialize optimizers
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=lr_actor) 
                               for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Hidden states
        self.hidden_states = [actor.init_hidden() for actor in self.actors]
        
        self.gamma = 0.99
    
    def select_actions(self, signals: List[torch.Tensor], 
                      prev_actions: torch.Tensor) -> Tuple[List[int], List[torch.Tensor]]:
        actions = []
        log_probs = []
        true_action_rates = []

        for i, actor in enumerate(self.actors):
            action_probs, self.hidden_states[i] = actor(signals[i], prev_actions, 
                                                      self.hidden_states[i])
            dist = Categorical(action_probs)
            action = dist.sample()
            actions.append(action.item())
            log_probs.append(dist.log_prob(action))
            true_action_rates.append(dist.probs[0][0].item())

        return actions, log_probs, true_action_rates
    
    def update(self, signals: List[torch.Tensor], actions: List[int], 
               log_probs: List[torch.Tensor], observed_rewards: List[float], true_rewards: List[float], 
               next_signals: List[torch.Tensor], prev_actions: torch.Tensor):
        
        # Convert rewards to tensor
        observed_rewards_tensor = torch.tensor(observed_rewards, dtype=torch.float32).view(-1)
        true_rewards_tensor = torch.tensor(true_rewards, dtype=torch.float32).view(-1)

        # Get current state values for all agents
        current_value = self.critic(self.hidden_states)
        
        # Get next state values
        with torch.no_grad():
            next_hidden_states = []
            for i, actor in enumerate(self.actors):
                _, hidden = actor(next_signals[i], prev_actions, self.hidden_states[i].detach())
                next_hidden_states.append(hidden)
            
            next_value = self.critic(next_hidden_states)
        
        # Compute advantage for each agent separately
        advantages = []
        for i in range(self.num_agents):
            advantage = (observed_rewards_tensor[i] + 
                        self.gamma * next_value[:, i].detach() - 
                        current_value[:, i].detach())
            advantages.append(advantage)
        
        # Update actors using their specific advantages
        actor_losses = []
        for i in range(self.num_agents):
            policy_loss = -log_probs[i] * advantages[i]
            actor_losses.append(policy_loss)
        
        # Update each actor
        for i in range(self.num_agents):
            self.actor_optimizers[i].zero_grad()
            actor_losses[i].backward(retain_graph=True)
            self.actor_optimizers[i].step()
        
        # Update critic using per-agent value targets
        value_targets = true_rewards_tensor + self.gamma * next_value.detach()
        value_loss = F.mse_loss(current_value, value_targets, reduction="sum")
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        
        # Update hidden states
        for i in range(self.num_agents):
            with torch.no_grad():
                _, self.hidden_states[i] = self.actors[i](signals[i], prev_actions, 
                                                        self.hidden_states[i].detach())

def train_ctde(num_agents: int = 4, num_steps: int = 10000, 
                     signal_accuracy: float = 0.75):
    env = MultiAgentEnvironment(num_agents, signal_accuracy)
    trainer = MultiAgentTrainer(num_agents)
    metrics = MultiAgentMetricsTracker(num_agents, num_steps)
    
    # Get initial signals and actions
    signals = env.get_signals()
    prev_actions = torch.zeros(1, num_agents)

    print(f"True state: {env.true_state}")
    # Training loop
    for t in range(num_steps):
        # Select actions
        actions, log_probs, true_action_rates = trainer.select_actions(signals, prev_actions)
        
        # Environment step
        next_signals, observed_rewards, true_rewards, mistakes = env.step(actions)
        
        # Update trainer
        trainer.update(signals, actions, log_probs, observed_rewards, true_rewards, next_signals, prev_actions)
        
        # Update metrics
        metrics.add_mistakes(mistakes)
        metrics.add_action_rate(true_action_rates)
        metrics.update_metrics()
        
        # Update states
        signals = next_signals
        prev_actions = torch.tensor([[float(a) for a in actions]])
        
        # Print progress
        if (t + 1) % 1000 == 0:
            print(f"\nStep {t + 1}")
            for i in range(num_agents):
                print(f"Agent {i+1}:")
                print(f"  Mistake Rate: {metrics.mistake_rates[i][-1]:.3f}")
                print(f"  Learning Rate: {metrics.learning_rates[i][-1]:.3f}")
            print("--------------------")
    
    return trainer, metrics


