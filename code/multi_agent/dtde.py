import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple

from environment import MultiAgentEnvironment
from nn import MultiAgentRNN
from metrics import MultiAgentMetricsTracker

class DecentralizedActor(nn.Module):
    """Actor network for each agent."""
    def __init__(self, hidden_size: int):
        super(DecentralizedActor, self).__init__()
        self.policy = nn.Linear(hidden_size, 2)
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.policy(hidden), dim=1)

class DecentralizedCritic(nn.Module):
    """Critic network for each agent."""
    def __init__(self, hidden_size: int):
        super(DecentralizedCritic, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(hidden))
        x = F.relu(self.fc2(x))
        return self.value(x)

class DecentralizedAgent:
    """Complete decentralized agent combining RNN, actor, and critic."""
    def __init__(self, num_agents: int, hidden_size: int = 64,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3):
        self.rnn = MultiAgentRNN(num_agents, hidden_size)
        self.actor = DecentralizedActor(hidden_size)
        self.critic = DecentralizedCritic(hidden_size)
        
        # Optimizers
        self.optimizer_rnn = torch.optim.Adam(self.rnn.parameters(), lr=lr_actor)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.hidden = torch.zeros(1, hidden_size)
        self.h_prev = torch.zeros(1, hidden_size)  # Track previous hidden state
        self.gamma = 0.99
    
    def select_action(self, signal: torch.Tensor, actions: torch.Tensor) -> Tuple[int, torch.Tensor]:
        # Save previous hidden state before updating
        self.h_prev = self.hidden.clone()

        with torch.no_grad():
            self.hidden = self.rnn(signal, actions, self.h_prev)
            action_probs = self.actor(self.hidden)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            true_action_rate = dist.probs[0][0].item()

        return action.item(), log_prob, true_action_rate
    
    def update(self, signal: torch.Tensor, action: int, log_prob: torch.Tensor,
               reward: float, next_signal: torch.Tensor, current_actions: torch.Tensor):
        # Compute current hidden state using previous hidden state (h_prev)
        with torch.no_grad():
            current_hidden = self.rnn(signal, current_actions, self.h_prev)
        
        # Compute current value
        current_value = self.critic(current_hidden)
        
        # Compute next hidden state and next value
        with torch.no_grad():
            next_hidden = self.rnn(next_signal, current_actions, current_hidden)
            next_value = self.critic(next_hidden)
            value_target = torch.tensor([reward], dtype=torch.float32) + self.gamma * next_value
            advantage = (value_target - current_value).detach()
        
        # Update critic
        self.optimizer_critic.zero_grad()
        value_loss = F.mse_loss(current_value, value_target)
        value_loss.backward()
        self.optimizer_critic.step()
        
        # Recompute action probabilities for policy gradient WITH GRADIENTS
        action_probs = self.actor(current_hidden)  # Removed torch.no_grad()
        dist = Categorical(action_probs)
        policy_loss = -dist.log_prob(torch.tensor(action)) * advantage
        
        # Update actor
        self.optimizer_actor.zero_grad()
        policy_loss.backward()  # Now has valid gradients
        self.optimizer_actor.step()
        
        # Update RNN
        self.optimizer_rnn.zero_grad()
        current_hidden_with_grad = self.rnn(signal, current_actions, self.h_prev)
        action_probs_with_grad = self.actor(current_hidden_with_grad)
        dist_with_grad = Categorical(action_probs_with_grad)
        rnn_loss = -dist_with_grad.log_prob(torch.tensor(action)) * advantage
        rnn_loss.backward()
        self.optimizer_rnn.step()

def train_dtde(num_agents: int = 4, num_steps: int = 10000,
                     signal_accuracy: float = 0.75):
    env = MultiAgentEnvironment(num_agents, signal_accuracy)
    agents = [DecentralizedAgent(num_agents) for _ in range(num_agents)]
    metrics = MultiAgentMetricsTracker(num_agents, num_steps)
    
    # Get initial signals and actions
    signals = env.get_signals()
    prev_actions = torch.zeros(1, num_agents)
    
    print(f"True state: {env.true_state}")
    # Training loop
    for t in range(num_steps):
        # Select actions
        actions = []
        log_probs = []
        true_action_rates = []

        for i, agent in enumerate(agents):
            action, log_prob, true_action_rate = agent.select_action(signals[i], prev_actions)
            actions.append(action)
            log_probs.append(log_prob)
            true_action_rates.append(true_action_rate)
            
        # Environment step
        next_signals, observed_rewards, _, mistakes = env.step(actions)
        
        # Convert current actions to tensor
        current_actions_tensor = torch.tensor([actions], dtype=torch.float32)
        
        # Update agents
        for i, agent in enumerate(agents):
            agent.update(signals[i], actions[i], log_probs[i], observed_rewards[i],
                        next_signals[i], current_actions_tensor)
        
        # Update metrics
        metrics.add_mistakes(mistakes)
        metrics.add_action_rate(true_action_rates)
        metrics.update_metrics()
        
        # Update states
        signals = next_signals
        prev_actions = current_actions_tensor
        
        # Print progress
        if (t + 1) % 1000 == 0:
            print(f"\nStep {t + 1}")
            for i in range(num_agents):
                print(f"Agent {i+1}:")
                print(f"  Mistake Rate: {metrics.mistake_rates[i][-1]:.3f}")
                print(f"  Learning Rate: {metrics.learning_rates[i][-1]:.3f}")
            print("--------------------")
    
    return agents, metrics
