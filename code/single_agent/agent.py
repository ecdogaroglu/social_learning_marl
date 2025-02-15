import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from typing import Tuple

class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        
        # GRU weights
        self.w_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.w_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.w_h = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((x, h), dim=1)
        z = torch.sigmoid(self.w_z(combined))
        r = torch.sigmoid(self.w_r(combined))
        h_tilde = torch.tanh(self.w_h(torch.cat((x, r * h), dim=1)))
        h_new = (1 - z) * h + z * h_tilde
        return h_new

class Actor(nn.Module):
    def __init__(self, hidden_size: int, action_size: int = 2):
        super(Actor, self).__init__()
        self.rnn = RNN(input_size=1, hidden_size=hidden_size)
        self.policy = nn.Linear(hidden_size, action_size)
    
    def forward(self, signal: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.rnn(signal, hidden)
        action_probs = F.softmax(self.policy(hidden), dim=1)
        return action_probs, hidden
    
    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(1, self.rnn.hidden_size)

class Critic(nn.Module):
    def __init__(self, hidden_size: int):
        super(Critic, self).__init__()
        self.rnn = RNN(input_size=1, hidden_size=hidden_size)
        self.value = nn.Linear(hidden_size, 1)
    
    def forward(self, signal: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.rnn(signal, hidden)
        state_value = self.value(hidden)
        return state_value, hidden
    
    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(1, self.rnn.hidden_size)

class Agent:
    def __init__(self, hidden_size: int = 64, lr_actor: float = 1e-4, lr_critic: float = 1e-3):
        self.actor = Actor(hidden_size, action_size=2)
        self.critic = Critic(hidden_size)
        
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = 0.99
        self.hidden_actor = self.actor.init_hidden()
        self.hidden_critic = self.critic.init_hidden()
    
    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        action_probs, self.hidden_actor = self.actor(state, self.hidden_actor)
        dist = Categorical(action_probs)
        action = dist.sample()
        # Return both action and log probability
        return action.item(), dist.log_prob(action)
    
    def update(self, state: torch.Tensor, log_prob: torch.Tensor, reward: float, next_state: torch.Tensor):
        # Get current state value
        current_value, new_critic_hidden = self.critic(state, self.hidden_critic.detach())
        
        # Get next state value
        with torch.no_grad():
            next_value, _ = self.critic(next_state, new_critic_hidden.detach())
        
        # Compute advantage
        advantage = reward + self.gamma * next_value.item() - current_value.item()
        
        # Policy gradient loss
        policy_loss = -log_prob * advantage  # Log prob already has gradients
        
        # Value function loss
        value_loss = F.mse_loss(current_value, torch.tensor([reward + self.gamma * next_value.item()]))
        
        # Update policy first
        self.optimizer_actor.zero_grad()
        policy_loss.backward()
        self.optimizer_actor.step()
        
        # Then update value function
        self.optimizer_critic.zero_grad()
        value_loss.backward()
        self.optimizer_critic.step()
        
        # Update hidden states after optimization
        with torch.no_grad():
            _, self.hidden_actor = self.actor(state, self.hidden_actor.detach())
            self.hidden_critic = new_critic_hidden.detach()
