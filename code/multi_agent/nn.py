import torch
from torch import nn   

class MultiAgentRNN(nn.Module):
    """GRU implementation for processing signal and action histories."""
    def __init__(self, num_agents: int, hidden_size: int):
        super(MultiAgentRNN, self).__init__()
        self.num_agents = num_agents
        self.hidden_size = hidden_size
        
        # Input size = signal + actions of all agents
        self.input_size = 1 + num_agents
        
        # GRU weights
        self.w_z = nn.Linear(self.input_size + hidden_size, hidden_size)
        self.w_r = nn.Linear(self.input_size + hidden_size, hidden_size)
        self.w_h = nn.Linear(self.input_size + hidden_size, hidden_size)
    
    def forward(self, signal: torch.Tensor, actions: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = torch.cat((signal, actions), dim=1)
        combined = torch.cat((x, h), dim=1)
        
        z = torch.sigmoid(self.w_z(combined))
        r = torch.sigmoid(self.w_r(combined))
        h_tilde = torch.tanh(self.w_h(torch.cat((x, r * h), dim=1)))
        h_new = (1 - z) * h + z * h_tilde
        
        return h_new