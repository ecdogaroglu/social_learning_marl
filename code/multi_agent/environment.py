import numpy as np
import torch
from typing import List, Tuple

class MultiAgentEnvironment:
    """Multi-agent social learning environment."""
    def __init__(self, num_agents: int, signal_accuracy: float = 0.75):
        self.num_agents = num_agents
        self.q = signal_accuracy
        #self.true_state = np.random.randint(1)
        self.true_state = 1

    def get_signals(self) -> List[torch.Tensor]:
        """Generate signals for all agents."""
        signals = []
        for _ in range(self.num_agents):
            if np.random.random() < self.q:
                signal = self.true_state
            else:
                signal = 1 - self.true_state
            signals.append(torch.FloatTensor([[signal]]))
        return signals
    
    def compute_rewards(self, actions: List[int], signals: List[torch.Tensor]) -> List[float]:
        """
        Compute rewards using the binary case reward function from the paper:
        R(a,ψ) = v(a,s) = (q·1_{a=s} - (1-q)·1_{a≠s})/(2q-1)
        """
        observed_rewards = []
        true_rewards = []
        for i in range(self.num_agents):
            signal = signals[i].item()
            action = actions[i]
            
            # Implementation of the binary case reward function
            indicator_match = 1.0 if action == signal else 0.0
            indicator_mismatch = 1.0 if action != signal else 0.0
            
            observed_reward = (self.q * indicator_match - (1 - self.q) * indicator_mismatch) / (2 * self.q - 1)
            observed_rewards.append(observed_reward)
            true_reward = 1.0 if action == self.true_state else 0.0
            true_rewards.append(true_reward)
        
        return observed_rewards, true_rewards
    
    def step(self, signals: List[int], actions: List[int]) -> Tuple[List[torch.Tensor], List[float], List[bool]]:
        observed_rewards, true_rewards = self.compute_rewards(actions,signals)
        mistakes = [action != self.true_state for action in actions]
        next_signals = self.get_signals()
        return next_signals, observed_rewards, true_rewards, mistakes
