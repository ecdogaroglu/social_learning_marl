import numpy as np
import torch

from typing import Tuple

class Environment:
    def __init__(self, signal_accuracy: float = 0.75):
        """Binary state environment with paper's reward function.
        
        Args:
            signal_accuracy: q in the paper (>0.5 for informative signals)
        """
        self.q = signal_accuracy
        # Draw true state once and keep it fixed throughout
        self.true_state = np.random.randint(2)  # Fixed true state ω
    
    def get_signal(self) -> torch.Tensor:
        """Generate signal based on current true state."""
        if np.random.random() < self.q:
            signal = self.true_state  # Correct signal
        else:
            signal = 1 - self.true_state  # Incorrect signal
        return torch.FloatTensor([[signal]])
    
    def compute_reward(self, action: int, signal: float) -> float:
        """Implements v(a,s) from the paper for binary case."""
        # v(a,s) = (q·1{a=s} - (1-q)·1{a≠s})/(2q-1)
        action_matches_signal = float(action == int(signal))
        numerator = self.q * action_matches_signal - (1 - self.q) * (1 - action_matches_signal)
        denominator = 2 * self.q - 1
        return numerator / denominator
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        # Get current signal
        current_signal = self.get_signal()
        
        # Compute reward using paper's reward function
        reward = self.compute_reward(action, current_signal.item())
        
        # Check if action was a mistake
        is_mistake = (action != self.true_state)
        
        # Get next signal
        next_signal = self.get_signal()
            
        return next_signal, reward, is_mistake
