import math

class MetricsTracker:
    def __init__(self):
        self.mistakes = []  # Full history of mistakes
        self.mistake_rates = []
        self.learning_rates = []
        self.total_steps = 0
    
    def add_mistake(self, mistake: bool):
        self.total_steps += 1
        self.mistakes.append(mistake)
    
    def compute_mistake_rate(self) -> float:
        """Compute current mistake rate."""
        if self.total_steps == 0:
            return 1.0  # Start with worst case
        return sum(self.mistakes) / self.total_steps
    
    def compute_learning_rate(self) -> float:
        """Compute learning rate as defined in the paper.
        r = -1/t log P(aᵗ ≠ aω)
        """
        if self.total_steps < 2:  # Need at least 2 steps
            return 0.0
            
        # Compute mistake rate over increasing horizons
        rates = []
        # Use exponentially increasing windows to approximate limit
        for window in [2**i for i in range(2, int(math.log2(self.total_steps + 1)))]:
            recent_mistakes = self.mistakes[-window:]
            if len(recent_mistakes) >= window:  # Only use full windows
                mistake_rate = sum(recent_mistakes) / window
                if mistake_rate == 0:  # Perfect learning
                    mistake_rate = 1/window  # Approximate with smallest possible rate
                rate = -1/window * math.log(mistake_rate)
                rates.append(rate)
        
        return min(rates) if rates else float('inf')  # lim inf
    
    def update_metrics(self):
        self.mistake_rates.append(self.compute_mistake_rate())
        self.learning_rates.append(self.compute_learning_rate())
