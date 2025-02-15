import math
import matplotlib.pyplot as plt
from typing import List

class MultiAgentMetricsTracker:
    """Track learning metrics for multiple agents."""
    def __init__(self, num_agents: int, num_steps: int):
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.mistakes = [[] for _ in range(num_agents)]
        self.mistake_rates = [[] for _ in range(num_agents)]
        self.learning_rates = [[] for _ in range(num_agents)]
        self.action_rate_history = [[] for _ in range(num_agents)]  # Track actions for each agent
        self.avg_action_rates = [[] for _ in range(num_agents)]
        self.window_size = int(self.num_steps / 100)
        self.total_steps = 0
    
    def add_mistakes(self, mistakes: List[bool]):
        self.total_steps += 1
        for i in range(self.num_agents):
            self.mistakes[i].append(mistakes[i])

    def add_action_rate(self, true_action_rates: List[float]):
        """Track actions chosen by each agent."""
        for i in range(self.num_agents):
            self.action_rate_history[i].append(true_action_rates[i])
    
    def compute_mistake_rate(self, agent_idx: int) -> float:
        if self.total_steps == 0:
            return 1.0
        return sum(self.mistakes[agent_idx]) / self.total_steps
    
    def compute_avg_action_rates(self, agent_idx: int) -> float:
        recent_rates = self.action_rate_history[agent_idx][-self.window_size:]
        
        avg_action_rate = sum(recent_rates) / len(recent_rates) if recent_rates else 0.5
        return avg_action_rate        
            
    def compute_learning_rate(self, agent_idx: int) -> float:
        r = -math.log(1-self.compute_avg_action_rates(agent_idx=agent_idx)) / self.total_steps
            
        return r


    
    def update_metrics(self):
        for i in range(self.num_agents):
            self.mistake_rates[i].append(self.compute_mistake_rate(i))
            self.learning_rates[i].append(self.compute_learning_rate(i))
            self.avg_action_rates[i].append(self.compute_avg_action_rates(i))

import numpy as np
import matplotlib.pyplot as plt
import math

def plot_multi_agent_metrics(metrics_tracker: MultiAgentMetricsTracker, 
                             signal_accuracy: float,
                             save_path: str = None):
    plt.style.use('grayscale')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    q = signal_accuracy
    p = 1 - q
    r_bdd = 2 * (2 * q - 1) * math.log(q/p)
    
    # Generate different grayscale shades for each agent
    num_agents = metrics_tracker.num_agents
    colors = [str(gray) for gray in np.linspace(0, 0.8, num_agents)]  # Shades of gray from black to light gray
    
    # Plot action proportions
    # Determine the top and bottom 5 agents based on the latest average action rates
    sorted_indices = np.argsort(metrics_tracker.avg_action_rates)
    top_5_indices = sorted_indices[-5:]
    bottom_5_indices = sorted_indices[:5]
    
    for i in range(num_agents):
        if i in top_5_indices or i in bottom_5_indices:
            ax1.plot(metrics_tracker.avg_action_rates[i], 
                     color=colors[i], 
                     label=f'Agent {i+1}',
                     alpha=0.7)
    
    ax1.axhline(y=0.5, color='black', linestyle='--', 
                label='Random Policy', alpha=0.5)
    ax1.set_xscale('log')  # Set x-axis to logarithmic scale
    ax1.set_title(f'(a) Policy Over Time (q={signal_accuracy}, window size={metrics_tracker.window_size})')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Proportion of True Action')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Plot learning rates
    for i in range(num_agents):
        if i in top_5_indices or i in bottom_5_indices:
            ax2.plot(metrics_tracker.learning_rates[i], 
                    color=colors[i], 
                    label=f'Agent {i+1}',
                    alpha=0.7)
        
    ax2.axhline(y=r_bdd, color='black', linestyle='--', 
                label='Theoretical Bound', alpha=0.5)
    ax2.set_title(f'(b) Learning Rates Over Time (q={signal_accuracy}, window size={metrics_tracker.window_size})')
    ax2.set_xscale('log')  # Set x-axis to logarithmic scale
    ax2.set_yscale('log')  # Set y-axis to logarithmic scale
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Learning Rate')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
