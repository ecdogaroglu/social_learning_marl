import math
import matplotlib.pyplot as plt
from typing import List
import numpy as np

class MultiAgentMetricsTracker:
    """Track learning metrics for multiple agents."""
    def __init__(self, num_agents: int, num_steps: int):
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.window_size = int(self.num_steps / 10)
        self.total_steps = 0
        self.signals = [[] for _ in range(num_agents)]
        self.mistakes = [[] for _ in range(num_agents)]
        self.mistake_rates = [[] for _ in range(num_agents)]
        self.learning_rates = [[] for _ in range(num_agents)]
        self.action_rate_history = [[] for _ in range(num_agents)] 
        self.avg_action_rates = [[] for _ in range(num_agents)]
        self.true_rewards = [[] for _ in range(num_agents)]
        self.observed_rewards = [[] for _ in range(num_agents)]
        self.avg_observed_rewards = [[] for _ in range(num_agents)]
        self.avg_true_rewards = [[] for _ in range(num_agents)]
        self.avg_signals = [[] for _ in range(num_agents)]
    
    def add_mistakes(self, mistakes: List[bool]):
        self.total_steps += 1
        for i in range(self.num_agents):
            self.mistakes[i].append(mistakes[i])

    def add_action_rate(self, true_action_rates: List[float]):
        """Track actions chosen by each agent."""
        for i in range(self.num_agents):
            self.action_rate_history[i].append(true_action_rates[i])
    
    def add_true_rewards(self, true_rewards: List[float]):
        """Track actions chosen by each agent."""
        for i in range(self.num_agents):
            self.true_rewards[i].append(true_rewards[i])

    def add_observed_rewards(self, observed_rewards: List[float]):
        """Track actions chosen by each agent."""
        for i in range(self.num_agents):
            self.observed_rewards[i].append(observed_rewards[i])
    
    def add_signals(self, signals: List[float]):
        """Track actions chosen by each agent."""
        for i in range(self.num_agents):
            self.signals[i].append(signals[i])
    
    def compute_mistake_rate(self, agent_idx: int) -> float:
        if self.total_steps == 0:
            return 1.0
        return sum(self.mistakes[agent_idx]) / self.total_steps
    
    def compute_running_avg(self, metric, agent_idx: int) -> float:
        recent_rates = metric[agent_idx][-self.window_size:]
        
        avg_metric = sum(recent_rates) / len(recent_rates) if recent_rates else 0.5
        return avg_metric  
            
    def compute_learning_rate(self, agent_idx: int) -> float:
        prob_mistake = 1 - self.compute_running_avg(self.action_rate_history, agent_idx=agent_idx)

        if prob_mistake != 0:
            r = -math.log(prob_mistake) / self.total_steps

        else:
            r = float('inf')
        return r
    
    def update_metrics(self):
        for i in range(self.num_agents):
            self.mistake_rates[i].append(self.compute_mistake_rate(i))
            self.learning_rates[i].append(self.compute_learning_rate(i))
            self.avg_action_rates[i].append(self.compute_running_avg(self.action_rate_history, i))
            self.avg_observed_rewards[i].append(self.compute_running_avg(self.observed_rewards, i))
            self.avg_true_rewards[i].append(self.compute_running_avg(self.true_rewards, i))
            self.avg_signals[i].append(self.compute_running_avg(self.signals, i))

def plot_multi_agent_metrics(metrics_tracker: MultiAgentMetricsTracker,
                           signal_accuracy: float,
                           save_path: str = None):
    if not 0 <= signal_accuracy <= 1:
        raise ValueError("signal_accuracy must be between 0 and 1")
        
    def plot_metric(ax, data, title, ylabel, x_scale="linear", include_theoretical=False):
        # Get indices of agents based on their maximum learning rates ever achieved
        max_learning_rates = [max(rates) for rates in metrics_tracker.learning_rates]
        sorted_indices = np.argsort(max_learning_rates)
        top_indices = sorted_indices[-2:]  # Two agents that achieved highest learning rates
        bottom_indices = sorted_indices[:2]  # Two agents that achieved lowest learning rates
        selected_indices = np.concatenate([top_indices, bottom_indices])
        
        # Use perceptually uniform colormap
        colors = plt.cm.Greys(np.linspace(0.2, 1, metrics_tracker.num_agents))
        
        # Only plot selected agents
        for idx in range(metrics_tracker.num_agents):
            ax.plot(data[idx], 
                   color=colors[idx],
                   label=f'Agent {idx+1}',
                   alpha=0.7)
        
        if include_theoretical:
            q, p = signal_accuracy, 1 - signal_accuracy
            r_bdd = 2 * (2 * q - 1) * math.log(q/p)
            ax.axhline(y=r_bdd, color='black', linestyle='--',
                      label='Theoretical Bound', alpha=1)
        
        ax.set_xscale(x_scale)
        
        ax.set_title(f'({title}) {ylabel} '
                    f'(q={signal_accuracy})')
        ax.set_xlabel('Time Step')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Place legend in top right with increased size
        legend = ax.legend(fontsize='small',  # Increased from x-small to small
                          framealpha=0.5,
                          loc='upper right',
                          handlelength=1.5,  # Increased from 1
                          handletextpad=0.5)  # Increased from 0.4

        
        ymin, ymax = ax.get_ylim()
        if 'Rate' in ylabel:  # For plots with log scale
            ax.set_ylim(ymin, ymax * 2)  # Double the upper limit for log scale
        else:  # For linear scale plots
            y_range = ymax - ymin
            ax.set_ylim(ymin, ymax + 0.2 * y_range)  # Add 20% of the range for linear scale
        
    plt.style.use('classic')  # More modern style
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot each metric
    plot_metric(ax1, metrics_tracker.avg_signals, 'a', 'Signals')
    plot_metric(ax2, metrics_tracker.avg_observed_rewards, 'b', 'Observed Rewards')
    plot_metric(ax3, metrics_tracker.avg_action_rates, 'c', 'True Action Rates')
    plot_metric(ax4, metrics_tracker.learning_rates, 'd', 'Learning Rates', x_scale="log", include_theoretical=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()