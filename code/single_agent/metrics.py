import math
import numpy as np

class MetricsTracker:
    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        self.window_size = int(self.num_steps / 10)
        self.total_steps = 0
        
        # Basic metrics from original implementation
        self.mistakes = []
        self.mistake_rates = []
        self.learning_rates = []
        
        # New metrics from multi-agent implementation
        self.signals = []
        self.action_rate_history = []
        self.true_rewards = []
        self.observed_rewards = []
        
        # Running averages
        self.avg_action_rates = []
        self.avg_observed_rewards = []
        self.avg_true_rewards = []
        self.avg_signals = []
    
    def add_mistake(self, mistake: bool):
        self.total_steps += 1
        self.mistakes.append(mistake)
    
    def add_action_rate(self, true_action_rate: float):
        """Track action chosen by the agent."""
        self.action_rate_history.append(true_action_rate)
    
    def add_true_reward(self, true_reward: float):
        """Track true reward."""
        self.true_rewards.append(true_reward)
    
    def add_observed_reward(self, observed_reward: float):
        """Track observed reward."""
        self.observed_rewards.append(observed_reward)
    
    def add_signal(self, signal: float):
        """Track received signal."""
        self.signals.append(signal)
    
    def compute_mistake_rate(self) -> float:
        if self.total_steps == 0:
            return 1.0
        return sum(self.mistakes) / self.total_steps
    
    def compute_running_avg(self, metric: list) -> float:
        recent_values = metric[-self.window_size:]
        return sum(recent_values) / len(recent_values) if recent_values else 0.5
    
    def compute_learning_rate(self) -> float:
        """Compute learning rate using action rate history."""
        prob_mistake = 1 - self.compute_running_avg(self.action_rate_history)
        
        if prob_mistake != 0:
            r = -math.log(prob_mistake) / self.total_steps
        else:
            r = float('inf')
        return r
    
    def update_metrics(self):
        """Update all metrics including running averages."""
        self.mistake_rates.append(self.compute_mistake_rate())
        self.learning_rates.append(self.compute_learning_rate())
        
        # Update running averages
        self.avg_action_rates.append(self.compute_running_avg(self.action_rate_history))
        self.avg_observed_rewards.append(self.compute_running_avg(self.observed_rewards))
        self.avg_true_rewards.append(self.compute_running_avg(self.true_rewards))
        self.avg_signals.append(self.compute_running_avg(self.signals))

def plot_metrics(metrics_tracker: MetricsTracker, 
                         signal_accuracy: float,
                         save_path: str = None):
    """Plot metrics with theoretical bounds."""
    import matplotlib.pyplot as plt
    
    plt.style.use('classic')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    def plot_metric(ax, data, title, ylabel, x_scale="linear", include_theoretical=False):
        ax.plot(data, color='black', label='Agent', alpha=0.7)
        
        if include_theoretical:
            q, p = signal_accuracy, 1 - signal_accuracy
            r_bdd = 2 * (2 * q - 1) * math.log(q/p)
            ax.axhline(y=r_bdd, color='black', linestyle='--',
                      label='Theoretical Bound', alpha=1)
        
        ax.set_xscale(x_scale)
        ax.set_title(f'({title}) {ylabel} (q={signal_accuracy})')
        ax.set_xlabel('Time Step')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.7)
        
    
    # Plot each metric
    plot_metric(ax1, metrics_tracker.avg_signals, 'a', 'Signals')
    plot_metric(ax2, metrics_tracker.avg_observed_rewards, 'b', 'Observed Rewards')
    plot_metric(ax3, metrics_tracker.avg_action_rates, 'c', 'True Action Rates')
    plot_metric(ax4, metrics_tracker.learning_rates, 'd', 'Learning Rates', 
                x_scale="log", include_theoretical=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()