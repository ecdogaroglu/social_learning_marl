import torch
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import math

from metrics import MetricsTracker
from environment import Environment
from agent import Agent
from pathlib import Path

# Current script path
current_path = Path(__file__).resolve()

# Move to the parent directory and specify the new folder
save_folder = current_path.parent.parent.parent / "charts"
save_folder.mkdir(exist_ok=True)  # Create folder if it doesn't exist

from ..multi_agent.metrics import plot_multi_agent_metrics

def plot_metrics(metrics_tracker: MetricsTracker, signal_accuracy: float = 0.75):
    """Plot learning curves with theoretical bound in academic black and white style."""
    # Set style for academic black and white plots
    plt.style.use('grayscale')
    
    # Create figure with side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calculate theoretical bound
    q_bound = 2 * signal_accuracy - 1  # For symmetric binary case
    r_bdd = -math.log(1 - q_bound)
    
    # Plot mistake rate
    ax1.plot(metrics_tracker.mistake_rates, color='black', label='Empirical Rate')
    ax1.set_title('(a) Mistake Rate Over Time', pad=10)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Mistake Rate')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(frameon=False)
    
    # Plot learning rate with different line styles for distinction
    ax2.plot(metrics_tracker.learning_rates, color='black', label='Empirical Rate')
    ax2.axhline(y=r_bdd, color='black', linestyle='--', 
                label=f'Theoretical Bound')
    ax2.set_title('(b) Learning Rate Over Time', pad=10)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Learning Rate')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(frameon=False)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save with high DPI for publication quality
    plt.savefig(save_folder/f'single_agent_learning_curves_q={signal_accuracy}.png', dpi=300, bbox_inches='tight')
    plt.show()

def train(num_steps: int = 100000, signal_accuracy: float = 0.75):
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    env = Environment(signal_accuracy=signal_accuracy)
    agent = Agent()
    metrics = MetricsTracker()
    
    # Get initial signal
    state = env.get_signal()
    
    # Run for specified number of steps
    for t in range(num_steps):
        # Select action using current policy
        with torch.no_grad():  # Don't track gradients during action selection
            action, log_prob = agent.select_action(state)
        
        # Take action in environment
        next_state, reward, is_mistake = env.step(action)
        
        # Update agent (create fresh log_prob for training)
        action_probs, _ = agent.actor(state, agent.hidden_actor.detach())
        dist = Categorical(action_probs)
        train_log_prob = dist.log_prob(torch.tensor(action))
        
        agent.update(state, train_log_prob, reward, next_state)
        
        # Update metrics
        metrics.add_mistake(is_mistake)
        metrics.update_metrics()
        
        # Move to next state
        state = next_state
        
        # Print progress
        if (t + 1) % 1000 == 0:
            print(f"Step {t + 1}")
            print(f"Current Mistake Rate: {metrics.compute_mistake_rate():.3f}")
            print(f"Current Learning Rate: {metrics.compute_learning_rate():.3f}")
            print("--------------------")
    
    return metrics

if __name__ == "__main__":
    metrics = train(num_steps=10000, signal_accuracy=0.75)
    plot_metrics(metrics)