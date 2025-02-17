import torch
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import math

from metrics import MetricsTracker, plot_metrics
from environment import Environment
from agent import Agent
from pathlib import Path

# Current script path
current_path = Path(__file__).resolve()

# Move to the parent directory and specify the new folder
save_folder = current_path.parent.parent.parent / "charts"
save_folder.mkdir(exist_ok=True)  # Create folder if it doesn't exist


def train(num_steps: int = 10000, signal_accuracy: float = 0.75):
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    env = Environment(signal_accuracy=signal_accuracy)
    agent = Agent()
    metrics = MetricsTracker(num_steps)
    
    # Get initial signal
    state = env.get_signal()
    
    # Run for specified number of steps
    for t in range(num_steps):
        # Select action using current policy
        with torch.no_grad():  # Don't track gradients during action selection
            action, log_prob = agent.select_action(state)
        
        # Take action in environment
        next_state, observed_reward, true_reward, is_mistake, current_signal = env.step(action)
        
        # Update agent (create fresh log_prob for training)
        action_probs, _ = agent.actor(state, agent.hidden_actor.detach())
        dist = Categorical(action_probs)
        train_log_prob = dist.log_prob(torch.tensor(action))
        
        agent.update(state, train_log_prob, observed_reward, next_state)
        
        # Update all metrics
        metrics.add_mistake(is_mistake)
        metrics.add_action_rate(float(action == env.true_state))  # True action rate
        metrics.add_true_reward(true_reward)
        metrics.add_observed_reward(observed_reward)
        metrics.add_signal(current_signal)
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
    # Set parameters
    num_steps = 10000
    signal_accuracy = 0.75
    
    # Train agent
    metrics = train(num_steps=num_steps, signal_accuracy=signal_accuracy)
    
    # Plot enhanced metrics
    save_path = save_folder/f'single_agent_learning_curves_q={signal_accuracy}.png'
    plot_metrics(metrics, signal_accuracy=signal_accuracy, save_path=save_path)