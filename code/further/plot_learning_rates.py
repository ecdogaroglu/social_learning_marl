import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
from scipy.optimize import curve_fit

def exponential_decay(t, r):
    """Exponential decay function e^(-rt) for fitting learning rates"""
    return np.exp(-r * t)

def plot_mistake_probabilities(mistake_probs_file, output_dir='.'):
    """
    Plot mistake probabilities over time for each agent
    
    Args:
        mistake_probs_file: Path to numpy file with mistake probabilities
        output_dir: Directory to save plots
    """
    # Load mistake probabilities
    mistake_probs = np.load(mistake_probs_file)
    
    # Get dimensions
    num_timesteps, num_agents = mistake_probs.shape
    
    # Plot overall average
    plt.figure(figsize=(12, 6))
    avg_mistakes = np.mean(mistake_probs, axis=1)
    plt.plot(np.arange(num_timesteps), avg_mistakes)
    plt.title('Average Mistake Probability Across All Agents')
    plt.xlabel('Timestep')
    plt.ylabel('Mistake Probability')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to better see exponential decay
    
    # Add trendline
    try:
        # Fit exponential decay
        params, _ = curve_fit(
            exponential_decay, 
            np.arange(num_timesteps), 
            avg_mistakes,
            p0=[0.0001],  # Initial guess for rate
            bounds=(0, np.inf)  # Rate must be positive
        )
        
        rate = params[0]
        fit_curve = exponential_decay(np.arange(num_timesteps), rate)
        plt.plot(np.arange(num_timesteps), fit_curve, 'r--', 
                 label=f'Exponential fit: r = {rate:.6f}')
        plt.legend()
    except:
        print("Warning: Could not fit exponential decay to average mistakes")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_mistake_probability.png'))
    plt.close()
    
    # Plot for each agent
    plt.figure(figsize=(14, 8))
    
    for i in range(num_agents):
        plt.plot(np.arange(num_timesteps), mistake_probs[:, i], label=f'Agent {i+1}')
    
    plt.title('Mistake Probability Over Time for Each Agent')
    plt.xlabel('Timestep')
    plt.ylabel('Mistake Probability')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'agent_mistake_probabilities.png'))
    plt.close()
    
    # Plot windowed mistake probabilities for better visualization
    window_size = min(100, num_timesteps // 20)  # Use reasonable window size
    
    windowed_mistakes = np.zeros((num_timesteps - window_size + 1, num_agents))
    for i in range(num_timesteps - window_size + 1):
        windowed_mistakes[i] = np.mean(mistake_probs[i:i+window_size], axis=0)
    
    plt.figure(figsize=(14, 8))
    for i in range(num_agents):
        plt.plot(np.arange(len(windowed_mistakes)), windowed_mistakes[:, i], label=f'Agent {i+1}')
    
    plt.title(f'Moving Average Mistake Probability (Window Size: {window_size})')
    plt.xlabel('Timestep')
    plt.ylabel('Mistake Probability')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'windowed_mistake_probabilities.png'))
    plt.close()
    
    return mistake_probs

def plot_learning_rates(learning_rates_file, output_dir='.'):
    """
    Plot learning rates over time
    
    Args:
        learning_rates_file: Path to JSON file with learning rates
        output_dir: Directory to save plots
    """
    # Load learning rates
    with open(learning_rates_file, 'r') as f:
        learning_rate_data = json.load(f)
    
    timesteps = learning_rate_data['timesteps']
    empirical_rates = learning_rate_data['empirical_rates']
    theoretical_bounds = learning_rate_data['theoretical_bounds']
    
    # Calculate average rates across agents for each timestep
    avg_empirical = [np.mean(rates) for rates in empirical_rates]
    avg_theoretical = [np.mean(bounds) for bounds in theoretical_bounds]
    
    # Calculate standard deviations for error bars
    std_empirical = [np.std(rates) for rates in empirical_rates]
    
    # Plot learning rates
    plt.figure(figsize=(12, 6))
    plt.errorbar(timesteps, avg_empirical, yerr=std_empirical, fmt='o-', label='Empirical Rate')
    plt.plot(timesteps, avg_theoretical, 's--', label='Theoretical Bound')
    
    plt.title('Learning Rate vs Training Progress')
    plt.xlabel('Timestep')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_during_training.png'))
    plt.close()
    
    # Plot convergence: theoretical/empirical ratio
    if len(avg_empirical) > 0 and len(avg_theoretical) > 0:
        ratio = [t/e if e > 0 else float('nan') for t, e in zip(avg_theoretical, avg_empirical)]
        plt.figure(figsize=(12, 6))
        plt.plot(timesteps, ratio, 'o-')
        plt.axhline(y=1.0, linestyle='--', color='r', label='Theoretical = Empirical')
        
        plt.title('Ratio of Theoretical to Empirical Learning Rate')
        plt.xlabel('Timestep')
        plt.ylabel('Ratio (Theoretical/Empirical)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_rate_ratio.png'))
        plt.close()
    
    return learning_rate_data

def analyze_learning_rate_by_window(mistake_probs, window_sizes=[50, 100, 200, 500], output_dir='.'):
    """
    Analyze how learning rate estimates change based on window size
    
    Args:
        mistake_probs: Numpy array of mistake probabilities
        window_sizes: List of window sizes to analyze
        output_dir: Directory to save plots
    """
    num_timesteps, num_agents = mistake_probs.shape
    
    # Store results for each window size and agent
    window_results = {}
    
    for window_size in window_sizes:
        if window_size >= num_timesteps:
            continue
            
        window_results[window_size] = []
        
        # Fit learning rates for different windows
        for window_start in range(0, num_timesteps - window_size, window_size):
            window_end = window_start + window_size
            window_mistakes = mistake_probs[window_start:window_end]
            
            agent_rates = []
            for i in range(num_agents):
                try:
                    # Fit exponential decay to this window
                    params, _ = curve_fit(
                        exponential_decay, 
                        np.arange(window_size), 
                        window_mistakes[:, i],
                        p0=[0.001],  # Initial guess
                        bounds=(0, np.inf)  # Rate must be positive
                    )
                    rate = params[0]
                except:
                    rate = 0.0
                    
                agent_rates.append(rate)
            
            window_results[window_size].append(agent_rates)
    
    # Plot how learning rate estimates change with window size
    plt.figure(figsize=(12, 6))
    
    for window_size in window_results:
        if len(window_results[window_size]) > 0:
            # Average across agents for each window
            avg_rates = [np.mean(rates) for rates in window_results[window_size]]
            windows = np.arange(len(avg_rates)) * window_size
            
            plt.plot(windows, avg_rates, 'o-', label=f'Window size = {window_size}')
    
    plt.title('Learning Rate Estimates Using Different Window Sizes')
    plt.xlabel('Starting Timestep of Window')
    plt.ylabel('Estimated Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_by_window.png'))
    plt.close()
    
    return window_results

def plot_learning_rate_vs_window_size(mistake_probs, max_window=None, output_dir='.'):
    """
    Plot how learning rate estimate changes with window size
    
    Args:
        mistake_probs: Numpy array of mistake probabilities
        max_window: Maximum window size to consider
        output_dir: Directory to save plots
    """
    num_timesteps, num_agents = mistake_probs.shape
    
    if max_window is None:
        max_window = num_timesteps // 2
    
    window_sizes = np.arange(50, max_window, 50)
    
    # Calculate learning rates for each window size
    avg_rates = []
    
    for window_size in window_sizes:
        agent_rates = []
        
        for i in range(num_agents):
            try:
                # Fit to first window_size timesteps
                params, _ = curve_fit(
                    exponential_decay, 
                    np.arange(window_size), 
                    mistake_probs[:window_size, i],
                    p0=[0.001],
                    bounds=(0, np.inf)
                )
                rate = params[0]
            except:
                rate = 0.0
                
            agent_rates.append(rate)
            
        avg_rates.append(np.mean(agent_rates))
    
    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes, avg_rates, 'o-')
    
    plt.title('Learning Rate Estimate vs Window Size')
    plt.xlabel('Window Size (timesteps)')
    plt.ylabel('Estimated Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_vs_window_size.png'))
    plt.close()
    
    return window_sizes, avg_rates

def main():
    parser = argparse.ArgumentParser(description='Analyze and plot learning rates')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='./analysis',
                        help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find data files
    mistake_probs_file = os.path.join(args.data_dir, 'all_mistake_probs.npy')
    learning_rates_file = os.path.join(args.data_dir, 'learning_rates.json')
    
    # Check if files exist
    if not os.path.exists(mistake_probs_file):
        print(f"Error: Mistake probabilities file not found at {mistake_probs_file}")
        return
    
    # Plot mistake probabilities
    print("Analyzing mistake probabilities...")
    mistake_probs = plot_mistake_probabilities(mistake_probs_file, args.output_dir)
    
    # Plot learning rates if file exists
    if os.path.exists(learning_rates_file):
        print("Analyzing learning rates...")
        learning_rate_data = plot_learning_rates(learning_rates_file, args.output_dir)
    
    # Additional analyses
    print("Performing additional learning rate analyses...")
    analyze_learning_rate_by_window(mistake_probs, output_dir=args.output_dir)
    plot_learning_rate_vs_window_size(mistake_probs, output_dir=args.output_dir)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 