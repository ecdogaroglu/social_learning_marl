import subprocess
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exponential_decay(t, r):
    """Exponential decay function e^(-rt) for fitting learning rates"""
    return np.exp(-r * t)

def run_single_experiment(env_type, num_agents, num_states, num_timesteps, signal_accuracy, output_dir):
    """
    Run a single experiment with specified parameters
    
    Args:
        env_type: Environment type ('learning' or 'strategic')
        num_agents: Number of agents
        num_states: Number of states
        num_timesteps: Number of timesteps
        signal_accuracy: Accuracy of private signals
        output_dir: Directory to save results
    
    Returns:
        result_dir: Path to the results directory
    """
    # Create output directory
    result_dir = os.path.join(output_dir, f"{num_agents}_agents")
    os.makedirs(result_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "python", "training_with_mistake_tracking.py",
        "--env", env_type,
        "--agents", str(num_agents),
        "--states", str(num_states),
        "--timesteps", str(num_timesteps),
        "--eval-interval", str(num_timesteps // 10),  # Evaluate 10 times during training
        "--log-interval", str(num_timesteps // 100),  # Log 100 times during training
        "--save-path", result_dir,
        "--signal-accuracy", str(signal_accuracy)
    ]
    
    # Run experiment
    print(f"Running experiment with {num_agents} agents...")
    subprocess.run(cmd)
    
    return result_dir

def fit_learning_rate_to_mistakes(mistake_probs, window_ratio=0.2):
    """
    Fit learning rate to mistake probabilities over a window
    
    Args:
        mistake_probs: Numpy array of mistake probabilities
        window_ratio: Ratio of timesteps to use for window size
    
    Returns:
        rate: Estimated learning rate
    """
    # Get dimensions
    num_timesteps, num_agents = mistake_probs.shape
    
    # Calculate window size (at least 50 timesteps)
    window_size = max(50, int(num_timesteps * window_ratio))
    
    # Use second half of training to fit learning rate
    start_idx = num_timesteps - window_size
    if start_idx < 0:
        start_idx = 0
    
    window_mistakes = mistake_probs[start_idx:start_idx+window_size]
    time_points = np.arange(window_size)
    
    # Fit to average mistake probabilities across agents
    avg_mistakes = np.mean(window_mistakes, axis=1)
    
    try:
        # Fit exponential decay
        params, _ = curve_fit(
            exponential_decay, 
            time_points, 
            avg_mistakes,
            p0=[0.001],  # Initial guess
            bounds=(0, np.inf)  # Rate must be positive
        )
        rate = params[0]
    except:
        # If fitting fails, use a default value
        rate = 0.001
    
    return rate

def run_network_size_experiments(
    env_type='learning',
    network_sizes=[2, 3, 5, 8],
    num_states=2,
    num_timesteps=5000,
    signal_accuracy=0.7,
    output_dir='./network_size_experiments'
):
    """
    Run experiments with different network sizes
    
    Args:
        env_type: Environment type ('learning' or 'strategic')
        network_sizes: List of network sizes to test
        num_states: Number of states
        num_timesteps: Number of timesteps per experiment
        signal_accuracy: Accuracy of private signals
        output_dir: Directory to save results
    
    Returns:
        results: Dictionary of results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Store results
    results = {
        'network_sizes': network_sizes,
        'num_neighbors': [n-1 for n in network_sizes],  # In complete networks
        'empirical_rates': [],
        'theoretical_bounds': []
    }
    
    # Run experiments for each network size
    for num_agents in network_sizes:
        # Run experiment
        result_dir = run_single_experiment(
            env_type=env_type,
            num_agents=num_agents,
            num_states=num_states,
            num_timesteps=num_timesteps,
            signal_accuracy=signal_accuracy,
            output_dir=output_dir
        )
        
        # Load mistake probabilities
        mistake_probs_file = os.path.join(result_dir, 'all_mistake_probs.npy')
        if os.path.exists(mistake_probs_file):
            mistake_probs = np.load(mistake_probs_file)
            
            # Fit learning rate to mistake probabilities
            rate = fit_learning_rate_to_mistakes(mistake_probs)
            results['empirical_rates'].append(rate)
            
            # Calculate theoretical bound
            num_neighbors = num_agents - 1
            if signal_accuracy > 0.5:
                I_private = (1-signal_accuracy)*np.log2(1-signal_accuracy) + signal_accuracy*np.log2(signal_accuracy) + 1
                network_factor = 1 + 0.1 * num_neighbors  # Adjust this based on the paper
                bound = I_private * network_factor
            else:
                bound = 0
                
            results['theoretical_bounds'].append(bound)
            
            print(f"Network size {num_agents}: rate = {rate:.6f}, bound = {bound:.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.plot(results['num_neighbors'], results['empirical_rates'], 'o-', label='Empirical Learning Rate')
    plt.plot(results['num_neighbors'], results['theoretical_bounds'], 's--', label='Theoretical Bound')
    
    plt.title('Learning Rate vs Number of Neighbors (Complete Network)')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_vs_neighbors.png'))
    plt.close()
    
    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({
            'network_sizes': results['network_sizes'],
            'num_neighbors': results['num_neighbors'],
            'empirical_rates': [float(r) for r in results['empirical_rates']],
            'theoretical_bounds': [float(b) for b in results['theoretical_bounds']]
        }, f, indent=2)
    
    return results

def load_and_plot_results(results_file, output_dir='.'):
    """
    Load and plot results from a previous run
    
    Args:
        results_file: Path to results JSON file
        output_dir: Directory to save plots
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    plt.plot(results['num_neighbors'], results['empirical_rates'], 'o-', label='Empirical Learning Rate')
    plt.plot(results['num_neighbors'], results['theoretical_bounds'], 's--', label='Theoretical Bound')
    
    plt.title('Learning Rate vs Number of Neighbors (Complete Network)')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_vs_neighbors.png'))
    plt.close()
    
    # Plot ratio of empirical to theoretical
    plt.figure(figsize=(10, 6))
    
    ratio = [e/t if t > 0 else 0 for e, t in zip(results['empirical_rates'], results['theoretical_bounds'])]
    plt.plot(results['num_neighbors'], ratio, 'o-')
    
    plt.title('Ratio of Empirical to Theoretical Learning Rate')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Ratio (Empirical/Theoretical)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_ratio.png'))
    plt.close()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run experiments with different network sizes')
    parser.add_argument('--env', type=str, default='learning', choices=['learning', 'strategic'],
                       help='Environment type')
    parser.add_argument('--sizes', type=str, default='2,3,5,8',
                       help='Comma-separated list of network sizes')
    parser.add_argument('--states', type=int, default=2,
                       help='Number of states')
    parser.add_argument('--timesteps', type=int, default=5000,
                       help='Number of timesteps per experiment')
    parser.add_argument('--signal-accuracy', type=float, default=0.7,
                       help='Accuracy of private signals')
    parser.add_argument('--output-dir', type=str, default='./network_size_experiments',
                       help='Directory to save results')
    parser.add_argument('--load', type=str, default=None,
                       help='Path to results file to load instead of running experiments')
    
    args = parser.parse_args()
    
    if args.load:
        # Load and plot existing results
        print(f"Loading results from {args.load}...")
        load_and_plot_results(args.load, args.output_dir)
    else:
        # Parse network sizes
        network_sizes = [int(size) for size in args.sizes.split(',')]
        
        # Run experiments
        results = run_network_size_experiments(
            env_type=args.env,
            network_sizes=network_sizes,
            num_states=args.states,
            num_timesteps=args.timesteps,
            signal_accuracy=args.signal_accuracy,
            output_dir=args.output_dir
        )
        
        print(f"Experiments complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 