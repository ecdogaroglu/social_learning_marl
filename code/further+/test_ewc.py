#!/usr/bin/env python3
"""
Test script for Elastic Weight Consolidation (EWC) in POLARIS.

This script runs a series of experiments to test the EWC implementation 
for mitigating catastrophic forgetting in the POLARIS framework.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from modules.environment import SocialLearningEnvironment
from modules.args import parse_args
from modules.simulation import run_agents


def main():
    """Main function to run EWC experiments."""
    # Parse command-line arguments but override for our specific test
    args = parse_args()
    
    # Set specific parameters for EWC testing
    args.exp_name = "ewc_test"
    args.num_agents = 2
    args.network_type = 'complete'
    args.save_model = True
    args.use_gnn = True
    args.horizon = 1000
    args.num_episodes = 4
    
    # Run experiments with different EWC settings
    run_ewc_experiment(args, use_ewc=False, ewc_online=False, name="baseline")
    run_ewc_experiment(args, use_ewc=True, ewc_online=False, name="standard_ewc")
    run_ewc_experiment(args, use_ewc=True, ewc_online=True, name="online_ewc")
    
    # Compare results
    plot_comparison(args)
    
    
def run_ewc_experiment(args, use_ewc, ewc_online, name):
    """Run an experiment with specific EWC settings."""
    # Set EWC parameters
    args.use_ewc = use_ewc
    args.ewc_online = ewc_online
    args.ewc_lambda = 100.0
    args.ewc_gamma = 0.95
    args.exp_name = f"ewc_test_{name}"
    
    print(f"\n{'='*80}\nRunning experiment: {args.exp_name}")
    print(f"EWC enabled: {args.use_ewc}, Online mode: {args.ewc_online}\n{'='*80}\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create environment with varying true states to test catastrophic forgetting
    env = SocialLearningEnvironment(
        num_agents=args.num_agents,
        signal_accuracy=args.signal_accuracy,
        network_type=args.network_type,
        network_params={'density': args.network_density} if args.network_type == 'random' else None,
        horizon=args.horizon,
        seed=args.seed
    )
    
    # Save true states for each episode to analyze later
    true_states = []
    
    # Run training
    args.eval_only = False
    for episode in range(args.num_episodes):
        print(f"\nTraining Episode {episode+1}/{args.num_episodes}")
        # Run agents for this episode
        run_agents(env, args, training=True)
        # Save the true state
        true_states.append(env.true_state)
        
    # Save true state sequence
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "true_states.txt", "w") as f:
        f.write(",".join(map(str, true_states)))
    
    # Run evaluation
    args.eval_only = True
    args.load_model = 'auto'
    run_agents(env, args, training=False)


def plot_comparison(args):
    """Plot comparison of mistake rates across different EWC settings."""
    baseline_dir = Path(args.output_dir) / "ewc_test_baseline"
    standard_ewc_dir = Path(args.output_dir) / "ewc_test_standard_ewc"
    online_ewc_dir = Path(args.output_dir) / "ewc_test_online_ewc"
    
    # Load evaluation metrics
    try:
        baseline_metrics = np.load(baseline_dir / "eval_network_complete_agents_2" / "metrics.npz", allow_pickle=True)
        standard_ewc_metrics = np.load(standard_ewc_dir / "eval_network_complete_agents_2" / "metrics.npz", allow_pickle=True)
        online_ewc_metrics = np.load(online_ewc_dir / "eval_network_complete_agents_2" / "metrics.npz", allow_pickle=True)
        
        # Extract mistake rates
        baseline_mistakes = baseline_metrics['mistake_rates']
        standard_ewc_mistakes = standard_ewc_metrics['mistake_rates']
        online_ewc_mistakes = online_ewc_metrics['mistake_rates']
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(baseline_mistakes, label='Baseline (No EWC)', linewidth=2)
        plt.plot(standard_ewc_mistakes, label='Standard EWC', linewidth=2)
        plt.plot(online_ewc_mistakes, label='Online EWC', linewidth=2)
        
        plt.xlabel('Step')
        plt.ylabel('Mistake Rate')
        plt.title('Comparison of Mistake Rates with Different EWC Settings')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        comparison_dir = Path(args.output_dir) / "ewc_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(comparison_dir / "mistake_rate_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to {comparison_dir / 'mistake_rate_comparison.png'}")
        
    except Exception as e:
        print(f"Error generating comparison plot: {e}")


if __name__ == "__main__":
    main() 