import os
import subprocess
import argparse

def train_network(num_agents, num_states, num_episodes, eval_interval, save_path, signal_accuracy=0.7):
    """
    Train a network configuration with specified parameters
    
    Args:
        num_agents: Number of agents in the network
        num_states: Number of states in the environment
        num_episodes: Number of training episodes
        eval_interval: Evaluation interval
        save_path: Path to save models
        signal_accuracy: Accuracy of private signals
    """
    # Create directory for this configuration if it doesn't exist
    config_dir = os.path.join(save_path, f"{num_agents}_agents")
    os.makedirs(config_dir, exist_ok=True)
    
    # Build command to run training script
    cmd = [
        "python", "training_script.py",
        "--env", "learning",
        "--agents", str(num_agents),
        "--states", str(num_states),
        "--episodes", str(num_episodes),
        "--eval-interval", str(eval_interval),
        "--save-path", config_dir
    ]
    
    # Run training
    print(f"Training network with {num_agents} agents...")
    subprocess.run(cmd)
    print(f"Training complete for {num_agents} agents. Models saved to {config_dir}.")

def main():
    parser = argparse.ArgumentParser(description='Train multiple network configurations')
    parser.add_argument('--agents', type=str, default='2,3,5,9',
                       help='Comma-separated list of numbers of agents to train')
    parser.add_argument('--states', type=int, default=2,
                       help='Number of states in the environment')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--eval-interval', type=int, default=10,
                       help='Evaluation interval')
    parser.add_argument('--save-path', type=str, default='./trained_networks',
                       help='Path to save models')
    parser.add_argument('--signal-accuracy', type=float, default=0.7,
                       help='Accuracy of private signals')
    
    args = parser.parse_args()
    
    # Parse number of agents
    num_agents_list = [int(n) for n in args.agents.split(',')]
    
    # Create base directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    # Train each network configuration
    for num_agents in num_agents_list:
        train_network(
            num_agents=num_agents,
            num_states=args.states,
            num_episodes=args.episodes,
            eval_interval=args.eval_interval,
            save_path=args.save_path,
            signal_accuracy=args.signal_accuracy
        )
    
    print(f"All network configurations trained successfully.")
    print(f"To analyze learning rates, run:")
    print(f"python analyze_learning_rates.py --agents-path {args.save_path} --neighbors {args.agents}")

if __name__ == "__main__":
    main() 