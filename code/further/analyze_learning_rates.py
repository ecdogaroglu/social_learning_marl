import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
import os
import json

from partially_observable_further import PartiallyObservableFURTHER
from learning_without_experimentation import LearningWithoutExperimentationEnv
from training_script import flatten_observation

def exponential_decay(t, r):
    """Exponential decay function e^(-rt) for fitting learning rates"""
    return np.exp(-r * t)

def calculate_mistake_probabilities(env, agents, num_timesteps=100, num_episodes=20):
    """
    Calculate mistake probabilities for each agent at each timestep
    
    Returns:
        mistake_probs: numpy array of shape (num_agents, num_timesteps) with average mistake probabilities
    """
    num_agents = len(agents)
    true_state = env.true_state
    
    # Initialize arrays to store mistake probabilities
    all_mistake_probs = np.zeros((num_episodes, num_agents, num_timesteps))
    
    for episode in range(num_episodes):
        # Reset environment and get initial observations
        observations = env.reset()
        true_state = env.true_state  # Get the ground truth state for this episode
        
        flat_observations = [flatten_observation(obs, 'learning', env.num_states) for obs in observations]
        
        # Initialize model states
        hidden_states = [None for _ in range(num_agents)]
        latent_strategies = [np.zeros(5) for _ in range(num_agents)]
        actions = [0 for _ in range(num_agents)]
        
        # Run for specified number of timesteps
        for t in range(num_timesteps):
            # Process actions
            processed_actions = []
            for i, action in enumerate(actions):
                if isinstance(action, int):
                    one_hot = np.zeros(env.num_states)
                    one_hot[action] = 1
                    processed_actions.append(one_hot)
                else:
                    processed_actions.append(action)
            
            # Update belief states and record action probabilities
            for i in range(num_agents):
                # Update belief state
                belief, hidden = agents[i].update_belief(
                    flat_observations[i],
                    processed_actions[i],
                    hidden_states[i]
                )
                hidden_states[i] = hidden
                
                # Get other agents' actions
                other_actions = np.concatenate([
                    processed_actions[j] for j in range(num_agents) if j != i
                ])
                
                # Infer latent strategies
                latent_strategies[i] = agents[i].infer_latent_strategies(
                    flat_observations[i],
                    processed_actions[i],
                    other_actions
                )
                
                # Get action probabilities for the current policy
                with torch.no_grad():
                    observation_tensor = torch.FloatTensor(flat_observations[i]).to(agents[i].device)
                    if observation_tensor.dim() == 1:
                        observation_tensor = observation_tensor.unsqueeze(0)
                        
                    latent_tensor = torch.FloatTensor(latent_strategies[i]).to(agents[i].device)
                    if latent_tensor.dim() == 1:
                        latent_tensor = latent_tensor.unsqueeze(0)
                    
                    # Get belief state
                    belief_tensor, _ = agents[i].belief_network(
                        observation_tensor, 
                        torch.zeros(1, env.num_states, device=agents[i].device)  # Dummy action
                    )
                    
                    # Get action probabilities
                    action_probs = agents[i].policy(belief_tensor, latent_tensor).cpu().numpy().flatten()
                    
                    # Calculate mistake probability (1 - probability of correct action)
                    mistake_prob = 1.0 - action_probs[true_state]
                    all_mistake_probs[episode, i, t] = mistake_prob
            
            # Take actions in the environment
            next_observations, rewards, done, info = env.step(actions)
            
            # Select actions for next timestep
            new_actions = []
            for i in range(num_agents):
                action, _ = agents[i].select_action(
                    flat_observations[i],
                    latent_strategies[i]
                )
                new_actions.append(action)
            
            # Update for next iteration
            flat_observations = [flatten_observation(obs, 'learning', env.num_states) for obs in next_observations]
            actions = new_actions
            
            if done:
                break
    
    # Average mistake probabilities across episodes
    mean_mistake_probs = np.mean(all_mistake_probs, axis=0)
    
    return mean_mistake_probs

def estimate_learning_rates(mistake_probs):
    """
    Estimate learning rates by fitting mistake probabilities to exponential decay
    
    Args:
        mistake_probs: numpy array of shape (num_agents, num_timesteps)
        
    Returns:
        learning_rates: list of estimated learning rates, one per agent
        fits: list of fitted curves, one per agent
    """
    num_agents, num_timesteps = mistake_probs.shape
    t = np.arange(num_timesteps)
    
    learning_rates = []
    fits = []
    
    for i in range(num_agents):
        # Get mistake probabilities for this agent
        agent_mistakes = mistake_probs[i]
        
        # Fit exponential decay
        params, _ = curve_fit(
            exponential_decay, 
            t, 
            agent_mistakes,
            p0=[0.1],  # Initial guess for rate
            bounds=(0, np.inf)  # Rate must be positive
        )
        
        rate = params[0]
        learning_rates.append(rate)
        
        # Generate fitted curve
        fit = exponential_decay(t, rate)
        fits.append(fit)
    
    return learning_rates, fits

def calculate_theoretical_bound(q, n_neighbors):
    """
    Calculate theoretical bound on learning rate based on signal accuracy and neighbors
    
    Args:
        q: signal accuracy
        n_neighbors: number of neighbors
        
    Returns:
        r_bound: theoretical upper bound on learning rate
    """
    # Implementation based on the social learning barrier paper
    if q <= 0.5:
        return 0.0  # No learning possible
    
    # Calculate the information gain from private signal
    I_private = (1 - 2*q) * np.log(1 - q) + (2*q - 1) * np.log(q) + np.log(2)
    
    # Calculate upper bound on learning rate
    # Here I'm assuming the formula as r ≤ I_private * (1 + c*n_neighbors)
    # where c is a constant representing the information gain from neighbors
    # This needs to be adjusted based on the exact formula in the paper
    c = 0.1  # This is a placeholder - adjust based on paper formula
    r_bound = I_private * (1 + c * n_neighbors)
    
    return r_bound

def run_analysis(agents_path, num_neighbors_list=[1, 2, 4, 8], num_states=2, 
                 signal_accuracy=0.7, num_timesteps=50, num_episodes=30):
    """
    Run analysis for different numbers of neighbors in a complete network
    
    Args:
        agents_path: path to directory containing trained agents
        num_neighbors_list: list of different numbers of neighbors to analyze
        num_states: number of states in the environment
        signal_accuracy: accuracy of private signals
        num_timesteps: number of timesteps to run for each episode
        num_episodes: number of episodes to average over
    """
    results = {
        'num_neighbors': [],
        'empirical_rates': [],
        'theoretical_bounds': []
    }
    
    for num_agents in num_neighbors_list:
        print(f"Analyzing with {num_agents} agents (complete network)...")
        
        # Each agent has num_agents-1 neighbors in a complete network
        num_neighbors = num_agents - 1
        
        # Load trained agents
        agents = []
        for i in range(num_agents):
            agent_path = os.path.join(agents_path, f"agent_{i}.pt")
            
            # Create agent with appropriate dimensions
            observation_dim = num_states + num_agents * num_states
            agent = PartiallyObservableFURTHER(
                observation_dim=observation_dim,
                action_dim=num_states,
                discrete_actions=True
            )
            
            # Load parameters
            agent.load_models(agent_path)
            agents.append(agent)
        
        # Create environment
        env = LearningWithoutExperimentationEnv(
            num_agents=num_agents,
            num_states=num_states,
            signal_accuracy=signal_accuracy,
            network_type='complete'
        )
        
        # Calculate mistake probabilities
        mistake_probs = calculate_mistake_probabilities(
            env, agents, num_timesteps, num_episodes
        )
        
        # Estimate learning rates
        learning_rates, fits = estimate_learning_rates(mistake_probs)
        
        # Calculate theoretical bounds
        theoretical_bounds = [calculate_theoretical_bound(signal_accuracy, num_neighbors) 
                             for _ in range(num_agents)]
        
        # Store results
        results['num_neighbors'].append(num_neighbors)
        results['empirical_rates'].append(learning_rates)
        results['theoretical_bounds'].append(theoretical_bounds)
        
        # Plot results for this network size
        plot_learning_curves(mistake_probs, fits, learning_rates, theoretical_bounds, num_agents)
    
    # Plot summary of learning rates vs number of neighbors
    plot_learning_rate_vs_neighbors(results)
    
    return results

def plot_learning_curves(mistake_probs, fits, learning_rates, theoretical_bounds, num_agents):
    """Plot mistake probabilities and fitted curves for each agent"""
    num_timesteps = mistake_probs.shape[1]
    t = np.arange(num_timesteps)
    
    plt.figure(figsize=(12, 8))
    
    for i in range(num_agents):
        plt.plot(t, mistake_probs[i], 'o', alpha=0.5, label=f'Agent {i+1} data')
        plt.plot(t, fits[i], '-', label=f'Agent {i+1} fit: r = {learning_rates[i]:.4f}')
        
        # Add horizontal line for theoretical bound
        plt.axhline(y=np.exp(-theoretical_bounds[i]), linestyle='--', 
                   color=f'C{i}', alpha=0.5, 
                   label=f'Theoretical bound for Agent {i+1}: {theoretical_bounds[i]:.4f}')
    
    plt.title(f'Mistake Probability Decay: {num_agents} Agents (Complete Network)')
    plt.xlabel('Time Step')
    plt.ylabel('Mistake Probability')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'learning_curves_{num_agents}_agents.png')
    plt.close()

def plot_learning_rate_vs_neighbors(results):
    """Plot how learning rate changes with number of neighbors"""
    plt.figure(figsize=(10, 6))
    
    num_neighbors = results['num_neighbors']
    
    # Average learning rates across agents for each network size
    avg_empirical = [np.mean(rates) for rates in results['empirical_rates']]
    avg_theoretical = [np.mean(bounds) for bounds in results['theoretical_bounds']]
    
    # Get error bars (std dev) for empirical rates
    err_empirical = [np.std(rates) for rates in results['empirical_rates']]
    
    # Plot results
    plt.errorbar(num_neighbors, avg_empirical, yerr=err_empirical, fmt='o-', 
                label='Empirical Learning Rate')
    plt.plot(num_neighbors, avg_theoretical, 's--', label='Theoretical Bound')
    
    plt.title('Learning Rate vs Number of Neighbors')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig('learning_rate_vs_neighbors.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze learning rates in social networks')
    parser.add_argument('--agents-path', type=str, required=True,
                        help='Path to directory containing trained agents')
    parser.add_argument('--neighbors', type=str, default='1,2,4,8',
                        help='Comma-separated list of numbers of neighbors to analyze')
    parser.add_argument('--states', type=int, default=2,
                        help='Number of states in the environment')
    parser.add_argument('--signal-accuracy', type=float, default=0.7,
                        help='Accuracy of private signals')
    parser.add_argument('--timesteps', type=int, default=50,
                        help='Number of timesteps per episode')
    parser.add_argument('--episodes', type=int, default=30,
                        help='Number of episodes to average over')
    
    args = parser.parse_args()
    
    # Parse neighbors list
    num_neighbors_list = [int(n) for n in args.neighbors.split(',')]
    num_agents_list = [n + 1 for n in num_neighbors_list]  # Convert to number of agents
    
    # Run analysis
    results = run_analysis(
        agents_path=args.agents_path,
        num_neighbors_list=num_agents_list,
        num_states=args.states,
        signal_accuracy=args.signal_accuracy,
        num_timesteps=args.timesteps,
        num_episodes=args.episodes
    )
    
    # Save results to file
    with open('learning_rate_analysis.json', 'w') as f:
        json.dump({
            'num_neighbors': results['num_neighbors'],
            'empirical_rates': [[float(r) for r in rates] for rates in results['empirical_rates']],
            'theoretical_bounds': [[float(b) for b in bounds] for bounds in results['theoretical_bounds']]
        }, f, indent=2)
    
    print("Analysis complete. Results saved to learning_rate_analysis.json") 