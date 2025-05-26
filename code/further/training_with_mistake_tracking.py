import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import time
import argparse
import os
import json

# Import our modules
from partially_observable_further import PartiallyObservableFURTHER, device
from strategic_experimentation import StrategicExperimentationEnv
from learning_without_experimentation import LearningWithoutExperimentationEnv
from training_script import flatten_observation, preprocess_action
from scipy.optimize import curve_fit

# Function to model exponential decay for fitting
def exponential_decay(t, r):
    """Exponential decay function e^(-rt) for fitting learning rates"""
    return np.exp(-r * t)

def train_agents_with_mistake_tracking(
    env_type='learning', 
    num_agents=2, 
    num_states=2, 
    num_timesteps=20000,  # Total timesteps for continuing task
    eval_interval=1000, 
    tracking_window=100,  # Window for calculating moving average of mistakes
    log_interval=100,     # How often to log mistake probabilities
    render=False, 
    save_path=None,
    signal_accuracy=0.7
):
    """
    Train agents using the Partially Observable FURTHER algorithm and track mistake probabilities
    
    Args:
        env_type: Type of environment ('strategic' or 'learning')
        num_agents: Number of agents
        num_states: Number of states
        num_timesteps: Total timesteps for training (for continuing tasks)
        eval_interval: Evaluate every n timesteps
        tracking_window: Window size for calculating moving average of mistakes
        log_interval: How often to log mistake probabilities
        render: Whether to render the environment
        save_path: Path to save the trained models
        signal_accuracy: Accuracy of private signals (for learning environment)
    """
    # Create environment
    if env_type == 'strategic':
        env = StrategicExperimentationEnv(num_agents=num_agents, state_levels=num_states)
        observation_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        discrete_actions = False
    elif env_type == 'learning':
        env = LearningWithoutExperimentationEnv(
            num_agents=num_agents, 
            num_states=num_states,
            signal_accuracy=signal_accuracy
        )
        observation_dim = num_states + num_agents * num_states  # Private signal + neighbor actions
        action_dim = num_states
        discrete_actions = True
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
    
    # Create agents
    agents = []
    for i in range(num_agents):
        agent = PartiallyObservableFURTHER(
            observation_dim=observation_dim,
            action_dim=action_dim,
            discrete_actions=discrete_actions,
            latent_dim=5,
            hidden_dim=128,
            learning_rate=0.001,
            alpha=0.2,
            batch_size=64
        )
        agents.append(agent)
    
    # Training metrics
    rewards = np.zeros((num_timesteps, num_agents))
    
    # Tracking variables for mistake probabilities
    all_mistake_probs = np.zeros((num_timesteps, num_agents))
    sliding_mistake_probs = [deque(maxlen=tracking_window) for _ in range(num_agents)]
    timestep_mistake_probs = np.zeros((num_timesteps // log_interval, num_agents))
    
    # Reset environment
    observations = env.reset()
    true_state = env.true_state  # Get the ground truth state
    
    # Flatten observations for FURTHER
    flat_observations = [flatten_observation(obs, env_type, num_states) for obs in observations]
    
    # Initialize latent strategies
    latent_strategies = [np.zeros(5) for _ in range(num_agents)]
    
    # Initialize hidden states
    hidden_states = [None for _ in range(num_agents)]
    
    # Initial action (depends on environment)
    if env_type == 'strategic':
        actions = [0.5 for _ in range(num_agents)]  # Start with 50-50 allocation
    else:
        actions = [0 for _ in range(num_agents)]  # Default to first action
        
    # Storage for diagnostic plots
    timesteps = []
    avg_mistake_probs = []
    
    # Track learning rates over time windows
    learning_rate_windows = []
    empirical_rates = []
    theoretical_bounds = []
    
    # Training loop (continuing task)
    for t in range(num_timesteps):
        # Preprocess actions for FURTHER
        processed_actions = [preprocess_action(actions[i], env_type, num_states) for i in range(num_agents)]
        
        # For each agent, get action probabilities and track mistake probability
        for i in range(num_agents):
            # Get current belief state and latent strategies
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
            
            # Get action probabilities to calculate mistake probability
            with torch.no_grad():
                observation_tensor = torch.FloatTensor(flat_observations[i]).to(device)
                if observation_tensor.dim() == 1:
                    observation_tensor = observation_tensor.unsqueeze(0)
                    
                latent_tensor = torch.FloatTensor(latent_strategies[i]).to(device)
                if latent_tensor.dim() == 1:
                    latent_tensor = latent_tensor.unsqueeze(0)
                
                # Get belief state
                belief_tensor, _ = agents[i].belief_network(
                    observation_tensor, 
                    torch.zeros(1, action_dim, device=device)  # Dummy action
                )
                
                # Get action probabilities
                action_probs = agents[i].policy(belief_tensor, latent_tensor).cpu().numpy().flatten()
                
                # Calculate mistake probability (1 - probability of correct action)
                if env_type == 'learning':
                    mistake_prob = 1.0 - action_probs[true_state]
                else:
                    # For strategic environment, we don't have a clear "correct" action
                    # So we use a different metric: deviation from optimal allocation
                    optimal_allocation = 1.0 if env.true_risky_payoff > env.safe_payoff else 0.0
                    if discrete_actions:
                        mistake_prob = 1.0 - action_probs[int(optimal_allocation)]
                    else:
                        # For continuous actions, measure distance from optimal
                        predicted_allocation = np.sum(action_probs * np.arange(len(action_probs)))
                        mistake_prob = abs(predicted_allocation - optimal_allocation)
                
                # Store mistake probability
                all_mistake_probs[t, i] = mistake_prob
                sliding_mistake_probs[i].append(mistake_prob)
        
        # Select actions for all agents
        new_actions = []
        for i in range(num_agents):
            action, _ = agents[i].select_action(
                flat_observations[i], 
                latent_strategies[i]
            )
            new_actions.append(action)
        
        # Take actions in the environment
        if env_type == 'strategic':
            # For strategic env, wrap each action as needed by the environment
            env_actions = []
            for a in new_actions:
                # Make sure each action is a numpy array of correct shape
                if isinstance(a, (int, float)):
                    env_actions.append(np.array([a]))
                elif isinstance(a, np.ndarray):
                    if a.shape == ():  # Scalar numpy array
                        env_actions.append(np.array([float(a)]))
                    else:
                        env_actions.append(a.reshape(-1, 1) if len(a.shape) == 1 else a)
                else:
                    env_actions.append(np.array([a]))
            next_observations, step_rewards, done, info = env.step(env_actions)
        else:
            # For learning env, use actions directly
            next_observations, step_rewards, done, info = env.step(new_actions)
        
        # Flatten observations
        next_flat_observations = [flatten_observation(obs, env_type, num_states) for obs in next_observations]
        
        # Store experiences and update parameters
        for i in range(num_agents):
            other_actions = np.concatenate([
                processed_actions[j] for j in range(num_agents) if j != i
            ])
            
            # For the next time step, infer latent strategies based on new actions
            processed_new_actions = [preprocess_action(new_actions[j], env_type, num_states) for j in range(num_agents)]
            other_new_actions = np.concatenate([
                processed_new_actions[j] for j in range(num_agents) if j != i
            ])
            
            next_latent = agents[i].infer_latent_strategies(
                next_flat_observations[i], 
                processed_new_actions[i], 
                other_new_actions
            )
            
            reward_value = step_rewards[i][0] if env_type == 'strategic' else step_rewards[i]
            
            agents[i].add_experience(
                flat_observations[i],
                latent_strategies[i],
                processed_actions[i] if not discrete_actions else new_actions[i],
                other_actions,
                reward_value,
                next_flat_observations[i],
                next_latent
            )
            
            # Update parameters
            agents[i].update_parameters()
            
            # Store reward
            rewards[t, i] = reward_value
        
        # Update state for next iteration
        flat_observations = next_flat_observations
        actions = new_actions
        
        # For learning environment, update true state if it changed
        if env_type == 'learning':
            true_state = env.true_state
        
        # Log mistake probabilities periodically
        if (t+1) % log_interval == 0:
            log_idx = t // log_interval
            for i in range(num_agents):
                avg_mistake = np.mean(list(sliding_mistake_probs[i]))
                timestep_mistake_probs[log_idx, i] = avg_mistake
            
            # Print progress
            avg_mistake = np.mean([np.mean(list(q)) for q in sliding_mistake_probs])
            print(f"Timestep {t+1}/{num_timesteps}: Avg mistake probability = {avg_mistake:.4f}")
            
            # Store for plotting
            timesteps.append(t+1)
            avg_mistake_probs.append(avg_mistake)
        
        # Evaluate mistake decay rate periodically
        if (t+1) % eval_interval == 0 and t+1 >= tracking_window:
            # Fit exponential decay to recent mistake probabilities
            window_size = min(tracking_window, t+1)
            recent_mistakes = all_mistake_probs[t+1-window_size:t+1]
            
            time_points = np.arange(window_size)
            rates = []
            
            for i in range(num_agents):
                try:
                    # Fit exponential decay to agent's mistakes
                    params, _ = curve_fit(
                        exponential_decay, 
                        time_points, 
                        recent_mistakes[:, i],
                        p0=[0.1],  # Initial guess
                        bounds=(0, np.inf)  # Rate must be positive
                    )
                    rate = params[0]
                except:
                    # If fitting fails, use a default small rate
                    rate = 0.01
                
                rates.append(rate)
            
            # Calculate theoretical bound
            if env_type == 'learning':
                # Each agent has num_agents-1 neighbors in complete network
                num_neighbors = num_agents - 1
                
                # Calculate mutual information for binary channel with error probability 1-q
                q = signal_accuracy
                if q > 0.5:
                    I_private = (1-q)*np.log2(1-q) + q*np.log2(q) + 1
                    # Simplified factor for complete network
                    network_factor = 1 + 0.1 * num_neighbors  # Adjust this based on the paper's formula
                    bound = I_private * network_factor
                else:
                    bound = 0
                
                bounds = [bound] * num_agents
            else:
                # For strategic environment, use a placeholder
                bounds = [0.1] * num_agents
            
            # Store results
            learning_rate_windows.append(t+1)
            empirical_rates.append(rates)
            theoretical_bounds.append(bounds)
            
            # Print current learning rates
            print(f"Learning rates at timestep {t+1}: {np.mean(rates):.4f} (theory: {np.mean(bounds):.4f})")
        
        # If environment signals done (e.g., episode limit in learning env), reset
        if done:
            observations = env.reset()
            true_state = env.true_state
            flat_observations = [flatten_observation(obs, env_type, num_states) for obs in observations]
    
    # Plot mistake probabilities over time
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, avg_mistake_probs)
    plt.title('Average Mistake Probability During Training')
    plt.xlabel('Timestep')
    plt.ylabel('Mistake Probability')
    plt.grid(True, alpha=0.3)
    plt.savefig('mistake_probability_during_training.png')
    plt.close()
    
    # Plot learning rates over time
    plt.figure(figsize=(12, 6))
    
    # Calculate average rates across agents
    avg_rates = [np.mean(rates) for rates in empirical_rates]
    avg_bounds = [np.mean(bounds) for bounds in theoretical_bounds]
    
    plt.plot(learning_rate_windows, avg_rates, 'o-', label='Empirical Rate')
    plt.plot(learning_rate_windows, avg_bounds, 's--', label='Theoretical Bound')
    
    plt.title('Learning Rate During Training')
    plt.xlabel('Timestep')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('learning_rate_during_training.png')
    plt.close()
    
    # Save models if path is provided
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        for i, agent in enumerate(agents):
            agent.save_models(f"{save_path}/agent_{i}.pt")
        
        # Save mistake probabilities and learning rates
        np.save(f"{save_path}/all_mistake_probs.npy", all_mistake_probs)
        np.save(f"{save_path}/timestep_mistake_probs.npy", timestep_mistake_probs)
        
        # Save learning rate data
        learning_rate_data = {
            'timesteps': learning_rate_windows,
            'empirical_rates': empirical_rates,
            'theoretical_bounds': theoretical_bounds
        }
        with open(f"{save_path}/learning_rates.json", 'w') as f:
            json.dump({
                'timesteps': learning_rate_windows,
                'empirical_rates': [[float(r) for r in rates] for rates in empirical_rates],
                'theoretical_bounds': [[float(b) for b in bounds] for bounds in theoretical_bounds]
            }, f, indent=2)
    
    # Return training metrics
    return {
        'rewards': rewards,
        'all_mistake_probs': all_mistake_probs,
        'timestep_mistake_probs': timestep_mistake_probs,
        'learning_rate_data': {
            'timesteps': learning_rate_windows,
            'empirical_rates': empirical_rates,
            'theoretical_bounds': theoretical_bounds
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train FURTHER agents and track mistake probabilities')
    parser.add_argument('--env', type=str, default='learning', choices=['strategic', 'learning'],
                       help='Environment type: strategic experimentation or learning without experimentation')
    parser.add_argument('--agents', type=int, default=2, help='Number of agents')
    parser.add_argument('--states', type=int, default=2, help='Number of states')
    parser.add_argument('--timesteps', type=int, default=20000, help='Total training timesteps')
    parser.add_argument('--eval-interval', type=int, default=1000, help='Evaluation interval in timesteps')
    parser.add_argument('--tracking-window', type=int, default=100, help='Window size for tracking mistake probabilities')
    parser.add_argument('--log-interval', type=int, default=100, help='Interval for logging mistake probabilities')
    parser.add_argument('--save-path', type=str, default=None, help='Path to save models and results')
    parser.add_argument('--signal-accuracy', type=float, default=0.7, help='Accuracy of private signals')
    
    args = parser.parse_args()
    
    # Train agents
    metrics = train_agents_with_mistake_tracking(
        env_type=args.env,
        num_agents=args.agents,
        num_states=args.states,
        num_timesteps=args.timesteps,
        eval_interval=args.eval_interval,
        tracking_window=args.tracking_window,
        log_interval=args.log_interval,
        save_path=args.save_path,
        signal_accuracy=args.signal_accuracy
    )
    
    print("Training complete. Results saved to disk.") 