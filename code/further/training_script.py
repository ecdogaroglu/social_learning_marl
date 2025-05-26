import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import time
import argparse

# Import our modules - make sure these match your file names
from partially_observable_further import PartiallyObservableFURTHER
from strategic_experimentation import StrategicExperimentationEnv
from learning_without_experimentation import LearningWithoutExperimentationEnv

def flatten_observation(obs, env_type, num_states):
    """Helper function to flatten observations for different environment types"""
    if env_type == 'strategic':
        # For strategic environment, ensure observation is the correct shape
        if isinstance(obs, np.ndarray):
            return obs.flatten()  # Flatten to 1D array to ensure consistent shape
        else:
            return np.array(obs).flatten()
    elif env_type == 'learning':
        # Concatenate private signal and flattened neighbor actions
        return np.concatenate([
            obs['private_signal'],
            obs['neighbor_actions'].flatten()
        ])

def preprocess_action(action, env_type, num_states):
    """Helper function to preprocess actions for different environment types"""
    if env_type == 'strategic':
        # Ensure action is a flat array of the correct shape
        if isinstance(action, (int, float)):
            return np.array([action])
        elif isinstance(action, np.ndarray):
            return action.flatten()
        else:
            return np.array([action])
    elif env_type == 'learning':
        # Convert discrete action to one-hot
        if isinstance(action, int) and action < num_states:
            one_hot = np.zeros(num_states)
            one_hot[action] = 1
            return one_hot
        elif isinstance(action, np.ndarray) and action.size == 1:
            idx = int(action.item())
            one_hot = np.zeros(num_states)
            one_hot[idx] = 1
            return one_hot
        else:
            # If already one-hot or otherwise formatted, ensure it's correct shape
            return np.array(action).flatten()[:num_states]

def train_agents(env_type='strategic', num_agents=2, num_states=2, num_episodes=1000, 
                eval_interval=10, render=False, save_path=None):
    """
    Train agents using the Partially Observable FURTHER algorithm
    
    Args:
        env_type: Type of environment ('strategic' or 'learning')
        num_agents: Number of agents
        num_states: Number of states
        num_episodes: Number of training episodes
        eval_interval: Evaluate every n episodes
        render: Whether to render the environment
        save_path: Path to save the trained models
    """
    # Create environment
    if env_type == 'strategic':
        env = StrategicExperimentationEnv(num_agents=num_agents, state_levels=num_states)
        observation_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        discrete_actions = False
    elif env_type == 'learning':
        env = LearningWithoutExperimentationEnv(num_agents=num_agents, num_states=num_states)
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
    episode_rewards = np.zeros((num_episodes, num_agents))
    episode_accuracies = np.zeros(num_episodes)
    
    # Evaluation metrics
    eval_rewards = []
    eval_accuracies = []
    
    # Training loop
    for episode in range(num_episodes):
        # Reset environment
        observations = env.reset()
        
        # Flatten observations for FURTHER
        flat_observations = [flatten_observation(obs, env_type, num_states) for obs in observations]
        
        # Initialize latent strategies
        latent_strategies = [np.zeros(5) for _ in range(num_agents)]
        
        # Initialize hidden states
        hidden_states = [None for _ in range(num_agents)]
        
        # Initial action (depends on environment)
        if env_type == 'strategic':
            actions = [0.5 for _ in range(num_agents)]  # Start with 50-50 allocation as a scalar
        else:
            actions = [0 for _ in range(num_agents)]  # Default to first action
            
        episode_reward = np.zeros(num_agents)
        done = False
        
        t = 0
        max_steps = 100  # Prevent infinite loops
        
        while not done and t < max_steps:
            t += 1
            
            # Preprocess actions for FURTHER
            processed_actions = [preprocess_action(actions[i], env_type, num_states) for i in range(num_agents)]
            
            # Update belief states
            belief_states = []
            for i in range(num_agents):
                belief, hidden = agents[i].update_belief(
                    flat_observations[i], 
                    processed_actions[i], 
                    hidden_states[i]
                )
                belief_states.append(belief)
                hidden_states[i] = hidden
            
            # Each agent observes other agents' actions
            for i in range(num_agents):
                other_actions = np.concatenate([
                    processed_actions[j] for j in range(num_agents) if j != i
                ])
                
                # Infer latent strategies
                latent_strategies[i] = agents[i].infer_latent_strategies(
                    flat_observations[i], 
                    processed_actions[i], 
                    other_actions
                )
            
            # Select actions
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
                next_observations, rewards, done, info = env.step(env_actions)
            else:
                # For learning env, use actions directly
                next_observations, rewards, done, info = env.step(new_actions)
            
            # Flatten observations
            next_flat_observations = [flatten_observation(obs, env_type, num_states) for obs in next_observations]
            
            # Store experiences
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
                
                reward_value = rewards[i][0] if env_type == 'strategic' else rewards[i]
                
                agents[i].add_experience(
                    flat_observations[i],
                    latent_strategies[i],
                    processed_actions[i] if not discrete_actions else new_actions[i],
                    other_actions,
                    reward_value,
                    next_flat_observations[i],
                    next_latent
                )
            
            # Update parameters for all agents
            for agent in agents:
                agent.update_parameters()
            
            # Update state for next iteration
            flat_observations = next_flat_observations
            actions = new_actions
            
            # Accumulate rewards
            for i in range(num_agents):
                if env_type == 'strategic':
                    episode_reward[i] += rewards[i][0]
                else:
                    episode_reward[i] += rewards[i]
        
        # Store episode metrics
        episode_rewards[episode] = episode_reward
        if env_type == 'learning':
            episode_accuracies[episode] = info['accuracy']
        
        # Evaluate every eval_interval episodes
        if episode % eval_interval == 0 or episode == num_episodes - 1:
            eval_reward, eval_accuracy = evaluate(
                env, agents, env_type, num_agents, num_states, num_eval_episodes=5
            )
            eval_rewards.append(eval_reward)
            eval_accuracies.append(eval_accuracy)
            
            print(f"Episode {episode}: Reward = {eval_reward}, Accuracy = {eval_accuracy}")
    
    # Save models if path is provided
    if save_path is not None:
        for i, agent in enumerate(agents):
            agent.save_models(f"{save_path}/agent_{i}.pt")
    
    # Return training metrics
    return {
        'episode_rewards': episode_rewards,
        'episode_accuracies': episode_accuracies,
        'eval_rewards': eval_rewards,
        'eval_accuracies': eval_accuracies
    }

def evaluate(env, agents, env_type, num_agents, num_states, num_eval_episodes=5):
    """Evaluate the performance of the agents"""
    total_rewards = np.zeros(num_agents)
    total_accuracy = 0.0
    
    for _ in range(num_eval_episodes):
        observations = env.reset()
        flat_observations = [flatten_observation(obs, env_type, num_states) for obs in observations]
        
        # Initialize latent strategies and belief states
        latent_strategies = [np.zeros(5) for _ in range(num_agents)]
        hidden_states = [None for _ in range(num_agents)]
        
        # Initialize actions
        if env_type == 'strategic':
            actions = [0.5 for _ in range(num_agents)]
        else:
            actions = [0 for _ in range(num_agents)]
            
        episode_rewards = np.zeros(num_agents)
        done = False
        
        t = 0
        max_steps = 100  # Prevent infinite loops
        
        while not done and t < max_steps:
            t += 1
            
            # Preprocess actions
            processed_actions = [preprocess_action(actions[i], env_type, num_states) for i in range(num_agents)]
            
            # Update belief states
            for i in range(num_agents):
                _, hidden = agents[i].update_belief(
                    flat_observations[i],
                    processed_actions[i],
                    hidden_states[i]
                )
                hidden_states[i] = hidden
            
            # Each agent observes other agents' actions
            for i in range(num_agents):
                other_actions = np.concatenate([
                    processed_actions[j] for j in range(num_agents) if j != i
                ])
                
                # Infer latent strategies
                latent_strategies[i] = agents[i].infer_latent_strategies(
                    flat_observations[i],
                    processed_actions[i],
                    other_actions
                )
            
            # Select actions (using evaluation mode)
            new_actions = []
            for i in range(num_agents):
                action, _ = agents[i].select_action(
                    flat_observations[i],
                    latent_strategies[i],
                    evaluate=True
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
                next_observations, rewards, done, info = env.step(env_actions)
            else:
                next_observations, rewards, done, info = env.step(new_actions)
            
            # Update state for next iteration
            flat_observations = [flatten_observation(obs, env_type, num_states) for obs in next_observations]
            actions = new_actions
            
            # Accumulate rewards
            for i in range(num_agents):
                if env_type == 'strategic':
                    episode_rewards[i] += rewards[i][0]
                else:
                    episode_rewards[i] += rewards[i]
        
        # Store episode metrics
        total_rewards += episode_rewards
        if env_type == 'learning':
            total_accuracy += info['accuracy']
    
    # Calculate averages
    avg_rewards = total_rewards / num_eval_episodes
    avg_accuracy = total_accuracy / num_eval_episodes if env_type == 'learning' else 0.0
    
    return avg_rewards, avg_accuracy

def plot_results(metrics, env_type, eval_interval=10, save_path=None):
    """Plot training metrics"""
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 1, 1)
    for i in range(metrics['episode_rewards'].shape[1]):
        plt.plot(metrics['episode_rewards'][:, i], label=f'Agent {i+1}')
    plt.plot([i * eval_interval for i in range(len(metrics['eval_rewards']))],
             [np.mean(r) for r in metrics['eval_rewards']], 'k--', label='Eval')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Training Rewards')
    plt.legend()
    
    # Plot accuracy for learning without experimentation
    if env_type == 'learning':
        plt.subplot(2, 1, 2)
        plt.plot(metrics['episode_accuracies'], label='Training')
        plt.plot([i * eval_interval for i in range(len(metrics['eval_accuracies']))],
                metrics['eval_accuracies'], 'k--', label='Eval')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')
        plt.title('State Prediction Accuracy')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train FURTHER agents in partially observable environments')
    parser.add_argument('--env', type=str, default='strategic', choices=['strategic', 'learning'],
                       help='Environment type: strategic experimentation or learning without experimentation')
    parser.add_argument('--agents', type=int, default=2, help='Number of agents')
    parser.add_argument('--states', type=int, default=2, help='Number of states')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--eval-interval', type=int, default=10, help='Evaluation interval')
    parser.add_argument('--save-path', type=str, default=None, help='Path to save models and results')
    
    args = parser.parse_args()
    
    # Set global variables
    env_type = args.env
    num_agents = args.agents
    num_states = args.states
    num_episodes = args.episodes
    eval_interval = args.eval_interval
    save_path = args.save_path
    
    # Train agents
    metrics = train_agents(
        env_type=env_type,
        num_agents=num_agents,
        num_states=num_states,
        num_episodes=num_episodes,
        eval_interval=eval_interval,
        save_path=save_path
    )
    
    # Plot results
    plot_results(metrics, env_type, eval_interval, save_path=f"{save_path}/results.png" if save_path else None)