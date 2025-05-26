"""
Core simulation logic for POLARIS experiments.
"""

import numpy as np
import time
import torch
from tqdm import tqdm
from modules.utils import (
    create_output_directory,
    calculate_observation_dimension,
    encode_observation, 
    setup_random_seeds,
    calculate_theoretical_bounds,
    display_theoretical_bounds,
    write_config_file,
    flatten_episodic_metrics,
    save_final_models,
    save_checkpoint_models,
    set_metrics,
    reset_agent_internal_states,
    update_total_rewards,
    select_agent_actions,
    update_progress_display,
    store_transition_in_buffer,
    load_agent_models)
from modules.metrics import (
    initialize_metrics, 
    update_metrics, 
    calculate_agent_learning_rates_from_metrics,
    display_learning_rate_summary,
    prepare_serializable_metrics,
    save_metrics_to_file
)
from modules.plotting import generate_plots
from modules.agent import POLARISAgent
from modules.networks import TemporalGNN
from modules.replay_buffer import ReplayBuffer

def run_agents(env, args, training=True, model_path=None):
    """
    Run POLARIS agents in the social learning environment.
    
    Args:
        env: The social learning environment
        args: Command-line arguments
        training: Whether to train the agents (True) or just evaluate (False)
        model_path: Path to load models from (optional)
    
    Returns:
        learning_rates: Dictionary of learning rates for each agent
        serializable_metrics: Dictionary of metrics for JSON serialization
    """
    # Setup directory
    output_dir = create_output_directory(args, env, training)
    
    # Initialize agents and metrics
    obs_dim = calculate_observation_dimension(env)
    agents = initialize_agents(env, args, obs_dim)
    load_agent_models(agents, model_path, env.num_agents, training=training)
    metrics = initialize_metrics(env, args, training)

    # Calculate and display theoretical bounds
    theoretical_bounds = calculate_theoretical_bounds(env)
    display_theoretical_bounds(theoretical_bounds)
    
    replay_buffers = initialize_replay_buffers(agents, args, obs_dim)

    # Write configuration if training
    if training:
        write_config_file(args, env, theoretical_bounds, output_dir)


    print(f"Running {args.num_episodes} episode(s) with {args.horizon} steps per episode")
    
    # Initialize episodic metrics to store each episode separately
    episodic_metrics = {
        'episodes': []
    }
    
    # Episode loop
    for episode in range(args.num_episodes):
        # Set a different seed for each episode based on the base seed
        episode_seed = args.seed + episode
        setup_random_seeds(episode_seed, env)
        print(f"\nStarting episode {episode+1}/{args.num_episodes} with seed {episode_seed}")
        
        # Initialize fresh metrics for this episode
        metrics = initialize_metrics(env, args, training)
        
        # Run simulation for this episode
        observations, episode_metrics = run_simulation(
            env, agents, replay_buffers, metrics, args,
            output_dir, training
        )
        
        # Store this episode's metrics separately
        episodic_metrics['episodes'].append(episode_metrics)
    
    # Create a flattened version of metrics
    combined_metrics = flatten_episodic_metrics(episodic_metrics, env.num_agents)
    
    # Process results
    learning_rates = calculate_agent_learning_rates_from_metrics(combined_metrics)
    display_learning_rate_summary(learning_rates, theoretical_bounds['bound_rate'])
    
    # Save metrics and models
    serializable_metrics = prepare_serializable_metrics(
        combined_metrics, learning_rates, theoretical_bounds, args.horizon, training
    )
    
    # Also save the episodic metrics for more detailed analysis
    episodic_serializable_metrics = {
        'episodic_data': episodic_metrics,
        'learning_rates': learning_rates,
        'theoretical_bounds': theoretical_bounds,
        'episode_length': args.horizon,
        'num_episodes': args.num_episodes
    }
    
    save_metrics_to_file(serializable_metrics, output_dir, training)
    save_metrics_to_file(episodic_serializable_metrics, output_dir, training, filename='episodic_metrics.json')
    
    if training and args.save_model:
        save_final_models(agents, output_dir)
    
    # Generate plots with LaTeX style if requested
    generate_plots(
        combined_metrics, 
        env, 
        args, 
        output_dir, 
        training, 
        episodic_metrics,
        use_latex=args.use_tex if hasattr(args, 'use_tex') else False
    )
    
    return learning_rates, serializable_metrics

def run_simulation(env, agents, replay_buffers, metrics, args, output_dir, training):
    """Run the main simulation loop."""
    mode_str = "training" if training else "evaluation"
    
    print(f"Starting {mode_str} for {args.horizon} steps...")
    start_time = time.time()
    
    # Ensure EWC attributes exist
    if not hasattr(args, 'use_ewc'):
        args.use_ewc = False
    
    # Initialize environment and agents
    observations = env.initialize()
    total_rewards = np.zeros(env.num_agents) if training else None
    
    # Print the true state
    print(f"True state for this episode: {env.true_state}")
    
    # Check if this is a new true state for EWC
    new_true_state_detected = False
    is_first_state = False
    if training and args.use_ewc:
        for agent_id, agent in agents.items():
            # Check if this is the first state the agent has seen
            if not hasattr(agent, 'seen_states') or len(agent.seen_states) == 0:
                is_first_state = True
                # Initialize seen_states if it doesn't exist
                if not hasattr(agent, 'seen_states'):
                    agent.seen_states = set()
            
            # Observe the new state
            if agent.observe_new_true_state(env.true_state):
                new_true_state_detected = True
        
    # Set global metrics for access in other functions
    set_metrics(metrics)
    
    # Reset and initialize agent internal states
    reset_agent_internal_states(agents)
    
    # Set agents to appropriate mode
    for agent_id, agent in agents.items():
        if training:
            agent.set_train_mode()
        else:
            agent.set_eval_mode()
    
    # Main simulation loop
    steps_iterator = tqdm(range(args.horizon), desc="Training" if training else "Evaluating")
    
    # Calculate Fisher matrices if a new true state was detected
    if training and args.use_ewc and (new_true_state_detected or is_first_state):
        action_str = "New true state detected" if new_true_state_detected else "First true state encountered"
        print(f"{action_str}, calculating Fisher matrices for EWC...")
        
        # For the first state, we need to gather some experience first
        if is_first_state:
            # Run a few steps to populate the replay buffer
            warmup_steps = min(100, args.horizon // 2)
            print(f"First state encountered. Running {warmup_steps} warmup steps to gather experience...")
            
            # Run warmup steps
            for step in range(warmup_steps):
                # Get agent actions based on current observations
                actions, action_probs = get_agent_actions(agents, observations, step, training)
                
                # Execute actions in environment
                next_observations, rewards, done, info = env.step(actions, action_probs)
                
                # Update agent states and store transitions in replay buffer
                update_agents(agents, observations, next_observations, actions, rewards, 
                              replay_buffers, metrics, env, args, training, step)
                
                # Move to next step
                observations = next_observations
                
                # Stop if done
                if done:
                    break
        
        # Now calculate Fisher matrices
        for agent_id, agent in agents.items():
            # Use the global replay buffer for this agent
            if agent_id in replay_buffers and len(replay_buffers[agent_id]) > 0:
                agent.calculate_fisher_matrices(replay_buffers[agent_id])
            else:
                print(f"Skipping Fisher calculation for agent {agent_id}: Insufficient data")
    
    for step in steps_iterator:
        # Get agent actions based on current observations
        actions, action_probs = get_agent_actions(agents, observations, step, training)
        
        # Execute actions in environment
        next_observations, rewards, done, info = env.step(actions, action_probs)
        
        if training:
            # Accumulate rewards
            for agent_id, reward in rewards.items():
                total_rewards[agent_id] += reward
        
        # Update metrics
        update_metrics(metrics, info, env, actions, action_probs, step, training)
        
        # Store belief and latent states if in evaluation mode or plot_internal_states enabled
        if not training or args.plot_internal_states:
            store_agent_internal_states(agents, metrics)
        
        # Update agent states and store transitions in replay buffer
        update_agents(agents, observations, next_observations, actions, rewards, 
                        replay_buffers, metrics, env, args, training, step)
        
        # Move to next step
        observations = next_observations
        
        # Stop if done
        if done:
            break
            
    # End episode for each agent
    for agent_id, agent in agents.items():
        agent.end_episode()
        
    print(f"Completed {mode_str} in {time.time() - start_time:.2f} seconds")
    
    # Return both observations and metrics
    return observations, metrics

def update_agents(agents, observations, next_observations, actions, rewards, 
                        replay_buffers, metrics, env, args, training, step):
    """Update agent states and store transitions in replay buffer."""
    for agent_id, agent in agents.items():
        # Get current and next observations
        obs_data = observations[agent_id]
        next_obs_data = next_observations[agent_id]
        
        # Extract signals from observations
        signal = obs_data['signal']
        next_signal = next_obs_data['signal']
        
        # Get neighbor actions from the observation
        neighbor_actions = obs_data['neighbor_actions']
        next_neighbor_actions = next_obs_data['neighbor_actions']

        # Encode observations
        signal_encoded, actions_encoded = encode_observation(
            signal=signal,
            neighbor_actions=neighbor_actions,
            num_agents=env.num_agents,
            num_states=env.num_states
        )
        next_signal_encoded, _ = encode_observation(
            signal=next_signal,
            neighbor_actions=next_neighbor_actions,
            num_agents=env.num_agents,
            num_states=env.num_states
        )

        # Get current belief and latent states (before observation update)
        belief = agent.current_belief.detach().clone()  # Make a copy to ensure we have the pre-update state
        latent = agent.current_latent.detach().clone()

        # Update agent belief state
        next_belief, next_dstr = agent.observe(signal_encoded, actions_encoded)
        # Infer latent state for next observation
        # This ensures we're using the correct latent state for the next observation
        next_latent = agent.infer_latent(
            signal_encoded,
            actions_encoded,
            rewards[agent_id] if rewards else 0.0,
            next_signal_encoded
        )
            
        # Store internal states for visualization if requested (for both training and evaluation)

        if args.plot_internal_states and 'belief_states' in metrics:
            current_belief = agent.get_belief_state()
            current_latent = agent.get_latent_state()
            metrics['belief_states'][agent_id].append(current_belief.detach().cpu().numpy())
            metrics['latent_states'][agent_id].append(current_latent.detach().cpu().numpy())
            
            # Store belief distribution if available
            belief_distribution = agent.get_belief_distribution()
            metrics['belief_distributions'][agent_id].append(belief_distribution.detach().cpu().numpy())
                
            # Store opponent belief distribution if available
            opponent_belief_distribution = agent.get_opponent_belief_distribution()
            metrics['opponent_belief_distributions'][agent_id].append(opponent_belief_distribution.detach().cpu().numpy())
        
        # Store transition in replay buffer if training
        if training and agent_id in replay_buffers:
            
            # Get mean and logvar from inference
            mean, logvar = agent.get_latent_distribution_params()
            
            # Store transition
            store_transition_in_buffer(
                replay_buffers[agent_id],
                signal_encoded,
                actions_encoded,
                belief,
                latent,
                actions[agent_id],
                rewards[agent_id] if rewards else 0.0,
                next_signal_encoded,
                next_belief,
                next_latent,
                mean,
                logvar
            )
            
            # Update networks if enough samples
            if len(replay_buffers[agent_id]) > args.batch_size and step % args.update_interval == 0:
                # Sample a batch from the replay buffer
                batch = replay_buffers[agent_id].sample(args.batch_size)
                # Update network parameters
                agent.update(batch)

def configure_gnn(agent, args):
    """Configure the GNN parameters for an agent."""
    # Update the inference module with the specified parameters
    agent.inference_module = TemporalGNN(
        hidden_dim=args.hidden_dim,
        action_dim=agent.num_states,
        latent_dim=args.latent_dim,
        num_agents=args.num_agents,
        device=args.device,
        num_belief_states=agent.num_states,
        num_gnn_layers=args.gnn_layers,
        num_attn_heads=args.attn_heads,
        dropout=0.1,
        temporal_window_size=args.temporal_window
    ).to(args.device)
    
    # Update the optimizer to use the new inference module
    agent.inference_optimizer = torch.optim.Adam(
        agent.inference_module.parameters(),
        lr=args.learning_rate
    )

def initialize_agents(env, args, obs_dim):
    """Initialize POLARIS agents."""
    print(f"Initializing {env.num_agents} agents{'...' if args.eval_only else ' for evaluation...'}")
    
    # Log if using GNN
    if args.use_gnn:
        print(f"Using Graph Neural Network with {args.gnn_layers} layers, {args.attn_heads} attention heads, and temporal window of {args.temporal_window}")
    else:
        print("Using traditional encoder-decoder inference module")
    
    # Ensure EWC attributes exist
    if not hasattr(args, 'use_ewc'):
        args.use_ewc = False
    if not hasattr(args, 'ewc_online'):
        args.ewc_online = False
    if not hasattr(args, 'ewc_lambda'):
        args.ewc_lambda = 100.0
    if not hasattr(args, 'ewc_gamma'):
        args.ewc_gamma = 0.95
        
    # Log if using EWC
    if args.use_ewc:
        print(f"Using Elastic Weight Consolidation with lambda={args.ewc_lambda}" + 
              f"{', online mode with gamma=' + str(args.ewc_gamma) if args.ewc_online else ''}")
        
    agents = {}
    
    for agent_id in range(env.num_agents):
        agent = POLARISAgent(
            agent_id=agent_id,
            num_agents=env.num_agents,
            num_states=env.num_states,
            observation_dim=obs_dim,
            action_dim=env.num_actions,
            hidden_dim=args.hidden_dim,
            belief_dim=args.belief_dim,
            latent_dim=args.latent_dim,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            entropy_weight=args.entropy_weight,
            kl_weight=args.kl_weight,
            buffer_capacity=args.buffer_capacity,
            device=args.device,
            use_gnn=args.use_gnn,
            use_ewc=args.use_ewc,
            ewc_lambda=args.ewc_lambda,
            ewc_online=args.ewc_online,
            ewc_gamma=args.ewc_gamma
        )
        
        # Configure GNN parameters if using GNN
        if args.use_gnn and hasattr(agent, 'inference_module'):
            configure_gnn(agent, args)
            
        # Store the agent in the dictionary
        agents[agent_id] = agent
        
    return agents

def initialize_replay_buffers(agents, args, obs_dim):
    """Initialize replay buffers for training."""
    replay_buffers = {}
    
    for agent_id in agents:
        replay_buffers[agent_id] = ReplayBuffer(
            capacity=args.buffer_capacity,
            observation_dim=obs_dim,
            belief_dim=args.belief_dim,
            latent_dim=args.latent_dim,
            device=args.device,
            sequence_length=8  # Default sequence length for sampling
        )
    return replay_buffers

def get_agent_actions(agents, observations, step, training):
    """Get actions from agents based on current observations."""
    actions = {}
    action_probs = {}
    
    # Have each agent select an action
    for agent_id, agent in agents.items():
        # Extract observation for current agent
        signal = observations[agent_id]['signal']
        neighbor_actions = observations[agent_id]['neighbor_actions']
        
        # Encode the observation
        signal_encoded, actions_encoded = encode_observation(
            signal=signal,
            neighbor_actions=neighbor_actions,
            num_agents=len(agents),
            num_states=agent.num_states
        )
        
        # Update agent belief with new observation
        agent.observe(signal_encoded, actions_encoded)
        
        # Select action based on belief
        action, action_prob = agent.select_action()
        actions[agent_id] = action
        action_probs[agent_id] = action_prob
    
    return actions, action_probs

def store_agent_internal_states(agents, metrics):
    """Store the internal states of agents for analysis."""
    if 'belief_states' not in metrics:
        return
        
    for agent_id, agent in agents.items():
        # Store belief state
        current_belief = agent.get_belief_state()
        if current_belief is not None:
            metrics['belief_states'][agent_id].append(current_belief.detach().cpu().numpy())
            
        # Store latent state
        current_latent = agent.get_latent_state()
        if current_latent is not None:
            metrics['latent_states'][agent_id].append(current_latent.detach().cpu().numpy())
            
        # Store belief distribution
        belief_distribution = agent.get_belief_distribution()
        if belief_distribution is not None:
            metrics['belief_distributions'][agent_id].append(belief_distribution.detach().cpu().numpy())
            
        # Store opponent belief distribution
        opponent_belief_distribution = agent.get_opponent_belief_distribution()
        if opponent_belief_distribution is not None:
            metrics['opponent_belief_distributions'][agent_id].append(opponent_belief_distribution.detach().cpu().numpy())