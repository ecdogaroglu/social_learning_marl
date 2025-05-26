import numpy as np
import gym
from gym import spaces

class StrategicExperimentationEnv(gym.Env):
    """
    A multi-agent environment for strategic experimentation based on Keller et al. (2020).
    
    Each agent faces a two-armed bandit problem where they continuously allocate their resources
    between a safe arm with known payoff and a risky arm whose expected payoff depends on
    an unknown state of the world.
    """
    
    def __init__(self, num_agents=2, state_levels=2, observation_dim=3, action_dim=1, 
                 safe_payoff=1, risky_payoffs=None, noise_std=0.1, signal_accuracy=0.8):
        super(StrategicExperimentationEnv, self).__init__()
        
        self.num_agents = num_agents
        self.state_levels = state_levels
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.safe_payoff = safe_payoff
        
        # If risky payoffs not provided, generate them
        if risky_payoffs is None:
            self.risky_payoffs = np.linspace(0, 2 * safe_payoff, state_levels)
        else:
            self.risky_payoffs = risky_payoffs
            
        self.noise_std = noise_std
        self.signal_accuracy = signal_accuracy
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=0, high=1, shape=(action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                           shape=(observation_dim,), dtype=np.float32)
        
        # Initialize state
        self.reset()
        
    def reset(self):
        """Reset the environment and return initial observations"""
        # Draw random state of the world
        self.true_state = np.random.randint(0, self.state_levels)
        self.true_risky_payoff = self.risky_payoffs[self.true_state]
        
        # Initialize observations for all agents
        observations = []
        for _ in range(self.num_agents):
            # Generate private signal about the state
            if np.random.random() < self.signal_accuracy:
                observed_state = self.true_state
            else:
                # Draw from other states uniformly
                other_states = list(range(self.state_levels))
                other_states.remove(self.true_state)
                observed_state = np.random.choice(other_states)
            
            # Initial observation includes:
            # 1. Private signal (one-hot encoding of observed state)
            # 2. Safe payoff (known)
            # 3. Placeholder for observed rewards (initially 0)
            signal = np.zeros(self.state_levels)
            signal[observed_state] = 1
            
            obs = np.concatenate([
                signal,
                np.array([self.safe_payoff]),
                np.zeros(1)  # Placeholder for observed reward
            ])
            
            observations.append(obs)
            
        self.last_actions = np.zeros((self.num_agents, self.action_dim))
        self.total_rewards = np.zeros(self.num_agents)
        self.t = 0
        
        return observations
    
    def step(self, actions):
        """
        Execute actions and return observations, rewards, done, and info
        
        Args:
            actions: List of actions for each agent, each action is a float in [0, 1]
                    representing allocation to the risky arm
        
        Returns:
            observations: List of observations for each agent
            rewards: List of rewards for each agent
            done: Whether the episode is done
            info: Additional information
        """
        self.t += 1
        
        # Ensure actions are properly formatted
        processed_actions = []
        for action in actions:
            # Convert to numpy array if not already
            if not isinstance(action, np.ndarray):
                action = np.array([action])
            # Ensure it's a 1D array
            action = action.flatten()
            # Clip to valid range [0, 1]
            action = np.clip(action, 0, 1)
            processed_actions.append(action)
            
        self.last_actions = np.array(processed_actions)
        
        # Calculate rewards for each agent
        rewards = []
        for i, action in enumerate(processed_actions):
            # Get allocation to risky arm (first value if array)
            allocation_to_risky = action[0]
            allocation_to_safe = 1 - allocation_to_risky
            
            # Calculate payoff from safe arm (deterministic)
            safe_payoff = allocation_to_safe * self.safe_payoff
            
            # Calculate payoff from risky arm (stochastic)
            risky_mean_payoff = allocation_to_risky * self.true_risky_payoff
            risky_noise = np.random.normal(0, self.noise_std * allocation_to_risky)
            risky_payoff = risky_mean_payoff + risky_noise
            
            # Total reward
            reward = safe_payoff + risky_payoff
            rewards.append(np.array([reward]))
            
            self.total_rewards[i] += reward
            
        # Generate new observations for all agents
        observations = []
        for i in range(self.num_agents):
            # Agents observe:
            # 1. Their private signal (unchanged)
            # 2. Safe payoff (unchanged)
            # 3. Their own reward
            signal = np.zeros(self.state_levels)
            if np.random.random() < self.signal_accuracy:
                observed_state = self.true_state
            else:
                other_states = list(range(self.state_levels))
                other_states.remove(self.true_state)
                observed_state = np.random.choice(other_states)
            signal[observed_state] = 1
            
            obs = np.concatenate([
                signal,
                np.array([self.safe_payoff]),
                rewards[i]
            ])
            
            observations.append(obs)
            
        # Episode never terminates (continuing task)
        done = False
        
        # Additional info
        info = {
            "true_state": self.true_state,
            "true_risky_payoff": self.true_risky_payoff,
            "total_rewards": self.total_rewards,
            "average_rewards": self.total_rewards / self.t
        }
        
        return observations, rewards, done, info