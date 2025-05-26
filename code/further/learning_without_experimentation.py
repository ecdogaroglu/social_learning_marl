import numpy as np
import gym
from gym import spaces
import networkx as nx

class LearningWithoutExperimentationEnv(gym.Env):
    """
    Multi-agent environment for learning without experimentation.
    
    Agents receive private signals about a fixed state of the world and observe
    the actions of their neighbors, but don't receive direct rewards from their actions.
    """
    
    def __init__(self, num_agents=5, num_states=2, signal_accuracy=0.7, 
                 network_type='complete', directed=False):
        """
        Initialize the environment.
        
        Args:
            num_agents: Number of agents
            num_states: Number of possible states of the world
            signal_accuracy: Accuracy of private signals
            network_type: Type of network topology ('complete', 'line', 'star', 'ring')
            directed: Whether the network is directed
        """
        super(LearningWithoutExperimentationEnv, self).__init__()
        
        self.num_agents = num_agents
        self.num_states = num_states
        self.signal_accuracy = signal_accuracy
        
        # Create social network
        self.network = self._create_network(network_type, directed)
        
        # Define action and observation spaces
        # Actions are discrete choices corresponding to states
        self.action_space = spaces.Discrete(num_states)
        
        # Observations consist of private signal and neighbor actions
        # Private signal is one-hot encoding of observed state
        # Neighbor actions are one-hot encodings of their last actions
        max_neighbors = max(len(list(self.network.neighbors(i))) for i in range(num_agents))
        self.observation_space = spaces.Dict({
            'private_signal': spaces.Box(low=0, high=1, shape=(num_states,), dtype=np.float32),
            'neighbor_actions': spaces.Box(low=0, high=1, 
                                         shape=(max_neighbors, num_states), dtype=np.float32)
        })
        
        # Initialize state
        self.reset()
        
    def _create_network(self, network_type, directed):
        """Create the social network based on the specified topology"""
        if network_type == 'complete':
            G = nx.complete_graph(self.num_agents)
        elif network_type == 'line':
            G = nx.path_graph(self.num_agents)
        elif network_type == 'star':
            G = nx.star_graph(self.num_agents - 1)
        elif network_type == 'ring':
            G = nx.cycle_graph(self.num_agents)
        else:
            raise ValueError(f"Unknown network type: {network_type}")
            
        if directed and not G.is_directed():
            G = G.to_directed()
            
        return G
    
    def reset(self):
        """Reset the environment and return initial observations"""
        # Draw random state of the world
        self.true_state = np.random.randint(0, self.num_states)
        
        # Initialize agent actions to None (no action taken yet)
        self.last_actions = [None] * self.num_agents
        
        # Generate observations for all agents
        observations = []
        for i in range(self.num_agents):
            # Generate private signal about the state
            signal = np.zeros(self.num_states)
            if np.random.random() < self.signal_accuracy:
                observed_state = self.true_state
            else:
                # Draw from other states uniformly
                other_states = list(range(self.num_states))
                other_states.remove(self.true_state)
                observed_state = np.random.choice(other_states)
            signal[observed_state] = 1
            
            # Get neighbor actions (initially all None)
            neighbors = list(self.network.neighbors(i))
            neighbor_actions = np.zeros((len(neighbors), self.num_states))
            
            obs = {
                'private_signal': signal,
                'neighbor_actions': neighbor_actions
            }
            
            observations.append(obs)
            
        self.t = 0
        return observations
        
    def step(self, actions):
        """
        Execute actions and return observations, rewards, done, and info
        
        Args:
            actions: List of actions for each agent (integer in [0, num_states-1])
        
        Returns:
            observations: List of observations for each agent
            rewards: List of computed rewards for each agent (not directly observed)
            done: Whether the episode is done
            info: Additional information
        """
        self.t += 1
        self.last_actions = actions.copy() if isinstance(actions, list) else actions
        
        # Calculate rewards using the observed reward function
        rewards = []
        for i, action in enumerate(self.last_actions):
            # Generate observation for reward calculation
            signal = np.zeros(self.num_states)
            if np.random.random() < self.signal_accuracy:
                observed_state = self.true_state
            else:
                other_states = list(range(self.num_states))
                other_states.remove(self.true_state)
                observed_state = np.random.choice(other_states)
            signal[observed_state] = 1
            
            # Calculate reward using the observed reward function
            reward = self._observed_reward_function(signal, action)
            rewards.append(reward)
            
        # Generate new observations for all agents
        observations = []
        for i in range(self.num_agents):
            # Generate new private signal
            signal = np.zeros(self.num_states)
            if np.random.random() < self.signal_accuracy:
                observed_state = self.true_state
            else:
                other_states = list(range(self.num_states))
                other_states.remove(self.true_state)
                observed_state = np.random.choice(other_states)
            signal[observed_state] = 1
            
            # Get neighbor actions
            neighbors = list(self.network.neighbors(i))
            neighbor_actions = np.zeros((len(neighbors), self.num_states))
            for j, neighbor in enumerate(neighbors):
                if self.last_actions[neighbor] is not None:
                    neighbor_actions[j, self.last_actions[neighbor]] = 1
            
            obs = {
                'private_signal': signal,
                'neighbor_actions': neighbor_actions
            }
            
            observations.append(obs)
            
        # Episode terminates after a fixed number of steps
        done = self.t >= 100
        
        # Additional info
        info = {
            "true_state": self.true_state,
            "accuracy": np.mean([1 if a == self.true_state else 0 for a in self.last_actions])
        }
        
        return observations, rewards, done, info
    
    def _observed_reward_function(self, observation, action):
        """
        Compute the observed reward function v(o, a) that preserves expected rewards.
        
        For a binary case (num_states=2), the function is:
        v(o, a) = (q * 1{a = φ(o)} - (1-q) * 1{a ≠ φ(o)}) / (2q - 1)
        
        where q is the signal accuracy and φ maps observations to corresponding actions.
        """
        q = self.signal_accuracy
        observed_state = np.argmax(observation)
        
        if self.num_states == 2:
            # Binary case
            if action == observed_state:
                return (q) / (2 * q - 1)
            else:
                return -(1 - q) / (2 * q - 1)
        else:
            # General case
            # Create mapping matrix M where M[i,j] = P(o_j | s_i)
            M = np.full((self.num_states, self.num_states), (1 - q) / (self.num_states - 1))
            np.fill_diagonal(M, q)
            
            # Create utility vector u where u[i] = 1 if i = true_state, 0 otherwise
            u = np.zeros(self.num_states)
            u[self.true_state] = 1
            
            # Calculate reward as v[o] = M^(-1) * u
            M_inv = np.linalg.inv(M)
            v = M_inv.dot(u)
            
            return v[observed_state] if action == observed_state else 0