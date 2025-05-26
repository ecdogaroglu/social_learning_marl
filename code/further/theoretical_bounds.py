import numpy as np
import matplotlib.pyplot as plt
import argparse

def calculate_mutual_information(q):
    """
    Calculate mutual information between state and private signal
    
    Args:
        q: signal accuracy
        
    Returns:
        I: mutual information
    """
    if q <= 0.5 or q >= 1:
        return 0.0
    
    # For binary state/signal setting, mutual information is:
    # I(S;X) = H(X) - H(X|S)
    # For binary symmetric channel with error probability 1-q:
    h_x_given_s = -(q * np.log2(q) + (1-q) * np.log2(1-q))
    h_x = 1  # Binary entropy is 1 bit
    
    return h_x - h_x_given_s

def calculate_learning_rate_bound(q, n_neighbors, network_type='complete'):
    """
    Calculate theoretical upper bound on learning rate
    
    Args:
        q: signal accuracy
        n_neighbors: number of neighbors
        network_type: type of network ('complete', 'star', etc.)
        
    Returns:
        r_bound: theoretical upper bound on learning rate
    """
    # Calculate mutual information from private signal
    I = calculate_mutual_information(q)
    
    # Factor for network structure
    if network_type == 'complete':
        # In complete network, all agents observe all others
        # The bound scales with the number of neighbors
        network_factor = 1 + 0.5 * n_neighbors
    elif network_type == 'star':
        # In star network, center sees all, leaves see only center
        network_factor = 2.0  # Simplified approximation
    else:
        # Default case
        network_factor = 1.0
    
    # Calculate bound from social learning barrier paper
    r_bound = I * network_factor
    
    return r_bound

def plot_learning_rate_vs_accuracy(n_neighbors_list=[1, 2, 4, 8], network_type='complete'):
    """
    Plot learning rate bound as a function of signal accuracy
    
    Args:
        n_neighbors_list: list of number of neighbors to plot
        network_type: type of network
    """
    # Signal accuracy range
    q_range = np.linspace(0.5, 1.0, 100)
    
    plt.figure(figsize=(10, 6))
    
    for n_neighbors in n_neighbors_list:
        # Calculate bounds for each accuracy
        bounds = [calculate_learning_rate_bound(q, n_neighbors, network_type) for q in q_range]
        
        # Plot bound
        plt.plot(q_range, bounds, label=f'{n_neighbors} neighbors')
    
    plt.title(f'Theoretical Learning Rate Bound vs Signal Accuracy ({network_type.capitalize()} Network)')
    plt.xlabel('Signal Accuracy (q)')
    plt.ylabel('Learning Rate Bound')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'learning_rate_vs_accuracy_{network_type}.png')
    plt.close()

def plot_learning_rate_vs_neighbors(q_list=[0.6, 0.7, 0.8, 0.9], max_neighbors=20, network_type='complete'):
    """
    Plot learning rate bound as a function of number of neighbors
    
    Args:
        q_list: list of signal accuracies to plot
        max_neighbors: maximum number of neighbors to plot
        network_type: type of network
    """
    # Number of neighbors range
    n_range = np.arange(1, max_neighbors + 1)
    
    plt.figure(figsize=(10, 6))
    
    for q in q_list:
        # Calculate bounds for each number of neighbors
        bounds = [calculate_learning_rate_bound(q, n, network_type) for n in n_range]
        
        # Plot bound
        plt.plot(n_range, bounds, label=f'q = {q}')
    
    plt.title(f'Theoretical Learning Rate Bound vs Number of Neighbors ({network_type.capitalize()} Network)')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Learning Rate Bound')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f'learning_rate_vs_neighbors_{network_type}.png')
    plt.close()

def plot_information_cascade_probability(q_list=[0.6, 0.7, 0.8, 0.9], max_agents=20):
    """
    Plot probability of information cascade as a function of number of agents
    
    Args:
        q_list: list of signal accuracies to plot
        max_agents: maximum number of agents to plot
    """
    # Number of agents range
    n_range = np.arange(2, max_agents + 1)
    
    plt.figure(figsize=(10, 6))
    
    for q in q_list:
        # Calculate cascade probability for each number of agents
        # This is a simplified model based on the paper
        cascade_probs = []
        for n in n_range:
            # Probability of cascade increases with number of agents and decreases with signal quality
            # This is a simplified formula - should be replaced with the actual formula from the paper
            p_cascade = 1 - np.exp(-(n-1) * (1-q))
            cascade_probs.append(p_cascade)
        
        # Plot probability
        plt.plot(n_range, cascade_probs, label=f'q = {q}')
    
    plt.title('Probability of Information Cascade vs Number of Agents')
    plt.xlabel('Number of Agents')
    plt.ylabel('Cascade Probability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('cascade_probability.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Calculate and visualize theoretical learning rate bounds')
    parser.add_argument('--network-type', type=str, default='complete',
                       choices=['complete', 'star', 'line', 'ring'],
                       help='Type of network structure')
    parser.add_argument('--max-neighbors', type=int, default=20,
                       help='Maximum number of neighbors to plot')
    parser.add_argument('--accuracies', type=str, default='0.6,0.7,0.8,0.9',
                       help='Comma-separated list of signal accuracies to plot')
    parser.add_argument('--neighbors', type=str, default='1,2,4,8',
                       help='Comma-separated list of neighbor counts to plot')
    
    args = parser.parse_args()
    
    # Parse accuracies and neighbors
    q_list = [float(q) for q in args.accuracies.split(',')]
    n_neighbors_list = [int(n) for n in args.neighbors.split(',')]
    
    # Generate plots
    plot_learning_rate_vs_accuracy(n_neighbors_list, args.network_type)
    plot_learning_rate_vs_neighbors(q_list, args.max_neighbors, args.network_type)
    plot_information_cascade_probability(q_list, args.max_neighbors)
    
    print("Theoretical analysis plots generated:")
    print(f"- learning_rate_vs_accuracy_{args.network_type}.png")
    print(f"- learning_rate_vs_neighbors_{args.network_type}.png")
    print("- cascade_probability.png")

if __name__ == "__main__":
    main() 