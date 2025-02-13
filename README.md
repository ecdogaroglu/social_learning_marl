# Social Learning via Multi-Agent Deep Reinforcement Learning

This repository implements a deep reinforcement learning approach to social learning in networks, as part of my ongoing master's thesis. The implementation currently handels the single-agent case with a binary state space and uses recurrent neural networks (RNNs) to process sequential observations.

## Theory Background

The model considers agents who must learn the true state of the world through:
- Private signals received in each period
- Observations of other agents' actions
- A reward function that provides feedback on their decisions

Key theoretical aspects:
- Agents receive signals with accuracy q > 0.5
- The true state ω remains fixed throughout the learning process
- Learning rate is measured by: r = liminf -1/t * log P(mistake)
- There exists a theoretical upper bound (r_bdd) on the learning rate

## Implementation Details

### Environment
- Binary state space (ω ∈ {0,1})
- Signal accuracy parameter q (default: 0.75)
- Reward function v(a,s) = (q·1{a=s} - (1-q)·1{a≠s})/(2q-1)

### Neural Architecture
The implementation uses two main components:

1. **Actor Network**
   - GRU-based RNN for processing observation history
   - Policy head that outputs action probabilities
   - Parameters optimized using policy gradient

2. **Critic Network**
   - GRU-based RNN for value estimation
   - Value head for computing state-value estimates
   - Trained using temporal difference learning

### Key Features
- Custom metrics tracking for mistake rates and learning rates
- Visualization tools for comparing empirical performance to theoretical bounds
- Implementation of the paper's reward function for binary case
- Efficient memory management using RNNs instead of growing state spaces

## Requirements

```
numpy
torch
matplotlib
```

## Usage

Basic usage example:

```python
# Train an agent
metrics = train(num_steps=10000, signal_accuracy=0.75)

# Plot learning curves
plot_metrics(metrics, signal_accuracy=0.75)
```

## Implementation Classes

### MetricsTracker
Tracks and computes various performance metrics including:
- Mistake rates over time
- Learning rates as defined in the paper
- Maintains full history of mistakes

### Environment
Implements the binary state environment with:
- Fixed true state
- Signal generation based on accuracy parameter
- Reward computation following the paper's formulation

### Agent
Combines actor and critic networks with:
- GRU-based memory for processing observation history
- Policy gradient optimization
- Value function learning

## Results Visualization

The implementation includes plotting functionality that shows:
1. Empirical mistake rate over time
2. Empirical learning rate compared to theoretical bound
3. Visualization of the learning process

## Theoretical Bounds

For the binary case with signal accuracy q, the theoretical bound on the learning rate is:
```
r_bdd = -log(1 - (2q - 1))
```

This bound represents the fundamental limit on how quickly any agent can learn the true state, regardless of their strategy.

