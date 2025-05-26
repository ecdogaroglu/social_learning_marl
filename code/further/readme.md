# Partially Observable FURTHER Implementation

This repository implements the FURTHER (FUlly Reinforcing acTive influence witH averagE Reward) algorithm for partially observable multi-agent environments, as described in the theoretical framework. The implementation supports both social learning scenarios: strategic experimentation with observable rewards and learning without direct experimentation.

## Overview

The implementation consists of the following components:

1. **Partially Observable FURTHER Algorithm**: An extension of the FURTHER algorithm that handles partial observability through belief state tracking using recurrent neural networks.

2. **Strategic Experimentation Environment**: A multi-agent environment where agents learn through trial and error with observable rewards, based on the model in Keller et al. (2020).

3. **Learning Without Experimentation Environment**: A multi-agent environment where agents learn primarily through observing others' actions, rather than direct reward feedback.

4. **Training Script**: A script to train and evaluate agents in either environment.

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Gym
- NetworkX
- Matplotlib

## Files Description

- `partially_observable_further.py`: The main implementation of the FURTHER algorithm adapted for partially observable settings.
- `strategic_experimentation.py`: Implementation of the strategic experimentation environment.
- `learning_without_experimentation.py`: Implementation of the learning without experimentation environment.
- `train.py`: Training script to run the algorithm in either environment.

## Partially Observable FURTHER Algorithm

The Partially Observable FURTHER algorithm extends the original FURTHER approach by:

1. **Belief State Representation**: Using GRUs to maintain and update belief states based on observations and actions.
2. **Variational Inference**: Inferring other agents' policies and learning dynamics through encoder/decoder networks.
3. **Average Reward Optimization**: Maximizing long-term average reward rather than discounted return.
4. **Active Influence**: Modeling how an agent's actions influence others' learning processes.

### Key Components:

- **BeliefStateNetwork**: Uses a GRU to track belief states over time.
- **EncoderNetwork/DecoderNetwork**: Variational inference components that infer and predict other agents' policies.
- **PolicyNetwork**: Maps belief states to actions.
- **QNetwork**: Estimates action-values for critic updates.

## Environments

### Strategic Experimentation

In this environment:
- Agents choose how to allocate resources between a safe arm with known payoff and a risky arm with unknown payoff.
- The true payoff of the risky arm depends on an unknown state of the world.
- Agents observe their own rewards and can infer information from others' actions.
- The goal is to maximize long-term average rewards.

### Learning Without Experimentation

In this environment:
- Agents need to learn the true state of the world through private signals and observing others' actions.
- Agents don't receive direct rewards from their actions.
- Rewards are constructed using the observed reward function that preserves expected rewards.
- The goal is to correctly identify the true state.

## Usage

You can run the training script with various command-line arguments:

```bash
python train.py --env [strategic|learning] --agents <num_agents> --states <num_states> --episodes <num_episodes> --eval-interval <interval> --save-path <path>
```

### Arguments:

- `--env`: Environment type ('strategic' or 'learning')
- `--agents`: Number of agents
- `--states`: Number of possible states
- `--episodes`: Number of training episodes
- `--eval-interval`: Evaluation interval
- `--save-path`: Path to save models and results

### Example:

```bash
# Train 3 agents in strategic experimentation environment for 1000 episodes
python train.py --env strategic --agents 3 --states 2 --episodes 1000 --save-path ./models

# Train 5 agents in learning without experimentation environment
python train.py --env learning --agents 5 --states 3 --episodes 2000 --save-path ./models
```

## Implementation Details

### Average Reward Formulation

Unlike typical reinforcement learning algorithms that use discounted returns, FURTHER maximizes the average reward over time:

```
ρ_θ^i = lim_{T→∞} E[1/T ∑_{t=0}^T R^i(s_t, a_t)]
```

This focus on long-term performance is crucial for multi-agent learning, as it encourages agents to consider how their actions influence others' learning processes over extended periods.

### Belief State Updates

The belief state is updated using a GRU network:

```
b_t^i = GRU(b_{t-1}^i, o_t^i, a_t^i)
```

Where `b_t^i` is agent i's belief state at time t, `o_t^i` is the observation, and `a_t^i` is the action.

### Variational Inference for Policy Estimation

The encoder-decoder architecture allows agents to infer and predict others' policies:

```
z_t^i = Encoder(o_t^i, a_t^i, a_t^{-i})
a_t^{-i} ≈ Decoder(o_t^i, z_t^i)
```

Where `z_t^i` represents the latent policy parameters of other agents as inferred by agent i.

## Theoretical Background

This implementation is based on the theoretical framework that extends Active Markov Games to partially observable settings. The key insights include:

1. Using belief states as sufficient statistics for histories in partially observable environments.
2. Modeling the joint learning dynamics of states, beliefs, and policy parameters.
3. Optimizing for long-term average rewards to focus on limiting behavior after convergence.
4. Using variational inference to estimate others' policies and learning dynamics from partial observations.

For more details on the theoretical framework, please refer to the accompanying theory files.

## Citation

If you use this code in your research, please cite:

```
@article{kim2022influencing,
  title={Influencing Long-Term Behavior in Multiagent Reinforcement Learning},
  author={Kim, Dong-Ki and Riemer, Matthew and Liu, Miao and Foerster, Jakob N and Everett, Michael and Sun, Chuangchuang and Tesauro, Gerald and How, Jonathan P},
  journal={Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```
