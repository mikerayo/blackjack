"""DQN Trainer with target network and experience replay."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import os
from datetime import datetime

from agent.dqn import DQN, create_dqn
from agent.replay_buffer import ReplayBuffer, Transition
from environment.blackjack_env import BlackjackEnv


class DQNTrainer:
    """
    DQN Trainer with target network and experience replay.

    Implements all standard DQN improvements:
    - Target network (periodic update)
    - Experience replay buffer
    - Epsilon-greedy exploration with decay
    - Gradient clipping
    - Huber loss (smooth L1)
    """

    def __init__(self,
                 env: BlackjackEnv,
                 state_dim: int = 9,
                 action_dim: int = 6,
                 hidden_dims: List[int] = [256, 256, 128],
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update_frequency: int = 1000,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay_steps: int = 100000,
                 network_type: str = 'standard',
                 save_dir: str = 'models'):
        """
        Initialize the DQN trainer.

        Args:
            env: Gymnasium environment
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer sizes
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update_frequency: Steps between target network updates
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps to decay epsilon
            network_type: Type of network ('standard' or 'dueling')
            save_dir: Directory to save models
        """
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.save_dir = save_dir

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Create policy and target networks
        self.policy_network = create_dqn(
            network_type=network_type,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        )
        self.target_network = create_dqn(
            network_type=network_type,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        )

        # Sync target network with policy network
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()  # Target network in evaluation mode

        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Training metrics
        self.step_count = 0
        self.episode_count = 0
        self.epsilon = epsilon_start

        # Statistics tracking
        self.episode_rewards: List[float] = []
        self.episode_losses: List[float] = []
        self.episode_lengths: List[int] = []

    def select_action(self, state: np.ndarray, valid_actions: Optional[List[int]] = None) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            valid_actions: List of valid action indices

        Returns:
            Selected action
        """
        # Get all actions if not specified
        if valid_actions is None:
            valid_actions = list(range(self.action_dim))

        return self.policy_network.select_action(state, self.epsilon, valid_actions)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step on a batch from replay buffer.

        Returns:
            Loss value if training was performed, None otherwise
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_arrays(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute current Q-values
        current_q_values = self.policy_network(states).gather(1, actions.unsqueeze(1))

        # Compute target Q-values
        with torch.no_grad():
            # Use target network for next state Q-values
            next_q_values = self.target_network(next_states).max(1)[0]

            # Compute target: r + gamma * max(Q_target(s', a')) * (1 - done)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10.0)

        self.optimizer.step()

        return loss.item()

    def update_epsilon(self) -> None:
        """Update epsilon using linear decay."""
        if self.step_count < self.epsilon_decay_steps:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * \
                          (self.step_count / self.epsilon_decay_steps)
        else:
            self.epsilon = self.epsilon_end

    def update_target_network(self) -> None:
        """Update target network with current policy network weights."""
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def train(self, num_episodes: int, max_steps_per_episode: int = 100,
              log_frequency: int = 1000, save_frequency: int = 10000) -> Dict[str, List]:
        """
        Train the agent for a specified number of episodes.

        Args:
            num_episodes: Number of episodes to train
            max_steps_per_episode: Maximum steps per episode
            log_frequency: Log progress every N episodes
            save_frequency: Save model every N episodes

        Returns:
            Dictionary of training metrics
        """
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Initial epsilon: {self.epsilon:.4f}")
        print(f"Replay buffer size: {self.replay_buffer.capacity}")

        episode_rewards = []
        episode_losses = []
        episode_wins = []

        for episode in tqdm(range(num_episodes), desc="Training"):
            state, _ = self.env.reset()
            state_vector = self.env.get_observation_vector(state)

            episode_reward = 0.0
            episode_loss = []
            done = False
            steps = 0

            while not done and steps < max_steps_per_episode:
                # Get valid actions
                valid_actions = list(range(self.action_dim))

                # Select and perform action
                action = self.select_action(state_vector, valid_actions)
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state_vector = self.env.get_observation_vector(next_state)

                # Store transition in replay buffer
                self.replay_buffer.push(
                    state_vector,
                    action,
                    reward,
                    next_state_vector,
                    done or truncated
                )

                # Train the network
                loss = self.train_step()
                if loss is not None:
                    episode_loss.append(loss)

                # Move to next state
                state_vector = next_state_vector
                episode_reward += reward
                steps += 1
                self.step_count += 1

                # Update target network
                if self.step_count % self.target_update_frequency == 0:
                    self.update_target_network()

                # Update epsilon
                self.update_epsilon()

            # Track episode statistics
            episode_rewards.append(episode_reward)
            if episode_loss:
                episode_losses.append(np.mean(episode_loss))
            episode_wins.append(1 if episode_reward > 0 else 0)

            # Log progress
            if (episode + 1) % log_frequency == 0:
                avg_reward = np.mean(episode_rewards[-log_frequency:])
                avg_loss = np.mean(episode_losses[-log_frequency:]) if episode_losses else 0.0
                win_rate = np.mean(episode_wins[-log_frequency:]) * 100

                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"  Average Reward (last {log_frequency}): {avg_reward:.4f}")
                print(f"  Average Loss: {avg_loss:.4f}")
                print(f"  Win Rate: {win_rate:.2f}%")
                print(f"  Epsilon: {self.epsilon:.4f}")
                print(f"  Buffer Size: {len(self.replay_buffer)}")

            # Save model checkpoint
            if (episode + 1) % save_frequency == 0:
                self.save_model(f'checkpoint_ep{episode + 1}.pt')

        print("\nTraining completed!")
        return {
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses,
            'episode_wins': episode_wins
        }

    def evaluate(self, num_episodes: int = 1000) -> Dict[str, float]:
        """
        Evaluate the current policy without exploration.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"Evaluating for {num_episodes} episodes...")

        self.policy_network.eval()

        episode_rewards = []
        episode_wins = []
        episode_blackjacks = []

        with torch.no_grad():
            for episode in tqdm(range(num_episodes), desc="Evaluating"):
                state, _ = self.env.reset()
                state_vector = self.env.get_observation_vector(state)

                episode_reward = 0.0
                done = False

                while not done:
                    # Get valid actions
                    valid_actions = list(range(self.action_dim))

                    # Greedy action selection (epsilon = 0)
                    action = self.policy_network.select_action(state_vector, 0.0, valid_actions)
                    next_state, reward, done, truncated, info = self.env.step(action)
                    state_vector = self.env.get_observation_vector(next_state)
                    episode_reward += reward

                episode_rewards.append(episode_reward)
                episode_wins.append(1 if episode_reward > 0 else 0)

                # Check for blackjack
                if 'game_result' in info:
                    is_blackjack = info['game_result'].result.name == 'BLACKJACK'
                    episode_blackjacks.append(1 if is_blackjack else 0)

        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'win_rate': np.mean(episode_wins) * 100,
            'total_profit': np.sum(episode_rewards),
            'blackjack_rate': np.mean(episode_blackjacks) * 100 if episode_blackjacks else 0,
            'roi': (np.sum(episode_rewards) / (num_episodes * 1.0)) * 100  # ROI percentage
        }

        print("\nEvaluation Results:")
        print(f"  Mean Reward: {metrics['mean_reward']:.4f}")
        print(f"  Std Reward: {metrics['std_reward']:.4f}")
        print(f"  Win Rate: {metrics['win_rate']:.2f}%")
        print(f"  Total Profit: {metrics['total_profit']:.2f}")
        print(f"  Blackjack Rate: {metrics['blackjack_rate']:.2f}%")

        self.policy_network.train()

        return metrics

    def save_model(self, filename: str) -> None:
        """
        Save model checkpoint.

        Args:
            filename: Name of the checkpoint file
        """
        checkpoint = {
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'epsilon': self.epsilon,
            'policy_network_state_dict': self.policy_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
        }

        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filename: str) -> None:
        """
        Load model checkpoint.

        Args:
            filename: Name of the checkpoint file
        """
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath)

        self.episode_count = checkpoint['episode_count']
        self.step_count = checkpoint['step_count']
        self.epsilon = checkpoint['epsilon']
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])

        print(f"Model loaded from {filepath}")
