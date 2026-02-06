"""Scalable DQN Trainer for Massive Training (5M+ episodes).

Optimized for long training runs with:
- Efficient checkpointing
- Progress tracking
- Memory optimization
- Resume capability
- TensorBoard integration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import os
import json
from datetime import datetime
from pathlib import Path

from agent.dqn import DQN, create_dqn
from agent.replay_buffer import ReplayBuffer
from environment.blackjack_env import BlackjackEnv
from game.rules import Action
from strategies import (
    create_consensus_system,
    get_all_strategies,
    create_betting_system,
    ConsensusResult,
    BettingDecision
)


class ScalableTrainer:
    """
    Scalable trainer optimized for 5M+ episodes.

    Features:
    - Periodic checkpointing
    - Training resumption
    - Memory-efficient replay buffer
    - Detailed logging
    - Expert strategy consensus integration
    - Variable betting systems
    """

    def __init__(self,
                 env: BlackjackEnv,
                 state_dim: int = 9,
                 action_dim: int = 6,
                 hidden_dims: List[int] = [512, 256, 128],
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 buffer_size: int = 500000,  # Larger buffer
                 batch_size: int = 128,  # Larger batch
                 target_update_frequency: int = 5000,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay_steps: int = 100000,  # Fixed: Was 2M (too slow), now 100K for proper learning
                 network_type: str = 'standard',
                 save_dir: str = 'models',
                 checkpoint_interval: int = 100000,  # Save every 100K episodes
                 log_interval: int = 10000,
                 use_consensus: bool = True,
                 consensus_type: str = 'hybrid',
                 use_variable_betting: bool = True,
                 betting_system: str = 'hilo',
                 initial_bankroll: float = 100000.0,
                 min_bet: float = 10.0,
                 max_bet: float = 1000.0):
        """
        Initialize scalable trainer.

        Args:
            env: Blackjack environment
            checkpoint_interval: Save checkpoint every N episodes
            log_interval: Log metrics every N episodes
            use_consensus: Use expert strategy consensus during training
            consensus_type: Type of consensus system
            use_variable_betting: Use variable betting system
            betting_system: Type of betting system
            initial_bankroll: Starting bankroll for variable betting
            min_bet: Minimum bet
            max_bet: Maximum bet
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
        self.save_dir = Path(save_dir)
        self.checkpoint_interval = checkpoint_interval
        self.log_interval = log_interval

        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.save_dir / 'logs').mkdir(exist_ok=True)
        (self.save_dir / 'metrics').mkdir(exist_ok=True)

        # Create networks
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
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.criterion = nn.SmoothL1Loss()

        # Experience replay
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Training state
        self.episode_count = 0
        self.step_count = 0
        self.epsilon = epsilon_start
        self.start_time = None

        # Metrics tracking
        self.episode_rewards: List[float] = []
        self.episode_losses: List[float] = []
        self.episode_win_rates: List[float] = []
        self.bankroll_history: List[float] = []

        # Expert strategies and consensus
        self.use_consensus = use_consensus
        if use_consensus:
            self.expert_strategies = get_all_strategies()
            self.consensus_system = create_consensus_system(
                consensus_type, self.expert_strategies
            )

        # Betting system
        self.use_variable_betting = use_variable_betting
        if use_variable_betting:
            self.betting_system = create_betting_system(
                betting_system,
                initial_bankroll=initial_bankroll,
                min_bet=min_bet,
                max_bet=max_bet
            )

        # Performance tracking
        self.best_win_rate = 0.0
        self.best_profit = float('-inf')

        print(f"Scalable Trainer Initialized")
        print(f"  Target episodes: 5,000,000+")
        print(f"  Checkpoint interval: {checkpoint_interval:,}")
        print(f"  Log interval: {log_interval:,}")
        print(f"  Buffer size: {buffer_size:,}")
        print(f"  Consensus: {use_consensus} ({consensus_type})")
        print(f"  Variable betting: {use_variable_betting} ({betting_system})")

    def train_step(self) -> Optional[float]:
        """Perform one training step."""
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample_arrays(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q_values = self.policy_network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def update_epsilon(self) -> None:
        """Update epsilon."""
        if self.step_count < self.epsilon_decay_steps:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * \
                          (self.step_count / self.epsilon_decay_steps)
        else:
            self.epsilon = self.epsilon_end

    def update_target_network(self) -> None:
        """Update target network."""
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def get_action_with_consensus(self, state_vector: np.ndarray,
                                  game,
                                  valid_actions: List[Action]) -> Tuple[int, Optional[ConsensusResult]]:
        """Get action using DQN with expert consensus."""
        # Convert Action enums to int indices for DQN
        valid_action_indices = [int(action.value) for action in valid_actions]

        # Get DQN action
        dqn_action = self.policy_network.select_action(
            state_vector, self.epsilon, valid_action_indices
        )

        # Get consensus action (need to access internal game state)
        if self.use_consensus and np.random.random() > self.epsilon:
            game_state = game.get_state()  # Access internal game state
            try:
                consensus_result = self.consensus_system.get_consensus(game_state, valid_actions)

                # If consensus confidence is high, use it
                if consensus_result.confidence > 0.6:
                    # Handle both Action enum and int
                    selected = consensus_result.selected_action
                    if isinstance(selected, int):
                        action_value = selected
                    elif hasattr(selected, 'value'):
                        action_value = int(selected.value)
                    else:
                        action_value = int(selected)
                    return action_value, consensus_result
            except Exception as e:
                # If consensus fails, fall back to DQN
                pass

        return dqn_action, None

    def get_bet_amount(self, game) -> float:
        """Get bet amount using betting system."""
        if self.use_variable_betting:
            game_state = game.get_state()  # Access internal game state
            decision = self.betting_system.get_bet(game_state)
            return decision.bet_amount
        else:
            return 1.0  # Flat betting

    def train(self,
              target_episodes: int = 5000000,
              resume_from: Optional[str] = None,
              verbose: bool = True) -> Dict:
        """
        Train for specified number of episodes.

        Args:
            target_episodes: Total number of episodes to train
            resume_from: Path to checkpoint to resume from
            verbose: Whether to show progress bar

        Returns:
            Training metrics dictionary
        """
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
            print(f"Resumed from episode {self.episode_count}")

        print(f"\n{'='*70}")
        print(f"STARTING TRAINING: {target_episodes:,} EPISODES")
        print(f"{'='*70}\n")

        self.start_time = datetime.now()

        # Training loop
        pbar = tqdm(range(self.episode_count, target_episodes),
                   desc="Training",
                   disable=not verbose)

        for episode in pbar:
            # Reset environment
            state, _ = self.env.reset()
            state_vector = self.env.get_observation_vector(state)

            # Get bet amount (access internal game)
            bet_amount = self.get_bet_amount(self.env.game)
            self.env.game.current_bet = bet_amount

            episode_reward = 0.0
            episode_loss = []
            done = False
            steps = 0

            while not done and steps < 50:
                # Get valid actions as Action enums from the game
                valid_action_enums = self.env.game.get_valid_actions()

                # Get action (with consensus)
                action, consensus_result = self.get_action_with_consensus(
                    state_vector, self.env.game, valid_action_enums
                )

                # Step environment
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state_vector = self.env.get_observation_vector(next_state)

                # Scale reward by bet amount
                scaled_reward = reward * bet_amount

                # Store transition
                self.replay_buffer.push(
                    state_vector, action, scaled_reward,
                    next_state_vector, done or truncated
                )

                # Train
                loss = self.train_step()
                if loss is not None:
                    episode_loss.append(loss)

                # Update
                state_vector = next_state_vector
                episode_reward += scaled_reward
                steps += 1
                self.step_count += 1

                # Update target network
                if self.step_count % self.target_update_frequency == 0:
                    self.update_target_network()

                # Update epsilon
                self.update_epsilon()

            # Update bankroll
            if self.use_variable_betting:
                self.betting_system.update_bankroll(episode_reward)

            # Track metrics
            self.episode_rewards.append(episode_reward)
            self.bankroll_history.append(self.betting_system.current_bankroll if self.use_variable_betting else episode_reward)
            if episode_loss:
                self.episode_losses.append(np.mean(episode_loss))

            # Logging
            if (episode + 1) % self.log_interval == 0:
                self._log_progress(episode, target_episodes)

            # Checkpointing
            if (episode + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(episode + 1)

            # Update progress bar
            if verbose and (episode + 1) % 100 == 0:
                recent_rewards = self.episode_rewards[-100:]
                avg_reward = np.mean(recent_rewards)
                win_rate = np.mean([1 if r > 0 else 0 for r in recent_rewards]) * 100
                pbar.set_postfix({
                    'avg_reward': f'{avg_reward:.3f}',
                    'win_rate': f'{win_rate:.1f}%',
                    'epsilon': f'{self.epsilon:.3f}',
                    'bankroll': f'${self.bankroll_history[-1]:.0f}' if self.use_variable_betting else 'N/A'
                })

        # Final save
        self.save_checkpoint(target_episodes)
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETED: {target_episodes:,} EPISODES")
        print(f"{'='*70}\n")

        return self.get_training_summary()

    def _log_progress(self, episode: int, target_episodes: int) -> None:
        """Log training progress."""
        recent_rewards = self.episode_rewards[-self.log_interval:]
        avg_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        win_rate = np.mean([1 if r > 0 else 0 for r in recent_rewards]) * 100

        recent_losses = self.episode_losses[-min(100, len(self.episode_losses)):]
        avg_loss = np.mean(recent_losses) if recent_losses else 0.0

        elapsed = (datetime.now() - self.start_time).total_seconds()
        episodes_per_sec = (episode + 1) / elapsed
        remaining = (target_episodes - episode) / episodes_per_sec if episodes_per_sec > 0 else 0

        print(f"\n{'='*70}")
        print(f"Episode {episode + 1:,}/{target_episodes:,} ({(episode+1)/target_episodes*100:.2f}%)")
        print(f"{'='*70}")
        print(f"  Recent Performance ({self.log_interval:,} episodes):")
        print(f"    Avg Reward: {avg_reward:.4f} ± {std_reward:.4f}")
        print(f"    Win Rate:   {win_rate:.2f}%")
        print(f"    Avg Loss:   {avg_loss:.4f}")
        print(f"  Training State:")
        print(f"    Epsilon:    {self.epsilon:.4f}")
        print(f"    Buffer:     {len(self.replay_buffer):,}/{self.replay_buffer.capacity:,}")
        print(f"    Steps:      {self.step_count:,}")
        print(f"  Bankroll:    ${self.bankroll_history[-1]:,.2f}" if self.use_variable_betting else "")
        print(f"  Timing:")
        print(f"    Elapsed:    {self._format_time(elapsed)}")
        print(f"    ETA:        {self._format_time(remaining)}")
        print(f"    Speed:      {episodes_per_sec:.1f} eps/sec")
        print(f"{'='*70}\n")

        # Update learning rate
        if recent_losses:
            self.scheduler.step(avg_loss)

        # Save metrics
        self._save_metrics(episode)

    def save_checkpoint(self, episode: int) -> None:
        """Save training checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        checkpoint = {
            'episode': episode,
            'step_count': self.step_count,
            'epsilon': self.epsilon,
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_win_rate': self.best_win_rate,
            'best_profit': self.best_profit,
        }

        # Save main checkpoint
        checkpoint_path = self.save_dir / 'checkpoints' / f'checkpoint_ep{episode}_{timestamp}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save as latest
        latest_path = self.save_dir / 'checkpoints' / 'latest.pt'
        torch.save(checkpoint, latest_path)

        print(f"Checkpoint saved: {checkpoint_path.name}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        self.episode_count = checkpoint['episode']
        self.step_count = checkpoint['step_count']
        self.epsilon = checkpoint['epsilon']
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_win_rate = checkpoint.get('best_win_rate', 0.0)
        self.best_profit = checkpoint.get('best_profit', float('-inf'))

        print(f"Loaded checkpoint from episode {self.episode_count:,}")

    def _save_metrics(self, episode: int) -> None:
        """Save training metrics to JSON."""
        metrics = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'recent_avg_reward': float(np.mean(self.episode_rewards[-self.log_interval:])),
            'recent_win_rate': float(np.mean([1 if r > 0 else 0 for r in self.episode_rewards[-self.log_interval:]])),
            'bankroll': float(self.bankroll_history[-1]) if self.bankroll_history else 0.0,
            'buffer_size': len(self.replay_buffer),
        }

        metrics_path = self.save_dir / 'metrics' / f'metrics_ep{episode}.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    def get_training_summary(self) -> Dict:
        """Get summary of training."""
        total_reward = sum(self.episode_rewards)
        win_rate = np.mean([1 if r > 0 else 0 for r in self.episode_rewards]) * 100

        return {
            'total_episodes': len(self.episode_rewards),
            'total_reward': total_reward,
            'win_rate': win_rate,
            'final_epsilon': self.epsilon,
            'final_bankroll': self.bankroll_history[-1] if self.bankroll_history else 0,
            'training_time': str(datetime.now() - self.start_time) if self.start_time else 'N/A',
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to readable time."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
