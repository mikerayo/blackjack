"""Curriculum Learning Trainer - Empieza con expertos, luego transiciona a DQN.

Idea clave:
1. Fase 1: Solo usar expertos (cold start)
2. Fase 2: Mezclar expertos + DQN gradually
3. Fase 3: Solo DQN fine-tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from .dqn import create_dqn
from .replay_buffer import ReplayBuffer
from ..environment.blackjack_env import BlackjackEnv
from ..game.rules import Action
from ..strategies import (
    get_all_strategies,
    create_consensus_system,
    create_betting_system
)


class CurriculumTrainer:
    """
    Curriculum Learning approach for faster convergence.

    Phases:
    1. Expert-only (cold start)
    2. Expert-guided DQN
    3. DQN-dominant
    4. Pure DQN
    """

    def __init__(self,
                 env: BlackjackEnv,
                 state_dim: int = 9,
                 action_dim: int = 6,
                 hidden_dims: List[int] = [512, 256, 128],
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 buffer_size: int = 200000,
                 batch_size: int = 128,
                 save_dir: str = 'models'):

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Networks
        self.policy_network = create_dqn(
            'standard', state_dim, action_dim, hidden_dims
        )
        self.target_network = create_dqn(
            'standard', state_dim, action_dim, hidden_dims
        )
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Expert strategies
        self.expert_strategies = get_all_strategies()
        self.consensus_system = create_consensus_system('hybrid', self.expert_strategies)

        # Betting system
        self.betting_system = create_betting_system('hilo', initial_bankroll=10000)
        self.betting_system.reset()

        # Training state
        self.episode_count = 0
        self.step_count = 0

        print(f"Curriculum Trainer Initialized")
        print(f"  Expert strategies: {len(self.expert_strategies)}")
        print(f"  Consensus system: Hybrid")
        print(f"  Betting system: Hi-Lo")

    def get_expert_action(self, game_state, valid_actions):
        """Get action from expert consensus."""
        result = self.consensus_system.get_consensus(game_state, valid_actions)
        action = result.selected_action

        # Convert to int if needed
        if hasattr(action, 'value'):
            action = int(action.value)
        return action

    def get_mixed_action(self, game_state, state_vector, valid_actions, expert_probability):
        """
        Mix expert and DQN actions.

        expert_probability: 0.0 = pure DQN, 1.0 = pure expert
        valid_actions: List of Action enums
        """
        # Use expert with given probability
        if np.random.random() < expert_probability:
            return self.get_expert_action(game_state, valid_actions)
        else:
            # Use DQN - convert Action enums to int indices
            valid_action_indices = [int(action.value) for action in valid_actions]
            action = self.policy_network.select_action(state_vector, epsilon=0.0, valid_actions=valid_action_indices)
            return action

    def train_step(self):
        """Perform one training step."""
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample_arrays(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Current Q values
        current_q = self.policy_network(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = self.criterion(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def train_curriculum(self,
                        total_episodes: int = 1000000,
                        phase1_episodes: int = 50000,    # Expert-only
                        phase2_episodes: int = 200000,   # Expert-guided
                        phase3_episodes: int = 500000,   # DQN-dominant
                        log_interval: int = 5000):
        """
        Train with curriculum learning.
        """

        print("\n" + "="*80)
        print("CURRICULUM LEARNING TRAINING")
        print("="*80)
        print(f"\nPhase 1: Expert-only ({phase1_episodes} episodes)")
        print(f"Phase 2: Expert-guided ({phase2_episodes} episodes)")
        print(f"Phase 3: DQN-dominant ({phase3_episodes} episodes)")
        print(f"Phase 4: Pure DQN ({total_episodes - phase1_episodes - phase2_episodes - phase3_episodes} episodes)")
        print("="*80 + "\n")

        start_time = datetime.now()
        episode_rewards = []
        episode_win_rates = []

        for episode in tqdm(range(total_episodes), desc="Curriculum Training"):
            # Determine current phase and expert probability
            if episode < phase1_episodes:
                # Phase 1: Pure expert
                expert_prob = 1.0
                phase = "Phase 1: Expert-only"
            elif episode < phase1_episodes + phase2_episodes:
                # Phase 2: High expert influence
                progress = (episode - phase1_episodes) / phase2_episodes
                expert_prob = 0.8 - 0.5 * progress  # 0.8 -> 0.3
                phase = "Phase 2: Expert-guided"
            elif episode < phase1_episodes + phase2_episodes + phase3_episodes:
                # Phase 3: DQN dominant
                progress = (episode - phase1_episodes - phase2_episodes) / phase3_episodes
                expert_prob = 0.3 - 0.25 * progress  # 0.3 -> 0.05
                phase = "Phase 3: DQN-dominant"
            else:
                # Phase 4: Pure DQN
                expert_prob = 0.05
                phase = "Phase 4: Pure DQN"

            # Run episode
            state, _ = self.env.reset()
            state_vector = self.env.get_observation_vector(state)

            # Get bet
            bet_amount = self.betting_system.get_bet(self.env.game.get_state())
            self.env.game.current_bet = bet_amount

            episode_reward = 0.0
            episode_loss = []
            done = False
            steps = 0

            while not done and steps < 50:
                # Get valid actions as Action enums from the game
                valid_action_enums = self.env.game.get_valid_actions()

                # Get action (mixed expert/DQN)
                action = self.get_mixed_action(
                    self.env.game,
                    state_vector,
                    valid_action_enums,
                    expert_prob
                )

                # Step
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state_vector = self.env.get_observation_vector(next_state)

                # Scale reward
                scaled_reward = reward * bet_amount

                # Store
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
                if self.step_count % 1000 == 0:
                    self.target_network.load_state_dict(self.policy_network.state_dict())

            # Update betting
            self.betting_system.update_bankroll(episode_reward)
            episode_rewards.append(episode_reward)
            episode_win_rates.append(1 if episode_reward > 0 else 0)

            # Log progress
            if (episode + 1) % log_interval == 0:
                recent_rewards = episode_rewards[-log_interval:]
                avg_reward = np.mean(recent_rewards)
                win_rate = np.mean([1 if r > 0 else 0 for r in recent_rewards]) * 100

                print(f"\nEpisode {episode + 1:,}/{total_episodes:,} ({phase})")
                print(f"  Expert Probability: {expert_prob:.2f}")
                print(f"  Avg Reward: ${avg_reward:.2f}")
                print(f"  Win Rate: {win_rate:.2f}%")
                print(f"  Bankroll: ${self.betting_system.current_bankroll:,.2f}")
                print(f"  Buffer: {len(self.replay_buffer):,}")

            # Checkpoint
            if (episode + 1) % 50000 == 0:
                self.save_checkpoint(episode + 1)

        # Final save
        self.save_checkpoint(total_episodes)

        elapsed = (datetime.now() - start_time).total_seconds()

        # Summary
        summary = {
            'total_episodes': total_episodes,
            'total_reward': sum(episode_rewards),
            'win_rate': np.mean(episode_win_rates) * 100,
            'final_bankroll': self.betting_system.current_bankroll,
            'training_time': str(elapsed),
        }

        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)
        print(f"Total Episodes: {summary['total_episodes']:,}")
        print(f"Win Rate: {summary['win_rate']:.2f}%")
        print(f"Total Reward: ${summary['total_reward']:,.2f}")
        print(f"Final Bankroll: ${summary['final_bankroll']:,.2f}")
        print(f"Training Time: {summary['training_time']}")
        print("="*80 + "\n")

        return summary

    def save_checkpoint(self, episode):
        """Save checkpoint."""
        checkpoint = {
            'episode': episode,
            'step_count': self.step_count,
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'betting_system_bankroll': self.betting_system.current_bankroll,
        }

        path = self.save_dir / f'curriculum_checkpoint_ep{episode}.pt'
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path.name}")

    def load_checkpoint(self, path):
        """Load checkpoint."""
        checkpoint = torch.load(path)
        self.episode_count = checkpoint['episode']
        self.step_count = checkpoint['step_count']
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'betting_system_bankroll' in checkpoint:
            self.betting_system.current_bankroll = checkpoint['betting_system_bankroll']
        print(f"Loaded checkpoint from episode {self.episode_count}")
