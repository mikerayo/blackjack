"""Massive Training Script - 5M+ Episodes.

Run this for serious training with expert consensus and variable betting.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environment.blackjack_env import BlackjackEnv
from agent.scalable_trainer import ScalableTrainer
from strategies import get_available_systems, get_available_betting_systems


def main():
    parser = argparse.ArgumentParser(
        description='Massive DQN Training - 5M+ Episodes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training scale
    parser.add_argument('--episodes', type=int, default=5000000,
                       help='Total number of training episodes')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')

    # Network architecture
    parser.add_argument('--hidden-dims', type=str, default='512,256,128',
                       help='Hidden layer dimensions')
    parser.add_argument('--network-type', type=str, default='standard',
                       choices=['standard', 'dueling'])

    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--buffer-size', type=int, default=500000,
                       help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--target-update', type=int, default=5000,
                       help='Target network update frequency')
    parser.add_argument('--epsilon-decay', type=int, default=2000000,
                       help='Epsilon decay steps')

    # Expert consensus
    parser.add_argument('--use-consensus', action='store_true', default=True,
                       help='Use expert strategy consensus during training')
    parser.add_argument('--consensus-type', type=str, default='hybrid',
                       choices=get_available_systems(),
                       help='Type of consensus system')

    # Variable betting
    parser.add_argument('--use-variable-betting', action='store_true', default=True,
                       help='Use variable betting system')
    parser.add_argument('--betting-system', type=str, default='hilo',
                       choices=get_available_betting_systems(),
                       help='Type of betting system')
    parser.add_argument('--initial-bankroll', type=float, default=100000.0,
                       help='Starting bankroll')
    parser.add_argument('--min-bet', type=float, default=10.0,
                       help='Minimum bet')
    parser.add_argument('--max-bet', type=float, default=1000.0,
                       help='Maximum bet')

    # Environment
    parser.add_argument('--num-decks', type=int, default=6,
                       help='Number of decks')
    parser.add_argument('--penetration', type=float, default=0.75,
                       help='Shoe penetration')
    parser.add_argument('--allow-surrender', action='store_true', default=True)
    parser.add_argument('--dealer-hits-soft-17', action='store_true', default=True)

    # Checkpointing and logging
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--checkpoint-interval', type=int, default=100000,
                       help='Save checkpoint every N episodes')
    parser.add_argument('--log-interval', type=int, default=10000,
                       help='Log progress every N episodes')

    args = parser.parse_args()

    print("="*80)
    print(" " * 20 + "MASSIVE BLACKJACK TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Target Episodes:     {args.episodes:,}")
    print(f"  Network:             {args.network_type} ({args.hidden_dims})")
    print(f"  Expert Consensus:    {args.use_consensus} ({args.consensus_type})")
    print(f"  Variable Betting:    {args.use_variable_betting} ({args.betting_system})")
    print(f"  Initial Bankroll:    ${args.initial_bankroll:,.2f}")
    print(f"  Checkpoint Interval: {args.checkpoint_interval:,}")
    print(f"  Resume From:         {args.resume or 'None'}")
    print("\n" + "="*80 + "\n")

    # Create environment
    env = BlackjackEnv(
        num_decks=args.num_decks,
        penetration=args.penetration,
        allow_surrender=args.allow_surrender,
        dealer_hits_soft_17=args.dealer_hits_soft_17
    )

    # Create trainer
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]

    trainer = ScalableTrainer(
        env=env,
        state_dim=9,
        action_dim=6,
        hidden_dims=hidden_dims,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_frequency=args.target_update,
        epsilon_decay_steps=args.epsilon_decay,
        network_type=args.network_type,
        save_dir=args.save_dir,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        use_consensus=args.use_consensus,
        consensus_type=args.consensus_type,
        use_variable_betting=args.use_variable_betting,
        betting_system=args.betting_system,
        initial_bankroll=args.initial_bankroll,
        min_bet=args.min_bet,
        max_bet=args.max_bet
    )

    # Train
    print("Starting training...\n")

    try:
        summary = trainer.train(
            target_episodes=args.episodes,
            resume_from=args.resume,
            verbose=True
        )

        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"Total Episodes:     {summary['total_episodes']:,}")
        print(f"Total Reward:       ${summary['total_reward']:,.2f}")
        print(f"Win Rate:           {summary['win_rate']:.2f}%")
        print(f"Final Bankroll:     ${summary['final_bankroll']:,.2f}")
        print(f"Training Time:      {summary['training_time']}")
        print("="*80 + "\n")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving emergency checkpoint...")
        trainer.save_checkpoint(trainer.episode_count)
        print(f"Emergency checkpoint saved at episode {trainer.episode_count:,}")
        print("Resume with --resume models/checkpoints/latest.pt")


if __name__ == '__main__':
    main()
