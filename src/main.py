"""Main entry point for Blackjack DQN training and evaluation."""

import argparse
import sys
import os
import json
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.blackjack_env import BlackjackEnv
from agent import DQNTrainer
from utils.metrics import (
    evaluate_basic_strategy,
    evaluate_random_strategy,
    print_metrics_report,
    calculate_metrics
)
from utils.visualization import (
    plot_training_rewards,
    plot_win_rate,
    plot_cumulative_profit,
    plot_loss_curve,
    create_dashboard
)


def train(args):
    """Train the DQN agent."""
    print("="*60)
    print("Training DQN Agent for Blackjack")
    print("="*60)

    # Create environment
    env = BlackjackEnv(
        num_decks=args.num_decks,
        penetration=args.penetration,
        allow_surrender=args.allow_surrender,
        dealer_hits_soft_17=args.dealer_hits_soft_17
    )

    # Create trainer
    trainer = DQNTrainer(
        env=env,
        state_dim=9,
        action_dim=6,
        hidden_dims=[int(x) for x in args.hidden_dims.split(',')],
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_frequency=args.target_update_frequency,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        network_type=args.network_type,
        save_dir=args.save_dir
    )

    # Train
    metrics = trainer.train(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        log_frequency=args.log_frequency,
        save_frequency=args.save_frequency
    )

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(args.save_dir, f'dqn_blackjack_{timestamp}.pt')
    trainer.save_model(f'dqn_blackjack_{timestamp}.pt')

    # Save training metrics
    metrics_path = os.path.join(args.save_dir, f'training_metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        # Convert to JSON-serializable format
        json.dump({
            'episode_rewards': metrics['episode_rewards'],
            'episode_losses': metrics['episode_losses'],
            'episode_wins': metrics['episode_wins'],
            'hyperparameters': {
                'learning_rate': args.learning_rate,
                'gamma': args.gamma,
                'batch_size': args.batch_size,
                'epsilon_decay_steps': args.epsilon_decay_steps,
                'num_decks': args.num_decks,
                'penetration': args.penetration
            }
        }, f, indent=2)

    print(f"\nModel saved to: {final_model_path}")
    print(f"Metrics saved to: {metrics_path}")

    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        viz_dir = os.path.join(args.save_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        plot_training_rewards(
            metrics['episode_rewards'],
            window=100,
            save_path=os.path.join(viz_dir, f'training_rewards_{timestamp}.png')
        )

        plot_win_rate(
            metrics['episode_rewards'],
            window=100,
            save_path=os.path.join(viz_dir, f'win_rate_{timestamp}.png')
        )

        plot_cumulative_profit(
            metrics['episode_rewards'],
            save_path=os.path.join(viz_dir, f'cumulative_profit_{timestamp}.png')
        )

        if metrics['episode_losses']:
            plot_loss_curve(
                metrics['episode_losses'],
                window=50,
                save_path=os.path.join(viz_dir, f'loss_curve_{timestamp}.png')
            )

        create_dashboard(
            metrics['episode_rewards'],
            metrics['episode_losses'] if metrics['episode_losses'] else None,
            window=100,
            save_path=os.path.join(viz_dir, f'dashboard_{timestamp}.png')
        )

        print(f"Visualizations saved to: {viz_dir}")


def evaluate(args):
    """Evaluate a trained DQN agent."""
    print("="*60)
    print("Evaluating DQN Agent")
    print("="*60)

    # Create environment
    env = BlackjackEnv(
        num_decks=args.num_decks,
        penetration=args.penetration,
        allow_surrender=args.allow_surrender,
        dealer_hits_soft_17=args.dealer_hits_soft_17
    )

    # Create trainer (will load model)
    trainer = DQNTrainer(
        env=env,
        state_dim=9,
        action_dim=6,
        hidden_dims=[int(x) for x in args.hidden_dims.split(',')],
        save_dir=args.save_dir
    )

    # Load model
    if args.model_path:
        trainer.load_model(args.model_path)
    else:
        print("Error: --model-path is required for evaluation")
        sys.exit(1)

    # Evaluate DQN
    dqn_metrics = trainer.evaluate(num_episodes=args.episodes)

    # Evaluate baselines
    print("\n" + "="*60)
    print("Evaluating Basic Strategy...")
    print("="*60)
    bs_metrics = evaluate_basic_strategy(env, num_episodes=args.episodes)
    print_metrics_report(bs_metrics, "Basic Strategy")

    print("\n" + "="*60)
    print("Evaluating Random Strategy...")
    print("="*60)
    random_metrics = evaluate_random_strategy(env, num_episodes=args.episodes)
    print_metrics_report(random_metrics, "Random Strategy")

    # Print comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"\nDQN vs Basic Strategy:")
    print(f"  DQN Win Rate: {dqn_metrics['win_rate']:.2f}%")
    print(f"  Basic Strategy Win Rate: {bs_metrics.win_rate:.2f}%")
    print(f"  Win Rate Difference: {dqn_metrics['win_rate'] - bs_metrics.win_rate:.2f}%")
    print(f"  DQN ROI: {dqn_metrics['roi']:.2f}%")
    print(f"  Basic Strategy ROI: {bs_metrics.roi:.2f}%")
    print(f"  ROI Difference: {dqn_metrics['roi'] - bs_metrics.roi:.2f}%")
    print(f"  Advantage Difference: ${dqn_metrics['mean_reward'] - bs_metrics.mean_reward:.4f} per hand")

    print(f"\nDQN vs Random Strategy:")
    print(f"  Win Rate Improvement: {dqn_metrics['win_rate'] - random_metrics.win_rate:.2f}%")
    print(f"  ROI Improvement: {dqn_metrics['roi'] - random_metrics.roi:.2f}%")


def test_game(args):
    """Test the blackjack game engine."""
    print("="*60)
    print("Testing Blackjack Game Engine")
    print("="*60)

    env = BlackjackEnv(
        num_decks=args.num_decks,
        penetration=args.penetration,
        allow_surrender=args.allow_surrender,
        dealer_hits_soft_17=args.dealer_hits_soft_17,
        render_mode='human'
    )

    print("\nPlaying 5 test games with random actions...\n")

    for episode in range(5):
        print(f"\n--- Episode {episode + 1} ---")
        state, _ = env.reset()

        done = False
        steps = 0

        while not done and steps < 10:
            # Random action
            action = env.action_space.sample()
            print(f"\nStep {steps + 1}: Action = {['HIT', 'STAND', 'DOUBLE', 'SPLIT', 'INSURANCE', 'SURRENDER'][action]}")

            next_state, reward, done, truncated, info = env.step(action)
            steps += 1

        if done:
            print(f"\nEpisode finished! Final Reward: {reward:.2f}")
            if 'game_result' in info:
                result = info['game_result']
                print(f"Result: {result.result.name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='DQN Blackjack Training and Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'evaluate', 'test'],
                       help='Mode to run: train, evaluate, or test')

    # Environment parameters
    parser.add_argument('--num-decks', type=int, default=6,
                       help='Number of decks in the shoe')
    parser.add_argument('--penetration', type=float, default=0.75,
                       help='Shoe penetration (0-1)')
    parser.add_argument('--allow-surrender', action='store_true', default=True,
                       help='Allow surrender rule')
    parser.add_argument('--dealer-hits-soft-17', action='store_true', default=True,
                       help='Dealer hits on soft 17')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=100000,
                       help='Number of episodes to train/evaluate')
    parser.add_argument('--max-steps', type=int, default=100,
                       help='Maximum steps per episode')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate for optimizer')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--buffer-size', type=int, default=100000,
                       help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--target-update-frequency', type=int, default=1000,
                       help='Steps between target network updates')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                       help='Initial epsilon for exploration')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                       help='Final epsilon for exploration')
    parser.add_argument('--epsilon-decay-steps', type=int, default=100000,
                       help='Steps to decay epsilon')
    parser.add_argument('--hidden-dims', type=str, default='256,256,128',
                       help='Hidden layer dimensions (comma-separated)')
    parser.add_argument('--network-type', type=str, default='standard',
                       choices=['standard', 'dueling'],
                       help='Type of network architecture')

    # Logging and saving
    parser.add_argument('--log-frequency', type=int, default=1000,
                       help='Log progress every N episodes')
    parser.add_argument('--save-frequency', type=int, default=10000,
                       help='Save model every N episodes')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model checkpoint (for evaluation)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations after training')

    args = parser.parse_args()

    # Run appropriate mode
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'test':
        test_game(args)


if __name__ == '__main__':
    main()
