"""Evaluate Expert Strategies and Consensus Systems.

Test all expert strategies against each other and against the DQN agent.
"""

import sys
import os
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environment.blackjack_env import BlackjackEnv
from game.blackjack import BlackjackGame
from strategies import (
    get_all_strategies,
    create_consensus_system,
    get_available_systems,
    create_betting_system,
    get_available_betting_systems
)
from utils.metrics import calculate_metrics, print_metrics_report


@dataclass
class StrategyEvaluation:
    """Results from evaluating a strategy."""
    name: str
    episodes: int
    total_reward: float
    win_rate: float
    roi: float
    sharpe_ratio: float
    mean_reward: float
    std_reward: float


def evaluate_strategy(strategy, env: BlackjackEnv,
                     num_episodes: int = 10000,
                     use_variable_betting: bool = False,
                     betting_system = None) -> StrategyEvaluation:
    """
    Evaluate a single strategy.

    Args:
        strategy: Strategy object with get_action method
        env: Blackjack environment
        num_episodes: Number of episodes to evaluate
        use_variable_betting: Whether to use variable betting
        betting_system: Bankroll management system

    Returns:
        StrategyEvaluation object
    """
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        game_state = env.get_state()

        # Get bet amount if using variable betting
        if use_variable_betting and betting_system:
            bet_decision = betting_system.get_bet(game_state)
            bet_amount = bet_decision.bet_amount
            env.game.current_bet = bet_amount
        else:
            bet_amount = 1.0

        episode_reward = 0.0
        done = False

        while not done:
            # Get valid actions
            valid_actions = list(range(env.action_space.n))

            # Get action from strategy
            try:
                action = strategy.get_action(game_state, valid_actions)
                # Convert Action enum to int if needed
                if hasattr(action, 'value'):
                    action = int(action.value)
            except:
                # Fallback to random action
                action = env.action_space.sample()

            # Step environment
            next_state, reward, done, truncated, info = env.step(action)
            game_state = env.get_state()

            # Scale reward by bet amount
            episode_reward += reward * bet_amount

        rewards.append(episode_reward)

        # Update betting system
        if use_variable_betting and betting_system:
            betting_system.update_bankroll(episode_reward)

    # Calculate metrics
    metrics = calculate_metrics(rewards, bet_size=1.0)

    return StrategyEvaluation(
        name=strategy.name if hasattr(strategy, 'name') else str(strategy),
        episodes=num_episodes,
        total_reward=metrics.total_profit,
        win_rate=metrics.win_rate,
        roi=metrics.roi,
        sharpe_ratio=metrics.sharpe_ratio,
        mean_reward=metrics.mean_reward,
        std_reward=metrics.std_reward
    )


def compare_all_strategies(num_episodes: int = 10000) -> Dict[str, StrategyEvaluation]:
    """
    Compare all expert strategies.

    Args:
        num_episodes: Number of episodes per strategy

    Returns:
        Dictionary mapping strategy names to evaluation results
    """
    print("="*80)
    print("EVALUATING ALL EXPERT STRATEGIES")
    print("="*80)

    # Create environment
    env = BlackjackEnv(num_decks=6, penetration=0.75)

    # Get all strategies
    strategies = get_all_strategies()

    results = {}

    for strategy in strategies:
        print(f"\nEvaluating: {strategy.name}")
        print("-" * 80)

        # Reset betting system for each strategy
        betting_system = create_betting_system('hilo', initial_bankroll=10000)
        betting_system.reset()

        result = evaluate_strategy(
            strategy,
            env,
            num_episodes=num_episodes,
            use_variable_betting=True,
            betting_system=betting_system
        )

        results[result.name] = result

        print(f"  Win Rate: {result.win_rate:.2f}%")
        print(f"  ROI: {result.roi:.2f}%")
        print(f"  Total Profit: ${result.total_reward:,.2f}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.4f}")

    return results


def compare_consensus_systems(num_episodes: int = 10000) -> Dict[str, StrategyEvaluation]:
    """
    Compare different consensus systems.

    Args:
        num_episodes: Number of episodes per system

    Returns:
        Dictionary mapping system names to evaluation results
    """
    print("\n" + "="*80)
    print("EVALUATING CONSENSUS SYSTEMS")
    print("="*80)

    # Create environment
    env = BlackjackEnv(num_decks=6, penetration=0.75)

    # Get all expert strategies
    expert_strategies = get_all_strategies()

    # Get available consensus systems
    system_types = get_available_systems()

    results = {}

    for system_type in system_types:
        print(f"\nEvaluating: {system_type.upper()} Consensus")
        print("-" * 80)

        # Create consensus system
        consensus = create_consensus_system(system_type, expert_strategies)

        # Create betting system
        betting_system = create_betting_system('hilo', initial_bankroll=10000)
        betting_system.reset()

        # Evaluate
        result = evaluate_strategy(
            consensus,
            env,
            num_episodes=num_episodes,
            use_variable_betting=True,
            betting_system=betting_system
        )

        results[system_type] = result

        print(f"  Win Rate: {result.win_rate:.2f}%")
        print(f"  ROI: {result.roi:.2f}%")
        print(f"  Total Profit: ${result.total_reward:,.2f}")

    return results


def compare_betting_systems(num_episodes: int = 10000) -> Dict[str, StrategyEvaluation]:
    """
    Compare different betting systems with fixed strategy.

    Args:
        num_episodes: Number of episodes per system

    Returns:
        Dictionary mapping system names to evaluation results
    """
    print("\n" + "="*80)
    print("EVALUATING BETTING SYSTEMS")
    print("="*80)

    # Create environment
    env = BlackjackEnv(num_decks=6, penetration=0.75)

    # Use Hi-Lo strategy as base
    from strategies import HiLoCountingStrategy
    base_strategy = HiLoCountingStrategy()

    # Get available betting systems
    system_types = get_available_betting_systems()

    results = {}

    for betting_type in system_types:
        print(f"\nEvaluating: {betting_type.upper()} Betting")
        print("-" * 80)

        # Create betting system
        betting_system = create_betting_system(
            betting_type,
            initial_bankroll=10000,
            min_bet=10,
            max_bet=500
        )
        betting_system.reset()

        # Evaluate
        result = evaluate_strategy(
            base_strategy,
            env,
            num_episodes=num_episodes,
            use_variable_betting=True,
            betting_system=betting_system
        )

        results[betting_type] = result

        print(f"  Win Rate: {result.win_rate:.2f}%")
        print(f"  ROI: {result.roi:.2f}%")
        print(f"  Final Bankroll: ${result.total_reward + 10000:,.2f}")

    return results


def print_comparison_table(results: Dict[str, StrategyEvaluation],
                          title: str = "STRATEGY COMPARISON"):
    """Print a formatted comparison table."""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)
    print(f"{'Strategy':<25} {'Win Rate':>12} {'ROI':>12} {'Profit':>15} {'Sharpe':>10}")
    print("-"*80)

    # Sort by total profit
    sorted_results = sorted(results.items(), key=lambda x: x[1].total_reward, reverse=True)

    for name, result in sorted_results:
        print(f"{name:<25} {result.win_rate:>10.2f}% "
              f"{result.roi:>10.2f}% "
              f"${result.total_reward:>13,.2f} "
              f"{result.sharpe_ratio:>9.4f}")

    print("="*80)


def main():
    """Run all comparisons."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Expert Strategies')
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of episodes per evaluation')
    parser.add_argument('--type', type=str, default='all',
                       choices=['all', 'strategies', 'consensus', 'betting'],
                       help='What to evaluate')

    args = parser.parse_args()

    if args.type in ['all', 'strategies']:
        print("\n" + "🎯"*40)
        print("COMPARING EXPERT STRATEGIES")
        print("🎯"*40)
        strategy_results = compare_all_strategies(args.episodes)
        print_comparison_table(strategy_results, "EXPERT STRATEGIES COMPARISON")

    if args.type in ['all', 'consensus']:
        print("\n" + "🤝"*40)
        print("COMPARING CONSENSUS SYSTEMS")
        print("🤝"*40)
        consensus_results = compare_consensus_systems(args.episodes)
        print_comparison_table(consensus_results, "CONSENSUS SYSTEMS COMPARISON")

    if args.type in ['all', 'betting']:
        print("\n" + "💰"*40)
        print("COMPARING BETTING SYSTEMS")
        print("💰"*40)
        betting_results = compare_betting_systems(args.episodes)
        print_comparison_table(betting_results, "BETTING SYSTEMS COMPARISON")

    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
