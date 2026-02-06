"""Metrics and evaluation utilities for Blackjack RL."""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from environment.blackjack_env import BlackjackEnv


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    total_episodes: int
    total_hands: int
    wins: int
    losses: int
    pushes: int
    blackjacks: int
    total_profit: float
    mean_reward: float
    std_reward: float
    win_rate: float
    roi: float  # Return on Investment
    sharpe_ratio: float  # Risk-adjusted return
    house_edge: float  # Negative advantage means house edge
    advantage_over_house: float  # Positive means player advantage


def calculate_metrics(rewards: List[float],
                      bet_size: float = 1.0) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from reward history.

    Args:
        rewards: List of episode rewards
        bet_size: Standard bet size per hand

    Returns:
        PerformanceMetrics object with calculated metrics
    """
    total_hands = len(rewards)

    # Count outcomes
    wins = sum(1 for r in rewards if r > 0)
    losses = sum(1 for r in rewards if r < 0)
    pushes = sum(1 for r in rewards if r == 0)

    # Count blackjacks (1.5 payout)
    blackjacks = sum(1 for r in rewards if r >= 1.4)  # Approximately 1.5

    # Basic statistics
    total_profit = sum(rewards)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # Win rate
    win_rate = (wins / total_hands * 100) if total_hands > 0 else 0

    # ROI (Return on Investment)
    total_invested = total_hands * bet_size
    roi = (total_profit / total_invested * 100) if total_invested > 0 else 0

    # Sharpe Ratio (risk-adjusted return)
    if std_reward > 0:
        sharpe_ratio = (mean_reward / std_reward) * np.sqrt(total_hands)
    else:
        sharpe_ratio = 0.0

    # House edge / Player advantage
    # In blackjack, house edge is typically around 0.5-1%
    # Negative = house advantage, Positive = player advantage
    house_edge = -mean_reward
    advantage_over_house = mean_reward

    return PerformanceMetrics(
        total_episodes=total_hands,
        total_hands=total_hands,
        wins=wins,
        losses=losses,
        pushes=pushes,
        blackjacks=blackjacks,
        total_profit=total_profit,
        mean_reward=mean_reward,
        std_reward=std_reward,
        win_rate=win_rate,
        roi=roi,
        sharpe_ratio=sharpe_ratio,
        house_edge=house_edge,
        advantage_over_house=advantage_over_house
    )


def evaluate_basic_strategy(env: BlackjackEnv, num_episodes: int = 10000) -> PerformanceMetrics:
    """
    Evaluate the performance of basic strategy.

    Args:
        env: Blackjack environment
        num_episodes: Number of episodes to evaluate

    Returns:
        PerformanceMetrics for basic strategy
    """
    rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()

        done = False
        episode_reward = 0

        while not done:
            # Get basic strategy action
            action = env.get_basic_strategy_action()

            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)

    return calculate_metrics(rewards)


def evaluate_random_strategy(env: BlackjackEnv, num_episodes: int = 10000) -> PerformanceMetrics:
    """
    Evaluate the performance of random strategy.

    Args:
        env: Blackjack environment
        num_episodes: Number of episodes to evaluate

    Returns:
        PerformanceMetrics for random strategy
    """
    rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()

        done = False
        episode_reward = 0

        while not done:
            # Random action
            action = env.action_space.sample()

            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)

    return calculate_metrics(rewards)


def compare_strategies(dqn_rewards: List[float],
                       basic_strategy_rewards: List[float],
                       random_rewards: List[float]) -> Dict:
    """
    Compare multiple strategies and return comparison metrics.

    Args:
        dqn_rewards: Rewards from DQN agent
        basic_strategy_rewards: Rewards from basic strategy
        random_rewards: Rewards from random strategy

    Returns:
        Dictionary with comparison results
    """
    dqn_metrics = calculate_metrics(dqn_rewards)
    bs_metrics = calculate_metrics(basic_strategy_rewards)
    random_metrics = calculate_metrics(random_rewards)

    comparison = {
        'dqn': dqn_metrics,
        'basic_strategy': bs_metrics,
        'random': random_metrics,
        'improvement_over_basic': {
            'win_rate': dqn_metrics.win_rate - bs_metrics.win_rate,
            'roi': dqn_metrics.roi - bs_metrics.roi,
            'advantage': dqn_metrics.advantage_over_house - bs_metrics.advantage_over_house
        },
        'improvement_over_random': {
            'win_rate': dqn_metrics.win_rate - random_metrics.win_rate,
            'roi': dqn_metrics.roi - random_metrics.roi,
            'advantage': dqn_metrics.advantage_over_house - random_metrics.advantage_over_house
        }
    }

    return comparison


def print_metrics_report(metrics: PerformanceMetrics, strategy_name: str = "Strategy") -> None:
    """
    Print a formatted metrics report.

    Args:
        metrics: PerformanceMetrics object
        strategy_name: Name of the strategy
    """
    print(f"\n{'='*60}")
    print(f"{strategy_name} Performance Report")
    print(f"{'='*60}")

    print(f"\nHands Played: {metrics.total_hands:,}")
    print(f"Wins: {metrics.wins:,} ({metrics.win_rate:.2f}%)")
    print(f"Losses: {metrics.losses:,}")
    print(f"Pushes: {metrics.pushes:,}")
    print(f"Blackjacks: {metrics.blackjacks:,}")

    print(f"\nFinancial Metrics:")
    print(f"  Total Profit/Loss: ${metrics.total_profit:.2f}")
    print(f"  Mean Reward per Hand: ${metrics.mean_reward:.4f}")
    print(f"  Std Deviation: ${metrics.std_reward:.4f}")
    print(f"  ROI: {metrics.roi:.2f}%")

    print(f"\nRisk Metrics:")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.4f}")

    print(f"\nAdvantage Analysis:")
    if metrics.advantage_over_house > 0:
        print(f"  Player Advantage: +{metrics.advantage_over_house:.4f} per hand")
        print(f"  Expected Return: +{metrics.advantage_over_house * 100:.2f}%")
    else:
        print(f"  House Edge: {abs(metrics.advantage_over_house):.4f} per hand")
        print(f"  Expected Return: {metrics.advantage_over_house * 100:.2f}%")

    # Theoretical comparison
    print(f"\nComparison to Casino Edge:")
    typical_house_edge = 0.005  # 0.5% typical house edge
    if metrics.advantage_over_house > typical_house_edge:
        advantage = metrics.advantage_over_house - typical_house_edge
        print(f"  [+] BEATING the house by +{advantage:.4f} per hand")
    elif metrics.advantage_over_house > 0:
        print(f"  [-] Winning but below typical advantage threshold")
    else:
        print(f"  [X] Still losing to the house")

    print(f"{'='*60}\n")


def calculate_confidence_interval(rewards: List[float],
                                   confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for mean reward.

    Args:
        rewards: List of rewards
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    n = len(rewards)
    mean = np.mean(rewards)
    std_err = np.std(rewards) / np.sqrt(n)

    # Z-score for 95% confidence
    z_score = 1.96

    margin_of_error = z_score * std_err

    return (mean - margin_of_error, mean + margin_of_error)


def is_profitable(metrics: PerformanceMetrics,
                   confidence_level: float = 0.95) -> Tuple[bool, float]:
    """
    Determine if a strategy is truly profitable using statistical testing.

    Args:
        metrics: PerformanceMetrics object
        confidence_level: Confidence level for statistical test

    Returns:
        Tuple of (is_profitable, p_value)
    """
    # Simple t-test: is mean reward significantly > 0?
    if metrics.std_reward == 0:
        return metrics.mean_reward > 0, 1.0

    # Calculate t-statistic
    t_stat = metrics.mean_reward / (metrics.std_reward / np.sqrt(metrics.total_hands))

    # For large samples, t-stat approximates normal distribution
    # At 95% confidence, we need t_stat > 1.96
    threshold = 1.96 if confidence_level == 0.95 else 1.645

    is_profitable = t_stat > threshold

    # Rough p-value approximation
    p_value = 2 * (1 - 0.5 * (1 + np.sign(t_stat) * (1 - np.exp(-2 * t_stat**2 / np.pi))))

    return is_profitable, p_value
