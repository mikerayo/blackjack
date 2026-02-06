"""Visualization utilities for Blackjack RL training."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
import pandas as pd

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_training_rewards(episode_rewards: List[float],
                          window: int = 100,
                          save_path: Optional[str] = None) -> None:
    """
    Plot training rewards over episodes.

    Args:
        episode_rewards: List of episode rewards
        window: Moving average window size
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot raw rewards
    ax.plot(episode_rewards, alpha=0.3, color='blue', linewidth=0.5, label='Raw Reward')

    # Plot moving average
    if len(episode_rewards) >= window:
        moving_avg = pd.Series(episode_rewards).rolling(window=window).mean()
        ax.plot(moving_avg, color='red', linewidth=2, label=f'{window}-Episode Moving Average')

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Training Rewards Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_win_rate(episode_rewards: List[float],
                  window: int = 100,
                  save_path: Optional[str] = None) -> None:
    """
    Plot win rate over episodes.

    Args:
        episode_rewards: List of episode rewards
        window: Moving average window size
        save_path: Path to save the figure
    """
    # Convert rewards to wins (1) or losses (0)
    wins = [1 if r > 0 else 0 for r in episode_rewards]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Calculate moving average
    if len(wins) >= window:
        win_rate = pd.Series(wins).rolling(window=window).mean() * 100
        ax.plot(win_rate, color='green', linewidth=2, label=f'{window}-Episode Win Rate')

    # Add 50% line (break-even)
    ax.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Break-even (50%)')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title('Win Rate Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_cumulative_profit(episode_rewards: List[float],
                           save_path: Optional[str] = None) -> None:
    """
    Plot cumulative profit over episodes.

    Args:
        episode_rewards: List of episode rewards
        save_path: Path to save the figure
    """
    cumulative_profit = np.cumsum(episode_rewards)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(cumulative_profit, color='green' if cumulative_profit[-1] > 0 else 'red',
            linewidth=2)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cumulative Profit ($)', fontsize=12)
    ax.set_title('Cumulative Profit Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add final profit text
    final_profit = cumulative_profit[-1]
    color = 'green' if final_profit > 0 else 'red'
    ax.text(0.02, 0.98, f'Final Profit: ${final_profit:.2f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_loss_curve(episode_losses: List[float],
                    window: int = 50,
                    save_path: Optional[str] = None) -> None:
    """
    Plot training loss over episodes.

    Args:
        episode_losses: List of episode losses
        window: Moving average window size
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot raw losses
    ax.plot(episode_losses, alpha=0.3, color='orange', linewidth=0.5, label='Raw Loss')

    # Plot moving average
    if len(episode_losses) >= window:
        moving_avg = pd.Series(episode_losses).rolling(window=window).mean()
        ax.plot(moving_avg, color='red', linewidth=2, label=f'{window}-Episode Moving Average')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_epsilon_decay(epsilon_values: List[float],
                       save_path: Optional[str] = None) -> None:
    """
    Plot epsilon decay over training steps.

    Args:
        epsilon_values: List of epsilon values over training
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(epsilon_values, color='purple', linewidth=2)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
    ax.set_title('Epsilon Decay Over Training', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_strategy_comparison(metrics_dict: Dict[str, Dict],
                             save_path: Optional[str] = None) -> None:
    """
    Plot comparison between different strategies.

    Args:
        metrics_dict: Dictionary of strategy names to metrics
        save_path: Path to save the figure
    """
    strategies = list(metrics_dict.keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Extract metrics
    win_rates = [metrics_dict[s]['win_rate'] for s in strategies]
    rois = [metrics_dict[s]['roi'] for s in strategies]
    advantages = [metrics_dict[s]['advantage_over_house'] for s in strategies]
    sharpe_ratios = [metrics_dict[s]['sharpe_ratio'] for s in strategies]

    # Plot win rates
    axes[0, 0].bar(strategies, win_rates, color='steelblue')
    axes[0, 0].axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[0, 0].set_ylabel('Win Rate (%)', fontsize=11)
    axes[0, 0].set_title('Win Rate Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Plot ROI
    colors = ['green' if roi > 0 else 'red' for roi in rois]
    axes[0, 1].bar(strategies, rois, color=colors)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[0, 1].set_ylabel('ROI (%)', fontsize=11)
    axes[0, 1].set_title('Return on Investment Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot advantage over house
    colors = ['green' if adv > 0 else 'red' for adv in advantages]
    axes[1, 0].bar(strategies, advantages, color=colors)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].axhline(y=0.005, color='blue', linestyle='--', linewidth=1,
                      alpha=0.5, label='Typical House Edge (0.5%)')
    axes[1, 0].set_ylabel('Advantage per Hand ($)', fontsize=11)
    axes[1, 0].set_title('Advantage Over House', fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot Sharpe ratios
    axes[1, 1].bar(strategies, sharpe_ratios, color='purple')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_ylabel('Sharpe Ratio', fontsize=11)
    axes[1, 1].set_title('Risk-Adjusted Returns (Sharpe Ratio)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_action_distribution(action_counts: Dict[int, int],
                            action_names: List[str],
                            save_path: Optional[str] = None) -> None:
    """
    Plot distribution of actions taken by the agent.

    Args:
        action_counts: Dictionary mapping action IDs to counts
        action_names: List of action names
        save_path: Path to save the figure
    """
    actions = [action_names[i] for i in action_counts.keys()]
    counts = list(action_counts.values())

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(actions, counts, color='steelblue')

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Action', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Action Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_dashboard(episode_rewards: List[float],
                    episode_losses: Optional[List[float]] = None,
                    window: int = 100,
                    save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive training dashboard.

    Args:
        episode_rewards: List of episode rewards
        episode_losses: List of episode losses (optional)
        window: Moving average window size
        save_path: Path to save the figure
    """
    fig = plt.figure(figsize=(16, 10))

    # Rewards plot
    ax1 = plt.subplot(2, 2, 1)
    if len(episode_rewards) >= window:
        moving_avg = pd.Series(episode_rewards).rolling(window=window).mean()
        ax1.plot(moving_avg, color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
    ax1.plot(episode_rewards, alpha=0.2, color='blue', linewidth=0.5)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Win rate plot
    ax2 = plt.subplot(2, 2, 2)
    wins = [1 if r > 0 else 0 for r in episode_rewards]
    if len(wins) >= window:
        win_rate = pd.Series(wins).rolling(window=window).mean() * 100
        ax2.plot(win_rate, color='green', linewidth=2)
    ax2.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Win Rate Over Time', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    # Cumulative profit plot
    ax3 = plt.subplot(2, 2, 3)
    cumulative_profit = np.cumsum(episode_rewards)
    color = 'green' if cumulative_profit[-1] > 0 else 'red'
    ax3.plot(cumulative_profit, color=color, linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Cumulative Profit ($)')
    ax3.set_title('Cumulative Profit', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Loss plot (if available)
    ax4 = plt.subplot(2, 2, 4)
    if episode_losses and len(episode_losses) > 0:
        if len(episode_losses) >= window:
            loss_avg = pd.Series(episode_losses).rolling(window=window).mean()
            ax4.plot(loss_avg, color='orange', linewidth=2)
        ax4.plot(episode_losses, alpha=0.2, color='red', linewidth=0.5)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training Loss', fontweight='bold')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Loss data not available',
                ha='center', va='center', fontsize=12,
                transform=ax4.transAxes)
        ax4.set_title('Training Loss', fontweight='bold')

    plt.suptitle('Training Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
