"""Evalúa el rendimiento de los expertos solos (sin DQN).

Esto nos dice cuál es el verdadero potencial del sistema.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environment.blackjack_env import BlackjackEnv
from strategies import get_all_strategies, create_consensus_system
from utils.metrics import calculate_metrics


def evaluate_pure_experts(num_episodes=10000):
    """Evalúa solo el sistema de expertos (sin DQN)."""

    print("="*80)
    print("EVALUANDO SISTEMA DE EXPERTOS PURO")
    print("="*80)
    print(f"\nEvaluando: {num_episodes:,} episodios\n")

    # Crear environment
    env = BlackjackEnv(num_decks=6, penetration=0.75)

    # Crear sistema de expertos
    strategies = get_all_strategies()
    consensus = create_consensus_system('hybrid', strategies)

    # Crear sistema de apuestas
    from strategies import create_betting_system
    betting = create_betting_system('hilo', initial_bankroll=10000)
    betting.reset()

    rewards = []

    print("Ejecutando episodios...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        game_state = env.game.get_state()

        # Get bet
        bet_decision = betting.get_bet(game_state)
        env.game.current_bet = bet_decision.bet_amount

        episode_reward = 0
        done = False

        while not done:
            valid_actions = list(range(env.action_space.n))

            # Get expert action
            result = consensus.get_consensus(game_state, valid_actions)
            action = result.selected_action

            # Convert to int (handle both Action enum and int)
            if hasattr(action, 'value'):
                action = int(action.value)
            elif not isinstance(action, int):
                action = int(action)

            # Step
            next_state, reward, done, truncated, info = env.step(action)
            game_state = env.game.get_state()

            episode_reward += reward * bet_decision.bet_amount

        rewards.append(episode_reward)

        # Update betting
        betting.update_bankroll(episode_reward)

        if (episode + 1) % 1000 == 0:
            recent = rewards[-1000:]
            win_rate = sum(1 for r in recent if r > 0) / len(recent) * 100
            print(f"Episode {episode + 1:,}: Win Rate (last 1K): {win_rate:.1f}%, Bankroll: ${betting.current_bankroll:,.2f}")

    # Calculate final metrics
    metrics = calculate_metrics(rewards, bet_size=1.0)

    print("\n" + "="*80)
    print("RESULTADOS FINALES - EXPERTOS PUROS")
    print("="*80)
    print(f"\nEpisodios: {num_episodes:,}")
    print(f"Win Rate: {metrics.win_rate:.2f}%")
    print(f"Total Reward: ${metrics.total_profit:,.2f}")
    print(f"Mean Reward per Hand: ${metrics.mean_reward:.4f}")
    print(f"Std Deviation: ${metrics.std_reward:.4f}")
    print(f"ROI: {metrics.roi:.2f}%")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
    print(f"Final Bankroll: ${betting.current_bankroll:,.2f}")

    if metrics.advantage_over_house > 0:
        print(f"\n[+] VENTAJA SOBRE LA CASA: +{metrics.advantage_over_house:.4f} por mano")
    else:
        print(f"\n[!] HOUSE EDGE: {abs(metrics.advantage_over_house):.4f} por mano")

    print("="*80 + "\n")

    return metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluar expertos puros')
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Número de episodios')

    args = parser.parse_args()

    evaluate_pure_experts(args.episodes)
