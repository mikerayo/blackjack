"""Evaluación SIMPLE de expertos - Solo primera decisión de cada mano.

Esto elimina el problema de decisiones múltiples por mano.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environment.blackjack_env import BlackjackEnv
from game.rules import Action
from strategies import get_all_strategies, create_consensus_system


def evaluate_first_decision_only(num_episodes=10000):
    """Evalúa solo la primera decisión de cada mano ( HIT/STAND/DOUBLE)."""

    print("="*80)
    print("EVALUACIÓN EXPERTOS - SOLO PRIMERA DECISIÓN")
    print("="*80)
    print(f"\nEvaluando: {num_episodes:,} episodios\n")

    env = BlackjackEnv(num_decks=6, penetration=0.75)
    strategies = get_all_strategies()
    consensus = create_consensus_system('hybrid', strategies)

    results = []

    print("Ejecutando episodios...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        game_state = env.game.get_state()

        # Solo evaluamos la primera acción
        valid_actions = list(range(env.action_space.n))
        result = consensus.get_consensus(game_state, valid_actions)
        action = result.selected_action

        if hasattr(action, 'value'):
            action = int(action.value)
        elif not isinstance(action, int):
            action = int(action)

        # Ejecutar UNA sola decisión
        next_state, reward, done, truncated, info = env.step(action)

        # Usar la reward resultante
        results.append(reward)

        if (episode + 1) % 1000 == 0:
            win_rate = sum(1 for r in results[-1000:] if r > 0) / len(results[-1000:]) * 100
            avg_reward = sum(results[-1000:]) / len(results[-1000:])
            print(f"Episode {episode + 1:,}: Win Rate (last 1K): {win_rate:.1f}%, Avg Reward: ${avg_reward:.2f}")

    # Métricas finales
    win_rate = sum(1 for r in results if r > 0) / len(results) * 100
    avg_reward = sum(results) / len(results)
    total_profit = sum(results)

    print("\n" + "="*80)
    print("RESULTADOS FINALES")
    print("="*80)
    print(f"\nEpisodios: {num_episodes:,}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Mean Reward: ${avg_reward:.4f}")

    if avg_reward > 0:
        print(f"\n[+] VENTAJA: ${avg_reward:.4f} por mano")
    else:
        print(f"\n[!] HOUSE EDGE: ${abs(avg_reward):.4f} por mano")

    print("="*80 + "\n")

    return {'win_rate': win_rate, 'avg_reward': avg_reward}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluar expertos - primera decisión')
    parser.add_argument('--episodes', type=int, default=10000)

    args = parser.parse_args()

    evaluate_first_decision_only(args.episodes)
