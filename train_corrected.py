"""Entrenamiento CORREGIDO - Soluciona los problemas del entrenamiento anterior.

Problemas identificados:
1. Epsilon decay demasiado lento (2M steps) -> epsilon sigue ~0.93 después de 100K
2. Sistema de consenso casi no se usa (por epsilon alto)
3. Apuestas variables + aleatoriedad = pérdidas masivas

Soluciones:
1. Curriculum learning: empieza con expertos, transiciona a DQN gradualmente
2. Epsilon decay más rápido
3. Fase inicial solo expertos para ver rendimiento real
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environment.blackjack_env import BlackjackEnv
from agent.curriculum_trainer import CurriculumTrainer


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Entrenamiento Corregido con Curriculum Learning')
    parser.add_argument('--episodes', type=int, default=100000,
                       help='Total episodes (default: 100K)')
    parser.add_argument('--phase1', type=int, default=10000,
                       help='Episodios solo expertos (default: 10K)')
    parser.add_argument('--phase2', type=int, default=30000,
                       help='Episodios expert-guided (default: 30K)')
    parser.add_argument('--phase3', type=int, default=60000,
                       help='Episodios DQN-dominant (default: 60K)')
    parser.add_argument('--log-interval', type=int, default=5000,
                       help='Log cada N episodios')

    args = parser.parse_args()

    print("="*80)
    print("ENTRENAMIENTO CORREGIDO - CURRICULUM LEARNING")
    print("="*80)
    print("\n📋 Configuración:")
    print(f"  Total: {args.episodes:,} episodios")
    print(f"  Fase 1 (Solo Expertos): {args.phase1:,} episodios")
    print(f"  Fase 2 (Expert-Guided): {args.phase2:,} episodios")
    print(f"  Fase 3 (DQN-Dominant): {args.phase3:,} episodios")
    print(f"  Fase 4 (Pure DQN): {args.episodes - args.phase1 - args.phase2 - args.phase3:,} episodios")
    print("\n💡 Por qué funciona:")
    print("  1. Empieza con expertos (win rate ~43-45%)")
    print("  2. Transiciona gradualmente a DQN")
    print("  3. DQN aprende de las decisiones de expertos")
    print("  4. Evita el problema de epsilon alto")
    print("="*80 + "\n")

    # Crear environment
    env = BlackjackEnv(num_decks=6, penetration=0.75)

    # Crear curriculum trainer
    trainer = CurriculumTrainer(
        env=env,
        state_dim=9,
        action_dim=6,
        hidden_dims=[512, 256, 128],
        learning_rate=0.0001,
        gamma=0.99,
        buffer_size=200000,
        batch_size=128,
        save_dir='models'
    )

    # Entrenar
    summary = trainer.train_curriculum(
        total_episodes=args.episodes,
        phase1_episodes=args.phase1,
        phase2_episodes=args.phase2,
        phase3_episodes=args.phase3,
        log_interval=args.log_interval
    )

    print("\n✅ ENTRENAMIENTO COMPLETADO!")
    print(f"Win Rate Final: {summary['win_rate']:.2f}%")
    print(f"Bankroll Final: ${summary['final_bankroll']:,.2f}")


if __name__ == '__main__':
    main()
