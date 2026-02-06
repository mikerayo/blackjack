"""Test ABSOLUTAMENTE simple - solo Basic Strategy del environment."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environment.blackjack_env import BlackjackEnv

print("="*80)
print("TEST SIMPLE - BASIC STRATEGY DEL ENVIRONMENT")
print("="*80)

env = BlackjackEnv(num_decks=6, penetration=0.75)

num_episodes = 1000
rewards = []

print(f"\nEjecutando {num_episodes} episodios con Basic Strategy...")

for episode in range(num_episodes):
    state, _ = env.reset()

    done = False
    episode_reward = 0

    while not done:
        # Usar el basic strategy del environment
        action = env.get_basic_strategy_action()

        # Step
        next_state, reward, done, truncated, info = env.step(action)
        episode_reward += reward

    rewards.append(episode_reward)

    if (episode + 1) % 100 == 0:
        recent = rewards[-100:]
        win_rate = sum(1 for r in recent if r > 0) / len(recent) * 100
        print(f"Episode {episode + 1}: Win Rate (last 100): {win_rate:.1f}%")

# Resultados
win_rate = sum(1 for r in rewards if r > 0) / len(rewards) * 100
avg_reward = sum(rewards) / len(rewards)

print("\n" + "="*80)
print("RESULTADOS")
print("="*80)
print(f"Episodios: {num_episodes}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Mean Reward: ${avg_reward:.4f}")
print("="*80 + "\n")
