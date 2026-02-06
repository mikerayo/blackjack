"""Script to monitor training progress."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time
import os
from pathlib import Path

def find_latest_metrics():
    """Find the latest training metrics file."""
    models_dir = Path('models')
    metrics_files = list(models_dir.glob('training_metrics_*.json'))

    if not metrics_files:
        return None

    # Sort by modification time
    latest = max(metrics_files, key=lambda f: f.stat().st_mtime)
    return latest

def get_progress():
    """Get training progress if training is active."""
    # Check if training process is running
    import subprocess
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'],
                              capture_output=True, text=True)
        if 'python.exe' not in result.stdout:
            return None, False
    except:
        pass

    # Find latest metrics file
    metrics_file = find_latest_metrics()
    if metrics_file is None:
        return None, False

    # Read metrics
    import json
    with open(metrics_file, 'r') as f:
        data = json.load(f)

    rewards = data['episode_rewards']
    wins = data['episode_wins']

    return {
        'episodes': len(rewards),
        'win_rate': sum(wins) / len(wins) * 100,
        'mean_reward': sum(rewards) / len(rewards),
        'last_1000_wr': sum(wins[-1000:]) / min(1000, len(wins)) * 100 if len(wins) >= 1000 else None,
        'last_1000_mr': sum(rewards[-1000:]) / min(1000, len(rewards)) if len(rewards) >= 1000 else None,
    }, True

def main():
    print("="*70)
    print("MONITOR DE ENTRENAMIENTO - 500K EPISODES")
    print("="*70)

    target_episodes = 500000
    last_episodes = 0

    try:
        while True:
            progress, training = get_progress()

            if progress is None:
                print("No se encontraron metrics. Esperando...")
                time.sleep(10)
                continue

            episodes = progress['episodes']
            percent = episodes / target_episodes * 100

            print("\n" + "="*70)
            print(f"PROGRESO: {episodes:,}/{target_episodes:,} ({percent:.2f}%)")
            print("="*70)
            print(f"\nEstadísticas Globales:")
            print(f"  Win Rate:   {progress['win_rate']:.2f}%")
            print(f"  Mean Reward: {progress['mean_reward']:.4f}")

            if progress['last_1000_wr']:
                print(f"\nÚltimos 1,000 episodios:")
                print(f"  Win Rate:   {progress['last_1000_wr']:.2f}%")
                print(f"  Mean Reward: {progress['last_1000_mr']:.4f}")

            # Estimate time remaining
            if episodes > last_episodes:
                eps_per_sec = episodes - last_episodes
                remaining = target_episodes - episodes
                eta_seconds = remaining / eps_per_sec if eps_per_sec > 0 else 0
                eta_hours = eta_seconds / 3600

                print(f"\nVelocidad: {eps_per_sec:,} eps/sec (últimos 10s)")
                print(f"Tiempo restante estimado: {eta_hours:.1f} horas")

            last_episodes = episodes

            if not training:
                print("\n[Training completed or stopped]")
                break

            print("\nPróxima actualización en 30 segundos...")
            time.sleep(30)

    except KeyboardInterrupt:
        print("\n\nMonitor detenido.")

if __name__ == "__main__":
    main()
