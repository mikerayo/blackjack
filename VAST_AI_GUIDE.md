"""Script de despliegue en vast.ai.

Este script configura el entorno y ejecuta el entrenamiento masivo
en una GPU alquilada de vast.ai.
"""

import subprocess
import sys
import os

print("="*80)
print("VAST.AI DEPLOYMENT SCRIPT")
print("="*80)

# Paso 1: Instalar dependencias
print("\n[1/4] Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch", "gymnasium", "numpy", "tensorboard",
                "matplotlib", "seaborn", "pandas", "tqdm"])
print("[OK] Dependencies installed")

# Paso 2: Verificar GPU
print("\n[2/4] Checking GPU availability...")
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[OK] GPU detected: {gpu_name}")
    print(f"     Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[WARNING] No GPU detected. Training will be slow.")

# Paso 3: Crear directorios
print("\n[3/4] Creating directories...")
os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("models/logs", exist_ok=True)
os.makedirs("models/metrics", exist_ok=True)
print("[OK] Directories created")

# Paso 4: Iniciar entrenamiento
print("\n[4/4] Starting massive training...")
print("="*80)

from agent.scalable_trainer import ScalableTrainer
from environment.blackjack_env import BlackjackEnv
from strategies import get_all_strategies, create_consensus_system

# Crear environment
env = BlackjackEnv(num_decks=6, penetration=0.75)

# Crear trainer optimizado para GPU
trainer = ScalableTrainer(
    env=env,
    state_dim=9,
    action_dim=6,
    hidden_dims=[512, 256, 128],
    learning_rate=0.0001,
    gamma=0.99,
    buffer_size=500000,
    batch_size=256,  # Batch más grande para GPU
    target_update_frequency=5000,
    epsilon_decay_steps=2000000,
    network_type='standard',
    save_dir='models',
    checkpoint_interval=100000,
    log_interval=10000,
    use_consensus=True,
    consensus_type='hybrid',
    use_variable_betting=True,
    betting_system='hilo',
    initial_bankroll=100000.0,
    min_bet=10.0,
    max_bet=1000.0
)

# Entrenar
print("\n🚀 STARTING GPU-ACCELERATED TRAINING 🚀")
print("="*80 + "\n")

summary = trainer.train(
    target_episodes=5000000,
    resume_from=None,
    verbose=True
)

# Guardar resumen final
import json
with open('models/training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("✅ TRAINING COMPLETED!")
print("="*80)
print(f"Total Episodes: {summary['total_episodes']:,}")
print(f"Total Reward: ${summary['total_reward']:,.2f}")
print(f"Win Rate: {summary['win_rate']:.2f}%")
print(f"Final Bankroll: ${summary['final_bankroll']:,.2f}")
print(f"Training Time: {summary['training_time']}")
print("="*80 + "\n")

# Crear archivo de señalización de completado
with open('models/TRAINING_COMPLETE.txt', 'w') as f:
    f.write("Training completed successfully!\n")
    f.write(f"Episodes: {summary['total_episodes']:,}\n")
    f.write(f"Win Rate: {summary['win_rate']:.2f}%\n")

print("🎉 All done! Download the 'models' folder to get your trained model.")
