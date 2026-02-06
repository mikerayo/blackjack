"""Entrenamiento optimizado para Vast.ai GPU."""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

print("="*80)
print("ML BLACKJACK - VAST.AI GPU TRAINING")
print("="*80)

# Verificar GPU
print("\n🎮 GPU Information:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   CUDA Version: {torch.version.cuda}")

# Importar después de verificar
from environment.blackjack_env import BlackjackEnv
from agent.scalable_trainer import ScalableTrainer

# Configuración optimizada para GPU
config = {
    # Entorno
    'num_decks': 6,
    'penetration': 0.75,

    # Red neuronal (más grande para GPU)
    'hidden_dims': [1024, 512, 256],  # Red más grande
    'network_type': 'standard',

    # Training optimizado para GPU
    'learning_rate': 0.0001,
    'gamma': 0.99,
    'buffer_size': 1000000,  # Buffer más grande
    'batch_size': 512,  # Batch mucho más grande para GPU
    'target_update_frequency': 10000,

    # Epsilon decay (más lento para más aprendizaje)
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay_steps': 3000000,

    # Expert consensus
    'use_consensus': True,
    'consensus_type': 'hybrid',

    # Variable betting
    'use_variable_betting': True,
    'betting_system': 'hilo',
    'initial_bankroll': 100000.0,
    'min_bet': 10.0,
    'max_bet': 1000.0,

    # Checkpointing
    'save_dir': 'models',
    'checkpoint_interval': 250000,  # Cada 250K episodios
    'log_interval': 25000,  # Log cada 25K
}

print(f"\n⚙️  Configuration:")
for key, value in config.items():
    print(f"   {key}: {value}")

# Crear environment
print("\n🎲 Creating environment...")
env = BlackjackEnv(
    num_decks=config['num_decks'],
    penetration=config['penetration'],
    allow_surrender=True,
    dealer_hits_soft_17=True
)

# Crear trainer
print("🤖 Creating trainer...")
trainer = ScalableTrainer(
    env=env,
    state_dim=9,
    action_dim=6,
    hidden_dims=config['hidden_dims'],
    learning_rate=config['learning_rate'],
    gamma=config['gamma'],
    buffer_size=config['buffer_size'],
    batch_size=config['batch_size'],
    target_update_frequency=config['target_update_frequency'],
    epsilon_decay_steps=config['epsilon_decay_steps'],
    network_type=config['network_type'],
    save_dir=config['save_dir'],
    checkpoint_interval=config['checkpoint_interval'],
    log_interval=config['log_interval'],
    use_consensus=config['use_consensus'],
    consensus_type=config['consensus_type'],
    use_variable_betting=config['use_variable_betting'],
    betting_system=config['betting_system'],
    initial_bankroll=config['initial_bankroll'],
    min_bet=config['min_bet'],
    max_bet=config['max_bet']
)

# Mover redes a GPU
if torch.cuda.is_available():
    trainer.policy_network = trainer.policy_network.cuda()
    trainer.target_network = trainer.target_network.cuda()
    print("🚀 Networks moved to GPU")

print("\n" + "="*80)
print("🚀 STARTING 10 MILLION EPISODE TRAINING ON GPU")
print("="*80)
print(f"⏱️  Estimated time: 2-6 hours on GPU")
print(f"💰 Estimated cost: $1.00 - $3.00")
print("="*80 + "\n")

# Entrenar
try:
    # Reanudar desde checkpoint si existe
    checkpoint_path = None
    checkpoint_file = Path('models/checkpoint_ep500000.pt')
    if checkpoint_file.exists():
        checkpoint_path = str(checkpoint_file)
        print(f"📂 Found checkpoint: {checkpoint_path}")
        print(f"   Resuming from 500K episodes to 10M total")

    summary = trainer.train(
        target_episodes=10000000,
        resume_from=checkpoint_path,
        verbose=True
    )

    # Guardar resumen
    import json
    summary_path = Path('models/training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"📊 Final Results:")
    print(f"   Episodes: {summary['total_episodes']:,}")
    print(f"   Win Rate: {summary['win_rate']:.2f}%")
    print(f"   Total Reward: ${summary['total_reward']:,.2f}")
    print(f"   Final Bankroll: ${summary['final_bankroll']:,.2f}")
    print(f"   Training Time: {summary['training_time']}")
    print("="*80)
    print("\n✨ Model saved in 'models/' directory")
    print("✨ Download the folder to get your trained model!")

except KeyboardInterrupt:
    print("\n⚠️  Training interrupted. Saving emergency checkpoint...")
    trainer.save_checkpoint(trainer.episode_count)
    print(f"✓ Checkpoint saved at episode {trainer.episode_count:,}")

except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
