"""
Fine-tuning del modelo con datos reales de casinos.

Usa este script DESPUÉS de entrenar el modelo de 10M episodios.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environment.blackjack_env import BlackjackEnv
from agent.dqn import DQN
from agent.scalable_trainer import ScalableTrainer


def load_casino_sessions(data_dir):
    """Cargar sesiones de datos reales de casino."""
    data_path = Path(data_dir)
    sessions = []

    print(f"📂 Cargando datos desde {data_path}...")

    # Buscar archivos JSON de sesiones
    for json_file in data_path.glob("session_*.json"):
        with open(json_file, 'r') as f:
            session = json.load(f)
            sessions.append(session)
            print(f"   ✅ {session['session_id']}: {len(session['hands'])} manos")

    if not sessions:
        print("❌ No se encontraron sesiones de datos.")
        return None

    # Combinar todas las manos
    all_hands = []
    for session in sessions:
        all_hands.extend(session['hands'])

    print(f"\n📊 Total manos cargadas: {len(all_hands):,}")
    return all_hands


def prepare_real_dataset(hands, env):
    """Preparar dataset para entrenamiento."""
    print("🔄 Preparando dataset...")

    X_states = []
    y_actions = []
    y_rewards = []

    for hand in hands:
        state = np.array(hand['state'])
        action = hand['action']
        reward = hand['reward']

        X_states.append(state)
        y_actions.append(action)
        y_rewards.append(reward)

    X_states = np.array(X_states)
    y_actions = np.array(y_actions)
    y_rewards = np.array(y_rewards)

    print(f"   States shape: {X_states.shape}")
    print(f"   Actions shape: {y_actions.shape}")
    print(f"   Rewards shape: {y_rewards.shape}")

    return X_states, y_actions, y_rewards


def fine_tune_model(model, X_states, y_actions, y_rewards, epochs=50, batch_size=64, learning_rate=1e-5):
    """Fine-tuning del modelo con datos reales."""
    print("\n" + "="*70)
    print("🎯 FINE-TUNING CON DATOS REALES")
    print("="*70)

    device = next(model.parameters()).device
    print(f"🎮 Device: {device}")

    # Optimizador con learning rate muy bajo
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Convertir a tensores
    states_tensor = torch.FloatTensor(X_states).to(device)
    actions_tensor = torch.LongTensor(y_actions).to(device)
    rewards_tensor = torch.FloatTensor(y_rewards).to(device)

    # Dividir en train/validation
    n_train = int(0.8 * len(X_states))
    X_train, X_val = states_tensor[:n_train], states_tensor[n_train:]
    y_train_act, y_val_act = actions_tensor[:n_train], actions_tensor[n_train:]
    y_train rew, y_val_rew = rewards_tensor[:n_train], rewards_tensor[n_train:]

    print(f"\n📊 Dataset:")
    print(f"   Train: {len(X_train):,} manos")
    print(f"   Val: {len(X_val):,} manos")

    # Entrenamiento
    print(f"\n🚀 Entrenando por {epochs} epochs...")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Batch size: {batch_size}")

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_states = X_train[i:i+batch_size]
            batch_actions = y_train_act[i:i+batch_size]
            batch_rewards = y_train_rew[i:i+batch_size]

            # Forward pass
            optimizer.zero_grad()

            # Obtener Q-values
            q_values = model.network(batch_states)

            # Usar las acciones tomadas para obtener Q-values predichos
            predicted_q = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze()

            # Target es el reward real
            target_q = batch_rewards

            # Loss
            loss = criterion(predicted_q, target_q)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_q_values = model.network(X_val)
            val_predicted_q = val_q_values.gather(1, y_val_act.unsqueeze(1)).squeeze()
            val_loss = criterion(val_predicted_q, y_val_rew).item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Guardar mejor modelo
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss/len(X_train):.4f}, Val Loss: {val_loss:.4f}")

        if patience_counter >= patience:
            print(f"\n✅ Early stopping en epoch {epoch+1}")
            break

    # Cargar mejor modelo
    model.load_state_dict(best_model_state)

    print(f"\n✅ Fine-tuning completado!")
    print(f"   Mejor Val Loss: {best_val_loss:.4f}")

    return model


def main():
    """Función principal."""
    print("="*70)
    print("🎰 FINE-TUNING DE MODELO CON DATOS REALES DE CASINO")
    print("="*70)

    # Configuración
    BASE_MODEL = 'models/checkpoints/latest.pt'  # Modelo de 10M
    REAL_DATA_DIR = 'casino_data'
    OUTPUT_MODEL = 'models/checkpoints/fine_tuned_real.pt'
    EPOCHS = 50
    LEARNING_RATE = 1e-5

    # 1. Cargar modelo base
    print(f"\n📦 Cargando modelo base: {BASE_MODEL}")
    if not Path(BASE_MODEL).exists():
        print(f"❌ Modelo no encontrado: {BASE_MODEL}")
        print("   Primero entrena el modelo de 10M episodios.")
        return

    checkpoint = torch.load(BASE_MODEL, map_location='cpu')

    # Crear environment
    env = BlackjackEnv(
        num_decks=6,
        penetration=0.75,
        allow_surrender=True,
        dealer_hits_soft_17=True
    )

    # Crear modelo
    model = DQN(
        state_dim=9,
        action_dim=6,
        hidden_dims=[1024, 512, 256],
        network_type='standard'
    )

    # Cargar pesos
    if 'policy_network_state_dict' in checkpoint:
        model.network.load_state_dict(checkpoint['policy_network_state_dict'])
    else:
        model.network.load_state_dict(checkpoint)

    model.network.eval()

    # Mover a GPU si disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.network = model.network.to(device)

    print(f"✅ Modelo cargado en {device}")

    # 2. Cargar datos reales
    hands = load_casino_sessions(REAL_DATA_DIR)

    if hands is None or len(hands) == 0:
        print("\n⚠️  No hay datos reales disponibles.")
        print("   Primero recopila datos usando: python collect_real_data.py")
        return

    # 3. Preparar dataset
    X_states, y_actions, y_rewards = prepare_real_dataset(hands, env)

    # 4. Fine-tuning
    fine_tune_model(
        model,
        X_states,
        y_actions,
        y_rewards,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

    # 5. Guardar modelo fine-tuned
    Path(OUTPUT_MODEL).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.network.state_dict(), OUTPUT_MODEL)

    print(f"\n✅ Modelo fine-tuned guardado en: {OUTPUT_MODEL}")
    print(f"   Para usarlo:")
    print(f"   python evaluate_strategies.py --model {OUTPUT_MODEL}")

    # 6. Comparar con modelo base
    print(f"\n📊 Para comparar rendimiento:")
    print(f"   python evaluate_strategies.py --model {BASE_MODEL}")
    print(f"   python evaluate_strategies.py --model {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
