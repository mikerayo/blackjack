"""Debug script para ver qué acciones toman los expertos."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environment.blackjack_env import BlackjackEnv
from strategies import get_all_strategies, create_consensus_system

print("="*80)
print("DEBUG: ACCIONES DE EXPERTOS")
print("="*80)

# Crear environment
env = BlackjackEnv(num_decks=6, penetration=0.75)

# Crear expertos
strategies = get_all_strategies()
consensus = create_consensus_system('hybrid', strategies)

# Probar algunas manos
test_cases = [
    {"player": 12, "dealer": 2, "desc": "12 vs 2 (should HIT)"},
    {"player": 16, "dealer": 10, "desc": "16 vs 10 (should STAND/SURRENDER)"},
    {"player": 11, "dealer": 10, "desc": "11 vs 10 (should DOUBLE)"},
    {"player": 20, "dealer": 10, "desc": "20 vs 10 (should STAND)"},
]

for i, test in enumerate(test_cases, 1):
    print(f"\nTest {i}: {test['desc']}")

    # Crear estado manual (simplificado)
    state, _ = env.reset()

    # Obtener estado actual
    game_state = env.game.get_state()

    # Mostrar estado real
    print(f"  Estado real:")
    print(f"    Player hand: {game_state.player_hand}")
    print(f"    Player value: {game_state.player_value}")
    print(f"    Dealer up card: {game_state.dealer_up_card}")
    print(f"    Valid actions: {env.game.get_valid_actions()}")

    # Obtener acción de consenso
    valid_actions = list(range(env.action_space.n))
    print(f"  Valid actions (int): {valid_actions}")

    result = consensus.get_consensus(game_state, valid_actions)

    print(f"  Consensus result:")
    print(f"    Selected action: {result.selected_action}")
    print(f"    Type: {type(result.selected_action)}")

    # Mostrar qué significa
    if hasattr(result.selected_action, 'name'):
        print(f"    Action name: {result.selected_action.name}")
    elif isinstance(result.selected_action, int):
        action_names = ['HIT', 'STAND', 'DOUBLE', 'SPLIT', 'INSURANCE', 'SURRENDER']
        if 0 <= result.selected_action < len(action_names):
            print(f"    Action name: {action_names[result.selected_action]}")

    print(f"    Confidence: {result.confidence:.2f}")
    print(f"    Votes: {result.votes}")

    # Simular un paso
    action_to_use = result.selected_action
    if hasattr(action_to_use, 'value'):
        action_to_use = int(action_to_use.value)
    elif not isinstance(action_to_use, int):
        action_to_use = int(action_to_use)

    print(f"  Action to use: {action_to_use}")

    # Ejecutar un paso
    next_state, reward, done, truncated, info = env.step(action_to_use)

    print(f"  Result:")
    print(f"    Reward: {reward}")
    print(f"    Done: {done}")

    if done and 'game_result' in info:
        gr = info['game_result']
        print(f"    Final result: {gr.result.name}")
        print(f"    Player hand: {gr.player_hand}")
        print(f"    Dealer hand: {gr.dealer_hand}")

print("\n" + "="*80)
print("DEBUG COMPLETADO")
print("="*80)
