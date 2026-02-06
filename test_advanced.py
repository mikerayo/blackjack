"""Quick test of advanced features."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("="*70)
print("Testing Advanced ML Blackjack System")
print("="*70)

# Test 1: Import all modules
print("\n[1/5] Testing imports...")
try:
    from environment.blackjack_env import BlackjackEnv
    from strategies import (
        get_all_strategies,
        create_consensus_system,
        get_available_systems,
        create_betting_system,
        get_available_betting_systems
    )
    from agent.scalable_trainer import ScalableTrainer
    print("[OK] All imports successful")
except Exception as e:
    print(f"[X] Import failed: {e}")
    sys.exit(1)

# Test 2: Create environment
print("\n[2/5] Testing environment...")
try:
    env = BlackjackEnv(num_decks=6)
    state, _ = env.reset()
    print(f"[OK] Environment created, state shape: {state['player_value'].shape}")
except Exception as e:
    print(f"[X] Environment failed: {e}")
    sys.exit(1)

# Test 3: Test expert strategies
print("\n[3/5] Testing expert strategies...")
try:
    strategies = get_all_strategies()
    print(f"[OK] Loaded {len(strategies)} expert strategies:")
    for strategy in strategies:
        print(f"  - {strategy.name}")
except Exception as e:
    print(f"[X] Strategies failed: {e}")
    sys.exit(1)

# Test 4: Test consensus system
print("\n[4/5] Testing consensus system...")
try:
    consensus_systems = get_available_systems()
    print(f"[OK] Available consensus systems: {', '.join(consensus_systems)}")

    # Create hybrid consensus
    hybrid = create_consensus_system('hybrid', strategies)
    print(f"[OK] Created hybrid consensus system")

    # Test it on a state - use internal game state
    state, _ = env.reset()
    game_state = env.game.get_state()
    valid_actions = list(range(env.action_space.n))
    result = hybrid.get_consensus(game_state, valid_actions)

    # Handle both Action enum and int
    action_name = result.selected_action.name if hasattr(result.selected_action, 'name') else str(result.selected_action)
    print(f"[OK] Consensus result: {action_name} (confidence: {result.confidence:.2f})")
except Exception as e:
    print(f"[X] Consensus failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test betting systems
print("\n[5/5] Testing betting systems...")
try:
    betting_systems = get_available_betting_systems()
    print(f"[OK] Available betting systems: {', '.join(betting_systems)}")

    # Create Hi-Lo betting
    hilo_betting = create_betting_system('hilo', initial_bankroll=10000)
    print(f"[OK] Created Hi-Lo betting system")

    # Test it on a state - use internal game state
    state, _ = env.reset()
    game_state = env.game.get_state()
    bet_decision = hilo_betting.get_bet(game_state)
    print(f"[OK] Bet decision: ${bet_decision.bet_amount:.2f}")
    print(f"[OK] Game state: player_value={game_state.player_value}, dealer_card={game_state.dealer_up_card}, TC={game_state.true_count:.2f}")
except Exception as e:
    print(f"[X] Betting failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("ALL TESTS PASSED! [OK]")
print("="*70)
print("\nNext steps:")
print("  1. Evaluate all strategies:")
print("     python evaluate_strategies.py --episodes 10000 --type all")
print("\n  2. Start massive training:")
print("     python train_massive.py --episodes 5000000")
print("\n" + "="*70 + "\n")
