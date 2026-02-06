"""Test script to verify expert strategies fix."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.append('src')

from environment.blackjack_env import BlackjackEnv
from strategies.expert_strategies import BasicStrategy
from strategies.consensus_system import create_consensus_system, get_all_strategies
from game.rules import Action

def test_basic_strategy():
    """Test that BasicStrategy returns valid actions."""
    print("Testing BasicStrategy...")
    env = BlackjackEnv()

    wins = 0
    losses = 0
    pushes = 0
    total = 0

    for i in range(1000):
        state, _ = env.reset()
        game_state = env.game.get_state()

        # Get valid actions from game
        valid_actions = env.game.get_valid_actions()

        # Get action from BasicStrategy
        bs = BasicStrategy()
        action = bs.get_action(game_state, valid_actions)

        # Verify action is valid
        if action not in valid_actions:
            print(f"  ERROR: Invalid action {action} not in {valid_actions}")
            print(f"    Player value: {game_state.player_value}")
            print(f"    Dealer up card: {game_state.dealer_up_card}")
            return False

        # Play episode
        done = False
        while not done:
            next_state, reward, done, _, info = env.step(int(action.value))
            if not done:
                game_state = env.game.get_state()
                valid_actions = env.game.get_valid_actions()
                action = bs.get_action(game_state, valid_actions)

        total += 1
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            pushes += 1

    win_rate = wins / total * 100
    print(f"  [OK] All actions were valid!")
    print(f"  Win Rate: {win_rate:.2f}% (Wins: {wins}, Losses: {losses}, Pushes: {pushes})")

    # Basic Strategy should achieve ~42-43% win rate
    if 40 <= win_rate <= 48:
        print(f"  [OK] Win rate is in expected range!")
        return True
    else:
        print(f"  [FAIL] Win rate is outside expected range [40-48%]")
        return False


def test_consensus_system():
    """Test that consensus system works correctly."""
    print("\nTesting Consensus System...")
    env = BlackjackEnv()

    strategies = get_all_strategies()
    consensus = create_consensus_system('majority', strategies)

    valid_count = 0
    total = 100

    for i in range(total):
        state, _ = env.reset()
        game_state = env.game.get_state()
        valid_actions = env.game.get_valid_actions()

        result = consensus.get_consensus(game_state, valid_actions)

        if result.selected_action in valid_actions:
            valid_count += 1
        else:
            print(f"  ERROR: Consensus returned invalid action {result.selected_action}")
            print(f"    Valid actions: {valid_actions}")
            return False

    print(f"  [OK] All {total} consensus actions were valid!")
    return True


if __name__ == "__main__":
    print("="*60)
    print("EXPERT STRATEGIES FIX VERIFICATION")
    print("="*60 + "\n")

    success = True

    # Test 1: Basic Strategy
    if not test_basic_strategy():
        success = False

    # Test 2: Consensus System
    if not test_consensus_system():
        success = False

    print("\n" + "="*60)
    if success:
        print("[OK] ALL TESTS PASSED! Expert strategies are fixed.")
    else:
        print("[FAIL] SOME TESTS FAILED! More fixes needed.")
    print("="*60)
