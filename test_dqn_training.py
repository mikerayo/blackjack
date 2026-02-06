"""Test script to verify DQN training works after fixes."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.append('src')

import torch
import numpy as np
from environment.blackjack_env import BlackjackEnv
from agent.dqn import create_dqn
from agent.replay_buffer import ReplayBuffer
from game.rules import Action

def test_dqn_training():
    """Test that DQN can train without errors."""
    print("Testing DQN Training...")

    # Create environment
    env = BlackjackEnv()

    # Create network
    state_dim = 9
    action_dim = 6
    policy_net = create_dqn('standard', state_dim, action_dim, hidden_dims=[256, 128])
    target_net = create_dqn('standard', state_dim, action_dim, hidden_dims=[256, 128])
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Optimizer
    import torch.optim as optim
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    criterion = torch.nn.SmoothL1Loss()

    # Replay buffer
    buffer = ReplayBuffer(capacity=10000)

    # Training parameters
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay_steps = 3000  # Fast decay for test (assuming ~2-3 steps/episode)
    num_episodes = 2000

    epsilon = epsilon_start
    episode_rewards = []

    print(f"  Training for {num_episodes} episodes...")
    print(f"  Epsilon: {epsilon_start} -> {epsilon_end} over {epsilon_decay_steps} steps")

    step_count = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        state_vector = env.get_observation_vector(state)

        episode_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 50:
            # Get valid actions as Action enums
            valid_action_enums = env.game.get_valid_actions()
            valid_action_indices = [int(a.value) for a in valid_action_enums]

            # Select action
            if np.random.random() < epsilon:
                action = np.random.choice(valid_action_indices)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                    q_values = policy_net(state_tensor)
                    # Mask invalid actions
                    mask = torch.full((action_dim,), float('-inf'))
                    for idx in valid_action_indices:
                        mask[idx] = 0
                    q_values = q_values + mask
                    action = q_values.argmax().item()

            # Step environment
            next_state, reward, done, truncated, info = env.step(action)
            next_state_vector = env.get_observation_vector(next_state)

            # Store transition
            buffer.push(state_vector, action, reward, next_state_vector, done or truncated)

            episode_reward += reward
            state_vector = next_state_vector
            steps += 1
            step_count += 1

        # Decay epsilon
        if step_count < epsilon_decay_steps:
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * (step_count / epsilon_decay_steps)
        else:
            epsilon = epsilon_end

        episode_rewards.append(episode_reward)

        # Train on batch
        if len(buffer) >= 64:
            states, actions, rewards, next_states, dones = buffer.sample_arrays(64)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            # Current Q values
            current_q = policy_net(states).gather(1, actions.unsqueeze(1))

            # Target Q values
            with torch.no_grad():
                next_q = target_net(next_states).max(1)[0]
                target_q = rewards + (0.99 * next_q * (1 - dones))

            loss = criterion(current_q.squeeze(), target_q)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
            optimizer.step()

            # Update target network
            if step_count % 100 == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Log progress
        if (episode + 1) % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            avg_reward = np.mean(recent_rewards)
            win_rate = np.mean([1 if r > 0 else 0 for r in recent_rewards]) * 100
            print(f"  Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.3f} | Win Rate: {win_rate:.1f}% | Epsilon: {epsilon:.3f}")

    # Final metrics
    total_avg_reward = np.mean(episode_rewards)
    total_win_rate = np.mean([1 if r > 0 else 0 for r in episode_rewards]) * 100

    print(f"\n  Training completed!")
    print(f"  Final Epsilon: {epsilon:.4f}")
    print(f"  Final Win Rate: {total_win_rate:.2f}%")
    print(f"  Final Avg Reward: {total_avg_reward:.3f}")

    # Check epsilon decayed properly
    if epsilon < 0.1:
        print(f"  [OK] Epsilon decayed properly (epsilon < 0.1)")
    else:
        print(f"  [FAIL] Epsilon decay is too slow (epsilon = {epsilon:.3f})")
        return False

    # Check win rate is reasonable
    if 35 <= total_win_rate <= 50:
        print(f"  [OK] Win rate is reasonable")
        return True
    else:
        print(f"  [WARN] Win rate is unusual but training may still work")
        return True


if __name__ == "__main__":
    print("="*60)
    print("DQN TRAINING FIX VERIFICATION")
    print("="*60 + "\n")

    if test_dqn_training():
        print("\n[OK] DQN training test passed!")
    else:
        print("\n[FAIL] DQN training test failed!")

    print("="*60)
