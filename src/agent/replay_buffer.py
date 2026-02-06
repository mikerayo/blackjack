"""Experience Replay Buffer for DQN training."""

import random
from typing import List, Tuple
import numpy as np

from collections import deque


class Transition:
    """A single transition in the environment."""

    def __init__(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def __repr__(self) -> str:
        return f"Transition(action={self.action}, reward={self.reward:.2f}, done={self.done})"


class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions.

    Implements a circular buffer with fixed capacity.
    """

    def __init__(self, capacity: int = 100000):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.position = 0

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """
        Add a transition to the buffer.

        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether the episode ended
        """
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            List of sampled transitions

        Raises:
            ValueError: If buffer has fewer samples than batch_size
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has only {len(self.buffer)} samples, "
                           f"but {batch_size} requested")

        return random.sample(self.buffer, batch_size)

    def sample_arrays(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch and return as separate arrays for efficient training.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays
        """
        transitions = self.sample(batch_size)

        states = np.array([t.state for t in transitions], dtype=np.float32)
        actions = np.array([t.action for t in transitions], dtype=np.int64)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        next_states = np.array([t.next_state for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Get the current size of the buffer."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough samples for training.

        Args:
            batch_size: Required batch size

        Returns:
            True if buffer has at least batch_size samples
        """
        return len(self.buffer) >= batch_size

    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self.buffer.clear()

    def get_stats(self) -> dict:
        """
        Get buffer statistics.

        Returns:
            Dictionary with buffer stats
        """
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'usage': 0.0,
                'mean_reward': 0.0,
                'std_reward': 0.0
            }

        rewards = [t.reward for t in self.buffer]

        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'usage': len(self.buffer) / self.capacity,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
        }
