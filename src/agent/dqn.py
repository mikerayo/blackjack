"""Deep Q-Network implementation for Blackjack."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class DQN(nn.Module):
    """
    Deep Q-Network for playing Blackjack.

    Architecture:
        - Input: 9 normalized features
        - Hidden 1: 256 neurons, ReLU
        - Hidden 2: 256 neurons, ReLU
        - Hidden 3: 128 neurons, ReLU (optional)
        - Output: 6 Q-values (one per action)
    """

    def __init__(self,
                 state_dim: int = 9,
                 action_dim: int = 6,
                 hidden_dims: list = [256, 256, 128],
                 use_dropout: bool = False):
        """
        Initialize the DQN.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer sizes
            use_dropout: Whether to use dropout for regularization
        """
        super(DQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_dropout = use_dropout

        # Build network layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor of shape (batch_size, state_dim)

        Returns:
            Q-values tensor of shape (batch_size, action_dim)
        """
        return self.network(state)

    def select_action(self,
                     state: np.ndarray,
                     epsilon: float = 0.0,
                     valid_actions: Optional[list] = None) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state observation
            epsilon: Exploration rate (0 = greedy, 1 = random)
            valid_actions: List of valid action indices (if None, all actions valid)

        Returns:
            Selected action
        """
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            # Random action
            if valid_actions:
                return np.random.choice(valid_actions)
            else:
                return np.random.randint(self.action_dim)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                # Move to same device as network
                device = next(self.network.parameters()).device
                state_tensor = state_tensor.to(device)
                q_values = self.forward(state_tensor)

                if valid_actions:
                    # Mask invalid actions
                    mask = torch.ones(self.action_dim) * float('-inf')
                    device = next(self.network.parameters()).device
                    mask = mask.to(device)
                    mask[valid_actions] = 0
                    q_values = q_values + mask

                return q_values.argmax(dim=1).item()

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for a state.

        Args:
            state: Current state observation

        Returns:
            Array of Q-values for each action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            # Move to same device as network
            device = next(self.network.parameters()).device
            state_tensor = state_tensor.to(device)
            q_values = self.forward(state_tensor)
            return q_values.squeeze(0).cpu().numpy()


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture.
    Separates value and advantage streams for better learning.

    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """

    def __init__(self,
                 state_dim: int = 9,
                 action_dim: int = 6,
                 hidden_dim: int = 256):
        """
        Initialize the Dueling DQN.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer size
        """
        super(DuelingDQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dueling network.

        Args:
            state: Input state tensor

        Returns:
            Q-values tensor
        """
        features = self.feature_layer(state)

        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine value and advantage: Q(s,a) = V(s) + (A(s,a) - mean(A))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


def create_dqn(network_type: str = 'standard',
               state_dim: int = 9,
               action_dim: int = 6,
               **kwargs) -> nn.Module:
    """
    Factory function to create DQN networks.

    Args:
        network_type: Type of network ('standard' or 'dueling')
        state_dim: State space dimension
        action_dim: Action space dimension
        **kwargs: Additional arguments for the network

    Returns:
        Initialized DQN network
    """
    if network_type == 'dueling':
        return DuelingDQN(state_dim, action_dim, **kwargs)
    else:
        return DQN(state_dim, action_dim, **kwargs)
