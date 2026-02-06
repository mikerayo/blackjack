"""Gymnasium environment for Blackjack RL training."""

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle
import numpy as np
from typing import Tuple, Dict, Any, Optional

from game.blackjack import BlackjackGame, GameState, GameResult
from game.rules import Action


class BlackjackEnv(gym.Env, EzPickle):
    """
    Blackjack environment compatible with Gymnasium.

    Observation Space:
        - player_value: 4-31 (discrete)
        - dealer_up_card: 1-10 (discrete)
        - is_soft: 0 or 1 (boolean)
        - true_count: -20 to 20 (continuous)
        - cards_remaining_ratio: 0-1 (continuous)
        - can_split: 0 or 1 (boolean)
        - can_double: 0 or 1 (boolean)

    Action Space:
        - Discrete(6): [HIT, STAND, DOUBLE, SPLIT, INSURANCE, SURRENDER]
    """

    metadata = {'render_modes': ['human']}

    def __init__(self,
                 num_decks: int = 6,
                 penetration: float = 0.75,
                 allow_surrender: bool = True,
                 dealer_hits_soft_17: bool = True,
                 render_mode: Optional[str] = None):
        """
        Initialize the Blackjack environment.

        Args:
            num_decks: Number of decks in the shoe
            penetration: Shoe penetration (0-1)
            allow_surrender: Whether surrender is allowed
            dealer_hits_soft_17: Whether dealer hits on soft 17
            render_mode: Rendering mode ('human' or None)
        """
        EzPickle.__init__(self, num_decks, penetration, allow_surrender,
                         dealer_hits_soft_17, render_mode)

        self.num_decks = num_decks
        self.penetration = penetration
        self.allow_surrender = allow_surrender
        self.dealer_hits_soft_17 = dealer_hits_soft_17
        self.render_mode = render_mode

        # Initialize game
        self.game = BlackjackGame(
            num_decks=num_decks,
            penetration=penetration,
            allow_surrender=allow_surrender,
            dealer_hits_soft_17=dealer_hits_soft_17
        )

        # Define action space (6 actions)
        # 0: HIT, 1: STAND, 2: DOUBLE, 3: SPLIT, 4: INSURANCE, 5: SURRENDER
        self.action_space = spaces.Discrete(6)

        # Define observation space
        # We use a Dict space for structured observations
        self.observation_space = spaces.Dict({
            'player_value': spaces.Box(low=4, high=31, shape=(1,), dtype=np.int32),
            'dealer_up_card': spaces.Box(low=1, high=10, shape=(1,), dtype=np.int32),
            'is_soft': spaces.Discrete(2),
            'true_count': spaces.Box(low=-20, high=20, shape=(1,), dtype=np.float32),
            'cards_remaining_ratio': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'can_split': spaces.Discrete(2),
            'can_double': spaces.Discrete(2),
            'can_surrender': spaces.Discrete(2),
            'can_insure': spaces.Discrete(2),
        })

        # For training with MLP, we also provide a flattened observation space
        self.flat_observation_dim = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1  # 9 features

        self.state: Optional[GameState] = None
        self.game_result: Optional[GameResult] = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment to start a new episode.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset the game
        self.state = self.game.reset()
        self.game_result = None

        observation = self._get_observation()
        info = {}

        if self.render_mode == 'human':
            self._render()

        return observation, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: The action to take (0-5)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert integer action to Action enum
        try:
            action_enum = Action(action)
        except ValueError:
            # Invalid action, use HIT as default
            action_enum = Action.HIT

        # Check if action is valid
        valid_actions = self.game.get_valid_actions()
        if action_enum not in valid_actions:
            # Invalid action, replace with HIT
            action_enum = Action.HIT

        # Execute action
        state, reward, done, info = self.game.step(action_enum)

        self.state = state

        if done and 'game_result' in info:
            self.game_result = info['game_result']

        observation = self._get_observation()

        if self.render_mode == 'human':
            self._render()

        return observation, reward, done, False, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Convert GameState to observation dict.

        Returns:
            Dictionary of observation features
        """
        state = self.state
        assert state is not None, "State not initialized. Call reset() first."

        player_value = state.player_value
        dealer_up_card = state.dealer_up_card.value if state.dealer_up_card else 1
        is_soft = 1 if state.is_soft else 0
        true_count = state.true_count
        cards_remaining_ratio = self.game.deck.cards_remaining() / self.game.deck.total_cards

        valid_actions = self.game.get_valid_actions()
        can_split = 1 if Action.SPLIT in valid_actions else 0
        can_double = 1 if Action.DOUBLE in valid_actions else 0
        can_surrender = 1 if Action.SURRENDER in valid_actions else 0
        can_insure = 1 if Action.INSURANCE in valid_actions else 0

        return {
            'player_value': np.array([player_value], dtype=np.int32),
            'dealer_up_card': np.array([dealer_up_card], dtype=np.int32),
            'is_soft': np.array(is_soft, dtype=np.int32),
            'true_count': np.array([true_count], dtype=np.float32),
            'cards_remaining_ratio': np.array([cards_remaining_ratio], dtype=np.float32),
            'can_split': np.array(can_split, dtype=np.int32),
            'can_double': np.array(can_double, dtype=np.int32),
            'can_surrender': np.array(can_surrender, dtype=np.int32),
            'can_insure': np.array(can_insure, dtype=np.int32),
        }

    def get_observation_vector(self, observation: Dict) -> np.ndarray:
        """
        Convert observation dict to flat vector for neural network input.

        Args:
            observation: Observation dictionary

        Returns:
            Flat numpy array of features
        """
        return np.array([
            observation['player_value'][0] / 31.0,  # Normalize to [0, 1]
            observation['dealer_up_card'][0] / 10.0,  # Normalize to [0, 1]
            float(observation['is_soft']),
            np.clip(observation['true_count'][0] / 20.0, -1, 1),  # Normalize to [-1, 1]
            observation['cards_remaining_ratio'][0],
            float(observation['can_split']),
            float(observation['can_double']),
            float(observation['can_surrender']),
            float(observation['can_insure']),
        ], dtype=np.float32)

    def _render(self) -> None:
        """Render the current state (human-readable)."""
        if self.state is None:
            return

        state = self.state

        print("\n" + "="*50)
        print("BLACKJACK GAME STATE")
        print("="*50)

        print(f"\nDealer's Hand:")
        if len(state.dealer_hand) > 0:
            print(f"  Up Card: {state.dealer_hand[0]}")
            if self.game_over():
                print(f"  Full Hand: {' '.join(str(c) for c in state.dealer_hand)}")
                from game.rules import calculate_hand_value
                dealer_value, _ = calculate_hand_value(state.dealer_hand)
                print(f"  Value: {dealer_value}")

        print(f"\nPlayer's Hand:")
        print(f"  {' '.join(str(c) for c in state.player_hand)}")
        print(f"  Value: {state.player_value} {'(Soft)' if state.is_soft else ''}")

        print(f"\nGame Info:")
        print(f"  True Count: {state.true_count:.2f}")
        print(f"  Cards Remaining: {self.game.deck.cards_remaining()}/{self.game.deck.total_cards}")
        print(f"  Current Bet: ${state.bet_amount:.2f}")

        if self.game_result:
            print(f"\nResult: {self.game_result.result.name}")
            print(f"Payout: ${self.game_result.payout:.2f}")
            print(f"Actions: {[a.name for a in self.game_result.actions_taken]}")

        print("="*50 + "\n")

    def game_over(self) -> bool:
        """Check if the current game is over."""
        return self.game.game_over if hasattr(self.game, 'game_over') else False

    def get_basic_strategy_action(self) -> int:
        """
        Get the action according to basic strategy.

        Returns:
            Integer action (0-5)
        """
        action = self.game.get_basic_strategy_action()
        return int(action)

    def close(self) -> None:
        """Clean up any resources."""
        pass
