"""Main Blackjack game engine."""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .deck import Deck, Card
from .rules import (
    Action, HandResult, calculate_hand_value, is_blackjack, is_busted,
    can_split, can_double_down, can_surrender, can_insure,
    get_valid_actions, compare_hands, get_dealer_action, get_payout_multiplier
)


@dataclass
class GameState:
    """Represents the current state of the game."""
    player_hand: List[Card]
    dealer_hand: List[Card]
    player_value: int
    is_soft: bool
    dealer_up_card: Card
    true_count: float
    is_split_hand: bool = False
    bet_amount: float = 1.0


@dataclass
class GameResult:
    """Represents the result of a completed game."""
    result: HandResult
    payout: float
    player_hand: List[Card]
    dealer_hand: List[Card]
    player_value: int
    dealer_value: int
    actions_taken: List[Action]


class BlackjackGame:
    """Main Blackjack game engine with all standard rules."""

    def __init__(self,
                 num_decks: int = 6,
                 penetration: float = 0.75,
                 allow_surrender: bool = True,
                 dealer_hits_soft_17: bool = True):
        """
        Initialize the blackjack game.

        Args:
            num_decks: Number of decks in the shoe
            penetration: Shoe penetration (0-1)
            allow_surrender: Whether surrender is allowed
            dealer_hits_soft_17: Whether dealer hits on soft 17
        """
        self.deck = Deck(num_decks=num_decks, penetration=penetration)
        self.allow_surrender = allow_surrender
        self.dealer_hits_soft_17 = dealer_hits_soft_17

        self.player_hand: List[Card] = []
        self.dealer_hand: List[Card] = []
        self.actions_taken: List[Action] = []
        self.current_bet: float = 1.0
        self.insurance_bet: float = 0.0
        self.is_split_hand = False
        self.game_over = False

    def reset(self) -> GameState:
        """
        Start a new game round.

        Returns:
            Initial GameState
        """
        self.player_hand = []
        self.dealer_hand = []
        self.actions_taken = []
        self.current_bet = 1.0
        self.insurance_bet = 0.0
        self.is_split_hand = False
        self.game_over = False

        # Deal initial cards
        self.player_hand = self.deck.draw_multiple(2)
        self.dealer_hand = self.deck.draw_multiple(2)

        return self.get_state()

    def get_state(self) -> GameState:
        """
        Get the current game state.

        Returns:
            Current GameState
        """
        player_value, is_soft = calculate_hand_value(self.player_hand)

        return GameState(
            player_hand=self.player_hand.copy(),
            dealer_hand=self.dealer_hand.copy(),
            player_value=player_value,
            is_soft=is_soft,
            dealer_up_card=self.dealer_hand[0] if self.dealer_hand else None,
            true_count=self.deck.get_true_count(),
            is_split_hand=self.is_split_hand,
            bet_amount=self.current_bet
        )

    def get_valid_actions(self) -> List[Action]:
        """
        Get valid actions for the current state.

        Returns:
            List of valid Action enums
        """
        if self.game_over:
            return []

        dealer_up_card = self.dealer_hand[0] if self.dealer_hand else None
        return get_valid_actions(
            self.player_hand,
            dealer_up_card,
            can_surrender_rule=self.allow_surrender
        )

    def step(self, action: Action) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """
        Execute an action in the game.

        Args:
            action: The action to take

        Returns:
            Tuple of (new_state, reward, done, info)
        """
        if self.game_over:
            raise ValueError("Game is over. Reset to play again.")

        self.actions_taken.append(action)

        # Handle different actions
        if action == Action.HIT:
            self._hit()
        elif action == Action.STAND:
            return self._stand()
        elif action == Action.DOUBLE:
            return self._double_down()
        elif action == Action.SPLIT:
            return self._split()
        elif action == Action.INSURANCE:
            return self._insurance()
        elif action == Action.SURRENDER:
            return self._surrender()
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check if player busted after hit
        if is_busted(self.player_hand):
            return self._end_game(HandResult.LOSE)

        # Return intermediate state
        state = self.get_state()
        return state, 0.0, False, {}

    def _hit(self) -> None:
        """Draw a card for the player."""
        card = self.deck.draw()
        self.player_hand.append(card)

    def _stand(self) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """Player stands, dealer plays."""
        # Dealer plays
        while True:
            dealer_action = get_dealer_action(self.dealer_hand)
            if dealer_action == Action.HIT:
                self.dealer_hand.append(self.deck.draw())
            else:
                break

        return self._end_game(compare_hands(self.player_hand, self.dealer_hand))

    def _double_down(self) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """Double down and receive one card."""
        if not can_double_down(self.player_hand):
            raise ValueError("Cannot double down with current hand")

        self.current_bet *= 2
        self.player_hand.append(self.deck.draw())

        # If busted, lose immediately
        if is_busted(self.player_hand):
            return self._end_game(HandResult.LOSE)

        # Otherwise stand (dealer plays)
        return self._stand()

    def _split(self) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """
        Split the hand into two separate hands.
        Note: This is a simplified implementation. In a full implementation,
        you would need to handle playing both hands.
        """
        if not can_split(self.player_hand):
            raise ValueError("Cannot split current hand")

        # For now, we'll just play the first split hand
        card = self.player_hand[1]
        self.player_hand = [self.player_hand[0]]
        self.player_hand.append(self.deck.draw())

        self.is_split_hand = True

        state = self.get_state()
        return state, 0.0, False, {"split": True}

    def _insurance(self) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """
        Place insurance bet (half of original bet).
        Pays 2:1 if dealer has blackjack.
        """
        if not can_insure(self.dealer_hand[0]):
            raise ValueError("Insurance not available")

        self.insurance_bet = self.current_bet / 2

        # Check if dealer has blackjack
        dealer_has_blackjack = is_blackjack(self.dealer_hand)

        if dealer_has_blackjack:
            insurance_payout = self.insurance_bet * 2
            return self._end_game(HandResult.LOSE, insurance_payout)
        else:
            # Insurance lost, continue play
            state = self.get_state()
            return state, -self.insurance_bet, False, {"insurance_lost": True}

    def _surrender(self) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """Surrender the hand and lose half the bet."""
        if not can_surrender(self.player_hand) or not self.allow_surrender:
            raise ValueError("Cannot surrender")

        self.game_over = True

        player_value, _ = calculate_hand_value(self.player_hand)
        dealer_value, _ = calculate_hand_value(self.dealer_hand)

        game_result = GameResult(
            result=HandResult.PUSH,
            payout=-0.5,
            player_hand=self.player_hand.copy(),
            dealer_hand=self.dealer_hand.copy(),
            player_value=player_value,
            dealer_value=dealer_value,
            actions_taken=self.actions_taken.copy()
        )

        state = self.get_state()

        return state, -0.5, True, {"game_result": game_result}

    def _end_game(self, result: HandResult, insurance_payout: float = 0.0) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """
        End the game and calculate rewards.

        Args:
            result: The game result
            insurance_payout: Any insurance payout

        Returns:
            Tuple of (state, reward, done, info)
        """
        self.game_over = True

        payout = get_payout_multiplier(result) * self.current_bet
        total_payout = payout + insurance_payout

        player_value, _ = calculate_hand_value(self.player_hand)
        dealer_value, _ = calculate_hand_value(self.dealer_hand)

        game_result = GameResult(
            result=result,
            payout=total_payout,
            player_hand=self.player_hand.copy(),
            dealer_hand=self.dealer_hand.copy(),
            player_value=player_value,
            dealer_value=dealer_value,
            actions_taken=self.actions_taken.copy()
        )

        state = self.get_state()

        return state, total_payout, True, {"game_result": game_result}

    def get_basic_strategy_action(self) -> Action:
        """
        Get the action according to basic strategy.
        This is a simplified version for baseline comparison.

        Returns:
            Recommended Action
        """
        player_value, is_soft = calculate_hand_value(self.player_hand)
        dealer_value = self.dealer_hand[0].value

        valid_actions = self.get_valid_actions()

        # Simplified basic strategy
        if is_soft:
            # Soft hands
            if player_value >= 20:
                return Action.STAND
            elif player_value == 19:
                return Action.STAND
            elif player_value == 18:
                if dealer_value in [2, 7, 8]:
                    return Action.DOUBLE if Action.DOUBLE in valid_actions else Action.HIT
                elif dealer_value in [9, 10, 11]:
                    return Action.HIT
                else:
                    return Action.STAND
            elif player_value == 17:
                return Action.HIT if Action.DOUBLE in valid_actions else Action.HIT
            else:
                return Action.HIT
        else:
            # Hard hands
            if player_value >= 17:
                return Action.STAND
            elif player_value in [13, 14, 15, 16]:
                if dealer_value in [2, 3, 4, 5, 6]:
                    return Action.STAND
                else:
                    return Action.HIT
            elif player_value == 12:
                if dealer_value in [4, 5, 6]:
                    return Action.STAND
                else:
                    return Action.HIT
            elif player_value == 11:
                return Action.DOUBLE if Action.DOUBLE in valid_actions else Action.HIT
            elif player_value == 10:
                if dealer_value < 10:
                    return Action.DOUBLE if Action.DOUBLE in valid_actions else Action.HIT
                else:
                    return Action.HIT
            elif player_value == 9:
                if dealer_value in [3, 4, 5, 6]:
                    return Action.DOUBLE if Action.DOUBLE in valid_actions else Action.HIT
                else:
                    return Action.HIT
            else:
                return Action.HIT
