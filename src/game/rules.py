"""Blackjack rules and hand evaluation logic."""

from typing import List, Tuple
from enum import IntEnum

from .deck import Card, Rank


class Action(IntEnum):
    """Possible actions in blackjack."""
    HIT = 0
    STAND = 1
    DOUBLE = 2
    SPLIT = 3
    INSURANCE = 4
    SURRENDER = 5


class HandResult(IntEnum):
    """Possible outcomes of a hand."""
    WIN = 1
    PUSH = 0
    LOSE = -1
    BLACKJACK = 2


def calculate_hand_value(cards: List[Card]) -> Tuple[int, bool]:
    """
    Calculate the value of a blackjack hand.

    Args:
        cards: List of cards in the hand

    Returns:
        Tuple of (hand_value, is_soft)
        - hand_value: Best possible value (<=21 if possible)
        - is_soft: True if Ace is being counted as 11
    """
    total = 0
    aces = 0

    for card in cards:
        if card.rank == Rank.ACE:
            aces += 1
            total += 11
        else:
            total += card.value

    # Adjust for aces if busted
    is_soft = aces > 0 and total <= 21
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
        is_soft = aces > 0 and total <= 21

    return total, is_soft


def is_blackjack(cards: List[Card]) -> bool:
    """Check if the hand is a natural blackjack (Ace + 10-value card)."""
    if len(cards) != 2:
        return False

    values = [c.value for c in cards]
    return sorted(values) == [11, 10]  # Ace (11) + 10-value card


def is_busted(cards: List[Card]) -> bool:
    """Check if the hand is busted (over 21)."""
    total, _ = calculate_hand_value(cards)
    return total > 21


def can_split(cards: List[Card]) -> bool:
    """Check if the hand can be split (two cards of same rank)."""
    if len(cards) != 2:
        return False
    return cards[0].rank == cards[1].rank


def can_double_down(cards: List[Card]) -> bool:
    """Check if the player can double down (typically allowed on any 2 cards)."""
    return len(cards) == 2


def can_surrender(cards: List[Card]) -> bool:
    """Check if surrender is allowed (typically only on initial 2 cards)."""
    return len(cards) == 2


def can_insure(dealer_card: Card) -> bool:
    """Check if insurance is offered (dealer shows Ace)."""
    return dealer_card.rank == Rank.ACE


def get_valid_actions(player_cards: List[Card], dealer_card: Card,
                     can_surrender_rule: bool = True) -> List[Action]:
    """
    Get list of valid actions for the current state.

    Args:
        player_cards: Player's current hand
        dealer_card: Dealer's up card
        can_surrender_rule: Whether surrender is allowed by casino rules

    Returns:
        List of valid Action enums
    """
    actions = [Action.HIT, Action.STAND]

    if can_double_down(player_cards):
        actions.append(Action.DOUBLE)

    if can_split(player_cards):
        actions.append(Action.SPLIT)

    if can_insure(dealer_card) and len(player_cards) == 2:
        actions.append(Action.INSURANCE)

    if can_surrender_rule and can_surrender(player_cards):
        actions.append(Action.SURRENDER)

    return actions


def compare_hands(player_cards: List[Card], dealer_cards: List[Card]) -> HandResult:
    """
    Compare player and dealer hands to determine winner.

    Args:
        player_cards: Player's final hand
        dealer_cards: Dealer's final hand

    Returns:
        HandResult indicating outcome
    """
    player_blackjack = is_blackjack(player_cards)
    dealer_blackjack = is_blackjack(dealer_cards)

    # Both have blackjack - push
    if player_blackjack and dealer_blackjack:
        return HandResult.PUSH

    # Only player has blackjack - win
    if player_blackjack:
        return HandResult.BLACKJACK

    # Only dealer has blackjack - lose
    if dealer_blackjack:
        return HandResult.LOSE

    player_value, _ = calculate_hand_value(player_cards)
    dealer_value, _ = calculate_hand_value(dealer_cards)

    # Player busted
    if player_value > 21:
        return HandResult.LOSE

    # Dealer busted
    if dealer_value > 21:
        return HandResult.WIN

    # Compare values
    if player_value > dealer_value:
        return HandResult.WIN
    elif player_value < dealer_value:
        return HandResult.LOSE
    else:
        return HandResult.PUSH


def get_dealer_action(dealer_cards: List[Card]) -> Action:
    """
    Determine dealer's action based on standard casino rules.
    Dealer hits on soft 17, stands on hard 17 or higher.

    Args:
        dealer_cards: Dealer's current hand

    Returns:
        Action.HIT or Action.STAND
    """
    value, is_soft = calculate_hand_value(dealer_cards)

    # Hit on soft 17 or less than 17
    if (is_soft and value == 17) or value < 17:
        return Action.HIT
    return Action.STAND


def get_payout_multiplier(result: HandResult, has_blackjack: bool = False) -> float:
    """
    Get the payout multiplier for a given result.

    Args:
        result: The hand result
        has_blackjack: Whether the player had blackjack

    Returns:
        Multiplier for the bet amount
    """
    if result == HandResult.BLACKJACK:
        return 1.5  # 3:2 payout
    elif result == HandResult.WIN:
        return 1.0
    elif result == HandResult.PUSH:
        return 0.0
    else:  # LOSE
        return -1.0
