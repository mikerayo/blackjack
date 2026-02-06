"""Deck and Card management for Blackjack with card counting support."""

from enum import IntEnum
from typing import List
import random


class Suit(IntEnum):
    """Card suits."""
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Rank(IntEnum):
    """Card ranks."""
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13


class Card:
    """Represents a playing card."""

    def __init__(self, suit: Suit, rank: Rank):
        self.suit = suit
        self.rank = rank

    @property
    def value(self) -> int:
        """Get the blackjack value of the card."""
        if self.rank in [Rank.JACK, Rank.QUEEN, Rank.KING]:
            return 10
        if self.rank == Rank.ACE:
            return 11  # Can be 1 or 11, handled in hand calculation
        return int(self.rank)

    @property
    def count_value(self) -> int:
        """
        Get the Hi-Lo card counting value.
        2-6: +1
        7-9: 0
        10-A: -1
        """
        if self.rank in [Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX]:
            return 1
        elif self.rank in [Rank.SEVEN, Rank.EIGHT, Rank.NINE]:
            return 0
        else:  # 10, J, Q, K, A
            return -1

    def __repr__(self) -> str:
        return f"Card({self.suit.name}, {self.rank.name})"

    def __str__(self) -> str:
        rank_str = self.rank.name
        if self.rank == Rank.ACE:
            rank_str = "A"
        elif self.rank == Rank.JACK:
            rank_str = "J"
        elif self.rank == Rank.QUEEN:
            rank_str = "Q"
        elif self.rank == Rank.KING:
            rank_str = "K"

        # Use ASCII characters for compatibility
        suit_symbols = {Suit.CLUBS: "C", Suit.DIAMONDS: "D", Suit.HEARTS: "H", Suit.SPADES: "S"}
        return f"{rank_str}{suit_symbols[self.suit]}"


class Deck:
    """Represents a deck of cards with card counting support."""

    def __init__(self, num_decks: int = 6, penetration: float = 0.75):
        """
        Initialize a deck.

        Args:
            num_decks: Number of decks in the shoe (default: 6)
            penetration: Shoe penetration percentage (0-1). Cards are reshuffled
                        when this percentage of cards have been dealt.
        """
        self.num_decks = num_decks
        self.penetration = penetration
        self.cards: List[Card] = []
        self.running_count = 0
        self.cards_dealt = 0
        self.total_cards = num_decks * 52
        self.reset()

    def reset(self) -> None:
        """Reset and shuffle the deck."""
        self.cards = []
        for _ in range(self.num_decks):
            for suit in Suit:
                for rank in Rank:
                    self.cards.append(Card(suit, rank))

        random.shuffle(self.cards)
        self.running_count = 0
        self.cards_dealt = 0

    def draw(self) -> Card:
        """
        Draw a card from the deck.

        Returns:
            Card: The drawn card

        Raises:
            ValueError: If the deck is empty
        """
        if not self.cards:
            raise ValueError("Cannot draw from an empty deck")

        card = self.cards.pop()
        self.running_count += card.count_value
        self.cards_dealt += 1

        # Auto-reshuffle if penetration reached
        if self.cards_dealt >= self.total_cards * (1 - self.penetration):
            self.reset()

        return card

    def draw_multiple(self, num_cards: int) -> List[Card]:
        """Draw multiple cards at once."""
        return [self.draw() for _ in range(num_cards)]

    def get_true_count(self) -> float:
        """
        Calculate the true count (running count divided by remaining decks).
        This is more accurate than running count for betting decisions.
        """
        remaining_decks = len(self.cards) / 52
        if remaining_decks == 0:
            return 0.0
        return self.running_count / remaining_decks

    def cards_remaining(self) -> int:
        """Get the number of cards remaining in the shoe."""
        return len(self.cards)

    def should_reshuffle(self) -> bool:
        """Check if the deck should be reshuffled based on penetration."""
        return self.cards_dealt >= self.total_cards * (1 - self.penetration)
