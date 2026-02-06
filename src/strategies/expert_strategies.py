"""Expert Blackjack Strategies Collection.

Implements multiple professional blackjack strategies and counting systems.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from game.blackjack import BlackjackGame, GameState
from game.rules import Action, calculate_hand_value, can_split, can_double_down
from game.deck import Card, Rank


class ExpertStrategy:
    """Base class for expert strategies."""

    def __init__(self, name: str):
        self.name = name
        self.decisions_count = 0

    def get_action(self, game_state: GameState, valid_actions: List[Action]) -> Action:
        """Get the recommended action for the current state."""
        raise NotImplementedError


class BasicStrategy(ExpertStrategy):
    """
    Standard Basic Strategy - mathematically optimal play without counting.
    Based on wizardofodds.com basic strategy.
    """

    def __init__(self):
        super().__init__("Basic Strategy")

    def get_action(self, game_state: GameState, valid_actions: List[Action]) -> Action:
        """Get basic strategy action."""
        player_value = game_state.player_value
        dealer_value = game_state.dealer_up_card.value if game_state.dealer_up_card else 10
        is_soft = game_state.is_soft

        # Surrender decisions (16 vs 9, 10, A / 15 vs 10 if allowed)
        if Action.SURRENDER in valid_actions:
            if player_value == 16 and dealer_value in [9, 10, 11]:
                return Action.SURRENDER
            if player_value == 15 and dealer_value == 10:
                return Action.SURRENDER

        # Pair splitting
        if Action.SPLIT in valid_actions:
            # Get pair rank
            if len(game_state.player_hand) == 2:
                pair_rank = game_state.player_hand[0].rank

                # Always split aces and 8s
                if pair_rank in [Rank.ACE, Rank.EIGHT]:
                    return Action.SPLIT

                # Never split 10s, 4s, 5s
                if pair_rank in [Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING]:
                    return Action.STAND

                # Split 2s, 3s, 7s vs dealer 2-7
                if pair_rank in [Rank.TWO, Rank.THREE, Rank.SEVEN] and dealer_value <= 7:
                    return Action.SPLIT

                # Split 6s vs dealer 2-6
                if pair_rank == Rank.SIX and dealer_value <= 6:
                    return Action.SPLIT

                # Split 9s vs dealer 2-9 except 7
                if pair_rank == Rank.NINE and 2 <= dealer_value <= 9 and dealer_value != 7:
                    return Action.SPLIT

        # Soft totals (Ace counts as 11)
        if is_soft:
            if player_value >= 20:
                return Action.STAND
            elif player_value == 19:
                return Action.STAND  # Double only vs 6 in some variations
            elif player_value == 18:
                if dealer_value in [2, 7, 8]:
                    return Action.STAND
                elif dealer_value in [9, 10, 11]:
                    return Action.HIT
                else:  # 3-6
                    return Action.DOUBLE if Action.DOUBLE in valid_actions else Action.HIT
            elif player_value == 17:
                return Action.HIT if Action.DOUBLE in valid_actions else Action.HIT
            else:  # 13-16
                if dealer_value in [4, 5, 6]:
                    return Action.DOUBLE if Action.DOUBLE in valid_actions else Action.HIT
                else:
                    return Action.HIT

        # Hard totals
        else:
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
            else:  # 8 or less
                return Action.HIT


class HiLoCountingStrategy(ExpertStrategy):
    """
    Hi-Lo Counting Strategy with index plays.
    Adjusts basic strategy based on the true count.
    """

    def __init__(self):
        super().__init__("Hi-Lo with Index Plays")

        # Illustrious 18 - most important index plays
        # (true count threshold for deviating from basic strategy)
        self.index_plays = {
            # Insurance
            'insurance': 3,  # Take insurance at TC +3 or higher

            # Hard 16 vs dealer upcard
            '16_vs_10': 0,   # Surrender/stand at TC >= 0 vs 10
            '16_vs_9': 5,    # Surrender at TC >= 5 vs 9

            # Hard 15 vs dealer upcard
            '15_vs_10': 4,   # Surrender at TC >= 4 vs 10

            # Hard 13 vs dealer upcard
            '13_vs_2': -1,   # Hit at TC < -1, otherwise stand
            '13_vs_3': -2,

            # Hard 12 vs dealer upcard
            '12_vs_2': 3,    # Stand at TC >= 3
            '12_vs_3': 2,    # Stand at TC >= 2
            '12_vs_4': 0,    # Stand at TC >= 0
            '12_vs_5': -2,   # Stand at TC >= -2
            '12_vs_6': -1,   # Stand at TC >= -1

            # Hard 11 vs dealer upcard
            '11_vs_ace': 1,  # Double at TC >= 1 vs A

            # Hard 10 vs dealer upcard
            '10_vs_10': 4,   # Double at TC >= 4 vs 10
            '10_vs_ace': 4,

            # Hard 9 vs dealer upcard
            '9_vs_2': 1,     # Double at TC >= 1
            '9_vs_7': 3,

            # 10,10 vs dealer upcard (splitting tens)
            '10_10_vs_5': 5,  # Split at TC >= 5
            '10_10_vs_6': 5,

            # Soft 19 vs dealer upcard
            'soft_19_vs_6': 1,  # Double at TC >= 1
            'soft_19_vs_5': 1,

            # Soft 18 vs dealer upcard
            'soft_18_vs_2': 1,  # Double at TC >= 1
            'soft_18_vs_5': 1,
            'soft_18_vs_6': 1,

            # Pair splitting
            '10_10_vs_4': 4,   # Split tens at TC >= 4
            '10_10_vs_5': 5,
            '10_10_vs_6': 5,
        }

        self.basic_strategy = BasicStrategy()

    def get_action(self, game_state: GameState, valid_actions: List[Action]) -> Action:
        """Get action adjusted for true count."""
        true_count = game_state.true_count
        player_value = game_state.player_value
        dealer_value = game_state.dealer_up_card.value if game_state.dealer_up_card else 10
        is_soft = game_state.is_soft

        # Insurance decision
        if Action.INSURANCE in valid_actions:
            if true_count >= self.index_plays['insurance']:
                return Action.INSURANCE

        # Check for index deviations
        # Hard 16
        if player_value == 16 and dealer_value == 10:
            if true_count >= self.index_plays['16_vs_10']:
                return Action.STAND if Action.STAND in valid_actions else Action.SURRENDER

        if player_value == 16 and dealer_value == 9:
            if Action.SURRENDER in valid_actions and true_count >= self.index_plays['16_vs_9']:
                return Action.SURRENDER

        # Hard 15
        if player_value == 15 and dealer_value == 10:
            if Action.SURRENDER in valid_actions and true_count >= self.index_plays['15_vs_10']:
                return Action.SURRENDER

        # Hard 12 vs 2-6
        if player_value == 12 and 2 <= dealer_value <= 6:
            index_key = f'12_vs_{dealer_value}'
            if index_key in self.index_plays:
                if true_count >= self.index_plays[index_key]:
                    return Action.STAND

        # Hard 10 doubling
        if player_value == 10 and Action.DOUBLE in valid_actions:
            if dealer_value in [10, 11] and true_count >= self.index_plays['10_vs_10']:
                return Action.DOUBLE

        # Splitting 10s
        if Action.SPLIT in valid_actions:
            if len(game_state.player_hand) == 2:
                if game_state.player_hand[0].rank in [Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING]:
                    if dealer_value in [4, 5, 6]:
                        index_key = f'10_10_vs_{dealer_value}'
                        if index_key in self.index_plays and true_count >= self.index_plays[index_key]:
                            return Action.SPLIT

        # Default to basic strategy
        return self.basic_strategy.get_action(game_state, valid_actions)


class KOCountingStrategy(ExpertStrategy):
    """
    Knock-Out (KO) Counting System - Unbalanced count system.
    Easier to use than Hi-Lo.
    """

    def __init__(self):
        super().__init__("KO Counting System")

    def get_running_count_key(self, true_count: float) -> int:
        """Convert true count to KO system key point."""
        # KO system uses running count directly
        return int(true_count * 2)  # Approximate conversion

    def get_action(self, game_state: GameState, valid_actions: List[Action]) -> Action:
        """Get action based on KO count."""
        rc_key = self.get_running_count_key(game_state.true_count)
        player_value = game_state.player_value
        dealer_value = game_state.dealer_up_card.value if game_state.dealer_up_card else 10

        # Insurance at +3 or higher (KO system)
        if Action.INSURANCE in valid_actions and rc_key >= 3:
            return Action.INSURANCE

        # Stand on 16 vs 10 at +1 or higher
        if player_value == 16 and dealer_value == 10:
            if rc_key >= 1 and Action.STAND in valid_actions:
                return Action.STAND

        # Stand on 12 vs 3 at +3 or higher
        if player_value == 12 and dealer_value == 3:
            if rc_key >= 3 and Action.STAND in valid_actions:
                return Action.STAND

        # Default to basic strategy
        basic = BasicStrategy()
        return basic.get_action(game_state, valid_actions)


class AceFiveCount(ExpertStrategy):
    """
    Ace-Five Count - Simplest effective counting system.
    Only tracks Aces and Fives.
    """

    def __init__(self):
        super().__init__("Ace-Five Count")
        self.ace_count = 0
        self.five_count = 0

    def update_count(self, card: Card):
        """Update count based on card seen."""
        if card.rank == Rank.ACE:
            self.ace_count -= 1
        elif card.rank == Rank.FIVE:
            self.five_count += 1

    def get_running_total(self) -> int:
        """Get the Ace-Five running total."""
        return self.five_count + self.ace_count

    def get_action(self, game_state: GameState, valid_actions: List[Action]) -> Action:
        """Get action based on Ace-Five count."""
        # Ace-Five mostly affects betting, not playing strategy
        # But we can make some adjustments

        running_total = self.get_running_total()

        # With very positive count, be more aggressive with doubles
        if running_total >= 2:
            player_value = game_state.player_value
            dealer_value = game_state.dealer_up_card.value if game_state.dealer_up_card else 10

            # Double 10 vs 10 (normally just hit)
            if player_value == 10 and dealer_value == 10:
                if Action.DOUBLE in valid_actions:
                    return Action.DOUBLE

        # Default to basic strategy
        basic = BasicStrategy()
        return basic.get_action(game_state, valid_actions)


class WizardOfOddsStrategy(ExpertStrategy):
    """
    Wizard of Odds strategy - slight variation of basic strategy.
    More aggressive on certain doubles.
    """

    def __init__(self):
        super().__init__("Wizard of Odds Strategy")

    def get_action(self, game_state: GameState, valid_actions: List[Action]) -> Action:
        """Get Wizard of Odds recommended action."""
        player_value = game_state.player_value
        dealer_value = game_state.dealer_up_card.value if game_state.dealer_up_card else 10
        is_soft = game_state.is_soft

        # Similar to basic strategy with slight variations
        basic = BasicStrategy()

        # Variation: Hit soft 18 vs A, 9, 10 (Wizard recommends hit)
        if is_soft and player_value == 18 and dealer_value in [9, 10, 11]:
            return Action.HIT

        # Variation: Double 11 vs all (including A)
        if player_value == 11 and Action.DOUBLE in valid_actions:
            return Action.DOUBLE

        return basic.get_action(game_state, valid_actions)


class ThorpStrategy(ExpertStrategy):
    """
    Edward O. Thorp's original strategy from "Beat the Dealer".
    The foundation of card counting.
    """

    def __init__(self):
        super().__init__("Thorp's Strategy")

    def get_ten_count(self, game_state: GameState) -> int:
        """Estimate ten-count (proportion of tens remaining)."""
        # Simplified - would need actual deck tracking
        return 0

    def get_action(self, game_state: GameState, valid_actions: List[Action]) -> Action:
        """Get Thorp's recommended action."""
        # Thorp's strategy is similar to basic strategy but more conservative
        basic = BasicStrategy()

        # More conservative on certain doubles
        player_value = game_state.player_value
        dealer_value = game_state.dealer_up_card.value if game_state.dealer_up_card else 10

        # Don't double 11 vs A in single deck (Thorp's recommendation)
        if player_value == 11 and dealer_value == 11:
            return Action.HIT

        return basic.get_action(game_state, valid_actions)


class WongHalvesStrategy(ExpertStrategy):
    """
    Stanford Wong's Halves System - Advanced balanced count.
    More accurate but harder to use.
    """

    def __init__(self):
        super().__init__("Wong Halves System")

    def get_action(self, game_state: GameState, valid_actions: List[Action]) -> Action:
        """Get action based on Wong Halves count."""
        true_count = game_state.true_count  # Approximated

        # Similar to Hi-Lo but with different thresholds
        if Action.INSURANCE in valid_actions and true_count >= 2.5:
            return Action.INSURANCE

        # More aggressive deviations
        if true_count >= 3:
            player_value = game_state.player_value
            dealer_value = game_state.dealer_up_card.value if game_state.dealer_up_card else 10

            # Stand on 16 vs 10 at TC +3
            if player_value == 16 and dealer_value == 10:
                return Action.STAND

        # Default to Hi-Lo strategy
        hilo = HiLoCountingStrategy()
        return hilo.get_action(game_state, valid_actions)


class ZenCountStrategy(ExpertStrategy):
    """
    Zen Count - Advanced balanced system.
    Considered one of the most powerful systems.
    """

    def __init__(self):
        super().__init__("Zen Count System")

    def get_action(self, game_state: GameState, valid_actions: List[Action]) -> Action:
        """Get action based on Zen count."""
        true_count = game_state.true_count

        # Zen count is more sensitive
        if Action.INSURANCE in valid_actions and true_count >= 2:
            return Action.INSURANCE

        # Use basic strategy with adjustments
        basic = BasicStrategy()

        # Very aggressive at high counts
        if true_count >= 4:
            player_value = game_state.player_value
            if player_value == 16 and Action.STAND in valid_actions:
                return Action.STAND

        return basic.get_action(game_state, valid_actions)


class AggressiveStrategy(ExpertStrategy):
    """
    Aggressive strategy - doubles and splits more often.
    """

    def __init__(self):
        super().__init__("Aggressive Strategy")

    def get_action(self, game_state: GameState, valid_actions: List[Action]) -> Action:
        """Get aggressive action."""
        player_value = game_state.player_value
        dealer_value = game_state.dealer_up_card.value if game_state.dealer_up_card else 10

        # Always double if possible with 9-11
        if Action.DOUBLE in valid_actions and 9 <= player_value <= 11:
            return Action.DOUBLE

        # Always split pairs
        if Action.SPLIT in valid_actions:
            return Action.SPLIT

        # Otherwise hit more
        if player_value < 17:
            return Action.HIT

        return Action.STAND


class ConservativeStrategy(ExpertStrategy):
    """
    Conservative strategy - minimizes risk.
    """

    def __init__(self):
        super().__init__("Conservative Strategy")

    def get_action(self, game_state: GameState, valid_actions: List[Action]) -> Action:
        """Get conservative action."""
        player_value = game_state.player_value

        # Never double
        # Surrender more often
        if Action.SURRENDER in valid_actions and player_value >= 15:
            return Action.SURRENDER

        # Stand earlier
        if player_value >= 16 and Action.STAND in valid_actions:
            return Action.STAND

        # Otherwise play basic strategy
        basic = BasicStrategy()
        return basic.get_action(game_state, valid_actions)


class AdaptiveStrategy(ExpertStrategy):
    """
    Adaptive strategy that changes based on game conditions.
    """

    def __init__(self):
        super().__init__("Adaptive Strategy")

    def get_action(self, game_state: GameState, valid_actions: List[Action]) -> Action:
        """Get adaptive action based on conditions."""
        true_count = game_state.true_count
        cards_remaining = game_state.dealer_up_card  # This is wrong, need to fix

        # Use different strategies based on count
        if true_count >= 3:
            # High count - be aggressive
            aggressive = AggressiveStrategy()
            return aggressive.get_action(game_state, valid_actions)
        elif true_count <= -2:
            # Low count - be conservative
            conservative = ConservativeStrategy()
            return conservative.get_action(game_state, valid_actions)
        else:
            # Neutral - use basic strategy
            basic = BasicStrategy()
            return basic.get_action(game_state, valid_actions)


# Strategy factory
def get_all_strategies() -> List[ExpertStrategy]:
    """Get all available expert strategies."""
    return [
        BasicStrategy(),
        HiLoCountingStrategy(),
        KOCountingStrategy(),
        AceFiveCount(),
        WizardOfOddsStrategy(),
        ThorpStrategy(),
        WongHalvesStrategy(),
        ZenCountStrategy(),
        AggressiveStrategy(),
        ConservativeStrategy(),
        AdaptiveStrategy(),
    ]


def get_strategy_by_name(name: str) -> Optional[ExpertStrategy]:
    """Get a specific strategy by name."""
    strategies = get_all_strategies()
    for strategy in strategies:
        if strategy.name == name:
            return strategy
    return None
