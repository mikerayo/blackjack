"""Bankroll Management and Variable Betting Systems.

Implements professional betting strategies based on deck composition
and bankroll management principles.
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass

from game.blackjack import GameState


@dataclass
class BettingDecision:
    """Result of betting system decision."""
    bet_amount: float
    reasoning: str
    confidence: float
    true_count: float


class BankrollManager:
    """
    Base class for bankroll management systems.
    """

    def __init__(self, initial_bankroll: float = 10000.0,
                 min_bet: float = 10.0,
                 max_bet: float = 500.0):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.total_wagered = 0.0
        self.total_profit = 0.0
        self.bets_count = 0

    def reset(self):
        """Reset bankroll to initial value."""
        self.current_bankroll = self.initial_bankroll
        self.total_wagered = 0.0
        self.total_profit = 0.0
        self.bets_count = 0

    def update_bankroll(self, profit: float):
        """Update bankroll after a hand."""
        self.current_bankroll += profit
        self.total_profit += profit
        self.bets_count += 1

    def get_bet(self, game_state: GameState) -> BettingDecision:
        """Determine bet amount for current state."""
        raise NotImplementedError


class FlatBetting(BankrollManager):
    """
    Flat betting - always bet the same amount.
    Most conservative approach.
    """

    def __init__(self, bet_amount: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.bet_amount = bet_amount

    def get_bet(self, game_state: GameState) -> BettingDecision:
        """Always return flat bet."""
        return BettingDecision(
            bet_amount=self.bet_amount,
            reasoning="Flat betting",
            confidence=1.0,
            true_count=game_state.true_count
        )


class KellyCriterion(BankrollManager):
    """
    Kelly Criterion - optimal bet sizing based on edge.
    Bet = (Edge / Odds) * Bankroll
    """

    def __init__(self, edge_per_true_count: float = 0.005,
                 safety_factor: float = 0.5,  # Use half-Kelly for safety
                 **kwargs):
        super().__init__(**kwargs)
        self.edge_per_true_count = edge_per_true_count
        self.safety_factor = safety_factor

    def estimate_edge(self, true_count: float) -> float:
        """
        Estimate player edge based on true count.

        Approximate: Edge ≈ True Count * 0.5%
        """
        return true_count * self.edge_per_true_count

    def get_bet(self, game_state: GameState) -> BettingDecision:
        """Calculate Kelly bet."""
        true_count = game_state.true_count

        # Estimate edge
        edge = self.estimate_edge(true_count)

        # Kelly formula: f* = (bp - q) / b
        # Where b = odds (1:1 for blackjack), p = win probability, q = loss probability
        # Simplified: f* = edge / odds
        if edge <= 0:
            # No edge, bet minimum
            bet = self.min_bet
            confidence = 0.0
        else:
            # Apply Kelly with safety factor
            kelly_fraction = edge * self.safety_factor
            bet = self.current_bankroll * kelly_fraction

            # Clamp to limits
            bet = max(self.min_bet, min(bet, self.max_bet))
            confidence = min(abs(edge) * 10, 1.0)

        return BettingDecision(
            bet_amount=bet,
            reasoning=f"Kelly: edge={edge:.3f}, kelly_frac={edge * self.safety_factor:.3f}",
            confidence=confidence,
            true_count=true_count
        )


class HiLoBetting(KellyCriterion):
    """
    Hi-Lo betting system - classic card counting bet spreading.
    Uses true count to determine bet size.
    """

    def __init__(self, bet_spread: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.bet_spread = bet_spread

    def get_bet(self, game_state: GameState) -> BettingDecision:
        """Calculate bet based on Hi-Lo true count."""
        true_count = game_state.true_count

        # Classic Hi-Lo betting ramp:
        # TC <= 0: 1 unit
        # TC = 1: 2 units
        # TC = 2: 4 units
        # TC = 3: 6 units
        # TC >= 4: 8-10 units

        if true_count <= 0:
            units = 1
        elif true_count == 1:
            units = 2
        elif true_count == 2:
            units = 4
        elif true_count == 3:
            units = 6
        else:  # TC >= 4
            units = min(8 + int(true_count - 4), self.bet_spread)

        bet = self.min_bet * units
        bet = min(bet, self.max_bet)  # Respect max bet

        confidence = min(abs(true_count) / 5.0, 1.0)

        return BettingDecision(
            bet_amount=bet,
            reasoning=f"Hi-Lo betting: TC={true_count:.1f}, units={units}",
            confidence=confidence,
            true_count=true_count
        )


class KOSystemBetting(BankrollManager):
    """
    KO (Knock-Out) system betting.
    Uses running count (simpler than true count).
    """

    def __init__(self, pivot_point: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.pivot_point = pivot_point

    def get_bet(self, game_state: GameState) -> BettingDecision:
        """Calculate bet based on KO running count."""
        # Approximate true count to KO key count
        true_count = game_state.true_count
        key_count = int(true_count * 2)  # Rough conversion

        if key_count < 2:
            units = 1
        elif key_count < 4:
            units = 2
        elif key_count < 6:
            units = 4
        else:
            units = min(6 + (key_count - 6) // 2, 8)

        bet = self.min_bet * units
        bet = min(bet, self.max_bet)

        confidence = min(max(key_count, 0) / 8.0, 1.0)

        return BettingDecision(
            bet_amount=bet,
            reasoning=f"KO betting: key={key_count}, units={units}",
            confidence=confidence,
            true_count=true_count
        )


class AdaptiveBetting(BankrollManager):
    """
    Adaptive betting that adjusts based on recent performance.
    """

    def __init__(self, window_size: int = 50,
                 aggressive_factor: float = 1.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.aggressive_factor = aggressive_factor
        self.recent_results: List[float] = []

    def update_bankroll(self, profit: float):
        """Update bankroll and track recent results."""
        super().update_bankroll(profit)
        self.recent_results.append(profit)

        # Keep only recent results
        if len(self.recent_results) > self.window_size:
            self.recent_results = self.recent_results[-self.window_size:]

    def get_trend(self) -> float:
        """Calculate recent trend."""
        if not self.recent_results:
            return 0.0
        return sum(self.recent_results) / len(self.recent_results)

    def get_bet(self, game_state: GameState) -> BettingDecision:
        """Calculate adaptive bet."""
        true_count = game_state.true_count
        trend = self.get_trend()

        # Start with Hi-Lo base bet
        if true_count <= 0:
            base_units = 1
        elif true_count == 1:
            base_units = 2
        elif true_count == 2:
            base_units = 4
        else:
            base_units = min(6 + int(true_count - 3), 10)

        # Adjust based on trend
        if trend > 0:
            # Winning recently, can be more aggressive
            multiplier = 1.0 + min(trend * self.aggressive_factor, 0.5)
        elif trend < 0:
            # Losing, be more conservative
            multiplier = max(1.0 + trend * self.aggressive_factor, 0.5)
        else:
            multiplier = 1.0

        units = base_units * multiplier
        bet = self.min_bet * units
        bet = max(self.min_bet, min(bet, self.max_bet))

        confidence = min(abs(true_count) / 5.0, 1.0)

        return BettingDecision(
            bet_amount=bet,
            reasoning=f"Adaptive: TC={true_count:.1f}, trend={trend:.2f}, units={units:.1f}",
            confidence=confidence,
            true_count=true_count
        )


class ConservativeBetting(BankrollManager):
    """
    Very conservative betting - minimizes risk of ruin.
    """

    def __init__(self, max_risk_per_hand: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.max_risk_per_hand = max_risk_per_hand

    def get_bet(self, game_state: GameState) -> BettingDecision:
        """Calculate conservative bet."""
        # Never risk more than X% of bankroll
        max_bet_by_bankroll = self.current_bankroll * self.max_risk_per_hand

        # Use small bet spread
        true_count = game_state.true_count
        if true_count >= 2:
            multiplier = 2
        elif true_count >= 4:
            multiplier = 3
        else:
            multiplier = 1

        bet = min(self.min_bet * multiplier, max_bet_by_bankroll, self.max_bet)

        return BettingDecision(
            bet_amount=bet,
            reasoning=f"Conservative: max_risk={self.max_risk_per_hand*100}%",
            confidence=0.9,
            true_count=true_count
        )


class AggressiveBetting(BankrollManager):
    """
    Aggressive betting - maximizes expected value.
    Higher risk, higher potential reward.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_bet(self, game_state: GameState) -> BettingDecision:
        """Calculate aggressive bet."""
        true_count = game_state.true_count

        # Very aggressive spread
        if true_count <= 0:
            units = 1
        elif true_count == 1:
            units = 3
        elif true_count == 2:
            units = 6
        elif true_count == 3:
            units = 10
        else:
            units = min(12 + (int(true_count) - 3) * 2, 20)

        bet = self.min_bet * units
        bet = min(bet, self.max_bet)

        # Cap at 20% of bankroll to avoid ruin
        bet = min(bet, self.current_bankroll * 0.2)
        bet = max(bet, self.min_bet)

        confidence = min(abs(true_count) / 4.0, 1.0)

        return BettingDecision(
            bet_amount=bet,
            reasoning=f"Aggressive: TC={true_count:.1f}, units={units}",
            confidence=confidence,
            true_count=true_count
        )


class ParlayBetting(BankrollManager):
    """
    Parlay betting - let winnings ride on winning streaks.
    """

    def __init__(self, max_parlay: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_parlay = max_parlay
        self.consecutive_wins = 0
        self.consecutive_losses = 0

    def update_bankroll(self, profit: float):
        """Update bankroll and track streaks."""
        super().update_bankroll(profit)

        if profit > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

    def get_bet(self, game_state: GameState) -> BettingDecision:
        """Calculate parlay bet."""
        true_count = game_state.true_count

        # Base bet
        if true_count <= 0:
            base_units = 1
        elif true_count <= 2:
            base_units = 2
        else:
            base_units = 4

        # Parlay multiplier based on win streak
        if self.consecutive_wins > 0:
            parlay_multiplier = min(2 ** self.consecutive_wins, 2 ** self.max_parlay)
        else:
            parlay_multiplier = 1

        # Reset after losses
        if self.consecutive_losses >= 2:
            parlay_multiplier = 1

        units = base_units * parlay_multiplier
        bet = self.min_bet * units
        bet = min(bet, self.max_bet)

        confidence = min(self.consecutive_wins / self.max_parlay, 1.0)

        return BettingDecision(
            bet_amount=bet,
            reasoning=f"Parlay: streak={self.consecutive_wins}, mult={parlay_multiplier}x",
            confidence=confidence,
            true_count=true_count
        )


# Betting system factory
def create_betting_system(system_type: str,
                          initial_bankroll: float = 10000.0,
                          min_bet: float = 10.0,
                          max_bet: float = 500.0,
                          **kwargs) -> BankrollManager:
    """Create a betting system by type."""
    systems = {
        'flat': FlatBetting,
        'kelly': KellyCriterion,
        'hilo': HiLoBetting,
        'ko': KOSystemBetting,
        'adaptive': AdaptiveBetting,
        'conservative': ConservativeBetting,
        'aggressive': AggressiveBetting,
        'parlay': ParlayBetting,
    }

    system_class = systems.get(system_type.lower(), FlatBetting)
    return system_class(
        initial_bankroll=initial_bankroll,
        min_bet=min_bet,
        max_bet=max_bet,
        **kwargs
    )


def get_available_betting_systems() -> List[str]:
    """Get list of available betting systems."""
    return [
        'flat',
        'kelly',
        'hilo',
        'ko',
        'adaptive',
        'conservative',
        'aggressive',
        'parlay',
    ]
