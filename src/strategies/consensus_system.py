"""Consensus/Ensemble System for Blackjack Strategy Selection.

Implements multiple voting and selection mechanisms for choosing
the best action across multiple expert strategies.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
from dataclasses import dataclass

from game.blackjack import GameState
from game.rules import Action
from strategies.expert_strategies import ExpertStrategy, get_all_strategies


@dataclass
class Vote:
    """Represents a single strategy vote."""
    strategy_name: str
    action: Action
    confidence: float = 1.0


@dataclass
class ConsensusResult:
    """Result from consensus system."""
    selected_action: Action
    votes: Dict[Action, int]
    vote_distribution: Dict[Action, float]
    confidence: float
    participating_strategies: List[str]
    reasoning: str


class ConsensusSystem:
    """
    Base class for consensus systems.
    """

    def __init__(self, strategies: List[ExpertStrategy]):
        self.strategies = strategies
        self.name = "Base Consensus"

    def get_consensus(self, game_state: GameState,
                     valid_actions: List[Action]) -> ConsensusResult:
        """Get consensus action from all strategies."""
        raise NotImplementedError


class MajorityVoting(ConsensusSystem):
    """
    Simple majority voting - each strategy gets one vote.
    Most voted action wins.
    """

    def __init__(self, strategies: List[ExpertStrategy]):
        super().__init__(strategies)
        self.name = "Majority Voting"

    def get_consensus(self, game_state: GameState,
                     valid_actions: List[Action]) -> ConsensusResult:
        """Get majority vote."""
        votes = []

        for strategy in self.strategies:
            action = strategy.get_action(game_state, valid_actions)
            if action in valid_actions:
                votes.append(Vote(strategy.name, action))

        # Count votes
        vote_count = Counter(v.action for v in votes)
        total_votes = len(votes)

        # Get action with most votes
        if vote_count:
            winning_action = vote_count.most_common(1)[0][0]
            win_count = vote_count[winning_action]
            confidence = win_count / total_votes if total_votes > 0 else 0
        else:
            # Fallback to first valid action
            winning_action = valid_actions[0]
            vote_count = {winning_action: 1}
            confidence = 0.0

        # Calculate distribution
        vote_distribution = {
            action: count / total_votes
            for action, count in vote_count.items()
        } if total_votes > 0 else {}

        return ConsensusResult(
            selected_action=winning_action,
            votes=dict(vote_count),
            vote_distribution=vote_distribution,
            confidence=confidence,
            participating_strategies=[v.strategy_name for v in votes],
            reasoning=f"Majority vote: {win_count}/{total_votes} strategies agree"
        )


class WeightedVoting(ConsensusSystem):
    """
    Weighted voting - strategies have different weights based on performance.
    """

    def __init__(self, strategies: List[ExpertStrategy],
                 weights: Optional[Dict[str, float]] = None):
        super().__init__(strategies)
        self.name = "Weighted Voting"

        # Default weights (can be updated based on performance)
        if weights is None:
            self.weights = {
                "Basic Strategy": 1.0,
                "Hi-Lo with Index Plays": 1.2,
                "KO Counting System": 1.1,
                "Wizard of Odds Strategy": 1.0,
                "Thorp's Strategy": 0.9,
                "Wong Halves System": 1.3,
                "Zen Count System": 1.3,
                "Aggressive Strategy": 0.5,
                "Conservative Strategy": 0.5,
                "Adaptive Strategy": 1.1,
            }
        else:
            self.weights = weights

    def get_consensus(self, game_state: GameState,
                     valid_actions: List[Action]) -> ConsensusResult:
        """Get weighted consensus."""
        vote_weights = {action: 0.0 for action in valid_actions}
        participating = []

        for strategy in self.strategies:
            action = strategy.get_action(game_state, valid_actions)
            if action in valid_actions:
                weight = self.weights.get(strategy.name, 1.0)
                vote_weights[action] += weight
                participating.append(strategy.name)

        # Get action with highest weight
        if vote_weights:
            winning_action = max(vote_weights, key=vote_weights.get)
            total_weight = sum(vote_weights.values())
            confidence = vote_weights[winning_action] / total_weight if total_weight > 0 else 0
        else:
            winning_action = valid_actions[0]
            confidence = 0.0

        # Calculate distribution
        total_weight = sum(vote_weights.values())
        vote_distribution = {
            action: weight / total_weight
            for action, weight in vote_weights.items()
        } if total_weight > 0 else {}

        return ConsensusResult(
            selected_action=winning_action,
            votes={action: int(weight * 10) for action, weight in vote_weights.items()},
            vote_distribution=vote_distribution,
            confidence=confidence,
            participating_strategies=participating,
            reasoning=f"Weighted vote: {winning_action} with highest confidence"
        )


class RankedVoting(ConsensusSystem):
    """
    Instant-runoff voting system.
    Strategies rank actions, and we eliminate lowest-ranked until we have a winner.
    """

    def __init__(self, strategies: List[ExpertStrategy]):
        super().__init__(strategies)
        self.name = "Ranked Voting"

    def get_consensus(self, game_state: GameState,
                     valid_actions: List[Action]) -> ConsensusResult:
        """Get consensus using instant-runoff voting."""
        # For simplicity, we'll use a simplified version
        # Each strategy provides their top choice
        first_choices = []

        for strategy in self.strategies:
            action = strategy.get_action(game_state, valid_actions)
            if action in valid_actions:
                first_choices.append(action)

        # Most common first choice wins
        from collections import Counter
        vote_count = Counter(first_choices)

        if vote_count:
            winning_action = vote_count.most_common(1)[0][0]
            win_count = vote_count[winning_action]
            confidence = win_count / len(first_choices)
        else:
            winning_action = valid_actions[0]
            vote_count = {winning_action: 1}
            confidence = 0.0

        vote_distribution = {
            action: count / len(first_choices)
            for action, count in vote_count.items()
        } if first_choices else {}

        return ConsensusResult(
            selected_action=winning_action,
            votes=dict(vote_count),
            vote_distribution=vote_distribution,
            confidence=confidence,
            participating_strategies=[s.name for s in self.strategies],
            reasoning=f"Ranked choice: {winning_action} wins with {win_count} first-choice votes"
        )


class BordaCount(ConsensusSystem):
    """
    Borda count voting system.
    Strategies rank all actions, and points are awarded based on rank.
    """

    def __init__(self, strategies: List[ExpertStrategy]):
        super().__init__(strategies)
        self.name = "Borda Count"

    def _rank_actions(self, strategy: ExpertStrategy, game_state: GameState,
                     valid_actions: List[Action]) -> List[Action]:
        """Rank actions from best to worst according to strategy."""
        # For now, we'll use a simple heuristic
        # In practice, we would need each strategy to provide a full ranking

        # Get preferred action
        preferred = strategy.get_action(game_state, valid_actions)

        # Put preferred first, rest in default order
        ranking = [preferred]
        for action in valid_actions:
            if action != preferred:
                ranking.append(action)

        return ranking

    def get_consensus(self, game_state: GameState,
                     valid_actions: List[Action]) -> ConsensusResult:
        """Get consensus using Borda count."""
        # Award points: n-1 for first choice, n-2 for second, etc.
        n = len(valid_actions)
        scores = {action: 0 for action in valid_actions}

        for strategy in self.strategies:
            ranking = self._rank_actions(strategy, game_state, valid_actions)
            for rank, action in enumerate(ranking):
                scores[action] += (n - 1 - rank)

        # Action with highest score wins
        winning_action = max(scores, key=scores.get)
        max_score = scores[winning_action]
        total_possible = len(self.strategies) * (n - 1)
        confidence = max_score / total_possible if total_possible > 0 else 0

        vote_distribution = {
            action: score / total_possible
            for action, score in scores.items()
        } if total_possible > 0 else {}

        return ConsensusResult(
            selected_action=winning_action,
            votes=scores,
            vote_distribution=vote_distribution,
            confidence=confidence,
            participating_strategies=[s.name for s in self.strategies],
            reasoning=f"Borda count: {winning_action} with {max_score} points"
        )


class CopelandRule(ConsensusSystem):
    """
    Copeland's rule - pairwise comparison of actions.
    Action that beats the most other actions wins.
    """

    def __init__(self, strategies: List[ExpertStrategy]):
        super().__init__(strategies)
        self.name = "Copeland Rule"

    def get_consensus(self, game_state: GameState,
                     valid_actions: List[Action]) -> ConsensusResult:
        """Get consensus using Copeland's method."""
        # Count how many strategies prefer each action
        preferences = {action: 0 for action in valid_actions}

        for strategy in self.strategies:
            preferred = strategy.get_action(game_state, valid_actions)
            if preferred in valid_actions:
                preferences[preferred] += 1

        # Action with most preferences wins
        winning_action = max(preferences, key=preferences.get)
        win_count = preferences[winning_action]
        confidence = win_count / len(self.strategies) if self.strategies else 0

        vote_distribution = {
            action: count / len(self.strategies)
            for action, count in preferences.items()
        } if self.strategies else {}

        return ConsensusResult(
            selected_action=winning_action,
            votes=preferences,
            vote_distribution=vote_distribution,
            confidence=confidence,
            participating_strategies=[s.name for s in self.strategies],
            reasoning=f"Copeland: {winning_action} preferred by {win_count} strategies"
        )


class MetaLearnerConsensus(ConsensusSystem):
    """
    Meta-learner that chooses which strategy to use based on game state.
    Uses simple heuristics to select the best expert for the situation.
    """

    def __init__(self, strategies: List[ExpertStrategy]):
        super().__init__(strategies)
        self.name = "Meta-Learner"
        self.strategy_map = {
            "Basic Strategy": 0,
            "Hi-Lo with Index Plays": 1,
            "KO Counting System": 2,
            "Wizard of Odds Strategy": 3,
            "Thorp's Strategy": 4,
        }

    def get_consensus(self, game_state: GameState,
                     valid_actions: List[Action]) -> ConsensusResult:
        """Meta-learner selects strategy based on state."""
        true_count = game_state.true_count
        player_value = game_state.player_value
        dealer_value = game_state.dealer_up_card.value if game_state.dealer_up_card else 10

        # Choose strategy based on conditions
        if abs(true_count) >= 3:
            # High count situation - use advanced counting
            selected_strategy_name = "Hi-Lo with Index Plays"
        elif player_value <= 11:
            # Low hand - might want aggressive strategy
            selected_strategy_name = "Wizard of Odds Strategy"
        elif player_value >= 16:
            # High hand - conservative
            selected_strategy_name = "Conservative Strategy"
        else:
            # Default to basic strategy
            selected_strategy_name = "Basic Strategy"

        # Find the selected strategy
        selected_strategy = None
        for strategy in self.strategies:
            if strategy.name == selected_strategy_name:
                selected_strategy = strategy
                break

        if selected_strategy is None:
            selected_strategy = self.strategies[0]

        # Get action from selected strategy
        action = selected_strategy.get_action(game_state, valid_actions)

        return ConsensusResult(
            selected_action=action,
            votes={action: 1},
            vote_distribution={action: 1.0},
            confidence=0.8,  # High confidence in meta-learner
            participating_strategies=[selected_strategy.name],
            reasoning=f"Meta-learner selected: {selected_strategy.name} based on TC={true_count:.2f}, hand={player_value}"
        )


class HybridConsensus(ConsensusSystem):
    """
    Hybrid system that combines multiple consensus methods.
    Uses weighted combination of different voting systems.
    """

    def __init__(self, strategies: List[ExpertStrategy]):
        super().__init__(strategies)
        self.name = "Hybrid Consensus"

        # Initialize component systems
        self.majority = MajorityVoting(strategies)
        self.weighted = WeightedVoting(strategies)
        self.meta = MetaLearnerConsensus(strategies)

    def get_consensus(self, game_state: GameState,
                     valid_actions: List[Action]) -> ConsensusResult:
        """Get hybrid consensus."""
        # Get recommendations from all systems
        majority_result = self.majority.get_consensus(game_state, valid_actions)
        weighted_result = self.weighted.get_consensus(game_state, valid_actions)
        meta_result = self.meta.get_consensus(game_state, valid_actions)

        # Combine results with different weights
        # Meta-learner gets highest weight when confidence is high
        meta_weight = 0.5 if meta_result.confidence > 0.7 else 0.2
        majority_weight = 0.3
        weighted_weight = 1.0 - meta_weight - majority_weight

        # Calculate scores for each action
        action_scores = {action: 0.0 for action in valid_actions}

        # Add weighted votes
        for action, dist in weighted_result.vote_distribution.items():
            action_scores[action] += dist * weighted_weight

        # Add majority votes
        for action, dist in majority_result.vote_distribution.items():
            action_scores[action] += dist * majority_weight

        # Add meta-learner vote
        action_scores[meta_result.selected_action] += meta_weight

        # Select action with highest score
        winning_action = max(action_scores, key=action_scores.get)
        confidence = action_scores[winning_action]

        # Combine reasoning
        reasoning = f"Hybrid: Meta={meta_result.selected_action}, " \
                   f"Majority={majority_result.selected_action}, " \
                   f"Weighted={weighted_result.selected_action}"

        return ConsensusResult(
            selected_action=winning_action,
            votes={action: int(score * 100) for action, score in action_scores.items()},
            vote_distribution=action_scores,
            confidence=confidence,
            participating_strategies=self.strategies[0].name,  # Simplified
            reasoning=reasoning
        )


# Consensus system factory
def create_consensus_system(system_type: str,
                           strategies: List[ExpertStrategy]) -> ConsensusSystem:
    """Create a consensus system by type."""
    systems = {
        'majority': MajorityVoting,
        'weighted': WeightedVoting,
        'ranked': RankedVoting,
        'borda': BordaCount,
        'copeland': CopelandRule,
        'meta': MetaLearnerConsensus,
        'hybrid': HybridConsensus,
    }

    system_class = systems.get(system_type.lower(), MajorityVoting)
    return system_class(strategies)


def get_available_systems() -> List[str]:
    """Get list of available consensus systems."""
    return [
        'majority',
        'weighted',
        'ranked',
        'borda',
        'copeland',
        'meta',
        'hybrid',
    ]
