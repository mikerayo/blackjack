"""Strategies module - Expert strategies and consensus systems."""

from .expert_strategies import (
    ExpertStrategy,
    BasicStrategy,
    HiLoCountingStrategy,
    KOCountingStrategy,
    AceFiveCount,
    WizardOfOddsStrategy,
    ThorpStrategy,
    WongHalvesStrategy,
    ZenCountStrategy,
    AggressiveStrategy,
    ConservativeStrategy,
    AdaptiveStrategy,
    get_all_strategies,
    get_strategy_by_name
)

from .consensus_system import (
    ConsensusSystem,
    MajorityVoting,
    WeightedVoting,
    RankedVoting,
    BordaCount,
    CopelandRule,
    MetaLearnerConsensus,
    HybridConsensus,
    create_consensus_system,
    get_available_systems,
    ConsensusResult
)

from .bankroll_management import (
    BankrollManager,
    FlatBetting,
    KellyCriterion,
    HiLoBetting,
    KOSystemBetting,
    AdaptiveBetting,
    ConservativeBetting,
    AggressiveBetting,
    ParlayBetting,
    create_betting_system,
    get_available_betting_systems,
    BettingDecision
)

__all__ = [
    # Expert Strategies
    'ExpertStrategy',
    'BasicStrategy',
    'HiLoCountingStrategy',
    'KOCountingStrategy',
    'AceFiveCount',
    'WizardOfOddsStrategy',
    'ThorpStrategy',
    'WongHalvesStrategy',
    'ZenCountStrategy',
    'AggressiveStrategy',
    'ConservativeStrategy',
    'AdaptiveStrategy',
    'get_all_strategies',
    'get_strategy_by_name',

    # Consensus Systems
    'ConsensusSystem',
    'MajorityVoting',
    'WeightedVoting',
    'RankedVoting',
    'BordaCount',
    'CopelandRule',
    'MetaLearnerConsensus',
    'HybridConsensus',
    'create_consensus_system',
    'get_available_systems',
    'ConsensusResult',

    # Bankroll Management
    'BankrollManager',
    'FlatBetting',
    'KellyCriterion',
    'HiLoBetting',
    'KOSystemBetting',
    'AdaptiveBetting',
    'ConservativeBetting',
    'AggressiveBetting',
    'ParlayBetting',
    'create_betting_system',
    'get_available_betting_systems',
    'BettingDecision',
]
