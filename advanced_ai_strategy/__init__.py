"""
Advanced AI Strategy System for Cryptocurrency Trading

A comprehensive, modular trading system with institutional-grade features:
- Market regime classification with 7 regime types
- Advanced strategy optimization (Bayesian, Genetic, Portfolio)
- Continuous learning and adaptation
- Real-time performance monitoring
- Risk management and emergency protocols

Author: Advanced AI Strategy Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Advanced AI Strategy Team"
__license__ = "MIT"

# Core imports
from .core import (
    BaseStrategy,
    StrategyRegistry,
    registry,
    MarketContext,
    StrategySignal,
    StrategyRecommendation,
    StrategyPerformance,
    MarketRegime,
    StrategyType
)

# Market Analysis
from .market_analysis import MarketRegimeClassifier

# Optimization
from .optimization import (
    BayesianOptimizer,
    GeneticOptimizer,
    PortfolioOptimizer,
    AdvancedBacktester,
    OptimizationResult,
    ParameterSpace,
    PortfolioWeights
)

# Learning & Adaptation
from .learning import (
    ContinuousLearner,
    PerformanceMonitor,
    AdaptiveManager
)

__all__ = [
    # Core Framework
    'BaseStrategy',
    'StrategyRegistry', 
    'registry',
    'MarketContext',
    'StrategySignal',
    'StrategyRecommendation',
    'StrategyPerformance',
    'MarketRegime',
    'StrategyType',
    
    # Market Analysis
    'MarketRegimeClassifier',
    
    # Optimization
    'BayesianOptimizer',
    'GeneticOptimizer', 
    'PortfolioOptimizer',
    'AdvancedBacktester',
    'OptimizationResult',
    'ParameterSpace',
    'PortfolioWeights',
    
    # Learning & Adaptation
    'ContinuousLearner',
    'PerformanceMonitor',
    'AdaptiveManager',
]