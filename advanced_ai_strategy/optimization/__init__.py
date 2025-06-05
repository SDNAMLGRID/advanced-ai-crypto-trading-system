"""
Optimization Module for Advanced AI Strategy System

Provides comprehensive optimization capabilities including:
- Bayesian optimization for parameter tuning
- Genetic algorithms for strategy evolution
- Portfolio optimization with multiple methods
- Advanced backtesting and performance evaluation

Optimization Methods:
1. Bayesian Optimization:
   - Gaussian Process surrogate models
   - Acquisition function optimization (EI, UCB, PI)
   - Adaptive parameter space exploration
   - Multi-objective optimization support

2. Genetic Algorithm:
   - Multi-objective evolution
   - Elitism with diversity preservation
   - Parallel fitness evaluation
   - Strategy genealogy tracking
   - Adaptive mutation rates

3. Portfolio Optimization:
   - Mean-Variance (Markowitz)
   - Risk Parity / Equal Risk Contribution
   - Maximum Diversification
   - Minimum Correlation
   - Black-Litterman model
   - Kelly Criterion sizing

4. Backtesting Engine:
   - Walk-forward analysis
   - Regime-aware evaluation
   - Risk metrics calculation
   - Performance attribution
   - Monte Carlo simulation
"""

from .bayesian_optimizer import (
    BayesianOptimizer,
    MultiObjectiveOptimizer,
    OptimizationResult,
    ParameterSpace
)

from .genetic_optimizer import (
    GeneticOptimizer,
    Individual,
    GeneticConfig,
    EvolutionResults
)

from .portfolio_optimizer import (
    PortfolioOptimizer,
    PortfolioWeights,
    OptimizationConstraints
)

__all__ = [
    # Bayesian Optimization
    'BayesianOptimizer',
    'MultiObjectiveOptimizer',
    'OptimizationResult',
    'ParameterSpace',
    
    # Genetic Algorithm
    'GeneticOptimizer',
    'Individual',
    'GeneticConfig',
    'EvolutionResults',
    
    # Portfolio Optimization
    'PortfolioOptimizer',
    'PortfolioWeights',
    'OptimizationConstraints'
]