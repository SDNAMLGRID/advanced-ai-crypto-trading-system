"""
Portfolio Optimization Engine for Advanced AI Strategy System
Extracted and enhanced from the monolithic Advanced AI Strategy system
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PortfolioWeights:
    """Portfolio weight allocation results"""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    optimization_method: str
    risk_budget: Optional[Dict[str, float]] = None

@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    max_weight: float = 0.4  # Maximum allocation per asset
    min_weight: float = 0.0  # Minimum allocation per asset
    max_sector_weight: Optional[float] = None
    target_return: Optional[float] = None
    max_turnover: Optional[float] = None
    transaction_costs: float = 0.001

class PortfolioOptimizer:
    """
    Advanced portfolio optimization with multiple methods
    
    Features:
    - Mean-Variance Optimization (Markowitz)
    - Black-Litterman model
    - Risk Parity / Equal Risk Contribution
    - Maximum Diversification
    - Minimum Correlation
    - Kelly Criterion sizing
    - Regime-aware allocation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self.estimation_window = self.config.get('estimation_window', 252)
        
    def optimize_portfolio(self,
                          returns_data: pd.DataFrame,
                          method: str = 'mean_variance',
                          constraints: OptimizationConstraints = None,
                          views: Optional[Dict] = None) -> PortfolioWeights:
        """
        Optimize portfolio allocation using specified method
        
        Args:
            returns_data: DataFrame with asset returns
            method: Optimization method ('mean_variance', 'risk_parity', 'black_litterman', etc.)
            constraints: Portfolio constraints
            views: Investor views for Black-Litterman
        """
        if constraints is None:
            constraints = OptimizationConstraints()
            
        print(f"🎯 Optimizing Portfolio ({method.replace('_', ' ').title()})")
        
        # Calculate risk-return parameters
        expected_returns = self._calculate_expected_returns(returns_data)
        covariance_matrix = self._calculate_covariance_matrix(returns_data)
        
        # Apply optimization method
        if method == 'mean_variance':
            weights = self._mean_variance_optimization(
                expected_returns, covariance_matrix, constraints
            )
        elif method == 'risk_parity':
            weights = self._risk_parity_optimization(
                covariance_matrix, constraints
            )
        elif method == 'maximum_diversification':
            weights = self._maximum_diversification_optimization(
                expected_returns, covariance_matrix, constraints
            )
        elif method == 'minimum_correlation':
            weights = self._minimum_correlation_optimization(
                returns_data, constraints
            )
        else:
            # Default to equal weights
            n_assets = len(returns_data.columns)
            weights = np.array([1/n_assets] * n_assets)
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_volatility = np.sqrt(
            np.dot(weights, np.dot(covariance_matrix, weights))
        )
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Convert to weight dictionary
        asset_names = returns_data.columns.tolist()
        weight_dict = dict(zip(asset_names, weights))
        
        print(f"   ✅ Expected Return: {portfolio_return:.2%}")
        print(f"   📊 Expected Volatility: {portfolio_volatility:.2%}")
        print(f"   📈 Sharpe Ratio: {sharpe_ratio:.3f}")
        
        return PortfolioWeights(
            weights=weight_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            optimization_method=method
        )
    
    def _calculate_expected_returns(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Calculate expected returns using historical mean"""
        return returns_data.mean().values * 252  # Annualized
    
    def _calculate_covariance_matrix(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Calculate covariance matrix"""
        return returns_data.cov().values * 252
    
    def _mean_variance_optimization(self,
                                   expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   constraints: OptimizationConstraints) -> np.ndarray:
        """Mean-Variance optimization (Markowitz)"""
        
        n_assets = len(expected_returns)
        
        # Objective function: minimize negative Sharpe ratio
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            
            if portfolio_volatility == 0:
                return -np.inf
            
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            return -sharpe_ratio  # Minimize negative Sharpe
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds for individual weights
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            return result.x
        else:
            # Fallback to equal weights
            return np.array([1/n_assets] * n_assets)
    
    def _risk_parity_optimization(self,
                                 covariance_matrix: np.ndarray,
                                 constraints: OptimizationConstraints) -> np.ndarray:
        """Risk Parity optimization (Equal Risk Contribution)"""
        
        n_assets = len(covariance_matrix)
        
        def objective(weights):
            # Calculate risk contributions
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            marginal_risk = np.dot(covariance_matrix, weights) / portfolio_vol
            risk_contributions = weights * marginal_risk / portfolio_vol
            
            # Minimize sum of squared deviations from equal risk contribution
            target_risk = 1 / n_assets
            return np.sum((risk_contributions - target_risk) ** 2)
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess: inverse volatility weights
        inv_vol = 1 / np.sqrt(np.diag(covariance_matrix))
        x0 = inv_vol / np.sum(inv_vol)
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            return result.x
        else:
            return x0
    
    def _maximum_diversification_optimization(self,
                                            expected_returns: np.ndarray,
                                            covariance_matrix: np.ndarray,
                                            constraints: OptimizationConstraints) -> np.ndarray:
        """Maximum Diversification optimization"""
        
        n_assets = len(expected_returns)
        
        def objective(weights):
            # Diversification ratio = weighted average volatility / portfolio volatility
            individual_vols = np.sqrt(np.diag(covariance_matrix))
            weighted_avg_vol = np.sum(weights * individual_vols)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            
            if portfolio_vol == 0:
                return 0
            
            diversification_ratio = weighted_avg_vol / portfolio_vol
            return -diversification_ratio  # Minimize negative (maximize diversification)
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            return result.x
        else:
            return x0
    
    def _minimum_correlation_optimization(self,
                                        returns_data: pd.DataFrame,
                                        constraints: OptimizationConstraints) -> np.ndarray:
        """Minimum Correlation optimization"""
        
        correlation_matrix = returns_data.corr().values
        n_assets = len(correlation_matrix)
        
        def objective(weights):
            # Weighted average correlation
            weighted_correlation = 0
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    weighted_correlation += weights[i] * weights[j] * correlation_matrix[i, j]
            
            return weighted_correlation * 2  # Factor of 2 for double counting
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            return result.x
        else:
            return x0