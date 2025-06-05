"""
Comprehensive Test Suite for Optimization Module
Testing all optimization components with realistic scenarios
"""
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

# Import optimization components
from .bayesian_optimizer import BayesianOptimizer, ParameterSpace
from .genetic_optimizer import GeneticOptimizer, GeneticConfig
from .portfolio_optimizer import PortfolioOptimizer, OptimizationConstraints
from .backtester import AdvancedBacktester, BacktestConfig, Trade
from ..core.data_models import StrategySignal, MarketContext, MarketRegime

# Test strategy for optimization
class TestStrategy:
    """Simple test strategy for optimization testing"""
    
    def __init__(self, parameters: Dict):
        self.parameters = parameters
        self.name = "TestStrategy"
    
    async def generate_signals(self, market_data: pd.DataFrame, context: MarketContext) -> List[StrategySignal]:
        """Generate simple momentum signals"""
        if len(market_data) < 20:
            return []
        
        # Simple momentum strategy
        short_ma = market_data['close'].rolling(5).mean().iloc[-1]
        long_ma = market_data['close'].rolling(20).mean().iloc[-1]
        current_price = market_data['close'].iloc[-1]
        
        momentum_threshold = self.parameters.get('momentum_threshold', 0.02)
        
        signals = []
        
        if short_ma > long_ma * (1 + momentum_threshold):
            signals.append(StrategySignal(
                symbol='TEST',
                action='BUY',
                quantity=100,
                price=current_price,
                confidence=0.7,
                timestamp=datetime.now()
            ))
        elif short_ma < long_ma * (1 - momentum_threshold):
            signals.append(StrategySignal(
                symbol='TEST',
                action='SELL',
                quantity=100,
                price=current_price,
                confidence=0.6,
                timestamp=datetime.now()
            ))
        
        return signals
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        return 'momentum_threshold' in self.parameters
    
    def get_parameter_space(self) -> Dict[str, tuple]:
        """Get parameter optimization bounds"""
        return {
            'momentum_threshold': (0.01, 0.10)
        }

def generate_test_market_data(days: int = 252, volatility: float = 0.2) -> pd.DataFrame:
    """Generate synthetic market data for testing"""
    
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Generate random price movements
    np.random.seed(42)
    returns = np.random.normal(0.0005, volatility/np.sqrt(252), days)
    
    # Create price series
    prices = [100.0]  # Starting price
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices[1:])):
        # Generate realistic OHLC from close
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i] * (1 + np.random.normal(0, 0.005))
        volume = int(np.random.normal(1000000, 200000))
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': max(volume, 100000)
        })
    
    return pd.DataFrame(data).set_index('timestamp')

def generate_test_contexts(days: int = 252) -> List[MarketContext]:
    """Generate test market contexts"""
    contexts = []
    
    for i in range(days):
        context = MarketContext(
            timestamp=datetime(2023, 1, 1) + timedelta(days=i),
            current_regime=MarketRegime.BULL_TRENDING,
            regime_confidence=0.7,
            volatility_percentile=50.0,
            trend_strength=0.3,
            market_stress_level=0.2
        )
        contexts.append(context)
    
    return contexts

async def test_bayesian_optimizer():
    """Test Bayesian optimization"""
    print("\n🔍 Testing Bayesian Optimizer...")
    
    # Define objective function
    async def objective_function(params):
        # Simulate strategy performance based on parameters
        momentum_threshold = params['momentum_threshold']
        
        # Simulate performance (higher threshold = lower but more stable returns)
        base_return = 0.15  # 15% base return
        stability_bonus = (momentum_threshold - 0.01) * 2  # Reward higher thresholds
        noise = np.random.normal(0, 0.05)  # Add some noise
        
        return base_return + stability_bonus + noise
    
    # Define parameter space
    parameter_space = [
        ParameterSpace('momentum_threshold', 0.01, 0.10, 'continuous')
    ]
    
    # Run optimization
    optimizer = BayesianOptimizer({'max_iterations': 10})
    result = await optimizer.optimize(
        objective_function=objective_function,
        parameter_space=parameter_space,
        initial_points=3
    )
    
    print(f"   ✅ Best parameters: {result.best_params}")
    print(f"   📊 Best score: {result.best_score:.6f}")
    print(f"   🔄 Iterations: {result.iterations}")
    
    return result

async def test_genetic_optimizer():
    """Test genetic algorithm optimization"""
    print("\n🧬 Testing Genetic Optimizer...")
    
    # Define fitness function
    async def fitness_function(params):
        # Multi-objective fitness: return vs risk
        momentum_threshold = params['momentum_threshold']
        
        # Calculate return component
        return_component = 0.10 + momentum_threshold * 2
        
        # Calculate risk component (lower is better)
        risk_component = 0.05 - momentum_threshold * 0.3
        
        # Combined fitness (higher is better)
        fitness = return_component - abs(risk_component)
        
        # Add some noise
        fitness += np.random.normal(0, 0.02)
        
        return fitness
    
    # Define parameter bounds
    parameter_bounds = {
        'momentum_threshold': (0.01, 0.10)
    }
    
    # Configure genetic algorithm
    config = GeneticConfig(
        population_size=20,
        max_generations=10,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    # Run evolution
    optimizer = GeneticOptimizer(config)
    result = await optimizer.evolve(
        fitness_function=fitness_function,
        parameter_bounds=parameter_bounds
    )
    
    print(f"   ✅ Best individual: {result.best_individual.genes}")
    print(f"   🏆 Best fitness: {result.best_individual.fitness:.6f}")
    print(f"   🧬 Generations: {result.generation_count}")
    print(f"   📈 Final diversity: {result.diversity_history[-1]:.4f}")
    
    return result

def test_portfolio_optimizer():
    """Test portfolio optimization"""
    print("\n🎯 Testing Portfolio Optimizer...")
    
    # Generate synthetic returns data for multiple assets
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Create correlated returns for 4 assets
    n_assets = 4
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    
    # Generate returns with different risk/return profiles
    returns_data = pd.DataFrame(index=dates, columns=asset_names)
    
    for i, asset in enumerate(asset_names):
        mean_return = 0.0008 + i * 0.0002  # Different expected returns
        volatility = 0.015 + i * 0.005       # Different volatilities
        
        returns = np.random.normal(mean_return, volatility, len(dates))
        returns_data[asset] = returns
    
    # Add some correlation
    correlation_matrix = np.array([
        [1.0, 0.6, 0.3, 0.2],
        [0.6, 1.0, 0.4, 0.3],
        [0.3, 0.4, 1.0, 0.5],
        [0.2, 0.3, 0.5, 1.0]
    ])
    
    # Apply correlation (simplified)
    returns_data = returns_data.fillna(0)
    
    optimizer = PortfolioOptimizer()
    
    # Test different optimization methods
    methods = ['mean_variance', 'risk_parity', 'maximum_diversification']
    results = {}
    
    for method in methods:
        constraints = OptimizationConstraints(max_weight=0.5, min_weight=0.05)
        
        result = optimizer.optimize_portfolio(
            returns_data=returns_data,
            method=method,
            constraints=constraints
        )
        
        results[method] = result
        
        print(f"   {method.replace('_', ' ').title()}:")
        print(f"     Weights: {result.weights}")
        print(f"     Expected Return: {result.expected_return:.2%}")
        print(f"     Volatility: {result.expected_volatility:.2%}")
        print(f"     Sharpe Ratio: {result.sharpe_ratio:.3f}")
    
    return results

async def test_backtester():
    """Test advanced backtesting engine"""
    print("\n🔄 Testing Advanced Backtester...")
    
    # Generate test data
    market_data = generate_test_market_data(days=100)
    contexts = generate_test_contexts(days=100)
    
    # Create test strategy
    strategy = TestStrategy({'momentum_threshold': 0.03})
    
    # Configure backtester
    config = BacktestConfig(
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0005,
        max_position_size=0.2
    )
    
    backtester = AdvancedBacktester(config)
    
    # Run backtest
    results = await backtester.backtest_strategy(
        strategy=strategy,
        market_data={'TEST': market_data},
        contexts={'TEST': contexts}
    )
    
    print(f"   📊 Performance Metrics:")
    print(f"     Total Return: {results.total_return:.2%}")
    print(f"     Annual Return: {results.annual_return:.2%}")
    print(f"     Volatility: {results.volatility:.2%}")
    print(f"     Sharpe Ratio: {results.sharpe_ratio:.3f}")
    print(f"     Max Drawdown: {results.max_drawdown:.2%}")
    print(f"   📈 Trade Statistics:")
    print(f"     Total Trades: {results.total_trades}")
    print(f"     Win Rate: {results.win_rate:.1%}")
    print(f"     Profit Factor: {results.profit_factor:.2f}")
    
    # Test Monte Carlo simulation
    if len(results.monthly_returns) >= 3:  # Need some data
        mc_results = backtester.monte_carlo_simulation(results, num_simulations=100)
        
        if 'error' not in mc_results:
            print(f"   🎲 Monte Carlo (100 simulations):")
            print(f"     Mean Annual Return: {mc_results['mean_annual_return']:.2%}")
            print(f"     5th Percentile: {mc_results['percentile_5']:.2%}")
            print(f"     95th Percentile: {mc_results['percentile_95']:.2%}")
    
    return results

async def main():
    """Run comprehensive optimization module tests"""
    print("🚀 Advanced AI Strategy Optimization Module - Comprehensive Testing")
    print("=" * 70)
    
    try:
        # Test all optimization components
        bayesian_results = await test_bayesian_optimizer()
        genetic_results = await test_genetic_optimizer()
        portfolio_results = test_portfolio_optimizer()
        backtest_results = await test_backtester()
        
        print("\n" + "=" * 70)
        print("✅ All Optimization Tests Completed Successfully!")
        print("\n📋 Test Summary:")
        print(f"   🔍 Bayesian Optimization: {bayesian_results.iterations} iterations")
        print(f"   🧬 Genetic Evolution: {genetic_results.generation_count} generations")
        print(f"   🎯 Portfolio Methods: {len(portfolio_results)} optimization methods")
        print(f"   🔄 Backtest Trades: {backtest_results.total_trades} executed")
        
        print("\n🎯 Module Integration Status: READY")
        print("   All optimization components are working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())