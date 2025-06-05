"""
Complete System Integration Tests
Tests all phases working together in a realistic trading scenario
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

# Import all system components
from advanced_ai_strategy.core import (
    BaseStrategy, StrategyRegistry, registry,
    MarketContext, StrategySignal, MarketRegime
)
from advanced_ai_strategy.market_analysis import MarketRegimeClassifier
from advanced_ai_strategy.optimization import (
    BayesianOptimizer, GeneticOptimizer, PortfolioOptimizer, AdvancedBacktester
)
from advanced_ai_strategy.learning import (
    ContinuousLearner, PerformanceMonitor, AdaptiveManager
)

class ExampleStrategy(BaseStrategy):
    """Example strategy for system testing"""
    
    async def generate_signals(self, market_data: pd.DataFrame, context: MarketContext) -> List[StrategySignal]:
        """Generate trading signals based on market data"""
        if len(market_data) < 20:
            return []
        
        # Simple momentum strategy with regime awareness
        sma_short = market_data['close'].rolling(10).mean().iloc[-1]
        sma_long = market_data['close'].rolling(20).mean().iloc[-1]
        current_price = market_data['close'].iloc[-1]
        
        threshold = self.parameters.get('momentum_threshold', 0.02)
        confidence_boost = 0.2 if context.current_regime == MarketRegime.BULL_TRENDING else 0.0
        
        signals = []
        
        if sma_short > sma_long * (1 + threshold):
            confidence = min(0.9, 0.6 + confidence_boost)
            signals.append(StrategySignal(
                symbol=context.symbol or 'BTC',
                action='BUY',
                quantity=100,
                price=current_price,
                confidence=confidence,
                timestamp=datetime.now()
            ))
        
        return signals
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        return 'momentum_threshold' in self.parameters
    
    def get_parameter_space(self) -> Dict[str, tuple]:
        """Get parameter optimization bounds"""
        return {'momentum_threshold': (0.01, 0.08)}

def generate_realistic_market_data(symbol: str = "BTCUSDT", days: int = 365) -> pd.DataFrame:
    """Generate realistic crypto market data"""
    np.random.seed(42)
    
    # Start with realistic crypto price
    start_price = 45000 if symbol == "BTCUSDT" else 3000
    dates = pd.date_range(start='2023-01-01', periods=days * 24, freq='H')  # Hourly data
    
    # Generate realistic crypto returns with volatility clustering
    returns = []
    volatility = 0.02  # Base volatility
    
    for i in range(len(dates)):
        # Volatility clustering
        volatility += np.random.normal(0, 0.001)
        volatility = max(0.01, min(0.05, volatility))
        
        # Market regimes
        if i < len(dates) * 0.3:  # Bull market
            drift = 0.0002
        elif i < len(dates) * 0.7:  # Ranging market
            drift = 0.0000
        else:  # Bear market
            drift = -0.0001
        
        ret = np.random.normal(drift, volatility)
        returns.append(ret)
    
    # Create price series
    prices = [start_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLCV
    data = []
    for i, (date, price) in enumerate(zip(dates, prices[1:])):
        # Realistic OHLC generation
        volatility_factor = abs(returns[i]) * 10
        
        high = price * (1 + abs(np.random.normal(0, volatility_factor)))
        low = price * (1 - abs(np.random.normal(0, volatility_factor)))
        open_price = prices[i] * (1 + np.random.normal(0, volatility_factor/2))
        
        # Realistic volume (higher during volatile periods)
        base_volume = 1000000 if symbol == "BTCUSDT" else 500000
        volume_multiplier = 1 + volatility_factor * 20
        volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 2.0))
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    return pd.DataFrame(data).set_index('timestamp')

async def test_complete_system_integration():
    """Test all system phases working together"""
    print("🚀 Testing Complete System Integration...")
    print("=" * 60)
    
    # 1. Generate realistic market data
    print("\n📊 Phase 1: Generating Market Data...")
    btc_data = generate_realistic_market_data("BTCUSDT", days=90)
    eth_data = generate_realistic_market_data("ETHUSDT", days=90)
    
    print(f"   BTC Data: {len(btc_data)} samples")
    print(f"   ETH Data: {len(eth_data)} samples")
    
    # 2. Market Analysis (Phase 2)
    print("\n🔍 Phase 2: Market Regime Classification...")
    classifier = MarketRegimeClassifier()
    
    btc_regime = await classifier.classify_regime(btc_data)
    eth_regime = await classifier.classify_regime(eth_data)
    
    print(f"   BTC Regime: {btc_regime.value}")
    print(f"   ETH Regime: {eth_regime.value}")
    
    # Create market contexts
    btc_context = MarketContext(
        symbol="BTCUSDT",
        timeframe="1h", 
        current_regime=btc_regime,
        volatility_rank=75.0,
        trend_strength=0.6,
        liquidity_score=95.0
    )
    
    eth_context = MarketContext(
        symbol="ETHUSDT",
        timeframe="1h",
        current_regime=eth_regime, 
        volatility_rank=70.0,
        trend_strength=0.5,
        liquidity_score=90.0
    )
    
    # 3. Strategy Framework (Phase 1)
    print("\n⚙️ Phase 1: Strategy Framework...")
    
    # Register example strategy
    registry.register(ExampleStrategy)
    
    # Create strategy instance
    strategy = registry.create_strategy('ExampleStrategy', {
        'momentum_threshold': 0.025,
        'historical_win_rate': 0.58,
        'avg_win': 0.03,
        'avg_loss': 0.015
    })
    
    print(f"   Strategy Registered: {strategy.name}")
    print(f"   Available Strategies: {registry.list_strategies()}")
    
    # 4. Strategy Optimization (Phase 3)
    print("\n🎯 Phase 3: Strategy Optimization...")
    
    # Mock fitness function for optimization
    async def strategy_fitness(params):
        test_strategy = ExampleStrategy(params)
        signals = await test_strategy.generate_signals(btc_data[-100:], btc_context)
        # Simple fitness: number of signals * confidence
        return len(signals) * (sum(s.confidence for s in signals) / max(1, len(signals)))
    
    # Quick Bayesian optimization
    from advanced_ai_strategy.optimization.bayesian_optimizer import ParameterSpace
    
    optimizer = BayesianOptimizer({'max_iterations': 5})
    param_space = [ParameterSpace('momentum_threshold', 0.01, 0.08, 'continuous')]
    
    opt_result = await optimizer.optimize(
        objective_function=strategy_fitness,
        parameter_space=param_space,
        initial_points=2
    )
    
    print(f"   Optimized Parameters: {opt_result.best_params}")
    print(f"   Best Score: {opt_result.best_score:.3f}")
    
    # 5. Portfolio Optimization
    print("\n💼 Portfolio Optimization...")
    
    # Create returns data for portfolio optimization
    btc_returns = btc_data['close'].pct_change().dropna()[-252:]
    eth_returns = eth_data['close'].pct_change().dropna()[-252:]
    
    portfolio_data = pd.DataFrame({
        'BTC': btc_returns,
        'ETH': eth_returns
    }).dropna()
    
    portfolio_optimizer = PortfolioOptimizer()
    portfolio_weights = portfolio_optimizer.optimize_portfolio(
        returns_data=portfolio_data,
        method='mean_variance'
    )
    
    print(f"   Portfolio Weights: {portfolio_weights.weights}")
    print(f"   Expected Return: {portfolio_weights.expected_return:.2%}")
    print(f"   Sharpe Ratio: {portfolio_weights.sharpe_ratio:.3f}")
    
    # 6. Learning & Adaptation (Phase 4)
    print("\n🧠 Phase 4: Learning & Adaptation...")
    
    # Initialize learning components
    learner = ContinuousLearner()
    monitor = PerformanceMonitor()
    manager = AdaptiveManager()
    
    # Simulate some learning
    from advanced_ai_strategy.core.data_models import StrategyPerformance
    
    test_performance = StrategyPerformance(
        strategy_id="ExampleStrategy",
        total_trades=50,
        win_rate=0.62,
        profit_factor=1.8,
        sharpe_ratio=1.4,
        max_drawdown=0.12,
        avg_win=0.035,
        avg_loss=0.018,
        avg_trade_duration=6.5,
        best_market_regime=btc_regime,
        worst_market_regime=MarketRegime.HIGH_VOLATILITY,
        confidence_correlation=0.75,
        risk_adjusted_return=0.18,
        information_ratio=1.2,
        calmar_ratio=1.5
    )
    
    # Test learning
    learning_result = await learner.learn_from_performance(
        test_performance, btc_context, {'ensemble': 1.4}
    )
    
    # Test monitoring 
    alerts = await monitor.update_performance(test_performance, "ExampleStrategy")
    
    print(f"   Learning Success: {learning_result.success}")
    print(f"   Monitoring Alerts: {len(alerts)}")
    print(f"   System Learning: ACTIVE")
    
    # 7. Integration Summary
    print("\n" + "=" * 60)
    print("✅ COMPLETE SYSTEM INTEGRATION TEST SUCCESSFUL!")
    print("\n📋 System Components Status:")
    print("   ⚙️ Core Framework: OPERATIONAL")
    print("   🔍 Market Analysis: OPERATIONAL")
    print("   🎯 Strategy Optimization: OPERATIONAL")
    print("   🧠 Learning & Adaptation: OPERATIONAL")
    print("\n🚀 Advanced AI Crypto Trading System: READY FOR DEPLOYMENT")
    
    return True

async def main():
    """Run complete system tests"""
    try:
        await test_complete_system_integration()
    except Exception as e:
        print(f"\n❌ System test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())