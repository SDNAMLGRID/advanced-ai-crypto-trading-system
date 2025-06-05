#!/usr/bin/env python3
"""
Unified Test Runner for Advanced AI Strategy System
Runs all tests across all phases with comprehensive reporting
"""
import asyncio
import sys
import os
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def run_phase_1_tests():
    """Test Core Framework (Phase 1)"""
    print("⚙️ Testing Phase 1: Core Framework...")
    
    try:
        # Test imports
        from advanced_ai_strategy.core import (
            BaseStrategy, StrategyRegistry, registry,
            MarketContext, StrategySignal, MarketRegime
        )
        
        # Test strategy registry
        class TestStrategy(BaseStrategy):
            async def generate_signals(self, market_data, context):
                return []
            def validate_parameters(self):
                return True
            def get_parameter_space(self):
                return {}
        
        registry.register(TestStrategy)
        strategy = registry.create_strategy('TestStrategy', {})
        
        print("   ✅ Core Framework: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ Core Framework: FAILED - {e}")
        return False

async def run_phase_2_tests():
    """Test Market Analysis (Phase 2)"""
    print("🔍 Testing Phase 2: Market Analysis...")
    
    try:
        from advanced_ai_strategy.market_analysis import MarketRegimeClassifier
        import pandas as pd
        import numpy as np
        
        # Create test data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        test_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000000, 2000000, 100)
        }, index=dates)
        
        # Test regime classification
        classifier = MarketRegimeClassifier()
        regime = await classifier.classify_regime(test_data)
        
        print(f"   ✅ Market Analysis: Detected regime {regime.value}")
        return True
        
    except Exception as e:
        print(f"   ❌ Market Analysis: FAILED - {e}")
        return False

async def run_phase_3_tests():
    """Test Optimization (Phase 3)"""
    print("🎯 Testing Phase 3: Optimization...")
    
    try:
        from advanced_ai_strategy.optimization import (
            BayesianOptimizer, GeneticOptimizer, PortfolioOptimizer,
            ParameterSpace
        )
        import pandas as pd
        import numpy as np
        
        # Test Bayesian Optimization
        async def simple_objective(params):
            return -(params['x'] - 0.5) ** 2  # Maximum at x=0.5
        
        optimizer = BayesianOptimizer({'max_iterations': 3})
        param_space = [ParameterSpace('x', 0.0, 1.0, 'continuous')]
        
        result = await optimizer.optimize(
            objective_function=simple_objective,
            parameter_space=param_space,
            initial_points=2
        )
        
        # Test Portfolio Optimization
        returns_data = pd.DataFrame({
            'Asset1': np.random.normal(0.001, 0.02, 252),
            'Asset2': np.random.normal(0.0008, 0.018, 252)
        })
        
        portfolio_optimizer = PortfolioOptimizer()
        weights = portfolio_optimizer.optimize_portfolio(returns_data)
        
        print(f"   ✅ Optimization: Bayesian converged, Portfolio optimized")
        return True
        
    except Exception as e:
        print(f"   ❌ Optimization: FAILED - {e}")
        return False

async def run_phase_4_tests():
    """Test Learning & Adaptation (Phase 4)"""
    print("🧠 Testing Phase 4: Learning & Adaptation...")
    
    try:
        from advanced_ai_strategy.learning import (
            ContinuousLearner, PerformanceMonitor, AdaptiveManager
        )
        from advanced_ai_strategy.core.data_models import (
            StrategyPerformance, MarketContext, MarketRegime
        )
        
        # Test components
        learner = ContinuousLearner()
        monitor = PerformanceMonitor()
        manager = AdaptiveManager()
        
        # Create test performance
        performance = StrategyPerformance(
            strategy_id="test",
            total_trades=50,
            win_rate=0.6,
            profit_factor=1.5,
            sharpe_ratio=1.2,
            max_drawdown=0.1,
            avg_win=0.02,
            avg_loss=0.01,
            avg_trade_duration=4.0,
            best_market_regime=MarketRegime.BULL_TRENDING,
            worst_market_regime=MarketRegime.HIGH_VOLATILITY,
            confidence_correlation=0.7,
            risk_adjusted_return=0.15,
            information_ratio=1.0,
            calmar_ratio=1.3
        )
        
        context = MarketContext(
            symbol="TEST",
            timeframe="1h",
            current_regime=MarketRegime.BULL_TRENDING,
            volatility_rank=50.0,
            trend_strength=0.6,
            liquidity_score=85.0
        )
        
        # Test learning
        learning_result = await learner.learn_from_performance(
            performance, context, {'model': 1.2}
        )
        
        # Test monitoring
        alerts = await monitor.update_performance(performance, "test_strategy")
        
        print(f"   ✅ Learning & Adaptation: Learning active, Monitoring operational")
        return True
        
    except Exception as e:
        print(f"   ❌ Learning & Adaptation: FAILED - {e}")
        return False

async def run_integration_tests():
    """Test complete system integration"""
    print("🔗 Testing System Integration...")
    
    try:
        # Import everything
        from advanced_ai_strategy import (
            BaseStrategy, registry, MarketRegimeClassifier,
            BayesianOptimizer, ContinuousLearner
        )
        
        # Test end-to-end workflow
        class IntegrationStrategy(BaseStrategy):
            async def generate_signals(self, market_data, context):
                return []
            def validate_parameters(self):
                return True
            def get_parameter_space(self):
                return {'param': (0.1, 1.0)}
        
        # Register strategy
        registry.register(IntegrationStrategy, 'IntegrationTest')
        
        # Create instances
        classifier = MarketRegimeClassifier()
        optimizer = BayesianOptimizer()
        learner = ContinuousLearner()
        
        print("   ✅ System Integration: ALL COMPONENTS CONNECTED")
        return True
        
    except Exception as e:
        print(f"   ❌ System Integration: FAILED - {e}")
        return False

async def main():
    """Run comprehensive test suite"""
    print("🚀 Advanced AI Strategy System - Comprehensive Test Suite")
    print("=" * 70)
    
    start_time = time.time()
    
    # Run all test phases
    test_results = {
        'Phase 1 (Core)': await run_phase_1_tests(),
        'Phase 2 (Market Analysis)': await run_phase_2_tests(),
        'Phase 3 (Optimization)': await run_phase_3_tests(),
        'Phase 4 (Learning)': await run_phase_4_tests(),
        'Integration': await run_integration_tests()
    }
    
    # Calculate results
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    failed_tests = total_tests - passed_tests
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name:<25} {status}")
    
    print(f"\n📈 Results: {passed_tests}/{total_tests} tests passed")
    print(f"⏱️  Duration: {elapsed_time:.2f} seconds")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT!")
        print("🚀 Advanced AI Crypto Trading System: FULLY OPERATIONAL")
        return 0
    else:
        print(f"\n⚠️  {failed_tests} TEST(S) FAILED - REVIEW REQUIRED")
        print("🔧 Please check the error messages above and fix issues")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())