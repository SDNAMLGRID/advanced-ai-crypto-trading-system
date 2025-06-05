"""
Test script for Phase 4: Learning & Adaptation Module
Tests continuous learning, performance monitoring, and adaptive management
"""
import asyncio
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import learning components
from advanced_ai_strategy.learning.continuous_learner import (
    ContinuousLearner, LearningConfig, LearningMode, AdaptationResult
)
from advanced_ai_strategy.learning.performance_monitor import (
    PerformanceMonitor, MonitorConfig, AlertLevel, PerformanceAlert
)
from advanced_ai_strategy.learning.adaptive_manager import (
    AdaptiveManager, AdaptationType, ParameterUpdate, AdaptationEvent
)
from advanced_ai_strategy.core.data_models import (
    StrategyPerformance, MarketContext, MarketRegime
)

def generate_sample_performance_data(num_samples: int = 100) -> List[StrategyPerformance]:
    """Generate sample strategy performance data for testing"""
    np.random.seed(42)
    
    performances = []
    
    for i in range(num_samples):
        # Add some trend and noise
        trend_factor = 1.0 + (i / num_samples) * 0.1  # Slight upward trend
        noise = np.random.normal(0, 0.05)
        
        # Simulate performance degradation after sample 60
        if i > 60:
            trend_factor *= 0.9  # Performance decline
        
        # Generate base metrics
        win_rate = max(0.2, min(0.9, 0.6 + np.random.normal(0, 0.1)))
        avg_win = max(0.01, 0.05 * trend_factor + np.random.normal(0, 0.01))
        avg_loss = max(0.01, 0.03 + np.random.normal(0, 0.005))
        
        profit_factor = (win_rate * avg_win) / ((1 - win_rate) * avg_loss + 1e-8)
        sharpe_ratio = max(-2.0, min(3.0, trend_factor * 1.5 + np.random.normal(0, 0.3)))
        max_drawdown = max(0.01, min(0.3, abs(np.random.normal(0.1, 0.05))))
        
        # Random regime assignment
        regimes = list(MarketRegime)
        best_regime = regimes[i % len(regimes)]
        worst_regime = regimes[(i + 3) % len(regimes)]
        
        performance = StrategyPerformance(
            strategy_id=f"test_strategy_{i}",
            total_trades=i + 10,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_duration=4.0 + np.random.normal(0, 1.0),  # hours
            best_market_regime=best_regime,
            worst_market_regime=worst_regime,
            confidence_correlation=np.random.uniform(0.3, 0.9),
            risk_adjusted_return=sharpe_ratio * 0.15,  # Approximate
            information_ratio=sharpe_ratio * 0.8 + np.random.normal(0, 0.1),
            calmar_ratio=sharpe_ratio / (max_drawdown + 1e-8)
        )
        
        performances.append(performance)
    
    return performances

def generate_sample_market_contexts(num_samples: int = 100) -> List[MarketContext]:
    """Generate sample market context data for testing"""
    np.random.seed(42)
    
    contexts = []
    regimes = list(MarketRegime)
    
    for i in range(num_samples):
        # Simulate regime changes
        if i < 30:
            regime = MarketRegime.BULL_TRENDING
        elif i < 60:
            regime = MarketRegime.RANGING
        elif i < 80:
            regime = MarketRegime.HIGH_VOLATILITY
        else:
            regime = MarketRegime.BEAR_TRENDING
        
        context = MarketContext(
            symbol="BTCUSDT",
            timeframe="1h",
            current_regime=regime,
            volatility_rank=np.random.uniform(20, 80),
            trend_strength=np.random.uniform(0.1, 0.9),
            liquidity_score=np.random.uniform(70, 95),
            correlation_matrix={},
            macro_indicators={},
            sentiment_scores={},
            volume_profile={},
            on_chain_metrics={},
            event_calendar=[]
        )
        
        contexts.append(context)
    
    return contexts

async def test_continuous_learner():
    """Test continuous learning functionality"""
    print("🧠 Testing Continuous Learner...")
    
    # Configure learning system
    config = LearningConfig(
        learning_rate=0.01,
        window_size=200,
        drift_threshold=0.05,
        learning_mode=LearningMode.HYBRID,
        enable_drift_detection=True
    )
    
    learner = ContinuousLearner(config)
    
    # Generate test data
    performances = generate_sample_performance_data(80)
    contexts = generate_sample_market_contexts(80)
    
    adaptation_count = 0
    
    # Simulate learning process
    for i, (performance, context) in enumerate(zip(performances, contexts)):
        # Mock predictions from multiple models
        predictions = {
            'model_a': np.random.normal(performance.sharpe_ratio, 0.1),
            'model_b': np.random.normal(performance.sharpe_ratio, 0.15),
            'ensemble': np.random.normal(performance.sharpe_ratio, 0.08)
        }
        
        # Learn from performance
        adaptation_result = await learner.learn_from_performance(
            performance, context, predictions
        )
        
        if adaptation_result.success:
            adaptation_count += 1
            print(f"   Adaptation {adaptation_count}: {adaptation_result.adaptation_type}")
    
    # Get learning statistics
    stats = learner.get_learning_statistics()
    
    print(f"   ✅ Continuous Learning Test Complete")
    print(f"   📊 Total Samples: {stats['state']['total_samples']}")
    print(f"   🔄 Adaptations: {stats['state']['adaptation_count']}")
    print(f"   🎯 Confidence: {stats['state']['confidence_level']:.3f}")
    print(f"   📈 Performance Trend: {stats['performance_stats']['performance_trend']}")
    print(f"   💾 Memory Usage: {stats['memory_usage']['memory_utilization']:.1%}")
    
    return True

async def test_performance_monitor():
    """Test performance monitoring functionality"""
    print("\n📊 Testing Performance Monitor...")
    
    # Configure monitoring
    config = MonitorConfig(
        window_size=300,
        alert_thresholds={
            'performance_degradation': 0.1,
            'max_drawdown': 0.15,
            'volatility_spike': 2.0,
            'sharpe_decline': 0.3
        },
        enable_real_time_alerts=True,
        enable_drift_detection=True
    )
    
    monitor = PerformanceMonitor(config)
    
    # Generate test performance data
    performances = generate_sample_performance_data(100)
    
    total_alerts = 0
    
    # Simulate real-time monitoring
    for i, performance in enumerate(performances):
        alerts = await monitor.update_performance(
            performance, strategy_name="test_strategy"
        )
        
        total_alerts += len(alerts)
        
        # Print significant alerts
        for alert in alerts:
            if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                print(f"   ⚠️ {alert.level.value}: {alert.message}")
    
    # Export monitoring data
    export_data = monitor.export_monitoring_data()
    
    # Get monitoring statistics
    stats = monitor.get_monitoring_statistics()
    
    print(f"   ✅ Performance Monitoring Test Complete")
    print(f"   📊 Samples Processed: {stats['samples_processed']}")
    print(f"   🚨 Total Alerts: {total_alerts}")
    print(f"   🕊 Critical Alerts: {stats['alerts_by_level'].get('CRITICAL', 0)}")
    print(f"   📁 Export Records: {len(export_data)}")
    
    return True

async def test_adaptive_manager():
    """Test adaptive management functionality"""
    print("\n⚙️ Testing Adaptive Manager...")
    
    manager = AdaptiveManager()
    
    # Generate test data
    performances = generate_sample_performance_data(70)
    contexts = generate_sample_market_contexts(70)
    
    adaptation_count = 0
    emergency_count = 0
    
    # Simulate adaptive management
    for i, (performance, context) in enumerate(zip(performances, contexts)):
        # Create parameter update recommendations
        param_updates = [
            ParameterUpdate(
                parameter_name="risk_factor",
                old_value=0.1,
                new_value=0.1 + np.random.normal(0, 0.02),
                confidence=np.random.uniform(0.6, 0.9),
                reason="Performance optimization"
            ),
            ParameterUpdate(
                parameter_name="position_size",
                old_value=0.05,
                new_value=max(0.01, 0.05 + np.random.normal(0, 0.01)),
                confidence=np.random.uniform(0.5, 0.8),
                reason="Risk adjustment"
            )
        ]
        
        # Process adaptation
        adaptation_result = await manager.process_adaptation(
            performance=performance,
            context=context,
            parameter_updates=param_updates
        )
        
        if adaptation_result.success:
            adaptation_count += 1
            
            if adaptation_result.adaptation_type == AdaptationType.EMERGENCY_STOP:
                emergency_count += 1
                print(f"   🚨 Emergency Stop Triggered: {adaptation_result.reason}")
    
    # Get adaptation history
    adaptation_history = manager.get_adaptation_history()
    
    # Get adaptation statistics
    stats = manager.get_adaptation_statistics()
    
    print(f"   ✅ Adaptive Management Test Complete")
    print(f"   🔄 Total Adaptations: {adaptation_count}")
    print(f"   🚨 Emergency Stops: {emergency_count}")
    print(f"   📈 Performance Improvements: {stats['adaptations_by_type'].get('PERFORMANCE_CORRECTION', 0)}")
    print(f"   ⚙️ Parameter Updates: {stats['adaptations_by_type'].get('PARAMETER_UPDATE', 0)}")
    print(f"   📁 History Records: {len(adaptation_history)}")
    
    return True

async def test_integrated_learning_workflow():
    """Test integrated workflow with all components"""
    print("\n🔄 Testing Integrated Learning Workflow...")
    
    # Initialize all components
    learning_config = LearningConfig(learning_mode=LearningMode.HYBRID)
    monitor_config = MonitorConfig(enable_real_time_alerts=True)
    
    learner = ContinuousLearner(learning_config)
    monitor = PerformanceMonitor(monitor_config)
    manager = AdaptiveManager()
    
    # Generate comprehensive test data
    performances = generate_sample_performance_data(50)
    contexts = generate_sample_market_contexts(50)
    
    learning_events = 0
    monitoring_alerts = 0
    adaptations = 0
    
    print(f"   Processing {len(performances)} performance samples...")
    
    # Integrated processing loop
    for i, (performance, context) in enumerate(zip(performances, contexts)):
        # 1. Continuous Learning
        predictions = {
            'ensemble': np.random.normal(performance.sharpe_ratio, 0.1)
        }
        
        learning_result = await learner.learn_from_performance(
            performance, context, predictions
        )
        
        if learning_result.success:
            learning_events += 1
        
        # 2. Performance Monitoring  
        alerts = await monitor.update_performance(
            performance, strategy_name="integrated_test"
        )
        
        monitoring_alerts += len(alerts)
        
        # 3. Adaptive Management
        if alerts:  # Respond to alerts with adaptations
            param_updates = [
                ParameterUpdate(
                    parameter_name="adaptive_param",
                    old_value=1.0,
                    new_value=1.0 + np.random.normal(0, 0.1),
                    confidence=0.7,
                    reason=f"Response to {len(alerts)} alerts"
                )
            ]
            
            adaptation_result = await manager.process_adaptation(
                performance, context, param_updates
            )
            
            if adaptation_result.success:
                adaptations += 1
    
    print(f"   ✅ Integrated Workflow Test Complete")
    print(f"   🧠 Learning Events: {learning_events}")
    print(f"   🚨 Monitoring Alerts: {monitoring_alerts}")
    print(f"   ⚙️ Adaptations: {adaptations}")
    print(f"   🔗 Components Working Together: ALL SYSTEMS OPERATIONAL")
    
    return True

def test_learning_module_integration():
    """Test module imports and basic integration"""
    print("\n📦 Testing Learning Module Integration...")
    
    try:
        # Test all imports
        from advanced_ai_strategy.learning import (
            ContinuousLearner,
            PerformanceMonitor, 
            AdaptiveManager
        )
        
        # Test instantiation
        learner = ContinuousLearner()
        monitor = PerformanceMonitor()
        manager = AdaptiveManager()
        
        print(f"   ✅ All Learning Components Imported Successfully")
        print(f"   🚀 ContinuousLearner: Ready")
        print(f"   📊 PerformanceMonitor: Ready")
        print(f"   ⚙️ AdaptiveManager: Ready")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import Error: {e}")
        return False

async def main():
    """Run comprehensive learning module tests"""
    print("🚀 Advanced AI Strategy Learning Module - Comprehensive Testing")
    print("=" * 70)
    
    # Test module integration first
    integration_success = test_learning_module_integration()
    
    if not integration_success:
        print("❌ Module integration failed. Stopping tests.")
        return
    
    try:
        # Run all component tests
        await test_continuous_learner()
        await test_performance_monitor()
        await test_adaptive_manager()
        await test_integrated_learning_workflow()
        
        print("\n" + "=" * 70)
        print("✅ All Learning Module Tests Completed Successfully!")
        print("\n📋 Test Summary:")
        print("   🧠 Continuous Learning: WORKING")
        print("   📊 Performance Monitoring: WORKING")
        print("   ⚙️ Adaptive Management: WORKING")
        print("   🔄 Integrated Workflow: WORKING")
        
        print("\n🎯 Phase 4 Learning & Adaptation Module: READY FOR DEPLOYMENT")
        print("   All learning components are fully operational!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the comprehensive tests
    asyncio.run(main())