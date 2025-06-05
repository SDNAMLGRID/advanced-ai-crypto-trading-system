"""
Adaptive Strategy Management System
Dynamically adjusts strategy parameters based on market conditions and performance
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import warnings
warnings.filterwarnings('ignore')

from ..core.data_models import StrategyPerformance, MarketContext, MarketRegime

class AdaptationType(Enum):
    """Types of strategy adaptations"""
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    REGIME_ADAPTATION = "regime_adaptation"
    PERFORMANCE_CORRECTION = "performance_correction"
    RISK_ADJUSTMENT = "risk_adjustment"
    MARKET_CONDITION_RESPONSE = "market_condition_response"

class AdaptationPriority(Enum):
    """Priority levels for adaptations"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AdaptationStrategy:
    """Strategy for parameter adaptation"""
    name: str
    description: str
    parameters: Dict[str, Any]
    conditions: Dict[str, Any]
    priority: AdaptationPriority
    enabled: bool = True

@dataclass
class ParameterUpdate:
    """Individual parameter update"""
    parameter_name: str
    old_value: Any
    new_value: Any
    adjustment_factor: float
    reason: str
    confidence: float
    timestamp: datetime

@dataclass
class AdaptationEvent:
    """Record of an adaptation event"""
    event_id: str
    adaptation_type: AdaptationType
    trigger_reason: str
    parameters_updated: List[ParameterUpdate]
    market_context: Optional[MarketContext]
    performance_before: Optional[StrategyPerformance]
    performance_after: Optional[StrategyPerformance] = None
    success: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

class AdaptiveManager:
    """
    Advanced adaptive strategy management system
    
    Features:
    - Dynamic parameter adjustment based on market conditions
    - Performance-driven adaptation strategies
    - Market regime-aware adaptations
    - Risk-based parameter scaling
    - Learning from adaptation history
    - Multi-strategy coordination
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Adaptation strategies
        self.adaptation_strategies = {}
        self._initialize_adaptation_strategies()
        
        # Parameter management
        self.current_parameters = {}
        self.parameter_bounds = {}
        self.parameter_history = {}
        
        # Adaptation history and analytics
        self.adaptation_history = []
        self.adaptation_success_rates = {}
        self.performance_tracking = {}
        
        # Market condition tracking
        self.current_market_context = None
        self.market_history = []
        
        # Risk management
        self.risk_limits = self.config.get('risk_limits', {})
        self.emergency_stops = {}
        
        print(f"🎛️  Adaptive Manager initialized")
        print(f"   Adaptation Strategies: {len(self.adaptation_strategies)}")
        print(f"   Risk Management: {'Enabled' if self.risk_limits else 'Disabled'}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for adaptive manager"""
        return {
            'adaptation_frequency': 300,  # seconds
            'min_performance_history': 20,
            'adaptation_sensitivity': 0.1,
            'risk_limits': {
                'max_drawdown': 0.2,
                'max_volatility': 0.5,
                'min_sharpe': -1.0
            },
            'enable_emergency_stops': True,
            'enable_regime_adaptation': True,
            'enable_performance_adaptation': True
        }
    
    def _initialize_adaptation_strategies(self):
        """Initialize built-in adaptation strategies"""
        
        # Performance degradation adaptation
        self.adaptation_strategies['performance_correction'] = AdaptationStrategy(
            name="Performance Correction",
            description="Adjust parameters when performance degrades",
            parameters={
                'sensitivity': 0.1,
                'adjustment_magnitude': 0.2,
                'cooldown_period': 3600  # 1 hour
            },
            conditions={
                'performance_decline_threshold': 0.15,
                'min_samples': 10
            },
            priority=AdaptationPriority.HIGH
        )
        
        # Market regime adaptation
        self.adaptation_strategies['regime_adaptation'] = AdaptationStrategy(
            name="Market Regime Adaptation",
            description="Adjust parameters based on market regime changes",
            parameters={
                'regime_sensitivity': 0.05,
                'adaptation_speed': 0.3
            },
            conditions={
                'regime_change_threshold': 0.7,
                'confirmation_period': 300  # 5 minutes
            },
            priority=AdaptationPriority.MEDIUM
        )
        
        # Risk management adaptation
        self.adaptation_strategies['risk_adjustment'] = AdaptationStrategy(
            name="Risk Adjustment",
            description="Dynamically adjust risk parameters",
            parameters={
                'volatility_target': 0.15,
                'adjustment_speed': 0.1
            },
            conditions={
                'volatility_breach_threshold': 1.5,
                'drawdown_threshold': 0.1
            },
            priority=AdaptationPriority.CRITICAL
        )
    
    async def update_market_context(self, market_context: MarketContext):
        """Update current market context for adaptation decisions"""
        
        self.current_market_context = market_context
        self.market_history.append({
            'timestamp': datetime.now(),
            'context': market_context
        })
        
        # Trim history to manage memory
        if len(self.market_history) > 1000:
            self.market_history = self.market_history[-500:]
        
        # Check for regime-based adaptations
        if self.config.get('enable_regime_adaptation', True):
            await self._check_regime_adaptations(market_context)
    
    async def update_performance(self, 
                                performance: StrategyPerformance,
                                strategy_name: str = "default"):
        """Update performance metrics and trigger adaptations if needed"""
        
        # Store performance data
        if strategy_name not in self.performance_tracking:
            self.performance_tracking[strategy_name] = []
        
        self.performance_tracking[strategy_name].append({
            'timestamp': datetime.now(),
            'performance': performance
        })
        
        # Trim performance history
        if len(self.performance_tracking[strategy_name]) > 500:
            self.performance_tracking[strategy_name] = self.performance_tracking[strategy_name][-300:]
        
        # Check for performance-based adaptations
        if self.config.get('enable_performance_adaptation', True):
            await self._check_performance_adaptations(performance, strategy_name)
        
        # Emergency risk checks
        if self.config.get('enable_emergency_stops', True):
            await self._check_emergency_conditions(performance, strategy_name)
    
    async def _check_emergency_conditions(self, 
                                        performance: StrategyPerformance,
                                        strategy_name: str):
        """Check for emergency conditions requiring immediate action"""
        
        emergency_triggered = False
        
        # Check maximum drawdown
        if (performance.max_drawdown > self.risk_limits.get('max_drawdown', 1.0) and
            strategy_name not in self.emergency_stops):
            
            emergency_triggered = True
            self.emergency_stops[strategy_name] = {
                'reason': 'max_drawdown_exceeded',
                'value': performance.max_drawdown,
                'timestamp': datetime.now()
            }
            print(f"🚨 EMERGENCY STOP: Max drawdown exceeded for {strategy_name}")
        
        # Check volatility spike (using information_ratio as proxy)
        volatility_proxy = abs(performance.information_ratio)
        if (volatility_proxy > self.risk_limits.get('max_volatility', 1.0) and
            strategy_name not in self.emergency_stops):
            
            emergency_triggered = True
            self.emergency_stops[strategy_name] = {
                'reason': 'volatility_spike',
                'value': volatility_proxy,
                'timestamp': datetime.now()
            }
            print(f"🚨 EMERGENCY STOP: Volatility spike for {strategy_name}")
        
        if emergency_triggered:
            await self._trigger_emergency_adaptation(performance, strategy_name)
    
    async def _check_performance_adaptations(self, 
                                           performance: StrategyPerformance,
                                           strategy_name: str):
        """Check if performance-based adaptations are needed"""
        
        performance_history = self.performance_tracking.get(strategy_name, [])
        
        if len(performance_history) < self.config.get('min_performance_history', 20):
            return
        
        # Calculate performance trend
        recent_performances = [entry['performance'] for entry in performance_history[-10:]]
        baseline_performances = [entry['performance'] for entry in performance_history[-20:-10]]
        
        recent_sharpe = np.mean([p.sharpe_ratio for p in recent_performances])
        baseline_sharpe = np.mean([p.sharpe_ratio for p in baseline_performances])
        
        performance_decline = baseline_sharpe - recent_sharpe
        threshold = self.adaptation_strategies['performance_correction'].conditions['performance_decline_threshold']
        
        if performance_decline > threshold:
            # Performance degradation detected
            adaptation_event = await self._adapt_to_performance_decline(
                performance, performance_decline, strategy_name
            )
            
            if adaptation_event:
                self.adaptation_history.append(adaptation_event)
                print(f"   📉 Performance adaptation triggered: decline of {performance_decline:.3f}")
    
    async def _adapt_to_performance_decline(self, 
                                          performance: StrategyPerformance,
                                          decline_magnitude: float,
                                          strategy_name: str) -> Optional[AdaptationEvent]:
        """Adapt strategy parameters due to performance decline"""
        
        if not self.current_parameters:
            return None
        
        # Performance-based parameter adjustments
        parameter_updates = []
        
        # Reduce position sizes during poor performance
        if 'position_size' in self.current_parameters:
            old_value = self.current_parameters['position_size']
            reduction_factor = max(0.5, 1.0 - decline_magnitude)
            new_value = old_value * reduction_factor
            
            # Apply bounds
            if 'position_size' in self.parameter_bounds:
                min_val, max_val = self.parameter_bounds['position_size']
                new_value = max(min_val, min(max_val, new_value))
            
            update = ParameterUpdate(
                parameter_name='position_size',
                old_value=old_value,
                new_value=new_value,
                adjustment_factor=reduction_factor,
                reason=f"performance_decline_{decline_magnitude:.3f}",
                confidence=0.7,
                timestamp=datetime.now()
            )
            
            parameter_updates.append(update)
            self.current_parameters['position_size'] = new_value
        
        # Tighten stop losses
        if 'stop_loss_threshold' in self.current_parameters:
            old_value = self.current_parameters['stop_loss_threshold']
            tightening_factor = max(0.7, 1.0 - decline_magnitude * 0.5)
            new_value = old_value * tightening_factor
            
            update = ParameterUpdate(
                parameter_name='stop_loss_threshold',
                old_value=old_value,
                new_value=new_value,
                adjustment_factor=tightening_factor,
                reason="performance_decline_risk_reduction",
                confidence=0.8,
                timestamp=datetime.now()
            )
            
            parameter_updates.append(update)
            self.current_parameters['stop_loss_threshold'] = new_value
        
        if parameter_updates:
            return AdaptationEvent(
                event_id=self._generate_event_id(),
                adaptation_type=AdaptationType.PERFORMANCE_CORRECTION,
                trigger_reason=f"Performance decline: {decline_magnitude:.3f}",
                parameters_updated=parameter_updates,
                market_context=self.current_market_context,
                performance_before=performance,
                timestamp=datetime.now()
            )
        
        return None
    
    async def _trigger_emergency_adaptation(self, 
                                          performance: StrategyPerformance,
                                          strategy_name: str):
        """Trigger emergency adaptations for critical conditions"""
        
        emergency_adjustments = {
            'position_size': 0.5,  # Halve position sizes
            'risk_tolerance': 0.3,  # Significantly reduce risk
            'stop_loss_threshold': 0.5  # Tighten stop losses
        }
        
        parameter_updates = []
        
        for param_name, reduction_factor in emergency_adjustments.items():
            if param_name in self.current_parameters:
                old_value = self.current_parameters[param_name]
                new_value = old_value * reduction_factor
                
                update = ParameterUpdate(
                    parameter_name=param_name,
                    old_value=old_value,
                    new_value=new_value,
                    adjustment_factor=reduction_factor,
                    reason="emergency_risk_reduction",
                    confidence=1.0,
                    timestamp=datetime.now()
                )
                
                parameter_updates.append(update)
                self.current_parameters[param_name] = new_value
        
        if parameter_updates:
            adaptation_event = AdaptationEvent(
                event_id=self._generate_event_id(),
                adaptation_type=AdaptationType.RISK_ADJUSTMENT,
                trigger_reason="Emergency risk conditions detected",
                parameters_updated=parameter_updates,
                market_context=self.current_market_context,
                performance_before=performance,
                timestamp=datetime.now()
            )
            
            self.adaptation_history.append(adaptation_event)
            print(f"   🚨 Emergency adaptation applied for {strategy_name}")
    
    async def _check_regime_adaptations(self, market_context: MarketContext):
        """Check if regime-based adaptations are needed"""
        
        if len(self.market_history) < 5:
            return
        
        # Detect regime changes
        recent_regimes = [entry['context'].current_regime for entry in self.market_history[-5:]]
        
        # Check for regime consistency/change
        unique_regimes = set(recent_regimes)
        
        if len(unique_regimes) > 1:
            # Regime transition detected
            most_recent_regime = recent_regimes[-1]
            adaptation_event = await self._adapt_to_regime(
                most_recent_regime, market_context
            )
            
            if adaptation_event:
                self.adaptation_history.append(adaptation_event)
                print(f"   🎯 Regime adaptation triggered: {most_recent_regime}")
    
    async def _adapt_to_regime(self, 
                             regime: MarketRegime,
                             market_context: MarketContext) -> Optional[AdaptationEvent]:
        """Adapt strategy parameters to new market regime"""
        
        if not self.current_parameters:
            return None
        
        # Simplified regime adaptations for demo
        return None
    
    def set_strategy_parameters(self, 
                               parameters: Dict[str, Any],
                               parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        """Set current strategy parameters and bounds"""
        
        self.current_parameters = parameters.copy()
        
        if parameter_bounds:
            self.parameter_bounds = parameter_bounds.copy()
        
        print(f"   📝 Strategy parameters updated: {len(parameters)} parameters")
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters"""
        return self.current_parameters.copy()
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive adaptation statistics"""
        
        if not self.adaptation_history:
            return {'no_adaptations': True}
        
        # Calculate success rates by adaptation type
        type_stats = {}
        for adaptation_type in AdaptationType:
            adaptations = [
                event for event in self.adaptation_history
                if event.adaptation_type == adaptation_type
            ]
            
            if adaptations:
                successful = len([event for event in adaptations if event.success])
                type_stats[adaptation_type.value] = {
                    'total': len(adaptations),
                    'successful': successful,
                    'success_rate': successful / len(adaptations)
                }
        
        return {
            'total_adaptations': len(self.adaptation_history),
            'adaptation_types': type_stats,
            'emergency_stops': len(self.emergency_stops),
            'active_strategies': len(self.adaptation_strategies),
            'current_parameters': len(self.current_parameters)
        }
    
    def get_parameter_history(self, parameter_name: str) -> List[Dict[str, Any]]:
        """Get history of changes for a specific parameter"""
        
        history = []
        
        for event in self.adaptation_history:
            for update in event.parameters_updated:
                if update.parameter_name == parameter_name:
                    history.append({
                        'timestamp': update.timestamp,
                        'old_value': update.old_value,
                        'new_value': update.new_value,
                        'reason': update.reason,
                        'confidence': update.confidence,
                        'adaptation_event_id': event.event_id
                    })
        
        return sorted(history, key=lambda x: x['timestamp'])
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        return f"ADAPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.adaptation_history):04d}"