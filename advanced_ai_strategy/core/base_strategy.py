"""
Abstract base class for all trading strategies
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
from .data_models import MarketContext, StrategySignal, MarketRegime

class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, parameters: Dict):
        self.parameters = parameters
        self.name = self.__class__.__name__
        self.is_active = False
        
    @abstractmethod
    async def generate_signals(self, 
                              market_data: pd.DataFrame, 
                              context: MarketContext) -> List[StrategySignal]:
        """Generate trading signals based on market data and context"""
        pass
        
    @abstractmethod
    def validate_parameters(self) -> bool:
        """Validate strategy parameters"""
        pass
        
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, tuple]:
        """Get parameter optimization bounds"""
        pass
        
    def calculate_confidence(self, market_data: pd.DataFrame, 
                           context: MarketContext) -> float:
        """Calculate strategy confidence based on market conditions"""
        # Default implementation - override in specific strategies
        if context.current_regime in self.get_optimal_regimes():
            return 0.8
        return 0.4
        
    def get_optimal_regimes(self) -> List[MarketRegime]:
        """Get market regimes where this strategy performs best"""
        return [MarketRegime.BULL_TRENDING, MarketRegime.BEAR_TRENDING]
        
    def calculate_position_size(self, confidence: float, 
                               risk_budget: float) -> float:
        """Calculate position size using Kelly criterion"""
        # Simplified Kelly criterion implementation
        win_rate = self.parameters.get('historical_win_rate', 0.55)
        avg_win = self.parameters.get('avg_win', 0.02)
        avg_loss = self.parameters.get('avg_loss', 0.01)
        
        if avg_loss > 0:
            b = avg_win / avg_loss  # Win/loss ratio
            kelly_fraction = (win_rate * b - (1 - win_rate)) / b
            
            # Apply safety factor and confidence adjustment
            safety_factor = 0.25  # Conservative Kelly
            position_size = kelly_fraction * safety_factor * confidence * risk_budget
            
            return max(0, min(position_size, risk_budget))
        
        return risk_budget * 0.01  # Default 1% risk
        
    def _calculate_stop_loss(self, market_data: pd.DataFrame, direction: str) -> float:
        """Calculate stop loss based on ATR"""
        current_price = market_data['close'].iloc[-1]
        
        # Calculate ATR (Average True Range)
        high_low = market_data['high'] - market_data['low']
        high_close = abs(market_data['high'] - market_data['close'].shift())
        low_close = abs(market_data['low'] - market_data['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # Set stop loss at configurable ATR multiple
        atr_multiplier = self.parameters.get('stop_loss_atr_multiplier', 2.0)
        stop_distance = atr * atr_multiplier
        
        if direction == 'BUY':
            return current_price - stop_distance
        else:
            return current_price + stop_distance
            
    def _calculate_take_profit(self, market_data: pd.DataFrame, direction: str) -> float:
        """Calculate take profit based on risk-reward ratio"""
        current_price = market_data['close'].iloc[-1]
        stop_loss = self._calculate_stop_loss(market_data, direction)
        
        risk_amount = abs(current_price - stop_loss)
        reward_ratio = self.parameters.get('risk_reward_ratio', 2.0)
        
        if direction == 'BUY':
            return current_price + (risk_amount * reward_ratio)
        else:
            return current_price - (risk_amount * reward_ratio)
            
    def __str__(self) -> str:
        return f"{self.name}({self.parameters})"
        
    def __repr__(self) -> str:
        return self.__str__()