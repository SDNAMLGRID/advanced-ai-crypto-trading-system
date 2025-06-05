"""
Advanced Risk Management Module
Handles position sizing, risk limits, and portfolio protection
"""

import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .logging_config import get_logger

logger = get_logger(__name__)

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class RiskParameters:
    max_position_size: float = 0.02
    max_daily_loss: float = 0.05
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    max_correlation: float = 0.7
    max_leverage: float = 1.0
    max_positions: int = 5
    trailing_stop_enabled: bool = False
    trailing_stop_activation: float = 0.03
    trailing_stop_pct: float = 0.01

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_params = RiskParameters(**config.get('risk_management', {}))
        
        # Track daily P&L
        self.daily_pnl = 0.0
        self.daily_trades = 0
        
        # Position tracking
        self.position_sizes = {}
        self.trailing_stops = {}
        
        logger.info("Risk Manager initialized")
    
    def calculate_position_size(self, signal, portfolio_value: float, current_positions: Dict) -> float:
        """Calculate appropriate position size based on risk parameters"""
        
        base_size = self.risk_params.max_position_size * portfolio_value
        
        # Adjust for signal confidence
        confidence_factor = getattr(signal, 'confidence', 0.5)
        adjusted_size = base_size * confidence_factor
        
        # Adjust for correlation with existing positions
        correlation_factor = self._calculate_correlation_factor(signal.symbol, current_positions)
        adjusted_size *= correlation_factor
        
        # Ensure we don't exceed maximum positions
        if len(current_positions) >= self.risk_params.max_positions:
            return 0.0
        
        # Check daily loss limits
        if self.daily_pnl < -self.risk_params.max_daily_loss * portfolio_value:
            logger.warning("Daily loss limit reached, reducing position size")
            adjusted_size *= 0.5
        
        return max(0, min(adjusted_size, base_size))
    
    def get_risk_levels(self, entry_price: float, signal_type: SignalType) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        
        if signal_type == SignalType.BUY:
            stop_loss = entry_price * (1 - self.risk_params.stop_loss_pct)
            take_profit = entry_price * (1 + self.risk_params.take_profit_pct)
        elif signal_type == SignalType.SELL:
            stop_loss = entry_price * (1 + self.risk_params.stop_loss_pct)
            take_profit = entry_price * (1 - self.risk_params.take_profit_pct)
        else:
            return entry_price, entry_price
        
        return stop_loss, take_profit
    
    def update_trailing_stop(self, position: Dict, current_price: float) -> Optional[float]:
        """Update trailing stop for position"""
        
        if not self.risk_params.trailing_stop_enabled:
            return None
        
        symbol = position['symbol']
        entry_price = position['entry_price']
        side = position['side']
        
        # Calculate unrealized P&L percentage
        if side == 'long':
            unrealized_pnl_pct = (current_price - entry_price) / entry_price
        else:
            unrealized_pnl_pct = (entry_price - current_price) / entry_price
        
        # Activate trailing stop if profit threshold reached
        if unrealized_pnl_pct >= self.risk_params.trailing_stop_activation:
            
            if symbol not in self.trailing_stops:
                # Initialize trailing stop
                if side == 'long':
                    self.trailing_stops[symbol] = current_price * (1 - self.risk_params.trailing_stop_pct)
                else:
                    self.trailing_stops[symbol] = current_price * (1 + self.risk_params.trailing_stop_pct)
                
                logger.info(f"Trailing stop activated for {symbol} at {self.trailing_stops[symbol]}")
            else:
                # Update trailing stop
                if side == 'long':
                    new_stop = current_price * (1 - self.risk_params.trailing_stop_pct)
                    if new_stop > self.trailing_stops[symbol]:
                        self.trailing_stops[symbol] = new_stop
                        logger.info(f"Trailing stop updated for {symbol} to {new_stop}")
                else:
                    new_stop = current_price * (1 + self.risk_params.trailing_stop_pct)
                    if new_stop < self.trailing_stops[symbol]:
                        self.trailing_stops[symbol] = new_stop
                        logger.info(f"Trailing stop updated for {symbol} to {new_stop}")
        
        return self.trailing_stops.get(symbol)
    
    def check_risk_limits(self, portfolio_value: float, positions: Dict) -> Dict[str, str]:
        """Check various risk limits and return violations"""
        violations = {}
        
        # Check daily loss limit
        daily_loss_pct = abs(self.daily_pnl) / portfolio_value if portfolio_value > 0 else 0
        if daily_loss_pct > self.risk_params.max_daily_loss:
            violations['daily_loss'] = f"Daily loss {daily_loss_pct:.2%} exceeds limit {self.risk_params.max_daily_loss:.2%}"
        
        # Check maximum positions
        if len(positions) > self.risk_params.max_positions:
            violations['max_positions'] = f"Position count {len(positions)} exceeds limit {self.risk_params.max_positions}"
        
        # Check individual position sizes
        for symbol, position in positions.items():
            position_value = position.get('value', 0)
            position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
            
            if position_pct > self.risk_params.max_position_size:
                violations[f'position_size_{symbol}'] = f"Position size {position_pct:.2%} exceeds limit {self.risk_params.max_position_size:.2%}"
        
        return violations
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking"""
        self.daily_pnl += pnl
        self.daily_trades += 1
        
        logger.debug(f"Daily P&L updated: {self.daily_pnl:.2f} ({self.daily_trades} trades)")
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of new trading day)"""
        logger.info(f"Resetting daily metrics. Previous day P&L: {self.daily_pnl:.2f}")
        self.daily_pnl = 0.0
        self.daily_trades = 0
    
    def _calculate_correlation_factor(self, symbol: str, positions: Dict) -> float:
        """Calculate correlation factor to reduce position size for correlated assets"""
        
        if not positions:
            return 1.0
        
        # Simplified correlation - in reality would use historical price correlation
        correlation_symbols = {
            'BTC': ['ETH', 'LTC', 'BCH'],
            'ETH': ['BTC', 'LTC', 'ADA'],
            'LTC': ['BTC', 'ETH', 'BCH'],
        }
        
        base_symbol = symbol.split('/')[0] if '/' in symbol else symbol[:3]
        correlated = correlation_symbols.get(base_symbol, [])
        
        correlation_exposure = 0.0
        for pos_symbol in positions.keys():
            pos_base = pos_symbol.split('/')[0] if '/' in pos_symbol else pos_symbol[:3]
            if pos_base in correlated:
                correlation_exposure += 1.0
        
        # Reduce position size if high correlation
        if correlation_exposure >= 2:
            return 0.5  # 50% reduction
        elif correlation_exposure >= 1:
            return 0.75  # 25% reduction
        
        return 1.0  # No reduction