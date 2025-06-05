"""
Core data models for the advanced AI strategy system
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending" 
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"

class StrategyType(Enum):
    """Trading strategy types"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"

@dataclass
class MarketContext:
    """Comprehensive market context for strategy decisions"""
    symbol: str
    timeframe: str
    current_regime: MarketRegime
    volatility_rank: float  # 0-100 percentile
    trend_strength: float   # -1 to 1
    liquidity_score: float  # 0-100
    correlation_matrix: Dict[str, float] = field(default_factory=dict)
    macro_indicators: Dict[str, float] = field(default_factory=dict)
    sentiment_scores: Dict[str, float] = field(default_factory=dict)
    volume_profile: Dict[str, float] = field(default_factory=dict)
    on_chain_metrics: Dict[str, float] = field(default_factory=dict)
    event_calendar: List[Dict] = field(default_factory=list)

@dataclass
class StrategySignal:
    """Trading signal from a strategy"""
    signal_type: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    price: float
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyRecommendation:
    """AI strategy recommendation"""
    strategy_type: StrategyType
    confidence: float
    parameters: Dict[str, Any]
    risk_allocation: float  # Percentage of capital
    entry_conditions: List[str]
    exit_conditions: List[str]
    stop_loss_method: str
    take_profit_method: str
    expected_return: float
    expected_risk: float
    reasoning: str
    market_conditions: List[str]

@dataclass
class StrategyPerformance:
    """Detailed strategy performance metrics"""
    strategy_id: str
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_win: float
    avg_loss: float
    avg_trade_duration: float
    best_market_regime: MarketRegime
    worst_market_regime: MarketRegime
    confidence_correlation: float  # Correlation between AI confidence and actual performance
    risk_adjusted_return: float
    information_ratio: float
    calmar_ratio: float 