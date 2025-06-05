"""
Advanced Signal Processing Module
Handles signal generation and combination from multiple sources
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from collections import deque
from enum import Enum

from utils.indicators import TechnicalIndicators, IndicatorConfig
from .logging_config import get_logger

logger = get_logger(__name__)

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class TradingSignal:
    """Trading signal data structure"""
    def __init__(self, symbol: str, signal_type: SignalType, confidence: float, 
                 price: float, timestamp: datetime, indicators: Dict, ai_reasoning: str):
        self.symbol = symbol
        self.signal_type = signal_type
        self.confidence = confidence
        self.price = price
        self.timestamp = timestamp
        self.indicators = indicators
        self.ai_reasoning = ai_reasoning

class SignalProcessor:
    """Processes multiple data sources to generate trading signals"""
    
    def __init__(self, ai_analyzer, config: Dict):
        self.ai_analyzer = ai_analyzer
        self.config = config
        self.signal_history = deque(maxlen=100)
        
        # Initialize technical indicators
        indicator_config = IndicatorConfig(**config.get('technical_indicators', {}))
        self.technical_indicators = TechnicalIndicators(indicator_config)
        
    async def generate_signals(self, market_data: pd.DataFrame, 
                             symbol: str) -> Optional[TradingSignal]:
        """Generate trading signals from market data"""
        
        # Calculate technical indicators
        indicators = self.technical_indicators.calculate_all(market_data)
        
        # Get AI analysis
        ai_analysis = await self.ai_analyzer.analyze_market_conditions(
            market_data, self._format_indicators_for_ai(indicators)
        )
        
        # Combine signals
        signal = self._combine_signals(indicators, ai_analysis, market_data)
        
        if signal and signal['confidence'] >= self.config.get('trading', {}).get('min_confidence', 0.6):
            trading_signal = TradingSignal(
                symbol=symbol,
                signal_type=signal['type'],
                confidence=signal['confidence'],
                price=float(market_data['close'].iloc[-1]),
                timestamp=datetime.now(),
                indicators=indicators,
                ai_reasoning=ai_analysis.get('reasoning', '')
            )
            
            self.signal_history.append(trading_signal)
            return trading_signal
        
        return None
    
    def _format_indicators_for_ai(self, indicators: Dict) -> Dict:
        """Format indicators for AI analysis"""
        formatted = {}
        for key, value in indicators.items():
            if isinstance(value, (int, float, np.number)):
                formatted[key] = float(value)
            elif hasattr(value, 'iloc') and len(value) > 0:
                formatted[key] = float(value.iloc[-1])
        return formatted
    
    def _combine_signals(self, indicators: Dict, ai_analysis: Dict, 
                        market_data: pd.DataFrame) -> Optional[Dict]:
        """Combine various signal sources into final signal"""
        signals = []
        
        # Technical indicator signals
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if hasattr(rsi, 'iloc'):
                rsi_value = rsi.iloc[-1] if len(rsi) > 0 else 50
            else:
                rsi_value = rsi
                
            if rsi_value < 30:
                signals.append(('BUY', 0.7, 'RSI oversold'))
            elif rsi_value > 70:
                signals.append(('SELL', 0.7, 'RSI overbought'))
        
        # MACD signals
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd']
            macd_signal = indicators['macd_signal']
            
            if hasattr(macd, 'iloc') and hasattr(macd_signal, 'iloc'):
                if len(macd) > 1 and len(macd_signal) > 1:
                    if macd.iloc[-1] > macd_signal.iloc[-1] and macd.iloc[-2] <= macd_signal.iloc[-2]:
                        signals.append(('BUY', 0.6, 'MACD bullish crossover'))
                    elif macd.iloc[-1] < macd_signal.iloc[-1] and macd.iloc[-2] >= macd_signal.iloc[-2]:
                        signals.append(('SELL', 0.6, 'MACD bearish crossover'))
        
        # Moving average signals
        if 'sma_20' in indicators and 'sma_50' in indicators:
            sma_20 = indicators['sma_20']
            sma_50 = indicators['sma_50']
            
            if hasattr(sma_20, 'iloc') and hasattr(sma_50, 'iloc'):
                if len(sma_20) > 0 and len(sma_50) > 0:
                    if sma_20.iloc[-1] > sma_50.iloc[-1]:
                        signals.append(('BUY', 0.5, 'Golden cross'))
                    else:
                        signals.append(('SELL', 0.5, 'Death cross'))
        
        # AI analysis signals
        if ai_analysis:
            confidence = ai_analysis.get('confidence', 0.5)
            if confidence > 0.6:
                recommendation = ai_analysis.get('recommendation', 'HOLD')
                if recommendation in ['BUY', 'SELL']:
                    signals.append((recommendation, confidence, 'AI analysis'))
        
        # Combine signals
        if not signals:
            return None
        
        strategy_weights = self.config.get('strategy_weights', {})
        
        buy_confidence = 0
        sell_confidence = 0
        
        for signal_type, confidence, reason in signals:
            weight = self._get_signal_weight(reason, strategy_weights)
            if signal_type == 'BUY':
                buy_confidence += confidence * weight
            elif signal_type == 'SELL':
                sell_confidence += confidence * weight
        
        # Normalize confidences
        total_signals = len(signals)
        if total_signals > 0:
            buy_confidence /= total_signals
            sell_confidence /= total_signals
        
        # Determine final signal
        if buy_confidence > sell_confidence and buy_confidence > 0.5:
            return {
                'type': SignalType.BUY,
                'confidence': buy_confidence,
                'reasoning': f'Combined signal: {signals}'
            }
        elif sell_confidence > buy_confidence and sell_confidence > 0.5:
            return {
                'type': SignalType.SELL,
                'confidence': sell_confidence,
                'reasoning': f'Combined signal: {signals}'
            }
        
        return None
    
    def _get_signal_weight(self, signal_name: str, strategy_weights: Dict) -> float:
        """Get weight for specific signal type"""
        signal_name_lower = signal_name.lower()
        
        for key, weight in strategy_weights.items():
            if key.lower() in signal_name_lower:
                return weight
        
        return 1.0  # Default weight