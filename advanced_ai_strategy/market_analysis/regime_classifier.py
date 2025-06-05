"""
Market regime classification using ensemble methods
Extracted and enhanced from the monolithic Advanced AI Strategy system
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats
from ..core.data_models import MarketRegime

class MarketRegimeClassifier:
    """
    Classify market regimes using multiple indicators and ensemble methods
    
    Supports 7 regime types:
    - BULL_TRENDING: Strong upward price movement
    - BEAR_TRENDING: Strong downward price movement
    - RANGING: Sideways movement within bounds
    - HIGH_VOLATILITY: High price fluctuations
    - LOW_VOLATILITY: Low price fluctuations
    - ACCUMULATION: Building positions, increasing volume
    - DISTRIBUTION: Selling positions, decreasing volume
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.lookback_period = self.config.get('regime_lookback', 100)
        
    async def classify_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Classify current market regime using ensemble methods"""
        if len(market_data) < self.lookback_period:
            return MarketRegime.RANGING
            
        # Extract regime features
        features = self._extract_regime_features(market_data)
        
        # Multi-factor regime determination
        trend_regime = self._classify_trend(features)
        volatility_regime = self._classify_volatility(features)
        structure_regime = self._classify_structure(features)
        
        # Combine classifications with weights
        regime_scores = {
            MarketRegime.BULL_TRENDING: 0,
            MarketRegime.BEAR_TRENDING: 0,
            MarketRegime.RANGING: 0,
            MarketRegime.HIGH_VOLATILITY: 0,
            MarketRegime.LOW_VOLATILITY: 0,
            MarketRegime.ACCUMULATION: 0,
            MarketRegime.DISTRIBUTION: 0
        }
        
        # Weight trend classification (40%)
        regime_scores[trend_regime] += 0.4
        
        # Weight volatility classification (30%)
        regime_scores[volatility_regime] += 0.3
        
        # Weight structure classification (30%)
        regime_scores[structure_regime] += 0.3
        
        # Return regime with highest score
        return max(regime_scores.keys(), key=lambda k: regime_scores[k])
        
    def _extract_regime_features(self, df: pd.DataFrame) -> Dict:
        """Extract comprehensive features for regime classification"""
        close = df['close'].values
        returns = df['close'].pct_change().dropna()
        
        # Trend indicators
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        sma_200 = df['close'].rolling(200).mean()
        
        # Calculate trend strength
        trend_strength = 0
        if len(df) > 200:
            if close[-1] > sma_20.iloc[-1] > sma_50.iloc[-1] > sma_200.iloc[-1]:
                trend_strength = 1  # Strong uptrend
            elif close[-1] < sma_20.iloc[-1] < sma_50.iloc[-1] < sma_200.iloc[-1]:
                trend_strength = -1  # Strong downtrend
            else:
                trend_strength = (close[-1] - sma_200.iloc[-1]) / sma_200.iloc[-1]
        
        # Volatility analysis
        volatility = returns.std() * np.sqrt(252)  # Annualized
        historical_volatility = returns.rolling(252).std() * np.sqrt(252)
        volatility_rank = stats.percentileofscore(
            historical_volatility.dropna(), 
            volatility
        ) if len(historical_volatility.dropna()) > 0 else 50
        
        # Market structure analysis
        higher_highs = 0
        lower_lows = 0
        
        for i in range(1, min(20, len(df))):
            if df['high'].iloc[-i] > df['high'].iloc[-i-1]:
                higher_highs += 1
            if df['low'].iloc[-i] < df['low'].iloc[-i-1]:
                lower_lows += 1
        
        # Volume analysis
        volume_trend = np.polyfit(range(20), df['volume'].tail(20).values, 1)[0] if len(df) >= 20 else 0
        
        # Price efficiency calculation
        price_efficiency = self._calculate_price_efficiency(df)
        
        # Fractal dimension
        fractal_dimension = self._calculate_fractal_dimension(close[-100:]) if len(close) >= 100 else 1.5
        
        return {
            'trend_strength': trend_strength,
            'volatility': volatility,
            'volatility_rank': volatility_rank,
            'higher_highs': higher_highs,
            'lower_lows': lower_lows,
            'volume_trend': volume_trend,
            'price_efficiency': price_efficiency,
            'fractal_dimension': fractal_dimension
        }
    
    def _classify_trend(self, features: Dict) -> MarketRegime:
        """Classify trend regime based on trend strength and structure"""
        trend_strength = features['trend_strength']
        
        if trend_strength > 0.1:
            return MarketRegime.BULL_TRENDING
        elif trend_strength < -0.1:
            return MarketRegime.BEAR_TRENDING
        else:
            return MarketRegime.RANGING
            
    def _classify_volatility(self, features: Dict) -> MarketRegime:
        """Classify volatility regime"""
        vol_percentile = features['volatility_rank']
        
        if vol_percentile > 80:
            return MarketRegime.HIGH_VOLATILITY
        elif vol_percentile < 20:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.RANGING
            
    def _classify_structure(self, features: Dict) -> MarketRegime:
        """Classify market structure regime"""
        trend_strength = features['trend_strength']
        volume_trend = features['volume_trend']
        price_efficiency = features['price_efficiency']
        
        # Accumulation: positive volume trend with moderate upward price movement
        if volume_trend > 0 and 0.1 < trend_strength < 0.5 and price_efficiency > 0.6:
            return MarketRegime.ACCUMULATION
            
        # Distribution: positive volume trend with moderate downward price movement
        elif volume_trend > 0 and -0.5 < trend_strength < -0.1 and price_efficiency > 0.6:
            return MarketRegime.DISTRIBUTION
            
        # Default to ranging if no clear structure pattern
        else:
            return MarketRegime.RANGING
    
    def _calculate_price_efficiency(self, df: pd.DataFrame) -> float:
        """Calculate market price efficiency (0-1, higher = more efficient)"""
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < 10:
            return 0.7  # Default assumption
        
        # Hurst exponent calculation (simplified)
        h = self._calculate_hurst_exponent(returns)
        
        # Efficiency ratio: closer to 0.5 (random walk) = more efficient
        efficiency = 1 - abs(h - 0.5) * 2
        
        return max(0, min(1, efficiency))
    
    def _calculate_hurst_exponent(self, returns: pd.Series) -> float:
        """Calculate Hurst exponent for market efficiency analysis"""
        try:
            # Simplified implementation of Hurst exponent
            lags = range(2, min(20, len(returns) // 2))
            tau = []
            
            for lag in lags:
                # Calculate variance of returns over different lags
                lag_returns = returns.rolling(lag).sum().dropna()
                if len(lag_returns) > 0:
                    tau.append(np.sqrt(np.var(lag_returns)))
                else:
                    tau.append(0)
            
            if len(tau) > 1 and np.std(np.log(lags)) > 0:
                # Linear regression to find Hurst exponent
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2
            else:
                return 0.5  # Random walk default
                
        except Exception:
            return 0.5  # Default to random walk if calculation fails
    
    def _calculate_fractal_dimension(self, prices: np.array) -> float:
        """Calculate fractal dimension of price series"""
        if len(prices) < 10:
            return 1.5  # Default for financial time series
            
        try:
            # Simplified box-counting method
            max_price = np.max(prices)
            min_price = np.min(prices)
            
            if max_price == min_price:
                return 1.0
            
            # Normalize prices
            normalized = (prices - min_price) / (max_price - min_price)
            
            # Calculate fractal dimension using correlation dimension approach
            # This is a simplified version - full implementation would be more complex
            fd = 1.5  # Default for financial time series
            
            # Estimate based on price variability
            price_variability = np.std(normalized)
            if price_variability > 0.1:
                fd = 1.3  # More chaotic
            elif price_variability < 0.05:
                fd = 1.7  # More structured
            
            return fd
            
        except Exception:
            return 1.5  # Default value
    
    def get_regime_probabilities(self, market_data: pd.DataFrame) -> Dict[MarketRegime, float]:
        """Get probabilities for each regime type"""
        if len(market_data) < self.lookback_period:
            # Return uniform probabilities if insufficient data
            return {regime: 1/7 for regime in MarketRegime}
            
        features = self._extract_regime_features(market_data)
        
        # Calculate weighted scores for each regime
        regime_scores = {}
        
        # Trend-based scores
        trend_strength = features['trend_strength']
        regime_scores[MarketRegime.BULL_TRENDING] = max(0, trend_strength) * 0.4
        regime_scores[MarketRegime.BEAR_TRENDING] = max(0, -trend_strength) * 0.4
        regime_scores[MarketRegime.RANGING] = (1 - abs(trend_strength)) * 0.4
        
        # Volatility-based scores
        vol_rank = features['volatility_rank'] / 100
        regime_scores[MarketRegime.HIGH_VOLATILITY] = vol_rank * 0.3
        regime_scores[MarketRegime.LOW_VOLATILITY] = (1 - vol_rank) * 0.3
        
        # Structure-based scores
        volume_trend = features['volume_trend']
        if volume_trend > 0 and trend_strength > 0.1:
            regime_scores[MarketRegime.ACCUMULATION] = 0.3
        elif volume_trend > 0 and trend_strength < -0.1:
            regime_scores[MarketRegime.DISTRIBUTION] = 0.3
        
        # Normalize to probabilities
        total_score = sum(regime_scores.values())
        if total_score > 0:
            return {regime: score / total_score for regime, score in regime_scores.items()}
        else:
            return {regime: 1/7 for regime in MarketRegime}
    
    def analyze_regime_stability(self, market_data: pd.DataFrame, periods: int = 10) -> Dict:
        """Analyze how stable the current regime is over recent periods"""
        if len(market_data) < periods * 2:
            return {'stability': 0.5, 'regime_changes': 0, 'dominant_regime': MarketRegime.RANGING}
        
        recent_regimes = []
        
        # Analyze regime over rolling windows
        for i in range(periods):
            window_data = market_data.iloc[-(periods - i):]
            if len(window_data) >= self.lookback_period:
                regime = self._get_regime_sync(window_data)
                recent_regimes.append(regime)
        
        if not recent_regimes:
            return {'stability': 0.5, 'regime_changes': 0, 'dominant_regime': MarketRegime.RANGING}
        
        # Calculate stability metrics
        regime_changes = sum(1 for i in range(1, len(recent_regimes)) 
                           if recent_regimes[i] != recent_regimes[i-1])
        
        stability = 1 - (regime_changes / max(1, len(recent_regimes) - 1))
        
        # Find dominant regime
        regime_counts = {}
        for regime in recent_regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        dominant_regime = max(regime_counts.keys(), key=lambda k: regime_counts[k])
        
        return {
            'stability': stability,
            'regime_changes': regime_changes,
            'dominant_regime': dominant_regime,
            'regime_history': recent_regimes
        }
    
    def _get_regime_sync(self, market_data: pd.DataFrame) -> MarketRegime:
        """Synchronous version for stability analysis"""
        # Simplified synchronous version of classify_regime
        if len(market_data) < 20:
            return MarketRegime.RANGING
            
        features = self._extract_regime_features(market_data)
        
        trend_strength = features['trend_strength']
        vol_rank = features['volatility_rank']
        
        if abs(trend_strength) > 0.7 and vol_rank < 70:
            return MarketRegime.BULL_TRENDING if trend_strength > 0 else MarketRegime.BEAR_TRENDING
        elif vol_rank > 80:
            return MarketRegime.HIGH_VOLATILITY
        elif vol_rank < 20:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.RANGING