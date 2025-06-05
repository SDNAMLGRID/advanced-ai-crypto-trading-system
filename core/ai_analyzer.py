"""
AI Strategy Analyzer Module
Uses OpenAI to analyze market conditions and provide trading insights
"""

import asyncio
import json
import os
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd
from openai import AsyncOpenAI

from .logging_config import get_logger

logger = get_logger(__name__)

class AIStrategyAnalyzer:
    """Uses OpenAI to analyze market conditions and refine strategies"""
    
    def __init__(self, config: Dict):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OPENAI_API_KEY environment variable not set - AI analysis disabled")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=api_key)
        
        self.config = config.get('ai_analysis', {})
        self.analysis_cache = {}
        
    async def analyze_market_conditions(self, market_data: pd.DataFrame, 
                                      technical_indicators: Dict) -> Dict:
        """Analyze current market conditions using AI"""
        
        if not self.client:
            logger.warning("OpenAI client not available - returning empty analysis")
            return {}
        
        # Prepare context for AI
        context = {
            "price_action": {
                "current_price": float(market_data['close'].iloc[-1]),
                "24h_change": float((market_data['close'].iloc[-1] / market_data['close'].iloc[-24] - 1) * 100),
                "volatility": float(market_data['close'].pct_change().std() * np.sqrt(24)),
                "volume_profile": float(market_data['volume'].iloc[-24:].mean())
            },
            "technical_indicators": technical_indicators,
            "market_structure": self._analyze_market_structure(market_data)
        }
        
        prompt = f"""
        Analyze the following cryptocurrency market data and provide trading insights:
        
        Market Data:
        {json.dumps(context, indent=2)}
        
        Please provide:
        1. Market regime (trending/ranging/volatile)
        2. Key support and resistance levels
        3. Recommended trading strategy adjustments
        4. Risk factors to consider
        5. Confidence level (0-1) for any trading signals
        
        Format your response as JSON.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.get('model', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": "You are an expert cryptocurrency trading analyst with deep knowledge of technical analysis and market dynamics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.get('temperature', 0.2),
                max_tokens=self.config.get('max_tokens', 1500),
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return {}
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze market structure patterns"""
        closes = df['close'].values
        
        # Find recent highs and lows
        highs = []
        lows = []
        
        for i in range(10, len(closes) - 10):
            if closes[i] == max(closes[i-10:i+11]):
                highs.append((i, closes[i]))
            if closes[i] == min(closes[i-10:i+11]):
                lows.append((i, closes[i]))
        
        return {
            "recent_highs": highs[-3:] if highs else [],
            "recent_lows": lows[-3:] if lows else [],
            "trend_strength": self._calculate_trend_strength(closes)
        }
    
    def _calculate_trend_strength(self, prices: np.array) -> float:
        """Calculate trend strength using linear regression"""
        if len(prices) < 2:
            return 0.0
        
        x = np.arange(len(prices))
        coeffs = np.polyfit(x, prices, 1)
        
        # Normalize by price volatility
        trend_strength = coeffs[0] / (np.std(prices) + 1e-8)
        return float(np.clip(trend_strength, -1, 1))