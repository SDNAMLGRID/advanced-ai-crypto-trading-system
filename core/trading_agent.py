"""
Advanced Trading Agent Module
Main orchestrator for the crypto trading system
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

from .signal_processor import SignalProcessor, TradingSignal, SignalType
from .risk_manager import RiskManager
from .exchange_manager import ExchangeManager
from .logging_config import get_logger

logger = get_logger(__name__)

class TradingAgent:
    """Main trading agent orchestrator"""
    
    def __init__(self, config: Optional[Dict] = None):
        # Load configuration if not provided
        if config is None:
            from config_loader import load_config
            config = load_config()
        
        self.config = config
        self.is_running = False
        self.positions = {}
        self.orders = {}
        
        # Initialize components
        self.signal_processor = None
        self.risk_manager = RiskManager(config)
        self.exchange_manager = ExchangeManager(config)
        
        # Import advanced strategy manager conditionally
        try:
            from core.strategy_manager import get_strategy_manager
            self.strategy_manager = get_strategy_manager()
            logger.info("Advanced strategy manager loaded")
        except ImportError:
            self.strategy_manager = None
            logger.info("Advanced strategies not available - using basic mode")
        
        # Initialize AI analyzer
        try:
            from .ai_analyzer import AIStrategyAnalyzer
            self.ai_analyzer = AIStrategyAnalyzer(config)
            self.signal_processor = SignalProcessor(self.ai_analyzer, config)
            logger.info("AI analyzer initialized")
        except Exception as e:
            logger.warning(f"AI analyzer not available: {e}")
            self.ai_analyzer = None
        
        # Trading parameters
        self.trading_symbols = config.get('trading', {}).get('symbols', ['BTC/USDT'])
        self.check_interval = config.get('trading', {}).get('check_interval', 60)
        
        logger.info("Trading Agent initialized")
    
    async def start(self):
        """Start the trading agent"""
        if self.is_running:
            logger.warning("Trading agent already running")
            return
        
        self.is_running = True
        logger.info("Starting trading agent...")
        
        # Start main loops
        tasks = [
            asyncio.create_task(self._trading_loop()),
            asyncio.create_task(self._monitoring_loop()),
            asyncio.create_task(self._risk_check_loop())
        ]
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the trading agent"""
        if not self.is_running:
            return
        
        logger.info("Stopping trading agent...")
        self.is_running = False
        
        # Close exchange connections
        try:
            await self.exchange_manager.close()
        except Exception as e:
            logger.error(f"Error closing exchange connections: {e}")
        
        logger.info("Trading agent stopped")
    
    async def _trading_loop(self):
        """Main trading loop"""
        logger.info("Starting trading loop")
        
        while self.is_running:
            try:
                for symbol in self.trading_symbols:
                    # Fetch market data
                    market_data = await self._fetch_market_data(symbol)
                    if market_data is None:
                        continue
                    
                    # Generate main signal
                    main_signal = None
                    if self.signal_processor:
                        main_signal = await self.signal_processor.generate_signals(market_data, symbol)
                    
                    # Get advanced strategy signals
                    advanced_signals = []
                    if self.strategy_manager:
                        try:
                            context = {'market_data': market_data, 'symbol': symbol}
                            advanced_signals = await self.strategy_manager.get_combined_signals(context)
                        except Exception as e:
                            logger.error(f"Error getting advanced signals: {e}")
                    
                    # Combine all signals
                    final_signal = await self._combine_all_signals(main_signal, advanced_signals, symbol)
                    
                    # Process signal if generated
                    if final_signal:
                        await self._process_signal(final_signal)
                
                # Wait before next iteration
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _fetch_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch market data for analysis"""
        try:
            ohlcv = await self.exchange_manager.fetch_ohlcv(symbol, '1h', 200)
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    async def _combine_all_signals(self, main_signal: Optional[TradingSignal], 
                                 advanced_signals: List[TradingSignal], 
                                 symbol: str) -> Optional[TradingSignal]:
        """Combine signals from different sources"""
        all_signals = []
        
        if main_signal:
            all_signals.append(main_signal)
        
        all_signals.extend(advanced_signals)
        
        if not all_signals:
            return None
        
        # If only one signal, return it
        if len(all_signals) == 1:
            return all_signals[0]
        
        # Combine multiple signals using weighted approach
        def calculate_weighted_confidence(typed_signals):
            if not typed_signals:
                return 0.0
            
            total_weight = 0.0
            weighted_sum = 0.0
            
            for signal in typed_signals:
                confidence = getattr(signal, 'confidence', 0.5)
                weight = 1.0  # Could be based on signal source
                
                weighted_sum += confidence * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Separate signals by type
        buy_signals = [s for s in all_signals if getattr(s, 'signal_type', None) == SignalType.BUY]
        sell_signals = [s for s in all_signals if getattr(s, 'signal_type', None) == SignalType.SELL]
        
        buy_confidence = calculate_weighted_confidence(buy_signals)
        sell_confidence = calculate_weighted_confidence(sell_signals)
        
        # Determine final signal
        min_confidence = self.config.get('trading', {}).get('min_confidence', 0.6)
        
        if buy_confidence > sell_confidence and buy_confidence >= min_confidence:
            # Create combined buy signal
            best_buy = max(buy_signals, key=lambda x: getattr(x, 'confidence', 0))
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                confidence=buy_confidence,
                price=getattr(best_buy, 'price', 0),
                timestamp=datetime.now(),
                indicators=getattr(best_buy, 'indicators', {}),
                ai_reasoning=f"Combined signal from {len(buy_signals)} buy signals"
            )
        elif sell_confidence > buy_confidence and sell_confidence >= min_confidence:
            # Create combined sell signal
            best_sell = max(sell_signals, key=lambda x: getattr(x, 'confidence', 0))
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                confidence=sell_confidence,
                price=getattr(best_sell, 'price', 0),
                timestamp=datetime.now(),
                indicators=getattr(best_sell, 'indicators', {}),
                ai_reasoning=f"Combined signal from {len(sell_signals)} sell signals"
            )
        
        return None
    
    async def _process_signal(self, signal: TradingSignal):
        """Process a trading signal"""
        try:
            logger.info(f"Processing signal: {signal.symbol} {signal.signal_type.value} "
                       f"confidence={signal.confidence:.2f}")
            
            # Check if we already have position in this symbol
            current_position = self.positions.get(signal.symbol)
            
            # Get portfolio value for position sizing
            balance = await self.exchange_manager.get_account_balance()
            portfolio_value = self._calculate_portfolio_value(balance)
            
            if signal.signal_type == SignalType.BUY:
                if current_position and current_position.get('side') == 'long':
                    logger.info(f"Already long {signal.symbol}, skipping buy signal")
                    return
                
                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(
                    signal, portfolio_value, self.positions
                )
                
                if position_size > 0:
                    await self._open_position(signal, position_size, 'long')
                
            elif signal.signal_type == SignalType.SELL:
                if current_position and current_position.get('side') == 'short':
                    logger.info(f"Already short {signal.symbol}, skipping sell signal")
                    return
                
                # If we have a long position, close it first
                if current_position and current_position.get('side') == 'long':
                    await self._close_position(signal.symbol, current_position, "Signal reversal")
                
                # Calculate position size for short
                position_size = self.risk_manager.calculate_position_size(
                    signal, portfolio_value, self.positions
                )
                
                if position_size > 0:
                    await self._open_position(signal, position_size, 'short')
        
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    async def _open_position(self, signal: TradingSignal, size: float, side: str):
        """Open a new position"""
        try:
            # Calculate risk levels
            stop_loss, take_profit = self.risk_manager.get_risk_levels(
                signal.price, signal.signal_type
            )
            
            # Execute order
            order_params = {
                'symbol': signal.symbol,
                'type': 'market',
                'side': 'buy' if side == 'long' else 'sell',
                'amount': size,
                'price': signal.price
            }
            
            order_result = await self.exchange_manager.execute_order(order_params)
            
            if order_result.get('status') == 'closed':
                # Record position
                self.positions[signal.symbol] = {
                    'symbol': signal.symbol,
                    'side': side,
                    'size': size,
                    'entry_price': signal.price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp': datetime.now(),
                    'order_id': order_result.get('id')
                }
                
                logger.info(f"Opened {side} position for {signal.symbol}: "
                           f"size={size:.4f}, price={signal.price:.2f}")
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
    
    async def _monitoring_loop(self):
        """Monitor existing positions"""
        logger.info("Starting monitoring loop")
        
        while self.is_running:
            try:
                for symbol, position in list(self.positions.items()):
                    # Get current price
                    ticker = await self.exchange_manager.fetch_ticker(symbol)
                    if not ticker:
                        continue
                    
                    current_price = ticker.get('last', 0)
                    if current_price == 0:
                        continue
                    
                    # Check exit conditions
                    should_exit, reason = self._should_exit_position(position, current_price)
                    
                    if should_exit:
                        await self._close_position(symbol, position, reason)
                    else:
                        # Update trailing stop
                        trailing_stop = self.risk_manager.update_trailing_stop(position, current_price)
                        if trailing_stop:
                            position['trailing_stop'] = trailing_stop
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
    
    def _should_exit_position(self, position: Dict, current_price: float) -> tuple[bool, str]:
        """Check if position should be closed"""
        entry_price = position['entry_price']
        side = position['side']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']
        
        # Check stop loss
        if side == 'long' and current_price <= stop_loss:
            return True, "Stop loss triggered"
        elif side == 'short' and current_price >= stop_loss:
            return True, "Stop loss triggered"
        
        # Check take profit
        if side == 'long' and current_price >= take_profit:
            return True, "Take profit triggered"
        elif side == 'short' and current_price <= take_profit:
            return True, "Take profit triggered"
        
        # Check trailing stop
        if 'trailing_stop' in position:
            trailing_stop = position['trailing_stop']
            if side == 'long' and current_price <= trailing_stop:
                return True, "Trailing stop triggered"
            elif side == 'short' and current_price >= trailing_stop:
                return True, "Trailing stop triggered"
        
        return False, ""
    
    async def _close_position(self, symbol: str, position: Dict, reason: str = "Manual close"):
        """Close an existing position"""
        try:
            side = position['side']
            size = position['size']
            
            # Execute closing order
            order_params = {
                'symbol': symbol,
                'type': 'market',
                'side': 'sell' if side == 'long' else 'buy',
                'amount': size
            }
            
            order_result = await self.exchange_manager.execute_order(order_params)
            
            if order_result.get('status') == 'closed':
                # Calculate P&L
                entry_price = position['entry_price']
                exit_price = order_result.get('price', 0)
                
                if side == 'long':
                    pnl = (exit_price - entry_price) * size
                else:
                    pnl = (entry_price - exit_price) * size
                
                # Update risk manager
                self.risk_manager.update_daily_pnl(pnl)
                
                # Remove position
                del self.positions[symbol]
                
                logger.info(f"Closed {side} position for {symbol}: "
                           f"P&L={pnl:.2f}, reason={reason}")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    async def _risk_check_loop(self):
        """Periodic risk checks"""
        logger.info("Starting risk check loop")
        
        while self.is_running:
            try:
                # Get portfolio value
                balance = await self.exchange_manager.get_account_balance()
                portfolio_value = self._calculate_portfolio_value(balance)
                
                # Check risk limits
                violations = self.risk_manager.check_risk_limits(portfolio_value, self.positions)
                
                if violations:
                    logger.warning(f"Risk violations detected: {violations}")
                    
                    # If daily loss limit exceeded, close all positions
                    if 'daily_loss' in violations:
                        logger.critical("Daily loss limit exceeded - closing all positions")
                        await self._close_all_positions()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in risk check loop: {e}")
                await asyncio.sleep(300)
    
    async def _close_all_positions(self):
        """Close all open positions"""
        for symbol, position in list(self.positions.items()):
            await self._close_position(symbol, position, "Risk management")
    
    def _calculate_portfolio_value(self, balance: Dict) -> float:
        """Calculate total portfolio value"""
        total_value = 0.0
        
        if 'total' in balance:
            for currency, amount in balance['total'].items():
                if currency == 'USDT' or currency == 'USD':
                    total_value += amount
                # For other currencies, you'd need to convert to USD
                # This is simplified
        
        return total_value
    
    async def _calculate_total_pnl(self) -> float:
        """Calculate total unrealized P&L"""
        total_pnl = 0.0
        
        for symbol, position in self.positions.items():
            try:
                ticker = await self.exchange_manager.fetch_ticker(symbol)
                if not ticker:
                    continue
                
                current_price = ticker.get('last', 0)
                entry_price = position['entry_price']
                size = position['size']
                side = position['side']
                
                if side == 'long':
                    pnl = (current_price - entry_price) * size
                else:
                    pnl = (entry_price - current_price) * size
                
                total_pnl += pnl
                
            except Exception as e:
                logger.error(f"Error calculating P&L for {symbol}: {e}")
        
        return total_pnl