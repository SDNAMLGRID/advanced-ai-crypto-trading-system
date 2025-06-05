"""
Advanced Exchange Management Module
Handles multiple exchanges, order execution, and account management
"""

import asyncio
import logging
from typing import Dict, List, Optional
import ccxt.async_support as ccxt

from .logging_config import get_logger

logger = get_logger(__name__)

class ExchangeManager:
    """Manages multiple cryptocurrency exchanges"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.exchanges = {}
        self.primary_exchange = None
        
        self._initialize_exchanges()
        
    def _initialize_exchanges(self):
        """Initialize configured exchanges"""
        exchange_configs = self.config.get('exchanges', {})
        
        for exchange_id, exchange_config in exchange_configs.items():
            if exchange_config.get('enabled', False):
                try:
                    exchange = self._create_exchange(exchange_id)
                    if exchange:
                        self.exchanges[exchange_id] = {
                            'client': exchange,
                            'config': exchange_config
                        }
                        
                        if exchange_config.get('primary', False):
                            self.primary_exchange = exchange_id
                            
                        logger.info(f"Initialized exchange: {exchange_id}")
                except Exception as e:
                    logger.error(f"Failed to initialize exchange {exchange_id}: {e}")
        
        if not self.primary_exchange and self.exchanges:
            self.primary_exchange = list(self.exchanges.keys())[0]
            logger.info(f"Set primary exchange to: {self.primary_exchange}")
    
    def _create_exchange(self, exchange_id: str):
        """Create exchange client"""
        exchange_config = self.config['exchanges'][exchange_id]
        
        # Map exchange IDs to ccxt classes
        exchange_classes = {
            'binance': ccxt.binance,
            'coinbase': ccxt.coinbasepro,
            'kraken': ccxt.kraken,
            'bitfinex': ccxt.bitfinex,
            'okx': ccxt.okx,
            'bybit': ccxt.bybit
        }
        
        if exchange_id not in exchange_classes:
            logger.error(f"Unsupported exchange: {exchange_id}")
            return None
        
        try:
            exchange_class = exchange_classes[exchange_id]
            exchange = exchange_class({
                'apiKey': exchange_config.get('api_key'),
                'secret': exchange_config.get('api_secret'),
                'password': exchange_config.get('passphrase'),
                'sandbox': exchange_config.get('sandbox', False),
                'enableRateLimit': True,
                'options': exchange_config.get('options', {})
            })
            
            return exchange
        except Exception as e:
            logger.error(f"Error creating exchange {exchange_id}: {e}")
            return None
    
    async def execute_order(self, order_params: Dict) -> Dict:
        """Execute order on the appropriate exchange"""
        exchange_id = order_params.get('exchange', self.primary_exchange)
        
        if exchange_id not in self.exchanges:
            raise ValueError(f"Exchange {exchange_id} not available")
        
        exchange_info = self.exchanges[exchange_id]
        exchange = exchange_info['client']
        
        try:
            result = await self._execute_on_exchange(exchange, order_params)
            logger.info(f"Order executed on {exchange_id}: {result}")
            return result
        except Exception as e:
            logger.error(f"Order execution failed on {exchange_id}: {e}")
            raise
    
    async def _execute_on_exchange(self, exchange, order_params: Dict) -> Dict:
        """Execute order on specific exchange"""
        # This would contain the actual order execution logic
        # For now, return a mock successful order
        return {
            'id': 'mock_order_123',
            'status': 'closed',
            'symbol': order_params.get('symbol'),
            'type': order_params.get('type'),
            'side': order_params.get('side'),
            'amount': order_params.get('amount'),
            'price': order_params.get('price')
        }
    
    async def get_account_balance(self) -> Dict:
        """Get account balance from primary exchange"""
        if not self.primary_exchange:
            return {}
        
        exchange = self.exchanges[self.primary_exchange]['client']
        try:
            balance = await exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return {}
    
    async def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch ticker data from primary exchange"""
        if not self.primary_exchange:
            return {}
        
        exchange = self.exchanges[self.primary_exchange]['client']
        try:
            ticker = await exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            return {}
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> List:
        """Fetch OHLCV data from primary exchange"""
        if not self.primary_exchange:
            return []
        
        exchange = self.exchanges[self.primary_exchange]['client']
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return []
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book data"""
        if not self.primary_exchange:
            return {}
        
        exchange = self.exchanges[self.primary_exchange]['client']
        try:
            order_book = await exchange.fetch_order_book(symbol, limit)
            return order_book
        except Exception as e:
            logger.error(f"Failed to fetch order book for {symbol}: {e}")
            return {}
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List:
        """Get open orders"""
        if not self.primary_exchange:
            return []
        
        exchange = self.exchanges[self.primary_exchange]['client']
        try:
            orders = await exchange.fetch_open_orders(symbol)
            return orders
        except Exception as e:
            logger.error(f"Failed to fetch open orders: {e}")
            return []
    
    async def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancel an order"""
        if not self.primary_exchange:
            return {}
        
        exchange = self.exchanges[self.primary_exchange]['client']
        try:
            result = await exchange.cancel_order(order_id, symbol)
            logger.info(f"Order {order_id} cancelled")
            return result
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return {}
    
    async def close(self):
        """Close all exchange connections"""
        for exchange_id, exchange_info in self.exchanges.items():
            try:
                await exchange_info['client'].close()
                logger.info(f"Closed connection to {exchange_id}")
            except Exception as e:
                logger.error(f"Error closing {exchange_id}: {e}")
    
    def get_available_exchanges(self) -> List[str]:
        """Get list of available exchanges"""
        return list(self.exchanges.keys())
    
    def is_exchange_available(self, exchange_id: str) -> bool:
        """Check if exchange is available"""
        return exchange_id in self.exchanges