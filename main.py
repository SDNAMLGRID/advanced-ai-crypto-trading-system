#!/usr/bin/env python3
"""
Advanced AI Crypto Trading System - Main Entry Point

This is the primary entry point for the trading system.
Run this file to start the complete trading system with all components.
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('TradingSystem')

# Import system components
from advanced_ai_strategy import (
    BaseStrategy,
    registry,
    MarketRegimeClassifier,
    BayesianOptimizer,
    ContinuousLearner,
    PerformanceMonitor,
    AdaptiveManager
)

class TradingSystemController:
    """Main controller for the AI trading system"""
    
    def __init__(self):
        self.running = False
        self.classifier = MarketRegimeClassifier()
        self.learner = ContinuousLearner()
        self.monitor = PerformanceMonitor()
        self.manager = AdaptiveManager()
        self.strategies = {}
        
    async def initialize(self):
        """Initialize the trading system"""
        logger.info("🚀 Initializing Advanced AI Trading System...")
        
        # Register example strategies
        await self._register_strategies()
        
        # Initialize components
        logger.info("⚙️ System components initialized")
        logger.info(f"📊 Market analysis: Ready")
        logger.info(f"🧠 Learning system: Ready")
        logger.info(f"📊 Monitoring system: Ready")
        logger.info(f"⚙️ Adaptive management: Ready")
        
    async def _register_strategies(self):
        """Register available trading strategies"""
        # This would be where you register your actual strategies
        logger.info("📁 Strategy registration complete")
        
    async def start(self):
        """Start the trading system"""
        await self.initialize()
        
        self.running = True
        logger.info("✅ Trading system started successfully")
        
        try:
            while self.running:
                await self._trading_loop()
                await asyncio.sleep(60)  # 1-minute cycle
                
        except KeyboardInterrupt:
            logger.info("⚠️ Shutdown signal received")
        except Exception as e:
            logger.error(f"❌ System error: {e}")
        finally:
            await self.shutdown()
    
    async def _trading_loop(self):
        """Main trading loop"""
        try:
            # 1. Market Analysis
            # In a real implementation, you would:
            # - Fetch current market data
            # - Classify market regime
            # - Update market context
            
            # 2. Strategy Execution
            # - Generate signals from active strategies
            # - Execute trades based on signals
            # - Update positions
            
            # 3. Performance Monitoring
            # - Track strategy performance
            # - Generate alerts if needed
            # - Update performance metrics
            
            # 4. Learning & Adaptation
            # - Learn from recent performance
            # - Adapt strategy parameters if needed
            # - Update model weights
            
            logger.info("🔄 Trading cycle completed")
            
        except Exception as e:
            logger.error(f"❌ Trading loop error: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown the system"""
        logger.info("📋 Shutting down trading system...")
        
        self.running = False
        
        # Close any open positions
        # Save system state
        # Export performance data
        
        logger.info("✅ System shutdown complete")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"🚨 Received signal {signum}")
    sys.exit(0)

async def main():
    """Main function"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 Advanced AI Crypto Trading System")
    print("=" * 50)
    print(f"Start time: {datetime.now()}")
    print(f"System mode: {'DEMO' if True else 'LIVE'}")
    print("=" * 50)
    
    # Create and start the trading system
    controller = TradingSystemController()
    
    try:
        await controller.start()
    except Exception as e:
        logger.error(f"❌ System startup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Check if this is a test run
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("🧪 Running system tests instead...")
        import subprocess
        result = subprocess.run([sys.executable, "run_tests.py"], capture_output=False)
        sys.exit(result.returncode)
    
    # Run the main trading system
    exit_code = asyncio.run(main())
    sys.exit(exit_code)