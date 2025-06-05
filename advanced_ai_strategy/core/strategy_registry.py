"""
Strategy Registry for dynamic strategy management
"""
from typing import Dict, List, Type, Optional
import importlib
import inspect
from .base_strategy import BaseStrategy

class StrategyRegistry:
    """Registry for managing and discovering trading strategies"""
    
    def __init__(self):
        self._strategies: Dict[str, Type[BaseStrategy]] = {}
        self._instances: Dict[str, BaseStrategy] = {}
        
    def register(self, strategy_class: Type[BaseStrategy], name: Optional[str] = None):
        """Register a strategy class"""
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(f"{strategy_class} must inherit from BaseStrategy")
            
        strategy_name = name or strategy_class.__name__
        self._strategies[strategy_name] = strategy_class
        print(f"📝 Registered strategy: {strategy_name}")
        
    def create_strategy(self, name: str, parameters: Dict) -> BaseStrategy:
        """Create strategy instance with parameters"""
        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' not found in registry")
            
        strategy_class = self._strategies[name]
        instance = strategy_class(parameters)
        
        if not instance.validate_parameters():
            raise ValueError(f"Invalid parameters for strategy '{name}'")
            
        self._instances[f"{name}_{id(instance)}"] = instance
        return instance
        
    def list_strategies(self) -> List[str]:
        """List all registered strategies"""
        return list(self._strategies.keys())
        
    def get_strategy_info(self, name: str) -> Dict:
        """Get strategy information"""
        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' not found")
            
        strategy_class = self._strategies[name]
        return {
            'name': name,
            'class': strategy_class.__name__,
            'module': strategy_class.__module__,
            'docstring': strategy_class.__doc__ or 'No description available'
        }

# Global registry instance
registry = StrategyRegistry()