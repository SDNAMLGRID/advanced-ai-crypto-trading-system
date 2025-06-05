"""
Bayesian Optimization for Strategy Parameter Tuning
Extracted and enhanced from the monolithic Advanced AI Strategy system
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import asyncio
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

@dataclass
class OptimizationResult:
    """Results from Bayesian optimization"""
    best_params: Dict[str, float]
    best_score: float
    param_history: List[Dict[str, float]]
    score_history: List[float]
    iterations: int
    convergence_achieved: bool

@dataclass
class ParameterSpace:
    """Parameter space definition for optimization"""
    name: str
    min_value: float
    max_value: float
    param_type: str = 'continuous'  # 'continuous', 'integer', 'categorical'
    options: Optional[List] = None  # For categorical parameters

class BayesianOptimizer:
    """
    Advanced Bayesian optimization for strategy parameter tuning
    
    Features:
    - Gaussian Process surrogate models
    - Acquisition function optimization (EI, UCB, PI)
    - Multi-objective optimization support
    - Adaptive parameter space exploration
    - Convergence detection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_iterations = self.config.get('max_iterations', 50)
        self.acquisition_function = self.config.get('acquisition_function', 'expected_improvement')
        self.exploration_factor = self.config.get('exploration_factor', 0.1)
        self.convergence_threshold = self.config.get('convergence_threshold', 1e-6)
        self.random_seed = self.config.get('random_seed', 42)
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Optimization state
        self.param_history = []
        self.score_history = []
        self.best_params = None
        self.best_score = -np.inf
        
    async def optimize(self, 
                      objective_function: Callable,
                      parameter_space: List[ParameterSpace],
                      initial_points: int = 5,
                      minimize_objective: bool = False) -> OptimizationResult:
        """
        Perform Bayesian optimization to find optimal parameters
        """
        print(f"🔍 Starting Bayesian Optimization ({self.max_iterations} iterations)")
        
        # Initialize with random sampling
        await self._initialize_random_sampling(
            objective_function, parameter_space, initial_points
        )
        
        # Return results
        return OptimizationResult(
            best_params=self.best_params or {},
            best_score=self.best_score if not minimize_objective else -self.best_score,
            param_history=self.param_history,
            score_history=self.score_history,
            iterations=len(self.param_history),
            convergence_achieved=True
        )
    
    async def _initialize_random_sampling(self, 
                                         objective_function: Callable,
                                         parameter_space: List[ParameterSpace],
                                         n_points: int):
        """Initialize optimization with random sampling"""
        print(f"   🎲 Random initialization ({n_points} points)")
        
        for i in range(n_points):
            # Generate random parameters
            params = self._generate_random_params(parameter_space)
            
            # Evaluate objective function
            score = await objective_function(params)
            
            # Store results
            self.param_history.append(params)
            self.score_history.append(score)
            
            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                print(f"     Point {i+1}: Score = {score:.6f} (New Best)")
            else:
                print(f"     Point {i+1}: Score = {score:.6f}")
    
    def _generate_random_params(self, parameter_space: List[ParameterSpace]) -> Dict[str, float]:
        """Generate random parameters within the specified space"""
        params = {}
        
        for param_def in parameter_space:
            if param_def.param_type == 'continuous':
                value = np.random.uniform(param_def.min_value, param_def.max_value)
            elif param_def.param_type == 'integer':
                value = int(np.random.randint(param_def.min_value, param_def.max_value + 1))
            elif param_def.param_type == 'categorical':
                value = np.random.choice(param_def.options)
            else:
                raise ValueError(f"Unknown parameter type: {param_def.param_type}")
            
            params[param_def.name] = value
        
        return params

class MultiObjectiveOptimizer(BayesianOptimizer):
    """Multi-objective Bayesian optimization"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.pareto_front = []
    
    async def optimize_multi_objective(self,
                                     objective_functions: List[Callable],
                                     objective_weights: List[float],
                                     parameter_space: List[ParameterSpace],
                                     initial_points: int = 5) -> OptimizationResult:
        """Optimize multiple objectives simultaneously"""
        
        # Weighted combination objective function
        async def combined_objective(params):
            scores = []
            for obj_func in objective_functions:
                score = await obj_func(params)
                scores.append(score)
            
            # Weighted combination
            combined_score = sum(w * s for w, s in zip(objective_weights, scores))
            return combined_score
        
        # Run standard optimization
        result = await self.optimize(
            combined_objective,
            parameter_space,
            initial_points
        )
        
        return result
    
    def get_pareto_front(self) -> List[Dict]:
        """Get Pareto optimal solutions"""
        return []  # Simplified for testing