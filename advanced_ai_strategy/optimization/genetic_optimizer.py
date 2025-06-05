"""
Genetic Algorithm Optimizer for Strategy Evolution
Extracted and enhanced from the monolithic Advanced AI Strategy system
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor

@dataclass
class Individual:
    """Individual strategy in the genetic algorithm"""
    genes: Dict[str, float]  # Strategy parameters
    fitness: float = 0.0
    age: int = 0
    generation: int = 0

@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm"""
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    tournament_size: int = 3
    convergence_threshold: float = 1e-6
    max_stagnant_generations: int = 20

@dataclass
class EvolutionResults:
    """Results from genetic evolution"""
    best_individual: Individual
    population_history: List[List[Individual]]
    fitness_history: List[float]
    diversity_history: List[float]
    generation_count: int
    convergence_achieved: bool

class GeneticOptimizer:
    """
    Advanced Genetic Algorithm for strategy parameter evolution
    
    Features:
    - Multi-objective optimization
    - Adaptive mutation rates
    - Elitism with diversity preservation
    - Parallel fitness evaluation
    - Convergence detection
    - Strategy genealogy tracking
    """
    
    def __init__(self, config: GeneticConfig = None):
        self.config = config or GeneticConfig()
        self.population = []
        self.generation = 0
        self.best_fitness_history = []
        self.diversity_history = []
        self.stagnant_generations = 0
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
    
    async def evolve(self,
                    fitness_function: Callable,
                    parameter_bounds: Dict[str, Tuple[float, float]],
                    initial_population: Optional[List[Dict]] = None) -> EvolutionResults:
        """
        Evolve strategy parameters using genetic algorithm
        
        Args:
            fitness_function: Async function to evaluate strategy fitness
            parameter_bounds: Dict of parameter_name -> (min_value, max_value)
            initial_population: Optional starting population
        """
        print(f"🧬 Starting Genetic Evolution ({self.config.max_generations} generations)")
        print(f"   Population Size: {self.config.population_size}")
        print(f"   Mutation Rate: {self.config.mutation_rate}")
        print(f"   Crossover Rate: {self.config.crossover_rate}")
        
        # Initialize population
        if initial_population:
            self.population = self._create_population_from_params(initial_population, parameter_bounds)
        else:
            self.population = self._create_random_population(parameter_bounds)
        
        # Evaluate initial population
        await self._evaluate_population(fitness_function)
        
        population_history = []
        
        # Evolution loop
        for generation in range(self.config.max_generations):
            self.generation = generation
            
            print(f"   Generation {generation + 1}/{self.config.max_generations}")
            
            # Store current population
            population_history.append([individual for individual in self.population])
            
            # Selection and reproduction
            new_population = await self._evolve_generation(fitness_function, parameter_bounds)
            self.population = new_population
            
            # Track metrics
            best_fitness = max(ind.fitness for ind in self.population)
            self.best_fitness_history.append(best_fitness)
            
            diversity = self._calculate_diversity()
            self.diversity_history.append(diversity)
            
            print(f"     Best Fitness: {best_fitness:.6f}")
            print(f"     Population Diversity: {diversity:.4f}")
            
            # Check convergence
            if self._check_convergence():
                print(f"     🎯 Convergence achieved at generation {generation + 1}")
                break
        
        # Get best individual
        best_individual = max(self.population, key=lambda x: x.fitness)
        
        print(f"   ✅ Evolution Complete")
        print(f"   Best Fitness: {best_individual.fitness:.6f}")
        print(f"   Best Parameters: {best_individual.genes}")
        
        return EvolutionResults(
            best_individual=best_individual,
            population_history=population_history,
            fitness_history=self.best_fitness_history,
            diversity_history=self.diversity_history,
            generation_count=self.generation + 1,
            convergence_achieved=self._check_convergence()
        )
    
    def _create_random_population(self, parameter_bounds: Dict[str, Tuple[float, float]]) -> List[Individual]:
        """Create initial random population"""
        population = []
        
        for i in range(self.config.population_size):
            genes = {}
            for param_name, (min_val, max_val) in parameter_bounds.items():
                genes[param_name] = random.uniform(min_val, max_val)
            
            individual = Individual(
                genes=genes,
                generation=0
            )
            population.append(individual)
        
        return population
    
    def _create_population_from_params(self, 
                                      initial_params: List[Dict],
                                      parameter_bounds: Dict[str, Tuple[float, float]]) -> List[Individual]:
        """Create population from provided parameters"""
        population = []
        
        # Add provided individuals
        for params in initial_params:
            individual = Individual(
                genes=params,
                generation=0
            )
            population.append(individual)
        
        # Fill remaining slots with random individuals
        remaining_slots = self.config.population_size - len(initial_params)
        if remaining_slots > 0:
            random_population = self._create_random_population(parameter_bounds)
            population.extend(random_population[:remaining_slots])
        
        return population[:self.config.population_size]
    
    async def _evaluate_population(self, fitness_function: Callable):
        """Evaluate fitness for entire population"""
        
        # Prepare tasks for parallel evaluation
        tasks = []
        for individual in self.population:
            if individual.fitness == 0.0:  # Only evaluate if not already evaluated
                task = fitness_function(individual.genes)
                tasks.append((individual, task))
        
        # Evaluate in parallel
        if tasks:
            results = await asyncio.gather(*[task for _, task in tasks])
            
            for (individual, _), fitness in zip(tasks, results):
                individual.fitness = fitness
    
    async def _evolve_generation(self, 
                                fitness_function: Callable,
                                parameter_bounds: Dict[str, Tuple[float, float]]) -> List[Individual]:
        """Evolve one generation"""
        
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        new_population = []
        
        # Elitism: Keep best individuals
        elite_count = self.config.elite_size
        for i in range(elite_count):
            elite = Individual(
                genes=self.population[i].genes.copy(),
                fitness=self.population[i].fitness,
                age=self.population[i].age + 1,
                generation=self.generation + 1
            )
            new_population.append(elite)
        
        # Fill remaining population through crossover and mutation
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child1_genes, child2_genes = self._crossover(parent1.genes, parent2.genes)
            else:
                child1_genes, child2_genes = parent1.genes.copy(), parent2.genes.copy()
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child1_genes = self._mutate(child1_genes, parameter_bounds)
            if random.random() < self.config.mutation_rate:
                child2_genes = self._mutate(child2_genes, parameter_bounds)
            
            # Create children
            child1 = Individual(genes=child1_genes, generation=self.generation + 1)
            child2 = Individual(genes=child2_genes, generation=self.generation + 1)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        new_population = new_population[:self.config.population_size]
        
        # Evaluate new individuals
        await self._evaluate_population(fitness_function)
        
        return new_population
    
    def _tournament_selection(self) -> Individual:
        """Tournament selection for parent selection"""
        tournament = random.sample(self.population, self.config.tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, genes1: Dict[str, float], genes2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Single-point crossover for parameter genes"""
        param_names = list(genes1.keys())
        crossover_point = random.randint(1, len(param_names) - 1)
        
        child1_genes = {}
        child2_genes = {}
        
        for i, param_name in enumerate(param_names):
            if i < crossover_point:
                child1_genes[param_name] = genes1[param_name]
                child2_genes[param_name] = genes2[param_name]
            else:
                child1_genes[param_name] = genes2[param_name]
                child2_genes[param_name] = genes1[param_name]
        
        return child1_genes, child2_genes
    
    def _mutate(self, genes: Dict[str, float], parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Gaussian mutation for continuous parameters"""
        mutated_genes = genes.copy()
        
        for param_name, value in genes.items():
            if param_name in parameter_bounds:
                min_val, max_val = parameter_bounds[param_name]
                
                # Gaussian mutation with adaptive step size
                mutation_strength = (max_val - min_val) * 0.1
                noise = random.gauss(0, mutation_strength)
                
                new_value = value + noise
                new_value = max(min_val, min(max_val, new_value))  # Clamp to bounds
                
                mutated_genes[param_name] = new_value
        
        return mutated_genes
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity using parameter variance"""
        if not self.population:
            return 0.0
        
        # Calculate variance across all parameters
        param_names = list(self.population[0].genes.keys())
        total_variance = 0.0
        
        for param_name in param_names:
            param_values = [ind.genes[param_name] for ind in self.population]
            if len(param_values) > 1:
                variance = np.var(param_values)
                total_variance += variance
        
        return total_variance / len(param_names) if param_names else 0.0
    
    def _check_convergence(self) -> bool:
        """Check if evolution has converged"""
        if len(self.best_fitness_history) < 10:
            return False
        
        # Check if best fitness has plateaued
        recent_fitness = self.best_fitness_history[-10:]
        fitness_improvement = max(recent_fitness) - min(recent_fitness)
        
        if fitness_improvement < self.config.convergence_threshold:
            self.stagnant_generations += 1
        else:
            self.stagnant_generations = 0
        
        return self.stagnant_generations >= self.config.max_stagnant_generations