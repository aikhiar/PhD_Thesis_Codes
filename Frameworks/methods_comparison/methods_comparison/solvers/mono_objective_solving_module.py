"""
Generic Mono-Objective Metaheuristics – Custom Operators (Numpy Arrays)
=========================================================================
Algorithms: GA, ES, CS, PSO, GWO.

All operators and the objective function receive a `problem_params` tuple.
The framework evaluates a population by looping over solutions and calling
`objective_function(solution, problem_params)` for each one.
Every algorithm uses ONLY the problem-provided operators:
  generate_solution, perturb, crossover, mutate, objective_function.
"""

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np
from numba import njit

from methods_comparison.solvers.defaults import DEFAULT_GENERATIONS, DEFAULT_POPULATION_SIZE


def _make_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Generator for one algorithm call. With seed=None the seed is DRAWN FROM
    NumPy's global RNG -- which methods_comparison seeds before every solver call
    (core.seeding.seed_everything) -- instead of from OS entropy, so every run
    is reproducible. np.random.default_rng(None) would ignore the framework's
    seeding entirely."""
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    return np.random.default_rng(seed)

def _progress(iterator, show: bool, desc: str):
    """tqdm progress bar when requested AND installed; plain iterator
    otherwise (tqdm is optional)."""
    if not show:
        return iterator
    try:
        from tqdm import tqdm
    except ImportError:
        return iterator
    return tqdm(iterator, desc=desc)

class _GenClock:
    """Wall-clock trace of a generation loop, and an optional time limit.

    `wrap(iterable)` yields the generation indices; it records the elapsed
    time (s, measured from the clock's creation, so initialisation counts)
    at the END of every completed generation in `times`, and stops the loop
    before a new generation once `time_limit` seconds have elapsed (at least
    one generation always runs). `include_start=True` also records the time
    at which the loop starts (multi-objective algorithms record a front
    before their first generation). `trim(history)` drops the entries of
    generations that were not run."""

    def __init__(self, time_limit: Optional[float] = None, include_start: bool = False):
        self.t0 = time.perf_counter()
        self.time_limit = time_limit
        self.include_start = include_start
        self.times = []
        self.n_done = 0

    def wrap(self, iterable):
        first = True
        for g in iterable:
            if first:
                if self.include_start:
                    self.times.append(time.perf_counter() - self.t0)
            else:
                el = time.perf_counter() - self.t0
                self.times.append(el)
                self.n_done += 1
                if self.time_limit is not None and el >= self.time_limit:
                    return
            first = False
            yield g
        if not first:
            self.times.append(time.perf_counter() - self.t0)
            self.n_done += 1

    def trim(self, history):
        return None if history is None else history[:self.n_done]


# ============================================================================
# GENERIC UTILITIES
# ============================================================================

@njit
def sort_population(
    fitness: np.ndarray,
    ascending: bool = False,
) -> np.ndarray:
    """
    Return indices that sort fitness.

    For minimization (lower = better):
      ascending=False (default) → best first  (lowest fitness first)
      ascending=True            → worst first (highest fitness first)
    """
    if ascending:
        return np.argsort(-fitness)  # worst first
    return np.argsort(fitness)       # best first


# ============================================================================
# SIMPLE ARCHIVE (used by PSO)
# ============================================================================

class SimpleArchive:
    """Maintains the best solutions found so far (single objective)."""
    def __init__(self, max_size: int, rng: Optional[np.random.Generator] = None):
        self.max_size = max_size
        self.rng = rng if rng is not None else _make_rng()
        self.solutions: List[np.ndarray] = []
        self.fitness: np.ndarray = np.empty(0)

    def __len__(self):
        return len(self.solutions)

    def insert(self, new_solutions: List[np.ndarray], new_fitness: np.ndarray) -> None:
        if len(self.solutions) > 0:
            combined_sol = self.solutions + list(new_solutions)
            combined_fit = np.concatenate([self.fitness, new_fitness])
        else:
            combined_sol = list(new_solutions)
            combined_fit = np.asarray(new_fitness, dtype=np.float64)

        combined_fit = np.asarray(combined_fit, dtype=np.float64)
        order = sort_population(combined_fit, ascending=False)
        keep = order[:self.max_size]

        self.solutions = [combined_sol[i] for i in keep]
        self.fitness = combined_fit[keep]

        _, unique_idx = np.unique(self.fitness, return_index=True)
        unique_idx.sort()
        self.solutions = [self.solutions[i] for i in unique_idx]
        self.fitness = self.fitness[unique_idx]

    def select_leaders(self, k: int = 1) -> List[np.ndarray]:
        if len(self.solutions) == 0:
            raise RuntimeError("Cannot select leaders from an empty archive.")
        if len(self.solutions) < k:
            idxs = self.rng.integers(0, len(self.solutions), size=k)
            return [self.solutions[i] for i in idxs]
        n = len(self.solutions)
        probs = (n - np.arange(n)).astype(np.float64)
        probs /= probs.sum()
        idxs = self.rng.choice(n, size=k, replace=False, p=probs)
        return [self.solutions[i] for i in idxs]

    def best(self) -> Tuple[np.ndarray, float]:
        if len(self.solutions) == 0:
            raise RuntimeError("Empty archive.")
        return self.solutions[0], self.fitness[0]


# ============================================================================
# PROBLEM CONTRACT
# ============================================================================

@dataclass
class Problem:
    """
    Mono-objective optimization problem with custom operators.

    Solutions are numpy arrays of shape `solution_shape`.
    Every operator and the objective function receive `problem_params` (a tuple).

    Attributes
    ----------
    solution_shape : tuple
    objective_function : callable
        objective_function(solution, problem_params) -> float (to minimize).
    generate_solution : callable
        generate_solution(problem_params) -> new solution.
    perturb : callable
        perturb(solution, problem_params) -> perturbed solution.
    crossover : callable
        crossover(parents_array, problem_params) -> child solution.
        parents_array is a numpy array of shape (n_parents, *solution_shape).
    mutate : callable, optional
        mutate(solution, problem_params) -> mutated solution. If None, perturb is used.
    problem_params : tuple
        Parameters passed to every operator and the objective.
    name : str
    """
    solution_shape: tuple
    objective_function: Callable[[np.ndarray, tuple], float]
    generate_solution: Callable[[tuple], np.ndarray]
    perturb: Callable[[np.ndarray, tuple], np.ndarray]
    crossover: Callable[[np.ndarray, tuple], np.ndarray]
    mutate: Optional[Callable[[np.ndarray, tuple], np.ndarray]] = None
    problem_params: tuple = ()
    name: str = "problem"

    def __post_init__(self):
        if self.mutate is None:
            self.mutate = self.perturb

    def evaluate(self, population: List[np.ndarray]) -> np.ndarray:
        return np.array([self.objective_function(sol, self.problem_params) for sol in population])


# ============================================================================
# COMMON RESULT CONTAINER
# ============================================================================

@dataclass
class OptimizationResult:
    algorithm: str
    best_solution: np.ndarray
    best_objective: float
    best_solutions: Optional[List[np.ndarray]] = None
    best_objectives: Optional[np.ndarray] = None
    history_best: Optional[np.ndarray] = None
    history_time: Optional[list] = None
    n_generations: int = 0
    population_size: int = 0


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

@njit
def _elite_probabilities(elite_count: int) -> np.ndarray:
    probs = (elite_count - np.arange(elite_count)).astype(np.float64)
    probs /= probs.sum()
    return probs


def _sort_pop_by_fitness(
    population: List[np.ndarray],
    fitness: np.ndarray,
    keep_size: int,
):
    """Sort population by fitness ascending (best first), keep top keep_size."""
    order = sort_population(fitness, ascending=False)[:keep_size]
    sorted_pop = [population[i] for i in order]
    sorted_fit = fitness[order]
    return sorted_pop, sorted_fit


# ============================================================================
# GA — Genetic Algorithm (tournament selection, crossover, mutation)
# Counterpart to: NSGA-II
# Operators: generate_solution, crossover, mutate, objective_function
# ============================================================================

def GA(
    problem: Problem,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    crossover_pool_size: int = 2,
    mutation_rate: float = 0.2,
    seed: Optional[int] = None,
    show_progress: bool = True,
    time_limit: Optional[float] = None,
) -> OptimizationResult:
    rng = _make_rng(seed)

    iterator = range(generations)
    clock = _GenClock(time_limit)
    iterator = _progress(clock.wrap(iterator), show_progress, f"GA[{problem.name}]")

    population, fitness = [], []
    for _ in range(population_size):
        sol = problem.generate_solution(problem.problem_params)
        population.append(sol)
        fitness.append(problem.objective_function(sol, problem.problem_params))
    fitness = np.array(fitness)
    population, fitness = _sort_pop_by_fitness(population, fitness, population_size)

    history = np.empty(generations)

    for gen in iterator:
        offspring, offspring_fitness = [], []
        for _ in range(population_size):
            parents = []
            for _ in range(crossover_pool_size):
                contestants = rng.choice(len(population), size=2, replace=False)
                winner = contestants[0] if fitness[contestants[0]] <= fitness[contestants[1]] else contestants[1]
                parents.append(population[winner])
            child = problem.crossover(np.array(parents), problem.problem_params)
            if rng.random() < mutation_rate:
                child = problem.mutate(child, problem.problem_params)
            offspring.append(child)
            offspring_fitness.append(problem.objective_function(child, problem.problem_params))

        combined_pop = population + offspring
        combined_fit = np.concatenate([fitness, np.array(offspring_fitness)])

        population, fitness = _sort_pop_by_fitness(combined_pop, combined_fit, population_size)
        history[gen] = fitness[0]

    return OptimizationResult(
        algorithm="GA",
        best_solution=population[0],
        best_objective=fitness[0],
        best_solutions=population[:min(10, len(population))],
        best_objectives=fitness[:min(10, len(fitness))],
        history_best=clock.trim(history),
        history_time=clock.times,
        n_generations=generations,
        population_size=population_size,
    )


# ============================================================================
# ES — (mu+lambda) Evolution Strategy with ranking selection
# Counterpart to: NSGA-III
# Operators: generate_solution, crossover, perturb, objective_function
# ============================================================================

def ES(
    problem: Problem,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    lambda_ratio: float = 1.0,
    elite_fraction: float = 0.5,
    crossover_pool_size: int = 2,
    mutation_rate: float = 1.0,
    seed: Optional[int] = None,
    show_progress: bool = True,
    time_limit: Optional[float] = None,
) -> OptimizationResult:
    """
    (mu+lambda) Evolution Strategy.

    Parents are selected via linear ranking. Offspring are produced via
    crossover of selected parents followed by perturbation. Environmental
    selection merges parents and offspring, keeping the best mu.

    Parameters
    ----------
    population_size : int
        Number of parents mu (also the final population size).
    lambda_ratio : float
        Offspring count = lambda_ratio * population_size.
    elite_fraction : float in (0, 1]
        Fraction of population used as the breeding pool for ranking selection.
    crossover_pool_size : int
        Number of parents combined by the crossover operator.
    mutation_rate : float in [0, 1]
        Probability of applying perturb after crossover.
    """
    rng = _make_rng(seed)

    mu = population_size
    lam = max(1, int(lambda_ratio * mu))
    elite_count = max(1, int(np.ceil(elite_fraction * mu)))
    elite_probs = _elite_probabilities(elite_count)

    iterator = range(generations)
    clock = _GenClock(time_limit)
    iterator = _progress(clock.wrap(iterator), show_progress, f"ES[{problem.name}]")

    population, fitness = [], []
    for _ in range(mu):
        sol = problem.generate_solution(problem.problem_params)
        population.append(sol)
        fitness.append(problem.objective_function(sol, problem.problem_params))
    fitness = np.array(fitness)
    population, fitness = _sort_pop_by_fitness(population, fitness, mu)

    history = np.empty(generations)

    for gen in iterator:
        offspring, offspring_fitness = [], []
        for _ in range(lam):
            parent_idxs = rng.choice(elite_count, size=crossover_pool_size,
                                      replace=(crossover_pool_size > elite_count), p=elite_probs)
            parents = np.array([population[i] for i in parent_idxs])
            child = problem.crossover(parents, problem.problem_params)
            if rng.random() < mutation_rate:
                child = problem.perturb(child, problem.problem_params)
            offspring.append(child)
            offspring_fitness.append(problem.objective_function(child, problem.problem_params))

        combined_pop = population + offspring
        combined_fit = np.concatenate([fitness, np.array(offspring_fitness)])
        population, fitness = _sort_pop_by_fitness(combined_pop, combined_fit, mu)
        history[gen] = fitness[0]

    return OptimizationResult(
        algorithm="ES",
        best_solution=population[0],
        best_objective=fitness[0],
        best_solutions=population[:min(10, len(population))],
        best_objectives=fitness[:min(10, len(fitness))],
        history_best=clock.trim(history),
        history_time=clock.times,
        n_generations=generations,
        population_size=population_size,
    )


# ============================================================================
# CS — Cuckoo Search
# Counterpart to: MOCS
# Operators: generate_solution, perturb, objective_function
# ============================================================================

def CS(
    problem: Problem,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    p_abandon: float = 0.25,
    elite_fraction: float = 0.3,
    seed: Optional[int] = None,
    show_progress: bool = True,
    time_limit: Optional[float] = None,
) -> OptimizationResult:
    rng = _make_rng(seed)

    population, fitness = [], []
    for _ in range(population_size):
        sol = problem.generate_solution(problem.problem_params)
        population.append(sol)
        fitness.append(problem.objective_function(sol, problem.problem_params))
    fitness = np.array(fitness)
    population, fitness = _sort_pop_by_fitness(population, fitness, population_size)

    elite_count = max(1, int(elite_fraction * population_size))
    elite_probs = _elite_probabilities(elite_count)
    n_abandon = int(p_abandon * population_size)

    iterator = range(generations)
    clock = _GenClock(time_limit)
    iterator = _progress(clock.wrap(iterator), show_progress, f"CS[{problem.name}]")

    history = np.empty(generations)

    for gen in iterator:
        new_population, new_fitness = [], []
        for _ in range(population_size):
            k = int(rng.choice(elite_count, p=elite_probs))
            child = problem.perturb(population[k], problem.problem_params)
            new_population.append(child)
            new_fitness.append(problem.objective_function(child, problem.problem_params))

        # Abandoned worst nests are rebuilt from scratch (a fresh, pure
        # schedule via generate_solution), not perturbed in place -- this is
        # the standard Cuckoo Search "abandon and rebuild" step, matching
        # MOCS's own abandon step (see multi_objective_solving_module.py).
        for j in range(1, n_abandon + 1):
            idx = population_size - j
            population[idx] = problem.generate_solution(problem.problem_params)
            fitness[idx] = problem.objective_function(population[idx], problem.problem_params)

        combined_population = population + new_population
        combined_fitness = np.concatenate([fitness, np.array(new_fitness)])

        population, fitness = _sort_pop_by_fitness(combined_population, combined_fitness, population_size)
        history[gen] = fitness[0]

    return OptimizationResult(
        algorithm="CS",
        best_solution=population[0],
        best_objective=fitness[0],
        best_solutions=population[:min(10, len(population))],
        best_objectives=fitness[:min(10, len(fitness))],
        history_best=clock.trim(history),
        history_time=clock.times,
        n_generations=generations,
        population_size=population_size,
    )


# ============================================================================
# PSO — Particle Swarm Optimization
# Counterpart to: MOPSO
# Operators: generate_solution, perturb, crossover, objective_function
# ============================================================================

def PSO(
    problem: Problem,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    archive_size: int = 100,
    mutation_rate: float = 1.0,
    seed: Optional[int] = None,
    show_progress: bool = True,
    time_limit: Optional[float] = None,
) -> OptimizationResult:
    """Particle-swarm-style move (crossover of the particle, its personal
    best, and the best solution found so far) followed by a probabilistic
    mutation step, matching GWO's crossover-then-mutate structure.
    `mutation_rate` gates the post-crossover mutation step (default 1.0
    preserves always-mutate)."""
    rng = _make_rng(seed)
    archive = SimpleArchive(archive_size, rng)

    population, fitness = [], []
    for _ in range(population_size):
        sol = problem.generate_solution(problem.problem_params)
        population.append(sol)
        fitness.append(problem.objective_function(sol, problem.problem_params))
    fitness = np.array(fitness)
    personal_best = population.copy()
    personal_best_fit = fitness.copy()
    archive.insert(population, fitness)

    iterator = range(generations)
    clock = _GenClock(time_limit)
    iterator = _progress(clock.wrap(iterator), show_progress, f"PSO[{problem.name}]")

    history = np.empty(generations)

    for gen in iterator:
        global_best, _ = archive.best()
        new_population = []
        new_fitness = np.empty(population_size)
        for i in range(population_size):
            parents = np.array([population[i], personal_best[i], global_best])
            child = problem.crossover(parents, problem.problem_params)
            if rng.random() < mutation_rate:
                child = problem.mutate(child, problem.problem_params)
            child_fit = problem.objective_function(child, problem.problem_params)

            new_population.append(child)
            new_fitness[i] = child_fit

            if child_fit < personal_best_fit[i] or (
                child_fit == personal_best_fit[i] and rng.random() < 0.5
            ):
                personal_best[i] = child
                personal_best_fit[i] = child_fit

        population = new_population
        fitness = new_fitness
        archive.insert(population, fitness)
        _, best_fit = archive.best()
        history[gen] = best_fit

    best_sol, best_fit = archive.best()
    return OptimizationResult(
        algorithm="PSO",
        best_solution=best_sol,
        best_objective=best_fit,
        best_solutions=archive.solutions[:min(10, len(archive.solutions))],
        best_objectives=archive.fitness[:min(10, len(archive.fitness))],
        history_best=clock.trim(history),
        history_time=clock.times,
        n_generations=generations,
        population_size=population_size,
    )


# ============================================================================
# GWO — Grey Wolf Optimizer
# Counterpart to: MOGWO
# Operators: generate_solution, crossover, mutate, objective_function
# ============================================================================

def GWO(
    problem: Problem,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    mutation_rate: float = 1.0,
    seed: Optional[int] = None,
    show_progress: bool = True,
    time_limit: Optional[float] = None,
) -> OptimizationResult:
    """Grey Wolf Optimizer using the problem's crossover + mutate operators.

    `mutation_rate` gates the post-crossover mutation step probabilistically
    (default 1.0 preserves the original always-mutate behaviour)."""
    rng = _make_rng(seed)

    population, fitness = [], []
    for _ in range(population_size):
        sol = problem.generate_solution(problem.problem_params)
        population.append(sol)
        fitness.append(problem.objective_function(sol, problem.problem_params))
    fitness = np.array(fitness)
    population, fitness = _sort_pop_by_fitness(population, fitness, population_size)

    iterator = range(generations)
    clock = _GenClock(time_limit)
    iterator = _progress(clock.wrap(iterator), show_progress, f"GWO[{problem.name}]")

    history = np.empty(generations)

    for gen in iterator:
        if population_size >= 3:
            leaders = np.array([population[0], population[1], population[2]])
        else:
            idxs = rng.choice(population_size, size=3, replace=True)
            leaders = np.array([population[int(i)] for i in idxs])

        new_population, new_fitness = [], []
        for _ in range(population_size):
            child = problem.crossover(leaders, problem.problem_params)
            if rng.random() < mutation_rate:
                child = problem.mutate(child, problem.problem_params)
            new_population.append(child)
            new_fitness.append(problem.objective_function(child, problem.problem_params))

        # Elitism: merge parent + offspring before environmental selection
        combined_population = population + new_population
        combined_fitness = np.concatenate([fitness, np.array(new_fitness)])
        population, fitness = _sort_pop_by_fitness(combined_population, combined_fitness, population_size)
        history[gen] = fitness[0]

    return OptimizationResult(
        algorithm="GWO",
        best_solution=population[0],
        best_objective=fitness[0],
        best_solutions=population[:min(10, len(population))],
        best_objectives=fitness[:min(10, len(fitness))],
        history_best=clock.trim(history),
        history_time=clock.times,
        n_generations=generations,
        population_size=population_size,
    )


# ============================================================================
# UNIFIED ENTRY POINT
# ============================================================================

_ALGORITHMS = {
    "GA": GA,
    "ES": ES,
    "CS": CS,
    "PSO": PSO,
    "GWO": GWO,
}


def solve(problem: Problem, algorithm: str = "GA", **kwargs) -> OptimizationResult:
    if algorithm not in _ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from {list(_ALGORITHMS)}.")
    return _ALGORITHMS[algorithm](problem, **kwargs)