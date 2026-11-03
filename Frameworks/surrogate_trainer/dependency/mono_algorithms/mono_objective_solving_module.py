"""
Generic Mono-Objective Metaheuristics - Custom Operators (Numpy Arrays)
=========================================================================
Standalone copy of methods_comparison's `methods_comparison/solvers/mono_objective_solving_module.py`,
adapted for the surrogate_trainer framework (no import back into methods_comparison --
see the module-level constraint in `surrogate_trainer/main.py`).

Algorithms: GA, ES, CS, PSO, GWO.

Change vs. the methods_comparison original: every algorithm accepts an optional
`log_callback(generation, population, fitness)` argument. When provided, it is
invoked once per generation (including the initial population, generation 0)
with the CURRENT population sorted best-first by fitness. The callback itself
decides how much of the population to keep (the training_data_generator wires
in a callback that keeps only the top `elite_log_fraction` of the population,
per config.py) -- this module has no opinion on that fraction. No other
algorithmic behaviour is changed: the optimization logic is untouched.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np
from numba import njit

from dependency.seeding import seed_everything


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
      ascending=False (default) -> best first  (lowest fitness first)
      ascending=True            -> worst first (highest fitness first)
    """
    if ascending:
        return np.argsort(-fitness)  # worst first
    return np.argsort(fitness)       # best first


@njit
def deduplicate_best(
    objectives: np.ndarray,
    best_idx: np.ndarray,
) -> np.ndarray:
    """Remove duplicate objective values from best-index array, keeping the
    earliest occurrence."""
    n = len(best_idx)
    if n <= 1:
        return best_idx
    keep_mask = np.ones(n, dtype=np.bool_)
    for i in range(n):
        if not keep_mask[i]:
            continue
        vi = objectives[best_idx[i]]
        for j in range(i + 1, n):
            if keep_mask[j] and objectives[best_idx[j]] == vi:
                keep_mask[j] = False
    return best_idx[keep_mask]


# ============================================================================
# SIMPLE ARCHIVE (used by PSO)
# ============================================================================

class SimpleArchive:
    """Maintains the best solutions found so far (single objective)."""
    def __init__(self, max_size: int, rng: Optional[np.random.Generator] = None):
        self.max_size = max_size
        self.rng = rng if rng is not None else np.random.default_rng()
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


LogCallback = Optional[Callable[[int, List[np.ndarray], np.ndarray], None]]


def _log_gen(log_callback: LogCallback, generation: int, population: List[np.ndarray], fitness: np.ndarray) -> None:
    """Invoke log_callback with the population already sorted best-first.
    No-op if log_callback is None -- callers pay nothing extra in that case."""
    if log_callback is None:
        return
    log_callback(generation, population, fitness)


# ============================================================================
# GA -- Genetic Algorithm (tournament selection, crossover, mutation)
# Counterpart to: NSGA-II
# ============================================================================

def GA(
    problem: Problem,
    population_size: int = 100,
    generations: int = 200,
    crossover_pool_size: int = 2,
    mutation_rate: float = 0.2,
    seed: Optional[int] = None,
    show_progress: bool = True,
    log_callback: LogCallback = None,
) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    if seed is not None:
        # `rng` above only seeds this function's own Python-level draws.
        # Generate_Solution/Perturb/Crossover are @njit and consume numba's
        # OWN separate internal RNG state, which a bare np.random.default_rng
        # (or np.random.seed) call does NOT reset. seed_everything() also
        # resets numba's RNG from within compiled code, making the run
        # genuinely reproducible.
        seed_everything(seed)

    iterator = range(generations)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"GA[{problem.name}]")

    population, fitness = [], []
    for _ in range(population_size):
        sol = problem.generate_solution(problem.problem_params)
        population.append(sol)
        fitness.append(problem.objective_function(sol, problem.problem_params))
    fitness = np.array(fitness)
    population, fitness = _sort_pop_by_fitness(population, fitness, population_size)
    _log_gen(log_callback, 0, population, fitness)

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
        _log_gen(log_callback, gen + 1, population, fitness)

    return OptimizationResult(
        algorithm="GA",
        best_solution=population[0],
        best_objective=fitness[0],
        best_solutions=population[:min(10, len(population))],
        best_objectives=fitness[:min(10, len(fitness))],
        history_best=history,
        n_generations=generations,
        population_size=population_size,
    )


# ============================================================================
# ES -- (mu+lambda) Evolution Strategy with ranking selection
# Counterpart to: NSGA-III
# ============================================================================

def ES(
    problem: Problem,
    population_size: int = 100,
    generations: int = 200,
    lambda_ratio: float = 1.0,
    elite_fraction: float = 0.5,
    crossover_pool_size: int = 2,
    mutation_rate: float = 1.0,
    seed: Optional[int] = None,
    show_progress: bool = True,
    log_callback: LogCallback = None,
) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    if seed is not None:
        # `rng` above only seeds this function's own Python-level draws.
        # Generate_Solution/Perturb/Crossover are @njit and consume numba's
        # OWN separate internal RNG state, which a bare np.random.default_rng
        # (or np.random.seed) call does NOT reset. seed_everything() also
        # resets numba's RNG from within compiled code, making the run
        # genuinely reproducible.
        seed_everything(seed)

    mu = population_size
    lam = max(1, int(lambda_ratio * mu))
    elite_count = max(1, int(np.ceil(elite_fraction * mu)))
    elite_probs = _elite_probabilities(elite_count)

    iterator = range(generations)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"ES[{problem.name}]")

    population, fitness = [], []
    for _ in range(mu):
        sol = problem.generate_solution(problem.problem_params)
        population.append(sol)
        fitness.append(problem.objective_function(sol, problem.problem_params))
    fitness = np.array(fitness)
    population, fitness = _sort_pop_by_fitness(population, fitness, mu)
    _log_gen(log_callback, 0, population, fitness)

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
        _log_gen(log_callback, gen + 1, population, fitness)

    return OptimizationResult(
        algorithm="ES",
        best_solution=population[0],
        best_objective=fitness[0],
        best_solutions=population[:min(10, len(population))],
        best_objectives=fitness[:min(10, len(fitness))],
        history_best=history,
        n_generations=generations,
        population_size=population_size,
    )


# ============================================================================
# CS -- Cuckoo Search
# Counterpart to: MOCS
# ============================================================================

def CS(
    problem: Problem,
    population_size: int = 100,
    generations: int = 200,
    p_abandon: float = 0.25,
    elite_fraction: float = 0.3,
    seed: Optional[int] = None,
    show_progress: bool = True,
    log_callback: LogCallback = None,
) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    if seed is not None:
        # `rng` above only seeds this function's own Python-level draws.
        # Generate_Solution/Perturb/Crossover are @njit and consume numba's
        # OWN separate internal RNG state, which a bare np.random.default_rng
        # (or np.random.seed) call does NOT reset. seed_everything() also
        # resets numba's RNG from within compiled code, making the run
        # genuinely reproducible.
        seed_everything(seed)

    population, fitness = [], []
    for _ in range(population_size):
        sol = problem.generate_solution(problem.problem_params)
        population.append(sol)
        fitness.append(problem.objective_function(sol, problem.problem_params))
    fitness = np.array(fitness)
    population, fitness = _sort_pop_by_fitness(population, fitness, population_size)
    _log_gen(log_callback, 0, population, fitness)

    elite_count = max(1, int(elite_fraction * population_size))
    elite_probs = _elite_probabilities(elite_count)
    n_abandon = int(p_abandon * population_size)

    iterator = range(generations)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"CS[{problem.name}]")

    history = np.empty(generations)

    for gen in iterator:
        new_population, new_fitness = [], []
        for _ in range(population_size):
            k = int(rng.choice(elite_count, p=elite_probs))
            child = problem.perturb(population[k], problem.problem_params)
            new_population.append(child)
            new_fitness.append(problem.objective_function(child, problem.problem_params))

        for j in range(1, n_abandon + 1):
            idx = population_size - j
            population[idx] = problem.generate_solution(problem.problem_params)
            fitness[idx] = problem.objective_function(population[idx], problem.problem_params)

        combined_population = population + new_population
        combined_fitness = np.concatenate([fitness, np.array(new_fitness)])

        population, fitness = _sort_pop_by_fitness(combined_population, combined_fitness, population_size)
        history[gen] = fitness[0]
        _log_gen(log_callback, gen + 1, population, fitness)

    return OptimizationResult(
        algorithm="CS",
        best_solution=population[0],
        best_objective=fitness[0],
        best_solutions=population[:min(10, len(population))],
        best_objectives=fitness[:min(10, len(fitness))],
        history_best=history,
        n_generations=generations,
        population_size=population_size,
    )


# ============================================================================
# PSO -- Particle Swarm Optimization
# Counterpart to: MOPSO
# ============================================================================

def PSO(
    problem: Problem,
    population_size: int = 100,
    generations: int = 200,
    archive_size: int = 100,
    mutation_rate: float = 1.0,
    seed: Optional[int] = None,
    show_progress: bool = True,
    log_callback: LogCallback = None,
) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    if seed is not None:
        # `rng` above only seeds this function's own Python-level draws.
        # Generate_Solution/Perturb/Crossover are @njit and consume numba's
        # OWN separate internal RNG state, which a bare np.random.default_rng
        # (or np.random.seed) call does NOT reset. seed_everything() also
        # resets numba's RNG from within compiled code, making the run
        # genuinely reproducible.
        seed_everything(seed)
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

    if log_callback is not None:
        order0 = sort_population(fitness, ascending=False)
        _log_gen(log_callback, 0, [population[i] for i in order0], fitness[order0])

    iterator = range(generations)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"PSO[{problem.name}]")

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

        if log_callback is not None:
            order = sort_population(fitness, ascending=False)
            _log_gen(log_callback, gen + 1, [population[i] for i in order], fitness[order])

    best_sol, best_fit = archive.best()
    return OptimizationResult(
        algorithm="PSO",
        best_solution=best_sol,
        best_objective=best_fit,
        best_solutions=archive.solutions[:min(10, len(archive.solutions))],
        best_objectives=archive.fitness[:min(10, len(archive.fitness))],
        history_best=history,
        n_generations=generations,
        population_size=population_size,
    )


# ============================================================================
# GWO -- Grey Wolf Optimizer
# Counterpart to: MOGWO
# ============================================================================

def GWO(
    problem: Problem,
    population_size: int = 100,
    generations: int = 200,
    mutation_rate: float = 1.0,
    seed: Optional[int] = None,
    show_progress: bool = True,
    log_callback: LogCallback = None,
) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    if seed is not None:
        # `rng` above only seeds this function's own Python-level draws.
        # Generate_Solution/Perturb/Crossover are @njit and consume numba's
        # OWN separate internal RNG state, which a bare np.random.default_rng
        # (or np.random.seed) call does NOT reset. seed_everything() also
        # resets numba's RNG from within compiled code, making the run
        # genuinely reproducible.
        seed_everything(seed)

    population, fitness = [], []
    for _ in range(population_size):
        sol = problem.generate_solution(problem.problem_params)
        population.append(sol)
        fitness.append(problem.objective_function(sol, problem.problem_params))
    fitness = np.array(fitness)
    population, fitness = _sort_pop_by_fitness(population, fitness, population_size)
    _log_gen(log_callback, 0, population, fitness)

    iterator = range(generations)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"GWO[{problem.name}]")

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

        combined_population = population + new_population
        combined_fitness = np.concatenate([fitness, np.array(new_fitness)])
        population, fitness = _sort_pop_by_fitness(combined_population, combined_fitness, population_size)
        history[gen] = fitness[0]
        _log_gen(log_callback, gen + 1, population, fitness)

    return OptimizationResult(
        algorithm="GWO",
        best_solution=population[0],
        best_objective=fitness[0],
        best_solutions=population[:min(10, len(population))],
        best_objectives=fitness[:min(10, len(fitness))],
        history_best=history,
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
