"""
Generic Multi-Objective Metaheuristics -- physical copy of methods_comparison's
methods_comparison/solvers/multi_objective_solving_module.py, so that this
framework runs EXACTLY the algorithms of the metaheuristics comparison.
Local differences: the default values are imported from dependency/algorithms/defaults.py instead of methods_comparison.solvers.defaults.

Generic Multi-Objective Metaheuristics – Custom Operators (Numpy Arrays)
=========================================================================
Algorithms: NSGA‑II, NSGA‑III, MOCS, MOPSO, MOGWO.

All operators and the objective function receive a `problem_params` tuple.
The framework evaluates a population by looping over solutions and calling
`objective_function(solution, problem_params)` for each one.
"""

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from numba import njit

from dependency.algorithms.defaults import (
    DEFAULT_ARCHIVE_SIZE, DEFAULT_BETA, DEFAULT_CROSSOVER_POOL_SIZE, DEFAULT_ELITE_FRACTION,
    DEFAULT_GENERATIONS, DEFAULT_N_DIVISIONS, DEFAULT_N_GRID,
    DEFAULT_P_ABANDON, DEFAULT_POPULATION_SIZE,
)


def _make_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Generator for one algorithm call. With seed=None the seed is DRAWN FROM
    NumPy's global RNG -- which methods_comparison seeds before every solver call
    (core.seeding.seed_everything) -- instead of from OS entropy, so every run
    is reproducible and paired runs (config['evaluation']['paired_seeds'])
    really share their random numbers. np.random.default_rng(None) would
    ignore the framework's seeding entirely."""
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
# GENERIC PARETO UTILITIES
# ============================================================================

@njit
def is_dominate(x: np.ndarray, y: np.ndarray) -> bool:
    """Return True if x dominates y (minimization)."""
    strictly_better = False
    for i in range(len(x)):
        if x[i] > y[i]:
            return False
        if x[i] < y[i]:
            strictly_better = True
    return strictly_better


@njit
def find_pareto_front_indexes(objective_values: np.ndarray) -> np.ndarray:
    N = len(objective_values)
    n = np.zeros(N, dtype=np.int32)
    for p in range(N):
        for q in range(N):
            if p != q and is_dominate(objective_values[q], objective_values[p]):
                n[p] += 1
    return np.where(n == 0)[0]


@njit
def fast_non_dominated_sort(objective_values: np.ndarray):
    N = len(objective_values)
    S = np.zeros((N, N), dtype=np.int32)
    S_count = np.zeros(N, dtype=np.int32)
    n = np.zeros(N, dtype=np.int32)
    ranks = np.zeros(N, dtype=np.int32)

    for p in range(N):
        for q in range(N):
            if p != q:
                if is_dominate(objective_values[p], objective_values[q]):
                    S[p, S_count[p]] = q
                    S_count[p] += 1
                elif is_dominate(objective_values[q], objective_values[p]):
                    n[p] += 1

    fronts_list = []
    current_front = np.zeros(N, dtype=np.int32)
    front_size = 0
    for p in range(N):
        if n[p] == 0:
            current_front[front_size] = p
            front_size += 1
            ranks[p] = 0
    fronts_list.append(current_front[:front_size].copy())

    rank = 0
    while front_size > 0:
        next_front = np.zeros(N, dtype=np.int32)
        next_size = 0
        for i in range(front_size):
            p = current_front[i]
            for j in range(S_count[p]):
                q = S[p, j]
                n[q] -= 1
                if n[q] == 0:
                    next_front[next_size] = q
                    next_size += 1
                    ranks[q] = rank + 1
        if next_size > 0:
            fronts_list.append(next_front[:next_size].copy())
            current_front = next_front
            front_size = next_size
            rank += 1
        else:
            break
    return fronts_list, ranks


@njit
def calculate_crowding_distance(fronts, objective_values: np.ndarray) -> np.ndarray:
    pop_size, num_objectives = objective_values.shape
    crowding_distances = np.zeros(pop_size, dtype=np.float64)
    for front in fronts:
        lf = len(front)
        if lf == 0:
            continue
        if lf <= 2:
            for idx in front:
                crowding_distances[idx] = 1e18
            continue
        distances = np.zeros(lf, dtype=np.float64)
        for k in range(num_objectives):
            obj_values = np.zeros(lf, dtype=np.float64)
            for i in range(lf):
                obj_values[i] = objective_values[front[i], k]
            sorted_indices = np.argsort(obj_values)
            distances[sorted_indices[0]] = 1e18
            distances[sorted_indices[-1]] = 1e18
            min_obj = obj_values[sorted_indices[0]]
            max_obj = obj_values[sorted_indices[-1]]
            if max_obj - min_obj < 1e-12:
                continue
            for j in range(1, lf - 1):
                idx_c = sorted_indices[j]
                idx_p = sorted_indices[j - 1]
                idx_n = sorted_indices[j + 1]
                distances[idx_c] += (obj_values[idx_n] - obj_values[idx_p]) / (max_obj - min_obj)
        for j in range(lf):
            crowding_distances[front[j]] = distances[j]
    return crowding_distances


@njit
def normalize_objectives(objective_values: np.ndarray) -> np.ndarray:
    n, n_obj = objective_values.shape
    mins = objective_values[0].copy()
    maxs = objective_values[0].copy()
    for i in range(1, n):
        for k in range(n_obj):
            if objective_values[i, k] < mins[k]:
                mins[k] = objective_values[i, k]
            if objective_values[i, k] > maxs[k]:
                maxs[k] = objective_values[i, k]

    span = maxs - mins
    for k in range(n_obj):
        if span[k] == 0:
            span[k] = 1.0

    result = np.empty_like(objective_values)
    for i in range(n):
        for k in range(n_obj):
            result[i, k] = (objective_values[i, k] - mins[k]) / span[k]
    return result


@njit
def deduplicate_pareto_front(
    objectives: np.ndarray,
    pf_idx: np.ndarray,
) -> np.ndarray:
    """Remove duplicate objective vectors from a Pareto front index array,
    keeping the earliest occurrence -- manual scan (np.unique(..., axis=0)
    is not supported in numba nopython mode), mirroring the mono module's
    deduplicate_best but comparing full objective ROWS instead of scalars."""
    n = len(pf_idx)
    if n <= 1:
        return pf_idx
    keep_mask = np.ones(n, dtype=np.bool_)
    for i in range(n):
        if not keep_mask[i]:
            continue
        vi = objectives[pf_idx[i]]
        for j in range(i + 1, n):
            if not keep_mask[j]:
                continue
            is_equal = True
            for k in range(len(vi)):
                if objectives[pf_idx[j], k] != vi[k]:
                    is_equal = False
                    break
            if is_equal:
                keep_mask[j] = False
    return pf_idx[keep_mask]


# ============================================================================
# NSGA-III REFERENCE POINTS
# ============================================================================

def generate_reference_points(n_objectives: int, n_divisions: int) -> np.ndarray:
    def _recurse(n_obj, left):
        if n_obj == 1:
            return [[left]]
        pts = []
        for i in range(left + 1):
            for tail in _recurse(n_obj - 1, left - i):
                pts.append([i] + tail)
        return pts

    raw = np.array(_recurse(n_objectives, n_divisions), dtype=np.float64)
    return raw / n_divisions


def associate_to_ref_points(
    norm_obj: np.ndarray,
    ref_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    ref_norm = ref_points / np.maximum(np.linalg.norm(ref_points, axis=1, keepdims=True), 1e-12)
    proj_len = norm_obj @ ref_norm.T
    proj = proj_len[:, :, None] * ref_norm[None, :, :]
    perp = np.linalg.norm(norm_obj[:, None, :] - proj, axis=2)
    association = np.argmin(perp, axis=1)
    distances = perp[np.arange(len(norm_obj)), association]
    return association, distances


def niching_selection(
    fronts: List[np.ndarray],
    objective_values: np.ndarray,
    population_size: int,
    ref_points: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    chosen = []
    k = 0
    front_idx = 0

    while front_idx < len(fronts) and k + len(fronts[front_idx]) <= population_size:
        chosen.extend(fronts[front_idx].tolist())
        k += len(fronts[front_idx])
        front_idx += 1

    if k == population_size:
        return np.array(chosen, dtype=np.int64)

    F_l = fronts[front_idx] if front_idx < len(fronts) else np.array([], dtype=np.int64)
    S_t = np.array(chosen + F_l.tolist(), dtype=np.int64)
    remaining = population_size - k

    norm_obj = normalize_objectives(objective_values[S_t])
    association, distances = associate_to_ref_points(norm_obj, ref_points)

    n_before = len(chosen)
    rho = np.zeros(len(ref_points), dtype=np.int64)
    for a in association[:n_before]:
        rho[a] += 1

    idx_Fl_local = np.arange(n_before, len(S_t))
    selected_local = []
    Z_r = np.arange(len(ref_points))

    while len(selected_local) < remaining and len(Z_r) > 0:
        rho_min = rho[Z_r].min()
        J_min = Z_r[rho[Z_r] == rho_min]
        j = rng.choice(J_min)

        candidates = [i for i in idx_Fl_local if association[i] == j and i not in selected_local]
        if candidates:
            if rho[j] == 0:
                i_best = min(candidates, key=lambda i: distances[i])
            else:
                i_best = rng.choice(candidates)
            selected_local.append(i_best)
            rho[j] += 1
        else:
            Z_r = Z_r[Z_r != j]

    chosen_final = chosen + [int(S_t[i]) for i in selected_local]
    assert len(chosen_final) == population_size, (
        f"niching_selection returned {len(chosen_final)} individuals, expected "
        f"population_size={population_size} -- reference points were exhausted "
        f"of candidates before the quota was filled (silent population shrinkage "
        f"instead of a loud failure)."
    )
    return np.array(chosen_final, dtype=np.int64)


# ============================================================================
# GRID-BASED EXTERNAL ARCHIVE (used by MOPSO and MOGWO)
#
# Free functions, not a class: `n_objectives` varies per problem, so
# `grid_count`'s shape (n_grid,)*n_obj is only known at RUNTIME. Numba needs a
# static shape at compile time, so `compute_grid_count`, `update_archive`,
# `select_leaders` -- and anything that calls it -- must stay pure Python. Archive state is threaded explicitly by the caller
# as a (solutions: list, objectives: ndarray) pair instead of held on self.
# ============================================================================

@njit
def compute_grid_count(objectives: np.ndarray, n_grid: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bin each point of `objectives` (shape (N, n_obj)) into an n_grid-per-
    axis hypercube. `n_obj` is only known at RUNTIME, so a genuinely
    (n_grid,)*n_obj-shaped array (dynamic rank) is not numba-compatible --
    the hypercube is flattened into a 1D array instead, indexed via a
    manually computed (row-major) flat index. Returns (grid_count, flat_idx):
      grid_count : ndarray, shape (n_grid**n_obj,) -- points per grid cell.
      flat_idx   : ndarray, shape (N,) -- each point's flat cell index.
    """
    n = objectives.shape[0]
    n_obj = objectives.shape[1]

    mins = objectives[0].copy()
    maxs = objectives[0].copy()
    for i in range(1, n):
        for k in range(n_obj):
            if objectives[i, k] < mins[k]:
                mins[k] = objectives[i, k]
            if objectives[i, k] > maxs[k]:
                maxs[k] = objectives[i, k]

    flat_idx = np.zeros(n, dtype=np.int64)
    for i in range(n):
        idx = 0
        for k in range(n_obj):
            span = maxs[k] - mins[k]
            if span < 1e-12:
                span = 1e-12
            c = int((objectives[i, k] - mins[k]) / span * n_grid)
            if c < 0:
                c = 0
            if c > n_grid - 1:
                c = n_grid - 1
            idx = idx * n_grid + c
        flat_idx[i] = idx

    grid_count = np.zeros(n_grid ** n_obj, dtype=np.int64)
    for i in range(n):
        grid_count[flat_idx[i]] += 1

    return grid_count, flat_idx


def update_archive(
    archive_solutions: List[np.ndarray],
    archive_objectives: np.ndarray,
    new_solutions: Sequence[np.ndarray],
    new_objectives: np.ndarray,
    max_size: int,
    n_grid: int,
    rng: np.random.Generator,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Merge `new_solutions`/`new_objectives` into the archive, keep only the
    non-dominated duplicate-free union, then -- if still over `max_size` --
    repeatedly evict one random member of the currently most-crowded grid
    cell (grid-density trimming). Returns the updated (solutions, objectives)."""
    if len(archive_solutions) > 0:
        combined_sol = archive_solutions + list(new_solutions)
        combined_obj = np.vstack([archive_objectives, new_objectives])
    else:
        combined_sol = list(new_solutions)
        combined_obj = np.asarray(new_objectives)

    pf_idx = find_pareto_front_indexes(combined_obj)
    keep_global_idx = deduplicate_pareto_front(combined_obj, pf_idx)

    solutions = [combined_sol[i] for i in keep_global_idx]
    objectives = combined_obj[keep_global_idx]

    if len(solutions) > max_size:
        _, flat_idx = compute_grid_count(objectives, n_grid)
        keep = list(range(len(solutions)))

        while len(keep) > max_size:
            cell_counts = {}
            for i in keep:
                key = int(flat_idx[i])
                cell_counts[key] = cell_counts.get(key, 0) + 1
            crowded_cell = max(cell_counts, key=lambda c: cell_counts[c])
            candidates = [i for i in keep if int(flat_idx[i]) == crowded_cell]
            remove_idx = candidates[rng.integers(len(candidates))]
            keep.remove(remove_idx)

        solutions = [solutions[i] for i in keep]
        objectives = objectives[keep]

    return solutions, objectives


def select_leaders(
    objectives: np.ndarray, k: int, n_grid: int, beta: float, rng: np.random.Generator,
) -> np.ndarray:
    """Sample k archive indices, weighted inversely by their grid cell's
    density**beta (less-crowded cells favoured). Falls back to sampling with
    replacement when the archive holds fewer than k points."""
    n = len(objectives)
    if n == 0:
        raise RuntimeError("Cannot select leaders from an empty archive.")
    if n < k:
        return rng.integers(0, n, size=k)

    grid_count, flat_idx = compute_grid_count(objectives, n_grid)
    counts = grid_count[flat_idx].astype(np.float64)
    weights = 1.0 / (counts ** beta)
    probs = weights / weights.sum()
    return rng.choice(n, size=k, replace=False, p=probs)


def _leader_draw_setup(objectives: np.ndarray, n_grid: int, beta: float):
    """Precomputes the per-generation-static pieces of select_leaders'
    k=1 draw (the grid-density weighting) once, so a caller drawing one
    leader per individual across a whole population doesn't pay
    compute_grid_count's O(archive_size) cost on every single draw --
    `objectives` (the archive) is fixed for the whole generation, only
    mutated afterwards by update_archive. Mirrors select_leaders' exact
    k=1 control flow/fallback so the draw sequence is unchanged; returns
    (n, probs) where probs is None in the (k=1-unreachable, kept only for
    fidelity) n < 1 fallback branch."""
    n = len(objectives)
    if n == 0:
        raise RuntimeError("Cannot select leaders from an empty archive.")
    if n < 1:
        return n, None
    grid_count, flat_idx = compute_grid_count(objectives, n_grid)
    counts = grid_count[flat_idx].astype(np.float64)
    weights = 1.0 / (counts ** beta)
    probs = weights / weights.sum()
    return n, probs


def _draw_leader(n: int, probs: Optional[np.ndarray], rng: np.random.Generator) -> int:
    """One leader draw using precomputed `_leader_draw_setup` output --
    exactly select_leaders' k=1 draw step, just without recomputing probs."""
    if probs is None:
        return int(rng.integers(0, n, size=1)[0])
    return int(rng.choice(n, size=1, replace=False, p=probs)[0])


# ============================================================================
# PROBLEM CONTRACT
# ============================================================================

@dataclass
class Problem:
    """
    Multi-objective minimisation problem with custom operators.

    Solutions are numpy arrays of shape `solution_shape`.
    Every operator and the objective function receive `problem_params` (a tuple).

    Attributes
    ----------
    n_objectives : int
    solution_shape : tuple
    objective_function : callable
        objective_function(solution, problem_params) -> 1D array of objective values.
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
    obj_signs : optional, array of ±1 for each objective (1 = minimize, -1 = maximize).
    name : str
    """
    n_objectives: int
    solution_shape: tuple
    objective_function: Callable[[np.ndarray, tuple], np.ndarray]
    generate_solution: Callable[[tuple], np.ndarray]
    perturb: Callable[[np.ndarray, tuple], np.ndarray]
    crossover: Callable[[np.ndarray, tuple], np.ndarray]
    mutate: Optional[Callable[[np.ndarray, tuple], np.ndarray]] = None
    problem_params: tuple = ()
    obj_signs: Optional[np.ndarray] = None
    name: str = "problem"

    def __post_init__(self):
        if self.obj_signs is None:
            self.obj_signs = np.ones(self.n_objectives)
        else:
            self.obj_signs = np.asarray(self.obj_signs, dtype=np.float64)

        if self.mutate is None:
            self.mutate = self.perturb

    def evaluate_min(self, population: List[np.ndarray]) -> np.ndarray:
        objs = [self.objective_function(sol, self.problem_params) for sol in population]
        raw = np.array(objs)
        return raw * self.obj_signs

    def evaluate_one(self, solution: np.ndarray) -> np.ndarray:
        """Single-solution counterpart to evaluate_min, for algorithms that
        build a population one solution at a time and want to evaluate each
        one inline instead of batching a separate full pass afterwards."""
        return self.objective_function(solution, self.problem_params) * self.obj_signs

    def to_raw(self, signed_objectives: np.ndarray) -> np.ndarray:
        return signed_objectives * self.obj_signs


# ============================================================================
# COMMON RESULT CONTAINER
# ============================================================================

@dataclass
class OptimizationResult:
    algorithm: str
    pareto_solutions: List[np.ndarray]
    pareto_objectives: np.ndarray
    history_hypervolume: Optional[np.ndarray] = None
    progress_fronts: Optional[list] = None
    history_time: Optional[list] = None
    n_generations: int = 0
    population_size: int = 0


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _tournament_nsga2(
    population: List[np.ndarray],
    ranks: np.ndarray,
    distances: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    i1, i2 = rng.choice(len(population), size=2, replace=False)
    if ranks[i1] < ranks[i2]:
        return population[i1]
    if ranks[i2] < ranks[i1]:
        return population[i2]
    return population[i1] if distances[i1] >= distances[i2] else population[i2]


def _select_next_generation_nsga2(
    population: List[np.ndarray],
    objectives: np.ndarray,
    population_size: int,
    rng: np.random.Generator,
):
    fronts, ranks = fast_non_dominated_sort(objectives)
    distances = calculate_crowding_distance(fronts, objectives)

    new_idx = []
    k = 0
    for front in fronts:
        if k + len(front) <= population_size:
            new_idx.extend(front.tolist())
            k += len(front)
        else:
            remaining = population_size - k
            order = np.argsort(-distances[front])
            new_idx.extend(front[order][:remaining].tolist())
            k = population_size
            break

    new_idx = np.array(new_idx, dtype=np.int64)
    new_pop = [population[i] for i in new_idx]
    return new_pop, objectives[new_idx], distances[new_idx], ranks[new_idx]


def _select_next_generation_nsga3(
    population: List[np.ndarray],
    objectives: np.ndarray,
    population_size: int,
    ref_points: np.ndarray,
    rng: np.random.Generator,
):
    fronts, ranks = fast_non_dominated_sort(objectives)
    sel_idx = niching_selection(fronts, objectives, population_size, ref_points, rng)
    new_pop = [population[i] for i in sel_idx]
    return new_pop, objectives[sel_idx], ranks[sel_idx]


def _sort_population_by_rank_crowding(
    population: List[np.ndarray],
    objectives: np.ndarray,
    keep_size: int,
):
    fronts, ranks = fast_non_dominated_sort(objectives)
    crowding = calculate_crowding_distance(fronts, objectives)
    order = np.lexsort((-crowding, ranks))[:keep_size]

    sorted_population = [population[i] for i in order]
    sorted_objectives = objectives[order]
    sorted_ranks = ranks[order]
    sorted_crowding = crowding[order]

    return sorted_population, sorted_objectives, sorted_ranks, sorted_crowding


@njit
def _elite_probabilities(elite_count: int) -> np.ndarray:
    probs = (elite_count - np.arange(elite_count)).astype(np.float64)
    probs /= probs.sum()
    return probs


def _record_front(
    problem: Problem,
    objectives: np.ndarray,
    progress: Optional[list],
    ranks: Optional[np.ndarray] = None,
) -> None:
    """Append the current generation's non-dominated front (raw, unsigned
    objectives) to `progress`, in place. No-op if `progress` is None (i.e.
    record_progress=False) -- this is the ONLY per-generation cost paid when
    convergence recording isn't requested, so tuning/evaluation runs are
    unaffected.

    `ranks` lets a caller that already ran `fast_non_dominated_sort` this
    generation (for selection) pass its rank array along, so the front is
    read off as `ranks == 0` instead of paying a second, independent O(N^2)
    `find_pareto_front_indexes` pass over the same objectives -- the two are
    mathematically identical (front 0 of a full non-dominated sort). Callers
    that never computed ranks this generation (e.g. MOPSO/MOGWO's archive
    snapshot, or a first-generation call before any selection has run) leave
    this as None and fall back to the original standalone computation."""
    if progress is None:
        return
    pf_idx = np.where(ranks == 0)[0] if ranks is not None else find_pareto_front_indexes(objectives)
    progress.append(problem.to_raw(objectives[pf_idx]).tolist())


# ============================================================================
# NSGA-II
# ============================================================================

def NSGA2(
    problem: Problem,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    crossover_pool_size: int = DEFAULT_CROSSOVER_POOL_SIZE,
    mutation_rate: float = 0.2,
    seed: Optional[int] = None,
    show_progress: bool = True,
    time_limit: Optional[float] = None,
    record_progress: bool = False,
) -> OptimizationResult:
    rng = _make_rng(seed)
    progress = [] if record_progress else None

    iterator = range(generations)
    clock = _GenClock(time_limit, include_start=True)
    iterator = _progress(clock.wrap(iterator), show_progress, f"NSGA2[{problem.name}]")

    population, objectives = [], []
    for _ in range(population_size):
        sol = problem.generate_solution(problem.problem_params)
        population.append(sol)
        objectives.append(problem.evaluate_one(sol))
    objectives = np.array(objectives)
    population, objectives, distances, ranks = _select_next_generation_nsga2(
        population, objectives, population_size, rng
    )
    _record_front(problem, objectives, progress, ranks)

    for _ in iterator:
        offspring, offspring_obj = [], []
        for _ in range(population_size):
            parents = [_tournament_nsga2(population, ranks, distances, rng) for _ in range(crossover_pool_size)]
            child = problem.crossover(np.array(parents), problem.problem_params)
            if rng.random() < mutation_rate:
                child = problem.mutate(child, problem.problem_params)
            offspring.append(child)
            offspring_obj.append(problem.evaluate_one(child))

        offspring_obj = np.array(offspring_obj)
        combined_pop = population + offspring
        combined_obj = np.concatenate([objectives, offspring_obj], axis=0)

        population, objectives, distances, ranks = _select_next_generation_nsga2(
            combined_pop, combined_obj, population_size, rng
        )
        _record_front(problem, objectives, progress, ranks)

    pf_idx = find_pareto_front_indexes(objectives)
    pf_idx = deduplicate_pareto_front(objectives, pf_idx)
    return OptimizationResult(
        algorithm="NSGA2",
        pareto_solutions=[population[i] for i in pf_idx],
        pareto_objectives=problem.to_raw(objectives[pf_idx]),
        progress_fronts=progress,
        history_time=clock.times,
        n_generations=generations,
        population_size=population_size,
    )


# ============================================================================
# NSGA-III
# ============================================================================

def NSGA3(
    problem: Problem,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    n_divisions: int = DEFAULT_N_DIVISIONS,
    crossover_pool_size: int = DEFAULT_CROSSOVER_POOL_SIZE,
    mutation_rate: float = 0.2,
    seed: Optional[int] = None,
    show_progress: bool = True,
    time_limit: Optional[float] = None,
    record_progress: bool = False,
) -> OptimizationResult:
    if crossover_pool_size > population_size:
        raise ValueError(
            f"crossover_pool_size ({crossover_pool_size}) must be <= population_size "
            f"({population_size}) -- NSGA3 draws crossover_pool_size parents without "
            f"replacement from the current population each time."
        )
    ref_points = generate_reference_points(problem.n_objectives, n_divisions)
    if len(ref_points) > population_size:
        raise ValueError(
            f"n_divisions={n_divisions} for {problem.n_objectives} objectives yields "
            f"{len(ref_points)} reference points, which exceeds population_size "
            f"({population_size}) -- niching_selection cannot give every reference "
            f"point a fair shot at a representative with fewer population slots than "
            f"reference points. Lower n_divisions or raise population_size."
        )
    rng = _make_rng(seed)
    progress = [] if record_progress else None

    iterator = range(generations)
    clock = _GenClock(time_limit, include_start=True)
    iterator = _progress(clock.wrap(iterator), show_progress, f"NSGA3[{problem.name}]")

    population, objectives = [], []
    for _ in range(population_size):
        sol = problem.generate_solution(problem.problem_params)
        population.append(sol)
        objectives.append(problem.evaluate_one(sol))
    objectives = np.array(objectives)
    _record_front(problem, objectives, progress)

    for _ in iterator:
        offspring, offspring_obj = [], []
        for _ in range(population_size):
            idxs = rng.choice(len(population), size=crossover_pool_size, replace=False)
            parents = [population[i] for i in idxs]
            child = problem.crossover(np.array(parents), problem.problem_params)
            if rng.random() < mutation_rate:
                child = problem.mutate(child, problem.problem_params)
            offspring.append(child)
            offspring_obj.append(problem.evaluate_one(child))

        offspring_obj = np.array(offspring_obj)
        combined_pop = population + offspring
        combined_obj = np.concatenate([objectives, offspring_obj], axis=0)

        population, objectives, ranks = _select_next_generation_nsga3(
            combined_pop, combined_obj, population_size, ref_points, rng
        )
        _record_front(problem, objectives, progress, ranks)

    pf_idx = find_pareto_front_indexes(objectives)
    pf_idx = deduplicate_pareto_front(objectives, pf_idx)
    return OptimizationResult(
        algorithm="NSGA3",
        pareto_solutions=[population[i] for i in pf_idx],
        pareto_objectives=problem.to_raw(objectives[pf_idx]),
        progress_fronts=progress,
        history_time=clock.times,
        n_generations=generations,
        population_size=population_size,
    )


# ============================================================================
# MOCS -- no archive, MOCS_main-like logic
# ============================================================================

def MOCS(
    problem: Problem,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    p_abandon: float = DEFAULT_P_ABANDON,
    elite_fraction: float = DEFAULT_ELITE_FRACTION,
    seed: Optional[int] = None,
    show_progress: bool = True,
    time_limit: Optional[float] = None,
    record_progress: bool = False,
) -> OptimizationResult:
    rng = _make_rng(seed)
    progress = [] if record_progress else None

    population, objectives = [], []
    for _ in range(population_size):
        sol = problem.generate_solution(problem.problem_params)
        population.append(sol)
        objectives.append(problem.evaluate_one(sol))
    objectives = np.array(objectives)
    population, objectives, ranks, crowding = _sort_population_by_rank_crowding(
        population, objectives, population_size
    )
    _record_front(problem, objectives, progress, ranks)

    elite_count = max(1, int(elite_fraction * population_size))
    elite_probs = _elite_probabilities(elite_count)
    n_abandon = int(p_abandon * population_size)

    iterator = range(generations)
    clock = _GenClock(time_limit, include_start=True)
    iterator = _progress(clock.wrap(iterator), show_progress, f"MOCS[{problem.name}]")

    for _ in iterator:
        new_population, new_objectives = [], []

        for _ in range(population_size):
            k = int(rng.choice(elite_count, p=elite_probs))
            child = problem.perturb(population[k], problem.problem_params)
            new_population.append(child)
            new_objectives.append(problem.evaluate_one(child))
        new_objectives = np.array(new_objectives)

        # Abandoned worst nests are rebuilt from scratch (a fresh, pure
        # schedule via generate_solution), not perturbed in place -- this is
        # the standard Cuckoo Search "abandon and rebuild" step, matching
        # every chapter's own description of this operator (e.g. chapter 10's
        # "abandon worst -> PlanChargingOperation"). Perturbing here instead
        # was a divergence from both the intended design and canonical CS.
        for j in range(1, n_abandon + 1):
            idx = population_size - j
            population[idx] = problem.generate_solution(problem.problem_params)
            objectives[idx] = problem.evaluate_one(population[idx])

        combined_population = population + new_population
        combined_objectives = np.concatenate([objectives, new_objectives], axis=0)

        population, objectives, ranks, crowding = _sort_population_by_rank_crowding(
            combined_population, combined_objectives, population_size
        )
        _record_front(problem, objectives, progress, ranks)

    pf_idx = np.where(ranks == 0)[0]
    pf_idx = deduplicate_pareto_front(objectives, pf_idx)
    return OptimizationResult(
        algorithm="MOCS",
        pareto_solutions=[population[i] for i in pf_idx],
        pareto_objectives=problem.to_raw(objectives[pf_idx]),
        progress_fronts=progress,
        history_time=clock.times,
        n_generations=generations,
        population_size=population_size,
    )


# ============================================================================
# MOPSO -- archive-based
# ============================================================================

def MOPSO(
    problem: Problem,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    archive_size: int = DEFAULT_ARCHIVE_SIZE,
    mutation_rate: float = 1.0,
    n_grid: int = DEFAULT_N_GRID,
    beta: float = DEFAULT_BETA,
    seed: Optional[int] = None,
    show_progress: bool = True,
    time_limit: Optional[float] = None,
    record_progress: bool = False,
) -> OptimizationResult:
    """Particle-swarm-style move (crossover of the particle, its personal
    best, and an archive leader) followed by a probabilistic mutation step,
    matching MOGWO's crossover-then-mutate structure. `mutation_rate` gates
    the post-crossover mutation step (default 1.0 preserves always-mutate)."""
    if archive_size < 1:
        raise ValueError(f"archive_size must be >= 1, got {archive_size}")
    if n_grid < 1:
        raise ValueError(f"n_grid must be >= 1, got {n_grid}")
    rng = _make_rng(seed)
    progress = [] if record_progress else None
    archive_solutions: List[np.ndarray] = []
    archive_objectives = np.empty((0, problem.n_objectives))

    population, objectives = [], []
    for _ in range(population_size):
        sol = problem.generate_solution(problem.problem_params)
        population.append(sol)
        objectives.append(problem.evaluate_one(sol))
    objectives = np.array(objectives)
    personal_best = population.copy()
    personal_best_obj = objectives.copy()
    archive_solutions, archive_objectives = update_archive(
        archive_solutions, archive_objectives, population, objectives, archive_size, n_grid, rng
    )
    _record_front(problem, archive_objectives, progress)

    iterator = range(generations)
    clock = _GenClock(time_limit, include_start=True)
    iterator = _progress(clock.wrap(iterator), show_progress, f"MOPSO[{problem.name}]")

    for _ in iterator:
        n_archive, leader_probs = _leader_draw_setup(archive_objectives, n_grid, beta)
        new_population = []
        new_objectives = np.empty((population_size, problem.n_objectives))
        for i in range(population_size):
            leader = archive_solutions[_draw_leader(n_archive, leader_probs, rng)]
            parents = np.array([population[i], personal_best[i], leader])
            child = problem.crossover(parents, problem.problem_params)
            if rng.random() < mutation_rate:
                child = problem.mutate(child, problem.problem_params)
            child_obj = problem.evaluate_one(child)

            new_population.append(child)
            new_objectives[i] = child_obj

            if is_dominate(child_obj, personal_best_obj[i]) or (
                not is_dominate(personal_best_obj[i], child_obj) and rng.random() < 0.5
            ):
                personal_best[i] = child
                personal_best_obj[i] = child_obj

        population, objectives = new_population, new_objectives
        archive_solutions, archive_objectives = update_archive(
            archive_solutions, archive_objectives, population, objectives, archive_size, n_grid, rng
        )
        _record_front(problem, archive_objectives, progress)

    return OptimizationResult(
        algorithm="MOPSO",
        pareto_solutions=archive_solutions,
        pareto_objectives=problem.to_raw(archive_objectives),
        progress_fronts=progress,
        history_time=clock.times,
        n_generations=generations,
        population_size=population_size,
    )


# ============================================================================
# MOGWO -- grid-based external archive, alpha-beta-delta crossover then mutation
# ============================================================================

def MOGWO(
    problem: Problem,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    archive_size: int = DEFAULT_ARCHIVE_SIZE,
    mutation_rate: float = 1.0,
    n_grid: int = DEFAULT_N_GRID,
    beta: float = DEFAULT_BETA,
    seed: Optional[int] = None,
    show_progress: bool = True,
    time_limit: Optional[float] = None,
    record_progress: bool = False,
) -> OptimizationResult:
    """Grey-wolf move (crossover of the three leaders) followed by a
    probabilistic mutation step, matching MOPSO's crossover-then-mutate
    structure.

    Following Mirjalili et al. (2016), the leaders alpha, beta and delta are
    drawn once per generation from an external archive of non-dominated
    solutions, by roulette wheel weighted inversely by their grid cell's
    density**beta, so that leadership is taken from the sparsely populated
    regions of the front; the archive is trimmed by the same grid-density
    criterion when it overflows. As in the original, the pack is replaced by
    the wolves it produces (no parent/offspring elitist merge) -- the archive,
    not the population, is what carries the elite.

    `mutation_rate` gates the post-crossover mutation step (default 1.0
    preserves always-mutate)."""
    if archive_size < 1:
        raise ValueError(f"archive_size must be >= 1, got {archive_size}")
    if n_grid < 1:
        raise ValueError(f"n_grid must be >= 1, got {n_grid}")
    rng = _make_rng(seed)
    progress = [] if record_progress else None
    archive_solutions: List[np.ndarray] = []
    archive_objectives = np.empty((0, problem.n_objectives))

    population, objectives = [], []
    for _ in range(population_size):
        sol = problem.generate_solution(problem.problem_params)
        population.append(sol)
        objectives.append(problem.evaluate_one(sol))
    objectives = np.array(objectives)
    archive_solutions, archive_objectives = update_archive(
        archive_solutions, archive_objectives, population, objectives,
        archive_size, n_grid, rng
    )
    _record_front(problem, archive_objectives, progress)

    iterator = range(generations)
    clock = _GenClock(time_limit, include_start=True)
    iterator = _progress(clock.wrap(iterator), show_progress, f"MOGWO[{problem.name}]")

    for _ in iterator:
        # alpha, beta, delta: three distinct archive members when the archive
        # holds at least three, drawn without replacement and weighted toward
        # sparse grid cells; select_leaders falls back to sampling with
        # replacement while the archive is still smaller than three.
        leader_idx = select_leaders(archive_objectives, 3, n_grid, beta, rng)
        leaders = np.array([archive_solutions[int(i)] for i in leader_idx])

        new_population, new_objectives = [], []
        for _ in range(population_size):
            child = problem.crossover(leaders, problem.problem_params)
            if rng.random() < mutation_rate:
                child = problem.mutate(child, problem.problem_params)
            new_population.append(child)
            new_objectives.append(problem.evaluate_one(child))

        population = new_population
        objectives = np.array(new_objectives)

        archive_solutions, archive_objectives = update_archive(
            archive_solutions, archive_objectives, population, objectives,
            archive_size, n_grid, rng
        )
        _record_front(problem, archive_objectives, progress)

    return OptimizationResult(
        algorithm="MOGWO",
        pareto_solutions=archive_solutions,
        pareto_objectives=problem.to_raw(archive_objectives),
        progress_fronts=progress,
        history_time=clock.times,
        n_generations=generations,
        population_size=population_size,
    )


# ============================================================================
# UNIFIED ENTRY POINT
# ============================================================================

_ALGORITHMS = {
    "NSGA2": NSGA2,
    "NSGA3": NSGA3,
    "MOCS": MOCS,
    "MOPSO": MOPSO,
    "MOGWO": MOGWO,
}


def solve(problem: Problem, algorithm: str = "NSGA2", **kwargs) -> OptimizationResult:
    if algorithm not in _ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from {list(_ALGORITHMS)}.")
    return _ALGORITHMS[algorithm](problem, **kwargs)