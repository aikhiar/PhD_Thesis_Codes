"""
Training data generation: runs each problem's configured metaheuristics, with
FIXED parameters, over the problem's own (larger) training-instance pool, and
collects logged solutions along the way -- at different generations, i.e.
different levels of optimisation, not just the final result -- as raw
material for the surrogate regressors.

Two things keep this bounded on a real machine instead of exhausting RAM:

  1. Records store the already-EXTRACTED feature vector (`feature_fn(sol,
     instance)`, a handful of floats), never the raw solution array (which
     for a scheduling problem can be tens of KB each). This is computed once,
     right when the solution is still available inside the log callback --
     `build_feature_target_pairs` below used to call `feature_fn` itself in a
     second pass over the raw solutions; now it just reads the field back
     off, since nothing downstream ever needs the raw solution again.
  2. Per-generation logging is capped: `elite_log_fraction`/`pareto_log_fraction`
     bound how much of each generation's population/front is kept, and
     `log_generation_stride` bounds how many of the generations themselves
     are logged (the true final generation is always kept regardless of
     stride). Without both, a multi-objective run logs its ENTIRE population
     at EVERY generation -- for many training instances across several
     algorithms, that reaches tens of GB.

`ml_config["datagen_n_jobs"]` controls how the (instance, algorithm) tasks
are executed: 1 runs them sequentially in-process with full live
per-generation progress; anything else dispatches them across that many
joblib worker processes (numba/GIL-bound work benefits from process-, not
thread-, parallelism). A live rich.Progress display cannot be shared across
process boundaries, so parallel workers run silently and only the per-task
(not per-generation) "datagen" progress is updated, from the parent process,
once results come back.

Contract expected of a problem module (problems/problem_chapter5.py, problem_chapter6.py)
--------------------------------------------------------------------------
    CODE: str
    n_objectives: int
    objective_names: List[str]
    SURROGATE_OBJECTIVE_NAMES: List[str]
        The objectives that get a surrogate (Chapter 5: undelivered energy is exact
        and has none).

    generate_training_instances() -> List[Tuple[str, Any]]
        (instance_id, Instance) pairs of the surrogate's own training pool
        (TRAINING_INSTANCES_PER_SIZE per size class in parameters.py), disjoint
        from the evaluation instances of methods_comparison.
    instance_class(instance_id) -> str
        Size class of a training instance (stratification of the split by
        instance, per-class benchmark).
    exact_objective_values(instance, solution) -> Dict[str, float]
        Objectives computed exactly, without a surrogate (Chapter 5's energy; Chapter 6: {}).

    solution_shape(instance) -> tuple
    build_problem_params(params: dict, instance) -> tuple
    Generate_Solution(problem_params) -> np.ndarray
    Perturb(solution, problem_params) -> np.ndarray
    Crossover(parents_array, problem_params) -> np.ndarray
    objective_function(solution, problem_params) -> np.ndarray
        The REAL (Monte Carlo, N_r = LABEL_REPLICATIONS) objective -- the label
        the surrogate learns.
    feature_fn(solution, instance) -> np.ndarray
        Cheap, instance-aware feature map phi(S) (problems/jN/model.py).

    ALGORITHMS: Dict[str, dict]
        name -> {"kind": "mono" | "multi", "params": {...fixed metaheuristic
        hyperparameters...}, "solve_kwargs": {...extra kwargs for solve(),
        e.g. population_size/generations; never a seed: every task is seeded
        here from (instance, algorithm)...}}
"""

import importlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dependency.mono_algorithms.mono_objective_solving_module import (
    Problem as MonoProblem,
    solve as mono_solve,
)
from dependency.multi_algorithms.multi_objective_solving_module import (
    Problem as MultiProblem,
    solve as multi_solve,
)
from dependency.reporting.progress import ProgressReporter
from dependency.seeding import seed_everything


def _task_seed(instance_id: Any, algorithm: str) -> int:
    """Deterministic seed of one (instance, algorithm) data-generation task."""
    import zlib
    return zlib.crc32(f"{instance_id}|{algorithm}".encode()) & 0x7FFFFFFF


@dataclass
class TrainingRecord:
    instance_id: Any
    algorithm: str
    generation: int
    features: np.ndarray  # feature_fn(solution, instance) -- NOT the raw solution
    objective_values: np.ndarray  # raw (unsigned), length == n_objectives


@dataclass
class TrainingDataset:
    records: List[TrainingRecord] = field(default_factory=list)
    instances_by_id: Dict[Any, Any] = field(default_factory=dict)
    # compute seconds of every (instance, algorithm) task, "instance|algorithm"
    task_seconds: Dict[str, float] = field(default_factory=dict)
    warmup_seconds: float = 0.0
    wall_seconds: float = 0.0          # wall clock of the generation (this session)

    def summary(self) -> Dict[str, Any]:
        per_algo: Dict[str, int] = {}
        for rec in self.records:
            per_algo[rec.algorithm] = per_algo.get(rec.algorithm, 0) + 1
        return {
            "n_records": len(self.records),
            "n_instances": len(self.instances_by_id),
            "records_per_algorithm": per_algo,
            "datagen_compute_seconds": float(sum(self.task_seconds.values())),
            "datagen_warmup_seconds": float(self.warmup_seconds),
            "datagen_wall_seconds": float(self.wall_seconds),
        }


def _progress_stride(total_generations: int) -> int:
    """How many generations between progress prints: every generation for
    short runs, otherwise roughly 10 prints across the whole run -- so a
    tiny smoke-scale run (generations=20) still gives frequent feedback
    instead of looking silent/stalled for the entire call. Purely a display
    cadence -- unrelated to `log_generation_stride`, which controls how much
    training data is actually kept."""
    return max(1, int(np.ceil(total_generations / 10.0)))


def _gen_label(generation: int, total_generations: int) -> str:
    """Solvers invoke log_callback once more after the configured generation
    loop finishes (to log the final returned front), so `generation` can
    reach `total_generations` itself -- label that extra call "final" rather
    than printing a confusing out-of-range ratio like "4/3"."""
    if generation >= total_generations:
        return "final"
    return f"{generation + 1}/{total_generations}"


def _should_log_generation(generation: int, total_generations: Optional[int], log_stride: int) -> bool:
    """True for every `log_stride`-th generation, and ALWAYS true for the
    final callback invocation (generation >= total_generations) -- the
    converged/best front must never be dropped by the stride."""
    if total_generations is not None and generation >= total_generations:
        return True
    return log_stride <= 1 or generation % log_stride == 0


def _make_mono_log_callback(
    sink: List[TrainingRecord], instance_id: Any, algorithm: str, instance: Any, feature_fn,
    elite_log_fraction: float, log_stride: int = 1,
    total_generations: int = None, verbose: bool = False, t_start: float = None,
    progress: Optional[ProgressReporter] = None, gen_task_key: Optional[str] = None,
):
    stride = _progress_stride(total_generations) if total_generations else None

    def _callback(generation: int, population: List[np.ndarray], fitness: np.ndarray) -> None:
        if _should_log_generation(generation, total_generations, log_stride):
            k = max(1, int(np.ceil(elite_log_fraction * len(population))))
            for sol, fit in zip(population[:k], fitness[:k]):
                sink.append(TrainingRecord(
                    instance_id=instance_id,
                    algorithm=algorithm,
                    generation=generation,
                    features=np.asarray(feature_fn(sol, instance), dtype=np.float64),
                    objective_values=np.array([float(fit)], dtype=np.float64),
                ))
        best_fit = float(np.min(fitness))
        if progress is not None and gen_task_key is not None:
            progress.update(
                gen_task_key,
                completed=min(generation + 1, total_generations),
                description=f"    [{instance_id}/{algorithm}] gen {_gen_label(generation, total_generations)} (best={best_fit:.4g})",
            )
        elif verbose and stride and (generation % stride == 0 or generation >= total_generations - 1):
            elapsed = time.perf_counter() - t_start if t_start is not None else float("nan")
            print(
                f"    [{instance_id}/{algorithm}] generation {_gen_label(generation, total_generations)} "
                f"-- best={best_fit:.4g} -- {elapsed:.1f}s elapsed",
                flush=True,
            )
    return _callback


def _make_multi_log_callback(
    sink: List[TrainingRecord], instance_id: Any, algorithm: str, instance: Any, feature_fn,
    pareto_log_fraction: float, log_stride: int = 1,
    total_generations: int = None, verbose: bool = False, t_start: float = None,
    progress: Optional[ProgressReporter] = None, gen_task_key: Optional[str] = None,
):
    stride = _progress_stride(total_generations) if total_generations else None

    def _callback(generation: int, solutions: List[np.ndarray], objectives: np.ndarray) -> None:
        if _should_log_generation(generation, total_generations, log_stride):
            n_front = len(solutions)
            k = max(1, int(np.ceil(pareto_log_fraction * n_front)))
            if k >= n_front:
                idx = range(n_front)
            else:
                # Evenly spread the kept subset across the whole front --
                # there is no "elite" ordering within a non-dominated set
                # (every point on it is mutually non-dominated), so taking a
                # prefix would arbitrarily bias toward whatever order solve()
                # happens to return them in. np.unique both sorts and dedupes
                # the rounded positions (rounding can otherwise collide).
                idx = np.unique(np.linspace(0, n_front - 1, k).round().astype(int))
            for i in idx:
                sink.append(TrainingRecord(
                    instance_id=instance_id,
                    algorithm=algorithm,
                    generation=generation,
                    features=np.asarray(feature_fn(solutions[i], instance), dtype=np.float64),
                    objective_values=np.asarray(objectives[i], dtype=np.float64).copy(),
                ))
        if progress is not None and gen_task_key is not None:
            progress.update(
                gen_task_key,
                completed=min(generation + 1, total_generations),
                description=f"    [{instance_id}/{algorithm}] gen {_gen_label(generation, total_generations)} ({len(objectives)} non-dom.)",
            )
        elif verbose and stride and (generation % stride == 0 or generation >= total_generations - 1):
            elapsed = time.perf_counter() - t_start if t_start is not None else float("nan")
            print(
                f"    [{instance_id}/{algorithm}] generation {_gen_label(generation, total_generations)} "
                f"-- {len(objectives)} non-dominated -- {elapsed:.1f}s elapsed",
                flush=True,
            )
    return _callback


def _warmup_algorithms(
    problem_module, instances: List[Tuple[Any, Any]], progress: Optional[ProgressReporter] = None,
) -> None:
    """Runs every algorithm in `problem_module.ALGORITHMS` exactly once, on a
    shrunk population/generations budget, purely to trigger numba JIT
    compilation ahead of the real timed runs. Without this, whichever
    (instance, algorithm) pair happens to run first pays the one-time
    compile cost, which both inflates that run's own timing and skews the
    live per-generation elapsed/ETA reporting for it. Every other
    algorithm-specific solve_kwarg (seed, crossover_pool_size, p_abandon,
    elite_fraction, archive_size, n_grid, beta, mutation_rate, ...) is kept
    as configured -- only population_size/generations shrink -- so the same
    internal code paths actually get compiled, not just a subset. Runs on
    the first training instance only; numba specializes by argument
    dtype/ndim, not exact array shape, so the compiled functions are reused
    for every other (larger) instance afterward.

    NOTE this only warms up THIS (the parent) process. When
    `datagen_n_jobs != 1`, each worker process gets its own fresh numba
    cache and pays its own one-time compile cost on its first task -- these
    njit functions are not decorated with `cache=True`, so there is no
    on-disk cache for other processes to reuse. That per-worker cost is a
    few seconds, paid once per worker, and is dwarfed by the wall-clock win
    of running many tasks concurrently."""
    if not instances:
        return
    _, warmup_instance = instances[0]

    warmup_key = "warmup"
    if progress is not None:
        progress.start(warmup_key, "Warming up JIT (metaheuristics)", total=len(problem_module.ALGORITHMS))

    for algo_name, algo_spec in problem_module.ALGORITHMS.items():
        kind = algo_spec["kind"]
        params = algo_spec.get("params", {})
        solve_kwargs = dict(algo_spec.get("solve_kwargs", {}))
        # one generation at the configured population: NSGA-III needs at least as
        # many individuals as reference points, so the population is not shrunk
        solve_kwargs["generations"] = 1
        solve_kwargs["show_progress"] = False

        problem_params = problem_module.build_problem_params(params, warmup_instance)
        shape = problem_module.solution_shape(warmup_instance)

        if progress is not None:
            progress.update(warmup_key, description=f"Warming up JIT -- [bold]{algo_name}[/bold]")

        if kind == "mono":
            problem = MonoProblem(
                solution_shape=shape,
                objective_function=problem_module.objective_function,
                generate_solution=problem_module.Generate_Solution,
                perturb=problem_module.Perturb,
                crossover=problem_module.Crossover,
                problem_params=problem_params,
                name=f"{problem_module.CODE}_warmup",
            )
            mono_solve(problem, algorithm=algo_name, **solve_kwargs)
        elif kind == "multi":
            problem = MultiProblem(
                n_objectives=problem_module.n_objectives,
                solution_shape=shape,
                objective_function=problem_module.objective_function,
                generate_solution=problem_module.Generate_Solution,
                perturb=problem_module.Perturb,
                crossover=problem_module.Crossover,
                problem_params=problem_params,
                name=f"{problem_module.CODE}_warmup",
            )
            multi_solve(problem, algorithm=algo_name, **solve_kwargs)
        else:
            raise ValueError(f"Unknown algorithm kind '{kind}' for '{algo_name}' (expected 'mono' or 'multi').")

        if progress is not None:
            progress.update(warmup_key, advance=1)

    if progress is not None:
        progress.finish(warmup_key)


def _run_one_task(
    problem_module, instance_id: Any, instance: Any, algo_name: str, algo_spec: Dict[str, Any],
    ml_config: Dict[str, Any], verbose: bool = True, progress: Optional[ProgressReporter] = None,
) -> Tuple[Any, str, List[TrainingRecord], float]:
    """Runs one (instance, algorithm) pair to completion and returns its
    logged records. The single unit of work dispatched either sequentially
    (with live per-generation progress, `progress` given) or in parallel via
    joblib (where `progress` is always None, since a live rich.Progress
    display cannot cross process boundaries)."""
    kind = algo_spec["kind"]
    params = algo_spec.get("params", {})
    solve_kwargs = dict(algo_spec.get("solve_kwargs", {}))
    solve_kwargs.setdefault("show_progress", False)
    # Reproducible task: Python, NumPy AND Numba RNGs seeded from (instance,
    # algorithm); the solver draws its own generator seed from the seeded NumPy
    # RNG (seed=None), as in methods_comparison.
    seed_everything(_task_seed(instance_id, algo_name))
    solve_kwargs.setdefault("seed", None)

    problem_params = problem_module.build_problem_params(params, instance)
    shape = problem_module.solution_shape(instance)
    total_generations = solve_kwargs.get("generations")

    t_run_start = time.perf_counter()
    if verbose and progress is None:
        pop = solve_kwargs.get("population_size", "?")
        print(
            f"  [generate_training_data] instance={instance_id} algorithm={algo_name} "
            f"(population_size={pop}, generations={total_generations}) ...",
            flush=True,
        )

    gen_task_key = f"datagen::{instance_id}::{algo_name}"
    if progress is not None:
        progress.start(
            gen_task_key,
            f"    [{instance_id}/{algo_name}] gen 0/{total_generations}",
            total=total_generations,
        )

    run_records: List[TrainingRecord] = []
    if kind == "mono":
        log_callback = _make_mono_log_callback(
            run_records, instance_id, algo_name, instance, problem_module.feature_fn,
            ml_config["elite_log_fraction"], log_stride=ml_config.get("log_generation_stride", 1),
            total_generations=total_generations, verbose=verbose, t_start=t_run_start,
            progress=progress, gen_task_key=gen_task_key,
        )
        problem = MonoProblem(
            solution_shape=shape,
            objective_function=problem_module.objective_function,
            generate_solution=problem_module.Generate_Solution,
            perturb=problem_module.Perturb,
            crossover=problem_module.Crossover,
            problem_params=problem_params,
            name=f"{problem_module.CODE}_{instance_id}",
        )
        mono_solve(problem, algorithm=algo_name, log_callback=log_callback, **solve_kwargs)

    elif kind == "multi":
        log_callback = _make_multi_log_callback(
            run_records, instance_id, algo_name, instance, problem_module.feature_fn,
            ml_config["pareto_log_fraction"], log_stride=ml_config.get("log_generation_stride", 1),
            total_generations=total_generations, verbose=verbose, t_start=t_run_start,
            progress=progress, gen_task_key=gen_task_key,
        )
        problem = MultiProblem(
            n_objectives=problem_module.n_objectives,
            solution_shape=shape,
            objective_function=problem_module.objective_function,
            generate_solution=problem_module.Generate_Solution,
            perturb=problem_module.Perturb,
            crossover=problem_module.Crossover,
            problem_params=problem_params,
            name=f"{problem_module.CODE}_{instance_id}",
        )
        multi_solve(problem, algorithm=algo_name, log_callback=log_callback, **solve_kwargs)

    else:
        raise ValueError(f"Unknown algorithm kind '{kind}' for '{algo_name}' (expected 'mono' or 'multi').")

    run_elapsed = time.perf_counter() - t_run_start
    if progress is not None:
        progress.finish(gen_task_key)
    if verbose and progress is None:
        print(
            f"    [{instance_id}/{algo_name}] done in {run_elapsed:.1f}s "
            f"-- {len(run_records)} records logged",
            flush=True,
        )
    return instance_id, algo_name, run_records, run_elapsed


def _run_one_task_by_module_name(
    problem_module_name: str, instance_id: Any, instance: Any, algo_name: str, algo_spec: Dict[str, Any],
    ml_config: Dict[str, Any],
) -> Tuple[Any, str, List[TrainingRecord], float]:
    """Process-safe wrapper around `_run_one_task`, used only by the
    parallel dispatch path below: re-imports the problem module by its
    dotted name inside the worker process via `importlib`, rather than
    relying on the live module object being pickled across the process
    boundary. This is the standard-library-only, most portable option --
    cloudpickle can often pickle plain file-backed modules by reference too,
    but importing by name sidesteps that entirely."""
    problem_module = importlib.import_module(problem_module_name)
    _, _, run_records, run_elapsed = _run_one_task(
        problem_module, instance_id, instance, algo_name, algo_spec, ml_config,
        verbose=False, progress=None,
    )
    return instance_id, algo_name, run_records, run_elapsed


def _task_settings(problem_module, instance_id, algo_name, algo_spec, ml_config) -> Dict[str, Any]:
    """What a cached task must have been run with to be reused on resume."""
    return {"problem": problem_module.CODE, "instance": str(instance_id), "algorithm": algo_name,
            "spec": algo_spec, "seed": _task_seed(instance_id, algo_name),
            "logging": [ml_config.get("elite_log_fraction"), ml_config.get("pareto_log_fraction"),
                        ml_config.get("log_generation_stride", 1)]}


def generate_training_data(
    problem_module, ml_config: Dict[str, Any], verbose: bool = True,
    progress: Optional[ProgressReporter] = None, cache_dir: Optional[str] = None,
) -> TrainingDataset:
    """Runs every algorithm of `problem_module.ALGORITHMS` on every training
    instance with its fixed parameters, logging non-dominated solutions at every
    `log_generation_stride`-th generation, after a one-off JIT warm-up. The
    records of all tasks are merged into one `TrainingDataset`, ALWAYS in task
    order (instance, then algorithm), whatever the completion order, so a resumed
    or parallel run builds the same dataset.

    `ml_config["datagen_n_jobs"]` selects sequential (1) vs. parallel (joblib
    process pool) execution of the (instance, algorithm) tasks.

    Resume: with `cache_dir`, every finished task (its records and compute time)
    is saved there at once (dependency/resume.py) and reused on resume when it
    was run with the same settings; every task is seeded from (instance,
    algorithm), so the result does not depend on the interruption.

    The compute seconds of every task are kept in `dataset.task_seconds` (the
    cost of the training data, reported by the cost accounting)."""
    from dependency.resume import load_task, save_task, task_path

    dataset = TrainingDataset()
    instances = problem_module.generate_training_instances()
    n_jobs = ml_config.get("datagen_n_jobs", 1)
    t_wall = time.perf_counter()
    for instance_id, instance in instances:
        dataset.instances_by_id[instance_id] = instance

    tasks = [
        (instance_id, instance, algo_name, algo_spec)
        for instance_id, instance in instances
        for algo_name, algo_spec in problem_module.ALGORITHMS.items()
    ]
    results: Dict[int, Tuple[List[TrainingRecord], float]] = {}

    def _path(instance_id, algo_name):
        return task_path(cache_dir, "datagen", f"{instance_id}__{algo_name}.pkl")

    pending = []
    for k, (instance_id, instance, algo_name, algo_spec) in enumerate(tasks):
        cached = load_task(_path(instance_id, algo_name),
                           _task_settings(problem_module, instance_id, algo_name, algo_spec, ml_config))
        if cached is not None:
            results[k] = (cached["records"], cached["seconds"])
        else:
            pending.append(k)
    if results and verbose and progress is None:
        print(f"  [generate_training_data] resumed: {len(results)}/{len(tasks)} tasks loaded from the cache", flush=True)

    def _store(k, run_records, run_elapsed):
        instance_id, _, algo_name, algo_spec = tasks[k]
        results[k] = (run_records, run_elapsed)
        save_task(_path(instance_id, algo_name),
                  _task_settings(problem_module, instance_id, algo_name, algo_spec, ml_config),
                  {"records": run_records, "seconds": run_elapsed})

    overall_key = "datagen"
    if progress is not None:
        progress.start(overall_key, "Generating training data", total=len(tasks))
        progress.update(overall_key, completed=len(results))

    if pending:
        n_algos = len(problem_module.ALGORITHMS)
        if verbose and progress is None:
            print(f"  [generate_training_data] warming up JIT for {n_algos} algorithm(s) ...", flush=True)
        t_warmup_start = time.perf_counter()
        _warmup_algorithms(problem_module, instances, progress=progress)
        dataset.warmup_seconds = time.perf_counter() - t_warmup_start
        if verbose and progress is None:
            print(f"  [generate_training_data] warm-up done in {dataset.warmup_seconds:.1f}s", flush=True)

    if pending and n_jobs in (1, None):
        for k in pending:
            instance_id, instance, algo_name, algo_spec = tasks[k]
            _, _, run_records, run_elapsed = _run_one_task(
                problem_module, instance_id, instance, algo_name, algo_spec, ml_config,
                verbose=verbose, progress=progress,
            )
            _store(k, run_records, run_elapsed)
            if progress is not None:
                progress.update(overall_key, advance=1)
    elif pending:
        if verbose and progress is None:
            print(
                f"  [generate_training_data] running {len(pending)} (instance, algorithm) tasks "
                f"across n_jobs={n_jobs} worker processes (task-level progress) ...",
                flush=True,
            )
        parallel_kwargs = dict(n_jobs=n_jobs, backend="loky")
        task_iter = (
            delayed(_run_one_task_by_module_name)(
                problem_module.__name__, tasks[k][0], tasks[k][1], tasks[k][2], tasks[k][3], ml_config,
            )
            for k in pending
        )
        try:
            result_iter = Parallel(return_as="generator", **parallel_kwargs)(task_iter)
            incremental = True
        except TypeError:
            result_iter = Parallel(verbose=(10 if (verbose and progress is None) else 0), **parallel_kwargs)(task_iter)
            incremental = False

        index_of = {(str(tasks[k][0]), tasks[k][2]): k for k in pending}
        n_done = len(results)
        for instance_id, algo_name, run_records, run_elapsed in result_iter:
            _store(index_of[(str(instance_id), algo_name)], run_records, run_elapsed)
            n_done += 1
            if incremental:
                if progress is not None:
                    progress.update(overall_key, advance=1)
                elif verbose:
                    print(
                        f"    [{instance_id}/{algo_name}] done in {run_elapsed:.1f}s "
                        f"-- {len(run_records)} records logged ({n_done}/{len(tasks)}) ...",
                        flush=True,
                    )
        if progress is not None and not incremental:
            progress.update(overall_key, completed=len(tasks))

    if progress is not None:
        progress.finish(overall_key)

    for k, (instance_id, _, algo_name, _) in enumerate(tasks):
        run_records, run_elapsed = results[k]
        dataset.records.extend(run_records)
        dataset.task_seconds[f"{instance_id}|{algo_name}"] = float(run_elapsed)
    dataset.wall_seconds = time.perf_counter() - t_wall
    return dataset


def build_feature_target_pairs(
    dataset: TrainingDataset, problem_module,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Step 2 of the pipeline: for every objective that actually needs a
    surrogate (`problem_module.SURROGATE_OBJECTIVE_NAMES` -- falls back to
    every objective if a problem doesn't define this), builds
    (X, y, groups, classes): each row of X is the feature vector already
    extracted for that record during logging, y is that objective's REAL
    (Monte Carlo) value computed by the metaheuristic run, `groups` the id of
    the training instance the schedule was logged on, and `classes` its size
    class (`problem_module.instance_class(id)` when defined, else the id's
    prefix before the last '_'). The groups are what the train/validation split
    and the cross-validation folds are made of (whole instances). Deterministic
    objectives that need no surrogate (e.g. Chapter 5's undelivered energy) are
    excluded here -- they are computed exactly via
    `problem_module.exact_objective_values`, not learned."""
    objective_names = problem_module.objective_names
    surrogate_names = getattr(problem_module, "SURROGATE_OBJECTIVE_NAMES", objective_names)
    class_of = getattr(problem_module, "instance_class", lambda iid: str(iid).rsplit("_", 1)[0])
    feats = [rec.features for rec in dataset.records]
    groups = np.asarray([str(rec.instance_id) for rec in dataset.records])
    classes = np.asarray([class_of(rec.instance_id) for rec in dataset.records])
    X = np.vstack(feats) if feats else np.empty((0, 0))
    out = {}
    for name in surrogate_names:
        k = objective_names.index(name)
        y = np.asarray([float(rec.objective_values[k]) for rec in dataset.records], dtype=np.float64)
        out[name] = (X, y, groups, classes)
    return out


def build_record_metadata(dataset: TrainingDataset, problem_module) -> Dict[str, np.ndarray]:
    """Per record (same row order as `build_feature_target_pairs`): training
    instance, size class, algorithm, generation, the generation as a fraction of
    its run's last logged generation, and every objective's value (learned and
    exact) -- what the per-class, late-generation, ranking and design-set reports
    are computed from."""
    class_of = getattr(problem_module, "instance_class", lambda iid: str(iid).rsplit("_", 1)[0])
    recs = dataset.records
    groups = np.asarray([str(r.instance_id) for r in recs])
    algorithms = np.asarray([r.algorithm for r in recs])
    generation = np.asarray([int(r.generation) for r in recs])
    last: Dict[Tuple[str, str], int] = {}
    for g, a, gen in zip(groups, algorithms, generation):
        last[(g, a)] = max(last.get((g, a), 0), int(gen))
    gen_frac = np.asarray([gen / last[(g, a)] if last[(g, a)] > 0 else 1.0
                           for g, a, gen in zip(groups, algorithms, generation)], dtype=np.float64)
    objectives = np.vstack([np.asarray(r.objective_values, dtype=np.float64) for r in recs]) if recs \
        else np.empty((0, problem_module.n_objectives))
    return {"groups": groups, "classes": np.asarray([class_of(r.instance_id) for r in recs]),
            "algorithms": algorithms, "generation": generation, "generation_fraction": gen_frac,
            "objectives": objectives}
