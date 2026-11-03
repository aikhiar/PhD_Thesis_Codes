"""
Evaluation: re-runs the winning tuned combo `n_eval_runs` times per
(instance, regime), pools every run's (objective, schedule) pairs, and
keeps the non-dominated union -- WITH the actual schedule objects, not
just objective vectors, since the resampling protocol needs to re-realize
and re-score real schedules, not abstract points.

`evaluate_all_instances` dispatches one `evaluate_one_regime` call per
instance, either sequentially (n_jobs=1, full live per-run progress) or
across joblib worker processes (each instance's whole n_eval_runs-run
evaluation is one task -- coarser than parallelising the runs themselves,
but keeps per-instance JIT warm-up and RegimeFront construction simple and
self-contained per worker).

Resume: with `cache_dir`, every evaluated instance is cached as soon as it is
done (dependency/resume.py) and skipped when the run is resumed.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed

from dependency.algorithms.multi_objective_solving_module import find_pareto_front_indexes
from dependency.reporting.progress import ProgressReporter
from dependency.resume import load_task, save_task, task_path


@dataclass
class RegimeFront:
    instance_id: int
    objectives: List[List[float]] = field(default_factory=list)
    schedules: List[Any] = field(default_factory=list)
    # the front of every evaluation run, kept separately for the run-level
    # analysis ({"objectives": [...], "schedules": [...]} per run, in run order)
    run_fronts: List[Dict[str, list]] = field(default_factory=list)

    def __post_init__(self):
        assert len(self.objectives) == len(self.schedules), (
            f"RegimeFront invariant violated for instance {self.instance_id}: "
            f"{len(self.objectives)} objectives vs {len(self.schedules)} schedules"
        )


def _warmup_once(problem_module, instance, regime: str, best_params: Dict[str, Any], seed: int) -> None:
    """One throwaway run at a shrunk population/generations budget, purely
    to trigger numba JIT compilation ahead of the real timed evaluation
    runs -- same pattern as surrogate_trainer's `_warmup_algorithms`.
    Discarded entirely; never contributes to the front."""
    warmup_params = dict(best_params)
    warmup_params["population_size"] = min(6, warmup_params.get("population_size", 6))
    warmup_params["generations"] = 1
    warmup_params["seed"] = seed
    problem_module.run_algorithm(instance, warmup_params, regime, progress_record=False)


def evaluate_one_regime(
    problem_module,
    regime: str,
    instance: Any,
    instance_id: int,
    best_params: Dict[str, Any],
    n_eval_runs: int,
    seed_base: int,
    warmup: bool = True,
    progress: Optional[ProgressReporter] = None,
    verbose: bool = True,
) -> RegimeFront:
    task_key = f"eval::{regime}::{instance_id}"
    if progress is not None:
        progress.start(task_key, f"Evaluating [bold]{regime}[/bold] -- instance {instance_id}", total=n_eval_runs)

    if warmup:
        _warmup_once(problem_module, instance, regime, best_params, seed=seed_base - 1)

    pooled_objectives: List[List[float]] = []
    pooled_schedules: List[Any] = []
    run_fronts: List[Dict[str, list]] = []

    t_start = time.perf_counter()
    for run_idx in range(n_eval_runs):
        params = dict(best_params)
        params["seed"] = seed_base + run_idx
        result = problem_module.run_algorithm(instance, params, regime, progress_record=False)
        pooled_objectives.extend(result["pareto_objectives"])
        pooled_schedules.extend(result["pareto_front"])
        run_fronts.append({"objectives": list(result["pareto_objectives"]), "schedules": list(result["pareto_front"])})

        if progress is not None:
            progress.update(task_key, advance=1)

    if progress is not None:
        progress.finish(task_key)
    if verbose and progress is None:
        print(
            f"  [eval::{regime}] instance={instance_id} done in {time.perf_counter() - t_start:.1f}s "
            f"-- {len(pooled_objectives)} pooled points",
            flush=True,
        )

    if not pooled_objectives:
        return RegimeFront(instance_id=instance_id, objectives=[], schedules=[])

    pooled_arr = np.asarray(pooled_objectives, dtype=np.float64)
    nd_idx = find_pareto_front_indexes(pooled_arr)
    nd_objectives = pooled_arr[nd_idx].tolist()
    nd_schedules = [pooled_schedules[i] for i in nd_idx]

    return RegimeFront(instance_id=instance_id, objectives=nd_objectives, schedules=nd_schedules,
                       run_fronts=run_fronts)


def _evaluate_one_regime_by_module_name(
    problem_module_name: str, regime: str, instance: Any, instance_id: int,
    best_params: Dict[str, Any], n_eval_runs: int, seed_base: int, warmup: bool,
) -> Tuple[int, RegimeFront, float]:
    """Process-safe wrapper around `evaluate_one_regime`, used only by the
    parallel dispatch path below: re-imports the problem module by its
    dotted name inside the worker process, and always runs silently
    (progress=None) since a live rich.Progress display cannot cross process
    boundaries. Returns (instance id, front, compute seconds)."""
    problem_module = importlib.import_module(problem_module_name)
    t0 = time.perf_counter()
    front = evaluate_one_regime(
        problem_module, regime, instance, instance_id, best_params,
        n_eval_runs, seed_base, warmup=warmup, progress=None, verbose=False,
    )
    return instance_id, front, time.perf_counter() - t0


def evaluate_all_instances(
    problem_module,
    regime: str,
    instances_by_seed: Dict[int, Any],
    best_params_by_seed: Dict[int, Dict[str, Any]],
    n_eval_runs: int,
    seed_base: int,
    warmup: bool = True,
    n_jobs: int = 1,
    progress: Optional[ProgressReporter] = None,
    verbose: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[Dict[int, RegimeFront], Dict[int, float]]:
    """Runs `evaluate_one_regime` for every instance. `n_jobs=1` (default)
    keeps the sequential path with full live per-run progress; anything else
    dispatches one worker process per instance (each worker still runs that
    instance's `n_eval_runs` runs sequentially).

    Resume: with `cache_dir`, the front of every evaluated instance is saved
    there as soon as it is done (dependency/resume.py) and loaded instead of
    re-run on resume, when it was obtained with the same parameters.
    Returns ({seed: front}, {seed: evaluation compute seconds})."""
    seeds = list(instances_by_seed.keys())
    fronts: Dict[int, RegimeFront] = {}
    seconds: Dict[int, float] = {}

    def _settings(seed):
        return {"regime": regime, "instance": seed, "best_params": best_params_by_seed[seed],
                "n_eval_runs": n_eval_runs, "seed_base": seed_base, "warmup": warmup}

    def _path(seed):
        return task_path(cache_dir, "evaluation", regime, f"instance{seed}.pkl")

    pending = []
    for seed in seeds:
        cached = load_task(_path(seed), _settings(seed))
        if cached is not None:
            fronts[seed], seconds[seed] = cached["front"], cached["seconds"]
        else:
            pending.append(seed)
    if fronts and verbose and progress is None:
        print(f"  [eval::{regime}] resumed: {len(fronts)}/{len(seeds)} instances loaded from the cache", flush=True)

    def _store(seed, front, run_seconds):
        fronts[seed], seconds[seed] = front, run_seconds
        save_task(_path(seed), _settings(seed), {"front": front, "seconds": run_seconds})

    if n_jobs in (1, None):
        for seed in pending:
            t0 = time.perf_counter()
            front = evaluate_one_regime(
                problem_module, regime, instances_by_seed[seed], seed, best_params_by_seed[seed],
                n_eval_runs, seed_base, warmup=warmup, progress=progress, verbose=verbose,
            )
            _store(seed, front, time.perf_counter() - t0)
        return {s: fronts[s] for s in seeds}, {s: seconds[s] for s in seeds}

    if not pending:
        return {s: fronts[s] for s in seeds}, {s: seconds[s] for s in seeds}

    if verbose and progress is None:
        print(
            f"  [eval::{regime}] running {len(pending)} instance evaluations "
            f"across n_jobs={n_jobs} worker processes ...",
            flush=True,
        )
    overall_key = f"eval::{regime}::overall"
    if progress is not None:
        progress.start(overall_key, f"Evaluating [bold]{regime}[/bold] (all instances)", total=len(seeds))
        progress.update(overall_key, completed=len(seeds) - len(pending))

    task_iter = (
        delayed(_evaluate_one_regime_by_module_name)(
            problem_module.__name__, regime, instances_by_seed[seed], seed,
            best_params_by_seed[seed], n_eval_runs, seed_base, warmup,
        )
        for seed in pending
    )
    parallel_kwargs = dict(n_jobs=n_jobs, backend="loky")
    try:
        result_iter = Parallel(return_as="generator", **parallel_kwargs)(task_iter)
        incremental = True
    except TypeError:
        result_iter = Parallel(verbose=(10 if (verbose and progress is None) else 0), **parallel_kwargs)(task_iter)
        incremental = False

    n_done = len(seeds) - len(pending)
    for instance_id, front, run_seconds in result_iter:
        _store(instance_id, front, run_seconds)
        n_done += 1
        if incremental:
            if progress is not None:
                progress.update(overall_key, advance=1)
            elif verbose:
                # the final non-dominated front size (post-filter), not the
                # pooled count printed by evaluate_one_regime (pre-filter)
                print(
                    f"    [eval::{regime}] instance={instance_id} done "
                    f"-- {len(front.objectives)} non-dominated points ({n_done}/{len(seeds)})",
                    flush=True,
                )
    if progress is not None:
        if not incremental:
            progress.update(overall_key, completed=len(seeds))
        progress.finish(overall_key)

    return {s: fronts[s] for s in seeds}, {s: seconds[s] for s in seeds}
