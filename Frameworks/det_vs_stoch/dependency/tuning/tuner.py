"""
Lightweight, single-algorithm tuning for one regime: a scaled-down version of
methods_comparison's tuning. This framework only ever tunes ONE, already-named algorithm
(`problem_module.BEST_ALGORITHM`) per regime, per instance; each regime gets its
own complete, independent tuning pass (hyperparameters are never shared between
regimes).

Scoring of a combination: as methods_comparison's `tuning.mo_score = "shared"`, every
front returned on ONE instance (all combinations x all runs) is normalised with
the same ideal/nadir (`compute_hypervolume_batch`), so the hypervolumes of the
combinations are comparable; a combination's score is the mean hypervolume of
its runs, and the best combination is kept per instance. (Normalising each
front on its own extent made combinations incomparable: a narrow front can fill
its own box.)

Every (instance, combination) is run `n_runs_per_combo` times; with `n_jobs`
!= 1 those tasks are dispatched across joblib worker processes, the scores
being computed in the parent once all fronts of an instance are in.

Resume: with `cache_dir`, every finished (instance, combination) task is saved
there (dependency/resume.py) and skipped when the run is resumed.

Cost: the compute time of every task (seconds spent in its runs) is summed per
instance -- the per-instance tuning cost reported by the cost-ratio tables.
"""

from __future__ import annotations

import importlib
import itertools
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed

from dependency.resampling.hypervolume import compute_hypervolume_batch
from dependency.reporting.progress import ProgressReporter
from dependency.resume import load_task, save_task, task_path


def _expand_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of a {param_name: [values]} grid into a flat list of
    {param_name: value} combos (methods_comparison's grid-expansion convention)."""
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*[param_grid[k] for k in keys])]


def _run_combo(problem_module, instance, regime: str, combo: Dict[str, Any],
               n_runs_per_combo: int, seed_base: int) -> Tuple[List[List[List[float]]], float]:
    """(fronts of `n_runs_per_combo` seeded runs of a combo, seconds spent)."""
    fronts = []
    t0 = time.perf_counter()
    for run_idx in range(n_runs_per_combo):
        params = dict(combo)
        params["seed"] = seed_base + run_idx
        result = problem_module.run_algorithm(instance, params, regime, progress_record=False)
        fronts.append(result["pareto_objectives"])
    return fronts, time.perf_counter() - t0


def _run_combo_by_module_name(
    problem_module_name: str, seed: int, combo_idx: int, instance: Any, regime: str,
    combo: Dict[str, Any], n_runs_per_combo: int, seed_base: int,
) -> Tuple[int, int, List[List[List[float]]], float]:
    """Process-safe wrapper of `_run_combo` for the parallel path: re-imports the
    problem module by name inside the worker."""
    problem_module = importlib.import_module(problem_module_name)
    fronts, seconds = _run_combo(problem_module, instance, regime, combo, n_runs_per_combo, seed_base)
    return seed, combo_idx, fronts, seconds


def _best_combo(fronts_by_combo: Dict[int, List[List[List[float]]]], combos: List[Dict[str, Any]]):
    """(best combo, its score, all scores) with one shared normalisation for every
    front of the instance."""
    hv = compute_hypervolume_batch({str(ci): fronts for ci, fronts in fronts_by_combo.items()})
    scores = {ci: float(np.nanmean(hv[str(ci)])) if len(hv[str(ci)]) else float("-inf")
              for ci in fronts_by_combo}
    best = max(scores, key=lambda ci: scores[ci])
    return combos[best], scores[best], scores


def tune_one_regime(
    problem_module,
    regime: str,
    instances_by_seed: Dict[int, Any],
    n_runs_per_combo: int,
    seed_base: int,
    n_jobs: int = 1,
    progress: Optional[ProgressReporter] = None,
    verbose: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, float]]:
    """For EACH instance independently: run every combo of
    `problem_module.param_grid_for_best_algorithm()` `n_runs_per_combo` times,
    score the combos by mean hypervolume in one normalisation shared by all the
    fronts of that instance, keep the best combo.
    Returns ({seed: best_params}, {seed: tuning compute seconds})."""
    grid = problem_module.param_grid_for_best_algorithm()
    combos = _expand_grid(grid)
    seeds = list(instances_by_seed.keys())
    total_tasks = len(seeds) * len(combos)

    task_key = f"tuning::{regime}"
    if progress is not None:
        progress.start(task_key, f"Tuning [bold]{regime}[/bold]", total=total_tasks)

    fronts: Dict[int, Dict[int, List]] = {seed: {} for seed in seeds}
    seconds: Dict[int, float] = {seed: 0.0 for seed in seeds}
    t_start = time.perf_counter()
    completed = 0

    def _settings(seed, combo):
        return {"regime": regime, "instance": seed, "combo": combo,
                "n_runs_per_combo": n_runs_per_combo, "seed_base": seed_base + seed * 1000}

    def _path(seed, combo_idx):
        return task_path(cache_dir, "tuning", regime, f"instance{seed}_combo{combo_idx}.pkl")

    pending = []
    for seed in seeds:
        for combo_idx, combo in enumerate(combos):
            cached = load_task(_path(seed, combo_idx), _settings(seed, combo))
            if cached is not None:
                fronts[seed][combo_idx] = cached["fronts"]
                seconds[seed] += cached["seconds"]
                completed += 1
            else:
                pending.append((seed, combo_idx, combo))
    if completed and verbose and progress is None:
        print(f"  [tuning::{regime}] resumed: {completed}/{total_tasks} tasks loaded from the cache", flush=True)
    if progress is not None and completed:
        progress.update(task_key, completed=completed)

    def _store(seed, combo_idx, run_fronts, run_seconds):
        fronts[seed][combo_idx] = run_fronts
        seconds[seed] += run_seconds
        save_task(_path(seed, combo_idx), _settings(seed, combos[combo_idx]),
                  {"fronts": run_fronts, "seconds": run_seconds})

    if n_jobs in (1, None):
        for seed, combo_idx, combo in pending:
            run_fronts, run_seconds = _run_combo(problem_module, instances_by_seed[seed], regime, combo,
                                                 n_runs_per_combo, seed_base + seed * 1000)
            _store(seed, combo_idx, run_fronts, run_seconds)
            completed += 1
            if progress is not None:
                progress.update(task_key, completed=completed)
            elif verbose:
                print(f"  [tuning::{regime}] instance={seed} combo={combo} ({completed}/{total_tasks})", flush=True)
    elif pending:
        if verbose and progress is None:
            print(f"  [tuning::{regime}] running {len(pending)} (instance, combo) tasks across n_jobs={n_jobs} "
                  f"worker processes ...", flush=True)
        task_iter = (
            delayed(_run_combo_by_module_name)(
                problem_module.__name__, seed, combo_idx, instances_by_seed[seed], regime,
                combo, n_runs_per_combo, seed_base + seed * 1000,
            )
            for seed, combo_idx, combo in pending
        )
        parallel_kwargs = dict(n_jobs=n_jobs, backend="loky")
        try:
            result_iter = Parallel(return_as="generator", **parallel_kwargs)(task_iter)
            incremental = True
        except TypeError:
            result_iter = Parallel(verbose=(10 if (verbose and progress is None) else 0), **parallel_kwargs)(task_iter)
            incremental = False
        for seed, combo_idx, run_fronts, run_seconds in result_iter:
            _store(seed, combo_idx, run_fronts, run_seconds)
            completed += 1
            if incremental:
                if progress is not None:
                    progress.update(task_key, completed=completed)
                elif verbose:
                    print(f"  [tuning::{regime}] instance={seed} combo={combos[combo_idx]} "
                          f"({completed}/{total_tasks})", flush=True)
        if progress is not None and not incremental:
            progress.update(task_key, completed=total_tasks)

    best_combo: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        best_combo[seed], score, _ = _best_combo(fronts[seed], combos)
        if verbose and progress is None:
            print(f"  [tuning::{regime}] instance={seed}: best combo {best_combo[seed]} "
                  f"(mean shared-normalisation HV {score:.4g})", flush=True)

    if progress is not None:
        progress.finish(task_key)
    if verbose and progress is None:
        print(f"  [tuning::{regime}] done in {time.perf_counter() - t_start:.1f}s", flush=True)
    return best_combo, seconds
