"""
The resampling protocol: the engine of this framework.

Per instance, repeated R times: draw ONE realization and apply it to every
schedule of EVERY regime's front, score the realized schedules, rebuild each
regime's non-dominated front from the realized vectors, and compute a
hypervolume for each regime that is comparable across regimes (one ideal/nadir
shared by all regimes for this instance and replication, via
`compute_hypervolume_batch`).

The realized non-dominated fronts are kept as well (one per regime and
replication): the realized-objective, promise-vs-delivery, indicator and
attainment-surface reports are computed from them.

Run level: the front of every single evaluation run is realized under the same
draw too, and all run fronts of all regimes share one normalization per
replication ("hv_runs"); the run-level analysis
(dependency/reporting/run_level.py) compares the regimes on these, so that the
variability of the search between runs enters the tests.

"The same realization applied to every front" is implemented as COMMON RANDOM
NUMBERS: the RNG is reseeded to the same value right before every schedule's
`realize_one_draw` call of a replication. When the realizer draws per vehicle
(Chapter 6) the realization is identical for every schedule; a realizer drawing per
charging interval would only share the random stream between schedules with
different interval counts.
"""

from __future__ import annotations

import importlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed

from dependency.algorithms.multi_objective_solving_module import find_pareto_front_indexes
from dependency.evaluation.evaluator import RegimeFront
from dependency.resampling.hypervolume import compute_hypervolume_batch
from dependency.reporting.progress import ProgressReporter


def _nondominated_front(objective_vectors: List[np.ndarray]) -> List[List[float]]:
    if not objective_vectors:
        return []
    arr = np.asarray(objective_vectors, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    nd_idx = find_pareto_front_indexes(arr)
    return arr[nd_idx].tolist()


def run_resampling_protocol(
    problem_module,
    instance: Any,
    instance_id: int,
    fronts: Dict[str, RegimeFront],
    n_replications: int,
    seed_base: int,
    eps: float = 1e-3,
    progress: Optional[ProgressReporter] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, list]]:
    """{"hv": {regime: [hv_r0, ...]}, "realized": {regime: [front_r0, ...]}} --
    R comparable hypervolumes and R realized non-dominated fronts per regime for
    this ONE instance. An empty planned front gives NaN hypervolumes and empty
    realized fronts for that regime."""
    regimes = list(fronts)
    task_key = f"resampling::{instance_id}"
    if progress is not None:
        progress.start(task_key, f"Resampling -- instance {instance_id}", total=n_replications)

    hv: Dict[str, List[float]] = {r: [] for r in regimes}
    realized_fronts: Dict[str, List[List[List[float]]]] = {r: [] for r in regimes}
    hv_runs: Dict[str, List[List[float]]] = {r: [] for r in regimes}      # [replication][run]

    t_start = time.perf_counter()
    for rep in range(n_replications):
        seed_r = seed_base + rep
        nd: Dict[str, List[List[float]]] = {}
        for regime in regimes:
            schedules = fronts[regime].schedules
            scored = [problem_module.score_realized(instance, s, problem_module.realize_one_draw(instance, s, seed_r))
                      for s in schedules]
            nd[regime] = _nondominated_front(scored)
            realized_fronts[regime].append(nd[regime])
        hv_rep = compute_hypervolume_batch({r: [nd[r]] for r in regimes}, eps=eps)
        for regime in regimes:
            hv[regime].append(hv_rep[regime][0] if nd[regime] else float("nan"))
        run_nd = {}
        for regime in regimes:
            run_nd[regime] = []
            for rf in getattr(fronts[regime], "run_fronts", None) or []:
                scored = [problem_module.score_realized(instance, s, problem_module.realize_one_draw(instance, s, seed_r))
                          for s in rf["schedules"]]
                run_nd[regime].append(_nondominated_front(scored))
        if all(run_nd[r] for r in regimes):
            hv_r = compute_hypervolume_batch({r: run_nd[r] for r in regimes}, eps=eps)
            for regime in regimes:
                hv_runs[regime].append([float(x) for x in hv_r[regime]])
        if progress is not None:
            progress.update(task_key, advance=1)

    if progress is not None:
        progress.finish(task_key)
    if verbose and progress is None:
        means = ", ".join(f"{r}={np.nanmean(hv[r]):.4g}" for r in regimes)
        print(f"  [resampling] instance={instance_id} done in {time.perf_counter() - t_start:.1f}s "
              f"-- mean HV {means}", flush=True)
    return {"hv": hv, "realized": realized_fronts, "hv_runs": hv_runs}


def _run_resampling_protocol_by_module_name(
    problem_module_name: str, instance: Any, instance_id: int, fronts: Dict[str, RegimeFront],
    n_replications: int, seed_base: int, eps: float,
) -> Tuple[int, Dict[str, Dict[str, list]]]:
    """Process-safe wrapper around `run_resampling_protocol` for the parallel
    path: re-imports the problem module by its dotted name inside the worker."""
    problem_module = importlib.import_module(problem_module_name)
    result = run_resampling_protocol(problem_module, instance, instance_id, fronts,
                                     n_replications, seed_base, eps=eps, progress=None, verbose=False)
    return instance_id, result


def run_resampling_for_all_instances(
    problem_module,
    instances_by_seed: Dict[int, Any],
    fronts_by_regime: Dict[str, Dict[int, RegimeFront]],
    n_replications: int,
    seed_base: int,
    eps: float = 1e-3,
    n_jobs: int = 1,
    progress: Optional[ProgressReporter] = None,
    verbose: bool = True,
) -> Tuple[Dict[int, Dict[str, List[float]]], Dict[int, Dict[str, list]]]:
    """Runs `run_resampling_protocol` for every instance (one worker process per
    instance when n_jobs != 1). Returns (per_instance_hv, per_instance_realized):
    {instance: {regime: [R hypervolumes]}} and {instance: {regime: [R fronts]}}."""
    seeds = list(instances_by_seed.keys())

    def _fronts_of(seed):
        return {regime: fronts_by_regime[regime][seed] for regime in fronts_by_regime}

    results: Dict[int, Dict[str, Dict[str, list]]] = {}
    if n_jobs in (1, None):
        for seed in seeds:
            results[seed] = run_resampling_protocol(problem_module, instances_by_seed[seed], seed, _fronts_of(seed),
                                                    n_replications, seed_base, eps=eps, progress=progress,
                                                    verbose=verbose)
    else:
        if verbose and progress is None:
            print(f"  [resampling] running {len(seeds)} instance resamplings across n_jobs={n_jobs} "
                  f"worker processes ...", flush=True)
        overall_key = "resampling::overall"
        if progress is not None:
            progress.start(overall_key, "Resampling (all instances)", total=len(seeds))
        task_iter = (
            delayed(_run_resampling_protocol_by_module_name)(
                problem_module.__name__, instances_by_seed[seed], seed, _fronts_of(seed),
                n_replications, seed_base, eps,
            )
            for seed in seeds
        )
        parallel_kwargs = dict(n_jobs=n_jobs, backend="loky")
        try:
            result_iter = Parallel(return_as="generator", **parallel_kwargs)(task_iter)
            incremental = True
        except TypeError:
            result_iter = Parallel(verbose=(10 if (verbose and progress is None) else 0), **parallel_kwargs)(task_iter)
            incremental = False
        for n_done, (instance_id, result) in enumerate(result_iter, start=1):
            results[instance_id] = result
            if incremental:
                if progress is not None:
                    progress.update(overall_key, advance=1)
                elif verbose:
                    means = ", ".join(f"{r}={np.nanmean(v):.4g}" for r, v in result["hv"].items())
                    print(f"    [resampling] instance={instance_id} done -- mean HV {means} "
                          f"({n_done}/{len(seeds)})", flush=True)
        if progress is not None:
            if not incremental:
                progress.update(overall_key, completed=len(seeds))
            progress.finish(overall_key)

    per_instance_hv = {seed: results[seed]["hv"] for seed in seeds}
    per_instance_realized = {seed: results[seed]["realized"] for seed in seeds}
    # run-level hypervolumes, read by main.py (the return signature is unchanged)
    run_resampling_for_all_instances.last_hv_runs = {seed: results[seed].get("hv_runs", {}) for seed in seeds}
    return per_instance_hv, per_instance_realized
