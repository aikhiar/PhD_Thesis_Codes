"""
det_vs_stoch/main.py -- runs the deterministic-versus-stochastic comparison of
one chapter of the thesis (Chapters 2, 3, 4 and 6):

  1. For each of the two regimes of the chapter's REGIMES (the deterministic
     planner, the reference, and the stochastic planner): build the instances,
     run an independent tuning pass of the chapter's BEST_ALGORITHM (MOCS), then
     evaluate the tuned combination to get a non-dominated front (WITH schedules)
     per instance.
  2. For each instance: the R-replication resampling protocol (draw one
     realization, apply it to both regimes' schedules, score them, rebuild each
     regime's realized front, compute comparable hypervolumes).
  3. Report (dependency/reporting/reporting.py): what the chapter reports, as
     set by its REPORT -- realized hypervolume per instance and per size class
     with the per-instance tests; for Chapter 6 also the across-instance and
     run-level tests, promise vs delivery and the realized fronts.
  4. Persist everything under runs/det_stoch_<code>_<8charkey>/, with a manifest
     (configuration, instances, library versions, machine) and run.log.

Resume: every finished tuning and evaluation task is cached under
<run>/cache/; `--resume <run folder>` restarts an interrupted run from there,
with identical results (every task is seeded from its own settings).

This framework is standalone: every algorithm and math utility it needs is a
physical copy under dependency/, and every problem model (problems/chapter2/,
chapter3/, chapter4/, chapter6/) is the same folder as in methods_comparison.
(Chapter 5 has no deterministic-vs-stochastic study.)

    python main.py                         # Chapters 2, 3, 4 and 6
    python main.py chapter6                # one chapter, by code
    python main.py --estimate              # expected duration, without running
    python main.py --resume runs/det_stoch_chapter6_ab12cd34
"""

import argparse
import contextlib
import json
import os
import random
import string
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dependency.requirements import ensure_requirements  # noqa: E402  (standard library only)

ensure_requirements()                                     # checks, and installs if missing, the required packages

import joblib

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dependency.config.config import RESAMPLING_CONFIG
from dependency.reporting.progress import ProgressReporter, make_progress, print_config_panel
from dependency.reporting.reporting import generate_report
from dependency.runlog import RunLog, environment
from dependency.tuning.tuner import _expand_grid, tune_one_regime
from dependency.evaluation.evaluator import evaluate_all_instances
from dependency.resampling.protocol import run_resampling_for_all_instances

REPORTING_ROOT = os.path.join(str(_ROOT), "runs")


def _random_key(n: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(n))


@contextlib.contextmanager
def _null_progress_ctx():
    yield None


def _add_wallclock(cache_dir: str, regime: str, phase: str, seconds: float) -> float:
    """Wall-clock of a phase accumulated over the sessions of a run (a resumed
    run adds its own time to the interrupted sessions'); returns the total."""
    path = os.path.join(cache_dir, "wallclock.json")
    os.makedirs(cache_dir, exist_ok=True)
    wall = json.load(open(path)) if os.path.exists(path) else {}
    wall.setdefault(regime, {}).setdefault(phase, 0.0)
    wall[regime][phase] += seconds
    with open(path + ".tmp", "w") as f:
        json.dump(wall, f)
    os.replace(path + ".tmp", path)
    return wall[regime][phase]


def _report_settings(problem_module):
    """(statistics, report) of the chapter: config.py, then the chapter's REPORT."""
    report = {**RESAMPLING_CONFIG["report"], **getattr(problem_module, "REPORT", {})}
    stats = dict(RESAMPLING_CONFIG["statistics"], correction=report["correction"])
    return stats, report


def _n_eval_runs(problem_module) -> int:
    """Evaluation runs per (instance, regime): the problem's N_EVAL_RUNS, if it
    keeps the protocol of its chapter, else the value of config.py."""
    return getattr(problem_module, "N_EVAL_RUNS", RESAMPLING_CONFIG["evaluation"]["n_eval_runs"])


def _regimes(problem_module):
    regimes = list(problem_module.REGIMES)
    if len(regimes) != 2:
        raise ValueError(f"{problem_module.CODE}: REGIMES has two regimes, the reference (deterministic) "
                         f"and the challenger (stochastic)")
    labels = getattr(problem_module, "REGIME_LABELS", {})
    return regimes, {r: labels.get(r, r.replace("_", " ").capitalize()) for r in regimes}


def run_problem(
    problem_module,
    reporting_root: str = REPORTING_ROOT,
    instance_seeds_override: Optional[list] = None,
    param_grid_override: Optional[Dict[str, Dict[str, Any]]] = None,
    verbose: bool = True,
    show_progress: bool = True,
    resume_dir: Optional[str] = None,
) -> str:
    """Runs the whole comparison for one problem module; returns the run folder.
    `instance_seeds_override` / `param_grid_override` ({regime: grid}) run a
    cheap pilot; `resume_dir` resumes an interrupted run from its cache."""
    if resume_dir:
        problem_dir = os.path.abspath(resume_dir)
    else:
        problem_dir = os.path.join(reporting_root, f"det_stoch_{problem_module.CODE}_{_random_key()}")
    os.makedirs(problem_dir, exist_ok=True)
    cache_dir = os.path.join(problem_dir, "cache")
    log = RunLog(problem_dir)
    regimes, labels = _regimes(problem_module)
    seeds = instance_seeds_override or problem_module.generate_instance_seeds()
    n_jobs = RESAMPLING_CONFIG["parallel"]["n_jobs"]

    manifest = {
        "code": problem_module.CODE, "best_algorithm": problem_module.BEST_ALGORITHM,
        "regimes": {r: {"label": labels[r], **{k: v for k, v in problem_module.REGIMES[r].items()}} for r in regimes},
        "instances": list(seeds), "param_grid": problem_module.param_grid_for_best_algorithm(),
        "param_grid_override": param_grid_override, "resampling_config": RESAMPLING_CONFIG,
        "environment": environment(), "resumed": bool(resume_dir),
    }
    log(f"{'RESUME' if resume_dir else 'START'} {problem_module.CODE}: regimes {regimes}, {len(seeds)} instances, "
        f"n_jobs={n_jobs}")

    if show_progress:
        print_config_panel(problem_module, RESAMPLING_CONFIG)
    elif verbose:
        print(f"=== {problem_module.CODE}: regime comparison ({', '.join(regimes)}) -> {problem_dir} ===", flush=True)

    try:
        with (make_progress() if show_progress else _null_progress_ctx()) as rich_progress:
            reporter = ProgressReporter(rich_progress) if show_progress else None
            fronts: Dict[str, Dict[int, Any]] = {}
            best_params_by_regime: Dict[str, Dict[int, Dict[str, Any]]] = {}
            execution_time: Dict[str, Dict[str, Any]] = {}

            for regime in regimes:
                instances_by_seed = {s: problem_module.build_instance(s, regime) for s in seeds}
                tuning_s, tuning_cpu = 0.0, {s: 0.0 for s in seeds}
                if not RESAMPLING_CONFIG["tuning"]["enabled"]:
                    default_params = problem_module.default_params_for_best_algorithm()
                    best_params = {s: dict(default_params) for s in seeds}
                else:
                    grid_override = (param_grid_override or {}).get(regime)
                    original_grid_fn = problem_module.param_grid_for_best_algorithm
                    if grid_override is not None:
                        problem_module.param_grid_for_best_algorithm = lambda: grid_override
                    try:
                        log(f"[{regime}] tuning started")
                        t0 = time.time()
                        best_params, tuning_cpu = tune_one_regime(
                            problem_module, regime, instances_by_seed,
                            n_runs_per_combo=RESAMPLING_CONFIG["tuning"]["n_runs_per_combo"],
                            seed_base=RESAMPLING_CONFIG["tuning"]["seed_base"],
                            n_jobs=n_jobs, progress=reporter, verbose=verbose, cache_dir=cache_dir,
                        )
                        session_s = time.time() - t0
                        tuning_s = _add_wallclock(cache_dir, regime, "tuning_s", session_s)
                        log(f"[{regime}] tuning done in {session_s:.0f} s (all sessions: {tuning_s:.0f} s)")
                    finally:
                        problem_module.param_grid_for_best_algorithm = original_grid_fn
                best_params_by_regime[regime] = best_params

                log(f"[{regime}] evaluation started")
                t0 = time.time()
                regime_fronts, eval_cpu = evaluate_all_instances(
                    problem_module, regime, instances_by_seed, best_params,
                    n_eval_runs=_n_eval_runs(problem_module),
                    seed_base=RESAMPLING_CONFIG["evaluation"]["seed_base"],
                    warmup=RESAMPLING_CONFIG["evaluation"]["warmup"],
                    n_jobs=n_jobs, progress=reporter, verbose=verbose, cache_dir=cache_dir,
                )
                session_s = time.time() - t0
                evaluation_s = _add_wallclock(cache_dir, regime, "evaluation_s", session_s)
                log(f"[{regime}] evaluation done in {session_s:.0f} s (all sessions: {evaluation_s:.0f} s)")
                fronts[regime] = regime_fronts
                execution_time[regime] = {
                    "tuning_s": tuning_s, "evaluation_s": evaluation_s,
                    "per_instance": {s: {"tuning_s": tuning_cpu.get(s, 0.0), "evaluation_s": eval_cpu.get(s, 0.0)}
                                     for s in seeds},
                }
                regime_dir = os.path.join(problem_dir, regime)
                os.makedirs(regime_dir, exist_ok=True)
                joblib.dump(best_params, os.path.join(regime_dir, "best_params.pkl"))
                joblib.dump(regime_fronts, os.path.join(regime_dir, "fronts.pkl"))

            # realizations are always drawn from the TRUE randomness model, i.e.
            # the "stochastic" regime's instance when that regime exists (for Chapters 3, 4 and 6
            # every regime uses the same instance)
            true_regime = "stochastic" if "stochastic" in regimes else regimes[-1]
            instances_for_resampling = {s: problem_module.build_instance(s, true_regime) for s in seeds}
            log("resampling started")
            per_instance_hv, per_instance_realized = run_resampling_for_all_instances(
                problem_module, instances_for_resampling, fronts,
                n_replications=RESAMPLING_CONFIG["resampling"]["n_replications"],
                seed_base=RESAMPLING_CONFIG["resampling"]["seed_base"],
                eps=RESAMPLING_CONFIG["hypervolume"]["eps"],
                n_jobs=n_jobs, progress=reporter, verbose=verbose,
            )
            log("resampling done")
            resampling_dir = os.path.join(problem_dir, "resampling")
            os.makedirs(resampling_dir, exist_ok=True)
            joblib.dump(per_instance_hv, os.path.join(resampling_dir, "per_instance_hv.pkl"))
            joblib.dump(per_instance_realized, os.path.join(resampling_dir, "per_instance_realized.pkl"))
            hv_runs = getattr(run_resampling_for_all_instances, "last_hv_runs", {}) or {}
            joblib.dump(hv_runs, os.path.join(resampling_dir, "per_instance_hv_runs.pkl"))
            stats_cfg, report_cfg = _report_settings(problem_module)
            run_level = None
            if report_cfg["run_level"]:
                from dependency.reporting.run_level import run_level_report
                run_level = run_level_report(per_instance_hv, hv_runs, regimes,
                                             os.path.join(problem_dir, "Run-level analysis"),
                                             alpha=stats_cfg.get("alpha", 0.05), verbose=verbose)
                log("run-level analysis " + ("written" if run_level else "skipped (no per-run fronts)"))

            overall_summary = generate_report(
                code=problem_module.CODE,
                objective_names=problem_module.problem_dict["objective_names"],
                regimes=regimes, regime_labels=labels,
                per_instance_hv=per_instance_hv, per_instance_realized=per_instance_realized,
                instance_size_classes=problem_module.INSTANCE_SIZE_CLASSES,
                fronts=fronts, best_params_by_regime=best_params_by_regime, out_dir=problem_dir,
                figures_cfg=RESAMPLING_CONFIG["figures"], stats_cfg=stats_cfg, report_cfg=report_cfg,
                eps=RESAMPLING_CONFIG["hypervolume"]["eps"],
            )
            log("report written")
            manifest.update({"execution_time": execution_time, "overall_summary": overall_summary,
                             "run_level_across_instances": (run_level or {}).get("across_instances")})
            with open(os.path.join(problem_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2, default=str)
    except BaseException:
        log.exception("run_problem")
        raise

    if verbose:
        print(f"=== {problem_module.CODE}: done, report at {problem_dir} ===", flush=True)
    return problem_dir


def estimate(problem_module, gens: int = 3) -> float:
    """Expected wall clock of a full run on this machine (hours): a short run of
    BEST_ALGORITHM under every regime on the first instance of each size class,
    extrapolated to the configured generations, tuning combinations, runs and
    instances, divided by the physical cores. The resampling is not included."""
    from joblib import cpu_count
    regimes, _ = _regimes(problem_module)
    seeds = problem_module.generate_instance_seeds()
    grid = problem_module.param_grid_for_best_algorithm()
    combos = len(_expand_grid(grid))
    default = problem_module.default_params_for_best_algorithm()
    G = int(default.get("generations", 150))
    cfg = RESAMPLING_CONFIG
    runs_per_instance = (combos * cfg["tuning"]["n_runs_per_combo"] if cfg["tuning"]["enabled"] else 0) \
        + _n_eval_runs(problem_module)
    workers = cpu_count(only_physical_cores=True) if cfg["parallel"]["n_jobs"] in (-1, None) \
        else max(1, int(cfg["parallel"]["n_jobs"]))
    total = 0.0
    print(f"{problem_module.CODE}: {len(seeds)} instances, regimes {regimes}, {combos} combinations x "
          f"{cfg['tuning']['n_runs_per_combo']} runs + {_n_eval_runs(problem_module)} evaluation runs per "
          f"instance, {workers} worker(s)")
    for cls, ids in problem_module.INSTANCE_SIZE_CLASSES.items():
        in_run = [s for s in seeds if s in ids]
        if not in_run:
            continue
        line = f"  {cls:10s}"
        for regime in regimes:
            inst = problem_module.build_instance(in_run[0], regime)
            params = dict(default, generations=gens, seed=0)
            problem_module.run_algorithm(inst, params, regime, progress_record=False)          # compile
            t0 = time.perf_counter()
            problem_module.run_algorithm(inst, params, regime, progress_record=False)
            run_s = (time.perf_counter() - t0) / (gens + 1) * (G + 1)
            total += run_s * runs_per_instance * len(in_run)
            line += f"  {regime} {run_s:7.1f} s/run"
        print(line)
    hours = total / 3600 / workers
    print(f"  => about {total / 3600:.1f} core-h, {hours:.1f} h on {workers} worker(s)")
    return hours


def _problems():
    import problems.problem_chapter2 as problem_chapter2
    import problems.problem_chapter3 as problem_chapter3
    import problems.problem_chapter4 as problem_chapter4
    import problems.problem_chapter6 as problem_chapter6
    return [problem_chapter2, problem_chapter3, problem_chapter4, problem_chapter6]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Planning-regime comparison (Chapters 2, 3, 4 and 6).")
    ap.add_argument("codes", nargs="*", help="problem codes to run (default: all)")
    ap.add_argument("--estimate", action="store_true", help="print the expected duration only")
    ap.add_argument("--resume", metavar="RUN_DIR", help="resume an interrupted run from its folder")
    a = ap.parse_args(argv)
    os.makedirs(REPORTING_ROOT, exist_ok=True)
    problems = _problems()
    if a.resume:
        with open(os.path.join(a.resume, "run.log"), encoding="utf-8") as f:
            first = f.readline()
        code = next((p.CODE for p in problems if f" {p.CODE}:" in first), None)
        if code is None:
            raise SystemExit(f"cannot tell the problem of {a.resume} from its run.log")
        problem_module = next(p for p in problems if p.CODE == code)
        run_problem(problem_module, REPORTING_ROOT, resume_dir=a.resume)
        return
    for problem_module in problems:
        if a.codes and problem_module.CODE not in a.codes:
            continue
        if a.estimate:
            estimate(problem_module)
        else:
            run_problem(problem_module, REPORTING_ROOT)


if __name__ == "__main__":
    main()
