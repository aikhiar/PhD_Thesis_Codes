"""
surrogate_trainer/main.py -- runs the full surrogate-training pipeline
for every problem of PROBLEMS (Chapters 5 and 6):

  1. Generate training data: each configured metaheuristic, with fixed
     parameters, on the problem's own training-instance pool; the logged
     non-dominated schedules of every LOG_GENERATION_STRIDE-th generation
     (a fraction of each front) are labelled by the real Monte Carlo objective.
  2. Build (features, objective value, training instance, size class) per
     learned objective, and the metadata of every record (algorithm,
     generation, every objective's value).
  3. Train + validate every candidate model per objective: train/validation
     split BY INSTANCE (stratified by size class), hyperparameter search with a
     cross-validation over folds of whole instances.
  4. Execution-time benchmark: per-schedule cost of feature extraction, of every
     model's prediction, and of the real objective (its label replication count),
     warm-up calls excluded.
  5. Select the deployed model of every objective by the two-stage rule fixed in
     config.SELECTION (accuracy practically equivalent to the most accurate
     candidate, then lowest end-to-end scoring cost).
  6. Reports, those of Chapters 5 and 6: one per objective (metrics, search
     spaces and retained hyperparameters, statistical tests with validation
     instances as blocks, selection table, within-instance ranking, predictions
     and model comparison figures), the joint fidelity for the optimizer
     (Pareto-dominance agreement, non-dominated set precision/recall), the
     training data, the execution time and the cost accounting (break-even);
     every table also as LaTeX.
  7. Export the deployed models (surrogate_export/best_models/: the .joblib
     files and model_card.json, to copy into
     methods_comparison/problems/<study>/surrogate_models/), with the library versions
     they were trained with.

Resume: every data-generation task and every trained objective is cached under
<run>/cache/; `--resume <run folder>` restarts an interrupted run from there,
with identical results. Every run writes run.log (stages, durations, errors).

This framework is standalone: its algorithms are a physical copy of
methods_comparison's solvers (dependency/multi_algorithms), and the problems'
models (problems/chapter5/, problems/chapter6/) are identical to methods_comparison's.

    python main.py                          # chapter5 then chapter6
    python main.py chapter6                       # one paper
    python main.py --estimate               # expected duration of the data generation
    python main.py --resume runs/surrogate_chapter6_ab12cd34
"""

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import random
import string
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dependency.requirements import ensure_requirements  # noqa: E402  (standard library only)

ensure_requirements()                                     # checks, and installs if missing, the required packages

import joblib
import numpy as np

from dependency.config.config import ML_CONFIG
from dependency.reporting.progress import ProgressReporter, make_progress, print_config_panel
from dependency.reporting.quality import (
    report_cost_accounting,
    report_objective_quality,
    report_optimizer_quality,
    report_training_data,
    validation_view,
)
from dependency.reporting.reporting import (
    _validation_significance_tests,
    benchmark_inference_time,
    report_execution_time,
    report_objective,
    select_deployed_model,
)
from dependency.reporting.tables import csv_to_tex
from dependency.resume import load_task, save_task, task_path
from dependency.runlog import RunLog, environment
from dependency.trainer.trainer import train_and_validate
from dependency.training_data_generator.generator import (
    _run_one_task,
    build_feature_target_pairs,
    build_record_metadata,
    generate_training_data,
)

import problems.problem_chapter5 as problem_chapter5
import problems.problem_chapter6 as problem_chapter6

PROBLEMS = [problem_chapter5, problem_chapter6]

REPORTING_ROOT = os.path.join(str(_ROOT), "runs")
_FIGURES = {"formats": ["png", "svg"], "dpi": 400}


def _random_key(n: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(n))


def _slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


@contextlib.contextmanager
def _null_progress_ctx():
    yield None


def _evaluations_per_run(problem_module) -> Optional[int]:
    """Objective evaluations of one optimizer run of the study: population x
    (generations + 1), from the problem's algorithm settings."""
    spec = next(iter(problem_module.ALGORITHMS.values()), {}).get("solve_kwargs", {})
    if "population_size" in spec and "generations" in spec:
        return int(spec["population_size"]) * (int(spec["generations"]) + 1)
    return None


def _training_settings(objective_name: str, X: np.ndarray, y: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """What a cached trained objective must have been trained with/on."""
    keys = ("candidate_models", "cv_folds", "search_n_iter", "scoring", "random_state", "slow_model_train_cap",
            "train_val_split")
    digest = hashlib.md5(np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(y).tobytes()).hexdigest()
    return {"objective": objective_name, "data": digest, **{k: cfg.get(k) for k in keys}}


def run_problem(
    problem_module,
    reporting_root: str = REPORTING_ROOT,
    instances_override=None,
    algorithms_override: Optional[Dict[str, Dict[str, Any]]] = None,
    verbose: bool = True,
    show_progress: bool = True,
    ml_config: Optional[Dict[str, Any]] = None,
    resume_dir: Optional[str] = None,
) -> str:
    """Runs the whole pipeline for one problem module; returns the run folder.

    `instances_override` / `algorithms_override` / `ml_config` let a caller run a
    cheap pilot; `resume_dir` resumes an
    interrupted run from its cache."""
    cfg = ml_config or ML_CONFIG
    figures_cfg = cfg.get("figures", _FIGURES)
    qcfg = cfg.get("quality_report", {})
    if resume_dir:
        problem_dir = os.path.abspath(resume_dir)
    else:
        problem_dir = os.path.join(reporting_root, f"surrogate_{problem_module.CODE}_{_random_key()}")
    os.makedirs(problem_dir, exist_ok=True)
    cache_dir = os.path.join(problem_dir, "cache")
    log = RunLog(problem_dir)
    log(f"{'RESUME' if resume_dir else 'START'} {problem_module.CODE}: "
        f"{len(problem_module.ALGORITHMS if algorithms_override is None else algorithms_override)} algorithms")

    original_instances_fn = problem_module.generate_training_instances
    original_algorithms = problem_module.ALGORITHMS
    if instances_override is not None:
        problem_module.generate_training_instances = lambda: instances_override
    if algorithms_override is not None:
        problem_module.ALGORITHMS = algorithms_override

    try:
        objective_names = list(problem_module.objective_names)
        surrogate_names = getattr(problem_module, "SURROGATE_OBJECTIVE_NAMES", objective_names)
        exact_names = [n for n in objective_names if n not in surrogate_names]
        evaluations_per_run = _evaluations_per_run(problem_module)

        if show_progress:
            print_config_panel(problem_module, cfg)
        elif verbose:
            print(f"=== {problem_module.CODE}: surrogate training -> {problem_dir} ===")
            print(f"  learned (surrogate trained): {surrogate_names}")
            print(f"  computed exactly (no surrogate): {exact_names or '(none)'}")

        with (make_progress() if show_progress else _null_progress_ctx()) as rich_progress:
            reporter = ProgressReporter(rich_progress) if show_progress else None

            # 1-2. training data
            log("data generation started")
            t0 = time.time()
            dataset = generate_training_data(problem_module, cfg, verbose=verbose, progress=reporter,
                                             cache_dir=cache_dir)
            log(f"data generation done in {time.time() - t0:.0f} s -- {dataset.summary()}")
            if verbose and not show_progress:
                print(f"  done in {time.time() - t0:.1f}s -- {dataset.summary()}")
            feature_target_pairs = build_feature_target_pairs(dataset, problem_module)
            metadata = build_record_metadata(dataset, problem_module)

            raw_data_dir = os.path.join(problem_dir, "raw_data")
            os.makedirs(raw_data_dir, exist_ok=True)
            joblib.dump(dataset.instances_by_id, os.path.join(raw_data_dir, "instances.pkl"))
            joblib.dump(dataset.records, os.path.join(raw_data_dir, "training_records.pkl"))
            joblib.dump(feature_target_pairs, os.path.join(raw_data_dir, "feature_target_pairs.pkl"))
            joblib.dump(metadata, os.path.join(raw_data_dir, "record_metadata.pkl"))
            joblib.dump(dataset.task_seconds, os.path.join(raw_data_dir, "datagen_task_seconds.pkl"))

            # 3. train and validate every candidate (split by instance), cached per objective
            training_results: Dict[str, Any] = {}
            for objective_name in surrogate_names:
                X, y, groups, classes = feature_target_pairs[objective_name]
                path = task_path(cache_dir, "training", f"{_slugify(objective_name)}.pkl")
                settings = _training_settings(objective_name, X, y, cfg)
                cached = load_task(path, settings)
                if cached is not None:
                    training_results[objective_name] = cached["result"]
                    log(f"training '{objective_name}' loaded from the cache")
                    continue
                if verbose and not show_progress:
                    print(f"  training objective '{objective_name}' (X={X.shape}, "
                          f"{len(set(groups.tolist()))} training instances) ...")
                log(f"training '{objective_name}' started ({X.shape[0]} records)")
                t0 = time.time()
                training_results[objective_name] = train_and_validate(
                    X, y, objective_name, cfg, progress=reporter, groups=groups, classes=classes)
                save_task(path, settings, {"result": training_results[objective_name]})
                log(f"training '{objective_name}' done in {time.time() - t0:.0f} s")

            # the benchmark below is about the problem's real size classes, not a pilot override
            problem_module.generate_training_instances = original_instances_fn
            problem_module.ALGORITHMS = original_algorithms

            # 4. execution-time benchmark (warm-up excluded)
            if verbose and not show_progress:
                print("  benchmarking scoring time (feature extraction, prediction, real objective) ...")
            log("benchmark started")
            models_by_objective = {obj: {m: r.estimator for m, r in tr.results.items()}
                                   for obj, tr in training_results.items()}
            inference_rows, real_rows = benchmark_inference_time(
                problem_module, models_by_objective, cfg.get("benchmark"), progress=reporter)
            log("benchmark done")

            # 5. two-stage selection of the deployed model
            significance_by_objective: Dict[str, Any] = {}
            for objective_name, result in training_results.items():
                significance = _validation_significance_tests(result)
                significance_by_objective[objective_name] = significance
                selection = select_deployed_model(result, significance, inference_rows, cfg.get("selection", {}),
                                                  real_rows=real_rows)
                result.deployed_model_name = selection["deployed"]
                result.selection = selection
                log(f"{objective_name}: deployed {selection['deployed']} (most accurate {selection['most_accurate']})")
                if verbose and not show_progress:
                    print(f"    {objective_name}: most accurate = {selection['most_accurate']}, "
                          f"stage-1 candidates = {selection['stage1_candidates']}, deployed = {selection['deployed']}")

            # 6. reports
            log("reports started")
            view = validation_view(training_results, metadata, objective_names)
            for objective_name, result in training_results.items():
                obj_dir = os.path.join(problem_dir, _slugify(objective_name))
                report_objective(result, obj_dir, figures_cfg=figures_cfg)
                report_objective_quality(result, view, obj_dir, qcfg)
                for rel in ("Metrics/leaderboard.csv", "Metrics/metrics.csv", "Statistical Tests/significance_tests.csv"):
                    csv_to_tex(os.path.join(obj_dir, rel))
                models_dir = os.path.join(obj_dir, "models")
                os.makedirs(models_dir, exist_ok=True)
                for model_name, model_result in result.results.items():
                    joblib.dump(model_result.estimator, os.path.join(models_dir, f"{model_name}.joblib"))
                joblib.dump(result, os.path.join(obj_dir, "training_result.pkl"))

            report_optimizer_quality(
                problem_module.CODE, view, objective_names, list(surrogate_names), metadata,
                os.path.join(problem_dir, "optimizer_quality"), figures_cfg, qcfg)
            report_training_data(problem_module.CODE, metadata, os.path.join(problem_dir, "training_data"))
            exec_dir = os.path.join(problem_dir, "execution_time_analysis")
            report_execution_time(problem_module.CODE, training_results, inference_rows, exec_dir,
                                  figures_cfg=figures_cfg, real_rows=real_rows)
            for name in ("training_time.csv", "inference_time.csv", "real_vs_surrogate_time.csv"):
                csv_to_tex(os.path.join(exec_dir, "Data", name))
            cost_rows = report_cost_accounting(problem_module.CODE, dataset.summary(), training_results,
                                               inference_rows, real_rows, evaluations_per_run, exec_dir)
            log("reports done")

            # 7. export of the deployed models
            _export_objective_function(problem_module, training_results, dataset.summary(), cfg,
                                       os.path.join(problem_dir, "surrogate_export"))

            summary_out = dataset.summary()
            summary_out["objectives_learned"] = list(surrogate_names)
            summary_out["objectives_computed_exactly"] = exact_names
            summary_out["deployed_models"] = {o: r.deployed_model_name for o, r in training_results.items()}
            summary_out["cost_accounting"] = cost_rows
            summary_out["environment"] = environment()
            with open(os.path.join(problem_dir, "dataset_summary.json"), "w") as f:
                json.dump(summary_out, f, indent=2, default=str)
            log("export done")

        if verbose:
            print(f"=== {problem_module.CODE}: done, report at {problem_dir} ===")
    except BaseException:
        log.exception("run_problem")
        raise
    finally:
        problem_module.generate_training_instances = original_instances_fn
        problem_module.ALGORITHMS = original_algorithms

    return problem_dir


def _export_objective_function(problem_module, training_results: Dict[str, Any], dataset_summary: Dict[str, Any],
                               cfg: Dict[str, Any], out_dir: str) -> None:
    """Writes under out_dir:
      best_models/<objective slug>.joblib   the DEPLOYED model of every learned objective
      best_models/model_card.json           what was deployed and why, the data it came from and
                                            the library versions it was trained with (the problem's
                                            load_surrogates checks them)
      model_card.json                       the same card
      surrogate_objective.py                a standalone objective_function(instance, solution) using them
    Copy everything in best_models/ into methods_comparison/problems/<jN>/surrogate_models/."""
    objective_names = list(problem_module.objective_names)
    surrogate_names = list(getattr(problem_module, "SURROGATE_OBJECTIVE_NAMES", objective_names))
    exact_names = [n for n in objective_names if n not in surrogate_names]

    models_dir = os.path.join(out_dir, "best_models")
    os.makedirs(models_dir, exist_ok=True)
    model_files = {}
    card = {"problem": problem_module.CODE,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "label_replications": getattr(problem_module, "LABEL_REPLICATIONS", None),
            "training_records": dataset_summary.get("n_records"),
            "training_instances": dataset_summary.get("n_instances"),
            "split": cfg["train_val_split"], "selection_rule": cfg.get("selection"),
            "feature_names": list(getattr(problem_module, "FEATURE_NAMES", [])),
            "environment": environment(),
            "objectives": {}}
    for name in surrogate_names:
        result = training_results[name]
        deployed = result.deployed_model_name or result.best_model_name
        fname = f"{_slugify(name)}.joblib"
        joblib.dump(result.results[deployed].estimator, os.path.join(models_dir, fname))
        model_files[name] = fname
        m = result.results[deployed]
        card["objectives"][name] = {
            "file": fname, "deployed_model": deployed, "most_accurate_model": result.best_model_name,
            "stage1_candidates": result.selection.get("stage1_candidates"),
            "val_rmse": m.val_rmse, "val_mae": m.val_mae, "val_r2": m.val_r2, "hyperparameters": m.fit_params,
        }
    card["objectives_computed_exactly"] = exact_names
    for path in (os.path.join(out_dir, "model_card.json"), os.path.join(models_dir, "model_card.json")):
        with open(path, "w") as f:
            json.dump(card, f, indent=2, default=str)

    code = f'''"""
Auto-generated by surrogate_trainer/main.py for problem
'{problem_module.CODE}': objective_function(instance, solution) predicting the
learned objectives with the deployed models (best_models/) and computing the
others exactly. ALWAYS re-evaluate the final returned front with the real
objective before reporting results.

Objectives predicted by a learned model: {surrogate_names!r}
Objectives computed exactly: {exact_names!r}
"""
import os
import joblib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_FILES = {model_files!r}
_MODELS = {{name: joblib.load(os.path.join(_HERE, "best_models", fname)) for name, fname in _MODEL_FILES.items()}}
_OBJECTIVE_NAMES = {objective_names!r}
_SURROGATE_NAMES = {surrogate_names!r}
_EXACT_NAMES = {exact_names!r}


def objective_function(instance, solution):
    from problems.{problem_module.__name__.rsplit(".", 1)[-1]} import feature_fn, exact_objective_values
    values = {{}}
    if _SURROGATE_NAMES:
        phi = np.asarray(feature_fn(solution, instance), dtype=np.float64).reshape(1, -1)
        for name in _SURROGATE_NAMES:
            values[name] = float(_MODELS[name].predict(phi)[0])
    if _EXACT_NAMES:
        values.update(exact_objective_values(instance, solution))
    return np.array([values[name] for name in _OBJECTIVE_NAMES], dtype=np.float64)
'''
    with open(os.path.join(out_dir, "surrogate_objective.py"), "w") as f:
        f.write(code)


def estimate(problem_module, cfg: Optional[Dict[str, Any]] = None, gens: int = 3) -> float:
    """Expected wall clock of the data generation on this machine (hours): a
    short run of every algorithm on the first training instance of each size
    class, extrapolated to the configured generations and to every training
    instance, divided by the physical cores. Model training is not included
    (minutes to a few hours, depending on the candidates)."""
    from joblib import cpu_count
    cfg = cfg or ML_CONFIG
    instances = problem_module.generate_training_instances()
    class_of = getattr(problem_module, "instance_class", lambda iid: str(iid).rsplit("_", 1)[0])
    by_class: Dict[str, list] = {}
    for iid, inst in instances:
        by_class.setdefault(class_of(iid), []).append((iid, inst))
    n_jobs = cfg.get("datagen_n_jobs", 1)
    workers = cpu_count(only_physical_cores=True) if n_jobs in (-1, None) else max(1, int(n_jobs))
    total = 0.0
    print(f"{problem_module.CODE}: {len(instances)} training instances x {len(problem_module.ALGORITHMS)} algorithms, "
          f"{workers} worker(s)")
    for cls, members in by_class.items():
        iid, inst = members[0]
        line = f"  {cls:10s}"
        for algo_name, spec in problem_module.ALGORITHMS.items():
            G = int(spec["solve_kwargs"].get("generations", 150))
            short = {**spec, "solve_kwargs": {**spec["solve_kwargs"], "generations": gens}}
            _run_one_task(problem_module, iid, inst, algo_name, short, cfg, verbose=False, progress=None)  # compile
            _, _, _, seconds = _run_one_task(problem_module, iid, inst, algo_name, short, cfg, verbose=False,
                                             progress=None)
            run_s = seconds / (gens + 1) * (G + 1)
            total += run_s * len(members)
            line += f"  {algo_name} {run_s:6.0f} s/run"
        print(line)
    hours = total / 3600 / workers
    print(f"  => data generation about {total / 3600:.1f} core-h, {hours:.1f} h on {workers} worker(s) "
          f"(+ model training)")
    return hours


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Surrogate training (Chapters 5 and 6).")
    ap.add_argument("codes", nargs="*", help="problem codes to run (default: all)")
    ap.add_argument("--estimate", action="store_true", help="print the expected duration only")
    ap.add_argument("--resume", metavar="RUN_DIR", help="resume an interrupted run from its folder")
    a = ap.parse_args(argv)
    os.makedirs(REPORTING_ROOT, exist_ok=True)
    if a.resume:
        with open(os.path.join(a.resume, "run.log"), encoding="utf-8") as f:
            first = f.readline()
        problem_module = next((p for p in PROBLEMS if f" {p.CODE}:" in first), None)
        if problem_module is None:
            raise SystemExit(f"cannot tell the problem of {a.resume} from its run.log")
        run_problem(problem_module, REPORTING_ROOT, resume_dir=a.resume)
        return
    for problem_module in PROBLEMS:
        if a.codes and problem_module.CODE not in a.codes:
            continue
        if a.estimate:
            estimate(problem_module)
        else:
            run_problem(problem_module, REPORTING_ROOT)


if __name__ == "__main__":
    main()
