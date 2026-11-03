"""
Live "top"-style progress reporting for the surrogate_trainer
pipeline, built on `rich`.

Two pieces:
  - `print_config_panel(problem_module, ml_config)`: a one-shot, non-live
    summary of everything about to run -- candidate models, tuning budget,
    train/validation split, the fixed-parameter algorithm portfolio (with
    population_size/generations per algorithm), instance counts, and the
    learned-vs-exact objective split. Printed once, before the live
    dashboard starts, so nothing about *what* is about to run is left
    implicit.
  - `ProgressReporter`: a thin wrapper around one shared `rich.progress.Progress`
    instance, used by every stage of the pipeline (data generation, model
    tuning, execution-time benchmarking) so elapsed time and ETA are always
    computed from real, per-iteration completion counts -- never faked.

All progress-reporting calls degrade to no-ops when `progress=None` is
passed through (the default), so every function that accepts a
`ProgressReporter` remains fully usable without `rich` in the loop, e.g.
for quick scripting or debugging.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()

# Human-readable display names for the candidate-model registry keys used
# throughout the pipeline (`ml_config["candidate_models"]`, trainer.py's
# `_model_factory`) -- kept here, rather than in trainer.py, so both
# trainer.py (console/progress descriptions) and reporting.py (figure
# titles/labels/legends) can use it without reporting.py having to import
# trainer.py's heavy ML-library dependencies (catboost/xgboost).
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "hist_gradient_boosting": "HistGradientBoosting",
    "gaussian_process": "Gaussian Process (Kriging)",
    "svr": "Support Vector Regression",
    "rbf": "RBF (Kernel Ridge)",
}


def display_model_name(name: str) -> str:
    """Professional display name for a candidate-model key, e.g.
    'gaussian_process' -> 'Gaussian Process (Kriging)'. Falls back to a
    title-cased, underscore-stripped version for any key not in the
    registry, so a future candidate model never leaks a raw Python
    identifier into a figure title/legend/axis label."""
    return MODEL_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def print_config_panel(problem_module, ml_config: Dict[str, Any]) -> None:
    """Prints, once, a complete picture of the run about to start."""
    cfg_table = Table(show_header=False, box=None, pad_edge=False)
    cfg_table.add_column("Setting", style="bold cyan", no_wrap=True)
    cfg_table.add_column("Value")

    split = ml_config["train_val_split"]
    cfg_table.add_row("Candidate models", ", ".join(ml_config["candidate_models"]))
    cfg_table.add_row("CV folds", str(ml_config["cv_folds"]))
    cfg_table.add_row("Randomized-search iterations / model", str(ml_config["search_n_iter"]))
    cfg_table.add_row(
        "Train / validation split",
        f"{1 - split['validation_fraction']:.0%} / {split['validation_fraction']:.0%} "
        + ("of the training INSTANCES (all their schedules), stratified by size class"
           if split.get("group_by_instance", True) else "of the rows")
        + f" (seed={split['random_state']}); CV folds of whole instances",
    )
    sel = ml_config.get("selection", {})
    cfg_table.add_row("Deployed-model selection",
                      f"|Cliff's delta| <= {sel.get('max_abs_cliff_delta', 0.33)} vs the most accurate, "
                      f"then lowest end-to-end scoring time")
    cfg_table.add_row("Slow-model training cap (GP/SVR/RBF)", str(ml_config.get("slow_model_train_cap", "--")))
    cfg_table.add_row("Elite log fraction (mono algorithms)", str(ml_config["elite_log_fraction"]))
    cfg_table.add_row("Pareto log fraction (multi algorithms)", str(ml_config.get("pareto_log_fraction", "--")))
    cfg_table.add_row("Log every Nth generation", str(ml_config.get("log_generation_stride", 1)))
    cfg_table.add_row("Data-generation n_jobs", str(ml_config.get("datagen_n_jobs", 1)))
    cfg_table.add_row("Scoring", ml_config["scoring"])
    cfg_table.add_row("Random state", str(ml_config["random_state"]))
    cfg_table.add_row("n_jobs (CV folds, sequential path only)", str(ml_config["n_jobs"]))
    cfg_table.add_row("Train n_jobs (parallel across models)", str(ml_config.get("train_n_jobs", 1)))

    algo_table = Table(title="Fixed-parameter algorithm portfolio", show_lines=False)
    algo_table.add_column("Algorithm", style="bold")
    algo_table.add_column("Population", justify="right")
    algo_table.add_column("Generations", justify="right")
    algo_table.add_column("Other solve_kwargs")
    for name, spec in problem_module.ALGORITHMS.items():
        sk = spec.get("solve_kwargs", {})
        others = ", ".join(f"{k}={v}" for k, v in sk.items() if k not in ("population_size", "generations", "seed"))
        algo_table.add_row(name, str(sk.get("population_size", "?")), str(sk.get("generations", "?")), others or "--")

    n_instances = len(problem_module.generate_training_instances())
    surrogate_names = getattr(problem_module, "SURROGATE_OBJECTIVE_NAMES", problem_module.objective_names)
    exact_names = [n for n in problem_module.objective_names if n not in surrogate_names]

    obj_table = Table(show_header=False, box=None, pad_edge=False)
    obj_table.add_column("Setting", style="bold cyan", no_wrap=True)
    obj_table.add_column("Value")
    obj_table.add_row("Training instances", str(n_instances))
    obj_table.add_row("Learned (surrogate trained)", ", ".join(surrogate_names) or "(none)")
    obj_table.add_row("Computed exactly (no surrogate)", ", ".join(exact_names) or "(none)")

    # `characters="-"` on both rules: Console.rule()'s default fill
    # character is a Unicode box-drawing "-" (U+2500), which is NOT covered
    # by Table/Panel's own automatic ASCII-fallback detection -- on a
    # Windows console using a legacy (non-UTF-8) codepage, the default
    # would raise UnicodeEncodeError. Forcing plain "-" is safe everywhere.
    console.rule(f"[bold]{problem_module.CODE}[/bold] -- surrogate training run configuration", characters="-")
    console.print(Panel(cfg_table, title="ML configuration", border_style="cyan", expand=False))
    console.print(Panel(algo_table, title="Data-generation portfolio", border_style="cyan", expand=False))
    console.print(Panel(obj_table, title="Problem", border_style="cyan", expand=False))
    console.rule(characters="-")


def make_progress() -> Progress:
    """One shared, multi-task progress display -- every stage of the
    pipeline (data generation, model tuning, execution-time benchmarking)
    adds/removes its own tasks into this same live view, each with a real
    elapsed-time and ETA column computed from actual completed/total
    counts."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("elapsed"),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


class ProgressReporter:
    """Thin convenience wrapper around a shared `rich.progress.Progress`:
    named tasks, so callers (generator.py, trainer.py, reporting.py) can
    start/update/finish a task by a short string key without passing
    `rich` task-id integers around."""

    def __init__(self, progress: Progress):
        self.progress = progress
        self._tasks: Dict[str, int] = {}

    def start(self, key: str, description: str, total: Optional[float] = None) -> None:
        if key in self._tasks:
            self.finish(key)
        self._tasks[key] = self.progress.add_task(description, total=total)

    def update(
        self,
        key: str,
        advance: float = 0,
        description: Optional[str] = None,
        completed: Optional[float] = None,
        total: Optional[float] = None,
    ) -> None:
        if key not in self._tasks:
            return
        kwargs: Dict[str, Any] = {}
        if description is not None:
            kwargs["description"] = description
        if total is not None:
            kwargs["total"] = total
        if completed is not None:
            kwargs["completed"] = completed
        elif advance:
            kwargs["advance"] = advance
        if kwargs:
            self.progress.update(self._tasks[key], **kwargs)

    def finish(self, key: str) -> None:
        task_id = self._tasks.pop(key, None)
        if task_id is not None:
            self.progress.remove_task(task_id)
