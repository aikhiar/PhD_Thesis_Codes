"""
Live "top"-style progress reporting for the det_vs_stoch pipeline,
built on `rich`. Adapted (not imported) from
surrogate_trainer/dependency/reporting/progress.py per this
framework's "full standalone copy" convention.

Two pieces:
  - `print_config_panel(problem_module, resampling_config)`: a one-shot,
    non-live summary of everything about to run -- BEST_ALGORITHM, both
    regimes' forced problem params, tuning/evaluation/resampling budgets,
    instance count. Printed once, before the live dashboard starts.
  - `ProgressReporter`: a thin wrapper around one shared
    `rich.progress.Progress` instance, used by every stage of the pipeline
    (tuning, evaluation, resampling) so elapsed time and ETA are always
    computed from real, per-iteration completion counts -- never faked.

All progress-reporting calls degrade to no-ops when `progress=None` is
passed through (the default), so every function that accepts a
`ProgressReporter` remains fully usable without `rich` in the loop.
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

# Human-readable display names for algorithm keys -- kept here for the same
# reason surrogate_trainer keeps its own copy: figures/console output
# should never leak a raw Python identifier.
ALGORITHM_DISPLAY_NAMES: Dict[str, str] = {
    "NSGA2": "NSGA-II",
    "NSGA3": "NSGA-III",
    "MOCS": "MOCS",
    "MOPSO": "MOPSO",
    "MOGWO": "MOGWO",
}


def display_algorithm_name(name: str) -> str:
    return ALGORITHM_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def print_config_panel(problem_module, resampling_config: Dict[str, Any]) -> None:
    """Prints, once, a complete picture of the run about to start."""
    cfg_table = Table(show_header=False, box=None, pad_edge=False)
    cfg_table.add_column("Setting", style="bold cyan", no_wrap=True)
    cfg_table.add_column("Value")

    tuning_enabled = resampling_config["tuning"]["enabled"]
    cfg_table.add_row("Best algorithm", display_algorithm_name(problem_module.BEST_ALGORITHM))
    cfg_table.add_row("Tuning", "enabled" if tuning_enabled else "disabled (using default params)")
    if tuning_enabled:
        cfg_table.add_row("Tuning runs per combo", str(resampling_config["tuning"]["n_runs_per_combo"]))
    cfg_table.add_row("Evaluation runs per (instance, regime)", str(getattr(problem_module, "N_EVAL_RUNS", resampling_config["evaluation"]["n_eval_runs"])))
    cfg_table.add_row("Resampling replications (R)", str(resampling_config["resampling"]["n_replications"]))
    cfg_table.add_row("Parallel n_jobs (tuning/eval/resampling)", str(resampling_config["parallel"]["n_jobs"]))
    cfg_table.add_row("Random state", str(resampling_config["random_state"]))

    regime_table = Table(title="Regimes", show_lines=False)
    regime_table.add_column("Regime", style="bold")
    regime_table.add_column("Forced problem params")
    for regime_name, regime_spec in problem_module.REGIMES.items():
        forced = regime_spec.get("forced_problem_params", {})
        regime_table.add_row(regime_name, ", ".join(f"{k}={v}" for k, v in forced.items()) or "(instance-level flag)")

    n_instances = len(problem_module.generate_instance_seeds())

    obj_table = Table(show_header=False, box=None, pad_edge=False)
    obj_table.add_column("Setting", style="bold cyan", no_wrap=True)
    obj_table.add_column("Value")
    obj_table.add_row("Instances", str(n_instances))
    obj_table.add_row("Objectives", ", ".join(problem_module.problem_dict["objective_names"]))

    # `characters="-"` on both rules: Console.rule()'s default fill
    # character is a Unicode box-drawing "-" (U+2500), which is NOT covered
    # by Table/Panel's own automatic ASCII-fallback detection -- on a
    # Windows console using a legacy (non-UTF-8) codepage, the default
    # would raise UnicodeEncodeError. Forcing plain "-" is safe everywhere.
    console.rule(f"[bold]{problem_module.CODE}[/bold] -- deterministic vs stochastic comparison", characters="-")
    console.print(Panel(cfg_table, title="Resampling configuration", border_style="cyan", expand=False))
    console.print(Panel(regime_table, title="Regime definitions", border_style="cyan", expand=False))
    console.print(Panel(obj_table, title="Problem", border_style="cyan", expand=False))
    console.rule(characters="-")


def make_progress() -> Progress:
    """One shared, multi-task progress display -- every stage of the
    pipeline (tuning, evaluation, resampling) adds/removes its own tasks
    into this same live view, each with a real elapsed-time and ETA column
    computed from actual completed/total counts."""
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
    named tasks, so callers (tuner.py, evaluator.py, protocol.py) can
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
