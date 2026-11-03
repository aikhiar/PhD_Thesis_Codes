"""
problem_chapter6.py -- Preemptive EV charging scheduling under joint arrival and
energy uncertainty (Chapter 6): the methods_comparison binding of the metaheuristics
comparison -- operators, surrogate-assisted solving methods, `problem_dict`,
and the problem-specific methods_comparison settings.
=============================================================================

Model        problems/chapter6/model.py (instances, operators, recourse policy, Monte
             Carlo objectives, surrogate feature map, verifiers), with its
             parameters in problems/chapter6/parameters.py. The chapter6/ folder is
             identical in methods_comparison, surrogate_trainer and
             det_vs_stoch.
Objectives   (mean tardiness in minutes, peak load kW), expectations over the
             joint error law kappa, both minimised.

Surrogate-assisted search (Algorithm 13 of the paper)
    1. the deployed surrogates (problems/chapter6/surrogate_models/, written by
       surrogate_trainer) are the search objective: methods_comparison's bundled
       solvers run on operators["objective"] = model.surrogate_objective;
    2. SurrogateAssistedMethod re-evaluates the returned front by Monte Carlo
       with REFINEMENT_REPLICATIONS (N_r^*) replications and keeps its
       non-dominated, de-duplicated subset.
Every reported objective value (hypervolume, indicators, fronts) is therefore a
Monte Carlo estimate; the convergence progress is recorded in the search's own
(surrogate) objective space.

Every methods_comparison setting (methods, runs, tuning protocol, parallelism, statistics,
figures) is in comparison_config.py; STUDY_CONFIG below holds the settings in
which this study departs from it (the protocol of the paper: population 80,
150 generations, MOPSO's n_grid tuned).

Usage (from methods_comparison):
    run_all.bat --only chapter6                     # the study
    python -m problems.problem_chapter6 estimate            # duration of the study on this machine
"""

import os
import time
from typing import Any, Dict

import numpy as np

from problems.chapter6 import model as M
from problems.chapter6 import parameters as P
from methods_comparison.solvers.bridge import OperatorMethod, attach_operator_methods
from methods_comparison.solvers.multi_objective_solving_module import (
    deduplicate_pareto_front, find_pareto_front_indexes,
)

PROBLEM_NAME = "chapter6"
CODE = "chapter6"


# ═══════════════════════════════════════════════════════════════════════════════
# OPERATORS FOR METHODS_COMPARISON'S BUNDLED SOLVERS
# ═══════════════════════════════════════════════════════════════════════════════

def finalize_solution(instance, solution: np.ndarray) -> np.ndarray:
    """A schedule of a returned front, as stored in the results: checked by the
    independent verifier (a schedule it rejects makes the run a crashed run,
    never a wrong value)."""
    M.check_schedule_feasibility(solution, instance, label="returned schedule")
    return np.asarray(solution, dtype=np.float64)


OPERATORS = {
    "objective": M.surrogate_objective,          # the search objective (stage 1)
    "generate_solution": M.Generate_Solution,
    "perturb": M.Perturb,
    "crossover": M.Crossover,
    "build_problem_params": M.build_problem_params,
    "solution_shape": M.solution_shape,
    "n_objectives": len(M.OBJECTIVE_NAMES),
    "module": __name__,                          # workers re-import this module
}


class SurrogateAssistedMethod(OperatorMethod):
    """A bundled solver searching on the surrogates, whose returned front is
    re-evaluated by Monte Carlo with N_r^* replications and filtered to its
    non-dominated, de-duplicated subset (Algorithm 13, lines 3-8)."""

    def __call__(self, instance, params: Dict, progress_record: bool) -> Dict:
        out = super().__call__(instance, params, progress_record)
        solutions = out["pareto_front"]
        if not solutions:
            return out
        pp = M.build_problem_params({**params, "num_of_replications": P.REFINEMENT_REPLICATIONS}, instance)
        objectives = np.array([M.objective_function(s, pp) for s in solutions], dtype=np.float64)
        idx = deduplicate_pareto_front(objectives, find_pareto_front_indexes(objectives))
        return {"pareto_front": [finalize_solution(instance, solutions[i]) for i in idx],
                "pareto_objectives": objectives[idx].tolist(),
                "progress": out["progress"]}


def attach_surrogate_methods(pdict: Dict, cfg: Dict) -> Dict:
    """problem_dict with the solving methods of cfg["solvers"] built from the
    operators (as methods_comparison does) and turned into surrogate-assisted methods."""
    out = attach_operator_methods(pdict, cfg)
    out["solving_methods"] = [SurrogateAssistedMethod(m.algorithm, m.module, m.solvers_cfg, m.n_obj)
                              for m in out["solving_methods"]]
    return out


def evaluate_objective(instance, solution) -> np.ndarray:
    """problem_dict["objective_function"]: both objectives by Monte Carlo with N_r^*."""
    return M.objective_mc(solution, P.REFINEMENT_REPLICATIONS, instance)


def self_check() -> None:
    """Run by methods_comparison at start-up: the deployed surrogates are present and
    consistent with the feature map, and the tuned problem-level parameter
    perc_perturb really changes the Generate -> Perturb -> Crossover -> objective
    pipeline (deterministic objective, so that only the operators vary)."""
    from methods_comparison.utils.param_wiring_check import verify_param_wiring
    M.load_surrogates()
    inst = M.build_instance(P.SELF_CHECK_N, P.SELF_CHECK_SEED, name="self-check")
    verify_param_wiring(
        build_problem_params=lambda combo, i: M.build_problem_params({**combo, "num_of_replications": 0}, i),
        instance=inst,
        generate_solution_fn=M.Generate_Solution,
        perturb_fn=M.Perturb,
        objective_fn=M.objective_function,
        crossover_fn=M.Crossover,
        param_grids={"perc_perturb": list(P.PROBLEM_PARAM_GRID["perc_perturb"])},
        baseline={"perc_perturb": P.UNTUNED_PERC_PERTURB},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# THE PROBLEM DICT AND THE PROBLEM-SPECIFIC METHODS_COMPARISON SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# settings in which this study departs from comparison_config.py: the protocol of
# the paper (population 80, 150 generations; MOPSO also tunes n_grid)
STUDY_CONFIG: Dict[str, Any] = {
    # the tests of the chapter: Friedman and Kendall's W, pairwise Wilcoxon (Holm),
    # Cliff's delta, and the per-instance Mann-Whitney tests against the control (MOCS)
    "statistics": {"mann_whitney": True},
    "solvers": {
        "population_size": 80,
        "generations": 150,
        "algorithms": {"MOPSO": {"grid": {"n_grid": [8, 10, 12]}}},
    },
}


def build_problem_dict() -> Dict[str, Any]:
    """problem_dict with the operators only (methods_comparison's generic route)."""
    return {
        "generate_instances": M.evaluation_instances,
        "objective_function": evaluate_objective,
        "name": "metaheuristics",
        "code": CODE,
        "objective_names": list(M.OBJECTIVE_NAMES),
        "operators": OPERATORS,
        "problem_param_grids": dict(P.PROBLEM_PARAM_GRID),
        "instance_size_classes": M.size_classes(),
        "instance_features": lambda inst: inst.features(),
        "self_check": self_check,
    }


def effective_config(extra: Dict = None) -> Dict[str, Any]:
    """The configuration methods_comparison uses for this problem, as run_all.bat applies it:
    comparison_config.py, then STUDY_CONFIG, then RUN["overrides"]["chapter6"]."""
    from methods_comparison.config import DEFAULT_CONFIG, RUN_SETTINGS
    from methods_comparison.run_all import merge
    cfg = merge(merge(DEFAULT_CONFIG, STUDY_CONFIG), RUN_SETTINGS.get("overrides", {}).get(CODE, {}))
    return merge(cfg, extra or {})


# the solving methods are built here (and not by methods_comparison) because they are the
# surrogate-assisted variants of the bundled solvers
problem_dict = attach_surrogate_methods(build_problem_dict(), effective_config())


# ═══════════════════════════════════════════════════════════════════════════════
# DURATION ESTIMATE ON THIS MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

def estimate() -> float:
    """Expected duration of the study: methods_comparison's generic estimate (short timed runs
    of every method on the surrogates, one instance per class), plus the Monte
    Carlo re-evaluation of the front returned by EVERY run (tuning and evaluation)
    with N_r^* replications, which the generic estimate does not time. The front
    size is bounded by the population size, so this second term is an upper
    bound. Returns hours on the configured workers."""
    from methods_comparison.solvers.bridge import estimate as _estimate
    from methods_comparison.utils.parallel import resolve_n_jobs
    cfg = effective_config()
    hours = _estimate(build_problem_dict(), cfg)
    workers = resolve_n_jobs(cfg["parallel"]["n_jobs"])
    workers = (os.cpu_count() or 1) if workers == -1 else max(1, int(workers or 1))
    pdict = attach_operator_methods(build_problem_dict(), cfg)
    ppg = pdict.get("problem_param_grids") or {}
    runs_per_instance = 0
    for m in pdict["solving_methods"]:
        grid = {**pdict["param_grids"].get(m.__name__, {}), **ppg}
        combos = int(np.prod([len(v) for v in grid.values()])) if grid else 1
        runs_per_instance += cfg["evaluation"]["n_runs_per_instance"]
        if cfg["tuning"]["enabled"]:
            runs_per_instance += cfg["tuning"]["n_runs_per_combo"] * combos
    front = int(cfg["solvers"]["population_size"])
    insts = M.evaluation_instances()
    core_s, line = 0.0, "  re-evaluation"
    for c, ids in M.size_classes().items():
        inst = insts[ids[0] - 1]
        pp = M.build_problem_params({"num_of_replications": 0}, inst)
        M.seed_all(0)
        s = M.Generate_Solution(pp)
        M.objective_mc(s, 2, inst)                                        # compile / warm up
        t0 = time.perf_counter()
        M.objective_mc(s, P.REFINEMENT_REPLICATIONS, inst)
        per_run = (time.perf_counter() - t0) * front
        core_s += per_run * runs_per_instance * len(ids)
        line += f"  {c} {per_run:5.1f} s/run"
    extra = core_s / 3600 / workers
    print(line)
    print(f"  Monte Carlo re-evaluation of the returned fronts (N_r^* = {P.REFINEMENT_REPLICATIONS}, "
          f"at most {front} schedules, {runs_per_instance} runs per instance): {core_s / 3600:.1f} core-h "
          f"=> about {hours + extra:.1f} h in all on {workers} worker(s)")
    return hours + extra


# ============================================================================
# COMMAND LINE
# ============================================================================

if __name__ == "__main__":
    estimate()
