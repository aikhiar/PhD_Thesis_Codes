"""
problem_chapter3.py -- Chapter 3 of the thesis (chapter9.tex of the manuscript;
Khiar et al., Omega 141 (2026) 103506: multi-objective EV charging scheduling
under stochastic charging durations): the methods_comparison binding of the model.
=============================================================================

Model        problems/chapter3/model.py: the functions of the thesis copied
             verbatim (construction, Generate_Solution, Perturb, Crossover,
             lognormal actualisation, objectives, verifiers) plus the adaptation
             layer (data source, instance suite, methods_comparison interface); parameters
             in problems/chapter3/parameters.py.
Instances    20 from the ACN-Data claimed sessions, n = 50, 100, 150, 200 (5 per
             size), m = ceil(n / 3).
Objectives   (conditional expected grid capacity kW by Monte Carlo with N_r = 30,
             conditional expected total tardiness h in closed form, unmet
             energy %), all minimised.

This module binds the code of the thesis to the current methods_comparison (the solving
methods are methods_comparison's bundled NSGA-II, NSGA-III, MOCS, MOPSO and MOGWO, built
from the operators); it changes no operator.

Usage (from methods_comparison):
    run_all.bat --only chapter3     # the study
    python -m problems.problem_chapter3 estimate # duration of the study on this machine
"""

from typing import Any, Dict

import numpy as np

from problems.chapter3 import model as M
from problems.chapter3 import parameters as P

PROBLEM_NAME = "chapter3"
CODE = "chapter3"


# ═══════════════════════════════════════════════════════════════════════════════
# OPERATORS FOR METHODS_COMPARISON'S BUNDLED SOLVERS (the thesis's, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def finalize_solution(instance, solution: np.ndarray) -> np.ndarray:
    """A schedule of a returned front, as stored in the results: checked by
    the chapter's verifier (a schedule it rejects makes the run a crashed run,
    never a wrong value)."""
    M.check_schedule_feasibility(solution, instance.data, label="returned schedule")
    return np.asarray(solution, dtype=np.float64)


OPERATORS = {
    "objective": M.objective_function,
    "generate_solution": M.Generate_Solution,
    "perturb": M.Perturb,
    "crossover": M.Crossover,
    "build_problem_params": M.build_problem_params,
    "solution_shape": M.solution_shape,
    "finalize": finalize_solution,
    "n_objectives": len(M.OBJECTIVE_NAMES),
    "module": __name__,                     # workers re-import this binding (the operators live in problems.chapter3.model)
}


def evaluate_objective(instance, solution) -> np.ndarray:
    """problem_dict["objective_function"]: the three objectives of a schedule (N_r = 30)."""
    return M.evaluate_objective(instance, solution)


def self_check() -> None:
    """Run by methods_comparison at start-up (the chapter's self_check): perc_perturb, lam and
    num_of_replications really change the Generate -> Perturb -> Crossover ->
    objective pipeline."""
    from methods_comparison.utils.param_wiring_check import verify_param_wiring
    inst = M.acn_instance(50, 999, name="self-check")
    verify_param_wiring(
        build_problem_params=M.build_problem_params,
        instance=inst,
        generate_solution_fn=M.Generate_Solution,
        perturb_fn=M.Perturb,
        objective_fn=M.objective_function,
        crossover_fn=M.Crossover,
        param_grids={
            "perc_perturb": [0.05, 0.10, 0.20],
            "lam": [0.5, 1.0, 2.0],
            "num_of_replications": [1, 10, 40],
        },
        baseline={"perc_perturb": P.DEFAULT_PERC_PERTURB, "lam": P.DEFAULT_LAM,
                  "num_of_replications": P.DEFAULT_NUM_REPLICATIONS},
        n_seeds=30,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# THE PROBLEM DICT AND THE PROBLEM-SPECIFIC METHODS_COMPARISON SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# settings in which this study departs from comparison_config.py: the chapter's
# protocol (population 80, 150 generations, the chapter's grids)
STUDY_CONFIG: Dict[str, Any] = {
    # the tests of the chapter: Friedman, pairwise Wilcoxon (Bonferroni), Cliff's delta
    "statistics": {"pairwise_correction": "bonferroni"},
    "solvers": {
        "population_size": P.DEFAULT_POPULATION_SIZE,
        "generations": P.DEFAULT_GENERATIONS,
        "algorithms": {a: {"grid": dict(g)} for a, g in P.ALGORITHM_GRIDS.items()},
    },
}


def build_problem_dict() -> Dict[str, Any]:
    """problem_dict with the operators (methods_comparison builds the solving methods)."""
    return {
        "generate_instances": M.generate_instances,
        "objective_function": evaluate_objective,
        "name": "metaheuristics",
        "code": CODE,
        "objective_names": list(M.OBJECTIVE_NAMES),
        "operators": OPERATORS,
        "problem_param_grids": {k: list(v) for k, v in P.PROBLEM_PARAM_GRID.items()},
        "instance_size_classes": M.size_classes(),
        "instance_features": M.instance_features,
        "self_check": self_check,
    }


def effective_config(extra: Dict = None) -> Dict[str, Any]:
    """The configuration methods_comparison uses for this problem, as run_all.bat applies it:
    comparison_config.py, then STUDY_CONFIG, then RUN["overrides"]["chapter3"]."""
    from methods_comparison.config import DEFAULT_CONFIG, RUN_SETTINGS
    from methods_comparison.run_all import merge
    cfg = merge(merge(DEFAULT_CONFIG, STUDY_CONFIG), RUN_SETTINGS.get("overrides", {}).get(CODE, {}))
    return merge(cfg, extra or {})


problem_dict = build_problem_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# DURATION ESTIMATE ON THIS MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

def estimate() -> float:
    """Expected duration of the study (methods_comparison's generic estimate: short timed runs
    of every method, one instance per class)."""
    from methods_comparison.solvers.bridge import estimate as _estimate
    return _estimate(build_problem_dict(), effective_config())


# ============================================================================
# COMMAND LINE
# ============================================================================

if __name__ == "__main__":
    estimate()
