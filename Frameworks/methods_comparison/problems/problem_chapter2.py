"""
problem_chapter2.py -- Chapter 2 of the thesis (bi-objective EV charging
scheduling under stochastic vehicle arrivals): the methods_comparison binding of the model:
operators, `problem_dict` and the settings of the study.
=============================================================================

Model        problems/chapter2/model.py (the code of the thesis: instance sampling,
             operators, objectives, verifiers), with its parameters in
             problems/chapter2/parameters.py.
Instances    20 from the ACN-Data claimed sessions, n = 50, 100, 150, 200 (5 per
             size), m = ceil(n / 3).
Objectives   (peak load kW, mean relative tardiness), expectations over the
             attendance of the vehicles (Monte Carlo with N_r replications for
             the peak load) on a stochastic instance.

This module binds the model of the thesis to methods_comparison: the solving
methods are the bundled NSGA-II, NSGA-III, MOCS, MOPSO and MOGWO, built from the
operators; it changes no operator.

Usage (from methods_comparison):
    run_all.bat --only chapter2     # the study
    python -m problems.problem_chapter2 estimate # duration of the study on this machine
"""

from typing import Any, Dict


from problems.chapter2 import model as M
from problems.chapter2 import parameters as P

PROBLEM_NAME = "chapter2"
CODE = "chapter2"


# ═══════════════════════════════════════════════════════════════════════════════
# OPERATORS FOR METHODS_COMPARISON'S BUNDLED SOLVERS
# ═══════════════════════════════════════════════════════════════════════════════

OPERATORS = {
    "objective": M.objective_function,
    "generate_solution": M.Generate_Solution,
    "perturb": M.Perturb,
    "crossover": M.Crossover,
    "build_problem_params": M.build_problem_params,
    "solution_shape": M.solution_shape,
    "n_objectives": len(M.OBJECTIVE_NAMES),
    "module": __name__,                     # workers re-import this binding
}


def evaluate_objective(instance, solution):
    """problem_dict["objective_function"]: the objectives of a schedule."""
    return M.evaluate_objective(instance, solution)


def self_check() -> None:
    """Run by methods_comparison at start-up: the problem-level parameters really change the
    Generate -> Perturb -> Crossover -> objective pipeline."""
    from methods_comparison.utils.param_wiring_check import verify_param_wiring
    verify_param_wiring(
        build_problem_params=M.build_problem_params,
        instance=M.simulate_instance(requests=M._load_or_mock_requests_hours(seed=99), n=50, m=15, tau=P.DEFAULT_TAU, seed=999, stochastic=True),
        generate_solution_fn=M.Generate_Solution,
        perturb_fn=M.Perturb,
        objective_fn=M.objective_function,
        crossover_fn=M.Crossover,
        param_grids={
            "perc_perturb": [0.05, 0.10, 0.20],
            "lam": [2.0, 5.0, 10.0],
            "num_of_replications": [5, 50, 100],
        },
        baseline={"perc_perturb": P.DEFAULT_PERC_PERTURB, "lam": P.DEFAULT_LAM,
                  "num_of_replications": P.DEFAULT_NUM_REPLICATIONS},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# THE PROBLEM DICT AND THE PROBLEM-SPECIFIC METHODS_COMPARISON SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# settings in which this study departs from comparison_config.py: the protocol of
# the thesis (population 80, 150 generations, the grids of parameters.py)
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
    comparison_config.py, then STUDY_CONFIG, then RUN["overrides"]["chapter2"]."""
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
