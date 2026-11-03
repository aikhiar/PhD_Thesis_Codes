"""
problem_appendix_c.py -- Appendix C of the thesis (bi-objective EV charging
scheduling for grid stability and user satisfaction): the methods_comparison binding of the
model: operators, `problem_dict`, the settings of the study and end-to-end
tests.
=============================================================================

Model        problems/appendix_c/model.py (the code of the thesis: instance sampling,
             operators, objectives, verifiers), with its parameters in
             problems/appendix_c/parameters.py.
Instances    20 from the ACN-Data claimed sessions, n = 50, 100, 150, 200 (5 per
             size), m = ceil(n / 3).
Objectives   (peak load kW, total completion time in slots), deterministic.

This module binds the model of the thesis to methods_comparison: the solving
methods are the bundled NSGA-II, NSGA-III, MOCS, MOPSO and MOGWO, built from the
operators; it changes no operator.

Usage (from methods_comparison):
    run_all.bat --only appendix_c     # the study
    python -m problems.problem_appendix_c estimate # duration of the study on this machine
"""

from typing import Any, Dict

import numpy as np

from problems.appendix_c import model as M
from problems.appendix_c import parameters as P

PROBLEM_NAME = "appendix_c"
CODE = "appendix_c"


# ═══════════════════════════════════════════════════════════════════════════════
# OPERATORS FOR METHODS_COMPARISON'S BUNDLED SOLVERS
# ═══════════════════════════════════════════════════════════════════════════════

OPERATORS = {
    "objective": M.objective_function,
    "generate_solution": M.Generate_Solution,
    # perturb (MOCS's nest replacement, MOPSO's non-guided move) uses FindNeighbour
    # (current charger allowed); mutate (NSGA-II, NSGA-III, MOGWO) uses Perturb
    # (forces a different charger)
    "perturb": M.FindNeighbour,
    "mutate": M.Perturb,
    "crossover": M.Crossover,
    "obj_signs": np.array([1.0, 1.0], dtype=np.float64),
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
        instance=M.simulate_instance(requests=M._load_or_mock_requests(), n=50, m=15, tau=P.DEFAULT_TAU, seed=999),
        generate_solution_fn=M.Generate_Solution,
        perturb_fn=M.Perturb,
        objective_fn=M.objective_function,
        crossover_fn=M.Crossover,
        param_grids={
            "perc_perturb": [0.05, 0.10, 0.20],
            "lam": [0.5, 1.0, 2.0],
        },
        baseline={"perc_perturb": P.DEFAULT_PERC_PERTURB, "lam": P.DEFAULT_LAM},
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
    comparison_config.py, then STUDY_CONFIG, then RUN["overrides"]["appendix_c"]."""
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
