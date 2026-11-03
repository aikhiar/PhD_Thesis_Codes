"""
problem_chapter4.py -- Chapter 4 of the thesis, det_vs_stoch binding: the deterministic-
vs-stochastic regime contract of main.py on the shared model
problems/chapter4/model.py (the same folder as in methods_comparison).

Regimes       MOCS is tuned and run independently under each regime (deterministic
              is the reference); the same instance under both, the regime lives in
              num_of_replications:
                deterministic  the plan is scored on the planned durations (0);
                stochastic     Monte Carlo over the lognormal durations (N_r).
Instances     8 of the 20 evaluation instances: the first 2 of each size class.
Resampling    realize_one_draw applies ONE lognormal draw to a planned schedule
              (common random numbers); score_realized scores it deterministically.
"""

from typing import Any, Dict, List

import numpy as np

from problems.chapter4 import model as M
from problems.chapter4 import parameters as P
from dependency.algorithms.multi_objective_solving_module import Problem, solve
from dependency.resampling.seeding import seed_everything

PROBLEM_NAME = "chapter4"
CODE = "chapter4"
BEST_ALGORITHM = "MOCS"
N_EVAL_RUNS = 5                              # evaluation runs per (instance, regime), as in the thesis

REGIMES = {
    "deterministic": {"forced_problem_params": {"num_of_replications": 0}},
    "stochastic": {"forced_problem_params": {"num_of_replications": P.DEFAULT_NUM_REPLICATIONS}},
}
REGIME_LABELS = {"deterministic": "Deterministic", "stochastic": "Stochastic"}

# what the chapter reports (dependency/reporting/reporting.py): the pooled-front test
# per instance, Bonferroni-corrected across the instances
REPORT = {"correction": "bonferroni", "across_instances": False, "run_level": False,
          "promise_vs_delivery": False, "realized_fronts": False}

# MOCS under the protocol of the thesis (population 80, 150 generations, the
# tuning grid of the metaheuristics comparison), plus the problem-level grid
# without num_of_replications, which the regimes set
problem_dict = {
    "objective_names": list(M.OBJECTIVE_NAMES),
    "instance_size_classes": M.size_classes(),
    "problem_param_grids": {k: v for k, v in P.PROBLEM_PARAM_GRID.items() if k != "num_of_replications"},
    "param_grids": {
        "MOCS": {"population_size": [80], "generations": [150], "p_abandon": [0.25],
                 "elite_fraction": [0.1, 0.2, 0.3]},
    },
}
INSTANCE_SIZE_CLASSES = problem_dict["instance_size_classes"]

# untuned values (used when RESAMPLING_CONFIG["tuning"]["enabled"] is False)
DEFAULT_PERC_PERTURB = P.DEFAULT_PERC_PERTURB
DEFAULT_POPULATION_SIZE = 80
DEFAULT_GENERATIONS = 150
DEFAULT_P_ABANDON = 0.25
DEFAULT_ELITE_FRACTION = 0.3


def param_grid_for_best_algorithm() -> Dict[str, List[Any]]:
    grid: Dict[str, List[Any]] = dict(problem_dict["problem_param_grids"])
    grid.update(problem_dict["param_grids"][BEST_ALGORITHM])
    return grid


def default_params_for_best_algorithm() -> Dict[str, Any]:
    return {key: globals()[f"DEFAULT_{key.upper()}"] for key in param_grid_for_best_algorithm()}


def run_algorithm(instance: Any, params: Dict[str, Any], regime: str, progress_record: bool) -> Dict:
    """One MOCS run under `regime`. params["seed"] seeds Python, NumPy and Numba
    (operators and Monte Carlo draws); the solver draws its own generator seed
    from the seeded NumPy RNG, as in methods_comparison."""
    params = dict(params)
    seed = params.pop("seed", None)
    if seed is not None:
        seed_everything(int(seed))
    problem_params = M.build_problem_params({**params, **REGIMES[regime]["forced_problem_params"]}, instance)
    problem = Problem(
        n_objectives=len(M.OBJECTIVE_NAMES),
        solution_shape=M.solution_shape(instance),
        objective_function=M.objective_function,
        generate_solution=M.Generate_Solution,
        perturb=M.Perturb,
        crossover=M.Crossover,
        problem_params=problem_params,
        name=f"{CODE}_{regime}",
    )
    res = solve(
        problem, algorithm=BEST_ALGORITHM, seed=None, show_progress=False, record_progress=bool(progress_record),
        population_size=int(params.get("population_size", DEFAULT_POPULATION_SIZE)),
        generations=int(params.get("generations", DEFAULT_GENERATIONS)),
        p_abandon=float(params.get("p_abandon", DEFAULT_P_ABANDON)),
        elite_fraction=float(params.get("elite_fraction", DEFAULT_ELITE_FRACTION)),
    )
    objectives = np.asarray(res.pareto_objectives)
    return {
        "pareto_front": [np.asarray(s) for s in res.pareto_solutions],
        "pareto_objectives": objectives.tolist() if objectives.size else [],
        "progress": res.progress_fronts if progress_record else None,
    }


def generate_instance_seeds() -> List[int]:
    """Evaluation ids of the comparison: the first 2 instances of each size class
    (the 'seed' of an instance in this framework is its 1-based evaluation id)."""
    seeds: List[int] = []
    for c in range(len(P.SIZE_CLASSES)):
        base = c * P.INSTANCES_PER_CLASS
        seeds.extend([base + 1, base + 2])
    return seeds


def build_instance(seed: int, regime: str) -> Any:
    """The same instance under both regimes (the regime lives in the problem
    parameters only)."""
    return M.suite_instance(int(seed))


def realize_one_draw(instance: Any, schedule: np.ndarray, seed: int) -> np.ndarray:
    """ONE lognormal realization of the durations, by the helper the Monte Carlo
    objective uses; it is @njit, so seed_everything (which also seeds Numba) makes
    the draw reproducible."""
    seed_everything(int(seed))
    updated_schedule, _durations = M._realize_schedule_once_lognormal(
        np.asarray(schedule, dtype=np.float64), instance, P.DEFAULT_VARIANCE_FACTOR,
    )
    return updated_schedule


def score_realized(instance: Any, original_schedule: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """Objectives of an already realized schedule (no further realization)."""
    return M._objective_deterministic(realized, instance)
