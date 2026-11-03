"""
problem_chapter2.py -- Chapter 2 of the thesis, det_vs_stoch binding: the deterministic-
vs-stochastic regime contract of main.py on the shared model
problems/chapter2/model.py (the same folder as in methods_comparison).

Regimes       MOCS is tuned and run independently under each regime (deterministic
              is the reference). The regime is a property of the instance:
                deterministic  full attendance (rho = 1, objectives without Monte Carlo);
                stochastic     rho = SYNTHETIC_RHO, Monte Carlo over the attendance.
              The two instances share the requests, the chargers and r_ij (same seed).
Instances     the 20 evaluation instances of methods_comparison (5 per size class).
Resampling    realize_one_draw draws ONE attendance (Bernoulli rho) from the stochastic
              instance; score_realized scores the realized schedule deterministically.
"""

from typing import Any, Dict, List

import numpy as np

from problems.chapter2 import model as M
from problems.chapter2 import parameters as P
from dependency.algorithms.multi_objective_solving_module import Problem, solve
from dependency.resampling.seeding import seed_everything

PROBLEM_NAME = "chapter2"
CODE = "chapter2"
BEST_ALGORITHM = "MOCS"
N_EVAL_RUNS = 5                              # evaluation runs per (instance, regime), as in the thesis

# Both regimes force the same num_of_replications: the regime lives in the instance
# (its stochastic flag), which objective_function dispatches on.
REGIMES = {
    "deterministic": {"forced_problem_params": {"num_of_replications": P.DEFAULT_NUM_REPLICATIONS}},
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
DEFAULT_LAM = P.DEFAULT_LAM


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
    """Evaluation ids of the comparison: the 20 instances (the 'seed' of an
    instance in this framework is its 1-based evaluation id)."""
    return list(range(1, len(P.SIZE_CLASSES) * P.INSTANCES_PER_CLASS + 1))


def build_instance(seed: int, regime: str) -> Any:
    """The instance `seed` under `regime` (see REGIMES)."""
    return M.suite_instance(int(seed), stochastic=(regime == "stochastic"))


def realize_one_draw(instance: Any, schedule: np.ndarray, seed: int) -> np.ndarray:
    """ONE Bernoulli attendance (rho of the stochastic instance). A vehicle that
    does not attend gets an empty window, [d + 1, d]: no load and no tardiness, so
    that the deterministic objective scores the realized schedule directly. A
    local generator: no @njit function consumes randomness here."""
    rng = np.random.default_rng(seed)
    n = int(instance[8])
    rho = np.asarray(instance[12], dtype=np.float64)
    departure_time = instance[1]
    active = rng.random(n) < rho
    realized = np.asarray(schedule, dtype=np.int32).copy()
    for i in range(n):
        if not active[i]:
            d = int(departure_time[i])
            realized[i, 1] = d + 1
            realized[i, 2] = d
    return realized


def score_realized(instance: Any, original_schedule: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """Objectives of an already realized schedule (no further realization)."""
    return M._objective_deterministic(realized, instance)
