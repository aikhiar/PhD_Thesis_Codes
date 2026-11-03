"""
problem_chapter6.py -- Chapter 6 (preemptive EV charging under joint arrival and
energy uncertainty), det_vs_stoch binding: the deterministic-vs-
stochastic regime contract of main.py on the shared model problems/chapter6/model.py
(identical in the three frameworks of the study).

Regimes       MOCS (the study's best algorithm) is tuned and run independently
              under each regime (deterministic is the reference):
                deterministic  the plan is scored on the declared arrivals and
                               energies (num_of_replications = 0);
                stochastic     the plan is scored by Monte Carlo over kappa with
                               LABEL_REPLICATIONS (N_r) replications.
Instances     the 20 evaluation instances of methods_comparison (5 per size class;
              DET_STOCH_INSTANCES_PER_SIZE in parameters.py).
Resampling    realize_one_draw applies ONE joint draw of kappa to a planned
              schedule through the recourse policy (common random numbers across
              the regimes' fronts); score_realized scores it.
"""

from typing import Any, Dict, List

import numpy as np

from problems.chapter6 import model as M
from problems.chapter6 import parameters as P
from dependency.algorithms.multi_objective_solving_module import Problem, solve
from dependency.resampling.seeding import seed_everything

PROBLEM_NAME = "chapter6"
CODE = "chapter6"
BEST_ALGORITHM = "MOCS"

# the deterministic regime is the reference of the comparison
REGIMES = {
    "deterministic": {"forced_problem_params": {"num_of_replications": 0}},
    "stochastic": {"forced_problem_params": {"num_of_replications": P.LABEL_REPLICATIONS}},
}
REGIME_LABELS = {"deterministic": "Deterministic", "stochastic": "Stochastic"}

# what the chapter reports (dependency/reporting/reporting.py): the pooled-front tests
# per instance (Holm) and across instances, the run-level tests, promise against
# delivery with the realized objectives, and the realized fronts
REPORT = {"correction": "holm", "across_instances": True, "run_level": True,
          "promise_vs_delivery": True, "realized_fronts": True}

# MOCS under the protocol of the paper (population 80, 150 generations, the
# tuning grid of the metaheuristics comparison), plus the problem-level grid
problem_dict = {
    "objective_names": list(M.OBJECTIVE_NAMES),
    "instance_size_classes": M.size_classes(),
    "problem_param_grids": dict(P.PROBLEM_PARAM_GRID),
    "param_grids": {
        "MOCS": {"population_size": [80], "generations": [150], "p_abandon": [0.25],
                 "elite_fraction": [0.1, 0.2, 0.3]},
    },
}
INSTANCE_SIZE_CLASSES = problem_dict["instance_size_classes"]

# untuned values (used when RESAMPLING_CONFIG["tuning"]["enabled"] is False)
DEFAULT_PERC_PERTURB = P.UNTUNED_PERC_PERTURB
DEFAULT_POPULATION_SIZE = 80
DEFAULT_GENERATIONS = 150
DEFAULT_P_ABANDON = 0.25
DEFAULT_ELITE_FRACTION = 0.3


def generate_instance_seeds() -> List[int]:
    """Evaluation ids of the comparison (the 'seed' of an instance in this
    framework is its 1-based evaluation id)."""
    return M.det_stoch_ids()


def build_instance(seed: int, regime: str) -> Any:
    """The same instance under both regimes (the regime lives in the problem
    parameters only)."""
    return M.evaluation_instance(int(seed))


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
    spec = REGIMES[regime]
    problem_params = M.build_problem_params({**params, **spec["forced_problem_params"]}, instance)
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
    solutions = [np.asarray(s) for s in res.pareto_solutions]
    objectives = np.asarray(res.pareto_objectives)
    return {
        "pareto_front": solutions,
        "pareto_objectives": objectives.tolist() if objectives.size else [],
        "progress": res.progress_fronts if progress_record else None,
    }


def realize_one_draw(instance: Any, schedule: np.ndarray, seed: int) -> np.ndarray:
    """ONE joint draw of kappa applied to `schedule` by the recourse policy; the
    seed fixes the draw (common random numbers across the fronts)."""
    seed_everything(int(seed))
    realized, _, _ = M.realize(schedule, instance)
    return realized


def score_realized(instance: Any, original_schedule: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """Objectives of an already realized schedule (no further realization)."""
    return M.score_realized(realized, instance)
