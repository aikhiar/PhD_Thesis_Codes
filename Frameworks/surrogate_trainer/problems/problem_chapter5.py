"""
problem_chapter5.py -- Chapter 5 (tri-objective preemptive EV charging under
stochastic charging durations), surrogate_trainer binding: the contract
of dependency/training_data_generator/generator.py on the shared model
problems/chapter5/model.py (identical in the two frameworks of the Chapter 5 study).

Training data: the five metaheuristics of the study, with the FIXED parameters
of ALGORITHMS below, run on the training pool (parameters.py:
TRAINING_INSTANCES_PER_SIZE per size class, seeds TRAINING_SEED_BASE + k,
disjoint from the evaluation seeds but drawn from the same ACN session pool);
every logged schedule is labelled by the Monte Carlo objective with
LABEL_REPLICATIONS (N_r) replications. The peak load and the total tardiness
are learned; the undelivered energy is deterministic and computed exactly.
"""

from typing import Any, Dict, List, Tuple

from problems.chapter5 import model as M
from problems.chapter5 import parameters as P

CODE = "chapter5"
n_objectives = len(M.OBJECTIVE_NAMES)
objective_names = list(M.OBJECTIVE_NAMES)          # [peak load, total tardiness, undelivered energy]
SURROGATE_OBJECTIVE_NAMES = list(M.SURROGATE_OBJECTIVE_NAMES)
FEATURE_NAMES = list(M.FEATURE_NAMES)
LABEL_REPLICATIONS = P.LABEL_REPLICATIONS


def generate_training_instances() -> List[Tuple[str, Any]]:
    """(id, instance) pairs of the training pool; ids '<size label>_<j>'."""
    return M.training_instances()


def instance_class(instance_id: str) -> str:
    """Size class of a training instance, from its id."""
    return str(instance_id).rsplit("_", 1)[0]


def exact_objective_values(instance: Any, solution) -> Dict[str, float]:
    """The undelivered energy, a deterministic function of the plan."""
    return {"Undelivered energy (kWh)": M.undelivered_energy(solution, instance)}


solution_shape = M.solution_shape
build_problem_params = M.build_problem_params     # num_of_replications defaults to N_r
Generate_Solution = M.Generate_Solution
Perturb = M.Perturb
Crossover = M.Crossover
objective_function = M.objective_function       # Monte Carlo with problem_params' replications
feature_fn = M.feature_fn
build_instance = M.build_instance

# Fixed-parameter portfolio of the training-data generation: the study's
# population and generations, and the middle of each tuning grid.
_PROBLEM = {"perc_perturb": P.UNTUNED_PERC_PERTURB, "lam": P.FREE_REGION_RATE,
            "num_of_replications": P.LABEL_REPLICATIONS}
_RUN = {"population_size": 80, "generations": 150}
ALGORITHMS: Dict[str, Dict[str, Any]] = {
    "NSGA2": {"kind": "multi", "params": dict(_PROBLEM),
              "solve_kwargs": {**_RUN, "crossover_pool_size": 2, "mutation_rate": 0.10}},
    "NSGA3": {"kind": "multi", "params": dict(_PROBLEM),
              "solve_kwargs": {**_RUN, "n_divisions": 6, "crossover_pool_size": 2, "mutation_rate": 0.10}},
    "MOCS": {"kind": "multi", "params": dict(_PROBLEM),
             "solve_kwargs": {**_RUN, "p_abandon": 0.25, "elite_fraction": 0.2}},
    "MOPSO": {"kind": "multi", "params": dict(_PROBLEM),
              "solve_kwargs": {**_RUN, "archive_size": 60, "n_grid": 10, "beta": 1.0, "mutation_rate": 0.10}},
    "MOGWO": {"kind": "multi", "params": dict(_PROBLEM),
              "solve_kwargs": {**_RUN, "archive_size": 60, "n_grid": 10, "beta": 1.0, "mutation_rate": 0.10}},
}
