"""
problem_appendix_d.py -- EV Charging Scheduling with Heterogeneous Discrete-Level Chargers
(Appendix D): model, instances, operators, solver bindings, `problem_dict` -- one file for this problem.
=============================================================================

Problem      n bookings, m chargers of several types (each type a set of power
             set-points, e.g. {11}, {11, 22}, {11, 22, 43} kW), H slots of length
             tau. An accepted booking occupies one compatible charger during its
             whole stay S_j = {r_j, ..., d_j}; in each slot of the stay it draws
             nothing or one effective level of that charger; it receives exactly
             e_j kWh, the last charging slot possibly delivering only part of its
             level (the level stays reserved on the grid); in every slot the
             levels drawn do not exceed the grid capacity W_t.
Objective    minimise the value of the rejected bookings, sum_j v_j (1 - x_j)
             (v_j = 1: number of rejected bookings) -- equivalently, maximise the
             value of the accepted bookings.

Solution     float64 array of shape (n, 1 + H):
                 sol[j, 0]      = charger k + 1 serving booking j, 0 if rejected
                 sol[j, 1 + t]  = level (kW) drawn by booking j in slot t (0 if none)
             Methods return a compact copy (charger per booking, and the non-zero
             levels as (booking, slot, level) triples) as `best_solution`.

Operators    (all njit, all feasibility-preserving)
    StochasticInsertion  compatible chargers in random order; the first one that is
                         free during the whole stay and whose residual grid
                         capacity lets the booking reach its energy receives it.
                         Slots of the stay are visited in random order and given
                         the largest level that fits (LEVEL_RULE "max") or a random
                         fitting level ("random"), raised to the largest fitting
                         levels if the energy is not reached; the levels are then trimmed
                         chronologically so that the booking stops charging once its
                         energy is reached, the last level being lowered to the
                         smallest one that completes the booking.
    Generate_Solution    StochasticInsertion of every booking, in random order.
    Perturb              ceil(pc n) random bookings are removed, then re-inserted in
                         random order (rejected ones may re-enter).
    Crossover            Generalized Crossover Operator: bookings in random order
                         inherit charger and levels from a random parent when they
                         fit the child; the others are re-inserted afterwards.

This file is the methods_comparison binding of Appendix D: the operators given to methods_comparison, the exact
method as a method of the study, problem_dict and the duration estimate. The model itself (instances, operators, verifier, exact-method
interface) is problems/appendix_d/model.py, with unit_flow.py, in the folder problems/appendix_d/
that is identical in AnnexeD (the exact-method framework); its
problem-specific values are in problems/appendix_d/parameters.py. Every methods_comparison setting (methods, population size, generations, solver parameters and
tuning grids, runs, tuning protocol, parallelism, figures) is in comparison_config.py
in the main folder. methods_comparison runs its bundled solvers with the operators given in
problem_dict["operators"].

Usage (from Frameworks/methods_comparison):
    run_all.bat                                       # every problem, incl. this one
    python -m problems.problem_appendix_d estimate            # duration of the study on this machine
"""

import os
import time
from typing import Any, Dict, List

import numpy as np

from problems.appendix_d import parameters as P
from problems.appendix_d.model import (
    CODE, Crossover, Generate_Solution, AppendixDInstance, APPENDIX_D_DIR, PROBLEM_NAME,
    Perturb, RULES, _build_problem_params, _size_classes, _unit_flow, acn_instance,
    evaluate_objective, exact_available, exact_solver, finalize_solution, generate_instances,
    objective_function, solution_shape,
)

# ═══════════════════════════════════════════════════════════════════════════════
# OPERATORS FOR METHODS_COMPARISON'S BUNDLED SOLVERS
# ═══════════════════════════════════════════════════════════════════════════════

OPERATORS = {
    "objective": objective_function,
    "generate_solution": Generate_Solution,
    "perturb": Perturb,
    "crossover": Crossover,
    "build_problem_params": _build_problem_params,
    "solution_shape": solution_shape,
    "finalize": finalize_solution,
    "module": __name__,                     # workers re-import this binding (the operators live in problems.appendix_d.model)
}


# ═══════════════════════════════════════════════════════════════════════════════
# THE EXACT METHOD AS A METHOD OF THE STUDY
# ═══════════════════════════════════════════════════════════════════════════════

def UF(instance: AppendixDInstance, params: Dict[str, Any], progress_record: bool) -> Dict[str, Any]:
    """The exact method as a METHOD of the methods_comparison study (added to the solving methods in
    build_problem_dict), compared with the metaheuristics. One thread per run, like them; the Gurobi seed is
    drawn from the seeded generator of the run. The time returned covers the model, the
    search, the construction of the schedule and its verification; if every Gurobi
    session of the licence is in use (runs are executed in parallel), the run waits, and
    the wait is not counted."""
    uf = _unit_flow()
    import gurobipy as gp
    limit = float(params.get("time_limit") or P.EXACT_TIME_LIMIT)
    threads = int(params.get("threads", P.EXACT_METHOD_THREADS))
    seed = int(np.random.randint(0, 2_000_000_000))
    for attempt in range(400):
        t0 = time.perf_counter()
        try:
            r = uf.solve(instance, limit, threads=threads, seed=seed)
            break
        except gp.GurobiError as exc:
            licence = any(w in str(exc).lower() for w in ("license", "licence", "token", "session"))
            if not licence or attempt == 399:
                raise
            time.sleep(3.0)
    if r["solution"] is None:
        raise RuntimeError(f"no schedule found by the exact method ({r['status']})")
    best = finalize_solution(instance, r["solution"])          # verified, compact
    return {"best_objective": evaluate_objective(instance, best)[0], "best_solution": best,
            "progress": None, "exec_time": time.perf_counter() - t0, "exact_status": r["status"]}


def extra_methods() -> List[Any]:
    return [UF] if (exact_available() and P.EXACT_AS_METHOD) else []


def self_check() -> None:
    """Run by methods_comparison at start-up: the tuned problem-level parameter pc (and the
    level rule) really change the Generate -> Perturb -> Crossover -> objective
    pipeline."""
    from methods_comparison.utils.param_wiring_check import verify_param_wiring
    inst = acn_instance(P.SELF_CHECK_N, P.SELF_CHECK_SEED)
    verify_param_wiring(
        build_problem_params=lambda combo, i: _build_problem_params(combo, i),
        instance=inst,
        generate_solution_fn=Generate_Solution,
        perturb_fn=Perturb,
        objective_fn=objective_function,
        crossover_fn=Crossover,
        param_grids={"pc": list(P.PROBLEM_PARAM_GRID["pc"]), "level_rule": list(RULES)},
        baseline={"pc": P.UNTUNED_PC, "level_rule": P.LEVEL_RULE},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DURATION ESTIMATE ON THIS MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

def effective_config() -> Dict[str, Any]:
    """The configuration methods_comparison uses for this problem: comparison_config.py, then the
    problem-specific settings of STUDY_CONFIG, then RUN["overrides"][CODE]
    (as run_all.bat applies them)."""
    from methods_comparison.config import DEFAULT_CONFIG, RUN_SETTINGS
    from methods_comparison.run_all import merge
    own = merge(STUDY_CONFIG, RUN_SETTINGS.get("overrides", {}).get(CODE, {}))
    return merge(DEFAULT_CONFIG, own)


def estimate() -> float:
    """Expected duration of the study on this machine (python -m problems.problem_appendix_d estimate):
    the generic methods_comparison estimate for the bundled solvers (short timed runs, one instance
    per class), plus timed runs of the exact method when it is a method of the study."""
    from methods_comparison.solvers.bridge import estimate as _estimate
    cfg = effective_config()
    hours = _estimate(_base_problem_dict(), cfg)
    if extra_methods():
        from methods_comparison.utils.parallel import resolve_n_jobs
        workers = resolve_n_jobs(cfg["parallel"]["n_jobs"])
        workers = (os.cpu_count() or 1) if workers == -1 else max(1, int(workers or 1))
        insts, total = generate_instances(), 0.0
        fixed = cfg["tuning"]["fixed_params"]["UF"]
        line = f"  {P.EXACT_METHOD_LABEL} (exact, not tuned)"
        for c, ids in _size_classes().items():
            run = UF(insts[ids[0] - 1], fixed, False)["exec_time"]
            total += run * cfg["evaluation"]["n_runs_per_instance"] * len(ids)
            line += f"  {c} {run:.1f} s/run"
        print(line)
        print(f"  exact method: {total / 3600:.2f} core-h => about {hours + total / 3600 / workers:.1f} h in all")
        hours += total / 3600 / workers
    return hours


# ═══════════════════════════════════════════════════════════════════════════════
# THE PROBLEM DICT AND THE PROBLEM-SPECIFIC METHODS_COMPARISON SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

def _base_problem_dict() -> Dict[str, Any]:
    """The problem as methods_comparison expects it: instances, objective, operators."""
    return {
        "generate_instances": generate_instances,
        "objective_function": evaluate_objective,
        "name": "metaheuristics",
        "code": CODE,
        "objective_names": ["Rejected bookings"],
        "operators": OPERATORS,                 # methods_comparison runs its bundled solvers on them
        "problem_param_grids": dict(P.PROBLEM_PARAM_GRID),
        "instance_size_classes": _size_classes(),
        "instance_features": lambda inst: inst.features(),
        "exact_solver": exact_solver if exact_available() else None,
        "exact_solver_timelimit": P.EXACT_TIME_LIMIT,
        "self_check": self_check,
    }


def build_problem_dict(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """The problem dict given to methods_comparison. Without the exact method: the problem and its
    operators, on which methods_comparison runs its bundled solvers. With the exact method as a
    method of the study (parameters.py, EXACT_AS_METHOD): the problem gives its own
    "solving_methods", i.e. the bundled solvers on the operators, built by methods_comparison's own
    function for the configuration of the run, followed by the exact method. Nothing of
    this is in methods_comparison: the exact method enters the study through this file only.
    `config`: the methods_comparison configuration of the run (default: effective_config())."""
    pdict = _base_problem_dict()
    if extra_methods():
        from methods_comparison.solvers.bridge import attach_operator_methods
        pdict = attach_operator_methods(pdict, config if config is not None else effective_config())
        pdict["solving_methods"] = list(pdict["solving_methods"]) + extra_methods()
        pdict["method_labels"] = {**pdict["method_labels"], "UF": P.EXACT_METHOD_LABEL}
    return pdict


# settings in which this study departs from comparison_config.py
STUDY_CONFIG: Dict[str, Any] = {
    # the tests of the appendix, on the objective and on the execution time: Friedman
    # and Kendall's W, Nemenyi critical difference, pairwise Wilcoxon (Holm), and against
    # the best method the Friedman post-hoc and Wilcoxon tests (Holm, A12), the Bayesian
    # signed-rank test and the per-instance Mann-Whitney tests
    "statistics": {"tested_blocks": ["objective", "time"], "nemenyi": True, "cliff_delta": False,
                   "mann_whitney": True, "holm": True, "bayesian": True},
    # the summary tables; the runs, one row each, for AnnexeD/complements.py
    "report": {"summary": True, "tidy_csv": True},
    # gap to the optimum over time, and share of the runs reaching the optimum
    "anytime": {"enabled": True, "targets": [0.0]},
    # computing time against the number of bookings
    "scaling": {"enabled": True, "features": ["n"], "blocks": ["time"]},
}
# the exact method has nothing to tune (fixed parameters: it is not part of the tuning phase)
if extra_methods():
    STUDY_CONFIG["tuning"] = {"fixed_params": {"UF": {"threads": P.EXACT_METHOD_THREADS,
                                                      "time_limit": P.EXACT_TIME_LIMIT}}}
problem_dict = build_problem_dict()


# ============================================================================
# COMMAND LINE
# ============================================================================

if __name__ == "__main__":
    estimate()
