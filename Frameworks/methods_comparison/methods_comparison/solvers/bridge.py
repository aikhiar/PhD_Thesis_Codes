"""
bridge.py  --  the bundled solvers as methods_comparison solving methods, built from a
problem's operators.

A study binding that does not write its own solving methods provides, in its
problem_dict, an entry

    "operators": {
        "objective":            f(solution, problem_params) -> float (mono) or 1D array (multi),
        "generate_solution":    f(problem_params) -> solution,
        "perturb":              f(solution, problem_params) -> solution,
        "crossover":            f(parents_array, problem_params) -> solution,
        "build_problem_params": f(params, instance) -> problem_params (tuple),
        "solution_shape":       f(instance) -> tuple,
        # optional
        "mutate":               f(solution, problem_params) -> solution (default: perturb),
        "on_generation":        f(problem_params, objectives) -> None, called after the
                                initialisation and after every generation (multi-objective
                                solvers only; see Problem.on_generation),
        "finalize":             f(instance, solution) -> solution as stored in the results
                                (e.g. a feasibility check and a compact copy),
        "n_objectives":         int (default: len(objective_names)),
        "obj_signs":            array of +1/-1 (multi-objective only),
        "module":               importable module name of the problem (default: derived),
    }

methods_comparison then runs the methods of CONFIG["solvers"]["mono_methods"] (or
"multi_methods"), with the parameters and tuning grids of
CONFIG["solvers"]["algorithms"] (comparison_config.py), merged with the problem's
"problem_param_grids". A problem may restrict the methods in its own
STUDY_CONFIG, e.g. {"solvers": {"mono_methods": ["CS"]}}.
"""

import importlib
import inspect
import sys
import time
from typing import Dict

import numpy as np

from . import mono_objective_solving_module as MONO
from . import multi_objective_solving_module as MULTI

_CALL_KEYS = {"problem", "seed", "show_progress", "time_limit", "record_progress"}
REQUIRED_OPERATORS = ("objective", "generate_solution", "perturb", "crossover",
                      "build_problem_params", "solution_shape")


def n_objectives(problem_dict: Dict) -> int:
    ops = problem_dict.get("operators") or {}
    return int(ops.get("n_objectives", len(problem_dict.get("objective_names") or [None])))


def solver_module(n_obj: int):
    return MULTI if n_obj > 1 else MONO


def _module_name(ops: Dict) -> str:
    """Importable name of the problem module (parallel workers re-import it)."""
    name = ops.get("module") or getattr(ops["build_problem_params"], "__module__", None)
    if name == "__main__":                      # run as `python -m problems.xxx`
        spec = getattr(sys.modules["__main__"], "__spec__", None)
        if spec is not None and spec.name:
            name = spec.name
    if not name:
        raise ValueError("cannot determine the module of the operators; set operators['module']")
    return name


def solver_kwargs(alg: str, params: Dict, solvers_cfg: Dict, module) -> Dict:
    """Keyword arguments of the bundled solver `alg`: untuned values of the
    config, overridden by the (tuned) params; population size and generations."""
    if alg not in solvers_cfg["algorithms"]:
        raise KeyError(f"no entry for {alg!r} in CONFIG['solvers']['algorithms'] (comparison_config.py)")
    spec = solvers_cfg["algorithms"][alg]
    kw = {k: params.get(k, v) for k, v in spec.get("params", {}).items()}
    for k in spec.get("grid", {}):
        if k in params:
            kw[k] = params[k]
    kw["population_size"] = int(params.get("population_size", solvers_cfg["population_size"]))
    kw["generations"] = int(params.get("generations", solvers_cfg["generations"]))
    accepted = set(inspect.signature(module._ALGORITHMS[alg]).parameters)
    unknown = sorted(set(kw) - accepted)
    if unknown:
        raise ValueError(f"{alg}: {unknown} are not parameters of the bundled solver "
                         f"(check CONFIG['solvers']['algorithms'][{alg!r}])")
    return kw


def evaluations_per_generation(alg: str, kw: Dict) -> int:
    """Objective evaluations per generation of the mono-objective bundled solvers
    (the initial population adds population_size evaluations)."""
    N = kw["population_size"]
    if alg == "CS":
        return N + int(kw["p_abandon"] * N)
    if alg == "ES":
        return max(1, int(kw["lambda_ratio"] * N))
    return N


class OperatorMethod:
    """One bundled solver applied to a problem's operators; a picklable callable
    with the methods_comparison method signature (instance, params, progress_record)."""

    def __init__(self, algorithm: str, module: str, solvers_cfg: Dict, n_obj: int):
        self.algorithm = algorithm
        self.module = module
        self.solvers_cfg = solvers_cfg
        self.n_obj = n_obj
        self.__name__ = self.__qualname__ = algorithm

    def __repr__(self):
        return f"OperatorMethod({self.algorithm} on {self.module})"

    def operators(self) -> Dict:
        mod = sys.modules.get(self.module) or importlib.import_module(self.module)
        return mod.problem_dict["operators"]

    def make_problem(self, instance, params: Dict, objective=None):
        ops = self.operators()
        common = dict(solution_shape=tuple(ops["solution_shape"](instance)),
                      objective_function=objective or ops["objective"],
                      generate_solution=ops["generate_solution"], perturb=ops["perturb"],
                      crossover=ops["crossover"], mutate=ops.get("mutate"),
                      problem_params=ops["build_problem_params"](params, instance),
                      name=self.module.rsplit(".", 1)[-1])
        if self.n_obj > 1:
            return MULTI.Problem(n_objectives=self.n_obj, obj_signs=ops.get("obj_signs"),
                                 on_generation=ops.get("on_generation"), **common)
        return MONO.Problem(**common)

    def __call__(self, instance, params: Dict, progress_record: bool) -> Dict:
        module = solver_module(self.n_obj)
        kw = solver_kwargs(self.algorithm, params, self.solvers_cfg, module)
        accepted = set(inspect.signature(module._ALGORITHMS[self.algorithm]).parameters)
        call = {"seed": None, "show_progress": False}
        if "time_limit" in accepted:
            call["time_limit"] = params.get("time_limit")
        if "record_progress" in accepted:
            call["record_progress"] = bool(progress_record)
        res = module.solve(self.make_problem(instance, params), algorithm=self.algorithm, **call, **kw)
        finalize = self.operators().get("finalize") or (lambda inst, sol: sol)
        if self.n_obj > 1:
            objs = np.asarray(res.pareto_objectives)
            return {"pareto_front": [finalize(instance, s) for s in res.pareto_solutions],
                    "pareto_objectives": objs.tolist() if objs.size else [],
                    "progress": res.progress_fronts if progress_record else None}
        out = {"best_objective": float(res.best_objective),
               "best_solution": finalize(instance, res.best_solution)}
        if progress_record and res.history_best is not None:
            hist = [float(x) for x in res.history_best]
            epg = evaluations_per_generation(self.algorithm, kw)
            out["progress"] = hist
            out["progress_time"] = [float(x) for x in (res.history_time or [])[:len(hist)]]
            out["progress_evals"] = [kw["population_size"] + (g + 1) * epg for g in range(len(hist))]
        return out


def method_names(problem_dict: Dict, cfg: Dict) -> list:
    key = "multi_methods" if n_objectives(problem_dict) > 1 else "mono_methods"
    return list(cfg["solvers"][key])


def attach_operator_methods(problem_dict: Dict, cfg: Dict) -> Dict:
    """problem_dict with "solving_methods", "param_grids" and "method_labels" built
    from its "operators" and cfg["solvers"]; unchanged when the problem provides
    its own "solving_methods" or no "operators"."""
    ops = problem_dict.get("operators")
    if not ops or problem_dict.get("solving_methods"):
        return problem_dict
    missing = [k for k in REQUIRED_OPERATORS if not callable(ops.get(k))]
    if missing:
        raise ValueError(f"problem_dict['operators'] misses the callables {missing}")
    S = cfg["solvers"]
    n_obj = n_objectives(problem_dict)
    module = _module_name(ops)
    names = method_names(problem_dict, cfg)
    algos = solver_module(n_obj)._ALGORITHMS
    bad = [a for a in names if a not in algos or a not in S["algorithms"]]
    if bad:
        raise ValueError(f"methods {bad} are not bundled {'multi' if n_obj > 1 else 'mono'}-objective "
                         f"solvers with an entry in CONFIG['solvers']['algorithms']")
    grids, labels = {}, {}
    for a in names:
        spec = S["algorithms"][a]
        grid = {"population_size": [S["population_size"]], "generations": [S["generations"]]}
        grid.update({k: [v] for k, v in spec.get("params", {}).items() if k not in spec.get("grid", {})})
        grid.update({k: list(v) for k, v in spec.get("grid", {}).items()})
        grids[a] = grid
        labels[a] = spec.get("label", a)
    out = dict(problem_dict)
    out["solving_methods"] = [OperatorMethod(a, module, S, n_obj) for a in names]
    out["param_grids"] = {**grids, **(problem_dict.get("param_grids") or {})}
    out["method_labels"] = {**labels, **(problem_dict.get("method_labels") or {})}
    return out


def estimate(problem_dict: Dict, cfg: Dict, gens: int = 3, workers: int = None) -> float:
    """Times a short run (`gens` generations) of every method on the first instance
    of each size class, extrapolates to the configured generations, and prints the
    expected wall clock of tuning (grid) and evaluation. Returns hours."""
    import os
    from ..utils.parallel import resolve_n_jobs, physical_cores
    pd = attach_operator_methods(problem_dict, cfg)
    if workers is None:
        workers = resolve_n_jobs(cfg["parallel"]["n_jobs"])
        workers = (os.cpu_count() or 1) if workers == -1 else max(1, int(workers or 1))
    insts = list(pd["generate_instances"]())
    classes = pd.get("instance_size_classes") or {f"#{k}": [k] for k in range(1, len(insts) + 1)}
    ppg = pd.get("problem_param_grids") or {}
    L = cfg["solvers"]["generations"]
    print(f"{pd.get('name', 'problem')}: {len(insts)} instances, methods "
          f"{[m.__name__ for m in pd['solving_methods']]}, {workers} worker(s) "
          f"(physical cores detected: {physical_cores()})")
    tot_eval = tot_tune = 0.0
    for c, ids in classes.items():
        inst = insts[ids[0] - 1]
        line = f"  {c:12s}"
        for m in pd["solving_methods"]:
            grid = {**pd["param_grids"].get(m.__name__, {}), **ppg}
            combos = int(np.prod([len(v) for v in grid.values()])) if grid else 1
            np.random.seed(0)
            m(inst, {"generations": gens}, False)                 # compile / warm up
            t0 = time.perf_counter()
            m(inst, {"generations": gens}, False)
            run = (time.perf_counter() - t0) / (gens + 1) * (L + 1)
            tot_eval += run * cfg["evaluation"]["n_runs_per_instance"] * len(ids)
            if cfg["tuning"]["enabled"]:
                tot_tune += run * cfg["tuning"]["n_runs_per_combo"] * combos * len(ids)
            line += f"  {m.__name__} {run:6.1f} s/run"
        print(line)
    hours = (tot_eval + tot_tune) / 3600 / workers
    print(f"  evaluation {tot_eval / 3600:.1f} core-h, tuning {tot_tune / 3600:.1f} core-h "
          f"=> about {hours:.1f} h on {workers} worker(s)")
    return hours
