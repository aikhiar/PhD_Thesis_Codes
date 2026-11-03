"""
problem.py  —  read-only view over the `problem_dict` of a study binding.

problem_dict contract (problems/problem_<study>.py)
===================================================

Required
--------
generate_instances : Callable[[], list]
    The fixed list of instances of the study. Called ONCE; its RNG is seeded
    by the framework before the call (config["instances"]["seed_base"]).
objective_function : Callable[[instance, solution], tuple[float, ...]]
    All objectives of a solution in one call; every objective is MINIMISED.
solving_methods : list[Callable[[instance, params: dict, progress_record: bool], dict]]
    Built from the study's "operators" by solvers/bridge.py (or by the binding
    itself, for the surrogate-assisted methods of Chapters 5 and 6).
    Mono  result dict : {"best_solution", "best_objective", "progress", ...}
    Multi result dict : {"pareto_front", "pareto_objectives", "progress"}
code : str
    Name of the study (e.g. "chapter2"); every file the framework saves is
    prefixed with it.
objective_names : list[str]
    One name per objective; one objective = mono-objective study.

Optional
--------
name                  : str
param_grids           : {method_name: {param: [values...]}}     (built by the bridge)
problem_param_grids   : {param: [values...]}
    Grid of the problem-level parameters (e.g. the perturbation rate of the
    shared Perturb operator), merged into the tuning grid of every method.
method_labels         : {internal_name: display_name}
instance_size_classes : {class_name: [instance_ids...]}
instance_features     : Callable[[instance], {name: number}]
    Numeric descriptors of an instance (size n, ...), used by the scaling
    analysis (Appendix D).
self_check            : Callable[[], None]
    Run once at start-up (see utils/param_wiring_check.py): the problem-level
    parameters really change the operators and the objective.
exact_solver          : Callable[[instance, timelimit], dict | float]   (Appendix D)
    MONO-OBJECTIVE ONLY. Called ONCE per instance; its value is the reference
    f* of the gap and of the anytime analysis.
exact_solver_timelimit : float | None
"""

from typing import Any, Callable, Dict, List, Optional

REQUIRED_KEYS = (
    "generate_instances",
    "objective_function",
    "solving_methods",
    "code",
)


class ProblemDescriptor:
    """Validated, read-only view over a `problem_dict`."""

    def __init__(self, problem_dict: Dict[str, Any]):
        d = problem_dict
        missing = [k for k in REQUIRED_KEYS if k not in d]
        if missing:
            raise ValueError(f"problem_dict missing required key(s): {missing}")

        self._d = d
        self.name: str = d.get("name", "Problem")
        self.code: str = d["code"]
        self.generate_instances: Callable = d["generate_instances"]
        self.objective_function: Callable = d["objective_function"]
        self.solving_methods: List[Callable] = list(d["solving_methods"])
        self.param_grids: Dict[str, dict] = d.get("param_grids", {}) or {}
        self.problem_param_grids: Dict[str, list] = d.get("problem_param_grids", {}) or {}
        self.self_check: Optional[Callable] = d.get("self_check", None)
        self.size_classes: Dict[str, List[int]] = d.get("instance_size_classes", {}) or {}
        self.method_labels: Dict[str, str] = d.get("method_labels", {}) or {}
        self.instance_features: Optional[Callable] = d.get("instance_features", None)

        # exact method (Appendix D)
        self.exact_solver: Optional[Callable] = d.get("exact_solver", None)
        self.exact_solver_timelimit = d.get("exact_solver_timelimit", None)

        # filled by the runner: {inst_id: f*}, {inst_id: {...}} exact-solver details,
        # {inst_id: {feature: value}}
        self.exact_reference_values: Optional[Dict[int, float]] = None
        self.exact_solver_info: Optional[Dict[int, dict]] = None
        self.instance_feature_values: Optional[Dict[int, dict]] = None

        names = d.get("objective_names")
        if not names:
            raise ValueError("problem_dict missing 'objective_names' (one name per objective)")
        self.objective_names: List[str] = list(names)
        self.n_objectives: int = len(self.objective_names)
        self.is_multiobjective: bool = self.n_objectives > 1

        if self.exact_solver is not None and self.is_multiobjective:
            raise ValueError("problem_dict provides an `exact_solver`, but the study is "
                             "multi-objective; the exact reference applies to mono-objective "
                             "studies only.")

        self._method_map = {fn.__name__: fn for fn in self.solving_methods}

    # ── metadata saved with every results pkl ───────────────────────────────
    def report_metadata(self) -> Dict[str, Any]:
        """The static, picklable description of the study saved next to the
        results of every method (see `ExperimentRunner._run_evaluation`)."""
        return {
            "name": self.name,
            "code": self.code,
            "objective_names": list(self.objective_names),
            "method_labels": dict(self.method_labels),
            "instance_size_classes": dict(self.size_classes),
            "exact_solver_timelimit": self.exact_solver_timelimit,
            "exact_reference_values": self.exact_reference_values,
            "exact_solver_info": self.exact_solver_info,
            "instance_feature_values": self.instance_feature_values,
        }

    # ── accessors ───────────────────────────────────────────────────────────
    @property
    def method_names(self) -> List[str]:
        return list(self._method_map.keys())

    def get_method(self, method_name: str) -> Callable:
        return self._method_map[method_name]

    def label(self, method_name: str) -> str:
        return self.method_labels.get(method_name, method_name)

    def get_param_grid(self, method_name: str) -> dict:
        """Tuning grid of `method_name`: the problem-level parameters
        (`problem_param_grids`, the same for every method) plus the method's own
        parameters (which win on a key collision)."""
        merged = dict(self.problem_param_grids)
        merged.update(self.param_grids.get(method_name, {}))
        return merged
