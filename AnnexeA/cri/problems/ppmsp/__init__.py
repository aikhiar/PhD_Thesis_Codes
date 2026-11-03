"""PPMSP plug-in: everything the problem-agnostic runner needs to know about the problem.

A problem plug-in provides
  name, sense ("min" | "max"), objective_name
  build_set(spec, base_seed) -> list of instances (each with .name, .family, .meta, .features())
  make_env(cfg)               -> solver environment shared by a worker (or None)
  make_context(inst, env, cfg)-> per-instance cache of costly artifacts (lower bounds, relaxations)
  evaluate(inst, x)           -> (objective, feasible)
  reference(inst, ctx, env, cfg) -> exact reference: {"opt", "bound", "proven", "time", "status"}
  instance_to_dict / instance_from_dict for storage
"""

from ...core.registry import register_problem
from . import context as _context
from . import generators as _generators
from . import methods as _methods  # noqa: F401  (registers the methods)
from . import sota as _sota  # noqa: F401  (registers the state-of-the-art baselines)
from .instance import UNIT, Instance


@register_problem("ppmsp")
class PPMSP:
    name = "ppmsp"
    sense = "min"
    objective_name = "peak power (kW)"
    resolution = 1.0 / UNIT

    @staticmethod
    def build_set(spec, base_seed=0):
        return _generators.build_set(spec, base_seed)

    @staticmethod
    def families():
        return sorted(_generators.FAMILIES)

    @staticmethod
    def make_env(cfg):
        try:
            import gurobipy as gp
        except ImportError:
            return None
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 1 if cfg.get("gurobi_output", False) else 0)
        # every model built in this process (LP relaxation, cover-cut and configuration LPs, iterative
        # rounding, schedule recovery, W-LNS subproblems) uses one thread unless a method sets more (the
        # -4t Gurobi variants): Gurobi's default would use all cores in every worker at once
        env.setParam("Threads", int(cfg.get("gurobi_threads", 1)))
        env.start()
        return env

    @staticmethod
    def warmup(env):
        """Load every Numba kernel and open Gurobi once in a fresh worker, so that no measured run pays
        for compilation or cache loading."""
        import numpy as np
        from ...core.registry import make_method
        inst = _generators.FAMILIES["staggered"](8, 12, np.random.default_rng(0))
        specs = [{"class": "G"}, {"class": "RR", "params": {"N": 50}}, {"class": "RLP", "params": {"trials": 2}},
                 {"class": "SA", "params": {"N": 300}}, {"class": "IR"}, {"class": "BK"},
                 {"class": "MT", "params": {"matrix": "lp"}}, {"class": "LLS"}, {"class": "WLNS"},
                 {"class": "NFDH"}, {"class": "ZAIDI_H"}, {"class": "ZAIDI_ILS", "params": {"max_trials": 50}}]
        specs += [{"class": "HUNSAKER", "params": {"alpha": 2.0}}, {"class": "AFSHAR_BB"}]
        specs += [{"class": "AN_HYBRID", "params": {"variant": v, "ls": ls, "pop": 6, "maxgen": 2}}
                  for v in ("DE", "GA", "PSO") for ls in ("LS1", "LS2")]
        specs += [{"class": "CE", "params": {"matrix": "lp", "estimator": e}} for e in ("excess", "union", "component")]
        # the cut and configuration matrices build the cover-cut LP and the configuration LP (knapsack pricing)
        specs += [{"class": "CE", "params": {"matrix": m, "estimator": "union"}} for m in ("cut", "conf")]
        # two passes: in the first, compilation times are charged to the methods (as the time of the
        # artifacts they use) and can exhaust their 0.2 s before their main loops are reached (LLS never got
        # to lls_iterate); the second pass, on a fresh context of compiled artifacts, reaches them
        for _ in range(2):
            ctx = _context.make_context(inst, env, {"conf_time_limit": 5})
            for sp in specs:
                try:
                    make_method(sp, defaults={}).run(inst, ctx, None if sp["class"] == "AN_HYBRID" else 0.2, 0)
                except Exception:
                    pass

    @staticmethod
    def uses_budget(spec, defaults=None):
        """True if the method normally runs until its budget (for the time estimate of a plan)."""
        prm = dict((defaults or {}).get(spec["class"], {}))
        prm.update(spec.get("params", {}) or {})
        c = spec["class"]
        if c in ("GUROBI", "LLS", "WLNS", "CRI"):
            return True
        if c == "ZAIDI_ILS":                 # the printed Algorithm 5 rarely ends before a budget of seconds
            return bool(prm.get("restart", False)) or prm.get("iter_count", "accepted") == "accepted"
        if c in ("AN_HYBRID", "AFSHAR_BB"):
            return True
        if c == "G":
            return prm.get("mode") == "budget"
        if c in ("RR", "RLP", "SA"):
            return prm.get("stop") == "budget"
        if c == "MT":
            return prm.get("mode") == "budget"
        return False

    @staticmethod
    def make_context(inst, env, cfg):
        return _context.make_context(inst, env, cfg)

    @staticmethod
    def evaluate(inst, x):
        if x is None:
            return float("nan"), False
        return inst.peak(x), bool(inst.is_feasible(x))

    @staticmethod
    def reference(inst, ctx, env, cfg):
        """Exact reference value (aggregated model, Theorem 3.3), warm-started by the certified lower
        bound; used to measure gaps to the optimum on the instances where it is known."""
        from . import exact as EX
        lb = ctx.get("lb")
        f = EX.solve_aggregated if cfg.get("model", "aggregated") == "aggregated" else EX.solve_standard
        r = f(inst, env, float(cfg.get("time_limit", 600)), int(cfg.get("threads", 1)), lb=lb, seed=0)
        return {"opt": r["peak"] if r["proven"] else float("nan"), "best": r["peak"], "bound": max(lb, r["bound"]),
                "proven": bool(r["proven"]), "time": r["runtime"], "status": r["status"]}

    @staticmethod
    def instance_to_dict(inst):
        return inst.to_dict()

    @staticmethod
    def instance_from_dict(d):
        return Instance.from_dict(d)
