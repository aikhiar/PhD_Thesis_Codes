"""
Exact models solved with Gurobi, with anytime traces (incumbent and dual bound over time):
  solve_standard    model (1)-(6) of the paper: one binary x_{i,t} per vehicle-slot pair
  solve_aggregated  Theorem 3.3: integer class counts y_{k,t}, continuous x, exact
Both accept a starting schedule (MIP start), a valid lower bound (cut P >= LB), a time limit and
a thread count, and report Gurobi's status, incumbent, dual bound and MIP gap (Minor comment 1).
"""

import time

import numpy as np

from .instance import UNIT

STATUS = {2: "OPTIMAL", 3: "INFEASIBLE", 9: "TIME_LIMIT", 11: "INTERRUPTED", 13: "SUBOPTIMAL"}


def classes(inst):
    pw = sorted(set(int(v) for v in inst.wu), reverse=True)
    cls = np.array([pw.index(int(v)) for v in inst.wu])
    return pw, cls


def _trace_callback(trace, t0):
    """Anytime trace [(t, incumbent, bound)]; Gurobi's +-1e100 sentinels (no incumbent / no bound yet)
    are stored as NaN."""
    from gurobipy import GRB

    def clean(v):
        return v if abs(v) < 1e20 else float("nan")

    def same(a, b, tol):
        return (a != a and b != b) or abs(a - b) <= tol

    def cb(model, where):
        if where == GRB.Callback.MIP:
            inc = clean(model.cbGet(GRB.Callback.MIP_OBJBST))
            bnd = clean(model.cbGet(GRB.Callback.MIP_OBJBND))
            if not trace or not same(trace[-1][1], inc, 1e-9) or not same(trace[-1][2], bnd, 1e-6):
                trace.append((time.time() - t0, inc, bnd))
        elif where == GRB.Callback.MIPSOL:
            trace.append((time.time() - t0, clean(model.cbGet(GRB.Callback.MIPSOL_OBJ)),
                          clean(model.cbGet(GRB.Callback.MIPSOL_OBJBND))))
    return cb


def _late(deadline):
    return deadline is not None and time.time() > deadline


def _not_built(t0, model):
    """The model could not be built before the deadline (no solve; the caller keeps its incumbent)."""
    return {"status": "BUILD_TIME_LIMIT", "runtime": time.time() - t0, "bound": np.nan, "peak": np.nan,
            "gap": np.nan, "x": None, "proven": False, "trace": [], "model": model}


def _finish(md, trace, t0, x_from, improve_on=None):
    """improve_on (kW): recover the schedule only if the incumbent is below it (CRI's completion stage starts
    from its own schedule: recovering that same peak would only cost time)."""
    from gurobipy import GRB
    out = {"status": STATUS.get(md.Status, str(md.Status)), "runtime": np.nan,
           "bound": md.ObjBound / UNIT if md.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL) else np.nan,
           "peak": np.nan, "gap": np.nan, "x": None, "proven": md.Status == GRB.OPTIMAL,
           "trace": [(t, inc / UNIT, bnd / UNIT) for t, inc, bnd in trace]}
    if md.SolCount:
        out["peak"] = md.ObjVal / UNIT
        out["gap"] = md.MIPGap if md.ObjVal > 0 else 0.0
        if improve_on is None or out["peak"] < improve_on - 1e-9:
            out["x"] = x_from()
    out["runtime"] = time.time() - t0          # building + solve + recovery of the schedule (all of it is charged)
    return out


def _env(env, threads, time_limit, seed):
    import gurobipy as gp
    md = gp.Model(env=env)
    md.Params.OutputFlag = 0
    md.Params.TimeLimit = max(0.01, float(time_limit))
    md.Params.MIPGap = 0.0
    md.Params.Seed = int(seed) % 2000000000
    if threads:
        md.Params.Threads = int(threads)
    return md


def solve_standard(inst, env=None, time_limit=60.0, threads=1, start=None, lb=None, seed=0, gurobi_params=None,
                   deadline=None, improve_on=None):
    """deadline (absolute time.time()): give up while building the model when it is passed (CRI's completion
    stage); None = always build (the Gurobi baselines: their whole time is charged)."""
    import gurobipy as gp
    from gurobipy import GRB
    t0 = time.time()
    md = _env(env, threads, time_limit, seed)
    for k, v in (gurobi_params or {}).items():
        md.setParam(k, v)
    x = {}
    for i in range(inst.n):
        for t in inst.win[i]:
            x[i, int(t)] = md.addVar(vtype=GRB.BINARY)
        if i % 200 == 199 and _late(deadline):
            return _not_built(t0, "standard")
    P = md.addVar(lb=0.0 if lb is None else float(lb) * UNIT)
    md.setObjective(P, GRB.MINIMIZE)
    for i in range(inst.n):
        md.addConstr(gp.quicksum(x[i, int(t)] for t in inst.win[i]) == int(inst.p[i]))
        if i % 200 == 199 and _late(deadline):
            return _not_built(t0, "standard")
    for t in range(inst.T):
        if inst.avail[t]:
            md.addConstr(gp.quicksum(int(inst.wu[i]) * x[i, t] for i in inst.avail[t]) <= P)
    if start is not None:
        for (i, t), v in x.items():
            v.Start = float(start[i, t])
        P.Start = float(inst.loads(start).max())
    trace = []
    md.Params.TimeLimit = max(0.01, float(time_limit) - (time.time() - t0))   # budget includes model building
    md.optimize(_trace_callback(trace, t0))

    def x_from():
        s = np.zeros((inst.n, inst.T), np.int8)
        for (i, t), v in x.items():
            s[i, t] = 1 if v.X > 0.5 else 0
        return s
    out = _finish(md, trace, t0, x_from, improve_on)
    out["model"] = "standard"
    return out


def solve_aggregated(inst, env=None, time_limit=60.0, threads=1, start=None, lb=None, seed=0, gurobi_params=None,
                     deadline=None, improve_on=None):
    """deadline: as in solve_standard."""
    import gurobipy as gp
    from gurobipy import GRB
    t0 = time.time()
    pw, cls = classes(inst)
    K, T = len(pw), inst.T
    md = _env(env, threads, time_limit, seed)
    for k, v in (gurobi_params or {}).items():
        md.setParam(k, v)
    x = {}
    for i in range(inst.n):
        for t in inst.win[i]:
            x[i, int(t)] = md.addVar(lb=0.0, ub=1.0)
        if i % 200 == 199 and _late(deadline):
            return _not_built(t0, "aggregated")
    y = {}
    members = {}
    for t in range(T):
        for i in inst.avail[t]:
            members.setdefault((int(cls[i]), t), []).append(i)
    for (k, t), mem in members.items():
        y[k, t] = md.addVar(lb=0, ub=len(mem), vtype=GRB.INTEGER)
    P = md.addVar(lb=0.0 if lb is None else float(lb) * UNIT)
    md.setObjective(P, GRB.MINIMIZE)
    for i in range(inst.n):
        md.addConstr(gp.quicksum(x[i, int(t)] for t in inst.win[i]) == int(inst.p[i]))
        if i % 200 == 199 and _late(deadline):
            return _not_built(t0, "aggregated")
    for (k, t), mem in members.items():
        md.addConstr(gp.quicksum(x[i, t] for i in mem) == y[k, t])
    if _late(deadline):
        return _not_built(t0, "aggregated")
    for t in range(T):
        ks = [k for k in range(K) if (k, t) in y]
        if ks:
            md.addConstr(gp.quicksum(pw[k] * y[k, t] for k in ks) <= P)
    if start is not None:
        for (k, t), mem in members.items():
            y[k, t].Start = float(sum(int(start[i, t]) for i in mem))
        for (i, t), v in x.items():
            v.Start = float(start[i, t])
        P.Start = float(inst.loads(start).max())
    trace = []
    md.Params.TimeLimit = max(0.01, float(time_limit) - (time.time() - t0))   # budget includes model building
    md.optimize(_trace_callback(trace, t0))

    def x_from():
        yv = {key: int(round(v.X)) for key, v in y.items()}
        return recover_schedule(inst, yv, members, env)
    out = _finish(md, trace, t0, x_from, improve_on)
    out["model"] = "aggregated"
    return out


def recover_schedule(inst, yv, members, env=None):
    """Integral schedule with the class counts yv (one transportation LP solved to a vertex)."""
    import gurobipy as gp
    md = gp.Model(env=env)
    md.Params.OutputFlag = 0
    md.Params.Method = 0
    x = {(i, int(t)): md.addVar(lb=0.0, ub=1.0) for i in range(inst.n) for t in inst.win[i]}
    for i in range(inst.n):
        md.addConstr(gp.quicksum(x[i, int(t)] for t in inst.win[i]) == int(inst.p[i]))
    for (k, t), mem in members.items():
        md.addConstr(gp.quicksum(x[i, t] for i in mem) == yv[k, t])
    md.optimize()
    s = np.zeros((inst.n, inst.T), np.int8)
    for (i, t), v in x.items():
        if abs(v.X - round(v.X)) > 1e-6:
            raise RuntimeError("transportation vertex not integral (should not happen)")
        s[i, t] = int(round(v.X))
    return s
