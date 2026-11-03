"""
covering.py -- the covering form (C) of formulation (P), a second baseline for the
unit-flow method.

(C) is (P) after Propositions "covering suffices" and "assignment to chargers": one binary
per booking and type, one binary y_jtw per booking, slot and level, and

    sum_w y_jtw <= x_j,     y_jtw <= sum of the types that offer w,
    tau * sum_{t,w} w y_jtw >= e_j x_j        (covering),
    sum_{j,w} w y_jtw <= W_t                  (grid),

without the energy variables q, the last-slot variables u and their constraints. It is the
strongest time-indexed formulation with one binary per level: comparing (UF) with (C)
isolates the effect of the units from the effect of dropping the exact-energy machinery.
"""
import math
import time

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = GRB = None

from problems.appendix_d.unit_flow import EPS, FEAS, STATUS, Units, _assign_chargers


def solve(inst, time_limit: float = 300.0, threads: int = 0, mip_gap: float = 0.0, seed: int = 0) -> dict:
    if gp is None:
        raise ImportError("gurobipy is not installed")
    t0 = time.perf_counter()
    U = Units(inst)
    J = U.J
    mdl = gp.Model("covering")
    mdl.Params.OutputFlag = 0
    mdl.Params.TimeLimit = float(time_limit)
    mdl.Params.Threads = int(threads)
    mdl.Params.MIPGap = float(mip_gap)
    mdl.Params.Seed = int(seed)
    stay = {j: range(int(inst.r[j]), int(inst.d[j]) + 1) for j in J}
    lam = {j: sorted({w for g in U.types[j] for w in U.levels[g, j]}) for j in J}
    x = {j: mdl.addVar(vtype=GRB.BINARY, obj=float(inst.v[j])) for j in J}
    b = {(g, j): mdl.addVar(vtype=GRB.BINARY) for j in J for g in U.types[j]}
    y = {(j, t, w): mdl.addVar(vtype=GRB.BINARY) for j in J for t in stay[j] for w in lam[j]
         if w <= inst.W[t] + FEAS}
    mdl.ModelSense = GRB.MAXIMIZE
    mdl.update()
    y_jt, y_t = {}, {}
    for (j, t, w), var in y.items():
        y_jt.setdefault((j, t), []).append((w, var))
        y_t.setdefault(t, []).append((w, var))
    for j in J:
        mdl.addConstr(gp.quicksum(b[g, j] for g in U.types[j]) == x[j])
    arrivals = sorted({int(inst.r[j]) for j in J})
    for g in range(inst.G):
        users = [j for j in J if (g, j) in b]
        for r in arrivals:
            present = [j for j in users if inst.r[j] <= r <= inst.d[j]]
            if len(present) > U.cap[g]:
                mdl.addConstr(gp.quicksum(b[g, j] for j in present) <= U.cap[g])
    for (j, t), lst in y_jt.items():
        mdl.addConstr(gp.quicksum(v for _, v in lst) <= x[j])
        for w, v in lst:
            allowed = [g for g in U.types[j] if any(abs(w - s) < EPS for s in U.levels[g, j])]
            if len(allowed) < len(U.types[j]):
                mdl.addConstr(v <= gp.quicksum(b[g, j] for g in allowed))
    for j in J:
        mdl.addConstr(gp.quicksum(w * v for t in stay[j] for w, v in y_jt.get((j, t), []))
                      >= (U.E[j] - FEAS) * x[j])
    for t, lst in y_t.items():
        mdl.addConstr(gp.quicksum(w * v for w, v in lst) <= inst.W[t] + FEAS)
    build_time = time.perf_counter() - t0
    mdl.optimize()
    has = mdl.SolCount > 0
    out = {"status": STATUS.get(mdl.Status, f"status_{mdl.Status}"), "accepted": float(mdl.ObjVal) if has else None,
           "bound": float(mdl.ObjBound), "build_time": build_time, "solver_time": float(mdl.Runtime),
           "nodes": float(mdl.NodeCount), "n_vars": int(mdl.NumVars), "n_bin": int(mdl.NumBinVars),
           "n_int": int(mdl.NumIntVars), "n_constrs": int(mdl.NumConstrs), "solution": None, "method": "covering"}
    if np.all(inst.v == np.round(inst.v)):
        out["bound_floor"] = math.floor(out["bound"] + 1e-6)
    if has:
        accepted = [j for j in J if x[j].X > 0.5]
        gtype = {j: next(g for g in U.types[j] if b[g, j].X > 0.5) for j in accepted}
        plan = {}
        for j in accepted:
            plan.setdefault(gtype[j], []).append(j)
        charger = _assign_chargers(inst, plan, U.chargers_of_type)
        sol = np.zeros((inst.n, 1 + inst.H))
        for j in accepted:
            sol[j, 0] = charger[j] + 1
            cum, last = 0.0, None
            for t in stay[j]:                                  # chronological truncation
                w = next((w for w, v in y_jt.get((j, t), []) if v.X > 0.5), 0.0)
                if w <= 0 or cum >= U.E[j] - FEAS:
                    continue
                sol[j, 1 + t] = w
                cum += w
                last = t
            before = cum - sol[j, 1 + last]
            sol[j, 1 + last] = next(w for w in sorted(U.levels[gtype[j], j]) if before + w >= U.E[j] - FEAS)
        out["solution"] = sol
    mdl.dispose()
    out["time"] = time.perf_counter() - t0
    return out
