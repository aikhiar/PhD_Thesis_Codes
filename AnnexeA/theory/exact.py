"""
exact.py -- exact class-aggregated formulation of the PPMSP (reference values of P*).

Vehicles with the same power form a class k. Integer y[k, t] = number of class-k vehicles
charging at t; continuous x[i, t] in [0, 1] carries one transportation problem per class
(vehicle i supplies p_i, slot t demands y[k, t], arcs t in T_i of capacity 1). For integer
y the transportation polytope is integral (bipartite incidence matrix, totally unimodular),
so the MILP with integer y only is exact: K*T integer variables instead of sum_i |T_i|.
"""

import numpy as np

import bounds as B


def classes(inst):
    """Power classes: (list of distinct powers in units, class index of each vehicle)."""
    pw = sorted(set(int(v) for v in inst.wu), reverse=True)
    cls = np.array([pw.index(int(v)) for v in inst.wu])
    return pw, cls


def solve_aggregated(inst, env=None, limit=600, threads=0):
    """Returns (P* or best incumbent in kW, dual bound in kW, proven, schedule x or None)."""
    import gurobipy as gp
    from gurobipy import GRB
    pw, cls = classes(inst)
    K, T = len(pw), inst.T
    md = gp.Model(env=env)
    md.Params.OutputFlag = 0
    md.Params.TimeLimit = limit
    md.Params.MIPGap = 0.0
    if threads:
        md.Params.Threads = threads
    x = {(i, int(t)): md.addVar(lb=0.0, ub=1.0) for i in range(inst.n) for t in inst.win[i]}
    y = {}
    for t in range(T):
        for k in range(K):
            cap = sum(1 for i in inst.avail[t] if cls[i] == k)
            if cap:
                y[k, t] = md.addVar(lb=0, ub=cap, vtype=GRB.INTEGER)
    P = md.addVar(lb=0.0)
    md.setObjective(P, GRB.MINIMIZE)
    for i in range(inst.n):
        md.addConstr(gp.quicksum(x[i, int(t)] for t in inst.win[i]) == int(inst.p[i]))
    for (k, t), v in y.items():
        md.addConstr(gp.quicksum(x[i, t] for i in inst.avail[t] if cls[i] == k) == v)
    for t in range(T):
        ks = [k for k in range(K) if (k, t) in y]
        if ks:
            md.addConstr(gp.quicksum(pw[k] * y[k, t] for k in ks) <= P)
    md.optimize()
    if not md.SolCount:
        return np.nan, md.ObjBound / B.UNIT, False, None
    yv = {key: int(round(v.X)) for key, v in y.items()}
    sched = recover_schedule(inst, yv, cls, env)
    peak = (inst.wu[:, None] * sched).sum(axis=0).max() / B.UNIT
    return peak, md.ObjBound / B.UNIT, md.Status == GRB.OPTIMAL, sched


def recover_schedule(inst, yv, cls, env=None):
    """An integral schedule with the class counts yv: one transportation LP per class, solved
    to a vertex (integral by total unimodularity)."""
    import gurobipy as gp
    md = gp.Model(env=env)
    md.Params.OutputFlag = 0
    md.Params.Method = 0
    x = {(i, int(t)): md.addVar(lb=0.0, ub=1.0) for i in range(inst.n) for t in inst.win[i]}
    md.setObjective(0)
    for i in range(inst.n):
        md.addConstr(gp.quicksum(x[i, int(t)] for t in inst.win[i]) == int(inst.p[i]))
    for (k, t), v in yv.items():
        md.addConstr(gp.quicksum(x[i, t] for i in inst.avail[t] if cls[i] == k) == v)
    md.optimize()
    s = np.zeros((inst.n, inst.T), int)
    for (i, t), v in x.items():
        if abs(v.X - round(v.X)) > 1e-6:
            raise RuntimeError("transportation vertex not integral (should not happen)")
        s[i, t] = int(round(v.X))
    return s
