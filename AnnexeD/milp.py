"""
milp.py -- the MILP (P) of the paper (Section 2.2), written with PuLP and solved by
Gurobi through PuLP's GUROBI interface, as in our previous studies.

    max  sum_j v_j x_j
    (C1) assignment             sum_i a_ij = x_j
    (C2) whole-stay occupancy   sum_{j in J_r} a_ij <= 1           for every charger i, arrival slot r
    (C3) discrete levels        sum_w y_jtw <= x_j ;  y_jtw <= sum_{i: w in L_ij} a_ij ;  p_jt = sum_w w y_jtw
    (C4) exact energy           sum_t q_jt = e_j x_j
    (C5) energy follows level   tau p_jt - tau Pbar_j u_jt <= q_jt <= tau p_jt ; sum_t u_jt = x_j ;
                                no level after the last charging slot
    (C6) grid capacity          sum_j p_jt <= W_t

MODEL = "per_type" solves the equivalent form of Proposition "Assignment to chargers"
(Section 3.1): one binary per (booking, compatible type) and occupancy counts per type
and arrival slot, which removes the symmetry between identical chargers; the individual
chargers are then recovered by the greedy interval colouring of that proposition.
MODEL = "per_charger" solves (P) exactly as written in Section 2.2.
The constraint "no level after the last charging slot" uses the running sum
c_js = sum_{t<s} u_jt (c_{j,r_j} = 0, c_{j,s+1} = c_js + u_js), an equivalent linear form
with O(|S_j|) instead of O(|S_j|^2) non-zeros. The power p_jt is the linear expression
sum_w w y_jtw, substituted in the constraints.

The solver is pulp.GUROBI (which drives Gurobi through gurobipy): the incumbent and the
upper bound are recorded over time by a Gurobi callback passed through PuLP, and the
statistics of the search (bound, gap, nodes, size) are read from prob.solverModel.
"""
import heapq
import math
import time

import numpy as np

try:
    from pulp import LpBinary, LpMaximize, LpProblem, LpVariable, lpSum, GUROBI
except ImportError:                                  # reported by build()
    LpProblem = None
try:
    from gurobipy import GRB
except ImportError:                                  # reported by solve()
    GRB = None


def _types_of(inst, j):
    return [g for g in range(inst.G) if inst.CT[j, g] == 1]


def _levels(inst, g, j):
    return [float(w) for w in inst.LV[g, j, :inst.NL[g, j]]]


def build(inst, model: str = "per_type"):
    """Returns (PuLP problem, variables dict)."""
    if LpProblem is None:
        raise ImportError("PuLP is not installed (python -m pip install pulp)")
    prob = LpProblem("".join(c if c.isalnum() else "_" for c in inst.name), LpMaximize)
    tau = inst.tau
    chargers_of_type = {g: [k for k in range(inst.m) if inst.charger_type[k] == g] for g in range(inst.G)}
    # bookings with no compatible charger on site (incompatible types, or compatible types
    # without any charger) are rejected: x_j = 0
    J = [j for j in range(inst.n) if any(chargers_of_type[g] for g in _types_of(inst, j))]
    x = {j: LpVariable(f"x_{j}", cat=LpBinary) for j in J}
    # assignment variables: per type (b_gj) or per charger (a_ij)
    if model == "per_type":
        asg = {(g, j): LpVariable(f"b_{g}_{j}", cat=LpBinary)
               for j in J for g in _types_of(inst, j) if chargers_of_type[g]}
        cap = {g: len(chargers_of_type[g]) for g in range(inst.G)}
        unit_type = {g: g for g in range(inst.G)}
    elif model == "per_charger":
        asg = {(k, j): LpVariable(f"a_{k}_{j}", cat=LpBinary)
               for j in J for k in range(inst.m) if inst.CT[j, inst.charger_type[k]] == 1}
        cap = {k: 1 for k in range(inst.m)}
        unit_type = {k: int(inst.charger_type[k]) for k in range(inst.m)}
    else:
        raise ValueError(f"MODEL must be 'per_type' or 'per_charger', not {model!r}")
    units_of = {j: [] for j in J}
    for (u, j) in asg:
        units_of[j].append(u)
    lam = {j: sorted({w for u in units_of[j] for w in _levels(inst, unit_type[u], j)}) for j in J}
    y, q, uu, c = {}, {}, {}, {}
    for j in J:
        for t in range(inst.r[j], inst.d[j] + 1):
            for li, w in enumerate(lam[j]):
                y[j, t, w] = LpVariable(f"y_{j}_{t}_{li}", cat=LpBinary)
            q[j, t] = LpVariable(f"q_{j}_{t}", lowBound=0)
            uu[j, t] = LpVariable(f"u_{j}_{t}", cat=LpBinary)
            c[j, t] = LpVariable(f"c_{j}_{t}", lowBound=0, upBound=1)
    p = {(j, t): lpSum(w * y[j, t, w] for w in lam[j])
         for j in J for t in range(inst.r[j], inst.d[j] + 1)}
    # objective
    prob += lpSum(float(inst.v[j]) * x[j] for j in J), "accepted_value"
    # (C1)
    for j in J:
        prob += lpSum(asg[u, j] for u in units_of[j]) == x[j], f"C1_{j}"
    # (C2) per unit (charger or type) and arrival slot
    arrivals = sorted({int(inst.r[j]) for j in J})
    for u_, capacity in cap.items():
        users = [j for j in J if (u_, j) in asg]
        for r in arrivals:
            present = [j for j in users if inst.r[j] <= r <= inst.d[j]]
            if len(present) > capacity:
                prob += lpSum(asg[u_, j] for j in present) <= capacity, f"C2_{u_}_{r}"
    # (C3), (C4), (C5)
    for j in J:
        stay = range(inst.r[j], inst.d[j] + 1)
        pbar = max(lam[j])
        for t in stay:
            prob += lpSum(y[j, t, w] for w in lam[j]) <= x[j], f"C3a_{j}_{t}"
            for li, w in enumerate(lam[j]):
                allowed = [u_ for u_ in units_of[j] if w in _levels(inst, unit_type[u_], j)]
                if len(allowed) < len(units_of[j]):
                    prob += y[j, t, w] <= lpSum(asg[u_, j] for u_ in allowed), f"C3b_{j}_{t}_{li}"
            prob += q[j, t] <= tau * p[j, t], f"C5a_{j}_{t}"
            prob += q[j, t] >= tau * p[j, t] - tau * pbar * uu[j, t], f"C5b_{j}_{t}"
            # running sum of the last-slot indicators before t, and no level after the last slot
            if t == inst.r[j]:
                prob += c[j, t] == 0, f"C5c_{j}_{t}"
            else:
                prob += c[j, t] == c[j, t - 1] + uu[j, t - 1], f"C5c_{j}_{t}"
            prob += lpSum(y[j, t, w] for w in lam[j]) + c[j, t] <= 1, f"C5d_{j}_{t}"
        prob += lpSum(q[j, t] for t in stay) == float(inst.e[j]) * x[j], f"C4_{j}"
        prob += lpSum(uu[j, t] for t in stay) == x[j], f"C5e_{j}"
    # (C6)
    for t in range(inst.H):
        terms = [p[j, t] for j in J if inst.r[j] <= t <= inst.d[j]]
        if terms:
            prob += lpSum(terms) <= float(inst.W[t]), f"C6_{t}"
    return prob, {"J": J, "x": x, "asg": asg, "y": y, "lam": lam, "model": model,
                  "chargers_of_type": chargers_of_type, "unit_type": unit_type}


def _assign_chargers(inst, plan_types, chargers_of_type):
    """Greedy interval colouring per type (Proposition 'Assignment to chargers')."""
    out = {}
    for g, js in plan_types.items():
        free = list(chargers_of_type[g])
        heapq.heapify(free)
        busy = []
        for j in sorted(js, key=lambda j: (inst.r[j], j)):
            while busy and busy[0][0] < inst.r[j]:
                heapq.heappush(free, heapq.heappop(busy)[1])
            if not free:
                raise RuntimeError(f"type {g} over capacity at slot {inst.r[j]}")
            k = heapq.heappop(free)
            out[j] = k
            heapq.heappush(busy, (int(inst.d[j]), k))
    return out


def _val(var) -> float:
    v = var.varValue
    return 0.0 if v is None else float(v)


def extract(inst, V) -> np.ndarray:
    """Schedule array (n, 1 + H) of the incumbent: charger + 1 (0 rejected), levels."""
    sol = np.zeros((inst.n, 1 + inst.H))
    accepted = [j for j in V["J"] if _val(V["x"][j]) > 0.5]
    units = {j: next(u for (u, jj), var in V["asg"].items() if jj == j and _val(var) > 0.5) for j in accepted}
    if V["model"] == "per_type":
        plan = {}
        for j in accepted:
            plan.setdefault(units[j], []).append(j)
        charger = _assign_chargers(inst, plan, V["chargers_of_type"])
    else:
        charger = units
    for j in accepted:
        sol[j, 0] = charger[j] + 1
        for t in range(inst.r[j], inst.d[j] + 1):
            for w in V["lam"][j]:
                if _val(V["y"][j, t, w]) > 0.5:
                    sol[j, 1 + t] = w
    return sol


def solve(inst, time_limit: float, threads: int = 1, model: str = "per_type", mip_gap=None,
          seed: int = 0, log_file: str = None) -> dict:
    """Solves the MILP with pulp.GUROBI; returns status, incumbent, bound, gap, time, nodes,
    model size, the trajectory [(time, incumbent, bound)] and the schedule of the incumbent."""
    if GRB is None:
        raise ImportError("gurobipy is not installed: PuLP's GUROBI interface needs it "
                          "(install the gurobipy of the same version as the Gurobi installation)")
    t0 = time.perf_counter()
    prob, V = build(inst, model)
    build_time = time.perf_counter() - t0
    params = {"Threads": int(threads), "Seed": int(seed)}
    if mip_gap is not None:
        params["MIPGap"] = float(mip_gap)
    if log_file:
        # msg=False would set OutputFlag = 0, which silences the log file as well: the
        # full solver log goes to log_file, nothing to the console
        solver = GUROBI(msg=True, timeLimit=float(time_limit), logPath=log_file,
                        LogToConsole=0, **params)
    else:
        solver = GUROBI(msg=False, timeLimit=float(time_limit), **params)
    traj = []

    def cb(model_, where):
        if where in (GRB.Callback.MIP, GRB.Callback.MIPSOL):
            if where == GRB.Callback.MIP:
                inc, bnd = model_.cbGet(GRB.Callback.MIP_OBJBST), model_.cbGet(GRB.Callback.MIP_OBJBND)
            else:
                inc, bnd = model_.cbGet(GRB.Callback.MIPSOL_OBJBST), model_.cbGet(GRB.Callback.MIPSOL_OBJBND)
            rt = model_.cbGet(GRB.Callback.RUNTIME)
            inc = None if inc is None or abs(inc) >= GRB.INFINITY else float(inc)
            bnd = None if bnd is None or abs(bnd) >= GRB.INFINITY else float(bnd)
            if not traj or (inc, bnd) != tuple(traj[-1][1:]):
                traj.append((float(rt), inc, bnd))

    prob.solve(solver, callback=cb)
    wall = time.perf_counter() - t0
    m = prob.solverModel                               # the gurobipy model built by PuLP
    status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit", GRB.INFEASIBLE: "infeasible",
              GRB.INTERRUPTED: "interrupted"}.get(m.Status, f"status_{m.Status}")
    has = m.SolCount > 0

    def attr(name, default=None):
        try:
            return float(getattr(m, name))
        except Exception:                              # e.g. a model without integer variables
            return default

    obj = attr("ObjVal") if has else None
    bound = attr("ObjBound", obj if status == "optimal" else None)
    gap = attr("MIPGap", 0.0 if status == "optimal" else None)
    # final point: incumbents found in presolve or at the end of the search are not
    # always reported to the callback
    final = (float(m.Runtime), obj, bound)
    if not traj or tuple(traj[-1][1:]) != final[1:]:
        traj.append(final)
    out = {"status": status, "accepted": obj, "bound": bound,
           "gap": gap * 100 if (has and gap is not None) else None, "time": float(m.Runtime),
           "build_time": build_time, "wall_time": wall, "nodes": attr("NodeCount", 0.0),
           "n_vars": int(m.NumVars), "n_bin": int(m.NumIntVars), "n_constrs": int(m.NumConstrs),
           "n_nonzeros": int(m.NumNZs), "trajectory": traj, "model": model, "threads": threads,
           "time_limit": time_limit, "interface": "pulp.GUROBI",
           "solution": extract(inst, V) if has else None}
    if has and out["bound"] is not None and np.all(inst.v == np.round(inst.v)):
        out["bound_floor"] = math.floor(out["bound"] + 1e-6)     # integer values: integer bound
    solver.close()
    return out
