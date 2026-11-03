"""
unit_flow.py -- exact method for the Appendix D problem: the unit-flow reformulation.

Idea
----
The time-indexed MILP (P) of the paper has one binary per booking, slot and level.
Most of them are not needed. Take a base step lam (the smallest effective level, 11 kW
in the instances) and write every level as

    w = a * lam - delta,      a = ceil(w / lam),   0 <= delta < lam.

* Regular levels of a booking on a type: lam, 2 lam, ..., kappa lam (a prefix of the
  multiples of lam; delta = 0). Drawing one of them in slot t is an integer number of
  units f_jt in {0, ..., kappa}.
* Special levels: every other level (43 = 4 * 11 - 1 in the instances). Drawing the
  special level w in slot t is a binary h_jtw; it counts for a units, with a deficit
  delta in the energy delivered AND in the grid power used.

With u_jt = f_jt + sum_w a_w h_jtw the units drawn by booking j in slot t:

    energy   lam * sum_t u_jt - sum_{t,w} delta_w h_jtw >= E_j      (E_j = e_j / tau)
       <=>   sum_t u_jt >= D_j,   D_j integer,  lam D_j >= E_j + sum_{t,w} delta_w h_jtw
    grid     lam * sum_j u_jt - sum_{j,w} delta_w h_jtw <= W_t
       <=>   sum_j u_jt <= C_t,   C_t integer,  lam C_t <= W_t + sum_{j,w} delta_w h_jtw

so that a special level raises by delta / lam the unit demand of its booking and the unit
capacity of its slot. Once the binaries (acceptance x, type b, special levels h) and the
integers (D, C) are fixed, the units f form a transportation problem with integral data,
whose polytope is integral: f is a CONTINUOUS variable of the model, and an integral f is
recovered afterwards by one small flow problem. Without special levels the model is the
flow model of Theorem "dispatch" (ii); the special levels are the only per-slot binaries.

By Proposition "covering suffices", covering the energy is enough: the exact-energy
schedule is obtained by the chronological truncation; by Proposition "assignment to
chargers", one binary per booking and type is enough: chargers are assigned by interval
colouring.

The model is exact for ANY level sets: a level that is not a regular multiple of lam is
simply special. It is small when most levels are regular.
"""
import heapq
import math
import time

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:                                     # reported by solve()
    gp = GRB = None

EPS = 1e-9          # comparison of levels
FEAS = 1e-7         # tolerance on energies (kW-slots) and powers (kW), below the verifier's


# ══════════════════════════════════════════════════════════════════════════════
# LEVELS: REGULAR UNITS AND SPECIAL LEVELS
# ══════════════════════════════════════════════════════════════════════════════

class Units:
    """Decomposition of the effective level sets of an instance on a base step lam."""

    def __init__(self, inst, lam=None):
        self.inst = inst
        chargers_of_type = {g: [k for k in range(inst.m) if inst.charger_type[k] == g] for g in range(inst.G)}
        self.chargers_of_type = chargers_of_type
        self.cap = {g: len(chargers_of_type[g]) for g in range(inst.G)}
        # compatible types that have at least one charger
        self.types = {j: [g for g in range(inst.G) if inst.CT[j, g] == 1 and self.cap[g] > 0]
                      for j in range(inst.n)}
        self.J = [j for j in range(inst.n) if self.types[j]]
        lv = {(g, j): [float(w) for w in inst.LV[g, j, :inst.NL[g, j]]] for j in self.J for g in self.types[j]}
        self.levels = lv
        self.lam = float(lam) if lam is not None else (min(min(v) for v in lv.values()) if lv else 1.0)
        self.kappa, self.special = {}, {}
        for key, levels in lv.items():
            k = 0
            while any(abs(w - (k + 1) * self.lam) < EPS for w in levels):
                k += 1
            self.kappa[key] = k
            self.special[key] = [w for w in levels if w > k * self.lam + EPS]
        self.kmax = {j: max(self.kappa[g, j] for g in self.types[j]) for j in self.J}
        self.spec = {j: sorted({w for g in self.types[j] for w in self.special[g, j]}) for j in self.J}
        self.E = {j: float(inst.e[j]) / inst.tau for j in self.J}          # energy in kW-slots

    def a(self, w: float) -> int:
        """Units of the special level w."""
        return int(math.ceil(w / self.lam - EPS))

    def delta(self, w: float) -> float:
        """Deficit of the special level w: a * lam - w, in [0, lam)."""
        return self.a(w) * self.lam - w


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build(inst, env=None, integer_units: bool = False, specials: bool = True, tight: bool = True):
    """Returns (gurobipy model, variables dict, Units). specials = False builds the
    restriction to the regular levels (no special level: a model without any binary
    indexed by time, whose solutions are feasible for the complete model)."""
    # tight = True uses the rounded forms of the constraints on the extra units and on the
    # bonus units (same integer solutions, stronger linear relaxation), see below
    U = Units(inst)
    lam, J = U.lam, U.J
    mdl = gp.Model("unit_flow", env=env) if env is not None else gp.Model("unit_flow")
    stay = {j: range(int(inst.r[j]), int(inst.d[j]) + 1) for j in J}
    x = {j: mdl.addVar(vtype=GRB.BINARY, obj=float(inst.v[j]), name=f"x_{j}") for j in J}
    b = {(g, j): mdl.addVar(vtype=GRB.BINARY, name=f"b_{g}_{j}") for j in J for g in U.types[j]}
    ftype = GRB.INTEGER if integer_units else GRB.CONTINUOUS
    f, h = {}, {}
    for j in J:
        for t in stay[j]:
            if U.kmax[j] >= 1 and inst.W[t] >= lam - FEAS:
                f[j, t] = mdl.addVar(lb=0.0, ub=U.kmax[j], vtype=ftype, name=f"f_{j}_{t}")
            for w in (U.spec[j] if specials else ()):
                if inst.W[t] >= w - FEAS:
                    h[j, t, w] = mdl.addVar(vtype=GRB.BINARY, name=f"h_{j}_{t}_{w:g}")
    mdl.ModelSense = GRB.MAXIMIZE
    mdl.update()
    hs_j = {j: [] for j in J}                         # special variables of each booking
    hs_t = {}                                         # ... and of each slot
    hs_t_j = {}                                       # bookings with a special variable in each slot
    hs_jt = {}
    for (j, t, w), var in h.items():
        hs_j[j].append((w, var))
        hs_t.setdefault(t, []).append((w, var))
        hs_t_j.setdefault(t, set()).add(j)
        hs_jt.setdefault((j, t), []).append((w, var))
    # (1) one type per accepted booking
    for j in J:
        mdl.addConstr(gp.quicksum(b[g, j] for g in U.types[j]) == x[j], name=f"assign_{j}")
    # (2) occupancy per type and arrival slot (Proposition "assignment to chargers")
    arrivals = sorted({int(inst.r[j]) for j in J})
    for g in range(inst.G):
        users = [j for j in J if (g, j) in b]
        for r in arrivals:
            present = [j for j in users if inst.r[j] <= r <= inst.d[j]]
            if len(present) > U.cap[g]:
                mdl.addConstr(gp.quicksum(b[g, j] for j in present) <= U.cap[g], name=f"occ_{g}_{r}")
    # (3) levels allowed by the type, one level per slot
    for (j, t), var in f.items():
        mdl.addConstr(var <= gp.quicksum(U.kappa[g, j] * b[g, j] for g in U.types[j]), name=f"lev_{j}_{t}")
    for (j, t, w), var in h.items():
        allowed = [g for g in U.types[j] if any(abs(w - s) < EPS for s in U.special[g, j])]
        mdl.addConstr(var <= gp.quicksum(b[g, j] for g in allowed), name=f"spe_{j}_{t}_{w:g}")
    for (j, t), lst in hs_jt.items():
        if (j, t) in f:
            mdl.addConstr(f[j, t] + U.kmax[j] * gp.quicksum(v for _, v in lst) <= U.kmax[j], name=f"one_{j}_{t}")
        if len(lst) > 1:
            mdl.addConstr(gp.quicksum(v for _, v in lst) <= x[j], name=f"ones_{j}_{t}")
    # (4) unit demand of each booking: sum_t u_jt >= D_j = q_j x_j + extra_j
    extra = {}
    for j in J:
        q = int(math.ceil((U.E[j] - FEAS) / lam - EPS))
        units = (gp.quicksum(f[j, t] for t in stay[j] if (j, t) in f)
                 + gp.quicksum(U.a(w) * v for w, v in hs_j[j]))
        if not hs_j[j]:
            mdl.addConstr(units >= q * x[j], name=f"units_{j}")
            continue
        slack = lam * q - (U.E[j] - FEAS)                     # deficit absorbed without extra unit
        slack = 0.0 if slack < 10 * FEAS else slack
        dmax = max(U.delta(w) for w, _ in hs_j[j])
        deficit = gp.quicksum(U.delta(w) * v for w, v in hs_j[j])
        nmax = len({t for (jj, t) in hs_jt if jj == j})       # special levels drawn by j: one per slot
        if tight:
            # the truncated schedule of a booking (Proposition "covering suffices") delivers less
            # than E_j before its last charging slot: at most ceil(E_j / w) special levels
            wmin = min(w for w, _ in hs_j[j])
            nmax = min(nmax, int(math.ceil((U.E[j] - FEAS) / wmin - EPS)))
            mdl.addConstr(gp.quicksum(v for _, v in hs_j[j]) <= nmax * x[j], name=f"nspec_{j}")
        ub = int(math.ceil(max(0.0, dmax * nmax - slack) / lam - EPS))
        if ub == 0:                                           # the deficits never cost a unit
            mdl.addConstr(units >= q * x[j], name=f"units_{j}")
            continue
        extra[j] = mdl.addVar(lb=0, ub=ub, vtype=GRB.INTEGER, name=f"extra_{j}")
        if tight and ub == 1:
            # extra_j = 0 forces deficit <= slack: stronger than lam extra_j >= deficit - slack
            mdl.addConstr(deficit - slack * x[j] <= (dmax * nmax - slack) * extra[j], name=f"dem_{j}")
        else:
            mdl.addConstr(lam * extra[j] >= deficit - slack * x[j], name=f"dem_{j}")
        mdl.addConstr(extra[j] <= ub * x[j], name=f"demx_{j}")
        mdl.addConstr(units >= q * x[j] + extra[j], name=f"units_{j}")
    # (5) unit capacity of each slot: sum_j u_jt <= C_t = C0_t + bonus_t
    bonus, C0 = {}, {}
    special_types = {g for (g, j), lst in U.special.items() if lst}
    special_chargers = sum(U.cap[g] for g in special_types)
    for t in range(inst.H):
        users = [j for j in J if (j, t) in f]
        lst = hs_t.get(t, [])
        if not users and not lst:
            continue
        C0[t] = int(math.floor((inst.W[t] + FEAS) / lam + EPS))
        units = gp.quicksum(f[j, t] for j in users) + gp.quicksum(U.a(w) * v for w, v in lst)
        if lst:
            rho = max(0.0, inst.W[t] + FEAS - lam * C0[t])    # power that whole units cannot use
            rho = 0.0 if rho < 10 * FEAS else rho
            wmin = min(w for w, _ in lst)
            deltas = sorted({round(U.delta(w), 9) for w, _ in lst})
            # special levels drawn in the slot: at most one per booking and per charger that has
            # one, and they must fit the grid
            k = min(len(hs_t_j[t]), special_chargers, int(math.floor((inst.W[t] + FEAS) / wmin)))
            ub = int(math.floor((rho + k * deltas[-1]) / lam + EPS))
            if ub >= 1:
                bonus[t] = mdl.addVar(lb=0, ub=ub, vtype=GRB.INTEGER, name=f"bonus_{t}")
                deficit = gp.quicksum(U.delta(w) * v for w, v in lst)
                if tight and ub == 1 and len(deltas) == 1 and deltas[0] > EPS:
                    # one bonus unit needs g special levels, g = ceil((lam - rho) / delta)
                    g = int(math.ceil((lam - rho) / deltas[0] - EPS))
                    mdl.addConstr(g * bonus[t] <= gp.quicksum(v for _, v in lst), name=f"bon_{t}")
                elif tight and ub == 1:
                    mdl.addConstr((lam - rho) * bonus[t] <= deficit, name=f"bon_{t}")
                else:
                    mdl.addConstr(lam * bonus[t] <= rho + deficit, name=f"bon_{t}")
                mdl.addConstr(units <= C0[t] + bonus[t], name=f"grid_{t}")
                continue
        mdl.addConstr(units <= C0[t], name=f"grid_{t}")
    V = {"x": x, "b": b, "f": f, "h": h, "extra": extra, "bonus": bonus, "C0": C0, "stay": stay}
    return mdl, V, U


# ══════════════════════════════════════════════════════════════════════════════
# FROM A SOLUTION OF THE MODEL TO A SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════

def _assign_chargers(inst, plan_types, chargers_of_type):
    """Greedy interval colouring per type (Proposition "assignment to chargers")."""
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


def integral_units(inst, U, V, accepted, gtype, hval, env=None):
    """Integral units f for fixed acceptance, types and special levels: a transportation
    problem with integral data (one small LP whose vertices are integral)."""
    lam = U.lam
    sub = gp.Model("units", env=env) if env is not None else gp.Model("units")
    sub.Params.OutputFlag = 0
    spec_jt, a_j, d_j, a_t, d_t = {}, {}, {}, {}, {}
    for (j, t, w) in hval:
        spec_jt[j, t] = w
        a_j[j] = a_j.get(j, 0) + U.a(w)
        d_j[j] = d_j.get(j, 0.0) + U.delta(w)
        a_t[t] = a_t.get(t, 0) + U.a(w)
        d_t[t] = d_t.get(t, 0.0) + U.delta(w)
    f = {}
    for j in accepted:
        k = U.kappa[gtype[j], j]
        for t in V["stay"][j]:
            if (j, t) in V["f"] and (j, t) not in spec_jt and k >= 1:
                f[j, t] = sub.addVar(lb=0, ub=k, vtype=GRB.INTEGER, obj=1.0)
    sub.update()
    for j in accepted:
        need = int(math.ceil((U.E[j] - FEAS + d_j.get(j, 0.0)) / lam - EPS)) - a_j.get(j, 0)
        if need > 0:
            sub.addConstr(gp.quicksum(f[j, t] for t in V["stay"][j] if (j, t) in f) >= need)
    by_slot = {}
    for (j, t), var in f.items():
        by_slot.setdefault(t, []).append(var)
    for t in set(by_slot) | set(a_t):
        cap = int(math.floor((inst.W[t] + FEAS + d_t.get(t, 0.0)) / lam + EPS)) - a_t.get(t, 0)
        if cap < 0:
            raise RuntimeError(f"slot {t}: special levels exceed the grid")
        if t in by_slot:
            sub.addConstr(gp.quicksum(by_slot[t]) <= cap)
    sub.optimize()
    if sub.Status != GRB.OPTIMAL:
        raise RuntimeError(f"no integral units for the fixed binaries (status {sub.Status})")
    out = {key: int(round(var.X)) for key, var in f.items()}
    sub.dispose()
    return out, spec_jt


def schedule(inst, U, V, xval, bval, hval, env=None) -> np.ndarray:
    """Schedule array (n, 1 + H): charger + 1 (0 = rejected), then the level of each slot,
    with the chronological truncation of Proposition "covering suffices"."""
    accepted = [j for j in U.J if xval[j] > 0.5]
    gtype = {j: next(g for g in U.types[j] if bval[g, j] > 0.5) for j in accepted}
    units, spec_jt = integral_units(inst, U, V, accepted, gtype, hval, env)
    plan = {}
    for j in accepted:
        plan.setdefault(gtype[j], []).append(j)
    charger = _assign_chargers(inst, plan, U.chargers_of_type)
    sol = np.zeros((inst.n, 1 + inst.H))
    for j in accepted:
        sol[j, 0] = charger[j] + 1
        levels = sorted(U.levels[gtype[j], j])
        cum, last = 0.0, None
        for t in V["stay"][j]:
            w = spec_jt.get((j, t), U.lam * units.get((j, t), 0))
            if w <= 0 or cum >= U.E[j] - FEAS:                 # nothing after completion
                continue
            sol[j, 1 + t] = w
            cum += w
            last = t
        if last is None or cum < U.E[j] - FEAS:
            raise RuntimeError(f"booking {j}: energy not covered")
        # last charging slot: the smallest level that completes the booking
        before = cum - sol[j, 1 + last]
        sol[j, 1 + last] = next(w for w in levels if before + w >= U.E[j] - FEAS)
    return sol


def contains(inst, sol, tight: bool = True) -> bool:
    """True if the schedule `sol` (array (n, 1 + H), feasible for (P)) is a solution of
    (UF): the variables x, b, h and f are fixed to the values read off the schedule, and the
    model must remain feasible. This is the direction "from a plan to (UF)" of the exactness
    theorem, checked numerically: (UF) cuts off no feasible schedule. With tight = True the
    schedule must stop charging once its energy is reached (truncated schedule), as the
    schedules of the operators and of every method do."""
    mdl, V, U = build(inst, tight=tight)
    mdl.Params.OutputFlag = 0
    fixed_h = set()
    for j in range(inst.n):
        k = int(round(sol[j, 0])) - 1
        if j not in V["x"]:
            if k >= 0:
                return False                                   # a booking without compatible type
            continue
        g_j = int(inst.charger_type[k]) if k >= 0 else None
        if k >= 0 and g_j not in U.types[j]:
            return False
        V["x"][j].LB = V["x"][j].UB = 1.0 if k >= 0 else 0.0
        for g in U.types[j]:
            V["b"][g, j].LB = V["b"][g, j].UB = 1.0 if g == g_j else 0.0
        if k < 0:
            continue
        for t in V["stay"][j]:
            w = float(sol[j, 1 + t])
            units = int(round(w / U.lam))
            if abs(w - units * U.lam) < EPS and units <= U.kappa[g_j, j]:       # zero or a regular level
                if (j, t) in V["f"]:
                    V["f"][j, t].LB = V["f"][j, t].UB = float(units)
                elif units > 0:
                    return False
                continue
            key = next((key for key in ((j, t, s) for s in U.spec[j]) if abs(key[2] - w) < EPS), None)
            if key is None or key not in V["h"]:
                return False                                   # not an effective level, or above the grid
            V["h"][key].LB = V["h"][key].UB = 1.0
            fixed_h.add(key)
    for key, var in V["h"].items():
        if key not in fixed_h:
            var.UB = 0.0
    mdl.optimize()
    ok = mdl.Status == GRB.OPTIMAL
    mdl.dispose()
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# SOLVE
# ══════════════════════════════════════════════════════════════════════════════

def _configure(mdl, time_limit, threads, mip_gap, seed, verbose, log_file, params):
    mdl.Params.OutputFlag = 1 if (verbose or log_file) else 0
    mdl.Params.LogToConsole = 1 if verbose else 0
    if log_file:
        mdl.Params.LogFile = log_file
    mdl.Params.TimeLimit = max(1.0, float(time_limit))
    mdl.Params.Threads = int(threads)
    mdl.Params.MIPGap = float(mip_gap)
    mdl.Params.Seed = int(seed)
    for key, value in (params or {}).items():
        mdl.setParam(key, value)


STATUS = {2: "optimal", 9: "time_limit", 3: "infeasible", 11: "interrupted"}     # Gurobi status codes


def solve(inst, time_limit: float = 300.0, threads: int = 0, mip_gap: float = 0.0, seed: int = 0,
          log_file: str = None, verbose: bool = False, integer_units: bool = False,
          warm_start: bool = False, stage1_share: float = 0.2, params: dict = None,
          tight: bool = True) -> dict:
    """Solves the unit-flow model (UF) and returns status, accepted value, bound, times,
    sizes, the trajectory [(time, incumbent, bound)] and the schedule of the incumbent.

    warm_start: two stages. Stage 1 solves the restriction of (UF) to the regular levels,
    which has no binary variable indexed by time and whose solutions are feasible for
    (UF); its optimal plan is the MIP start of stage 2, the complete model. The optimality
    proof is the one of stage 2 alone. Stage 1 gets at most `stage1_share` of the time
    limit and is skipped when the instance has no special level (it is then the model).
    threads = 0: all the cores (Gurobi default). params: further Gurobi parameters.
    tight = False: the plain rounding constraints (for tests and ablation)."""
    if gp is None:
        raise ImportError("gurobipy is not installed")
    t0 = time.perf_counter()
    mdl, V, U = build(inst, integer_units=integer_units, tight=tight)
    build_time = time.perf_counter() - t0
    traj, stage1 = [], None
    if warm_start and V["h"]:
        m1, V1, _ = build(inst, integer_units=integer_units, specials=False, tight=tight)
        _configure(m1, stage1_share * time_limit, threads, mip_gap, seed, False, None, params)
        m1.optimize()
        stage1 = {"status": STATUS.get(m1.Status, f"status_{m1.Status}"), "time": float(m1.Runtime),
                  "accepted": float(m1.ObjVal) if m1.SolCount > 0 else None}
        if m1.SolCount > 0:
            for key in ("x", "b", "f"):
                for k_, var in V1[key].items():
                    V[key][k_].Start = var.X
            for group in ("h", "extra", "bonus"):
                for var in V[group].values():
                    var.Start = 0.0
            traj.append((time.perf_counter() - t0, stage1["accepted"], None))
        m1.dispose()
    offset = time.perf_counter() - t0
    _configure(mdl, time_limit - offset, threads, mip_gap, seed, verbose, log_file, params)

    def cb(model_, where):
        if where in (GRB.Callback.MIP, GRB.Callback.MIPSOL):
            if where == GRB.Callback.MIP:
                inc, bnd = model_.cbGet(GRB.Callback.MIP_OBJBST), model_.cbGet(GRB.Callback.MIP_OBJBND)
            else:
                inc, bnd = model_.cbGet(GRB.Callback.MIPSOL_OBJBST), model_.cbGet(GRB.Callback.MIPSOL_OBJBND)
            inc = None if abs(inc) >= GRB.INFINITY else float(inc)
            bnd = None if abs(bnd) >= GRB.INFINITY else float(bnd)
            if not traj or (inc, bnd) != tuple(traj[-1][1:]):
                traj.append((offset + float(model_.cbGet(GRB.Callback.RUNTIME)), inc, bnd))

    mdl.optimize(cb)
    status = STATUS.get(mdl.Status, f"status_{mdl.Status}")
    has = mdl.SolCount > 0
    out = {"status": status, "accepted": float(mdl.ObjVal) if has else None,
           "bound": float(mdl.ObjBound), "build_time": build_time,
           "solver_time": float(mdl.Runtime) + (stage1["time"] if stage1 else 0.0),
           "stage1": stage1, "stage2_time": float(mdl.Runtime),
           "nodes": float(mdl.NodeCount), "n_vars": int(mdl.NumVars), "n_bin": int(mdl.NumBinVars),
           "n_int": int(mdl.NumIntVars), "n_constrs": int(mdl.NumConstrs), "n_nonzeros": int(mdl.NumNZs),
           "lam": U.lam, "n_special": len(V["h"]), "solution": None, "trajectory": traj,
           "method": "unit_flow", "threads": threads, "time_limit": time_limit}
    if np.all(inst.v == np.round(inst.v)):
        out["bound_floor"] = math.floor(out["bound"] + 1e-6)
    if has:
        xval = {j: v.X for j, v in V["x"].items()}
        bval = {k_: v.X for k_, v in V["b"].items()}
        hval = [k_ for k_, v in V["h"].items() if v.X > 0.5]
        out["solution"] = schedule(inst, U, V, xval, bval, hval)
        out["specials_used"] = len(hval)
        out["bonus_units"] = int(round(sum(v.X for v in V["bonus"].values())))
    mdl.dispose()
    out["time"] = time.perf_counter() - t0          # everything: both models, both stages, the schedule
    traj.append((out["time"], out["accepted"], out["bound"]))
    return out
