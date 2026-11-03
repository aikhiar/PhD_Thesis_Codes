"""
model.py -- the Appendix D problem (EV charging scheduling with heterogeneous
discrete-level chargers): instances, njit operators, objective, solution forms,
independent verifier, exact method interface and test helpers.

This folder (problems/appendix_d/: parameters.py, model.py, unit_flow.py,
acn_sessions.pkl) is IDENTICAL in the two frameworks that use it:
    methods_comparison   the metaheuristics study (problems/problem_appendix_d.py, the methods_comparison binding)
    AnnexeD  the exact method against Gurobi and the certificates
Neither framework imports from the other. Change a file here, then copy the
folder to the other framework.

Problem      n bookings, m chargers of several types (each type a set of power
             set-points, e.g. {11}, {11, 22}, {11, 22, 43} kW), H slots of length
             tau. An accepted booking occupies one compatible charger during its
             whole stay S_j = {r_j, ..., d_j}; in each slot of the stay it draws
             nothing or one effective level of that charger; it receives exactly
             e_j kWh, the last charging slot possibly delivering only part of its
             level (the level stays reserved on the grid); in every slot the
             levels drawn do not exceed the grid capacity W_t.
Objective    minimise the value of the rejected bookings, sum_j v_j (1 - x_j).
Solution     float64 array of shape (n, 1 + H): sol[j, 0] = charger k + 1 serving
             booking j (0 if rejected), sol[j, 1 + t] = level (kW) drawn in slot t.
Operators    StochasticInsertion, Generate_Solution, Perturb, Crossover (all njit,
             feasibility-preserving; see problems/problem_appendix_d.py of methods_comparison for
             their description).
"""

import math
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from numba import njit
from scipy.stats import gaussian_kde

from . import parameters as P

PROBLEM_NAME = "appendix_d"
CODE = "appendix_d"
APPENDIX_D_DIR = Path(__file__).resolve().parent
TOL = 1e-9
RULES = {"max": 0, "random": 1}
_ACN = None


def _check_parameters() -> None:
    """Validates problems/appendix_d/parameters.py at import, so that a wrong value stops
    the study at once with a clear message instead of failing inside a run."""
    err = []

    def need(cond, msg):
        if not cond:
            err.append(msg)

    need(P.TAU > 0, "TAU must be positive")
    need(P.BOOKING_VALUE > 0, "BOOKING_VALUE must be positive")
    need(len(P.ACCEPTANCE_LIMITS_KW) >= 1 and all(x is None or x > 0 for x in P.ACCEPTANCE_LIMITS_KW),
         "ACCEPTANCE_LIMITS_KW: values None or positive")
    need(len(P.ACCEPTANCE_LIMIT_SHARES) == len(P.ACCEPTANCE_LIMITS_KW) and min(P.ACCEPTANCE_LIMIT_SHARES) >= 0
         and abs(sum(P.ACCEPTANCE_LIMIT_SHARES) - 1.0) < 1e-9,
         "ACCEPTANCE_LIMIT_SHARES: one non-negative share per limit, sum 1")
    need(len(P.LEVEL_SETS) >= 1 and all(len(lv) >= 1 and all(w > 0 for w in lv) for lv in P.LEVEL_SETS),
         "LEVEL_SETS: non-empty sets of positive set-points")
    need(len(P.TYPE_SHARES) == len(P.LEVEL_SETS) and min(P.TYPE_SHARES) >= 0
         and abs(sum(P.TYPE_SHARES) - 1.0) < 1e-9, "TYPE_SHARES: one non-negative share per type, sum 1")
    need(P.CHARGERS_PER_PEAK_VEHICLE > 0, "CHARGERS_PER_PEAK_VEHICLE must be positive")
    need(isinstance(P.GRID_FRACTION, dict) and set(P.GRID_FRACTION) >= {"ACN", "SIM"}
         and all(x > 0 for x in P.GRID_FRACTION.values()), "GRID_FRACTION: positive values for ACN and SIM")
    need(0 <= P.GRID_VARIATION < 1, "GRID_VARIATION in [0, 1)")
    need(P.GRID_PERIOD_H >= P.TAU, "GRID_PERIOD_H: at least one slot")
    need(len(P.SIZES) >= 1 and all(isinstance(n, int) and n >= 1 for n in P.SIZES), "SIZES: positive integers")
    need(isinstance(P.INSTANCES_PER_SIZE, int) and P.INSTANCES_PER_SIZE >= 1, "INSTANCES_PER_SIZE >= 1")
    need((APPENDIX_D_DIR / P.ACN_DATA_FILE).exists(), f"missing data file appendix_d/{P.ACN_DATA_FILE}")
    need(P.LEVEL_RULE in RULES, f"LEVEL_RULE must be one of {sorted(RULES)}")
    need(set(P.PROBLEM_PARAM_GRID) == {"pc"} and len(P.PROBLEM_PARAM_GRID["pc"]) >= 1
         and all(0 < x <= 1 for x in P.PROBLEM_PARAM_GRID["pc"]), "PROBLEM_PARAM_GRID: pc values in (0, 1]")
    need(0 < P.UNTUNED_PC <= 1, "UNTUNED_PC in (0, 1]")
    if err:
        raise ValueError("problems/appendix_d/parameters.py:\n  " + "\n  ".join(err))


_check_parameters()


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL & INSTANCE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

class AppendixDInstance:
    """One instance. Raw data (hours mapped to slots) plus the arrays read by the
    njit operators (`arrays`)."""

    def __init__(self, name, source, seed, tau, W, type_levels, charger_type,
                 r, d, e, v, limit, meta=None):
        self.name, self.source, self.seed = name, source, int(seed)
        self.tau = float(tau)
        self.W = np.asarray(W, dtype=np.float64)
        self.type_levels = [tuple(float(w) for w in lv) for lv in type_levels]
        self.charger_type = np.asarray(charger_type, dtype=np.int64)
        self.r = np.asarray(r, dtype=np.int64)
        self.d = np.asarray(d, dtype=np.int64)
        self.e = np.asarray(e, dtype=np.float64)
        self.v = np.asarray(v, dtype=np.float64)
        self.limit = np.asarray(limit, dtype=np.float64)
        self.meta = dict(meta or {})
        self.n, self.m, self.H, self.G = len(self.r), len(self.charger_type), len(self.W), len(self.type_levels)
        # effective levels L_gj = {min(w, P_j) : w in K_g} \ {0}, ascending
        K = max(len(lv) for lv in self.type_levels)
        LV = np.zeros((self.G, self.n, K))
        NL = np.zeros((self.G, self.n), dtype=np.int64)
        CT = np.zeros((self.n, self.G), dtype=np.int64)
        for g, lv in enumerate(self.type_levels):
            for j in range(self.n):
                eff = sorted({min(w, self.limit[j]) for w in lv if min(w, self.limit[j]) > 0})
                LV[g, j, :len(eff)] = eff
                NL[g, j] = len(eff)
                stay = self.d[j] - self.r[j] + 1
                if eff and self.tau * stay * eff[-1] >= self.e[j] - TOL:
                    CT[j, g] = 1                     # compatible type
        self.LV, self.NL, self.CT = LV, NL, CT
        self.arrays = (self.r, self.d, self.e, self.v, self.tau, self.W,
                       self.charger_type, self.LV, self.NL, self.CT)

    def effective_levels(self, j, k):
        g = int(self.charger_type[k])
        return [float(w) for w in self.LV[g, j, :self.NL[g, j]]]

    def compatible(self, j, k):
        return bool(self.CT[j, int(self.charger_type[k])])

    def features(self) -> Dict[str, float]:
        """Numeric descriptors of the instance (scaling analysis of methods_comparison)."""
        present = np.zeros(self.H)
        for j in range(self.n):
            present[self.r[j]:self.d[j] + 1] += 1
        busy = present > 0
        return {"n": float(self.n), "m": float(self.m), "H": float(self.H),
                "vehicles_per_charger": float(present[busy].mean() / self.m) if busy.any() else 0.0,
                "energy_to_grid": float(self.e.sum() / (self.tau * self.W[busy].sum())) if busy.any() else 0.0,
                "compatible_share": float(self.CT.any(axis=1).mean())}

    def __repr__(self):
        return f"AppendixDInstance({self.name}: n={self.n}, m={self.m}, H={self.H})"


def _load_acn() -> np.ndarray:
    """ACN-Data claimed sessions: (arrival h, departure h, energy kWh)."""
    global _ACN
    if _ACN is None:
        with open(APPENDIX_D_DIR / P.ACN_DATA_FILE, "rb") as fh:
            _ACN = np.asarray(pickle.load(fh), dtype=np.float64)
    return _ACN


def _sample_from_requests(n: int, requests: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """n distinct sessions whose arrival times follow the KDE of the ACN arrivals
    (Khiar et al., SNCS): draw arrivals from the KDE, keep the session whose
    arrival is the closest, without repetition."""
    arrivals = requests[:, 0]
    kde = gaussian_kde(arrivals)
    chosen, seen, attempts = [], set(), 0
    max_attempts = n * P.ACN_MAX_ATTEMPTS_PER_BOOKING
    while len(chosen) < n and attempts < max_attempts:
        sample = kde.resample(P.ACN_KDE_BATCH, seed=int(rng.integers(1_000_000_000)))[0]
        for i in np.argmin(np.abs(arrivals[:, None] - sample[None, :]), axis=0):
            if int(i) not in seen:
                seen.add(int(i))
                chosen.append(int(i))
                if len(chosen) == n:
                    break
        attempts += P.ACN_KDE_BATCH
    if len(chosen) < n:
        raise RuntimeError(f"could not draw {n} distinct ACN sessions")
    return np.array(chosen)


def _slots(arr_h, dep_h, tau):
    """Arrival slot: first slot starting at or after the arrival; departure slot:
    last slot ending at or before the departure (included in the stay). A vehicle
    present during no whole slot gets the one-slot stay {r} for indexing and is
    flagged in `empty` (it can never be served, see _build)."""
    r = np.ceil(np.asarray(arr_h) / tau - 1e-9).astype(np.int64)
    d = np.floor(np.asarray(dep_h) / tau + 1e-9).astype(np.int64) - 1
    return r, np.maximum(d, r), d < r


def _simulated_requests(n: int, rng: np.random.Generator):
    """Protocol of Zaidi et al. and Golabi et al. (2025, Section 6)."""
    e = rng.uniform(*P.SIM_ENERGY_RANGE, n)
    arr = rng.uniform(0.0, P.SIM_ARRIVAL_SPAN_PER_BOOKING * n, n)
    p1 = e / P.SIM_REFERENCE_POWER
    alpha = np.empty(n)
    for k, p in enumerate(p1):
        lo, hi = next((a for (plo, phi), a in P.SIM_ALPHA if plo <= p <= phi), P.SIM_ALPHA_DEFAULT)
        alpha[k] = rng.uniform(lo, hi)
    return arr, arr + (1.0 + alpha) * p1, e


def _chargers(m: int, shares):
    """Charger types in the given shares (largest remainders)."""
    raw = np.asarray(shares, dtype=float) * m
    counts = np.floor(raw).astype(int)
    for k in np.argsort(-(raw - counts))[:m - counts.sum()]:
        counts[k] += 1
    return np.repeat(np.arange(len(shares)), counts)


def _peak_presence(r, d) -> int:
    """Largest number of bookings present in one slot."""
    pres = np.zeros(int(d.max()) + 2)
    np.add.at(pres, r, 1)
    np.add.at(pres, d + 1, -1)
    return int(np.cumsum(pres).max())


def _peak_average_power(r, d, e) -> float:
    """Largest aggregate power over the slots if every booking drew its energy
    uniformly over its stay: max_t sum_{j present in t} e_j / (tau |S_j|) (kW)."""
    prof = np.zeros(int(d.max()) + 2)
    rate = np.asarray(e) / (P.TAU * (d - r + 1))
    np.add.at(prof, r, rate)
    np.add.at(prof, d + 1, -rate)
    return float(np.cumsum(prof).max())


def _servable(r, d, e, limit, levels) -> np.ndarray:
    """Bookings that some charger type can serve on its own, i.e. that have a
    compatible type: tau |S_j| max_g max L_gj >= e_j, with a non-empty level set."""
    best = np.minimum(max(max(lv) for lv in levels), limit)
    return (best > 0) & (P.TAU * (d - r + 1) * best >= np.asarray(e) - TOL)


def _acceptance_limits(n: int, seed: int) -> np.ndarray:
    """Acceptance limit P_j of each vehicle (inf = no limit), drawn among
    ACCEPTANCE_LIMITS_KW with ACCEPTANCE_LIMIT_SHARES. Its own random stream, derived
    from the seed of the instance, leaves the bookings of that seed unchanged."""
    values = np.array([np.inf if x is None else float(x) for x in P.ACCEPTANCE_LIMITS_KW])
    rng = np.random.default_rng([int(seed), 1])
    return values[rng.choice(len(values), size=n, p=np.asarray(P.ACCEPTANCE_LIMIT_SHARES, dtype=float))]


def _grid_profile(H: int, nominal: float, seed: int) -> np.ndarray:
    """W_t = nominal * (1 - GRID_VARIATION * R_h): one draw R_h ~ U(0, 1) per period of
    GRID_PERIOD_H hours (own random stream, derived from the seed of the instance)."""
    k = max(1, int(round(P.GRID_PERIOD_H / P.TAU)))
    R = np.random.default_rng([int(seed), 2]).random(math.ceil(H / k))
    return float(nominal) * (1.0 - P.GRID_VARIATION * np.repeat(R, k)[:H])


def _build(name, source, seed, n, arr, dep, e, m, levels, shares, WG, meta):
    """m None: chargers sized on the peak presence; WG None: nominal grid capacity sized
    on the peak average power, with the time-varying profile (a given WG is constant)."""
    r, d, empty = _slots(arr, dep, P.TAU)
    H = int(d.max()) + 1
    limit = _acceptance_limits(n, seed)
    # a vehicle absent during every whole slot draws no power (no effective level),
    # so that no schedule uses time during which it is absent
    limit[empty] = 0.0
    # the infrastructure is sized on the bookings it can serve: a booking without any
    # compatible type is rejected whatever the schedule, and its average power
    # e_j / (tau |S_j|), which no charger can deliver, would inflate the grid capacity
    ok = _servable(r, d, e, limit, levels)
    if not ok.any():
        ok[:] = True
    if m is None:
        m = max(1, math.ceil(P.CHARGERS_PER_PEAK_VEHICLE * _peak_presence(r[ok], d[ok])))
    ctype = _chargers(m, shares)
    if WG is None:
        WG = P.GRID_FRACTION[source] * _peak_average_power(r[ok], d[ok], np.asarray(e)[ok])
        W = _grid_profile(H, WG, seed)
    else:
        W = np.full(H, float(WG))
    return AppendixDInstance(name, source, seed, P.TAU, W, levels, ctype,
                      r, d, e, np.full(n, P.BOOKING_VALUE), limit, {**meta, "grid_nominal": float(WG)})


def acn_instance(n: int, seed: int) -> AppendixDInstance:
    rng = np.random.default_rng(seed)
    data = _load_acn()[_sample_from_requests(n, _load_acn(), rng)]
    return _build(f"ACN-n{n}-s{seed}", "ACN", seed, n, data[:, 0], data[:, 1], data[:, 2], None,
                  P.LEVEL_SETS, P.TYPE_SHARES, None, {"size": n})


def simulated_instance(n: int, seed: int) -> AppendixDInstance:
    rng = np.random.default_rng(seed)
    arr, dep, e = _simulated_requests(n, rng)
    return _build(f"SIM-n{n}-s{seed}", "SIM", seed, n, arr, dep, e, None,
                  P.LEVEL_SETS, P.TYPE_SHARES, None, {"size": n})


def generate_instances() -> List[Any]:
    """ACN instances (ids 1-20) then simulated ones (21-40), by increasing size."""
    out = []
    for maker, base in ((acn_instance, P.ACN_SEED_BASE), (simulated_instance, P.SIM_SEED_BASE)):
        k = 0
        for n in P.SIZES:
            for _ in range(P.INSTANCES_PER_SIZE):
                out.append(maker(n, base + k))
                k += 1
    return out


def instance_class(inst) -> str:
    return f"{inst.source} n={inst.n}"


def _size_classes() -> Dict[str, List[int]]:
    """Instance ids of each class, from the parameters (same order as
    generate_instances, without generating the instances)."""
    out, k = {}, 0
    for source in ("ACN", "SIM"):
        for n in P.SIZES:
            out.setdefault(f"{source} n={n}", []).extend(range(k + 1, k + 1 + P.INSTANCES_PER_SIZE))
            k += P.INSTANCES_PER_SIZE
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# NUMBA OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════

@njit
def _state(sol, A):
    """Grid load per slot and charger occupancy (1 = busy) of a schedule."""
    r, d, e, v, tau, W, ctype, LV, NL, CT = A
    n, H, m = sol.shape[0], W.shape[0], ctype.shape[0]
    load = np.zeros(H)
    busy = np.zeros((m, H), dtype=np.int8)
    for j in range(n):
        k = int(sol[j, 0]) - 1
        if k < 0:
            continue
        for t in range(r[j], d[j] + 1):
            busy[k, t] = 1
            load[t] += sol[j, 1 + t]
    return load, busy


@njit
def _remove(j, sol, load, busy, A):
    r, d = A[0], A[1]
    k = int(sol[j, 0]) - 1
    if k < 0:
        return
    for t in range(r[j], d[j] + 1):
        busy[k, t] = 0
        load[t] -= sol[j, 1 + t]
        sol[j, 1 + t] = 0.0
    sol[j, 0] = 0.0


@njit
def stochastic_insertion(j, sol, load, busy, rule, A):
    """Inserts booking j into the schedule (in place). Returns True if accepted."""
    r, d, e, v, tau, W, ctype, LV, NL, CT = A
    m = ctype.shape[0]
    cand = np.empty(m, dtype=np.int64)
    nc = 0
    for k in range(m):
        if CT[j, ctype[k]] == 1:
            cand[nc] = k
            nc += 1
    if nc == 0:
        return False
    cand = cand[:nc]
    np.random.shuffle(cand)
    L = d[j] - r[j] + 1
    lev = np.zeros(L)
    order = np.arange(L)
    for k in cand:
        free = True
        for t in range(r[j], d[j] + 1):
            if busy[k, t] != 0:
                free = False
                break
        if not free:
            continue
        g = ctype[k]
        nl = NL[g, j]
        lev[:] = 0.0
        np.random.shuffle(order)
        rem = e[j]
        for i in order:
            t = r[j] + i
            res = W[t] - load[t]
            nf = 0
            while nf < nl and LV[g, j, nf] <= res + TOL:
                nf += 1
            if nf == 0:
                continue
            if rule == 0:
                w = LV[g, j, nf - 1]
            else:
                w = LV[g, j, np.random.randint(nf)]
            lev[i] = w
            rem -= tau * w
            if rem <= TOL:
                break
        if rem > TOL:
            # top-up: raise every visited level to the largest one that fits, so that
            # the booking is accepted whenever it fits alone in the residual capacity
            for i in order:
                if rem <= TOL:
                    break
                t = r[j] + i
                res = W[t] - load[t]
                nf = 0
                while nf < nl and LV[g, j, nf] <= res + TOL:
                    nf += 1
                if nf > 0 and LV[g, j, nf - 1] > lev[i]:
                    rem -= tau * (LV[g, j, nf - 1] - lev[i])
                    lev[i] = LV[g, j, nf - 1]
        if rem > TOL:
            continue
        # chronological trim: charging stops once the energy is reached
        cum = 0.0
        last = -1
        for i in range(L):
            if lev[i] > 0.0:
                if cum >= e[j] - TOL:
                    lev[i] = 0.0
                else:
                    cum += tau * lev[i]
                    last = i
        # the last level is lowered to the smallest one completing the booking
        need = e[j] - (cum - tau * lev[last])
        for q in range(nl):
            if tau * LV[g, j, q] >= need - TOL:
                if LV[g, j, q] < lev[last]:
                    lev[last] = LV[g, j, q]
                break
        sol[j, 0] = k + 1
        for i in range(L):
            t = r[j] + i
            sol[j, 1 + t] = lev[i]
            busy[k, t] = 1
            load[t] += lev[i]
        return True
    return False


@njit
def generate_random_schedule(rule, A):
    r, d, e, v, tau, W, ctype, LV, NL, CT = A
    n, H, m = r.shape[0], W.shape[0], ctype.shape[0]
    sol = np.zeros((n, 1 + H))
    load = np.zeros(H)
    busy = np.zeros((m, H), dtype=np.int8)
    order = np.arange(n)
    np.random.shuffle(order)
    for j in order:
        stochastic_insertion(j, sol, load, busy, rule, A)
    return sol


@njit
def perturb_schedule(sol, pc, rule, A):
    n = sol.shape[0]
    new = sol.copy()
    load, busy = _state(new, A)
    q = max(1, int(math.ceil(pc * n)))
    idx = np.arange(n)
    np.random.shuffle(idx)
    sub = idx[:q].copy()
    for j in sub:
        _remove(j, new, load, busy, A)
    np.random.shuffle(sub)
    for j in sub:
        stochastic_insertion(j, new, load, busy, rule, A)
    return new


@njit
def generalized_crossover(parents, rule, A):
    r, d, e, v, tau, W, ctype, LV, NL, CT = A
    n_par, n, width = parents.shape
    H, m = W.shape[0], ctype.shape[0]
    child = np.zeros((n, width))
    load = np.zeros(H)
    busy = np.zeros((m, H), dtype=np.int8)
    unplaced = np.empty(n, dtype=np.int64)
    nu = 0
    order = np.arange(n)
    np.random.shuffle(order)
    for j in order:
        p = np.random.randint(n_par)
        k = int(parents[p, j, 0]) - 1
        ok = k >= 0
        if ok:
            for t in range(r[j], d[j] + 1):
                if busy[k, t] != 0 or load[t] + parents[p, j, 1 + t] > W[t] + TOL:
                    ok = False
                    break
        if ok:
            child[j, 0] = k + 1
            for t in range(r[j], d[j] + 1):
                child[j, 1 + t] = parents[p, j, 1 + t]
                busy[k, t] = 1
                load[t] += parents[p, j, 1 + t]
        else:
            unplaced[nu] = j
            nu += 1
    for i in range(nu):
        stochastic_insertion(unplaced[i], child, load, busy, rule, A)
    return child


@njit
def _rejected_value(sol, A):
    v = A[3]
    s = 0.0
    for j in range(sol.shape[0]):
        if sol[j, 0] < 0.5:
            s += v[j]
    return s


# ============================================================================
# SOLVER-FACING FUNCTIONS   (problem_params = (pc, rule, arrays))
# ============================================================================

@njit
def objective_function(solution, problem_params):
    return _rejected_value(solution, problem_params[2])


@njit
def Generate_Solution(problem_params):
    pc, rule, A = problem_params
    return generate_random_schedule(rule, A)


@njit
def Perturb(solution, problem_params):
    pc, rule, A = problem_params
    return perturb_schedule(solution, pc, rule, A)


@njit
def Crossover(parents_array, problem_params):
    pc, rule, A = problem_params
    return generalized_crossover(parents_array, rule, A)


def _build_problem_params(params: Dict[str, Any], inst: AppendixDInstance) -> tuple:
    pc = float(params.get("pc", P.UNTUNED_PC))
    rule = RULES[params.get("level_rule", P.LEVEL_RULE)]
    return (pc, rule, inst.arrays)


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTIONS: COMPACT FORM, OBJECTIVE, FEASIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

def compact(sol: np.ndarray) -> Dict[str, np.ndarray]:
    """Charger per booking (-1 = rejected) and the non-zero levels as triples."""
    j, t = np.nonzero(sol[:, 1:] > 0)
    return {"charger": (np.rint(sol[:, 0]).astype(np.int32) - 1),
            "j": j.astype(np.int32), "t": t.astype(np.int32),
            "level": sol[j, 1 + t].astype(np.float64)}


def expand(inst: AppendixDInstance, sol) -> np.ndarray:
    if isinstance(sol, dict):
        full = np.zeros((inst.n, 1 + inst.H))
        full[:, 0] = np.asarray(sol["charger"]) + 1
        full[np.asarray(sol["j"]), 1 + np.asarray(sol["t"])] = np.asarray(sol["level"])
        return full
    return np.asarray(sol, dtype=np.float64)


def evaluate_objective(instance: AppendixDInstance, solution: Any) -> tuple:
    """problem_dict["objective_function"]: (value of the rejected bookings,)."""
    full = expand(instance, solution)
    return (float(np.sum(instance.v[np.rint(full[:, 0]) < 1])),)


def check_solution_feasibility(inst: AppendixDInstance, sol, label="solution", eps=None):
    """Independent verifier of every constraint (plain Python, not the operators'
    code). Raises AssertionError with a description of the first violation."""
    eps = P.VERIFIER_EPS if eps is None else eps
    sol = expand(inst, sol)
    assert sol.shape == (inst.n, 1 + inst.H), f"[{label}] SHAPE FAIL {sol.shape}"
    load = np.zeros(inst.H)
    on_charger = {}
    for j in range(inst.n):
        k = int(round(sol[j, 0])) - 1
        row = sol[j, 1:]
        if k < 0:
            assert not np.any(row > eps), f"[{label}] REJECTED-POWER FAIL booking {j}"
            continue
        assert k < inst.m, f"[{label}] CHARGER FAIL booking {j} on charger {k}"
        assert inst.compatible(j, k), f"[{label}] COMPATIBILITY FAIL booking {j}, charger {k}"
        slots = np.nonzero(row > eps)[0]
        assert slots.size > 0, f"[{label}] NO-CHARGE FAIL accepted booking {j}"
        assert slots.min() >= inst.r[j] and slots.max() <= inst.d[j], \
            f"[{label}] STAY FAIL booking {j} charges outside {inst.r[j]}..{inst.d[j]}"
        allowed = np.asarray(inst.effective_levels(j, k))
        for t in slots:
            assert np.any(np.abs(allowed - row[t]) <= eps), \
                f"[{label}] LEVEL FAIL booking {j} slot {t}: {row[t]} not in {allowed.tolist()}"
        total = inst.tau * row[slots].sum()
        before_last = total - inst.tau * row[slots.max()]
        assert total >= inst.e[j] - eps, f"[{label}] ENERGY FAIL booking {j}: {total} < {inst.e[j]}"
        assert before_last <= inst.e[j] + eps, \
            f"[{label}] LAST-SLOT FAIL booking {j} completes before its last charging slot"
        load += row
        on_charger.setdefault(k, []).append((int(inst.r[j]), int(inst.d[j]), j))
    for k, iv in on_charger.items():
        iv.sort()
        for (r1, d1, j1), (r2, d2, j2) in zip(iv, iv[1:]):
            assert r2 > d1, f"[{label}] OCCUPANCY FAIL charger {k}: bookings {j1}, {j2} overlap"
    over = np.nonzero(load > inst.W + eps)[0]
    assert over.size == 0, f"[{label}] GRID FAIL slot {over[:1].tolist()}: {load[over[:1]]} > {inst.W[over[:1]]}"


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION SHAPE AND STORED FORM
# ═══════════════════════════════════════════════════════════════════════════════

def solution_shape(instance: AppendixDInstance) -> tuple:
    return (instance.n, 1 + instance.H)


def finalize_solution(instance: AppendixDInstance, solution: np.ndarray) -> Dict[str, np.ndarray]:
    """The best schedule of a run, as stored in the results: checked by the
    independent verifier (a schedule it rejects makes the run a crashed run,
    never a wrong value) and kept in compact form."""
    check_solution_feasibility(instance, solution, label="best solution")
    return compact(solution)


# ═══════════════════════════════════════════════════════════════════════════════
# EXACT METHOD: THE UNIT-FLOW FORMULATION (problems/appendix_d/unit_flow.py)
# ═══════════════════════════════════════════════════════════════════════════════

def exact_available() -> bool:
    """The exact method can be used: enabled in the parameters, its module
    (unit_flow.py, in this folder) is present, and gurobipy is installed."""
    import importlib.util
    return bool(P.EXACT_SOLVER) and (APPENDIX_D_DIR / "unit_flow.py").exists() \
        and importlib.util.find_spec("gurobipy") is not None


def _unit_flow():
    from . import unit_flow
    return unit_flow


def exact_solver(instance: AppendixDInstance, timelimit=None) -> Dict[str, Any]:
    """problem_dict["exact_solver"]: the instance solved by the unit-flow exact method.
    Returns the value of the rejected bookings of its schedule (the optimum when the status
    is "optimal"), the lower bound, the status, the nodes and the time. The schedule is
    checked by the independent verifier before its value is used."""
    r = _unit_flow().solve(instance, float(timelimit) if timelimit else P.EXACT_TIME_LIMIT)
    if r["solution"] is None:
        raise RuntimeError(f"no schedule found by the exact method ({r['status']})")
    check_solution_feasibility(instance, r["solution"], label="exact method")
    rejected = evaluate_objective(instance, r["solution"])[0]
    total = float(instance.v.sum())
    assert abs((total - rejected) - r["accepted"]) < 1e-6, "exact method: value differs from its schedule"
    return {"best_objective": rejected, "bound": total - float(r.get("bound_floor", r["bound"])),
            "status": r["status"], "nodes": r["nodes"], "time": r["time"]}


@njit
def _nb_seed(seed):
    np.random.seed(seed)


def seed_all(seed: int) -> None:
    """Seeds Python's random, NumPy and Numba (the operators draw from Numba's
    generator), as methods_comparison's seed_everything does."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    _nb_seed(seed)
