"""
Baselines from the literature (Section 6.6 of the paper).

Implemented (Numba kernels, like the methods of the paper)
  BK          Burgess & Killebrew (1962) sum-of-squares leveling, the classical resource-leveling
              heuristic, adapted to preemption: every vehicle starts at its earliest slots (ASAP);
              vehicles are visited as in the original procedure, from the last activity backwards
              (latest start first), and each one moves to its best position within its float (one
              charged slot to a free slot of the window whenever this strictly lowers sum_t L_t^2,
              until none does: the p least-loaded slots, i.e. the sum-of-squares best response); passes
              repeat until no move applies. Deterministic, no parameters.
  NFDH        Next-Fit Decreasing Height, preemptive version (Ranjan, Khargonekar & Sahni 2014,
              Sec. V): vehicles by non-increasing power (Coffman et al. 1980) are packed on stacked
              levels, next fit along the time axis; a vehicle that does not fit on the current level
              keeps what fits there and the remainder starts a new level. Ranjan et al. assume a
              common window; on a common window this code is exactly their NFDH_Preemptive. On
              general windows it is the natural adaptation (earliest free slots of the vehicle's
              window on the current level); info["common_window"] says which case applied.
  ZAIDI_H     the constructive heuristic of Zaidi, Oulamara, Idoumghar & Basset (EJOR 2023/2024,
              "Minimizing grid capacity in preemptive electric vehicle charging orchestration",
              Algorithms 3-4) for a fixed charger assignment: vehicles by non-decreasing departure,
              ties by non-increasing energy then non-decreasing arrival; lower bound (17); each vehicle
              takes, in chronological order, the slots where it does not exceed max(lb, wG), then the
              least-loaded slots of its window.
  ZAIDI_ILS   their iterated local search (Algorithm 5) with simulated annealing as local search
              (Algorithm 6: peak-to-off-peak moves of one vehicle, q in {1..p} moves, Lundy-Mees
              cooling as in Connolly 1990, T0 = mu f(S0)); parameters of their Table 4. Adaptation:
              in the PPMSP the charger assignment is fixed, so the perturbation cannot change a
              vehicle's charger type; it selects a vehicle with their rules (random, or roulette wheel
              on the number of its charging slots with load >= wG/2) and re-places it on random slots
              of its window. Stops when the perturbation level reaches its maximum (as published) or
              at the budget; restart=True restarts from the best solution until the budget is used.
              The counter iter of their Algorithm 5 grows when a worse solution is accepted (line 14,
              as printed); iter_count="non_improving" follows their prose instead. The threshold
              max(wG, lb) of Algorithm 4 is applied exactly on integer loads (floor of lb).

  HUNSAKER    the optimal online alpha-policy of Hunsaker, Kleywegt, Savelsbergh & Tovey (2003) for
              minimax resource scheduling: level r(t) = alpha_T v_T(t) (v_T: best offline value given the
              work known at t, their Proposition 4.4; alpha_T from their LP, Corollary 4.13, checked against
              their Table 4.1), earliest-deadline-first service. Made integral for the PPMSP: vehicles that
              must charge to meet their departure charge first, then EDF while the load stays <= r(t).
              Online information: a vehicle is known from its arrival slot only.
  AFSHAR_BB   the depth-first branch and bound of Afshar-Nadjafi, Khalaj & Mehdizadeh (2013) for
              preemptive resource leveling (node = vehicles charging in the current slot, branches = all
              subsets, rules (a) remaining duration vs remaining window, (c) bound, (d) dominance as a
              transposition table), with the peak as objective; exact when it ends within the budget.
  AN_HYBRID   the hybrid metaheuristics of Aristotelous & Nearchou (2024), Table 1 settings (population
              100, 5000 generations, DE CR 0.6 F 0.5, GA SBX 0.6 + polynomial mutation 0.01, PSO w 1,
              c1 2, c2 2.5): SER / PAR / DE / GA / PSO with leveling scheme LS1 or LS2 on the best member
              at every generation; random keys decoded by least-loaded placement; fitness = peak (ties
              by DEV) or DEV (as published). LS1/LS2 follow their pseudo-code (one random extension or
              delay r in [0, TS] per non-critical vehicle, kept if DEV decreases), in preemptive form.
              DE/best/1/bin with distinct indices and strict acceptance, GA without elitism (as described),
              PSO with random initial velocities (their Eq. (10)); velocities and keys are clamped to keep
              the keys in [0, 1] (w = 1 is not stable otherwise).
(Hartmann & Briskorn 2022 is a survey; it is used to position the problem, not as a method.)

How to integrate a method
  Subclass SotaMethod and implement
      solve(self, inst, budget, seed, deadline) -> x            (0/1 array of shape (n, T))
  or  solve(...) -> (x, trace)                                   (trace = [(seconds, peak kW)])
  using inst.s, inst.e (0-based inclusive windows), inst.p, inst.w (kW) or inst.wu (integer units),
  and deadline.expired() / deadline.remaining() for anytime methods. The base class times the run,
  checks feasibility and builds the result record. Give the defaults of its parameters in the
  `defaults:` section of the configuration file and add it to the experiments (block E10).
"""

import time

import numpy as np
from numba import njit

from ...core.registry import register_method
from ...core.results import Deadline
from . import kernels as K
from .instance import UNIT
from .methods import Method, _result


class SotaMethod(Method):
    """Common wrapper: timing, feasibility check, result record."""
    stochastic = True
    reference = ""            # bibliographic key of the original paper
    implemented = True

    def __init__(self, **params):
        self.params = params

    def solve(self, inst, budget, seed, deadline):
        raise NotImplementedError

    def prepare(self, inst):
        """Constants of the method that do not depend on the instance data beyond its size (computed once,
        before the clock starts), e.g. the competitive-ratio parameter of HUNSAKER."""

    def run(self, inst, ctx, budget, seed, lower_bound=None):
        if not self.implemented:
            return {"status": "not_available", "info": {"reason": "to be integrated", "reference": self.reference}}
        self.prepare(inst)
        t0 = time.time()
        dl = Deadline(budget)
        self._info = {}
        out = self.solve(inst, budget, seed, dl)
        x, trace = out if isinstance(out, tuple) else (out, None)
        x = np.asarray(x, np.int8)
        t_own = time.time() - t0
        pk = inst.peak(x)
        if trace is None:
            trace = [(t_own, pk)]
        info = {"reference": self.reference}
        info.update(self._info)
        return _result(pk, x, trace, t_own, 0.0, feasible=bool(inst.is_feasible(x)), info=info)


@register_method("BK")
class BurgessKillebrew(SotaMethod):
    """Burgess-Killebrew sum-of-squares leveling (1962), adapted to preemptive unit moves."""
    stochastic = False
    reference = "Burgess1962"

    def __init__(self, order="latest_start", **params):
        super().__init__(**params)
        self.order = order

    def solve(self, inst, budget, seed, deadline):
        x = np.zeros((inst.n, inst.T), np.int8)
        for i in range(inst.n):
            x[i, inst.s[i]:inst.s[i] + inst.p[i]] = 1
        L = K.loads_of(x, inst.wu)
        if self.order == "decreasing_power":
            order = np.argsort(-inst.wu, kind="stable").astype(np.int64)
        else:                                   # latest-starting first, as in the original method
            order = np.lexsort((-inst.e, -inst.s)).astype(np.int64)
        K._descent(x, L, inst.s, inst.e, inst.wu, order)
        return x


# ── NFDH (Ranjan et al. 2014) ───────────────────────────────────────────────────

@njit(cache=True)
def nfdh_kernel(s, e, p, order, T):
    n = s.shape[0]
    x = np.zeros((n, T), np.int8)
    occ = np.zeros(T, np.bool_)
    levels = 1
    for oi in range(n):
        i = order[oi]
        need = p[i]
        for t in range(s[i], e[i] + 1):                 # current level, next fit (earliest free slots)
            if need == 0:
                break
            if not occ[t]:
                occ[t] = True
                x[i, t] = 1
                need -= 1
        while need > 0:                                 # the remainder starts a new level
            occ[:] = False
            levels += 1
            for t in range(s[i], e[i] + 1):
                if need == 0:
                    break
                if x[i, t] == 0:
                    occ[t] = True
                    x[i, t] = 1
                    need -= 1
    return x, levels


@register_method("NFDH")
class NFDH(SotaMethod):
    """Next-Fit Decreasing Height, preemptive (Ranjan et al. 2014); window-aware on general windows."""
    stochastic = False
    reference = "Ranjan2014"

    def solve(self, inst, budget, seed, deadline):
        order = np.lexsort((-inst.p, -inst.wu)).astype(np.int64)    # non-increasing power, then duration
        x, levels = nfdh_kernel(inst.s, inst.e, inst.p, order, inst.T)
        self._info = {"levels": int(levels),
                      "common_window": bool((inst.s == inst.s[0]).all() and (inst.e == inst.e[0]).all())}
        return x


# ── Zaidi et al. (EJOR): heuristic and ILS ──────────────────────────────────────

def zaidi_lb(inst):
    """Lower bound (17) for a fixed assignment, in units (real): max(total energy / availability span, w_max)."""
    span = int(inst.e.max()) + 1 - int(inst.s.min())
    return max(float((inst.wu * inst.p).sum()) / span, float(inst.wu.max()))


def zaidi_cap(inst):
    """Algorithm 4 keeps a slot in H1 when w_j + w_t^G <= max(w^G, lb); loads are integers (units), so the
    exact integer threshold is floor(lb)."""
    return int(np.floor(zaidi_lb(inst) + 1e-9))


def zaidi_order(inst):
    """Non-decreasing departure; ties: non-increasing energy, then non-decreasing arrival."""
    return np.lexsort((inst.s, -(inst.wu * inst.p), inst.e)).astype(np.int64)


@njit(cache=True)
def _zaidi_alloc(i, x, L, s, e, p, wu, cap):
    """Algorithm 4 for vehicle i (row i of x empty): slots t with w_i + L_t <= cap in chronological
    order, then the other slots of the window by non-decreasing load. Returns the new max load on i's slots."""
    need = p[i]
    m = e[i] - s[i] + 1
    rest = np.empty(m, np.int64)
    loads = np.empty(m, np.int64)
    k = 0
    for t in range(s[i], e[i] + 1):
        if need > 0 and wu[i] + L[t] <= cap:
            x[i, t] = 1
            need -= 1
        else:
            rest[k] = t
            loads[k] = L[t]
            k += 1
    if need > 0:
        idx = np.argsort(loads[:k], kind="mergesort")
        for j in range(need):
            x[i, rest[idx[j]]] = 1
    mx = 0
    for t in range(s[i], e[i] + 1):
        if x[i, t]:
            L[t] += wu[i]
            if L[t] > mx:
                mx = L[t]
    return mx


@njit(cache=True)
def zaidi_heuristic_kernel(s, e, p, wu, T, order, lb):
    n = s.shape[0]
    x = np.zeros((n, T), np.int8)
    L = np.zeros(T, np.int64)
    wG = 0
    for oi in range(n):
        i = order[oi]
        mx = _zaidi_alloc(i, x, L, s, e, p, wu, max(lb, wG))
        if mx > wG:
            wG = mx
    return x, L


@njit(cache=True)
def _peak(L):
    m = L[0]
    for t in range(1, L.shape[0]):
        if L[t] > m:
            m = L[t]
    return m


@njit(cache=True)
def _sa_neighbor(x, L, s, e, p, wu, flex, mv_i, mv_a, mv_b):
    """One SA neighbour (Algorithm 6): a random flexible vehicle j makes q in {1..p_j} moves of one
    charged slot from its most loaded charging slot to its least loaded free slot. Records the moves."""
    j = flex[np.random.randint(flex.shape[0])]
    q = 1 + np.random.randint(p[j])
    nm = 0
    for _ in range(q):
        t1 = -1; t2 = -1; v1 = -1; v2 = 1 << 62
        for t in range(s[j], e[j] + 1):
            if x[j, t]:
                if L[t] > v1:
                    v1 = L[t]; t1 = t
            else:
                if L[t] < v2:
                    v2 = L[t]; t2 = t
        if t1 < 0 or t2 < 0:
            break
        x[j, t1] = 0; x[j, t2] = 1
        L[t1] -= wu[j]; L[t2] += wu[j]
        mv_i[nm] = j; mv_a[nm] = t1; mv_b[nm] = t2
        nm += 1
    return nm


@njit(cache=True)
def _revert(x, L, wu, mv_i, mv_a, mv_b, nm):
    for k in range(nm - 1, -1, -1):
        j, a, b = mv_i[k], mv_a[k], mv_b[k]
        x[j, b] = 0; x[j, a] = 1
        L[b] -= wu[j]; L[a] += wu[j]


@njit(cache=True)
def zaidi_sa(x, L, s, e, p, wu, flex, unit, mu, max_gen, max_trials, acc_ratio, Tf):
    """Algorithm 6 (objective: peak in kW). Works in place on (x, L); returns the best (x, L) found."""
    best_x = x.copy(); best_L = L.copy()
    f = _peak(L) / unit
    fb = f
    T0 = mu * f if mu * f > Tf else Tf * 10.0
    temp = T0
    M = max(1.0, max_trials / max_gen)
    b = (T0 - Tf) / (M * T0 * Tf)
    max_acc = max(1, int(acc_ratio * max_gen))
    pmax = 0
    for i in range(flex.shape[0]):
        if p[flex[i]] > pmax:
            pmax = p[flex[i]]
    mv_i = np.empty(pmax, np.int64); mv_a = np.empty(pmax, np.int64); mv_b = np.empty(pmax, np.int64)
    trial = 0
    while True:
        accepted = 0
        generated = 0
        while generated < max_gen and accepted < max_acc:
            nm = _sa_neighbor(x, L, s, e, p, wu, flex, mv_i, mv_a, mv_b)
            f2 = _peak(L) / unit
            d = f2 - f
            generated += 1
            trial += 1
            if d < 0 or np.random.random() <= np.exp(-d / temp):
                f = f2
                accepted += 1
                if f < fb - 1e-12:
                    fb = f
                    best_x[:, :] = x
                    best_L[:] = L
            else:
                _revert(x, L, wu, mv_i, mv_a, mv_b, nm)
        temp = temp / (1.0 + b * temp)
        if trial >= max_trials or accepted == 0:
            break
    return best_x, best_L


@njit(cache=True)
def zaidi_perturb(x, L, s, e, p, wu, k, roulette_prob):
    """k perturbations: select a vehicle (uniformly, or by roulette wheel on its number of charging
    slots with load >= wG/2) and re-place it on p_i random slots of its window."""
    n = s.shape[0]
    a = np.zeros(n, np.float64)
    for _ in range(k):
        if np.random.random() < roulette_prob:
            thr = _peak(L) / 2.0
            tot = 0.0
            for i in range(n):
                c = 0
                for t in range(s[i], e[i] + 1):
                    if x[i, t] and L[t] >= thr:
                        c += 1
                a[i] = c
                tot += c
            u = np.random.random() * tot
            j = n - 1
            acc = 0.0
            for i in range(n):
                acc += a[i]
                if acc >= u and a[i] > 0:
                    j = i
                    break
        else:
            j = np.random.randint(n)
        for t in range(s[j], e[j] + 1):
            if x[j, t]:
                x[j, t] = 0
                L[t] -= wu[j]
        m = e[j] - s[j] + 1
        perm = np.random.permutation(m)
        for r in range(p[j]):
            t = s[j] + perm[r]
            x[j, t] = 1
            L[t] += wu[j]


@register_method("ZAIDI_H")
class ZaidiHeuristic(SotaMethod):
    """Constructive heuristic of Zaidi et al. (EJOR), Algorithms 3-4, fixed charger assignment."""
    stochastic = False
    reference = "Zaidi2024a"

    def solve(self, inst, budget, seed, deadline):
        x, L = zaidi_heuristic_kernel(inst.s, inst.e, inst.p, inst.wu, inst.T, zaidi_order(inst), zaidi_cap(inst))
        self._info = {"lb_17": zaidi_lb(inst) / UNIT}
        return x


@register_method("ZAIDI_ILS")
class ZaidiILS(SotaMethod):
    """Iterated local search of Zaidi et al. (EJOR), Algorithms 5-6, parameters of their Table 4."""
    reference = "Zaidi2024a"

    def __init__(self, pert0=3, pertmax=20, itermax=20, r=0.75, p0=0.5, mu=0.1, max_generated=50,
                 max_trials=500, acceptance_ratio=0.5, Tf=0.001, roulette_prob=0.5, restart=False,
                 iter_count="accepted", **params):
        super().__init__(**params)
        self.pert0, self.pertmax, self.itermax = int(pert0), int(pertmax), int(itermax)
        self.r, self.p0, self.mu = float(r), float(p0), float(mu)
        self.max_generated, self.max_trials = int(max_generated), int(max_trials)
        self.acceptance_ratio, self.Tf = float(acceptance_ratio), float(Tf)
        self.roulette_prob, self.restart = float(roulette_prob), bool(restart)
        # Algorithm 5, line 14: iter grows when a worse solution is accepted ("accepted", the pseudo-code);
        # "non_improving" counts every non-improving iteration, as the prose of their Section 5.2.3 reads
        self.iter_count = iter_count

    def solve(self, inst, budget, seed, deadline):
        t0 = time.time()
        K.seed_numba(int(seed) + 101)
        s, e, p, wu = inst.s, inst.e, inst.p, inst.wu
        flex = np.where(e - s + 1 > p)[0].astype(np.int64)
        x, L = zaidi_heuristic_kernel(s, e, p, wu, inst.T, zaidi_order(inst), zaidi_cap(inst))
        best_x, best_L = x.copy(), L.copy()
        fb = L.max()
        trace = [(time.time() - t0, fb / UNIT)]
        if flex.shape[0] == 0:
            return best_x, trace
        cur_x, cur_L = x, L
        it_total, restarts = 0, 0
        restart = self.restart and budget is not None
        while not deadline.expired():
            it, pert, pmax = 0, self.pert0, self.pertmax * self.pert0          # Algorithm 5, line 1
            while pert < pmax and not deadline.expired():
                k = 1 + int(np.random.default_rng(int(seed) * 7919 + it_total).integers(pert))
                nx, nL = cur_x.copy(), cur_L.copy()
                zaidi_perturb(nx, nL, s, e, p, wu, k, self.roulette_prob)
                nx, nL = zaidi_sa(nx, nL, s, e, p, wu, flex, float(UNIT), self.mu, self.max_generated,
                                  self.max_trials, self.acceptance_ratio, self.Tf)
                it_total += 1
                f = nL.max()
                if f < fb:
                    best_x, best_L, fb = nx, nL, f
                    cur_x, cur_L = nx, nL
                    it, pert = 0, self.pert0
                    trace.append((time.time() - t0, fb / UNIT))
                else:
                    if np.random.default_rng(int(seed) * 104729 + it_total).random() < self.p0 * self.r ** (it - 1):
                        cur_x, cur_L = nx, nL
                        if self.iter_count == "accepted":
                            it += 1               # Algorithm 5, line 14
                    if self.iter_count != "accepted":
                        it += 1                   # every non-improving iteration
                if it >= self.itermax:
                    it, pert = 0, pert + self.pert0
                    cur_x, cur_L = best_x.copy(), best_L.copy()
            if not restart:
                break
            restarts += 1
            cur_x, cur_L = best_x.copy(), best_L.copy()
        self._info = {"ils_iterations": it_total, "restarts": restarts}
        return best_x, trace


# ── Hunsaker, Kleywegt, Savelsbergh & Tovey (2003): optimal online alpha-policy ─

_ALPHA_CACHE = {}


def hunsaker_alpha(T):
    """Optimal competitive ratio alpha_T of the ORMP (Hunsaker et al., Corollary 4.13): from the table
    data/hunsaker_alpha.json (tools/hunsaker_alpha_table.py, T <= 300), else solved here with the same
    prefix-sum LP: min sum_t x(t) s.t. S(T) = 1, (T-i+1) x(t) >= S(t) - S(i-1), alpha_T = 1 / optimum."""
    if T in _ALPHA_CACHE:
        return _ALPHA_CACHE[T]
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hunsaker_alpha.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            tab = json.load(f)["alpha"]
        _ALPHA_CACHE.update({int(k): float(v) for k, v in tab.items()})
        if T in _ALPHA_CACHE:
            return _ALPHA_CACHE[T]
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    na, nS = T, T + 1
    nv = na + nS + T
    er, ec, ev, eb = [0], [na], [1.0], [0.0]                     # S(0) = 0
    for t in range(1, T + 1):                                    # S(t) - S(t-1) - a(t) = 0
        r = len(eb)
        er += [r, r, r]; ec += [na + t, na + t - 1, t - 1]; ev += [1.0, -1.0, -1.0]; eb.append(0.0)
    r = len(eb); er.append(r); ec.append(na + T); ev.append(1.0); eb.append(1.0)   # S(T) = 1
    rr, cc, vv, k = [], [], [], 0
    for t in range(1, T + 1):
        for i in range(1, t + 1):
            rr += [k, k, k]; cc += [na + t, na + i - 1, na + nS + t - 1]; vv += [1.0, -1.0, -float(T - i + 1)]
            k += 1
    c = np.zeros(nv); c[na + nS:] = 1.0
    res = linprog(c, A_ub=coo_matrix((vv, (rr, cc)), shape=(k, nv)).tocsr(), b_ub=np.zeros(k),
                  A_eq=coo_matrix((ev, (er, ec)), shape=(len(eb), nv)).tocsr(), b_eq=np.array(eb),
                  bounds=[(0, None)] * nv, method="highs")
    _ALPHA_CACHE[T] = 1.0 / res.fun
    return _ALPHA_CACHE[T]


@njit(cache=True)
def hunsaker_kernel(s, e, p, wu, T, alpha, edf):
    """alpha-policy, made integral: at slot t the level is r(t) = alpha * v_T(t), with v_T(t) the largest
    density of the energy known at t (Proposition 4.4); vehicles that must charge to meet their departure
    charge first, then the others in earliest-deadline-first order while the load stays <= r(t)."""
    n = s.shape[0]
    C = np.zeros((T + 1, T + 1), np.float64)        # C[a+1][b+1]: energy of vehicles with s <= a, e <= b
    for i in range(n):
        C[s[i] + 1, e[i] + 1] += wu[i] * p[i]
    for a in range(1, T + 1):
        for b in range(1, T + 1):
            C[a, b] += C[a - 1, b] + C[a, b - 1] - C[a - 1, b - 1]
    x = np.zeros((n, T), np.int8)
    rem = p.copy()
    L = np.zeros(T, np.int64)
    levels = np.zeros(T, np.float64)
    v = 0.0
    for t in range(T):
        for i in range(t + 1):                       # v_T(t): known work, windows inside [i, j]
            for j in range(i, T):
                k = min(j, t)
                E = C[k + 1, j + 1] - C[i, j + 1]
                d = E / (j - i + 1)
                if d > v:
                    v = d
        r = alpha * v
        levels[t] = r
        load = 0
        for oi in range(n):                          # forced vehicles
            i = edf[oi]
            if s[i] <= t and t <= e[i] and rem[i] > 0 and rem[i] == e[i] - t + 1:
                x[i, t] = 1; rem[i] -= 1; load += wu[i]
        for oi in range(n):                          # EDF fill up to the level
            i = edf[oi]
            if s[i] <= t and t <= e[i] and rem[i] > 0 and x[i, t] == 0 and load + wu[i] <= r + 1e-9:
                x[i, t] = 1; rem[i] -= 1; load += wu[i]
        L[t] = load
    return x, L, levels


@register_method("HUNSAKER")
class HunsakerOnline(SotaMethod):
    """Optimal online alpha-policy of Hunsaker et al. (2003) for minimax resource scheduling (ORMP), made
    integral for the PPMSP. Online: a vehicle is known only from its arrival slot."""
    stochastic = False
    reference = "Hunsaker2003"

    def __init__(self, alpha=None, **params):
        super().__init__(**params)
        self.alpha = alpha

    def prepare(self, inst):
        self._alpha = float(self.alpha) if self.alpha is not None else hunsaker_alpha(inst.T)

    def solve(self, inst, budget, seed, deadline):
        alpha = self._alpha
        edf = np.lexsort((np.arange(inst.n), inst.s, inst.e)).astype(np.int64)
        x, L, levels = hunsaker_kernel(inst.s, inst.e, inst.p, inst.wu, inst.T, alpha, edf)
        self._info = {"alpha_T": alpha, "final_level": float(levels[-1]) / UNIT}
        return x


# ── Afshar-Nadjafi, Khalaj & Mehdizadeh (2013): branch and bound ───────────────

@njit(cache=True)
def _bb_next_comb(comb, m, k):
    """Next m-combination of range(k) in lexicographic order; False when exhausted."""
    i = m - 1
    while i >= 0 and comb[i] == k - m + i:
        i -= 1
    if i < 0:
        return False
    comb[i] += 1
    for j in range(i + 1, m):
        comb[j] = comb[j - 1] + 1
    return True


@njit(cache=True)
def bb_run(s, e, p, wu, x, rem, t0, D, opt, nopt, forc, nforc, sidx, comb, started, tried, applied, peak_at,
           hstack, Z, ZD, memo, memo_cap, st, best_x, max_nodes):
    """Depth-first branch and bound over the slots t0 .. t0+D-1 (resumable: the search state lives in the
    arrays). st = [depth, UB, nodes, done, leaves, memo_hits]. A node is the set of vehicles charging in the
    current slot; branches are the subsets of the optional vehicles (sizes 1, 2, ..., then the empty set),
    forced vehicles always charge (rule (a): remaining duration = remaining window). Rule (c): a branch whose
    load reaches the incumbent is cut. Rule (d): a transposition table on (slot, remaining durations) cuts a
    node reached again with a peak that is not lower."""
    n = s.shape[0]
    d = st[0]
    UB = st[1]
    count = 0
    while count < max_nodes:
        if d < 0:
            st[3] = 1
            break
        if d == D:                                   # leaf: a complete schedule
            if peak_at[D] < UB:
                UB = peak_at[D]
                best_x[:, :] = x
                st[4] += 1
            d -= 1
            continue
        t = t0 + d
        if not started[d]:
            k = 0; f = 0; fl = 0
            for i in range(n):
                if rem[i] > 0 and s[i] <= t and t <= e[i]:
                    if rem[i] == e[i] - t + 1:
                        forc[d, f] = i; f += 1; fl += wu[i]
                    else:
                        opt[d, k] = i; k += 1
            for a in range(1, k):                      # optional vehicles by non-decreasing power
                v = opt[d, a]; b = a - 1
                while b >= 0 and wu[opt[d, b]] > wu[v]:
                    opt[d, b + 1] = opt[d, b]; b -= 1
                opt[d, b + 1] = v
            nopt[d] = k; nforc[d] = f
            sidx[d] = 0                               # sizes 1..k, then 0
            m = 1 if k > 0 else 0
            for j in range(m):
                comb[d, j] = j
            if m > 0 and fl + wu[opt[d, 0]] >= UB:
                sidx[d] = k                           # even the lightest single vehicle is cut
            started[d] = True; tried[d] = False; applied[d] = False
        else:
            k = nopt[d]
        if tried[d]:
            if applied[d]:                            # undo the current subset
                m = sidx[d] + 1 if sidx[d] < k else 0
                for j in range(nforc[d]):
                    i = forc[d, j]; x[i, t] = 0; rem[i] += 1
                for j in range(m):
                    i = opt[d, comb[d, j]]; x[i, t] = 0; rem[i] += 1
                applied[d] = False
            m = sidx[d] + 1 if sidx[d] < k else 0
            if not (m > 0 and _bb_next_comb(comb[d], m, k)):
                sidx[d] += 1
                if sidx[d] > k or (k == 0):
                    started[d] = False
                    d -= 1
                    continue
                m = sidx[d] + 1 if sidx[d] < k else 0
                for j in range(m):
                    comb[d, j] = j
                if m > 0:                             # rule (c) for a whole size: the m lightest already cut
                    lo_load = 0
                    for j in range(nforc[d]):
                        lo_load += wu[forc[d, j]]
                    for j in range(m):
                        lo_load += wu[opt[d, j]]
                    if lo_load >= UB:
                        sidx[d] = k                   # every larger subset is cut too: go to the empty set
                        m = 0
        tried[d] = True
        count += 1
        m = sidx[d] + 1 if sidx[d] < k else 0
        load = 0
        for j in range(nforc[d]):
            load += wu[forc[d, j]]
        for j in range(m):
            load += wu[opt[d, comb[d, j]]]
        newpeak = peak_at[d] if peak_at[d] > load else load
        if newpeak >= UB:                             # rule (c)
            continue
        h = hstack[d]
        for j in range(nforc[d]):
            i = forc[d, j]; x[i, t] = 1; h ^= Z[i, rem[i]] ^ Z[i, rem[i] - 1]; rem[i] -= 1
        for j in range(m):
            i = opt[d, comb[d, j]]; x[i, t] = 1; h ^= Z[i, rem[i]] ^ Z[i, rem[i] - 1]; rem[i] -= 1
        applied[d] = True
        key = h ^ ZD[d + 1]
        if key in memo and memo[key] <= newpeak:      # rule (d)
            st[5] += 1
            continue
        if len(memo) < memo_cap:
            memo[key] = newpeak
        hstack[d + 1] = h
        peak_at[d + 1] = newpeak
        d += 1
        st[2] += 1
    st[0] = d
    st[1] = UB


@register_method("AFSHAR_BB")
class AfsharBB(SotaMethod):
    """Branch and bound of Afshar-Nadjafi et al. (2013) for preemptive resource leveling, adapted to the
    PPMSP (peak objective; vehicles = activities without precedence; windows = release dates and deadlines;
    unlimited preemption). Exact if it ends within the budget; otherwise returns its incumbent."""
    stochastic = False
    reference = "AfsharNadjafi2013"

    def __init__(self, chunk_nodes=200000, memo_cap=2000000, **params):
        super().__init__(**params)
        self.chunk_nodes, self.memo_cap = int(chunk_nodes), int(memo_cap)

    def solve(self, inst, budget, seed, deadline):
        from numba import types
        from numba.typed import Dict
        t0_wall = time.time()
        n, T = inst.n, inst.T
        t0, t1 = int(inst.s.min()), int(inst.e.max()) + 1
        D = t1 - t0
        x = np.zeros((n, T), np.int8)
        rem = inst.p.astype(np.int64).copy()
        rng = np.random.default_rng(12345)
        Z = rng.integers(1, 2 ** 62, size=(n, int(inst.p.max()) + 1), dtype=np.int64)
        ZD = rng.integers(1, 2 ** 62, size=D + 2, dtype=np.int64)
        h0 = 0
        for i in range(n):
            h0 ^= int(Z[i, rem[i]])
        hstack = np.zeros(D + 2, np.int64); hstack[0] = h0
        memo = Dict.empty(key_type=types.int64, value_type=types.int64)
        st = np.array([0, int(inst.wu.sum()) + 1, 0, 0, 0, 0], np.int64)   # UB: total power + 1 (any schedule)
        best_x = np.zeros((n, T), np.int8)
        args = (np.zeros((D, n), np.int64), np.zeros(D, np.int64), np.zeros((D, n), np.int64), np.zeros(D, np.int64),
                np.zeros(D, np.int64), np.zeros((D, n + 1), np.int64), np.zeros(D, np.bool_), np.zeros(D, np.bool_),
                np.zeros(D, np.bool_), np.zeros(D + 1, np.int64))
        trace = []
        while True:
            leaves = st[4]
            bb_run(inst.s, inst.e, inst.p, inst.wu, x, rem, t0, D, *args, hstack, Z, ZD, memo, self.memo_cap, st,
                   best_x, self.chunk_nodes)
            if st[4] > leaves:
                trace.append((time.time() - t0_wall, st[1] / UNIT))
            if st[3] or deadline.expired():
                break
        if st[4] == 0:                               # no complete schedule within the budget: greedy fallback
            best_x, _ = K.greedy_vf(inst.s, inst.e, inst.p, inst.wu, T, 0)
        self._info = {"nodes": int(st[2]), "complete_search": bool(st[3]), "leaves": int(st[4]),
                      "memo_cuts": int(st[5]), "fallback": bool(st[4] == 0)}
        return best_x, (trace or None)


# ── Aristotelous & Nearchou (2024): hybrid metaheuristics with leveling schemes ─

@njit(cache=True)
def an_decode(keys, s, e, p, wu, T):
    """Random-key decoder: vehicles by priority (increasing key); each one takes the p_i least-loaded slots
    of its window (earliest first on ties). (Serial SGS adapted: without capacities, 'as early as possible'
    would give every individual the same schedule.)"""
    n = s.shape[0]
    order = np.argsort(keys)
    x = np.zeros((n, T), np.int8)
    L = np.zeros(T, np.int64)
    for oi in range(n):
        i = order[oi]
        m = e[i] - s[i] + 1
        loads = np.empty(m, np.int64)
        for j in range(m):
            loads[j] = L[s[i] + j]
        idx = np.argsort(loads, kind="mergesort")
        for j in range(p[i]):
            t = s[i] + idx[j]
            x[i, t] = 1
            L[t] += wu[i]
    return x, L


@njit(cache=True)
def an_dev(L, mu):
    tot = 0.0
    for t in range(L.shape[0]):
        tot += abs(L[t] - mu)
    return tot


@njit(cache=True)
def an_fitness(L, mu, scale, objective):
    """objective 0: peak, ties by DEV (lexicographic); 1: DEV (as published)."""
    if objective == 1:
        return an_dev(L, mu)
    pk = 0
    for t in range(L.shape[0]):
        if L[t] > pk:
            pk = L[t]
    return pk + an_dev(L, mu) / scale


@njit(cache=True)
def an_level(x, L, s, e, p, wu, mu, scheme):
    """Leveling schemes LS1 / LS2 of their pseudo-code, preemptive adaptation. The non-critical vehicles
    (window longer than the duration) are visited once, by non-increasing total slack TS = |window| - p. For
    each one an integer r is drawn in [0, TS], a single modification is tried and kept only if DEV decreases.
    LS1, 'increase the duration by r': the vehicle is re-scheduled on the p least-loaded slots of the span
    [ES, ES + p - 1 + r], ES being its first charging slot (its charging is spread over p + r slots, the
    preemptive form of a longer, less intensive job). LS2 makes, with probability 1/2 each, this move or
    'increase ES by r': its whole charging pattern is shifted r slots later (r clipped to the window)."""
    n = s.shape[0]
    T = L.shape[0]
    fl = np.empty(n, np.int64)
    for i in range(n):
        fl[i] = -(e[i] - s[i] + 1 - p[i])
    order = np.argsort(fl, kind="mergesort")
    dev = an_dev(L, mu)
    old = np.empty(T, np.int64)
    new = np.empty(T, np.int64)
    for oi in range(n):
        i = order[oi]
        TS = e[i] - s[i] + 1 - p[i]
        if TS <= 0:
            continue
        w = wu[i]
        r = np.random.randint(TS + 1)
        no = 0
        for t in range(s[i], e[i] + 1):
            if x[i, t]:
                old[no] = t
                no += 1
        for k in range(no):
            t = old[k]
            x[i, t] = 0
            L[t] -= w
        if scheme == 2 and np.random.random() >= 0.5:      # increase ES by r: delay the whole pattern
            rr = min(r, e[i] - old[no - 1])
            for k in range(no):
                new[k] = old[k] + rr
        else:                                               # increase the duration by r
            es = old[0]
            hi = min(e[i], es + p[i] - 1 + r)
            m = hi - es + 1
            loads = np.empty(m, np.int64)
            for j in range(m):
                loads[j] = L[es + j]
            idx = np.argsort(loads, kind="mergesort")
            for k in range(p[i]):
                new[k] = es + idx[k]
        for k in range(no):
            t = new[k]
            x[i, t] = 1
            L[t] += w
        d2 = an_dev(L, mu)
        if d2 < dev - 1e-9:
            dev = d2
        else:                                               # not better: back to the previous pattern
            for k in range(no):
                t = new[k]
                x[i, t] = 0
                L[t] -= w
            for k in range(no):
                t = old[k]
                x[i, t] = 1
                L[t] += w


@njit(cache=True)
def an_reencode(x, s, T, keys):
    """Keys of a schedule: earlier first charging slot = higher priority (plus a random tie-break)."""
    n = x.shape[0]
    for i in range(n):
        f = T
        for t in range(s[i], T):
            if x[i, t]:
                f = t
                break
        keys[i] = (f + 0.5 * np.random.random()) / (T + 1.0)


@njit(cache=True)
def an_eval_pop(P, fit, s, e, p, wu, T, mu, scale, objective, lo, hi):
    for k in range(lo, hi):
        _, L = an_decode(P[k], s, e, p, wu, T)
        fit[k] = an_fitness(L, mu, scale, objective)


@njit(cache=True)
def an_de_step(P, fit, s, e, p, wu, T, mu, scale, objective, F, CR, lo, hi):
    """DE/best/1/bin on the members lo..hi-1 (their Eqs. (6)-(8)): distinct r1, r2, both different from the
    target; the trial replaces the target if its cost is lower."""
    D = P.shape[1]
    m = hi - lo
    b = lo
    for k in range(lo, hi):
        if fit[k] < fit[b]:
            b = k
    trial = np.empty(D)
    for k in range(lo, hi):
        r1 = lo + np.random.randint(m); r2 = lo + np.random.randint(m)
        if m >= 3:
            while r1 == k:
                r1 = lo + np.random.randint(m)
            while r2 == k or r2 == r1:
                r2 = lo + np.random.randint(m)
        jr = np.random.randint(D)
        for j in range(D):
            if np.random.random() < CR or j == jr:
                v = P[b, j] + F * (P[r1, j] - P[r2, j])
                trial[j] = min(1.0, max(0.0, v))
            else:
                trial[j] = P[k, j]
        _, L = an_decode(trial, s, e, p, wu, T)
        f = an_fitness(L, mu, scale, objective)
        if f < fit[k]:
            P[k, :] = trial
            fit[k] = f


@njit(cache=True)
def an_ga_step(P, fit, s, e, p, wu, T, mu, scale, objective, pc, pm, eta_c, eta_m, lo, hi, elitism):
    """Real-coded GA: binary tournaments, SBX crossover (pc), polynomial mutation (pm per gene); the children
    replace the population (their Section 4.4); elitism=True keeps the best member (not in the paper)."""
    D = P.shape[1]
    m = hi - lo
    Q = np.empty((m, D))
    b = lo
    for k in range(lo, hi):
        if fit[k] < fit[b]:
            b = k
    c = 0
    if elitism:
        Q[0, :] = P[b]
        c = 1
    while c < m:
        a1 = lo + np.random.randint(m); a2 = lo + np.random.randint(m)
        p1 = a1 if fit[a1] <= fit[a2] else a2
        a1 = lo + np.random.randint(m); a2 = lo + np.random.randint(m)
        p2 = a1 if fit[a1] <= fit[a2] else a2
        c1 = P[p1].copy(); c2 = P[p2].copy()
        if np.random.random() < pc:
            for j in range(D):
                u = np.random.random()
                beta = (2 * u) ** (1 / (eta_c + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
                y1 = 0.5 * ((1 + beta) * c1[j] + (1 - beta) * c2[j])
                y2 = 0.5 * ((1 - beta) * c1[j] + (1 + beta) * c2[j])
                c1[j] = min(1.0, max(0.0, y1)); c2[j] = min(1.0, max(0.0, y2))
        for cc in (c1, c2):
            for j in range(D):
                if np.random.random() < pm:
                    u = np.random.random()
                    dq = (2 * u) ** (1 / (eta_m + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                    cc[j] = min(1.0, max(0.0, cc[j] + dq))
        Q[c, :] = c1; c += 1
        if c < m:
            Q[c, :] = c2; c += 1
    for k in range(m):
        P[lo + k, :] = Q[k]
    an_eval_pop(P, fit, s, e, p, wu, T, mu, scale, objective, lo, hi)


@njit(cache=True)
def an_pso_step(P, V, PB, PBf, fit, s, e, p, wu, T, mu, scale, objective, w, c1, c2, vmax, lo, hi):
    D = P.shape[1]
    g = lo
    for k in range(lo, hi):
        if PBf[k] < PBf[g]:
            g = k
    for k in range(lo, hi):
        for j in range(D):
            v = w * V[k, j] + c1 * np.random.random() * (PB[k, j] - P[k, j]) + c2 * np.random.random() * (PB[g, j] - P[k, j])
            v = min(vmax, max(-vmax, v))
            V[k, j] = v
            P[k, j] = min(1.0, max(0.0, P[k, j] + v))
    an_eval_pop(P, fit, s, e, p, wu, T, mu, scale, objective, lo, hi)
    for k in range(lo, hi):
        if fit[k] < PBf[k]:
            PBf[k] = fit[k]
            PB[k, :] = P[k]


@register_method("AN_HYBRID")
class AristotelousNearchou(SotaMethod):
    """Hybrid metaheuristics of Aristotelous & Nearchou (2024), Table 1 settings: variant SER (DE, then GA,
    then PSO; their best with LS2), PAR (DE | GA | PSO on three partitions), DE, GA or PSO, each with the
    leveling scheme LS1 or LS2 applied to the best member of every (sub)population at every generation.
    Adapted to the PPMSP: random-key decoder that places a vehicle in the least-loaded slots of its window,
    preemptive leveling moves, fitness = peak (ties by DEV) unless objective='dev' (as published)."""
    reference = "Aristotelous2024"

    def __init__(self, variant="SER", ls="LS2", pop=100, maxgen=5000, de_F=0.5, de_CR=0.6, ga_pc=0.6, ga_pm=0.01,
                 eta_c=20.0, eta_m=20.0, pso_w=1.0, pso_c1=2.0, pso_c2=2.5, pso_vmax=0.25, objective="peak",
                 ga_elitism=False, **params):
        super().__init__(**params)
        self.variant, self.ls, self.pop, self.maxgen = variant, ls, int(pop), int(maxgen)
        self.de_F, self.de_CR, self.ga_pc, self.ga_pm = float(de_F), float(de_CR), float(ga_pc), float(ga_pm)
        self.eta_c, self.eta_m = float(eta_c), float(eta_m)
        self.pso_w, self.pso_c1, self.pso_c2, self.pso_vmax = float(pso_w), float(pso_c1), float(pso_c2), float(pso_vmax)
        self.objective = 1 if objective == "dev" else 0
        self.ga_elitism = bool(ga_elitism)

    def solve(self, inst, budget, seed, deadline):
        t_start = time.time()
        K.seed_numba(int(seed) + 211)
        rng = np.random.default_rng(int(seed) + 211)
        s, e, p, wu, T = inst.s, inst.e, inst.p, inst.wu, inst.T
        n, N = inst.n, self.pop
        mu = float((wu * p).sum()) / T
        scale = float(T) * float((wu * p).sum()) + 1.0
        obj = self.objective
        P = rng.random((N, n)); fit = np.empty(N)
        V = rng.uniform(-self.pso_vmax, self.pso_vmax, (N, n))     # their Eq. (10), [vmin, vmax] = [-vmax, vmax]
        an_eval_pop(P, fit, s, e, p, wu, T, mu, scale, obj, 0, N)
        PB, PBf = P.copy(), fit.copy()
        scheme = 2 if self.ls == "LS2" else 1
        best_x, best_L = an_decode(P[int(np.argmin(fit))], s, e, p, wu, T)
        best_pk = best_L.max()
        trace = [(time.time() - t_start, best_pk / UNIT)]
        if self.variant == "PAR":
            thirds = [(0, N // 3, "DE"), (N // 3, 2 * N // 3, "GA"), (2 * N // 3, N, "PSO")]
        gen = 0
        budget_s = None if budget is None else float(budget)
        while gen < self.maxgen and not deadline.expired():
            if self.variant == "SER":
                # phases of MAXGEN/3 generations; under a time budget each phase also gets at most a third
                ph = min(2, gen * 3 // self.maxgen)
                if budget_s is not None:
                    ph = max(ph, min(2, int(3 * (time.time() - t_start) / budget_s)))
                parts = [(0, N, ("DE", "GA", "PSO")[ph])]
            elif self.variant == "PAR":
                parts = thirds
            else:
                parts = [(0, N, self.variant)]
            for lo, hi, op in parts:
                if op == "DE":
                    an_de_step(P, fit, s, e, p, wu, T, mu, scale, obj, self.de_F, self.de_CR, lo, hi)
                elif op == "GA":
                    an_ga_step(P, fit, s, e, p, wu, T, mu, scale, obj, self.ga_pc, self.ga_pm, self.eta_c, self.eta_m, lo, hi,
                               self.ga_elitism)
                else:
                    an_pso_step(P, V, PB, PBf, fit, s, e, p, wu, T, mu, scale, obj, self.pso_w, self.pso_c1, self.pso_c2,
                                self.pso_vmax, lo, hi)
                k = lo + int(np.argmin(fit[lo:hi]))      # leveling scheme on the best member of the part
                x, L = an_decode(P[k], s, e, p, wu, T)
                an_level(x, L, s, e, p, wu, mu, scheme)
                f = an_fitness(L, mu, scale, obj)
                if f < fit[k]:                          # the improved schedule goes back into the population:
                    keys = P[k].copy()                  # its keys (by first charging slot) replace the member's
                    an_reencode(x, s, T, keys)          # if what they decode to is not worse
                    _, L2 = an_decode(keys, s, e, p, wu, T)
                    f2 = an_fitness(L2, mu, scale, obj)
                    if f2 <= fit[k]:
                        P[k, :] = keys; fit[k] = f2
                        if f2 < PBf[k]:
                            PB[k, :] = keys; PBf[k] = f2
                if L.max() < best_pk:
                    best_pk, best_x = L.max(), x.copy()
                    trace.append((time.time() - t_start, best_pk / UNIT))
            gen += 1
        self._info = {"generations": gen}
        return best_x, trace


SOTA = ("BK", "NFDH", "ZAIDI_H", "ZAIDI_ILS", "HUNSAKER", "AFSHAR_BB", "AN_HYBRID")
