"""
certificates.py -- a priori certificates on the optimal peak P* of the PPMSP, and the
algorithms that attain them (Section 4 of the paper).

All certificates are computed from the instance and one LP solution alpha (no search).
They are valid for ANY alpha in the LP feasible set; alpha* = an LP vertex is the default.

Structure used (see Section 4 of the paper):
  V_t      random vehicles at slot t (0 < alpha_it < 1); D_t deterministic load
  G_alpha  slots t ~ t' iff V_t and V_t' share a vehicle; its connected components are
           mutually independent blocks under any product strategy
Certificates (upper bounds on P*):
  closed forms   Hoeffding / Kearns-Saul / Bernstein, union or local lemma, local quantities
  per-slot       union, componentwise union, asymmetric local lemma (exact or analytic tails)
  averaging      P* <= E[max_t S_t] (Meilijson-Nadas form, and componentwise product form)
  deterministic  iterative LP rounding: P* <= P_LP + w_max (Lau-Ravi-Singh style)
Lower bound:
  cut_lower_bound  smallest attainable load P at which the LP with the cover cuts at level P
                   is feasible (binary search; >= max(ceil P_LP, w_max)); its vertex alpha_cut
                   is a second marginal matrix for every certificate and algorithm
Algorithms:
  derandomized_union_by_component   peak <= componentwise-union certificate
  iterative_rounding                peak <= P_LP + w_max
  moser_tardos                      Las Vegas; expected resamplings <= sum x_t/(1-x_t)
"""

import math

import numpy as np

import bounds as B

UNIT = B.UNIT
MARGIN = 1e-9   # a failure-probability sum must be below 1 - MARGIN (exact ties are not success)
IR_LP_COUNT = 0  # number of LPs solved by iterative_rounding since import (Theorem 4.21 bound check)


# ── per-vehicle sampler with prescribed marginals ───────────────────────────────

def systematic_sample(slots, probs, rng):
    """Exactly sum(probs) distinct slots, slot k with probability probs[k] (Madow 1949).
    Requires probs in [0, 1] with an integer sum."""
    c = np.concatenate([[0.0], np.cumsum(probs)])
    r = int(round(c[-1]))
    c *= r / c[-1]                      # remove rounding drift: the last cut point is exactly r
    u = rng.random()
    pts = u + np.arange(r)
    idx = np.searchsorted(c, pts, side="right") - 1
    return slots[idx]


# ── Kearns-Saul optimal sub-Gaussian variance proxy of a centred Bernoulli ──────

def ks_proxy(a):
    """sigma^2(a) with log E exp(l (X - a)) <= l^2 sigma^2 / 2 for X ~ Bernoulli(a)
    (Kearns & Saul 1998; optimality: Berend & Kontorovich 2013). Equals 1/4 at a = 1/2."""
    a = np.asarray(a, float)
    out = np.full(a.shape, 0.25)
    m = np.abs(a - 0.5) > 1e-9
    am = a[m]
    out[m] = (1.0 - 2.0 * am) / (2.0 * np.log((1.0 - am) / am))
    return out


# ── the certificate engine ──────────────────────────────────────────────────────

class Certificates:
    def __init__(self, inst, alpha, p_lp=None):
        self.inst, self.alpha = inst, alpha
        L = B.SlotLaws(inst, alpha)
        self.laws = L
        T = inst.T
        self.p_lp = p_lp
        self.H = int(L.maxload.max()) + 1                         # grid 0..H (units)
        u = np.arange(self.H + 1)
        surv = np.zeros((T, self.H + 1))                          # surv[t, u] = P(S_t > u)
        for t in range(T):
            law = L.law[t]
            sR = np.concatenate([np.cumsum(law[::-1])[::-1], [0.0]])   # sR[k] = P(R >= k)
            k = u - L.det[t] + 1
            surv[t] = np.where(k <= 0, 1.0, sR[np.clip(k, 0, len(law))])
        self.surv = surv
        # lower end of every "smallest u" search: below the largest deterministic load no
        # certificate condition can hold. (Not floor(max mu): for an alpha that is not optimal
        # for a relaxation, the optimal rounding can lie below the largest expected load.)
        self.lo = int(L.det.max())
        # components of G_alpha (union-find over random vehicles)
        parent = list(range(T))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        frac = {}
        for t in range(T):
            for i, _, _ in L.rand[t]:
                frac.setdefault(i, []).append(t)
        for slots in frac.values():
            for t in slots[1:]:
                ra, rb = find(slots[0]), find(t)
                if ra != rb:
                    parent[rb] = ra
        comp = {}
        for t in range(T):
            comp.setdefault(find(t), []).append(t)
        self.components = list(comp.values())
        self.random_comps = [c for c in self.components if any(L.rand[t] for t in c)]
        self.frac_slots = frac

    # smallest grid point satisfying a monotone predicate
    def _first(self, ok):
        lo, hi = self.lo, self.H
        while lo < hi:
            mid = (lo + hi) // 2
            if ok(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    # ---- exact per-slot certificates ----
    def union(self):
        tot = self.surv.sum(axis=0)
        return self._first(lambda u: tot[u] < 1 - MARGIN) / UNIT

    def component_union(self):
        sums = np.array([self.surv[c].sum(axis=0) for c in self.components])
        worst = sums.max(axis=0)
        return self._first(lambda u: worst[u] < 1 - MARGIN) / UNIT

    def lll(self):
        return self._first(lambda u: B.lll_holds(self.surv[:, u], self.laws.nbr)) / UNIT

    def success_probability(self, u_kw):
        """Lower bound on P(a sample meets max_t S_t <= u): prod_c (1 - sum_{t in c} P(S_t > u))^+."""
        u = int(round(u_kw * UNIT))
        prod = 1.0
        for c in self.components:
            prod *= max(0.0, 1.0 - self.surv[c, u].sum())
        return prod

    # ---- averaging certificates ----
    def averaging(self):
        """E[max] <= sum_u min(1, sum_t P(S_t > u)) (Meilijson-Nadas)."""
        return np.minimum(1.0, self.surv.sum(axis=0)).sum() / UNIT

    def component_averaging(self):
        """E[max] = sum_u (1 - prod_c P(M_c <= u)) <= sum_u (1 - prod_c (1 - sum_{t in c} P(S_t > u))^+)."""
        good = np.ones(self.H + 1)
        for c in self.components:
            good *= np.clip(1.0 - self.surv[c].sum(axis=0), 0.0, 1.0)
        return (1.0 - good).sum() / UNIT

    # ---- analytic per-slot tails (closed forms per slot) ----
    def _analytic_surv(self, kind):
        L, T = self.laws, self.inst.T
        u = np.arange(self.H + 1)
        out = np.zeros((T, self.H + 1))
        for t in range(T):
            if not L.rand[t]:
                out[t] = (u < L.det[t]).astype(float)
                continue
            eps = u - L.mu[t]
            w = np.array([wk for _, wk, _ in L.rand[t]], float)
            a = np.array([ak for _, _, ak in L.rand[t]], float)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                if kind == "hoeffding":
                    expo = 2.0 * eps ** 2 / (w ** 2).sum()
                elif kind == "kearns_saul":
                    expo = eps ** 2 / (2.0 * (w ** 2 * ks_proxy(a)).sum())
                elif kind == "bernstein":
                    V = (w ** 2 * a * (1 - a)).sum(); M = (w * (1 - a)).max()
                    expo = eps ** 2 / (2.0 * V + 2.0 * M * eps / 3.0)
                else:  # bennett
                    V = (w ** 2 * a * (1 - a)).sum(); M = (w * (1 - a)).max()
                    x = M * eps / V
                    expo = (V / M ** 2) * ((1 + x) * np.log1p(x) - x)
            out[t] = np.where(eps > 0, np.exp(-np.where(eps > 0, expo, 0.0)), 1.0)
            # a load above the maximum possible value has probability 0
            out[t][u >= L.maxload[t]] = 0.0
        return out

    def analytic(self, kind):
        s = self._analytic_surv(kind)
        tot = s.sum(axis=0)
        sums = np.array([s[c].sum(axis=0) for c in self.components]).max(axis=0)
        return {"union": self._first(lambda u: tot[u] < 1 - MARGIN) / UNIT,
                "component_union": self._first(lambda u: sums[u] < 1 - MARGIN) / UNIT,
                "lll": self._first(lambda u: B.lll_holds(s[:, u], self.laws.nbr)) / UNIT}

    # ---- closed forms with a single threshold (drop-in for the non-localized theorems) ----
    def closed_forms(self):
        L, T = self.laws, self.inst.T
        rs = [t for t in range(T) if L.rand[t]]
        base = L.mu.max() / UNIT
        out = {}
        if not rs:
            return {k: base for k in ("hoeffding_lll", "ks_lll", "bernstein_lll", "hoeffding_comp", "ks_comp")}
        W = max((sum(wk ** 2 for _, wk, _ in L.rand[t]) for t in rs)) / UNIT ** 2
        KS = max(float((np.array([wk for _, wk, _ in L.rand[t]], float) ** 2
                        * ks_proxy([ak for _, _, ak in L.rand[t]])).sum()) for t in rs) / UNIT ** 2
        V = L.var[rs].max() / UNIT ** 2
        M = L.rng[rs].max() / UNIT
        d_r = max(len(nb) for nb in L.nbr)
        cmax = max(sum(1 for t in c if L.rand[t]) for c in self.random_comps)
        for name, Lg in (("lll", math.log(math.e * max(d_r, 1))), ("comp", math.log(max(cmax, 2)))):
            out[f"hoeffding_{name}"] = base + math.sqrt(W * Lg / 2.0)
            out[f"ks_{name}"] = base + math.sqrt(2.0 * KS * Lg)
            out[f"bernstein_{name}"] = base + M * Lg / 3.0 * (1.0 + math.sqrt(1.0 + 18.0 * V / (M * M * Lg)))
        out.update({"d_random": d_r, "largest_component": cmax, "n_components": len(self.random_comps)})
        return out

    # ---- integrality rounding of both sides ----
    def attainable_loads(self):
        """All values a slot load can take: union over t of the subset sums of {w_i : i in I(t)} (units).
        P* is the load of some slot under an optimal schedule, so it lies in this set."""
        if not hasattr(self, "_attain"):
            acc = 0
            for t in range(self.inst.T):
                bits = 1
                for i in self.inst.avail[t]:
                    bits |= bits << int(self.inst.wu[i])
                acc |= bits
            self._attain = np.array([k for k in range(acc.bit_length()) if (acc >> k) & 1], int)
        return self._attain

    def round_up_lb(self, lb_kw):
        a = self.attainable_loads()
        k = int(np.ceil(lb_kw * UNIT - 1e-6))
        return a[np.searchsorted(a, k)] / UNIT

    def round_down_ub(self, ub_kw):
        a = self.attainable_loads()
        k = int(np.floor(ub_kw * UNIT + 1e-6))
        return a[np.searchsorted(a, k, side="right") - 1] / UNIT

    def all(self):
        out = {"union": self.union(), "component_union": self.component_union(), "lll": self.lll(),
               "averaging": self.averaging(), "component_averaging": self.component_averaging()}
        for kind in ("hoeffding", "kearns_saul", "bernstein", "bennett"):
            for k, v in self.analytic(kind).items():
                out[f"{kind}_{k}"] = v
        out.update({f"closed_{k}": v for k, v in self.closed_forms().items()})
        return out


# ── algorithms ──────────────────────────────────────────────────────────────────

def derandomized_union_by_component(cert, u_kw=None):
    """Fix every random vehicle (its fractional slots only; integral entries kept) to minimize
    the conditional componentwise estimator sum_{t in c} P(S_t > u). Each component starts
    below 1 and its estimator never increases, so every slot ends <= u."""
    inst, alpha, L = cert.inst, cert.alpha, cert.laws
    u = int(round((u_kw if u_kw is not None else cert.component_union()) * UNIT))
    det = L.det.astype(float).copy()
    rand = [dict((i, (wk, ak)) for i, wk, ak in L.rand[t]) for t in range(inst.T)]
    law = [l.copy() for l in L.law]
    x = (alpha >= 1 - 1e-9).astype(int)
    for i in sorted(cert.frac_slots, key=lambda i: -inst.wu[i]):
        slots = np.array(cert.frac_slots[i])
        r = int(round(alpha[i, slots].sum()))
        wi = int(inst.wu[i])
        lm, delta = [], []
        for t in slots:
            rest = [(w2, a2) for j, (w2, a2) in rand[t].items() if j != i]
            l2 = B.bernoulli_sum_law(rest) if len(rest) <= 12 else B.remove_bernoulli(law[t], *rand[t][i])
            vals = det[t] + np.arange(len(l2))
            g0 = l2[vals > u].sum()
            g1 = l2[vals + wi > u].sum()
            lm.append(l2); delta.append(g1 - g0)
        order = np.lexsort((det[slots], np.array(delta)))
        chosen = set(slots[order[:r]].tolist())
        for k, t in enumerate(slots):
            on = 1 if t in chosen else 0
            det[t] += wi * on; law[t] = lm[k]; rand[t].pop(i, None); x[i, t] = on
    loads = (inst.wu[:, None] * x).sum(axis=0)
    return x, loads.max() / UNIT


def iterative_rounding(inst, p_lp, env=None, tol=1e-9):
    """Iterative LP rounding (Lau, Ravi & Singh 2011, ch. 3; cf. Shmoys & Tardos 1993).
    Keeps every vehicle row; drops a slot row only when it has one fractional variable,
    or two summing to at least 1. Final peak <= P_LP + w_max. Returns the schedule, the
    peak, the instance-dependent guarantee P_LP + max over dropped slots of
    sum w (1 - x), and whether the drop rule always found a slot (as the proof claims)."""
    import gurobipy as gp
    from gurobipy import GRB
    n, T = inst.n, inst.T
    w = inst.w
    edges = {(i, int(t)) for i in range(n) for t in inst.win[i]}
    one = set()
    active = set(range(T))
    guarantee_excess = 0.0
    rule_ok = True
    while edges:
        md = gp.Model(env=env)
        md.Params.OutputFlag = 0
        md.Params.Method = 0
        xv = {e: md.addVar(lb=0.0, ub=1.0) for e in edges}
        by_v, by_t = {}, {}
        for (i, t) in edges:
            by_v.setdefault(i, []).append((i, t)); by_t.setdefault(t, []).append((i, t))
        fixed_v = {}
        fixed_load = np.zeros(T)
        for (i, t) in one:
            fixed_v[i] = fixed_v.get(i, 0) + 1
            fixed_load[t] += w[i]
        for i, es in by_v.items():
            md.addConstr(gp.quicksum(xv[e] for e in es) == int(inst.p[i]) - fixed_v.get(i, 0))
        for t in active:
            if t in by_t:
                md.addConstr(gp.quicksum(w[e[0]] * xv[e] for e in by_t[t]) <= p_lp - fixed_load[t] + 1e-7)
        md.optimize()
        global IR_LP_COUNT
        IR_LP_COUNT += 1
        if md.Status != GRB.OPTIMAL:
            raise RuntimeError("iterative rounding: LP infeasible (should not happen)")
        val = {e: xv[e].X for e in edges}
        changed = False
        for e, v in val.items():
            if v <= tol:
                edges.discard(e); changed = True
            elif v >= 1 - tol:
                edges.discard(e); one.add(e); changed = True
        if changed:
            continue
        # all remaining variables fractional: drop a slot constraint
        cand = None
        for t in sorted(active):
            es = by_t.get(t, [])
            s = sum(val[e] for e in es)
            if len(es) <= 1 or (len(es) == 2 and s >= 1 - 1e-7):
                cand = (t, es)
                break
        if cand is None:
            rule_ok = False
            t = min(active, key=lambda t: len(by_t.get(t, [])))
            cand = (t, by_t.get(t, []))
        t, es = cand
        guarantee_excess = max(guarantee_excess, sum(w[e[0]] * (1 - val[e]) for e in es))
        active.discard(t)
    x = np.zeros((n, T), int)
    for (i, t) in one:
        x[i, t] = 1
    peak = (inst.wu[:, None] * x).sum(axis=0).max() / UNIT
    return x, peak, p_lp + guarantee_excess, rule_ok


def moser_tardos(cert, u_kw, rng, max_steps=10 ** 6):
    """Moser-Tardos resampling with the vehicle schedules as variables (systematic sampling
    with marginals alpha_i). Returns schedule, peak, number of resamplings."""
    inst, alpha, L = cert.inst, cert.alpha, cert.laws
    u = int(round(u_kw * UNIT))
    x = (alpha >= 1 - 1e-9).astype(int)
    frac = cert.frac_slots
    for i, slots in frac.items():
        sl = np.array(slots)
        x[i, sl] = 0
        x[i, systematic_sample(sl, alpha[i, sl], rng)] = 1
    loads = (inst.wu[:, None] * x).sum(axis=0)
    steps = 0
    while steps < max_steps:
        bad = np.where(loads > u)[0]
        if len(bad) == 0:
            break
        t = bad[rng.integers(len(bad))]
        for i, _, _ in L.rand[t]:
            sl = np.array(frac[i])
            loads[sl] -= inst.wu[i] * x[i, sl]
            x[i, sl] = 0
            x[i, systematic_sample(sl, alpha[i, sl], rng)] = 1
            loads[sl] += inst.wu[i] * x[i, sl]
        steps += 1
    return x, loads.max() / UNIT, steps


def lll_weights(cert, u_kw):
    """The least fixed point x of the asymmetric local lemma at threshold u (None if none)."""
    u = int(round(u_kw * UNIT))
    p = cert.surv[:, u]
    nbr = cert.laws.nbr
    x = p.copy()
    for _ in range(5000):
        logs = np.array([np.log1p(-x[nb]).sum() if len(nb) else 0.0 for nb in nbr])
        new = np.where(p > 0, p * np.exp(-logs), 0.0)
        if (new >= 1 - MARGIN).any():
            return None
        if np.abs(new - x).max() < 1e-13:
            return new
        x = new
    return None


# ── strengthened relaxation: cover cuts and the cut lower bound ─────────────────

def cover_cuts(inst, P, kmax=None):
    """Cover cuts valid for every schedule of peak <= P (units): triples (t, vehicles, k) meaning
    sum of x_{i,t} over the vehicles <= k. At slot t, with I(t) sorted by non-increasing power,
    the vehicles are the r largest, r the largest index such that the k+1 smallest of them
    still exceed P together; k runs over 0, 1, ... while the k+1 largest exceed P."""
    out = []
    for t in range(inst.T):
        I = sorted(inst.avail[t], key=lambda i: -inst.wu[i])
        S = np.concatenate([[0], np.cumsum([int(inst.wu[i]) for i in I])])
        m, r = len(I), 0
        for k in range(m if kmax is None else min(m, kmax + 1)):
            if S[k + 1] <= P:
                continue
            r = max(r, k + 1)
            while r < m and S[r + 1] - S[r - k] > P:
                r += 1
            out.append((t, I[:r], k))
    return out


def cut_lp(inst, P, env=None, vertex=False, kmax=None):
    """min Q over the LP feasible set with slot loads <= Q and the cover cuts at level P (units).
    Returns (Q in units, alpha), or (None, None) if the cuts leave nothing feasible."""
    import gurobipy as gp
    from gurobipy import GRB
    md = gp.Model(env=env)
    md.Params.OutputFlag = 0
    if vertex:
        md.Params.Method = 0
    x = {(i, int(t)): md.addVar(lb=0.0, ub=1.0) for i in range(inst.n) for t in inst.win[i]}
    Q = md.addVar(lb=0.0)
    md.setObjective(Q, GRB.MINIMIZE)
    for i in range(inst.n):
        md.addConstr(gp.quicksum(x[i, int(t)] for t in inst.win[i]) == int(inst.p[i]))
    for t in range(inst.T):
        if inst.avail[t]:
            md.addConstr(gp.quicksum(int(inst.wu[i]) * x[i, t] for i in inst.avail[t]) <= Q)
    for t, S, k in cover_cuts(inst, P, kmax):
        md.addConstr(gp.quicksum(x[i, t] for i in S) <= k)
    md.optimize()
    if md.Status != GRB.OPTIMAL:
        return None, None
    alpha = np.zeros((inst.n, inst.T))
    for (i, t), v in x.items():
        alpha[i, t] = v.X
    return Q.X, alpha


def cut_lower_bound(cert, ub_kw, env=None, kmax=None):
    """LB_cut: the smallest attainable load P such that the LP with peak <= P and the cover cuts
    at level P is feasible. Feasibility is monotone in P, so a binary search over the attainable
    loads between ceil(P_LP) and a known upper bound (any feasible schedule's peak) finds it.
    Returns (LB_cut in kW, alpha_cut: a vertex of the cut LP at that level, its value Q in kW)."""
    inst, A = cert.inst, cert.attainable_loads()
    lo = int(np.searchsorted(A, int(np.ceil(cert.p_lp * UNIT - 1e-6))))
    hi = int(np.searchsorted(A, int(np.floor(ub_kw * UNIT + 1e-6)), side="right")) - 1

    def ok(j):
        q, _ = cut_lp(inst, int(A[j]), env, kmax=kmax)
        return q is not None and q <= A[j] + 1e-6

    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    q, alpha = cut_lp(inst, int(A[lo]), env, vertex=True, kmax=kmax)
    return A[lo] / UNIT, alpha, q / UNIT


# ── dependent rounding by power class ───────────────────────────────────────────

def power_classes(inst):
    pw = sorted(set(int(v) for v in inst.wu), reverse=True)
    return pw, np.array([pw.index(int(v)) for v in inst.wu])


class DRCertificates(Certificates):
    """Certificates for bipartite dependent rounding (Gandhi, Khuller, Parthasarathy & Srinivasan
    2006) run separately on each power class: every vehicle keeps exactly p_i slots, and the
    number of class-k vehicles at slot t is floor(y_kt) or ceil(y_kt), y_kt = sum_{i in k} alpha_it,
    with P(ceil) = frac(y_kt). Slot loads are therefore sums of at most K two-point variables,
    independent across classes; components of G_alpha stay independent. The union, componentwise
    and averaging certificates apply unchanged; the local lemma does not (long-range dependence)."""

    def __init__(self, inst, alpha, p_lp=None, tol=1e-7):
        super().__init__(inst, alpha, p_lp)
        pw, cls = power_classes(inst)
        T, K = inst.T, len(pw)
        y = np.zeros((K, T))
        for t in range(T):
            for i in inst.avail[t]:
                y[cls[i], t] += alpha[i, t]
        y = np.where(np.abs(y - np.rint(y)) < tol, np.rint(y), y)
        fl = np.floor(y)
        fr = y - fl
        self.y, self.pw = y, np.array(pw)
        self.ceiling_load = (self.pw[:, None] * np.ceil(y)).sum(axis=0)
        u = np.arange(self.H + 1)
        surv = np.zeros((T, self.H + 1))
        for t in range(T):
            base = int((self.pw * fl[:, t]).sum())
            law = B.bernoulli_sum_law([(int(pw[k]), float(fr[k, t])) for k in range(K) if fr[k, t] > 0])
            sR = np.concatenate([np.cumsum(law[::-1])[::-1], [0.0]])
            k = u - base + 1
            surv[t] = np.where(k <= 0, 1.0, sR[np.clip(k, 0, len(law))])
        self.surv = surv

    def ceiling(self):
        """Deterministic: some schedule has every class count rounded up at most, so
        P* <= max_t sum_k w_k ceil(y_kt)."""
        return self.ceiling_load.max() / UNIT

    def lll(self):
        raise NotImplementedError("dependent rounding couples whole components: use the componentwise bounds")

    def all(self):
        return {"dr_union": self.union(), "dr_component_union": self.component_union(),
                "dr_averaging": self.averaging(), "dr_component_averaging": self.component_averaging(),
                "dr_ceiling": self.ceiling()}


def dependent_round(inst, alpha, rng, tol=1e-9):
    """Bipartite dependent rounding of alpha, separately on each power class (vehicle nodes i,
    slot nodes (class, t)). Repeatedly picks a cycle or a maximal path of fractional edges,
    splits it into alternate sets M1, M2, and moves by +a/-a or -b/+b with probabilities
    b/(a+b) and a/(a+b): marginals are preserved, every vehicle keeps p_i slots, every slot node
    ends at floor or ceil of its fractional degree. Returns an integral schedule."""
    _, cls = power_classes(inst)
    x = alpha.copy()
    adj = {}
    for i in range(inst.n):
        for t in inst.win[i]:
            if tol < x[i, t] < 1 - tol:
                adj.setdefault(("v", i), set()).add(("s", int(cls[i]), int(t)))
                adj.setdefault(("s", int(cls[i]), int(t)), set()).add(("v", i))

    def edge(a, b):
        return (a[1], b[2]) if a[0] == "v" else (b[1], a[2])

    def drop(i, t):
        a, b = ("v", i), ("s", int(cls[i]), t)
        adj[a].discard(b); adj[b].discard(a)
        if not adj[a]: del adj[a]
        if not adj[b]: del adj[b]

    # connected components of the fractional support, rounded one after the other, so that the
    # outcome of a component depends only on its own coin flips (independence across components).
    # Sets are visited in sorted order: Python randomizes the hashes of strings in every process, and
    # the same random stream must give the same rounding in every run
    comps, seen = [], set()
    for v0 in list(adj):
        if v0 in seen:
            continue
        stack, comp = [v0], set()
        seen.add(v0)
        while stack:
            v = stack.pop(); comp.add(v)
            for nb in sorted(adj[v]):
                if nb not in seen:
                    seen.add(nb); stack.append(nb)
        comps.append(comp)
    queue = list(comps)
    while queue:
        comp = queue[0]
        live = sorted(v for v in comp if v in adj)
        if not live:
            queue.pop(0)
            continue
        start = next((v for v in live if len(adj[v]) == 1), live[0])
        walk, pos, used = [start], {start: 0}, set()
        cycle = None
        while True:
            cur = walk[-1]
            nxt = next((v for v in sorted(adj[cur]) if frozenset((cur, v)) not in used), None)
            if nxt is None:
                break
            used.add(frozenset((cur, nxt)))
            if nxt in pos:
                cycle = walk[pos[nxt]:] + [nxt]
                break
            pos[nxt] = len(walk)
            walk.append(nxt)
        nodes = cycle if cycle is not None else walk
        es = [edge(nodes[j], nodes[j + 1]) for j in range(len(nodes) - 1)]
        m1, m2 = es[0::2], es[1::2]
        a = min([1 - x[e] for e in m1] + [x[e] for e in m2])
        b = min([x[e] for e in m1] + [1 - x[e] for e in m2])
        if rng.random() < b / (a + b):
            for e in m1: x[e] += a
            for e in m2: x[e] -= a
        else:
            for e in m1: x[e] -= b
            for e in m2: x[e] += b
        for (i, t) in es:
            if x[i, t] <= tol or x[i, t] >= 1 - tol:
                x[i, t] = round(x[i, t])
                drop(i, t)
    return np.rint(x).astype(int)


def dr_las_vegas(drc, u_kw, rng, max_rounds=10 ** 4):
    """Resample the dependent rounding until every slot is <= u (expected rounds <= 1/(1 - sum_t P(S_t > u))
    under the union certificate). Returns schedule, peak, rounds."""
    u = int(round(u_kw * UNIT))
    for r in range(1, max_rounds + 1):
        x = dependent_round(drc.inst, drc.alpha, rng)
        loads = (drc.inst.wu[:, None] * x).sum(axis=0)
        if loads.max() <= u:
            return x, loads.max() / UNIT, r
    return x, loads.max() / UNIT, max_rounds


# ── configuration LP lower bound (Dantzig-Wolfe on the slot knapsacks) ──────────

def _knapsack(values, weights, cap):
    """0-1 knapsack, integer weights: (best value, chosen indices) with total weight <= cap."""
    best = np.zeros(cap + 1)
    take = np.zeros((len(values), cap + 1), bool)
    for j, (v, w) in enumerate(zip(values, weights)):
        if w > cap or v <= 0:
            continue
        cand = np.full(cap + 1, -np.inf)
        cand[w:] = best[:cap + 1 - w] + v
        take[j] = cand > best + 1e-12
        best = np.maximum(best, cand)
    c, chosen = cap, []
    for j in range(len(values) - 1, -1, -1):
        if take[j, c]:
            chosen.append(j)
            c -= weights[j]
    return best[cap], chosen


def config_lp(inst, P, env=None, pool=None, tol=1e-9, max_iter=10 ** 4):
    """Is the configuration LP feasible at level P (units)? Slot t picks a convex combination of
    subsets S of I(t) with w(S) <= P; vehicle i must be covered p_i times. Phase I by column
    generation (pricing = 0-1 knapsack per slot). Returns (feasible, alpha or None, pool)."""
    import gurobipy as gp
    from gurobipy import GRB
    pool = [] if pool is None else [(t, S) for t, S in pool if inst.wu[list(S)].sum() <= P]
    md = gp.Model(env=env)
    md.Params.OutputFlag = 0
    s = [md.addVar(lb=0.0, obj=1.0) for _ in range(inst.n)]
    cv = [md.addConstr(s[i] >= int(inst.p[i])) for i in range(inst.n)]
    cs = {t: md.addConstr(gp.LinExpr() <= 1) for t in range(inst.T) if inst.avail[t]}
    lam = []

    def add(t, S):
        col = gp.Column([1.0] * len(S) + [1.0], [cv[i] for i in S] + [cs[t]])
        lam.append((t, tuple(S), md.addVar(lb=0.0, obj=0.0, column=col)))
    seen = set()
    for t, S in pool:
        if (t, S) not in seen:
            seen.add((t, S)); add(t, S)
    for t in cs:
        for i in inst.avail[t]:
            if inst.wu[i] <= P and (t, (i,)) not in seen:
                seen.add((t, (i,))); add(t, (i,))
    for _ in range(max_iter):
        md.optimize()
        pi = np.array([c.Pi for c in cv])
        new = 0
        for t, c in cs.items():
            I = inst.avail[t]
            val, ch = _knapsack(pi[I], [int(inst.wu[i]) for i in I], int(P))
            if val + c.Pi > tol:                      # reduced cost -(sum pi) - mu_t < 0
                S = tuple(sorted(I[j] for j in ch))
                if (t, S) not in seen:
                    seen.add((t, S)); add(t, S); new += 1
        if not new:
            break
    feasible = md.ObjVal <= 1e-7
    alpha = None
    if feasible:
        alpha = np.zeros((inst.n, inst.T))
        for t, S, v in lam:
            if v.X > 1e-12:
                alpha[list(S), t] += v.X
        for i in range(inst.n):                       # trim over-coverage (subsets of configurations stay feasible)
            alpha[i] = np.minimum(alpha[i], 1.0)
            excess = alpha[i].sum() - inst.p[i]
            for t in np.argsort(-alpha[i]):
                if excess <= 1e-12:
                    break
                d = min(excess, alpha[i, t]); alpha[i, t] -= d; excess -= d
    return feasible, alpha, [(t, S) for t, S, _ in lam]


def config_lower_bound(cert, lo_kw, ub_kw, env=None):
    """LB_conf: smallest attainable load P at which the configuration LP is feasible, searched
    between lo (e.g. LB_cut, which it dominates) and a feasible schedule's peak.
    Returns (LB_conf in kW, alpha_conf: its fractional schedule)."""
    inst, A = cert.inst, cert.attainable_loads()
    lo = int(np.searchsorted(A, int(np.ceil(lo_kw * UNIT - 1e-6))))
    hi = int(np.searchsorted(A, int(np.floor(ub_kw * UNIT + 1e-6)), side="right")) - 1
    pool, best_alpha = [], None
    while lo < hi:
        mid = (lo + hi) // 2
        ok, al, pool = config_lp(inst, int(A[mid]), env, pool)
        if ok:
            hi, best_alpha = mid, al
        else:
            lo = mid + 1
    if best_alpha is None:
        ok, best_alpha, pool = config_lp(inst, int(A[lo]), env, pool)
    return A[lo] / UNIT, best_alpha


# ── exact computations on the fractional support of an LP vertex (pseudo-forest DP) ──

def systematic_outcomes(slots, probs):
    """Exact outcome distribution of systematic sampling: list of (frozenset of chosen slots, probability)."""
    r = int(round(float(np.sum(probs))))
    c = np.concatenate([[0.0], np.cumsum(probs)])
    c *= r / c[-1]                      # remove rounding drift: the last cut point is exactly r
    br = np.unique(np.concatenate([[0.0, 1.0], np.mod(c[:-1], 1.0)]))
    out = {}
    for lo, hi in zip(br[:-1], br[1:]):
        if hi - lo <= 1e-15:
            continue
        U = 0.5 * (lo + hi)
        pts = U + np.arange(r)
        S = frozenset(int(slots[k]) for k in range(len(slots)) if np.any((pts >= c[k]) & (pts < c[k + 1])))
        if len(S) != r:
            raise RuntimeError("systematic outcome of the wrong size")
        out[S] = out.get(S, 0.0) + (hi - lo)
    return list(out.items())


def compatible_outcomes(slots, probs):
    """All subsets of the fractional slots with the right size (uniform weights; for existence only)."""
    from itertools import combinations
    r = int(round(sum(probs)))
    if math.comb(len(slots), r) > 4096:
        raise RuntimeError("too many compatible subsets")
    subs = [frozenset(int(s) for s in S) for S in combinations(slots, r)]
    return [(S, 1.0 / len(subs)) for S in subs]


class ForestDP:
    """Exact quantities for the strategy "integral entries of alpha fixed, each random vehicle drawn
    independently from its option list", by dynamic programming over the fractional support graph
    (vehicles - slots). At an LP vertex every connected component has at most one cycle (pseudo-forest);
    cycles are broken by conditioning on one vehicle per cycle.
      mode systematic: options = outcomes of systematic sampling (exact probabilities)
      mode compatible: options = all subsets of the right size (existence = best rounding of alpha)"""

    def __init__(self, cert, mode="systematic", max_cut=4096):
        self.cert, self.inst, self.mode, self.max_cut = cert, cert.inst, mode, max_cut
        al = cert.alpha
        gen = systematic_outcomes if mode == "systematic" else compatible_outcomes
        self.opts = {i: gen(np.array(sl), al[i, sl]) for i, sl in cert.frac_slots.items()}
        self.w = {i: int(self.inst.wu[i]) for i in cert.frac_slots}
        self.D = cert.laws.det.astype(int)
        self.comps = []
        for comp in cert.random_comps:
            cs = set(comp)
            veh = sorted(i for i, sl in cert.frac_slots.items() if sl[0] in cs)
            self.comps.append((sorted(comp), veh))
        rnd = {t for comp, _ in self.comps for t in comp}
        self.det_max = max([int(self.D[t]) for t in range(self.inst.T) if t not in rnd], default=0)
        self.cut_sizes = []

    # --- one forest (no cycles) given fixed extra loads ---
    def _forest(self, slots, veh, extra, u, boolean):
        fs = self.cert.frac_slots
        vset = set(veh)
        adj_t = {t: [] for t in slots}
        for i in veh:
            for t in fs[i]:
                adj_t[t].append(i)
        seen_t, seen_v, total = set(), set(), 1.0
        for root in slots:
            if root in seen_t:
                continue
            order, stack = [], [("t", root, None)]
            seen_t.add(root)
            while stack:
                kind, node, par = stack.pop()
                order.append((kind, node, par))
                if kind == "t":
                    for i in adj_t[node]:
                        if i not in seen_v:
                            seen_v.add(i)
                            stack.append(("v", i, node))
                else:
                    for t in fs[node]:
                        if t not in seen_t:
                            seen_t.add(t)
                            stack.append(("t", t, node))
            g, h, kids = {}, {}, {}
            for kind, node, par in order:
                if par is not None:
                    kids.setdefault((("t" if kind == "v" else "v"), par), []).append(node)
            for kind, node, par in reversed(order):
                if kind == "v":
                    hv = np.zeros(2)
                    for S, pw in self.opts[node]:
                        val = 1.0 if boolean else pw
                        for tc in kids.get(("v", node), []):
                            val *= g[tc][1 if tc in S else 0]
                            if val == 0.0:
                                break
                        hv[1 if par in S else 0] += val
                    h[node] = np.minimum(hv, 1.0) if boolean else hv
                else:
                    arr = np.ones(1)
                    for c in kids.get(("t", node), []):
                        wc = self.w[c]
                        new = np.zeros(len(arr) + wc)
                        new[:len(arr)] += arr * h[c][0]
                        new[wc:] += arr * h[c][1]
                        arr = np.minimum(new, 1.0) if boolean else new
                    cs = np.concatenate([[0.0], np.cumsum(arr)])
                    cap = u - int(self.D[node]) - int(extra.get(node, 0))

                    def upto(k):
                        return 0.0 if k < 0 else float(cs[min(k + 1, len(arr))])
                    if par is None:
                        val = upto(cap)
                        total *= min(val, 1.0) if boolean else val
                    else:
                        wp = self.w[par]
                        gt = np.array([upto(cap), upto(cap - wp)])
                        g[node] = np.minimum(gt, 1.0) if boolean else gt
            if total == 0.0:
                return 0.0
        return total

    def _cycle_cut(self, veh):
        """Vehicles whose removal makes the support acyclic (greedy: one vehicle per cycle found)."""
        fs = self.cert.frac_slots
        cut, rest = [], list(veh)
        while True:
            adj = {}
            for i in rest:
                for t in fs[i]:
                    adj.setdefault(("v", i), []).append(("t", t))
                    adj.setdefault(("t", t), []).append(("v", i))
            visited, found = set(), None
            for s in adj:
                if s in visited:
                    continue
                stack = [(s, None)]
                while stack and found is None:
                    node, par = stack.pop()
                    if node in visited:
                        found = node if node[0] == "v" else par
                        break
                    visited.add(node)
                    for nb in adj[node]:
                        if nb != par:
                            stack.append((nb, node))
                if found is not None:
                    break
            if found is None:
                return cut
            cut.append(found[1])
            rest.remove(found[1])

    def comp_value(self, comp, u, boolean):
        from itertools import product
        slots, veh = comp
        if not hasattr(self, "_cuts"):
            self._cuts = {}
        key = tuple(veh)
        if key not in self._cuts:
            self._cuts[key] = self._cycle_cut(veh)
            self.cut_sizes.append(len(self._cuts[key]))
        cut = self._cuts[key]
        rest = [i for i in veh if i not in cut]
        if not cut:
            return self._forest(slots, rest, {}, u, boolean)
        combos = 1
        for i in cut:
            combos *= len(self.opts[i])
        if combos > self.max_cut:
            raise RuntimeError("cycle cutset too large")
        total = 0.0
        for choice in product(*[self.opts[i] for i in cut]):
            extra, wt = {}, 1.0
            for i, (S, pw) in zip(cut, choice):
                wt *= 1.0 if boolean else pw
                for t in S:
                    extra[t] = extra.get(t, 0) + self.w[i]
            v = self._forest(slots, rest, extra, u, boolean)
            total += wt * v
            if boolean and total > 0:
                return 1.0
        return total

    def prob_ok(self, u):
        """P(max_t S_t <= u) exactly (u in units)."""
        if u < self.det_max:
            return 0.0
        p = 1.0
        for comp in self.comps:
            p *= self.comp_value(comp, u, False)
            if p == 0.0:
                break
        return p

    def exists(self, u):
        if u < self.det_max:
            return False
        return all(self.comp_value(comp, u, True) > 0 for comp in self.comps)

    def threshold(self):
        """Smallest u with P(max <= u) > 0: the exact existence certificate of the strategy
        (for mode compatible: the peak of the best rounding of alpha)."""
        lo, hi = max(self.det_max, self.cert.lo), self.cert.H
        while lo < hi:
            mid = (lo + hi) // 2
            if self.exists(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo / UNIT

    def expected_max(self):
        """E[max_t S_t] exactly: sum over integer u of P(max > u) (units)."""
        lo = max(self.det_max, 0)
        tot = float(lo)
        for u in range(lo, self.cert.H + 1):
            q = 1.0 - self.prob_ok(u)
            if q <= 1e-15:
                break
            tot += q
        return tot / UNIT
