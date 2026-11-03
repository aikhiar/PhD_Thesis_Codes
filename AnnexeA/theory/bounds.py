"""
bounds.py -- probabilistic upper bounds on the optimal peak power P* of the PPMSP
(Section 4 of the paper).

A sampling strategy draws the schedule of every vehicle independently, with
marginals alpha[i, t] = P(X_it = 1). Every bound below is valid for ANY alpha in the
LP feasible set (row sums p_i, entries in [0, 1], zero outside the window), because
each row of such an alpha is a convex combination of p_i-subsets of the window.

Non-localized bounds (Corollary 4.12 of the paper):
    hoeffding_union, hoeffding_lll, bernstein_union, bernstein_lll
New bounds:
    local_hoeffding_union, local_bernstein_union    per-slot variances, one threshold u
    exact_union                                      exact per-slot tails, union bound
    exact_lll                                        exact tails, asymmetric local lemma,
                                                     dependency through random vehicles only
    expected_max                                     averaging: P* <= E[max_t S_t]
                                                     <= min_u u + sum_t E[(S_t - u)^+]
Algorithms (method of conditional expectations; deterministic, polynomial):
    derandomize(..., estimator="excess")  peak <= the expected_max bound at its minimizer u
    derandomize(..., estimator="union")   peak <= u whenever sum_t P(S_t > u) < 1

Powers are integers in units of 0.1 kW, so slot loads have exact discrete laws.
"""

import math

import numpy as np

UNIT = 10  # 1 unit = 0.1 kW


# ── instance and LP ─────────────────────────────────────────────────────────────

class Instance:
    def __init__(self, tup):
        _, s, e, p, w, n, m, T, _ = tup
        self.n, self.T = int(n), int(T)
        self.s, self.e = np.asarray(s, int) - 1, np.asarray(e, int) - 1     # 0-based, inclusive
        self.p = np.asarray(p, int)
        self.wu = np.rint(np.asarray(w, float) * UNIT).astype(int)          # power in units
        self.w = self.wu / UNIT                                               # power in kW
        self.win = [np.arange(self.s[i], self.e[i] + 1) for i in range(self.n)]
        self.avail = [[i for i in range(self.n) if self.s[i] <= t <= self.e[i]] for t in range(self.T)]


def solve_lp(inst, env=None):
    """LP relaxation solved by primal simplex (vertex solution). Returns (P_LP in kW, alpha)."""
    import gurobipy as gp
    from gurobipy import GRB
    md = gp.Model(env=env)
    md.Params.OutputFlag = 0
    md.Params.Method = 0
    x = {(i, t): md.addVar(lb=0.0, ub=1.0) for i in range(inst.n) for t in inst.win[i]}
    P = md.addVar(lb=0.0)
    md.setObjective(P, GRB.MINIMIZE)
    for i in range(inst.n):
        md.addConstr(gp.quicksum(x[i, t] for t in inst.win[i]) == int(inst.p[i]))
    for t in range(inst.T):
        if inst.avail[t]:
            md.addConstr(gp.quicksum(inst.w[i] * x[i, t] for i in inst.avail[t]) <= P)
    md.optimize()
    alpha = np.zeros((inst.n, inst.T))
    for (i, t), v in x.items():
        alpha[i, t] = v.X
    alpha[alpha < 1e-9] = 0.0
    alpha[alpha > 1 - 1e-9] = 1.0
    return P.X, alpha


def uniform_alpha(inst):
    alpha = np.zeros((inst.n, inst.T))
    for i in range(inst.n):
        alpha[i, inst.win[i]] = inst.p[i] / len(inst.win[i])
    return alpha


# ── exact slot laws ─────────────────────────────────────────────────────────────

def bernoulli_sum_law(items):
    """Law of sum_k w_k B_k for independent Bernoulli(a_k): array P[value]."""
    law = np.ones(1)
    for wk, ak in items:
        new = np.zeros(len(law) + wk)
        new[:len(law)] += (1.0 - ak) * law
        new[wk:] += ak * law
        law = new
    return law


def remove_bernoulli(law, wk, ak):
    """Law of R' given law of R' + w B(a): stable recursion (forward if a <= 1/2, else backward)."""
    L = len(law) - wk
    out = np.zeros(L)
    if ak <= 0.5:
        for k in range(L):
            out[k] = (law[k] - (ak * out[k - wk] if k >= wk else 0.0)) / (1.0 - ak)
    else:
        for k in range(len(law) - 1, wk - 1, -1):
            hi = out[k] if k < L else 0.0
            out[k - wk] = (law[k] - (1.0 - ak) * hi) / ak
    out = np.clip(out, 0.0, None)
    return out / out.sum()


class SlotLaws:
    """Per slot t: deterministic load det[t] (units), random part with exact law, moments."""

    def __init__(self, inst, alpha, tol=1e-9):
        self.inst, self.alpha = inst, alpha
        T = inst.T
        self.det = np.zeros(T, int)
        self.rand = [[] for _ in range(T)]          # (i, w_units, a) with 0 < a < 1
        for t in range(T):
            for i in inst.avail[t]:
                a = alpha[i, t]
                if a >= 1 - tol:
                    self.det[t] += inst.wu[i]
                elif a > tol:
                    self.rand[t].append((i, int(inst.wu[i]), float(a)))
        self.law = [bernoulli_sum_law([(wk, ak) for _, wk, ak in self.rand[t]]) for t in range(T)]
        self.mu = np.array([self.det[t] + sum(wk * ak for _, wk, ak in self.rand[t]) for t in range(T)])  # units
        self.var = np.array([sum(wk * wk * ak * (1 - ak) for _, wk, ak in self.rand[t]) for t in range(T)])
        self.rng = np.array([max([wk * (1 - ak) for _, wk, ak in self.rand[t]], default=0.0) for t in range(T)])
        self.sq = np.array([sum(wk * wk for _, wk, _ in self.rand[t]) for t in range(T)], float)
        self.maxload = np.array([self.det[t] + len(self.law[t]) - 1 for t in range(T)])
        # dependency through random vehicles only: N(t) = {t' != t : V_t and V_t' share a vehicle}
        frac_slots = [set() for _ in range(inst.n)]
        for t in range(T):
            for i, _, _ in self.rand[t]:
                frac_slots[i].add(t)
        self.nbr = []
        for t in range(T):
            nb = set()
            for i, _, _ in self.rand[t]:
                nb |= frac_slots[i]
            nb.discard(t)
            self.nbr.append(np.array(sorted(nb), int))

    def tail_gt(self, u):
        """P(S_t > u) for every slot (u in units, integer)."""
        out = np.zeros(self.inst.T)
        for t in range(self.inst.T):
            k = u - self.det[t] + 1            # need R >= k
            law = self.law[t]
            if k <= 0:
                out[t] = 1.0
            elif k < len(law):
                out[t] = law[k:].sum()
        return out

    def excess_curve(self, umax):
        """E[(S_t - u)^+] summed over t, for every integer u in 0..umax."""
        total = np.zeros(umax + 1)
        u = np.arange(umax + 1)
        for t in range(self.inst.T):
            law = self.law[t]
            d = int(self.det[t])
            P = np.zeros(max(umax + 2, d + len(law)))
            P[d:d + len(law)] = law
            v = np.arange(len(P))
            mass = np.cumsum(P[::-1])[::-1]               # mass[k] = P(S >= k)
            mom = np.cumsum((v * P)[::-1])[::-1]          # mom[k]  = E[S; S >= k]
            total += mom[u + 1] - u * mass[u + 1]        # E[(S - u)^+] = E[S - u; S > u]
        return total


# ── non-localized bounds (Corollary 4.12) ───────────────────────────────────────

def dependency_degree_all(inst):
    d = 0
    for t in range(inst.T):
        nb = set()
        for i in inst.avail[t]:
            nb.update(inst.win[i].tolist())
        nb.discard(t)
        d = max(d, len(nb))
    return d


def old_bounds(inst, alpha):
    w = inst.w
    mu = (w[:, None] * alpha).sum(axis=0)
    base = mu.max()
    sumw2 = float((w ** 2).sum())
    wmax = float(w.max())
    V = float(((w[:, None] ** 2) * alpha * (1 - alpha)).sum(axis=0).max())
    d = dependency_degree_all(inst)
    out = {"d_all": d}
    for name, L in (("union", math.log(inst.T)), ("lll", math.log(math.e * max(d, 1)))):
        out[f"hoeffding_{name}"] = base + math.sqrt(sumw2 * L / 2.0)
        out[f"bernstein_{name}"] = base + wmax * L / 3.0 * (1.0 + math.sqrt(1.0 + 18.0 * V / (wmax ** 2 * L)))
    return out


# ── new bounds ──────────────────────────────────────────────────────────────────

def _smallest_u(lo, hi, ok):
    """Smallest integer u in [lo, hi] with ok(u) (ok monotone); hi must satisfy ok."""
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def lll_holds(p, nbr, iters=5000, slack=1e-6):
    """Asymmetric local lemma: is there x in [0,1) with p_t <= x_t prod_{t' in N(t)} (1 - x_t')?
    Least fixed point of x <- q / prod(1 - x) by monotone iteration, for the tails q = (1 + slack) p;
    True only if the limit is then checked to satisfy the condition for p itself (an explicit
    certificate: floating-point convergence alone does not prove that a fixed point exists)."""
    if (p >= 1 - 1e-9).any():
        return False
    act = np.where(p > 0)[0]
    if len(act) == 0:
        return True
    q = p * (1.0 + slack)
    x = q.copy()
    for _ in range(iters):
        logs = np.zeros_like(x)
        for t in act:
            nb = nbr[t]
            if len(nb):
                xs = x[nb]
                if (xs >= 1).any():
                    return False
                logs[t] = np.log1p(-xs).sum()
        new = np.where(p > 0, q * np.exp(-logs), 0.0)
        if (new >= 1 - 1e-9).any():
            return False
        if np.abs(new - x).max() < 1e-13:
            for t in act:                                  # explicit check of the condition at x = new
                nb = nbr[t]
                prod = np.exp(np.log1p(-new[nb]).sum()) if len(nb) else 1.0
                if not p[t] <= new[t] * prod:
                    return False
            return True
        x = new
    return False


def new_bounds(laws):
    T = laws.inst.T
    lo = int(np.floor(laws.mu.max()))
    hi = int(laws.maxload.max())
    out = {}

    def p_local(u, kind):
        p = np.zeros(T)
        for t in range(T):
            eps = u - laws.mu[t]
            if laws.var[t] == 0:
                p[t] = 0.0 if u >= laws.det[t] else 1.0
            elif eps <= 0:
                p[t] = 1.0
            elif kind == "hoeffding":
                p[t] = math.exp(-2.0 * eps * eps / laws.sq[t])
            else:
                p[t] = math.exp(-eps * eps / (2.0 * laws.var[t] + 2.0 * laws.rng[t] * eps / 3.0))
        return p

    for kind in ("hoeffding", "bernstein"):
        top = hi * 4 + 10
        out[f"local_{kind}_union"] = _smallest_u(lo, top, lambda u: p_local(u, kind).sum() < 1 - 1e-9) / UNIT
        out[f"local_{kind}_lll"] = _smallest_u(lo, top, lambda u: lll_holds(p_local(u, kind), laws.nbr)) / UNIT

    # closed forms: Theorems 4.10 / 4.13 with local quantities (random vehicles only)
    random_slots = [t for t in range(T) if laws.rand[t]]
    base = laws.mu.max() / UNIT
    d_r = max((len(nb) for nb in laws.nbr), default=0)
    if random_slots:
        W = laws.sq[random_slots].max() / UNIT ** 2
        V = laws.var[random_slots].max() / UNIT ** 2
        M = laws.rng[random_slots].max() / UNIT
        for name, L in (("union", math.log(max(len(random_slots), 2))), ("lll", math.log(math.e * max(d_r, 1)))):
            out[f"closed_hoeffding_{name}"] = base + math.sqrt(W * L / 2.0)
            out[f"closed_bernstein_{name}"] = base + M * L / 3.0 * (1.0 + math.sqrt(1.0 + 18.0 * V / (M * M * L)))
    else:
        for key in ("closed_hoeffding_union", "closed_hoeffding_lll", "closed_bernstein_union", "closed_bernstein_lll"):
            out[key] = base
    out["random_slots"] = len(random_slots)
    out["exact_union"] = _smallest_u(lo, hi, lambda u: laws.tail_gt(u).sum() < 1 - 1e-9) / UNIT
    out["exact_lll"] = _smallest_u(lo, hi, lambda u: lll_holds(laws.tail_gt(u), laws.nbr)) / UNIT
    curve = laws.excess_curve(hi + 1)
    f = np.arange(hi + 2) + curve
    u_star = int(np.argmin(f[lo:]) + lo)
    out["expected_max"] = f[u_star] / UNIT
    out["u_star_expected_max"] = u_star
    out["d_random"] = int(max((len(nb) for nb in laws.nbr), default=0))
    return out


# ── derandomization (method of conditional expectations) ────────────────────────

def derandomize(inst, alpha, u, estimator="excess", order=None, passes=0):
    """Fix the vehicles one at a time; each takes the p_i slots of its window that minimize
    the conditional value of the estimator. The estimator never increases, so
    'excess': final peak <= u + initial sum_t E[(S_t - u)^+]
    'union' : final peak <= u when initial sum_t P(S_t > u) < 1.
    Extra 'passes' re-optimize every vehicle against the others (still non-increasing)."""
    laws = SlotLaws(inst, alpha)
    det = laws.det.astype(float).copy()
    rand = [dict((i, (wk, ak)) for i, wk, ak in laws.rand[t]) for t in range(inst.T)]
    law = [l.copy() for l in laws.law]
    x = np.zeros((inst.n, inst.T), int)

    def value(c, lw):
        vals = c + np.arange(len(lw))
        if estimator == "excess":
            return float((lw * np.clip(vals - u, 0, None)).sum())
        return float(lw[vals > u].sum())

    if order is None:
        order = np.argsort(-inst.wu, kind="stable")
    seq = list(order) + [i for _ in range(passes) for i in order]
    fixed = np.zeros(inst.n, bool)
    for i in seq:
        wi = int(inst.wu[i])
        slots = inst.win[i]
        dminus, lminus = [], []
        for t in slots:
            if not fixed[i] and i in rand[t]:
                wk, ak = rand[t][i]
                rest = [(w2, a2) for j, (w2, a2) in rand[t].items() if j != i]
                lm = bernoulli_sum_law(rest) if len(rest) <= 12 else remove_bernoulli(law[t], wk, ak)
                dminus.append(det[t]); lminus.append(lm)
            else:
                cur = x[i, t] if fixed[i] else (1 if alpha[i, t] >= 1 - 1e-9 else 0)
                dminus.append(det[t] - wi * cur); lminus.append(law[t])
        delta = np.array([value(dm + wi, lm) - value(dm, lm) for dm, lm in zip(dminus, lminus)])
        # ties: prefer the slot with the lowest current load
        key = np.lexsort((np.array(dminus), delta))
        chosen = set(slots[key[:inst.p[i]]].tolist())
        for k, t in enumerate(slots):
            on = 1 if t in chosen else 0
            det[t] = dminus[k] + wi * on
            law[t] = lminus[k]
            rand[t].pop(i, None)
            x[i, t] = on
        fixed[i] = True
    loads = (inst.wu[:, None] * x).sum(axis=0)
    return x, loads.max() / UNIT
