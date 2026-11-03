"""
verify_theory.py -- end-to-end numerical verification of every result of Sections 3 and 4 of
the paper (the statements are numbered as in the paper).

Each check is tied to one statement and reports the number of tests and of failures. Exact
optima come from the class-aggregated MILP (exact.py) and, for the aggregation theorem itself,
from the original binary model. Monte Carlo checks use vectorized samplers and z-scores
restricted to levels where the normal approximation is valid. The closed forms are recomputed
here from the formulas of the paper, independently of certificates.py.

Run from this folder:  python verify_theory.py [--quick] [--workers N] [--out DIR]
(parallel by default: one worker process per physical core but one; --workers 1 = sequential)
Writes theory_verification.md and theory_verification.csv to DIR (default: results/ next to this file;
run_all.bat passes the campaign's 05_logs folder, so nothing is written inside the code).
"""

import itertools
import math
import os
import sys
import time

import numpy as np
import pandas as pd

import bounds as B
import certificates as C
import exact as X
import validate as V

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
EPS = 1e-6
REPORT = []
QUICK = "--quick" in sys.argv


def record(tag, statement, tests, fails, detail=""):
    REPORT.append({"result": tag, "statement": statement, "tests": int(tests), "failures": int(fails), "detail": detail})
    print(f"[{'PASS' if fails == 0 else 'FAIL'}] {tag:14s} {statement}  tests={tests} failures={fails}  {detail}", flush=True)


def make(s, e, p, w, T):
    n = len(s)
    return B.Instance((np.ones(n, int), np.array(s), np.array(e), np.array(p), np.array(w, float), n, 1, T, 0.0))


def xform_exact(inst, env, limit):
    import gurobipy as gp
    from gurobipy import GRB
    md = gp.Model(env=env)
    md.Params.OutputFlag = 0; md.Params.TimeLimit = limit; md.Params.MIPGap = 0.0
    x = {(i, int(t)): md.addVar(vtype=GRB.BINARY) for i in range(inst.n) for t in inst.win[i]}
    P = md.addVar(lb=0.0)
    md.setObjective(P, GRB.MINIMIZE)
    for i in range(inst.n):
        md.addConstr(gp.quicksum(x[i, int(t)] for t in inst.win[i]) == int(inst.p[i]))
    for t in range(inst.T):
        if inst.avail[t]:
            md.addConstr(gp.quicksum(int(inst.wu[i]) * x[i, t] for i in inst.avail[t]) <= P)
    md.optimize()
    return (md.ObjVal / B.UNIT if md.SolCount else np.nan), md.Status == GRB.OPTIMAL


# ── vectorized samplers ─────────────────────────────────────────────────────────

def frac_rows(inst, alpha, tol=1e-9):
    rows = []
    for i in range(inst.n):
        sl = np.array([t for t in inst.win[i] if tol < alpha[i, t] < 1 - tol], int)
        if len(sl):
            pr = alpha[i, sl]
            r = int(round(pr.sum()))
            c = np.concatenate([[0.0], np.cumsum(pr)]); c *= r / c[-1]
            rows.append((i, sl, c, r))
    return rows


def det_loads(inst, alpha, tol=1e-9):
    return (inst.wu[:, None] * (alpha >= 1 - tol)).sum(axis=0).astype(np.int64)


def sample_loads(inst, alpha, R, rng):
    """R samples of the slot loads (units) under the systematic strategy of alpha."""
    L = np.tile(det_loads(inst, alpha), (R, 1))
    rows_idx = np.arange(R)[:, None]
    for i, sl, c, r in frac_rows(inst, alpha):
        U = rng.random(R)[:, None] + np.arange(r)[None, :]
        idx = np.searchsorted(c, U, side="right") - 1
        np.add.at(L, (np.repeat(rows_idx, r, axis=1), sl[idx]), int(inst.wu[i]))
    return L


def ztest(emp, th, R):
    ok = (R * th >= 10) & (R * (1 - th) >= 10)
    if not ok.any():
        return 0.0
    se = np.sqrt(th * (1 - th) / R)
    return float(np.max(np.abs(emp - th)[ok] / se[ok]))


# ── instance pools ──────────────────────────────────────────────────────────────

def pool_specs(rng):
    """The candidate instances of the pool, drawn in a fixed order (the only use of rng in the pool)."""
    sizes = ((8, 12), (12, 24), (20, 24), (30, 48))
    reps = 2 if QUICK else 4
    return [(fam, n, T, V.FAMILIES[fam](n, T, rng)) for fam in V.FAMILIES for n, T in sizes for _ in range(reps)]


def pool_entry(spec, env):
    """Solve one candidate exactly; None if its optimum is not proven within 60 s."""
    fam, n, T, raw = spec
    inst = B.Instance(raw)
    ps, lb, opt, _ = X.solve_aggregated(inst, env, 60, threads=2)
    if not opt:
        return None
    p_lp, a = B.solve_lp(inst, env)
    cert = C.Certificates(inst, a, p_lp)
    ub = cert.round_down_ub(p_lp + inst.w.max())
    lbc, ac, q = C.cut_lower_bound(cert, ub, env)
    lbf, af = C.config_lower_bound(cert, lbc, ub, env)
    au = B.uniform_alpha(inst)
    mats = {"lp": a, "cut": ac, "conf": af, "uniform": au, "mixed": 0.5 * a + 0.5 * au}
    return dict(fam=fam, n=n, T=T, inst=inst, P=ps, p_lp=p_lp, cert=cert, lb_cut=lbc, lb_conf=lbf, q=q, mats=mats)


def small_pool(env, rng):
    return [e for e in (pool_entry(sp, env) for sp in pool_specs(rng)) if e is not None]


# ── Section 3 ───────────────────────────────────────────────────────────────────

def check_prop31(pool):
    f = sum(d["p_lp"] > d["P"] + EPS for d in pool)
    record("Prop 3.1", "P_LP <= P*", len(pool), f)


def check_prop32(env, rng):
    tests = fails = 0
    for _ in range(12 if QUICK else 30):
        m = int(rng.integers(2, 4)); n = int(rng.integers(4, 8))
        a = rng.integers(1, 21, n).astype(float)
        best = min(max(np.bincount(np.array(asg), weights=a, minlength=m)) for asg in itertools.product(range(m), repeat=n))
        inst = make([1] * n, [m] * n, [1] * n, a, m)
        ps, _, opt, _ = X.solve_aggregated(inst, env, 60, threads=2)
        tests += 1; fails += (not opt) or abs(ps - best) > EPS
    record("Prop 3.2", "PPMSP with p_i=1, common window = P||Cmax (brute force)", tests, fails)


def check_thm33(pool, env):
    tests = fails = 0
    for d in pool:
        inst = d["inst"]
        xs, opt = xform_exact(inst, env, 60)
        ps, _, opt2, sched = X.solve_aggregated(inst, env, 60, threads=2)
        if not (opt and opt2):
            continue
        tests += 1
        ok = abs(xs - ps) < EPS and (sched.sum(1) == inst.p).all()
        ok &= all(sched[i, t] == 0 for i in range(inst.n) for t in range(inst.T) if t not in set(inst.win[i].tolist()))
        ok &= abs((inst.wu[:, None] * sched).sum(0).max() / B.UNIT - ps) < EPS
        fails += not ok
    record("Thm 3.3", "aggregated MILP = binary MILP; recovered schedule integral, feasible, optimal", tests, fails)


def check_cor34(env, rng):
    tests = fails = 0
    for w in (3.3, 7.2, 11.5, 22.0):
        for n, T in ((10, 24), (20, 24), (40, 48), (80, 48)):
            for _ in range(2 if QUICK else 3):
                L = rng.integers(2, 12, n)
                s = np.array([rng.integers(1, T - l + 2) for l in L]); e = s + L - 1
                p = np.array([rng.integers(1, l + 1) for l in L])
                inst = make(s, e, p, [w] * n, T)
                ps, _, opt, _ = X.solve_aggregated(inst, env, 60, threads=2)
                p_lp, _ = B.solve_lp(inst, env)
                pred = w * math.ceil(p_lp / w - 1e-9)
                tests += 1; fails += (not opt) or abs(ps - pred) > 1e-6
    record("Cor 3.4", "single power class: P* = w ceil(P_LP / w)", tests, fails)


# ── Section 4 ───────────────────────────────────────────────────────────────────

def check_lem42(pool, rng):
    tests = fails = 0; zmax = 0.0
    for d in pool[: (6 if QUICK else 18)]:
        for name in ("lp", "mixed"):
            al = d["mats"][name]
            for i, sl, c, r in frac_rows(d["inst"], al):
                outs = C.systematic_outcomes(sl, al[i, sl])
                tot = sum(pw for _, pw in outs)
                marg = np.array([sum(pw for S, pw in outs if int(t) in S) for t in sl])
                tests += 1
                fails += abs(tot - 1) > 1e-9 or np.abs(marg - al[i, sl]).max() > 1e-9 or any(len(S) != r for S, _ in outs)
            R = 20000
            X0 = np.zeros((R, d["inst"].T))
            for i, sl, c, r in frac_rows(d["inst"], al)[:5]:
                U = rng.random(R)[:, None] + np.arange(r)[None, :]
                idx = np.searchsorted(c, U, side="right") - 1
                freq = np.array([(sl[idx] == t).any(axis=1).mean() for t in sl])
                zmax = max(zmax, ztest(freq, al[i, sl], R))
    record("Lem 4.2", "systematic sampling: exact sizes, exact marginals (outcome law), sampler frequencies", tests,
           fails + (zmax > 4.5), f"max |z| = {zmax:.2f}")


def check_lem43(pool, rng):
    zmax = 0.0; tests = 0
    for d in pool[: (6 if QUICK else 18)]:
        al = d["mats"]["mixed"]
        cert = C.Certificates(d["inst"], al)
        R = 20000
        Ls = sample_loads(d["inst"], al, R, rng)
        for u in np.linspace(cert.lo, cert.H, 10).astype(int):
            emp = (Ls > u).mean(axis=0)
            zmax = max(zmax, ztest(emp, cert.surv[:, u], R)); tests += 1
    record("Lem 4.3", "exact slot laws vs Monte Carlo", tests, int(zmax > 4.5), f"max |z| = {zmax:.2f}")


def check_lem46():
    a = np.linspace(0.001, 0.999, 199)
    lam = np.concatenate([np.linspace(-60, -1e-3, 12000), np.linspace(1e-3, 60, 12000)])
    A, Lm = np.meshgrid(a, lam)
    psi = np.logaddexp(np.log(1 - A), np.log(A) + Lm) - Lm * A          # log E exp(lambda (X - a)), stable
    sup = (2 * psi / Lm ** 2).max(axis=0)
    s2 = C.ks_proxy(a)
    viol = int((sup > s2 * (1 + 1e-9) + 1e-12).sum())
    gap = float(np.max(1 - sup / s2))
    record("Lem 4.6", "sub-Gaussian proxy valid on the grid and optimal (sup_lambda 2 psi / lambda^2 = sigma^2)", len(a),
           viol + int(gap > 2e-3), f"max relative gap to optimality {gap:.1e}")


def check_lem48(pool, rng):
    zind = 0.0; zdep = 0.0; tests = 0
    for d in pool[: (6 if QUICK else 18)]:
        al = d["mats"]["mixed"]
        cert = C.Certificates(d["inst"], al)
        comps = [c for c in cert.random_comps]
        R = 20000
        Ls = sample_loads(d["inst"], al, R, rng).astype(float)
        # components: maxima pairwise uncorrelated
        M = [Ls[:, c].max(axis=1) for c in comps]
        for j, k in itertools.combinations(range(len(M)), 2):
            if M[j].std() > 0 and M[k].std() > 0:
                zind = max(zind, abs(np.corrcoef(M[j], M[k])[0, 1]) * math.sqrt(R)); tests += 1
        # slot pairs: non-adjacent uncorrelated, adjacent typically correlated
        nbr = cert.laws.nbr
        rs = [t for t in range(d["inst"].T) if cert.laws.rand[t]]
        for t, u in itertools.combinations(rs, 2):
            if Ls[:, t].std() == 0 or Ls[:, u].std() == 0:
                continue
            z = abs(np.corrcoef(Ls[:, t], Ls[:, u])[0, 1]) * math.sqrt(R)
            if u in set(nbr[t].tolist()):
                zdep = max(zdep, z)
            else:
                zind = max(zind, z); tests += 1
    record("Lem 4.8", "independence across components and of non-adjacent slots (correlation z-tests)", tests,
           int(zind > 5.0), f"max |z| independent pairs {zind:.2f}; max |z| adjacent pairs {zdep:.1f} (detection check)")


def lp_vertex_structure(inst, env):
    p_lp, a = B.solve_lp(inst, env)
    F = [(i, t) for i in range(inst.n) for t in inst.win[i] if 1e-9 < a[i, t] < 1 - 1e-9]
    J = {i for i, _ in F}
    loads = (inst.w[:, None] * a).sum(axis=0)
    Teq = int((loads >= p_lp - 1e-7).sum())
    # components of the fractional bipartite graph
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, t in F:
        ra, rb = find(("v", i)), find(("s", int(t)))
        if ra != rb:
            parent[ra] = rb
    nodes, edges = {}, {}
    for i, t in F:
        r = find(("v", i)); edges[r] = edges.get(r, 0) + 1
    for x in list(parent):
        r = find(x); nodes[r] = nodes.get(r, 0) + 1
    pf_ok = all(edges.get(r, 0) <= nodes[r] for r in nodes)
    return len(J) <= Teq - 1, len(F) <= 2 * Teq - 2, pf_ok, len(F), len(J), Teq


def check_lem49_410(env, rng):
    tests = f49 = f410 = 0
    sizes = ((20, 24), (50, 48), (200, 96), (500, 96)) + (() if QUICK else ((1000, 96),))
    for fam in V.FAMILIES:
        for n, T in sizes:
            for _ in range(2 if QUICK else 4):
                inst = B.Instance(V.FAMILIES[fam](n, T, rng))
                okJ, okF, okP, *_ = lp_vertex_structure(inst, env)
                tests += 1; f49 += not (okJ and okF); f410 += not okP
    record("Lem 4.9", "LP vertex: |J| <= |T=| - 1 and sum_t |V_t| <= 2|T=| - 2", tests, f49)
    record("Lem 4.10", "LP vertex: every component of the fractional support has #edges <= #vertices", tests, f410)


def closed_forms_indep(inst, alpha):
    """Theorem 4.11 recomputed from the formulas (kW). Returns dict (choice, radius) -> bound and parts."""
    L = B.SlotLaws(inst, alpha)
    rs = [t for t in range(inst.T) if L.rand[t]]
    mu = (inst.w[:, None] * alpha).sum(axis=0).max()
    if not rs:
        return None
    nu = max(sum((w / B.UNIT) ** 2 * C.ks_proxy(a) for _, w, a in L.rand[t]) for t in rs)
    v = max(sum((w / B.UNIT) ** 2 * a * (1 - a) for _, w, a in L.rand[t]) for t in rs)
    b = max(max((w / B.UNIT) * (1 - a) for _, w, a in L.rand[t]) for t in rs)
    cert = C.Certificates(inst, alpha)
    m = len(rs)
    mbar = max(sum(1 for t in c if L.rand[t]) for c in cert.random_comps)
    d = max(len(nb) for nb in L.nbr)
    out = {}
    for name, Lc in (("a", math.log(m)), ("b", math.log(mbar)), ("c", math.log(math.e * d) if d >= 1 else None)):
        if Lc is None:
            continue
        rsg = math.sqrt(2 * nu * Lc)
        rb = (b * Lc / 3) * (1 + math.sqrt(1 + 18 * v / (b * b * Lc))) if Lc > 0 else 0.0
        out[(name, "SG")] = mu + rsg; out[(name, "B")] = mu + rb
    return dict(bounds=out, mu=mu, m=m, mbar=mbar, d=d, nu=nu, v=v, b=b)


def check_thm411_cor412(pool):
    tests = fails = fdom = tdom = frate = trate = frem = trem = 0
    for d in pool:
        inst, P = d["inst"], d["P"]
        for name, al in d["mats"].items():
            cf = closed_forms_indep(inst, al)
            if cf is None:
                continue
            for k, v in cf["bounds"].items():
                tests += 1; fails += v < P - EPS
            # Corollary 4.12: the non-localized forms dominate the localized ones and are valid
            pub = B.old_bounds(inst, al)
            for (ch, rad), key in ((("a", "SG"), "hoeffding_union"), (("a", "B"), "bernstein_union"),
                                   (("c", "SG"), "hoeffding_lll"), (("c", "B"), "bernstein_lll")):
                if (ch, rad) in cf["bounds"]:
                    tdom += 1; fdom += pub[key] < cf["bounds"][(ch, rad)] - EPS or pub[key] < P - EPS
            # Remark 4.13: (c) < (a) iff e d < m (SG radius)
            if ("c", "SG") in cf["bounds"] and cf["m"] > 1:
                trem += 1
                lhs = cf["bounds"][("c", "SG")] < cf["bounds"][("a", "SG")] - 1e-12
                rhs = math.e * cf["d"] < cf["m"]
                frem += lhs != rhs
        # eq (22) at the LP vertex
        L = B.SlotLaws(inst, d["mats"]["lp"])
        kstar = max(len(r) for r in L.rand)
        trate += 1
        frate += P - d["p_lp"] > inst.w.max() * math.sqrt(0.5 * kstar * math.log(inst.T)) + EPS
    record("Thm 4.11", "closed forms (a), (b), (c) x (sub-Gaussian, Bernstein) >= P* at 5 matrices", tests, fails)
    record("Cor 4.12", "distribution-free forms >= localized forms (same alpha) and >= P*", tdom, fdom)
    record("Rem 4.13", "local lemma closed form beats union bound iff e d_alpha < m_alpha", trem, frem)
    record("Eq (22)", "P* - P_LP <= w_max sqrt(k* ln T / 2)", trate, frate)


def check_thm414(pool, rng):
    tests = fails = 0; fsucc = tsucc = 0; flll = 0
    for d in pool:
        inst, P = d["inst"], d["P"]
        for name in ("lp", "cut", "conf", "mixed"):
            cert = C.Certificates(inst, d["mats"][name])
            u1, u2, u3 = cert.union(), cert.component_union(), cert.lll()
            tests += 1
            fails += u1 < P - EPS or u2 < P - EPS or u3 < P - EPS or u2 > u1 + EPS
            xi = C.lll_weights(cert, u3)
            if xi is not None:
                uu = int(round(u3 * B.UNIT))
                p = cert.surv[:, uu]
                rhs = np.array([xi[t] * np.prod(1 - xi[nb]) if len(nb) else xi[t] for t, nb in enumerate(cert.laws.nbr)])
                flll += bool((p > rhs + 1e-9).any() or (xi >= 1).any())
            if name in ("lp", "mixed"):
                R = 20000
                Ls = sample_loads(inst, d["mats"][name], R, rng)
                uu = int(round(u2 * B.UNIT))
                emp = (Ls.max(axis=1) <= uu).mean()
                bnd = cert.success_probability(u2)
                tsucc += 1; fsucc += emp < bnd - 3 * math.sqrt(max(bnd * (1 - bnd), 1e-12) / R) - 1e-9
    record("Thm 4.14", "union, componentwise, LLL certificates >= P*; U_Gamma <= U_union", tests, fails)
    record("Thm 4.14", "least fixed point of the LLL iteration satisfies condition (iii)", tests, flll)
    record("Thm 4.14", "success probability bound <= Monte Carlo success frequency", tsucc, fsucc)


def check_thm415(pool, rng):
    tests = fails = fid = 0
    for d in pool[: (12 if QUICK else 40)]:
        inst, P = d["inst"], d["P"]
        for name in ("lp", "mixed"):
            al = d["mats"][name]
            cert = C.Certificates(inst, al)
            R = 20000
            pk = sample_loads(inst, al, R, rng).max(axis=1) / B.UNIT
            em, se = pk.mean(), pk.std() / math.sqrt(R)
            ue, uec = cert.averaging(), cert.component_averaging()
            tests += 1
            fails += em < P - EPS or em > uec + 4.5 * se + 1e-9 or uec > ue + EPS
            exc = cert.laws.excess_curve(cert.H)
            mn = min(u + exc[u] for u in range(cert.H + 1)) / B.UNIT
            fid += abs(mn - ue) > 1e-9
    record("Thm 4.15", "P* <= E[max] (MC) <= U_EGamma <= U_E", tests, fails)
    record("Thm 4.15", "U_E = min_u {u + sum_t E[(S_t - u)^+]} (identity, exact)", tests, fid)


def check_thm417(pool, rng):
    from scipy.stats import binom
    tests = fails = fs = ts = 0
    for d in pool[: (10 if QUICK else 30)]:
        inst, P = d["inst"], d["P"]
        for name in ("lp", "mixed"):
            drc = C.DRCertificates(inst, d["mats"][name])
            vals = drc.all()
            tests += 1; fails += any(v < P - EPS for v in vals.values())
            u = vals["dr_union"]; uu = int(round(u * B.UNIT))
            R = 400 if QUICK else 1000
            ok = 0
            for _ in range(R):
                x = C.dependent_round(inst, d["mats"][name], rng)
                ok += (inst.wu[:, None] * x).sum(0).max() <= uu
            bnd = 1 - drc.surv[:, uu].sum()
            # exact binomial test (near 1 the normal approximation fails): a failure is a success count
            # this low with probability below 1e-6 if the theorem holds
            ts += 1; fs += bnd > 0 and binom.cdf(ok, R, min(bnd, 1.0)) < 1e-6
    record("Thm 4.17", "dependent-rounding certificates and class ceiling >= P*", tests, fails)
    record("Thm 4.17", "dependent rounding meets u with probability >= 1 - sum_t P(S~_t > u) (MC)", ts, fs)


def brute_optimal_rounding(inst, alpha, cap=50000):
    rows = frac_rows(inst, alpha)
    opts = [[np.array(S) for S in itertools.combinations(sl.tolist(), r)] for _, sl, _, r in rows]
    if not rows or math.prod(len(o) for o in opts) > cap:
        return None
    base = det_loads(inst, alpha)
    best = np.inf
    for ch in itertools.product(*opts):
        L = base.copy()
        for (i, _, _, _), S in zip(rows, ch):
            L[S] += inst.wu[i]
        best = min(best, L.max())
    return best / B.UNIT


def check_thm418(pool, rng):
    tests = fails = tb = fb = tz = 0; zmax = 0.0; zem = 0.0
    for d in pool:
        inst, P = d["inst"], d["P"]
        for name in ("lp", "cut", "conf", "mixed"):
            al = d["mats"][name]
            cert = C.Certificates(inst, al)
            try:
                fs = C.ForestDP(cert, "systematic"); fc = C.ForestDP(cert, "compatible")
                ue, ur = fs.threshold(), fc.threshold()
            except RuntimeError:
                continue
            tests += 1
            fails += ur < P - EPS or ur > ue + EPS or ue > cert.component_union() + EPS or ue > cert.lll() + EPS
            br = brute_optimal_rounding(inst, al)
            if br is not None:
                tb += 1; fb += abs(br - ur) > EPS
            if name in ("lp", "mixed") and len(cert.frac_slots) > 0:
                R = 20000
                pk = sample_loads(inst, al, R, rng).max(axis=1)
                for u in np.unique(pk)[:8]:
                    th = fs.prob_ok(int(u))
                    zmax = max(zmax, ztest(np.array([(pk <= u).mean()]), np.array([th]), R)); tz += 1
                if name == "lp":
                    em = fs.expected_max()
                    zem = max(zem, abs(pk.mean() / B.UNIT - em) / max(pk.std() / B.UNIT / math.sqrt(R), 1e-12))
                    fails += em > cert.component_averaging() + EPS or em < ue - EPS
    record("Thm 4.18", "P* <= U_R <= U_exists <= U_Gamma, U_LLL;  U_exists <= E[max] <= U_EGamma", tests, fails)
    record("Thm 4.18", "optimal rounding by dynamic programming = brute-force optimal rounding", tb, fb)
    record("Thm 4.18", "exact P(max <= u) and exact E[max] vs Monte Carlo", tz, int(zmax > 4.5 or zem > 4.5),
           f"max |z| probability {zmax:.2f}, expectation {zem:.2f}")


def check_thm419(pool, env, rng):
    # validity by brute force over slot configurations
    bad = tv = 0
    for _ in range(100 if QUICK else 300):
        inst = B.Instance(V.gen_staggered(12, 10, rng))
        P = int(rng.integers(50, 400))
        for t, S, k in C.cover_cuts(inst, P):
            I = inst.avail[t][:10]
            Sset = set(S)
            for z in itertools.product([0, 1], repeat=len(I)):
                on = [i for i, b in zip(I, z) if b]
                tv += 1
                if inst.wu[on].sum() <= P and len(Sset & set(on)) > k:
                    bad += 1
    record("Thm 4.19", "every load-feasible slot configuration satisfies every cover inequality (brute force)", tv, bad)
    tests = fails = fmono = 0
    for d in pool:
        cert, inst = d["cert"], d["inst"]
        tests += 1
        fails += d["lb_cut"] > d["P"] + EPS or d["lb_cut"] < max(cert.round_up_lb(d["p_lp"]), inst.w.max()) - EPS
        A = cert.attainable_loads()
        A = A[(A >= d["p_lp"] * B.UNIT - 1e-6) & (A <= d["P"] * B.UNIT + 1e-6)][:25]
        f = []
        for Pv in A:
            q, _ = C.cut_lp(inst, int(Pv), env)
            f.append(q is not None and q <= Pv + 1e-6)
        fmono += any(f[j] and not f[j + 1] for j in range(len(f) - 1))
    record("Thm 4.19", "max(ceil P_LP, w_max) <= LB_cut <= P*", tests, fails)
    record("Thm 4.19", "feasibility of the cut LP is monotone in the level P", tests, fmono)
    # special case P||Cmax
    tp = fp = 0
    for _ in range(10 if QUICK else 25):
        m = int(rng.integers(2, 5)); n = int(rng.integers(m + 2, 4 * m + 3))
        a = rng.choice(V.LEVEL2, n)
        inst = make([1] * n, [m] * n, [1] * n, a, m)
        p_lp, al = B.solve_lp(inst, env)
        cert = C.Certificates(inst, al, p_lp)
        lb, _, _ = C.cut_lower_bound(cert, cert.round_down_ub(p_lp + inst.w.max()), env)
        srt = np.sort(inst.w)[::-1]
        cls = max([srt[k * m - k:k * m + 1].sum() for k in range(1, (n - 1) // m + 1)] + [srt[0]])
        tp += 1; fp += lb < cls - EPS
    record("Rem (Sec 4.6)", "P||Cmax special case: LB_cut >= max_k (w_(kT-k+1) + ... + w_(kT+1))", tp, fp)


def explicit_config_feasible(inst, P, env):
    import gurobipy as gp
    md = gp.Model(env=env); md.Params.OutputFlag = 0
    lam = {}
    for t in range(inst.T):
        I = inst.avail[t]
        for z in itertools.product([0, 1], repeat=len(I)):
            S = tuple(i for i, b in zip(I, z) if b)
            if inst.wu[list(S)].sum() <= P:
                lam[t, S] = md.addVar(lb=0.0)
    for t in range(inst.T):
        if inst.avail[t]:
            md.addConstr(gp.quicksum(v for (tt, S), v in lam.items() if tt == t) == 1)
    for i in range(inst.n):
        md.addConstr(gp.quicksum(v for (t, S), v in lam.items() if i in S) == int(inst.p[i]))
    md.optimize()
    return md.Status == 2


def check_thm420(env, rng, pool):
    tests = fails = 0
    for d in pool:
        tests += 1; fails += d["lb_conf"] < d["lb_cut"] - EPS or d["lb_conf"] > d["P"] + EPS
    record("Thm 4.20", "LB_cut <= LB_conf <= P*", tests, fails)
    tc = fc = 0
    for _ in range(8 if QUICK else 20):
        inst = B.Instance(V.gen_staggered(7, 12, rng))
        if max(len(a) for a in inst.avail) > 9:
            continue
        p_lp, a = B.solve_lp(inst, env)
        cert = C.Certificates(inst, a, p_lp)
        ps, _, _, _ = X.solve_aggregated(inst, env, 30, threads=2)
        A = cert.attainable_loads()
        for Pv in A[(A >= p_lp * B.UNIT - 1e-6) & (A <= ps * B.UNIT + 1e-6)]:
            ok_cg, _, _ = C.config_lp(inst, int(Pv), env)
            tc += 1; fc += ok_cg != explicit_config_feasible(inst, int(Pv), env)
    record("Thm 4.20", "column generation decides feasibility exactly (vs explicit enumeration of configurations)", tc, fc)


def check_thm421(pool, env):
    tests = fails = 0
    for d in pool:
        inst = d["inst"]
        C.IR_LP_COUNT = 0
        x, pk, guar, ok = C.iterative_rounding(inst, d["p_lp"], env)
        nlp = C.IR_LP_COUNT
        tests += 1
        fails += (not ok) or pk > d["p_lp"] + inst.w.max() + EPS or pk < d["P"] - EPS or pk > guar + EPS \
            or nlp > sum(len(w) for w in inst.win) + inst.T or (x.sum(1) != inst.p).any()
    record("Thm 4.21", "iterative rounding: rule applies, schedule feasible, peak <= P_LP + w_max, #LP within bound", tests, fails)


def check_ex422(env):
    tests = fails = 0
    for Lw, w in ((10.0, 20.0), (3.3, 22.0), (7.2, 7.2)):
        inst = make([1, 2, 1], [1, 2, 2], [1, 1, 1], [Lw, Lw, w], 2)
        p_lp, a = B.solve_lp(inst, env)
        ps, _, _, _ = X.solve_aggregated(inst, env, 30, threads=2)
        cert = C.Certificates(inst, a, p_lp)
        lb, _, _ = C.cut_lower_bound(cert, ps, env)
        tests += 1
        fails += abs(p_lp - (Lw + w / 2)) > 1e-6 or abs(ps - (Lw + w)) > 1e-6 or abs(lb - ps) > 1e-6
    record("Ex 4.22", "P_LP = L + w/2, P* = L + w, LB_cut = P* on the gap family", tests, fails)


def check_thm423(pool):
    tests = fails = 0
    for d in pool:
        inst, cert = d["inst"], d["cert"]
        ubs = [d["p_lp"] + inst.w.max()]
        for name in ("lp", "cut", "conf"):
            c = C.Certificates(inst, d["mats"][name])
            ubs += [c.union(), c.component_union(), c.lll(), c.averaging(), c.component_averaging()]
        U = cert.round_down_ub(min(ubs))
        tests += 1; fails += not (d["lb_conf"] - EPS <= d["P"] <= U + EPS)
    record("Thm 4.23", "LB_conf <= P* <= U (rounded)", tests, fails)


def check_cor424(env, rng):
    rows = []; tests = fails = 0
    ns = (50, 100, 200, 400) + (() if QUICK else (800, 1600))
    for fam in ("overnight", "staggered"):
        for n in ns:
            inst = B.Instance(V.FAMILIES[fam](n, 48, rng))
            p_lp, a = B.solve_lp(inst, env)
            cert = C.Certificates(inst, a, p_lp)
            U = cert.round_down_ub(min(p_lp + inst.w.max(), cert.union(), cert.component_union(), cert.lll()))
            L = B.SlotLaws(inst, a)
            kstar = max(len(r) for r in L.rand)
            rhs = 1 + min(inst.w.max(), inst.w.max() * math.sqrt(0.5 * kstar * math.log(inst.T))) / p_lp
            tests += 1; fails += U / p_lp > rhs + EPS
            rows.append(f"{fam} n={n}: U/P_LP = {U / p_lp:.4f} (bound {rhs:.4f})")
    record("Cor 4.24", "U / P_LP <= 1 + min(w_max, w_max sqrt(k* ln T / 2)) / P_LP; trend with n at fixed T", tests, fails,
           "; ".join(rows))


def check_thm425(pool, rng):
    tmt = fmt = tce = fce = tlv = flv = 0
    for d in pool[: (12 if QUICK else 40)]:
        inst = d["inst"]
        for name in ("lp", "conf"):
            al = d["mats"][name]
            cert = C.Certificates(inst, al)
            u = cert.lll()
            xi = C.lll_weights(cert, u)
            if xi is not None:
                steps = []
                for _ in range(100):
                    _, pk, st = C.moser_tardos(cert, u, rng)
                    steps.append(st); fmt += pk > u + EPS
                bound = float((xi / (1 - xi)).sum())
                tmt += 1; fmt += np.mean(steps) > bound + 4.5 * np.std(steps, ddof=1) / 10 + 1e-9
            tot = cert.surv.sum(axis=0)
            us = int(np.argmax(tot <= 1)) if (tot <= 1).any() else cert.H
            g = (us + sum(float((cert.laws.law[t] * np.clip(cert.laws.det[t] + np.arange(len(cert.laws.law[t])) - us, 0, None)).sum())
                          for t in range(inst.T))) / B.UNIT
            _, pe = B.derandomize(inst, al, us, "excess")
            _, pu = B.derandomize(inst, al, int(round(cert.union() * B.UNIT)), "union")
            _, pc = C.derandomized_union_by_component(cert)
            tce += 1; fce += pe > g + EPS or pu > cert.union() + EPS or pc > cert.component_union() + EPS
            drc = C.DRCertificates(inst, al)
            ud = drc.union(); uu = int(round(ud * B.UNIT))
            bnd = 1.0 / (1.0 - drc.surv[:, uu].sum())
            rounds = [C.dr_las_vegas(drc, ud, rng, max_rounds=5000)[2] for _ in range(30)]
            tlv += 1; flv += np.mean(rounds) > bnd + 4.5 * np.std(rounds, ddof=1) / math.sqrt(30) + 1e-9
    record("Thm 4.25(a)", "Moser-Tardos: peak <= u, mean resamplings <= sum xi/(1-xi) (+4.5 SE)", tmt, fmt)
    record("Thm 4.25(b)", "conditional expectations: peaks within their guarantees", tce, fce)
    record("Thm 4.25(c)", "repeated dependent rounding: mean rounds <= 1/(1 - sum_t P(S~_t > u)) (+4.5 SE)", tlv, flv)


# ── parallel execution ──────────────────────────────────────────────────────────
# The checks are independent: each one reads the pool and draws its own random instances. In parallel,
# each check gets its own random stream (seeded by its position), the pool is built exactly as in the
# sequential run (same candidates; the exact solves are distributed), and the report keeps the paper order.

SEED = 2026102101
CHECKS = ["check_prop31", "check_prop32", "check_thm33", "check_cor34", "check_lem42", "check_lem43", "check_lem46",
          "check_lem48", "check_lem49_410", "check_thm411_cor412", "check_thm414", "check_thm415", "check_thm417",
          "check_thm418", "check_thm419", "check_thm420", "check_thm421", "check_ex422", "check_thm423",
          "check_cor424", "check_thm425"]
HEAVY_FIRST = ["check_thm419", "check_cor424", "check_thm418", "check_lem48", "check_thm425", "check_thm414",
               "check_thm420", "check_lem42", "check_thm415", "check_thm417", "check_lem43", "check_thm411_cor412"]
_W = {}


def _make_env(threads):
    import gurobipy as gp
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.setParam("Threads", threads); env.start()
    return env


def _init_worker(threads):
    import warnings
    warnings.filterwarnings("ignore")
    _W["env"] = _make_env(threads)


def _pool_task(spec):
    return pool_entry(spec, _W["env"])


def _check_task(name, idx, pool):
    import inspect
    fn = globals()[name]
    REPORT.clear()
    args = {"pool": pool, "env": _W["env"], "rng": np.random.default_rng([SEED, idx])}
    fn(*[args[q] for q in inspect.signature(fn).parameters])
    return list(REPORT)


def _workers():
    """--workers N, else one per physical core but one (at most one per check)."""
    if "--workers" in sys.argv:
        return max(1, int(sys.argv[sys.argv.index("--workers") + 1]))
    try:
        import psutil
        cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except Exception:
        cores = os.cpu_count() or 1
    return max(1, min(len(CHECKS), cores - 1, 61 if os.name == "nt" else 10 ** 6))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    workers = _workers()
    if workers <= 1:                                   # the original sequential run
        env = _make_env(2)
        pool = small_pool(env, rng)
        print(f"pool: {len(pool)} instances with a proven optimum ({time.time() - t0:.0f}s)", flush=True)
        for name in CHECKS:
            fn = globals()[name]
            import inspect
            args = {"pool": pool, "env": env, "rng": rng}
            fn(*[args[q] for q in inspect.signature(fn).parameters])
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        print(f"parallel run: {workers} worker processes", flush=True)
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                                 initargs=(2,)) as ex:
            pool = [e for e in ex.map(_pool_task, pool_specs(rng)) if e is not None]
            print(f"pool: {len(pool)} instances with a proven optimum ({time.time() - t0:.0f}s)", flush=True)
            order = HEAVY_FIRST + [c for c in CHECKS if c not in HEAVY_FIRST]
            futs = {name: ex.submit(_check_task, name, CHECKS.index(name), pool) for name in order}
            for name in CHECKS:                        # the report keeps the order of the paper
                REPORT.extend(futs[name].result())
    res = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else RES
    os.makedirs(res, exist_ok=True)
    df = pd.DataFrame(REPORT)
    df.to_csv(os.path.join(res, "theory_verification.csv"), index=False)
    lines = ["# Numerical verification of Sections 3-4 (verify_theory.py)", "",
             f"Pool: {len(pool)} small instances with a proven optimum (3 families, n = 8-30). "
             f"Run time {time.time() - t0:.0f} s. Failures in total: {int(df.failures.sum())}.", "",
             "| result | statement | tests | failures | detail |", "|---|---|---|---|---|"]
    for r in REPORT:
        lines.append(f"| {r['result']} | {r['statement']} | {r['tests']} | {r['failures']} | {r['detail']} |")
    open(os.path.join(res, "theory_verification.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"\nTOTAL failures: {int(df.failures.sum())} over {int(df.tests.sum())} tests ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
