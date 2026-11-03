"""
validate.py -- validation suite of the certificates and algorithms of Section 4.

  1. unit checks: Kearns-Saul proxy (log-MGF inequality on a grid), systematic sampler
     (inclusion frequencies), exact slot laws (Monte Carlo tail frequencies)
  2. small instances of three families, solved to proven optimality by Gurobi:
     every certificate >= P*, every algorithm between P* and its own guarantee,
     iterative rounding's drop rule always applicable
  3. medium and large instances: certificate / LP and algorithm / LP ratios

Run from this folder:  python validate.py [--quick]
Writes results/validation_small.csv, results/validation_large.csv, results/validation_log.txt
"""

import os
import sys
import time

import numpy as np
import pandas as pd

import bounds as B
import certificates as C
import exact as X

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
LEVEL2 = np.array([3.3, 6.2, 7.2, 9.6, 11.5, 12.0, 14.4, 19.2, 22.0])
BATT = np.array([24.0, 30.0, 40.0, 52.0, 58.0, 62.0, 64.0, 75.0, 77.0, 82.0, 90.0, 100.0, 105.0])
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


# ── instance families ───────────────────────────────────────────────────────────

def gen_omega(n, T, rng):
    """The omega generator, used with the earlier methods (vehicles of a charger have disjoint windows)."""
    m = int(np.ceil(n / 3))
    part = np.zeros(n, int); part[:m] = np.arange(1, m + 1); part[m:] = rng.integers(1, m + 1, n - m)
    part = part[rng.permutation(n)]
    mp = LEVEL2[rng.integers(0, len(LEVEL2), m)]
    s = np.zeros(n, int); e = np.zeros(n, int); p = np.zeros(n, int); w = np.zeros(n)
    dt = 24.0 / T
    for j in range(1, m + 1):
        tasks = np.where(part == j)[0]; N = len(tasks)
        if 2 * N > T:
            step = max(1, T // N)
            for k, i in enumerate(tasks):
                s[i] = k * step + 1; e[i] = min(s[i] + step - 1, T)
        else:
            tt = np.sort(rng.choice(np.arange(1, T + 1), 2 * N, replace=False)).reshape(N, 2)
            for k, i in enumerate(tasks):
                s[i], e[i] = tt[k]
        for i in tasks:
            L = e[i] - s[i] + 1; b = rng.choice(BATT); tgt = max(1, int(0.8 * L))
            phmax = min(0.95, tgt * mp[j - 1] * dt / b); phmin = min(0.10, phmax)
            soc = rng.uniform(phmin, phmax)
            p[i] = int(np.clip(int(np.ceil(b * soc / (mp[j - 1] * dt))), 1, tgt)); w[i] = mp[j - 1]
    return (part, s, e, p, w, n, m, T, 0.0)


def gen_staggered(n, T, rng):
    """Short staggered windows (3 to 10 slots): sparse dependency, the local lemma's regime."""
    L = rng.integers(3, 11, n)
    s = np.array([rng.integers(1, T - l + 2) for l in L])
    e = s + L - 1
    p = np.array([rng.integers(1, max(1, int(0.8 * l)) + 1) for l in L])
    w = LEVEL2[rng.integers(0, len(LEVEL2), n)]
    return (np.ones(n, int), s, e, p, w, n, 1, T, 0.0)


def gen_overnight(n, T, rng):
    """Long overnight windows: arrival in the first quarter, departure in the last quarter."""
    s = rng.integers(1, T // 4 + 1, n)
    e = rng.integers(3 * T // 4, T + 1, n)
    L = e - s + 1
    p = np.array([rng.integers(1, max(1, int(0.4 * l)) + 1) for l in L])
    w = LEVEL2[rng.integers(0, len(LEVEL2), n)]
    return (np.ones(n, int), s, e, p, w, n, 1, T, 0.0)


FAMILIES = {"omega": gen_omega, "staggered": gen_staggered, "overnight": gen_overnight}


def solve_exact(inst, env, limit):
    """Reference optimum: the exact class-aggregated MILP (exact.py; same optimum as the x-model,
    integer variables only on the class counts). Returns (incumbent, dual bound, proven)."""
    ub, lb, opt, _ = X.solve_aggregated(inst, env, limit)
    return ub, lb, opt


# ── one instance ────────────────────────────────────────────────────────────────

def derandomized(inst, cert, alpha, pre=""):
    """The conditional-expectation algorithms at marginals alpha, with their guarantees."""
    row = {}
    tot = cert.surv.sum(axis=0)
    u_star = int(np.argmax(tot <= 1)) if (tot <= 1).any() else cert.H     # minimizer of the averaging bound
    _, row[f"alg_{pre}derand_excess"] = B.derandomize(inst, alpha, u_star, "excess")
    row[f"guar_{pre}derand_excess"] = (u_star + sum(
        float((cert.laws.law[t] * np.clip(cert.laws.det[t] + np.arange(len(cert.laws.law[t])) - u_star, 0, None)).sum())
        for t in range(inst.T))) / B.UNIT
    _, row[f"alg_{pre}derand_excess_3p"] = B.derandomize(inst, alpha, u_star, "excess", passes=3)
    _, row[f"alg_{pre}derand_component"] = C.derandomized_union_by_component(cert)
    return row


def evaluate(inst, env, rng, mt=200, kmax=None, conf=True):
    p_lp, alpha = B.solve_lp(inst, env)
    cert = C.Certificates(inst, alpha, p_lp)
    row = {"P_LP": p_lp, "w_max": inst.w.max(), "LP_plus_wmax": p_lp + inst.w.max()}
    row.update(cert.all())
    old = B.old_bounds(inst, alpha)
    row.update({f"published_{k}": v for k, v in old.items()})
    # algorithms and their guarantees
    row.update(derandomized(inst, cert, alpha))
    _, row["alg_derand_union"] = B.derandomize(inst, alpha, int(round(row["union"] * B.UNIT)), "union")
    x, row["alg_iterative_rounding"], row["guar_iterative_rounding"], row["ir_rule_ok"] = C.iterative_rounding(inst, p_lp, env)
    if mt:
        xw = C.lll_weights(cert, row["lll"])
        row["mt_expected_bound"] = float((xw / (1 - xw)).sum()) if xw is not None else np.nan
        steps = []
        for _ in range(mt):
            _, pk, st = C.moser_tardos(cert, row["lll"], rng)
            steps.append(st)
            row["alg_moser_tardos"] = max(row.get("alg_moser_tardos", 0), pk)
        row["mt_mean_steps"] = float(np.mean(steps))
        row["mt_se"] = float(np.std(steps, ddof=1) / np.sqrt(len(steps)))
    row["success_prob_at_component_union"] = cert.success_probability(row["component_union"])
    # strengthened relaxation: cover-cut lower bound, then certificates and algorithms at its marginals
    ub = min(row[a] for a, _ in ALGS if a in row)
    row["lb_lp_rounded"] = cert.round_up_lb(p_lp)
    row["lb_cut"], alpha_c, q = C.cut_lower_bound(cert, ub, env, kmax)
    cert_c = C.Certificates(inst, alpha_c, q)
    for k in ("union", "component_union", "lll", "averaging", "component_averaging"):
        row[f"cut_{k}"] = getattr(cert_c, k)()
    row.update(derandomized(inst, cert_c, alpha_c, "cut_"))
    # configuration LP lower bound (dominates the cuts), then certificates and algorithms at its marginals
    if conf:
        ub = min(row[a] for a, _ in ALGS if a in row)
        row["lb_conf"], alpha_f = C.config_lower_bound(cert, row["lb_cut"], ub, env)
        cert_f = C.Certificates(inst, alpha_f, row["lb_conf"])
        for k in ("union", "component_union", "lll", "averaging", "component_averaging"):
            row[f"conf_{k}"] = getattr(cert_f, k)()
        row.update(derandomized(inst, cert_f, alpha_f, "conf_"))
    # dependent rounding by power class, at alpha* and alpha_cut
    for tag, al, q0 in (("", alpha, p_lp), ("cut_", alpha_c, q)):
        drc = C.DRCertificates(inst, al, q0)
        row.update({f"{tag}{k}": v for k, v in drc.all().items()})
        if tag == "":
            u = int(round(row["dr_union"] * B.UNIT))
            row["dr_lv_bound"] = 1.0 / (1.0 - drc.surv[:, u].sum())
            _, row["alg_dr_las_vegas"], row["dr_lv_rounds"] = C.dr_las_vegas(drc, row["dr_union"], rng, max_rounds=5000)
    # exact certificates on the fractional support (Theorem L): existence thresholds, exact expected peak
    marg = [("", cert), ("cut_", cert_c)] + ([("conf_", cert_f)] if conf else [])
    for tag, cc in marg:
        for mode, name in (("systematic", "exist_sys"), ("compatible", "exist_full")):
            try:
                f = C.ForestDP(cc, mode)
                row[f"{tag}{name}"] = f.threshold()
                if tag == "" and mode == "systematic":
                    row["forest_max_cut"] = max(f.cut_sizes, default=0)
                    if inst.n <= 200:
                        row["emax_exact"] = f.expected_max()
            except RuntimeError:
                row[f"{tag}{name}"] = np.nan
    best = float(np.nanmin([row[c] for c in CERTS if c in row and not c.startswith("published")]))
    row["best_certificate"] = best
    row["certified_lb"] = max(row["lb_cut"], row.get("lb_conf", -np.inf))
    row["certified_ub"] = cert.round_down_ub(best)
    row["best_algorithm"] = min(row[a] for a, _ in ALGS if a in row)
    row["a_priori_optimal"] = abs(row["certified_ub"] - row["certified_lb"]) < 1e-9
    row["solver_free_optimal"] = abs(row["best_algorithm"] - row["certified_lb"]) < 1e-9
    return row


CERTS = ["published_hoeffding_lll", "published_bernstein_lll", "published_bernstein_union",
         "closed_hoeffding_lll", "closed_ks_lll", "closed_bernstein_lll", "closed_ks_comp",
         "hoeffding_component_union", "kearns_saul_union", "kearns_saul_component_union", "kearns_saul_lll",
         "bennett_component_union", "union", "component_union", "lll", "averaging", "component_averaging",
         "LP_plus_wmax", "guar_iterative_rounding",
         "cut_union", "cut_component_union", "cut_lll", "cut_averaging", "cut_component_averaging",
         "conf_union", "conf_component_union", "conf_lll", "conf_averaging", "conf_component_averaging",
         "dr_union", "dr_component_union", "dr_averaging", "dr_component_averaging", "dr_ceiling",
         "cut_dr_union", "cut_dr_component_union", "cut_dr_averaging", "cut_dr_component_averaging", "cut_dr_ceiling",
         "exist_sys", "exist_full", "emax_exact", "cut_exist_sys", "cut_exist_full", "conf_exist_sys", "conf_exist_full"]
EXTRA = ["best_certificate", "certified_ub", "lb_lp_rounded", "lb_cut", "lb_conf", "certified_lb", "best_algorithm"]
ALGS = [("alg_derand_excess", "guar_derand_excess"), ("alg_derand_union", "union"),
        ("alg_derand_component", "component_union"), ("alg_iterative_rounding", "LP_plus_wmax"),
        ("alg_moser_tardos", "lll"), ("alg_derand_excess_3p", "guar_derand_excess"),
        ("alg_cut_derand_excess", "guar_cut_derand_excess"), ("alg_cut_derand_excess_3p", "guar_cut_derand_excess"),
        ("alg_cut_derand_component", "cut_component_union"),
        ("alg_conf_derand_excess", "guar_conf_derand_excess"), ("alg_conf_derand_excess_3p", "guar_conf_derand_excess"),
        ("alg_conf_derand_component", "conf_component_union"), ("alg_dr_las_vegas", "dr_union")]


# ── unit checks ─────────────────────────────────────────────────────────────────

def unit_checks(rng, env):
    p = np.linspace(0.001, 0.999, 999)
    lam = np.linspace(-60, 60, 2401)
    P, Lm = np.meshgrid(p, lam)
    logmgf = np.log((1 - P) * np.exp(-Lm * P) + P * np.exp(Lm * (1 - P)))
    viol = (logmgf - Lm ** 2 * C.ks_proxy(P) / 2 > 1e-10).sum()
    log(f"[unit] Kearns-Saul proxy: log-MGF inequality violated at {viol} of {P.size} grid points")
    slots = np.arange(7); probs = np.array([0.9, 0.1, 0.5, 0.5, 0.3, 0.7, 1.0])
    cnt = np.zeros(7); R = 200000
    for _ in range(R):
        cnt[C.systematic_sample(slots, probs, rng)] += 1
    log(f"[unit] systematic sampler: max |freq - prob| = {np.abs(cnt / R - probs).max():.4f} (R = {R}),"
        f" exactly {int(probs.sum())} slots each draw")
    inst = B.Instance(gen_omega(40, 48, rng))
    p_lp, alpha = B.solve_lp(inst, env)
    cert = C.Certificates(inst, alpha, p_lp)
    R = 40000
    loads = np.zeros((R, inst.T))
    for r in range(R):
        x = (alpha >= 1 - 1e-9).astype(int)
        for i, sl in cert.frac_slots.items():
            sl = np.array(sl); x[i, sl] = 0; x[i, C.systematic_sample(sl, alpha[i, sl], rng)] = 1
        loads[r] = (inst.wu[:, None] * x).sum(axis=0)
    zmax = 0.0
    for t in range(inst.T):
        if not cert.laws.rand[t]:
            continue
        for q in np.unique(loads[:, t])[:-1]:
            emp = (loads[:, t] > q).mean(); ex = cert.surv[t, int(q)]
            se = max(np.sqrt(ex * (1 - ex) / R), 1e-12)
            zmax = max(zmax, abs(emp - ex) / se)
    log(f"[unit] exact slot laws vs Monte Carlo (R = {R}): max |z| over all slots and levels = {zmax:.2f}")


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    import gurobipy as gp
    quick = "--quick" in sys.argv
    sys.stdout.reconfigure(encoding="utf-8")
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    rng = np.random.default_rng(20261021)
    os.makedirs(OUT, exist_ok=True)
    unit_checks(rng, env)

    small = [(fam, n, T) for fam in FAMILIES for n, T in ((10, 24), (15, 24), (20, 48), (30, 48))]
    reps = 2 if quick else 8
    rows = []
    for fam, n, T in small:
        for k in range(reps):
            inst = B.Instance(FAMILIES[fam](n, T, rng))
            t0 = time.time()
            ub, lb, opt = solve_exact(inst, env, 120)
            row = {"family": fam, "n": n, "T": T, "rep": k, "P_star": ub, "gurobi_bound": lb, "proven": opt}
            row.update(evaluate(inst, env, rng))
            row["sec"] = time.time() - t0
            rows.append(row)
        log(f"[small] {fam:9s} n={n:3d} T={T:3d}: {reps} instances, "
            f"proven {sum(r['proven'] for r in rows if r['family'] == fam and r['n'] == n)}")
    ds = pd.DataFrame(rows)
    ds.to_csv(os.path.join(OUT, "validation_small.csv"), index=False)
    pr = ds[ds.proven]
    log(f"\n[validity] {len(pr)} instances with a proven optimum")
    for c in CERTS:
        bad = int((pr[c] < pr.P_star - 1e-6).sum())
        log(f"  certificate {c:32s} below P*: {bad}")
    for a, g in ALGS:
        below = int((pr[a] < pr.P_star - 1e-6).sum()); above = int((ds[a] > ds[g] + 1e-6).sum())
        log(f"  algorithm   {a:32s} below P*: {below}   above its guarantee: {above}")
    log(f"  LP above P*: {int((pr.P_LP > pr.P_star + 1e-6).sum())}   iterative-rounding rule failures: {int((~ds.ir_rule_ok).sum())}")
    log(f"  rounded LP bound above P*: {int((pr.lb_lp_rounded > pr.P_star + 1e-6).sum())}   "
        f"cut lower bound above P*: {int((pr.lb_cut > pr.P_star + 1e-6).sum())}   "
        f"rounded upper certificate below P*: {int((pr.certified_ub < pr.P_star - 1e-6).sum())}")
    log(f"  configuration-LP bound above P*: {int((pr.lb_conf > pr.P_star + 1e-6).sum())}   below the cut bound: "
        f"{int((ds.lb_conf < ds.lb_cut - 1e-6).sum())}")
    log(f"  lower bound equal to P*: rounded LP {int((abs(pr.lb_lp_rounded - pr.P_star) < 1e-6).sum())}, "
        f"cut bound {int((abs(pr.lb_cut - pr.P_star) < 1e-6).sum())}, "
        f"configuration LP {int((abs(pr.lb_conf - pr.P_star) < 1e-6).sum())} of {len(pr)}")
    un = ds[~ds.proven]
    if len(un):
        log(f"  {len(un)} instances not closed by Gurobi: configuration-LP bound minus Gurobi's dual bound, mean "
            f"{(un.lb_conf - un.gurobi_bound).mean():.2f} kW (above it on {int((un.lb_conf > un.gurobi_bound + 1e-6).sum())}), "
            f"best algorithm / Gurobi incumbent {(un.best_algorithm / un.P_star).mean():.4f}")
    log(f"  dependent-rounding Las Vegas: mean rounds {ds.dr_lv_rounds.mean():.2f} vs mean bound {ds.dr_lv_bound.mean():.2f}")
    log(f"  a priori optimality certificates (cut LB = rounded UB): {int(pr.a_priori_optimal.sum())} of {len(pr)}; "
        f"best certificate equal to P*: {int((abs(pr.certified_ub - pr.P_star) < 1e-6).sum())}")
    log(f"  optimality proved without a MILP solver (best algorithm = cut LB): {int(pr.solver_free_optimal.sum())} of {len(pr)}; "
        f"best algorithm equal to P*: {int((abs(pr.best_algorithm - pr.P_star) < 1e-6).sum())}")
    log("  by family (instances / a priori optimal / solver-free optimal / certified LB = P*):")
    for fam, g in pr.groupby("family"):
        log(f"    {fam:9s} {len(g):3d} / {int(g.a_priori_optimal.sum()):3d} / {int(g.solver_free_optimal.sum()):3d} / "
            f"{int((abs(g.certified_lb - g.P_star) < 1e-6).sum()):3d}")
    if "mt_mean_steps" in ds:
        above = int((ds.mt_mean_steps > ds.mt_expected_bound + 3 * ds.mt_se + 1e-9).sum())
        log(f"  Moser-Tardos (200 runs per instance): mean resamplings {ds.mt_mean_steps.mean():.3f} vs mean "
            f"theoretical bound {ds.mt_expected_bound.mean():.3f}; instances whose mean exceeds the bound by "
            f"more than 3 standard errors: {above}")
    ratio = pr[CERTS + EXTRA + [a for a, _ in ALGS]].div(pr.P_star, axis=0)
    ratio["family"] = pr.family
    log("\n[tightness, small] mean ratio to P*, by family:")
    log(ratio.groupby("family").mean().T.round(3).to_string())

    large = [(fam, n, T) for fam in FAMILIES for n, T in (((200, 96), (1000, 96)) if quick else ((200, 96), (500, 96), (1000, 96), (2000, 144)))]
    rows = []
    for fam, n, T in large:
        inst = B.Instance(FAMILIES[fam](n, T, rng))
        t0 = time.time()
        row = {"family": fam, "n": n, "T": T}
        row.update(evaluate(inst, env, rng, mt=20 if n <= 500 else 0, kmax=8, conf=n <= 200))
        row["sec"] = time.time() - t0
        rows.append(row)
        log(f"[large] {fam:9s} n={n:4d} T={T:3d}: best certificate / LP = "
            f"{row['best_certificate'] / row['P_LP']:.4f}, "
            f"best algorithm / LP = {min(row[a] for a, _ in ALGS if a in row) / row['P_LP']:.4f}, "
            f"certified LB / LP = {row['certified_lb'] / row['P_LP']:.4f}, solver-free optimal = {row['solver_free_optimal']}, "
            f"published LLL-Bernstein / LP = {row['published_bernstein_lll'] / row['P_LP']:.3f} ({row['sec']:.0f}s)")
    dl = pd.DataFrame(rows)
    dl.to_csv(os.path.join(OUT, "validation_large.csv"), index=False)
    cols = [c for c in CERTS + EXTRA + [a for a, _ in ALGS] if c in dl]
    ratio = dl[cols].div(dl.P_LP, axis=0)
    ratio["family"] = dl.family; ratio["n"] = dl.n
    log("\n[tightness, large] ratio to P_LP (an upper bound on the ratio to P*):")
    log(ratio.groupby(["family", "n"]).mean().T.round(4).to_string())
    ratio = dl[cols].div(dl.certified_lb, axis=0)
    ratio["family"] = dl.family; ratio["n"] = dl.n
    log("\n[tightness, large] ratio to the cut lower bound (a sharper upper bound on the ratio to P*):")
    log(ratio.groupby(["family", "n"]).mean().T.round(4).to_string())
    open(os.path.join(OUT, "validation_log.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
