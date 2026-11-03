"""
stats.py  —  statistical primitives (minimisation convention aware).

All inputs are dicts ``{method_label: [score_per_instance, ...]}`` where the
i-th element of every method is the SAME instance (paired/blocked data).  This
alignment is essential: Friedman and Wilcoxon are blocked/paired tests.

`higher_is_better` selects ranking direction:
  * False (default) — lower score is better (mono objective, gap, time);
  * True            — higher score is better (multi-objective hypervolume).
"""

import numpy as np
from itertools import combinations
from typing import Dict, List
from scipy.stats import friedmanchisquare, wilcoxon, rankdata


# ── NaN coverage report ──────────────────────────────────────────────────────
def check_nan_coverage(data: Dict[str, List[float]],
                       min_valid_ratio: float = 0.5) -> dict:
    warns, excluded = [], []
    for mname, vals in data.items():
        arr = np.asarray(vals, dtype=float)
        n_total = len(arr)
        n_nan = int(np.isnan(arr).sum())
        ratio = (n_total - n_nan) / n_total if n_total else 0.0
        if n_nan:
            warns.append(f"Method '{mname}': {n_nan}/{n_total} NaN "
                         f"({100*(1-ratio):.1f}% missing)")
        if ratio < min_valid_ratio:
            excluded.append(mname)
            warns.append(f"--> '{mname}' EXCLUDED "
                         f"(valid {ratio:.0%} < {min_valid_ratio:.0%})")
    return {"warnings": warns, "excluded": excluded}


# ── aligned matrix (listwise deletion preserves blocking) ────────────────────
def _aligned_matrix(data: Dict[str, List[float]]):
    """Return (method_names, M) where M is (n_methods, n_common_blocks), no NaN."""
    names = list(data.keys())
    if not names:
        return [], np.empty((0, 0))
    L = min(len(v) for v in data.values())
    M = np.array([np.asarray(data[m][:L], dtype=float) for m in names])  # (k, L)
    keep = ~np.isnan(M).any(axis=0)            # drop blocks where ANY method is NaN
    return names, M[:, keep]


# ── Friedman omnibus ──────────────────────────────────────────────────────────
def friedman_test(data: Dict[str, List[float]], alpha: float = 0.05,
                  higher_is_better: bool = False) -> dict:
    """Friedman test on aligned per-instance scores. Needs >=3 methods, >=2 blocks."""
    names, M = _aligned_matrix(data)
    k, N = (M.shape if M.size else (len(names), 0))

    if k < 3 or N < 2:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "significant": False, "average_ranks": {}, "best_method": None,
                "n_methods": k, "n_blocks": int(N),
                "conclusion": f"Skipped (need >=3 methods & >=2 blocks; "
                              f"got {k} methods, {N} blocks)"}

    # per-block ranks with proper tie handling; rank 1 = best
    rank_input = -M if higher_is_better else M
    ranks = np.vstack([rankdata(rank_input[:, b]) for b in range(N)]).T  # (k, N)
    avg = ranks.mean(axis=1)

    if np.all(np.ptp(M, axis=0) == 0):
        # every method ties on every block: the statistic is 0/0 in scipy (NaN);
        # report the exact answer instead -- no difference, W = 0
        return {"statistic": 0.0, "p_value": 1.0, "significant": False,
                "average_ranks": {names[i]: float(avg[i]) for i in range(k)},
                "best_method": names[0], "n_methods": k, "n_blocks": int(N),
                "kendall_w": 0.0, "higher_is_better": higher_is_better,
                "conclusion": "All methods tie on every block (no difference)"}
    stat, p = friedmanchisquare(*[M[i] for i in range(k)])

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < alpha),
        "average_ranks": {names[i]: float(avg[i]) for i in range(k)},
        "best_method": names[int(np.argmin(avg))],
        "n_methods": k, "n_blocks": int(N),
        "kendall_w": kendall_w(float(stat), int(N), k),
        "higher_is_better": higher_is_better,
        "conclusion": (f"Significant difference (p={p:.4g} < {alpha})"
                       if p < alpha else
                       f"No significant difference (p={p:.4g} >= {alpha})"),
    }


# ── Nemenyi critical difference ───────────────────────────────────────────────
# Demšar (2006) q_alpha for the two-tailed Nemenyi test (k = number of methods).
_Q_ALPHA = {
    0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
           9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
           15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544},
    0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780,
           9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030, 13: 3.077, 14: 3.120,
           15: 3.159, 16: 3.196, 17: 3.230, 18: 3.261, 19: 3.291, 20: 3.319},
}


def nemenyi_critical_difference(k: int, n_blocks: int, alpha: float = 0.05) -> float:
    """CD = q_alpha * sqrt(k(k+1)/(6N)).  Two methods within CD are indistinguishable."""
    # alpha = 0.05 -> q_0.05 table; alpha = 0.10 -> q_0.10 table
    table = _Q_ALPHA[0.10] if alpha > 0.075 else _Q_ALPHA[0.05]
    q = table.get(k, table[max(table)])
    if n_blocks <= 0:
        return float("nan")
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n_blocks)))


# ── Wilcoxon signed-rank (paired, pairwise) + Cliff's delta ──────────────────
def wilcoxon_pairs(data: Dict[str, List[float]], alpha: float = 0.05,
                   bonferroni: bool = True, higher_is_better: bool = False,
                   correction: str = None) -> dict:
    """
    Pairwise PAIRED comparison across instances (Wilcoxon signed-rank), with
    Cliff's delta and Vargha-Delaney A12 as effect sizes and an explicit
    'better' method per pair. `correction` ("bonferroni", "holm", "none";
    None -> "bonferroni" if `bonferroni` else "none") adjusts for the number
    of pairs; a pair is significant when its adjusted p-value is < alpha.
    """
    if correction is None:
        correction = "bonferroni" if bonferroni else "none"
    names = list(data.keys())
    n = len(names)
    p_mat = np.ones((n, n))
    d_mat = np.zeros((n, n))
    a_mat = np.full((n, n), 0.5)
    pairs = list(combinations(range(n), 2))
    corr_alpha = alpha / len(pairs) if (correction == "bonferroni" and pairs) else alpha
    warns, sig = [], []
    tested = []

    for i, j in pairs:
        x = np.asarray(data[names[i]], dtype=float)
        y = np.asarray(data[names[j]], dtype=float)
        L = min(len(x), len(y))
        x, y = x[:L], y[:L]
        keep = ~(np.isnan(x) | np.isnan(y))    # pairwise listwise deletion
        x, y = x[keep], y[keep]
        if len(x) < 3 or np.allclose(x, y):
            warns.append(f"Pair ({names[i]}, {names[j]}) skipped "
                         f"(n={len(x)} paired, or identical).")
            continue
        try:
            _, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        except Exception:
            p = float("nan")
        delta = cliff_delta(x, y)
        p_mat[i, j] = p_mat[j, i] = float(p)
        d_mat[i, j] = delta
        d_mat[j, i] = -delta
        a = a12(x, y, higher_is_better)
        a_mat[i, j], a_mat[j, i] = a, 1.0 - a
        tested.append((i, j, float(p), delta, x, y))

    # adjusted p-values over the tested pairs
    raw = [t[2] for t in tested]
    if correction == "holm":
        adj = holm_adjust(raw)
    elif correction == "bonferroni":
        adj = [min(1.0, p * len(pairs)) if np.isfinite(p) else p for p in raw]
    else:
        adj = raw
    padj = np.ones((n, n))
    sig_mat = np.zeros((n, n), dtype=bool)
    for (i, j, p, delta, x, y), pa in zip(tested, adj):
        padj[i, j] = padj[j, i] = pa
        if np.isfinite(pa) and pa < alpha:
            sig_mat[i, j] = sig_mat[j, i] = True
            md = np.median(x - y)
            if higher_is_better:
                better = names[i] if md > 0 else names[j]
            else:
                better = names[i] if md < 0 else names[j]
            sig.append({"method_a": names[i], "method_b": names[j],
                        "p_value": float(p), "p_adjusted": float(pa),
                        "cliff_delta": float(delta),
                        "interpretation": cliff_interpretation(delta),
                        "a12": float(a_mat[i, j]),
                        "better": better})

    return {"method_names": names, "p_matrix": p_mat.tolist(),
            "p_adjusted_matrix": padj.tolist(), "significant_matrix": sig_mat.tolist(),
            "cliff_delta_matrix": d_mat.tolist(), "a12_matrix": a_mat.tolist(),
            "corrected_alpha": corr_alpha, "correction": correction,
            "warnings": warns, "significant_pairs": sig,
            "higher_is_better": higher_is_better}


# ── Cliff's delta ─────────────────────────────────────────────────────────────
def cliff_delta(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    return float((gt - lt) / (len(x) * len(y))) if len(x) and len(y) else 0.0


def cliff_interpretation(d: float) -> str:
    d = abs(d)
    if d < 0.147:
        return "negligible"
    if d < 0.330:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


# ══════════════════════════════════════════════════════════════════════════════
#  tests against the control method, Bayesian test, bootstrap, markers
# ══════════════════════════════════════════════════════════════════════════════
def holm_adjust(pvals) -> list:
    """Holm (1979) step-down adjusted p-values (monotone, capped at 1);
    NaN p-values stay NaN and do not count in the number of tests."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    idx = [i for i in range(len(p)) if np.isfinite(p[i])]
    m = len(idx)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda k: p[k])):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def kendall_w(chi2: float, n_blocks: int, k: int) -> float:
    """Kendall's coefficient of concordance from the Friedman statistic:
    W = chi2 / (N (k - 1)), in [0, 1] (effect size of the Friedman test)."""
    if not np.isfinite(chi2) or n_blocks <= 0 or k < 2:
        return float("nan")
    return float(chi2 / (n_blocks * (k - 1)))


def a12(x, y, higher_is_better: bool = False) -> float:
    """Vargha & Delaney (2000) A12: probability that a value of x is BETTER
    than a value of y (ties count 1/2). 0.5 = no difference."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if not len(x) or not len(y):
        return float("nan")
    if higher_is_better:
        better = np.sum(x[:, None] > y[None, :])
    else:
        better = np.sum(x[:, None] < y[None, :])
    ties = np.sum(x[:, None] == y[None, :])
    return float((better + 0.5 * ties) / (len(x) * len(y)))


def a12_interpretation(a: float) -> str:
    """Vargha & Delaney thresholds on |A12 - 0.5|: < .06 negligible, < .14
    small, < .21 medium, else large."""
    if not np.isfinite(a):
        return "n/a"
    d = abs(a - 0.5)
    return ("negligible" if d < 0.06 else "small" if d < 0.14
            else "medium" if d < 0.21 else "large")


def friedman_vs_control(avg_ranks: Dict[str, float], n_blocks: int, control: str,
                        alpha: float = 0.05) -> dict:
    """Post-hoc of the Friedman test against ONE control method (Demsar 2006;
    Garcia & Herrera 2008): z = (R_i - R_c) / sqrt(k (k + 1) / (6 N)), two-sided
    normal p-values, Holm step-down adjustment over the k - 1 comparisons."""
    from scipy.stats import norm
    k = len(avg_ranks)
    if control not in avg_ranks or k < 2 or n_blocks <= 0:
        return {"control": control, "comparisons": [],
                "note": "not applicable (control missing or < 2 methods)"}
    se = np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    others = [m for m in avg_ranks if m != control]
    z = [(avg_ranks[m] - avg_ranks[control]) / se for m in others]
    p = [float(2 * norm.sf(abs(v))) for v in z]
    padj = holm_adjust(p)
    comps = []
    for m, zi, pi, pa in zip(others, z, p, padj):
        comps.append({"method": m, "rank": float(avg_ranks[m]),
                      "z": float(zi), "p_value": pi, "p_holm": float(pa),
                      "significant": bool(pa < alpha),
                      "better": (control if zi > 0 else m) if pa < alpha else None})
    return {"control": control, "control_rank": float(avg_ranks[control]),
            "alpha": alpha, "comparisons": comps}


def wilcoxon_vs_control(data: Dict[str, List[float]], control: str,
                        alpha: float = 0.05, higher_is_better: bool = False) -> dict:
    """Wilcoxon signed-rank of every method against the control on the
    per-instance values (paired), Holm-adjusted, with A12 and Cliff's delta
    (control vs method)."""
    if control not in data:
        return {"control": control, "comparisons": []}
    xc = np.asarray(data[control], dtype=float)
    rows = []
    for m in data:
        if m == control:
            continue
        y = np.asarray(data[m], dtype=float)
        L = min(len(xc), len(y))
        keep = ~(np.isnan(xc[:L]) | np.isnan(y[:L]))
        a, b = xc[:L][keep], y[:L][keep]
        if len(a) < 3 or np.allclose(a, b):
            p = float("nan")
        else:
            try:
                p = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
            except Exception:
                p = float("nan")
        rows.append({"method": m, "n": int(len(a)), "p_value": p,
                     "a12_control": a12(a, b, higher_is_better),
                     "cliff_delta": cliff_delta(a, b) if len(a) else float("nan")})
    adj = holm_adjust([r["p_value"] for r in rows])
    for r, pa in zip(rows, adj):
        r["p_holm"] = pa
        r["significant"] = bool(np.isfinite(pa) and pa < alpha)
        r["a12_interpretation"] = a12_interpretation(r["a12_control"])
    return {"control": control, "alpha": alpha, "comparisons": rows}


def bayes_signed_rank(diff, rope: float, prior: float = 0.5,
                      n_samples: int = 20000, seed: int = 0):
    """Bayesian signed-rank test (Benavoli, Corani, Demsar & Zaffalon 2017).

    `diff` = per-instance differences (positive = the FIRST method is better,
    after the caller has oriented them). A Dirichlet process posterior with a
    pseudo-observation at 0 of weight `prior` is sampled; in each sample the
    probability mass of the Walsh averages (d_i + d_j) / 2 falling below
    -rope, inside [-rope, rope], and above rope is computed, and the sample
    votes for the largest. Returns (p_second_better, p_rope, p_first_better)
    and the (n_samples, 3) matrix of per-sample masses (for the simplex plot).
    """
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return (float("nan"),) * 3, np.empty((0, 3))
    z = np.concatenate(([0.0], d))
    S = z[:, None] + z[None, :]
    left = (S < -2 * rope).astype(float)
    right = (S > 2 * rope).astype(float)
    mid = 1.0 - left - right
    rng = np.random.default_rng(seed)
    alpha_vec = np.concatenate(([prior], np.ones(len(d))))
    out = np.empty((n_samples, 3))
    chunk = 5000
    for s in range(0, n_samples, chunk):
        w = rng.dirichlet(alpha_vec, size=min(chunk, n_samples - s))
        out[s:s + len(w), 0] = ((w @ left) * w).sum(axis=1)
        out[s:s + len(w), 1] = ((w @ mid) * w).sum(axis=1)
        out[s:s + len(w), 2] = ((w @ right) * w).sum(axis=1)
    win = np.bincount(np.argmax(out, axis=1), minlength=3) / len(out)
    return (float(win[0]), float(win[1]), float(win[2])), out


def bootstrap_mean_ci(values, n_boot: int = 2000, level: float = 0.95, seed: int = 0):
    """Percentile bootstrap CI of the mean (NaNs dropped); (mean, lo, hi)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    a = (1 - level) / 2
    return float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def marker_vs_control(x, c, alpha: float = 0.05, higher_is_better: bool = False) -> str:
    """Per-instance comparison of a method (runs `x`) with the control (runs
    `c`): two-sided Mann-Whitney U at `alpha`. Returns "+" (method
    significantly better than the control), "-" (significantly worse) or "="
    (no significant difference / not testable)."""
    from scipy.stats import mannwhitneyu
    x = np.asarray(x, dtype=float); c = np.asarray(c, dtype=float)
    x, c = x[np.isfinite(x)], c[np.isfinite(c)]
    if len(x) == 0 or len(c) == 0:
        return "="
    if len(x) + len(c) < 3 or (np.ptp(np.concatenate([x, c])) == 0):
        return "="
    try:
        p = mannwhitneyu(x, c, alternative="two-sided").pvalue
    except Exception:
        return "="
    if not np.isfinite(p) or p >= alpha:
        return "="
    a = a12(x, c, higher_is_better)
    return "+" if a > 0.5 else "-" if a < 0.5 else "="


# ── Ranking-direction self-check (remark 2) ───────────────────────────────────
def verify_ranking_direction() -> None:
    """Confirms friedman_test/wilcoxon_pairs rank correctly in BOTH directions:
    higher_is_better=False (mono objective / gap / time -- lower score wins)
    and higher_is_better=True (multi-objective hypervolume -- higher score
    wins). Uses synthetic per-block scores with a known best/worst method by
    construction, so the correct answer is known independently of the code
    under test. Raises AssertionError with a diagnostic message on any
    mismatch; does nothing (silently) on success.
    """
    rng = np.random.default_rng(0)
    n_blocks = 30

    # BEST has the lowest raw values, WORST the highest, MID in between, with
    # per-block noise small enough to never invert the ordering.
    best = 1.0 + rng.normal(0, 0.01, n_blocks)
    mid = 2.0 + rng.normal(0, 0.01, n_blocks)
    worst = 3.0 + rng.normal(0, 0.01, n_blocks)
    data = {"BEST": best.tolist(), "MID": mid.tolist(), "WORST": worst.tolist()}

    # -- mono convention: lower raw value = better --------------------------
    res_lo = friedman_test(data, higher_is_better=False)
    if res_lo["best_method"] != "BEST":
        raise AssertionError(
            f"[verify_ranking_direction] MONO FAIL -- friedman_test(higher_is_better=False) "
            f"picked best_method={res_lo['best_method']!r}, expected 'BEST' (lowest raw "
            f"values). average_ranks={res_lo['average_ranks']}"
        )
    ranks_lo = res_lo["average_ranks"]
    if not (ranks_lo["BEST"] < ranks_lo["MID"] < ranks_lo["WORST"]):
        raise AssertionError(
            f"[verify_ranking_direction] MONO FAIL -- average_ranks not monotonic with "
            f"raw value order under higher_is_better=False: {ranks_lo}"
        )

    pairs_lo = wilcoxon_pairs(data, higher_is_better=False)
    better_lo = {(p["method_a"], p["method_b"]): p["better"] for p in pairs_lo["significant_pairs"]}
    for (a, b), expected in [(("BEST", "WORST"), "BEST"), (("BEST", "MID"), "BEST"), (("MID", "WORST"), "MID")]:
        got = better_lo.get((a, b))
        if got != expected:
            raise AssertionError(
                f"[verify_ranking_direction] MONO FAIL -- wilcoxon_pairs(higher_is_better=False) "
                f"pair ({a},{b}) reported better={got!r}, expected {expected!r}. "
                f"significant_pairs={pairs_lo['significant_pairs']}"
            )

    # -- multi convention: higher raw value (e.g. hypervolume) = better -----
    res_hi = friedman_test(data, higher_is_better=True)
    if res_hi["best_method"] != "WORST":  # "WORST" holds the highest raw values here
        raise AssertionError(
            f"[verify_ranking_direction] MULTI FAIL -- friedman_test(higher_is_better=True) "
            f"picked best_method={res_hi['best_method']!r}, expected 'WORST' (highest raw "
            f"values, i.e. the best hypervolume). average_ranks={res_hi['average_ranks']}"
        )
    ranks_hi = res_hi["average_ranks"]
    if not (ranks_hi["WORST"] < ranks_hi["MID"] < ranks_hi["BEST"]):
        raise AssertionError(
            f"[verify_ranking_direction] MULTI FAIL -- average_ranks not inverted relative "
            f"to the mono case under higher_is_better=True: {ranks_hi}"
        )

    pairs_hi = wilcoxon_pairs(data, higher_is_better=True)
    better_hi = {(p["method_a"], p["method_b"]): p["better"] for p in pairs_hi["significant_pairs"]}
    for (a, b), expected in [(("BEST", "WORST"), "WORST"), (("BEST", "MID"), "MID"), (("MID", "WORST"), "WORST")]:
        got = better_hi.get((a, b))
        if got != expected:
            raise AssertionError(
                f"[verify_ranking_direction] MULTI FAIL -- wilcoxon_pairs(higher_is_better=True) "
                f"pair ({a},{b}) reported better={got!r}, expected {expected!r}. "
                f"significant_pairs={pairs_hi['significant_pairs']}"
            )


if __name__ == "__main__":
    verify_ranking_direction()
    print("verify_ranking_direction: PASSED (mono lower-is-better and multi "
          "higher-is-better both rank correctly)")
