"""
stats.py  —  non-parametric statistical-comparison primitives, physically
copied and adapted from methods_comparison's `methods_comparison/utils/stats.py` (this
framework is standalone -- nothing here imports across frameworks).

Implements the Demsar (2006) protocol ("Statistical Comparisons of
Classifiers over Multiple Data Sets", JMLR 7:1-30) for comparing several
learners on paired/blocked measurements: a Friedman omnibus test across all
candidates, followed by pairwise Wilcoxon signed-rank tests with a
Bonferroni-corrected significance threshold and Cliff's delta as effect
size.

All inputs are dicts ``{model_name: [value_per_block, ...]}`` where the
i-th element of every model is the SAME block (paired data) -- for
`surrogate_trainer` a "block" is one held-out validation sample, and
the value is that model's absolute prediction error on it. This alignment
is essential: Friedman and Wilcoxon are blocked/paired tests, and pairing
by validation sample (rather than by k-fold CV fold) is what keeps the
blocks genuinely close to independent -- CV folds share most of their
training data with each other, which biases a naive paired test toward
false positives (Dietterich, 1998).

`higher_is_better` is always False here (lower prediction error is
better); kept as a parameter for symmetry with the methods_comparison original and in
case a future caller scores something where higher is better.
"""

import numpy as np
from itertools import combinations
from typing import Dict, List
from scipy.stats import friedmanchisquare, wilcoxon, rankdata


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
    """Friedman test on aligned per-block scores. Needs >=3 methods, >=2 blocks."""
    names, M = _aligned_matrix(data)
    k, N = (M.shape if M.size else (len(names), 0))

    if k < 3 or N < 2:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "significant": False, "average_ranks": {}, "best_method": None,
                "n_methods": k, "n_blocks": int(N),
                "conclusion": f"Skipped (need >=3 methods & >=2 blocks; "
                              f"got {k} methods, {N} blocks)"}

    stat, p = friedmanchisquare(*[M[i] for i in range(k)])

    # per-block ranks with proper tie handling; rank 1 = best
    rank_input = -M if higher_is_better else M
    ranks = np.vstack([rankdata(rank_input[:, b]) for b in range(N)]).T  # (k, N)
    avg = ranks.mean(axis=1)

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < alpha),
        "average_ranks": {names[i]: float(avg[i]) for i in range(k)},
        "best_method": names[int(np.argmin(avg))],
        "n_methods": k, "n_blocks": int(N),
        "higher_is_better": higher_is_better,
        "conclusion": (f"Significant difference (p={p:.4g} < {alpha})"
                       if p < alpha else
                       f"No significant difference (p={p:.4g} >= {alpha})"),
    }


# ── Wilcoxon signed-rank (paired, pairwise) + Cliff's delta ──────────────────
def wilcoxon_pairs(data: Dict[str, List[float]], alpha: float = 0.05,
                   bonferroni: bool = True, higher_is_better: bool = False) -> dict:
    """
    Pairwise PAIRED comparison across blocks (Wilcoxon signed-rank), with
    Cliff's delta as effect size and an explicit 'better' method per pair.
    """
    names = list(data.keys())
    n = len(names)
    p_mat = np.ones((n, n))
    d_mat = np.zeros((n, n))
    pairs = list(combinations(range(n), 2))
    corr_alpha = alpha / len(pairs) if (bonferroni and pairs) else alpha
    warns, sig = [], []

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

        if np.isfinite(p) and p < corr_alpha:
            md = np.median(x - y)
            if higher_is_better:
                better = names[i] if md > 0 else names[j]
            else:
                better = names[i] if md < 0 else names[j]
            sig.append({"method_a": names[i], "method_b": names[j],
                        "p_value": float(p), "cliff_delta": float(delta),
                        "interpretation": cliff_interpretation(delta),
                        "better": better})

    return {"method_names": names, "p_matrix": p_mat.tolist(),
            "cliff_delta_matrix": d_mat.tolist(), "corrected_alpha": corr_alpha,
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


# ── Ranking-direction self-check ──────────────────────────────────────────────
def verify_ranking_direction() -> None:
    """Confirms friedman_test/wilcoxon_pairs rank correctly (lower error is
    better, the only convention this framework ever uses). Uses synthetic
    per-block scores with a known best/worst method by construction, so the
    correct answer is known independently of the code under test. Raises
    AssertionError with a diagnostic message on any mismatch; does nothing
    (silently) on success."""
    rng = np.random.default_rng(0)
    n_blocks = 30

    best = 1.0 + rng.normal(0, 0.01, n_blocks)
    mid = 2.0 + rng.normal(0, 0.01, n_blocks)
    worst = 3.0 + rng.normal(0, 0.01, n_blocks)
    data = {"BEST": best.tolist(), "MID": mid.tolist(), "WORST": worst.tolist()}

    res = friedman_test(data, higher_is_better=False)
    if res["best_method"] != "BEST":
        raise AssertionError(
            f"[verify_ranking_direction] FAIL -- friedman_test picked "
            f"best_method={res['best_method']!r}, expected 'BEST' (lowest error). "
            f"average_ranks={res['average_ranks']}"
        )
    ranks = res["average_ranks"]
    if not (ranks["BEST"] < ranks["MID"] < ranks["WORST"]):
        raise AssertionError(
            f"[verify_ranking_direction] FAIL -- average_ranks not monotonic "
            f"with raw error order: {ranks}"
        )

    pairs = wilcoxon_pairs(data, higher_is_better=False)
    better = {(p["method_a"], p["method_b"]): p["better"] for p in pairs["significant_pairs"]}
    for (a, b), expected in [(("BEST", "WORST"), "BEST"), (("BEST", "MID"), "BEST"), (("MID", "WORST"), "MID")]:
        got = better.get((a, b))
        if got != expected:
            raise AssertionError(
                f"[verify_ranking_direction] FAIL -- wilcoxon_pairs pair ({a},{b}) "
                f"reported better={got!r}, expected {expected!r}. "
                f"significant_pairs={pairs['significant_pairs']}"
            )


if __name__ == "__main__":
    verify_ranking_direction()
    print("verify_ranking_direction: PASSED")
