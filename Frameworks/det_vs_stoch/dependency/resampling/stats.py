"""
stats.py -- statistical primitives of the regime comparison.

Paired data: the i-th values of the two regimes are the SAME block (one
replication of one instance, or one instance), which is what makes the
signed-rank test the right one.

  paired_test         one paired comparison: Wilcoxon p-value, Cliff's delta with
                      a bootstrap CI, mean difference with a bootstrap CI
  holm                Holm step-down adjustment of a family of p-values
  bootstrap_ci        percentile bootstrap CI of a mean
  cliff_delta         Cliff's delta effect size, and its interpretation

`higher_is_better` selects the direction (True for the hypervolume).
"""

import numpy as np
from scipy.stats import wilcoxon


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


# ── Percentile bootstrap CI ───────────────────────────────────────────────────
def bootstrap_ci(values, n_boot: int = 2000, ci: float = 0.95, seed: int = 0):
    """Percentile bootstrap CI for the mean of `values`. Returns (mean, lo, hi).
    Makes no distributional assumption beyond resampling with replacement
    from the observed data -- more defensible than a normal-approximation CI
    (1.96*SEM) for a quantity like hypervolume, which is bounded below at 0
    and can be skewed at the replication counts used here, unlike SEM-based
    CIs which implicitly assume approximate normality of the sampling
    distribution of the mean."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if n == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    alpha = 1.0 - ci
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


# ── One paired comparison with confidence intervals ──────────────────────────
def paired_test(x, y, higher_is_better: bool = True, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired comparison of y against x (same blocks): two-sided Wilcoxon
    signed-rank p-value, Cliff's delta d(y, x) (> 0: y tends to be larger) with
    a percentile bootstrap CI over blocks, and the mean difference y - x with
    its bootstrap CI. `better` names the side favoured by the median
    difference ("y", "x" or "tie"). Blocks with a NaN on either side are
    dropped; fewer than 3 blocks give NaN statistics."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    L = min(len(x), len(y))
    x, y = x[:L], y[:L]
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = len(x)
    out = {"n": n, "p_value": float("nan"), "cliff_delta": float("nan"), "delta_lo": float("nan"),
           "delta_hi": float("nan"), "mean_diff": float("nan"), "diff_lo": float("nan"),
           "diff_hi": float("nan"), "better": "tie"}
    if n < 3:
        return out
    diff = y - x
    out["mean_diff"] = float(diff.mean())
    out["cliff_delta"] = cliff_delta(y, x)
    if np.allclose(x, y):
        out.update(p_value=1.0, delta_lo=0.0, delta_hi=0.0, diff_lo=0.0, diff_hi=0.0)
        return out
    try:
        out["p_value"] = float(wilcoxon(y, x, zero_method="wilcox", alternative="two-sided")[1])
    except Exception:
        pass
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_diff = diff[idx].mean(axis=1)
    out["diff_lo"], out["diff_hi"] = (float(v) for v in np.percentile(boot_diff, [2.5, 97.5]))
    boot_delta = np.array([cliff_delta(y[i], x[i]) for i in idx[: min(n_boot, 1000)]])
    out["delta_lo"], out["delta_hi"] = (float(v) for v in np.percentile(boot_delta, [2.5, 97.5]))
    md = float(np.median(diff))
    if md != 0.0:
        out["better"] = ("y" if md > 0 else "x") if higher_is_better else ("y" if md < 0 else "x")
    return out


# ── Holm step-down adjustment ────────────────────────────────────────────────
def holm(p_values) -> np.ndarray:
    """Holm-adjusted p-values (same order as the input; NaN stays NaN): reject
    H_i at level alpha when its adjusted p-value is below alpha. Controls the
    family-wise error rate like Bonferroni, with more power."""
    p = np.asarray(p_values, dtype=float)
    adj = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    m = len(ok)
    if m == 0:
        return adj
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj
