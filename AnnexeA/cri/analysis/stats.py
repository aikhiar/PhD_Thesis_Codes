"""
Statistical tests on per-instance values (blocks = instances, treatments = methods). Runs of a
stochastic method are first averaged per instance, so that seeds are not counted as independent
instances (pooling all runs would overstate the evidence).

  friedman            Friedman chi-square and the Iman-Davenport F correction, average ranks
  nemenyi_cd          Nemenyi critical difference (studentized range, Demsar 2006)
  pairwise_wilcoxon   Wilcoxon signed-rank for every pair, Holm-adjusted (and Benjamini-Hochberg)
  versus_control      Wilcoxon of every method against one control, Holm-adjusted
  effect sizes        Cliff's delta, Vargha-Delaney A12, matched-pairs rank-biserial r
  bootstrap_ci        percentile bootstrap CI of a mean
  win_tie_loss        counts per pair with a tolerance
  sign_test           exact binomial sign test
All functions take a DataFrame (index: instance, columns: methods) of a lower-is-better metric.
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats

TIE = 1e-9


def _complete(M):
    return M.dropna(axis=0, how="any")


def avg_ranks(M):
    M = _complete(M)
    return M.rank(axis=1, method="average").mean(axis=0).sort_values()


def friedman(M):
    M = _complete(M)
    N, k = M.shape
    if N < 2 or k < 3:
        return {"N": N, "k": k, "chi2": np.nan, "p": np.nan, "F_ID": np.nan, "p_ID": np.nan,
                "ranks": avg_ranks(M) if N else pd.Series(dtype=float)}
    R = M.rank(axis=1, method="average")
    rj = R.mean(axis=0)
    # tie-corrected Friedman statistic
    chi2 = 12 * N / (k * (k + 1)) * ((rj - (k + 1) / 2) ** 2).sum()
    ties = sum(((np.unique(row, return_counts=True)[1] ** 3) - np.unique(row, return_counts=True)[1]).sum()
               for row in M.values)
    corr = 1 - ties / (N * k * (k * k - 1))
    chi2 = chi2 / corr if corr > 0 else np.nan
    p = stats.chi2.sf(chi2, k - 1) if chi2 == chi2 else np.nan
    F = (N - 1) * chi2 / (N * (k - 1) - chi2) if chi2 == chi2 and N * (k - 1) - chi2 > 0 else np.nan
    pF = stats.f.sf(F, k - 1, (k - 1) * (N - 1)) if F == F else np.nan
    return {"N": N, "k": k, "chi2": float(chi2), "p": float(p), "F_ID": float(F) if F == F else np.nan,
            "p_ID": float(pF) if pF == pF else np.nan, "ranks": rj.sort_values()}


def nemenyi_cd(k, N, alpha=0.05):
    if k < 2 or N < 1:
        return np.nan
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * N)))


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    pv = p[ok][idx]
    m = len(pv)
    adj = np.maximum.accumulate((m - np.arange(m)) * pv)
    adj = np.minimum(adj, 1.0)
    tmp = np.empty(m); tmp[idx] = adj
    out[ok] = tmp
    return out


def bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = stats.false_discovery_control(p[ok])
    return out


def wilcoxon(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if len(d) == 0 or np.all(np.abs(d) <= TIE):
        return 1.0, 0.0
    nz = np.abs(d[np.abs(d) > TIE])
    ties = len(nz) < len(d) or len(np.unique(np.round(nz, 9))) < len(nz)
    # exact null distribution without ties (n <= 50); otherwise the tie-corrected normal approximation
    method = "exact" if (not ties and len(d) <= 50) else "approx"
    try:
        res = stats.wilcoxon(d, zero_method="pratt", alternative="two-sided", method=method)
        return float(res.pvalue), float(res.statistic)
    except ValueError:
        return 1.0, 0.0


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if not len(a) or not len(b):
        return np.nan
    gt = (a[:, None] > b[None, :] + TIE).sum()
    lt = (a[:, None] < b[None, :] - TIE).sum()
    return float((gt - lt) / (len(a) * len(b)))


def a12(a, b):
    """Vargha-Delaney A12: probability that a is LOWER (better) than b, ties count one half."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if not len(a) or not len(b):
        return np.nan
    lt = (a[:, None] < b[None, :] - TIE).sum()
    eq = (np.abs(a[:, None] - b[None, :]) <= TIE).sum()
    return float((lt + 0.5 * eq) / (len(a) * len(b)))


def rank_biserial(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    d = d[np.abs(d) > TIE]
    if not len(d):
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d < 0].sum() - r[d > 0].sum()) / r.sum())     # > 0: a better (lower)


def magnitude_cliff(d):
    d = abs(d)
    return "negligible" if d < 0.147 else "small" if d < 0.33 else "medium" if d < 0.474 else "large"


def bootstrap_ci(x, B=2000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (B, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def sign_test(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    w, l = int((d < -TIE).sum()), int((d > TIE).sum())
    if w + l == 0:
        return 1.0
    return float(stats.binomtest(w, w + l, 0.5).pvalue)


def win_tie_loss(a, b, tol=TIE):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    return int((d < -tol).sum()), int((np.abs(d) <= tol).sum()), int((d > tol).sum())


def pairwise(M, alpha=0.05):
    """All pairs: Wilcoxon (Holm and BH adjusted), sign test, W/T/L, effect sizes."""
    M = _complete(M)
    rows = []
    for a, b in itertools.combinations(M.columns, 2):
        p, st = wilcoxon(M[a], M[b])
        w, t, l = win_tie_loss(M[a], M[b])
        cd = cliffs_delta(M[a], M[b])
        rows.append({"A": a, "B": b, "N": len(M), "mean_A": M[a].mean(), "mean_B": M[b].mean(),
                     "median_diff": float(np.median(M[a] - M[b])), "W": st, "p": p, "p_sign": sign_test(M[a], M[b]),
                     "win_A": w, "tie": t, "win_B": l, "cliff_delta": cd, "cliff_mag": magnitude_cliff(cd) if cd == cd else "",
                     "A12": a12(M[a], M[b]), "r_rb": rank_biserial(M[a], M[b])})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(df.p.values)
        df["p_bh"] = bh(df.p.values)
        df["significant"] = df.p_holm < alpha
    return df


def versus_control(M, control, alpha=0.05):
    M = _complete(M)
    rows = []
    for b in M.columns:
        if b == control:
            continue
        p, st = wilcoxon(M[control], M[b])
        w, t, l = win_tie_loss(M[control], M[b])
        cd = cliffs_delta(M[control], M[b])
        rows.append({"control": control, "method": b, "N": len(M), "mean_control": M[control].mean(),
                     "mean_method": M[b].mean(), "p": p, "wins": w, "ties": t, "losses": l, "cliff_delta": cd,
                     "cliff_mag": magnitude_cliff(cd) if cd == cd else "", "A12": a12(M[control], M[b])})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(df.p.values)
        df["significant"] = df.p_holm < alpha
    return df


def pvalue_matrix(pw, methods, col="p_holm"):
    P = pd.DataFrame(np.nan, index=methods, columns=methods)
    for _, r in pw.iterrows():
        P.loc[r.A, r.B] = P.loc[r.B, r.A] = r[col]
    return P


def describe(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return {"mean": np.nan, "sd": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan, "min": np.nan,
                "max": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "count": 0}
    lo, hi = bootstrap_ci(x)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "median": float(np.median(x)), "q1": float(np.quantile(x, 0.25)), "q3": float(np.quantile(x, 0.75)),
            "min": float(x.min()), "max": float(x.max()), "ci_lo": lo, "ci_hi": hi, "count": int(len(x))}
