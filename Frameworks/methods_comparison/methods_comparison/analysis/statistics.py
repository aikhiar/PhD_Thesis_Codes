"""
statistics.py  —  the statistical tests of a study, per tested block (the
objective or hypervolume, and the execution time where the chapter tests it),
written as JSON and as a text report.

The unit of analysis is the INSTANCE: each method is summarised by its
per-instance mean over the runs, giving vectors aligned by instance. Each study
runs the tests its chapter reports (config["statistics"]):
  * Friedman test, average ranks and Kendall's W        (friedman)
  * Nemenyi critical difference                         (nemenyi)
  * pairwise Wilcoxon signed-rank tests, Holm or
    Bonferroni adjusted                                 (wilcoxon, pairwise_correction)
  * Cliff's delta of every pair                         (cliff_delta)
  * against the control method: Friedman post-hoc and
    Wilcoxon tests with the Holm adjustment, and A12    (holm)
  * Bayesian signed-rank test against the control       (bayesian)
The per-instance Mann-Whitney tests against the control (mann_whitney) are the
markers of the instance tables (report_tables.py).
"""

import os
from itertools import combinations
import numpy as np

from ..utils.io import save_json, slug
from ..utils.stats import (friedman_test, wilcoxon_pairs,
                           nemenyi_critical_difference, check_nan_coverage,
                           cliff_interpretation, friedman_vs_control,
                           wilcoxon_vs_control, bayes_signed_rank)


def run_all_statistical_tests(problem, blocks, paths, stats_cfg, verbose=True, control=None):
    """`blocks`: list of (block_name, scores, label, higher_is_better) to test.
    `control`: internal name of the control method (tests against it when
    stats_cfg["holm"] or stats_cfg["bayesian"] is set). Returns {slug: results}."""
    results = {}
    for block, scores, label, hib in blocks:
        if scores is None:
            continue
        if verbose:
            print(f"  [Stats] {label}")
        results[slug(block)] = _run_block(problem, scores, label, slug(block),
                                          hib, paths, stats_cfg, verbose, control)
    return results


def _run_block(problem, scores, label, s, higher_is_better,
               paths, stats_cfg, verbose, control=None):
    alpha = stats_cfg.get("alpha", 0.05)

    method_names = list(scores.keys())
    all_ids = sorted({i for m in scores for i in scores[m]})
    labels = {m: problem.label(m) for m in method_names}

    # aligned per-instance means (paired across instances)
    agg = {}
    for m in method_names:
        agg[labels[m]] = [_safe_mean(scores[m].get(i, [])) for i in all_ids]

    nan_info = check_nan_coverage(agg)
    if verbose:
        for w in nan_info["warnings"]:
            print(f"      {w}")
    # methods with too few valid instances leave the tests (listwise deletion
    # would otherwise discard every instance on which they failed, for ALL methods)
    for m in nan_info["excluded"]:
        agg.pop(m, None)

    res = {"label": label, "higher_is_better": higher_is_better,
           "nan_info": nan_info, "n_instances": len(all_ids)}

    # ── Friedman + Nemenyi ────────────────────────────────────────────────────
    fr = {}
    if stats_cfg.get("friedman", True):
        fr = friedman_test(agg, alpha, higher_is_better)
        res["friedman"] = fr
        save_json(fr, os.path.join(paths[f"{s}/friedman"], "friedman.json"), code=problem.code)
        if verbose and "conclusion" in fr:
            print(f"      Friedman: {fr['conclusion']}")

    cd = float("nan")
    if stats_cfg.get("nemenyi", False) and fr.get("average_ranks"):
        cd = nemenyi_critical_difference(fr["n_methods"], fr["n_blocks"], alpha)
        res["nemenyi"] = {"critical_difference": cd,
                          "average_ranks": fr["average_ranks"]}
        save_json(res["nemenyi"], os.path.join(paths[f"{s}/nemenyi"], "nemenyi.json"),
                  code=problem.code)

    # ── Wilcoxon signed-rank (paired) + Cliff's delta ────────────────────────
    wx = {}
    if stats_cfg.get("wilcoxon", True):
        wx = wilcoxon_pairs(agg, alpha, higher_is_better=higher_is_better,
                            correction=stats_cfg.get("pairwise_correction", "holm"))
        res["wilcoxon"] = wx
        out = {k: v for k, v in wx.items()
               if stats_cfg.get("cliff_delta", True) or k not in ("cliff_delta_matrix",)}
        save_json(out, os.path.join(paths[f"{s}/wilcoxon"], "wilcoxon.json"), code=problem.code)
        if stats_cfg.get("cliff_delta", True):
            save_json({"method_names": wx.get("method_names", []),
                       "cliff_delta_matrix": wx.get("cliff_delta_matrix", []),
                       "significant_pairs": wx.get("significant_pairs", [])},
                      os.path.join(paths[f"{s}/cliffdelta"], "cliff_delta.json"),
                      code=problem.code)
        if verbose:
            print(f"      Wilcoxon: {len(wx.get('significant_pairs', []))} significant pair(s)")

    # ── comparisons against the control method ───────────────────────────────
    ctl = labels.get(control) if control else None
    vs = {}
    if (ctl and ctl in agg and len(agg) >= 2
            and (stats_cfg.get("holm", False) or stats_cfg.get("bayesian", False))):
        vs["control"] = ctl
        if stats_cfg.get("holm", False):
            if fr.get("average_ranks"):
                vs["friedman_holm"] = friedman_vs_control(
                    fr["average_ranks"], fr["n_blocks"], ctl, alpha)
            vs["wilcoxon_holm"] = wilcoxon_vs_control(agg, ctl, alpha, higher_is_better)
        if stats_cfg.get("bayesian", False):
            vs["bayesian"] = _bayes_vs_control(agg, ctl, higher_is_better, stats_cfg)
        save_json(vs, os.path.join(paths[f"{s}/control"], "tests_vs_control.json"),
                  code=problem.code)
    res["vs_control"] = vs
    res["aligned_means"] = agg

    # ── text report ───────────────────────────────────────────────────────────
    _write_report(label, problem, fr, wx, cd, nan_info, all_ids, alpha,
                  paths[f"{s}/stats"], vs, stats_cfg.get("cliff_delta", True))
    return res


def _bayes_vs_control(agg, ctl, higher_is_better, stats_cfg):
    """Bayesian signed-rank test of the control against every other method on
    the per-instance means. Differences are oriented so that a POSITIVE value
    means the control is better, and relative (d / mean magnitude), with the
    region of practical equivalence `bayes_rope`."""
    rope = float(stats_cfg.get("bayes_rope", 0.01))
    n_samples = int(stats_cfg.get("bayes_samples", 20000))
    c = np.asarray(agg[ctl], dtype=float)
    out = []
    for m, v in agg.items():
        if m == ctl:
            continue
        v = np.asarray(v, dtype=float)
        L = min(len(c), len(v))
        d = (c[:L] - v[:L]) if higher_is_better else (v[:L] - c[:L])
        scale = (np.abs(c[:L]) + np.abs(v[:L])) / 2.0
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(scale > 0, d / scale, 0.0)
        d = d[np.isfinite(d)]
        (p_m, p_rope, p_c), _ = bayes_signed_rank(d, rope, n_samples=n_samples, seed=0)
        out.append({"method": m, "n": int(len(d)), "rope": rope, "rope_type": "relative",
                    "p_control_better": p_c, "p_equivalent": p_rope,
                    "p_method_better": p_m})
    return {"control": ctl, "comparisons": out}


# ── report writer ────────────────────────────────────────────────────────────
def _write_report(label, problem, fr, wx, cd, nan_info, all_ids, alpha, stat_dir, vs=None,
                  with_cliff=True):
    L = ["=" * 72,
         f"  STATISTICAL TESTS REPORT  --  {label}",
         f"  Study     : {problem.code} ({problem.name})",
         f"  Type      : {'Multi-objective' if problem.is_multiobjective else 'Mono-objective'}",
         f"  Instances : {len(all_ids)}",
         f"  Alpha     : {alpha}",
         "=" * 72]

    if nan_info["warnings"]:
        L += ["", "  DATA QUALITY:", ""] + [f"  {w}" for w in nan_info["warnings"]]

    if fr:
        L += ["", "-" * 72, "  FRIEDMAN TEST (omnibus)", "-" * 72]
        if "statistic" in fr:
            L += [f"  Chi2 = {fr['statistic']:.6f}   p = {fr['p_value']:.6g}   "
                  f"Kendall's W = {fr.get('kendall_w', float('nan')):.3f}",
                  f"  {fr['conclusion']}", "",
                  "  Average ranks (1 = best):"]
            for m, r in sorted(fr.get("average_ranks", {}).items(), key=lambda x: x[1]):
                mark = "  <- BEST" if m == fr.get("best_method") else ""
                L.append(f"    {m:32s} {r:7.4f}{mark}")
            if np.isfinite(cd):
                L += ["", f"  Nemenyi critical difference (CD) = {cd:.4f}",
                      "  Two methods whose average ranks differ by >= CD are "
                      "significantly different."]
        else:
            L.append(f"  {fr.get('conclusion', '(n/a)')}")

    if wx:
        L += ["", "-" * 72,
              "  WILCOXON SIGNED-RANK (pairwise, paired across instances)",
              "-" * 72,
              f"  Correction: {wx.get('correction', 'holm')}; a pair is significant "
              f"when its adjusted p < {alpha}"
              + (f" (equivalently raw p < {wx.get('corrected_alpha', alpha):.6g})"
                 if wx.get('correction') == 'bonferroni' else "")]
        sig = wx.get("significant_pairs", [])
        if sig:
            L += ["", "  Significant pairs (better method in brackets):"]
            for p in sig:
                L.append(f"    {p['method_a']:24s} vs {p['method_b']:24s}  "
                         f"p={p['p_value']:.4g}"
                         + (f"  d={p['cliff_delta']:+.3f} [{p['interpretation']}]"
                            if with_cliff else "")
                         + f"  -> {p['better']}")
        else:
            L.append("  No significant pairs after correction.")

    # ── Cliff's delta: EVERY pair, not gated by significance ────────────────
    names = wx.get("method_names", [])
    p_mat = wx.get("p_matrix", [])
    d_mat = wx.get("cliff_delta_matrix", [])
    hib = wx.get("higher_is_better", False)
    corr_alpha = wx.get("corrected_alpha", alpha)
    if with_cliff and len(names) >= 2 and p_mat and d_mat:
        L += ["", "-" * 72,
              "  CLIFF'S DELTA (effect size) -- ALL PAIRS",
              "-" * 72,
              "  delta in [-1,1]; |delta| < .147 negligible, < .330 small,",
              "  < .474 medium, else large (Romano et al., 2006).",
              "  '*' marks pairs significant at the corrected alpha above;",
              "  '->' names the favoured method by direction of delta alone.",
              ""]
        for i, j in combinations(range(len(names)), 2):
            delta = float(d_mat[i][j])
            p = float(p_mat[i][j])
            interp = cliff_interpretation(delta)
            star = "*" if np.isfinite(p) and p < corr_alpha else " "
            if abs(delta) < 1e-12:
                favor = "(no directional difference)"
            else:
                i_favored = (delta > 0) if hib else (delta < 0)
                favor = f"-> {names[i] if i_favored else names[j]}"
            L.append(f"  {star} {names[i]:24s} vs {names[j]:24s}  "
                     f"d={delta:+.3f} [{interp:10s}]  p={p:.4g}  {favor}")

    L += _control_lines(vs or {})
    L += ["", "=" * 72, ""]
    os.makedirs(stat_dir, exist_ok=True)
    fname = f"{problem.code}_Statistical Tests Report.txt"
    with open(os.path.join(stat_dir, fname), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def _safe_mean(vals):
    clean = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(clean)) if clean else np.nan


def _control_lines(vs):
    """Report section for the comparisons against the control method."""
    if not vs or not vs.get("control"):
        return []
    c = vs["control"]
    L = ["", "-" * 72, f"  AGAINST THE CONTROL METHOD: {c}", "-" * 72]
    fh = vs.get("friedman_holm", {})
    if fh.get("comparisons"):
        L += ["  Friedman post-hoc vs control (z test on average ranks, Holm):"]
        for r in fh["comparisons"]:
            L.append(f"    {r['method']:28s} rank={r['rank']:.3f}  z={r['z']:+.3f}  "
                     f"p={r['p_value']:.4g}  p_Holm={r['p_holm']:.4g}"
                     + (f"  -> {r['better']} better" if r['significant'] else "  (n.s.)"))
    wh = vs.get("wilcoxon_holm", {})
    if wh.get("comparisons"):
        L += ["", "  Wilcoxon signed-rank vs control (per-instance means, Holm):",
              "  A12 = probability that the control beats the method on an instance."]
        for r in wh["comparisons"]:
            L.append(f"    {r['method']:28s} n={r['n']:3d}  p={r['p_value']:.4g}  "
                     f"p_Holm={r['p_holm']:.4g}  A12={r['a12_control']:.3f} "
                     f"[{r['a12_interpretation']}]" + ("  *" if r['significant'] else ""))
    by = vs.get("bayesian", {})
    if by.get("comparisons"):
        L += ["", "  Bayesian signed-rank test vs control (Benavoli et al. 2017):"]
        for r in by["comparisons"]:
            L.append(f"    {r['method']:28s} P(control better)={r['p_control_better']:.3f}  "
                     f"P(equivalent)={r['p_equivalent']:.3f}  "
                     f"P(method better)={r['p_method_better']:.3f}  "
                     f"(rope {r['rope']:g}, {r['rope_type']})")
    return L
