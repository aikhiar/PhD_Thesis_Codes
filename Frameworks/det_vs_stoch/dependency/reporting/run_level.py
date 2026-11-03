"""
Run-level analysis of the regime comparison (written next to the main report,
in "Run-level analysis/").

The main report compares ONE pooled front per regime (the non-dominated union
of the evaluation runs) over R realizations: its per-instance test is
conditional on those two fronts and does not see how much they would change if
the search were run again with other seeds. Here every evaluation run is an
independent front: for instance i and regime r, run k has the run-level
hypervolume h_{r,k} = mean over the R realizations of the hypervolume of run k's
realized front (one normalization shared by all run fronts of all regimes, per
realization).
  per instance   Mann-Whitney U (two-sided) on the K vs K run-level values,
                 Cliff's delta and mean gap; Holm over the instances. With K
                 runs per regime the smallest attainable p-value is
                 2 / C(2K, K); when, after Holm, it cannot fall below alpha
                 (K = 5 with 20 instances), per-instance significance is
                 unattainable and the report says so: read the effect sizes
                 and the across-instance verdict
  across         Wilcoxon signed-rank on the per-instance run-level means
                 (paired over the instances): the verdict that includes the
                 run-to-run variability of the search
The pooled-front verdicts of the main report are listed next to it.
"""
from __future__ import annotations

import json
import os
from math import comb
from typing import Dict, List

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon


def _cliff(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    gt = sum(int(x > y) for x in a for y in b)
    lt = sum(int(x < y) for x in a for y in b)
    return float(gt - lt) / (len(a) * len(b)) if len(a) and len(b) else float("nan")


def _holm(p: Dict[int, float]) -> Dict[int, float]:
    keys = sorted(p, key=lambda k: p[k])
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def run_level_report(per_instance_hv, hv_runs, regimes: List[str], out_dir: str, alpha: float = 0.05,
                     verbose: bool = True) -> dict:
    """Writes run_level_summary.json and .md in out_dir; returns the summary
    ({} when no run-level hypervolumes are available, e.g. a run resumed from a
    cache written before the run-level analysis existed)."""
    ref, ch = regimes[0], regimes[1]
    usable = [i for i in sorted(per_instance_hv)
              if hv_runs.get(i) and hv_runs[i].get(ref) and hv_runs[i].get(ch)]
    if not usable:
        if verbose:
            print("  [run-level] skipped: no per-run fronts (resumed from an older cache?)", flush=True)
        return {}
    rows, p_pool, p_run = [], {}, {}
    K = None
    for inst in usable:
        pooled_ref = np.asarray(per_instance_hv[inst][ref], dtype=float)
        pooled_ch = np.asarray(per_instance_hv[inst][ch], dtype=float)
        d = pooled_ch - pooled_ref
        try:
            p_pool[inst] = float(wilcoxon(pooled_ch, pooled_ref).pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            p_pool[inst] = 1.0
        r_ref = np.nanmean(np.asarray(hv_runs[inst][ref], dtype=float), axis=0)
        r_ch = np.nanmean(np.asarray(hv_runs[inst][ch], dtype=float), axis=0)
        K = min(len(r_ref), len(r_ch)) if K is None else min(K, len(r_ref), len(r_ch))
        if np.allclose(r_ch, r_ref):
            p_run[inst] = 1.0
        else:
            p_run[inst] = float(mannwhitneyu(r_ch, r_ref, alternative="two-sided").pvalue)
        rows.append({"instance": int(inst),
                     "pooled_mean_" + ref: float(pooled_ref.mean()), "pooled_mean_" + ch: float(pooled_ch.mean()),
                     "pooled_" + ch + "_wins_of_R": int(np.sum(pooled_ch > pooled_ref)), "R": int(len(pooled_ref)),
                     "runs_" + ref: r_ref.tolist(), "runs_" + ch: r_ch.tolist(),
                     "run_mean_" + ref: float(r_ref.mean()), "run_mean_" + ch: float(r_ch.mean()),
                     "run_sd_" + ref: float(r_ref.std(ddof=1)) if len(r_ref) > 1 else float("nan"),
                     "run_sd_" + ch: float(r_ch.std(ddof=1)) if len(r_ch) > 1 else float("nan"),
                     "cliff_run": _cliff(r_ch, r_ref)})
    hp_pool, hp_run = _holm(p_pool), _holm(p_run)
    min_p = 2.0 / comb(2 * K, K) if K else 1.0
    attainable = min_p * len(usable) < alpha

    def verdict(m_ch, m_ref, p):
        if p >= alpha:
            return "no difference"
        return f"{ch} better" if m_ch > m_ref else f"{ref} better"

    for r in rows:
        i = r["instance"]
        r["p_pooled_holm"], r["p_run_holm"] = hp_pool[i], hp_run[i]
        r["verdict_pooled"] = verdict(r["pooled_mean_" + ch], r["pooled_mean_" + ref], hp_pool[i])
        r["verdict_run"] = (verdict(r["run_mean_" + ch], r["run_mean_" + ref], hp_run[i]) if attainable
                            else "not testable per instance (see across)")
    m_ref = np.array([r["run_mean_" + ref] for r in rows])
    m_ch = np.array([r["run_mean_" + ch] for r in rows])
    try:
        p_across = float(wilcoxon(m_ch, m_ref).pvalue) if np.any(m_ch != m_ref) else 1.0
    except ValueError:
        p_across = 1.0
    gap = m_ch - m_ref
    summary = {"reference": ref, "challenger": ch, "runs_per_regime": K, "alpha": alpha,
               "per_instance_testable": attainable, "min_attainable_p_per_instance": min_p,
               "instances": rows,
               "across_instances": {"mean_gap": float(gap.mean()),
                                    "mean_relative_gap_pct": float(100 * np.mean(gap / np.where(m_ref == 0, np.nan, m_ref))),
                                    "p_wilcoxon": p_across,
                                    "instances_challenger_higher": int(np.sum(gap > 0)), "n_instances": len(rows),
                                    "verdict": verdict(m_ch.mean(), m_ref.mean(), p_across)}}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_level_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    a = summary["across_instances"]
    lines = [f"# Run-level analysis: {ch} vs {ref}", "",
             f"{K} independent evaluation runs per regime and instance; run-level hypervolume = mean over "
             f"the R realizations of the hypervolume of one run's realized front.", ""]
    if not attainable:
        lines += [f"With {K} runs per regime the smallest attainable per-instance p-value is {min_p:.3g}, which "
                  f"cannot fall below {alpha} after Holm over {len(rows)} instances: per-instance verdicts are not "
                  f"testable, read the effect sizes and the across-instance verdict (raise n_eval_runs, e.g. to 10, "
                  f"for per-instance tests).", ""]
    lines += ["| instance | pooled: mean HV ref / ch | pooled: ch wins of R | pooled verdict (Holm) | "
              "runs: mean +- sd ref | runs: mean +- sd ch | Cliff (runs) | run-level verdict (Holm) |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['instance']} | {r['pooled_mean_' + ref]:.4f} / {r['pooled_mean_' + ch]:.4f} | "
                     f"{r['pooled_' + ch + '_wins_of_R']}/{r['R']} | {r['verdict_pooled']} (p={r['p_pooled_holm']:.2g}) | "
                     f"{r['run_mean_' + ref]:.4f} +- {r['run_sd_' + ref]:.4f} | "
                     f"{r['run_mean_' + ch]:.4f} +- {r['run_sd_' + ch]:.4f} | {r['cliff_run']:+.2f} | "
                     f"{r['verdict_run']} (p={r['p_run_holm']:.2g}) |")
    lines += ["", f"Across instances (run-level means, Wilcoxon paired over the instances): {a['verdict']}; "
              f"mean gap {a['mean_gap']:+.4f} ({a['mean_relative_gap_pct']:+.2f} %), {ch} higher on "
              f"{a['instances_challenger_higher']}/{a['n_instances']} instances, p = {a['p_wilcoxon']:.3g}"]
    with open(os.path.join(out_dir, "run_level_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    if verbose:
        print("  [run-level] " + lines[-1], flush=True)
    return summary
