"""
Reporting of the regime comparison: tables, figures and statistical tests.
One call, `generate_report`, writes, for one chapter's run, what the chapter
reports (the problem module's REPORT):

    <problem_dir>/
      Data/                 raw exports (hypervolume per replication, planned and
                            realized fronts, tuned parameters)
      Metrics/              realized hypervolume per instance, per size class and
                            overall; Chapter 6: promise vs delivery and the
                            realized objective values
      Statistical Tests/    per-instance tests over the R replications; Chapter 6:
                            across-instance tests over the instances
      Figures/<category>/   PNG/ and SVG/ subfolders and a caption file each

Every table is written as .csv and as a LaTeX (booktabs) .tex file. Every
folder carries a `_HOW_IT_IS_COMPUTED.txt` caption.

Regimes: two, the deterministic planner (the reference) and the stochastic
planner (the challenger).
"""

import csv
import itertools
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import style
from dependency.resampling import indicators as IND
from dependency.resampling.hypervolume import compute_hypervolume_batch
from dependency.resampling.stats import bootstrap_ci, cliff_interpretation, holm, paired_test

style.apply_thesis_style()

SIZE_CLASS_ORDER = ["small", "medium", "moderate", "large"]
TIE_EPS = 1e-9
_REGIME_PALETTE = [style.PALETTE[i] for i in (0, 1, 3, 4, 6, 7)]


# ============================================================================
# HELPERS
# ============================================================================

def _pretty_problem_name(code: str) -> str:
    if re.fullmatch(r"[A-Z]\d+", code):
        return code
    m = re.match(r"^([a-zA-Z]+?)(\d+)$", code)
    if m:
        return f"{m.group(1).capitalize()} {m.group(2)}"
    return code.replace("_", " ").title()


def _save_all_formats(fig, out_dir: str, stem: str, figures_cfg: Dict[str, Any]) -> None:
    for fmt in figures_cfg.get("formats", ["png", "svg"]):
        d = os.path.join(out_dir, fmt.upper())
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{stem}.{fmt}"), dpi=figures_cfg.get("dpi", 400), bbox_inches="tight", format=fmt)
    plt.close(fig)


def _write_caption(out_dir: str, title: str, paragraphs: List[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    lines = [title, "=" * len(title), ""]
    for p in paragraphs:
        lines += [p, ""]
    with open(os.path.join(out_dir, "_HOW_IT_IS_COMPUTED.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _fmt(v, digits: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{digits}g}" if np.isfinite(v) else ""
    return str(v)


def _tex_escape(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(a, b)
    return text


def write_table(rows: List[Dict[str, Any]], columns: Sequence[Tuple[str, str]], out_dir: str, stem: str) -> None:
    """rows -> <stem>.csv (all keys of `columns`, full precision) and <stem>.tex
    (booktabs tabular, 4 significant digits)."""
    os.makedirs(out_dir, exist_ok=True)
    keys = [k for k, _ in columns]
    with open(os.path.join(out_dir, f"{stem}.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for row in rows:
            writer.writerow([_fmt(row.get(k), 8) for k in keys])
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule",
             " & ".join(_tex_escape(h) for _, h in columns) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(_tex_escape(_fmt(row.get(k))) for k in keys) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out_dir, f"{stem}.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _class_lookup(instance_size_classes: Dict[str, List[int]]) -> Dict[int, str]:
    return {i: c for c, ids in instance_size_classes.items() for i in ids}


def _ordered_classes(instance_size_classes: Dict[str, List[int]]) -> List[str]:
    present = set(instance_size_classes)
    ordered = [c for c in SIZE_CLASS_ORDER if c in present]
    return ordered + sorted(present - set(ordered))


def _scopes(instance_size_classes, ids: List[int]) -> List[Tuple[str, List[int]]]:
    """[(class, its instances present in this run), ..., ("overall", all)]."""
    present = set(ids)
    out = [(c, sorted(i for i in instance_size_classes[c] if i in present)) for c in _ordered_classes(instance_size_classes)]
    return [(c, v) for c, v in out if v] + [("overall", sorted(ids))]


def _finite_mean(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def _rel_change(new: float, ref: float) -> float:
    """100 (new - ref) / |ref|; NaN when ref is 0 or not finite."""
    if not (np.isfinite(new) and np.isfinite(ref)) or ref == 0:
        return float("nan")
    return 100.0 * (new - ref) / abs(ref)


def _hv_gap(ref_mean: float, new_mean: float) -> float:
    """100 (HV_new - HV_ref) / HV_ref (positive: the challenger realized more)."""
    if not (np.isfinite(ref_mean) and np.isfinite(new_mean)) or ref_mean <= 0:
        return float("nan")
    return 100.0 * (new_mean - ref_mean) / ref_mean


def _adjust(p_values, correction: str) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if correction == "bonferroni":
        m = int(np.isfinite(p).sum())
        return np.where(np.isfinite(p), np.minimum(1.0, p * max(m, 1)), np.nan)
    if correction in (None, "none"):
        return p
    return holm(p)


# ============================================================================
# DERIVED QUANTITIES (computed once, used by tables and figures)
# ============================================================================

def compute_derived(regimes, objective_names, per_instance_hv, per_instance_realized, fronts, eps,
                    promise: bool) -> Dict[str, Any]:
    """Per instance and regime: the mean realized hypervolume; with `promise`
    (Chapter 6), also the realized knee point and best value of every objective
    and the planned-vs-realized hypervolumes in one normalisation shared by every
    planned and realized front of the instance."""
    K = len(objective_names)
    out = {"hv_mean": {}, "knee": {}, "best": {}, "promise": {}}
    for inst, hv in per_instance_hv.items():
        out["hv_mean"][inst] = {r: _finite_mean(hv[r]) for r in regimes}
        if not promise:
            continue
        realized = per_instance_realized[inst]
        R = max(len(realized[r]) for r in regimes)
        knee = {r: [] for r in regimes}
        best = {r: [] for r in regimes}
        for rep in range(R):
            raw = {r: (realized[r][rep] if rep < len(realized[r]) else []) for r in regimes}
            norm, ideal, nadir = IND.normalise_together(raw)
            for r in regimes:
                has = len(norm[r]) > 0
                knee[r].append(IND.knee_point(raw[r], ideal, nadir) if has else np.full(K, np.nan))
                best[r].append(np.asarray(raw[r], dtype=float).min(axis=0) if has else np.full(K, np.nan))
        out["knee"][inst] = {r: np.nanmean(np.asarray(knee[r], dtype=float), axis=0) for r in regimes}
        out["best"][inst] = {r: np.nanmean(np.asarray(best[r], dtype=float), axis=0) for r in regimes}

        batch = {}
        for r in regimes:
            batch[f"planned::{r}"] = [fronts[r][inst].objectives]
            batch[f"realized::{r}"] = [f for f in realized[r]]
        hvb = compute_hypervolume_batch(batch, eps=eps)
        planned_best = {r: (np.asarray(fronts[r][inst].objectives, dtype=float).min(axis=0)
                            if fronts[r][inst].objectives else np.full(K, np.nan)) for r in regimes}
        out["promise"][inst] = {r: {"planned_hv": hvb[f"planned::{r}"][0],
                                    "realized_hv": _finite_mean(hvb[f"realized::{r}"]),
                                    "planned_best": planned_best[r],
                                    "realized_best": out["best"][inst][r]} for r in regimes}
    return out


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def per_instance_tests(regimes, per_instance_hv, stats_cfg) -> Dict[str, Any]:
    """Per instance: paired Wilcoxon over the R replications (the same
    realization at each replication), Cliff's delta with a bootstrap CI;
    p-values adjusted over the instances (Bonferroni in Chapters 2 to 4, Holm
    in Chapter 6)."""
    ref = regimes[0]
    alpha, corr = stats_cfg.get("alpha", 0.05), stats_cfg.get("correction", "holm")
    ids = sorted(per_instance_hv)
    out: Dict[str, Any] = {"pairs": {}}
    for c in regimes[1:]:
        res = {i: paired_test(per_instance_hv[i][ref], per_instance_hv[i][c], higher_is_better=True,
                              n_boot=stats_cfg.get("bootstrap_samples", 2000)) for i in ids}
        adj = _adjust([res[i]["p_value"] for i in ids], corr)
        for i, p_adj in zip(ids, adj):
            r = res[i]
            r["p_adjusted"] = float(p_adj)
            r["significant"] = bool(np.isfinite(p_adj) and p_adj < alpha)
            r["winner"] = (c if r["better"] == "y" else ref if r["better"] == "x" else "tie") if r["significant"] \
                else "not significant"
        out["pairs"][c] = res
    return out


def across_instance_tests(regimes, derived, instance_size_classes, stats_cfg) -> Dict[str, Any]:
    """Chapter 6. Per scope (each size class, overall): paired Wilcoxon on the
    per-instance mean hypervolumes (the instances are the blocks), Cliff's delta
    and mean difference with bootstrap CIs. The class-level p-values are adjusted
    together; the overall test is the primary one."""
    ref = regimes[0]
    alpha, corr = stats_cfg.get("alpha", 0.05), stats_cfg.get("correction", "holm")
    ids = sorted(derived["hv_mean"])
    scopes = _scopes(instance_size_classes, ids)
    rows = []
    for c in regimes[1:]:
        block = []
        for scope, sids in scopes:
            x = [derived["hv_mean"][i][ref] for i in sids]
            y = [derived["hv_mean"][i][c] for i in sids]
            t = paired_test(x, y, higher_is_better=True, n_boot=stats_cfg.get("bootstrap_samples", 2000))
            gaps = [_hv_gap(a, b) for a, b in zip(x, y)]
            block.append({"indicator": "Hypervolume", "challenger": c, "reference": ref, "scope": scope,
                          "n_instances": t["n"], "mean_reference": _finite_mean(x), "mean_challenger": _finite_mean(y),
                          "mean_difference": t["mean_diff"], "diff_ci_lo": t["diff_lo"], "diff_ci_hi": t["diff_hi"],
                          "mean_relative_gap_pct": _finite_mean(gaps),
                          "cliff_delta": t["cliff_delta"], "delta_ci_lo": t["delta_lo"], "delta_ci_hi": t["delta_hi"],
                          "effect": cliff_interpretation(t["cliff_delta"]) if np.isfinite(t["cliff_delta"]) else "",
                          "p_value": t["p_value"], "_better": t["better"]})
        class_rows = [b for b in block if b["scope"] != "overall"]
        adj = _adjust([b["p_value"] for b in class_rows], corr)
        for b, p_adj in zip(class_rows, adj):
            b["p_adjusted"] = float(p_adj)
        overall = next(b for b in block if b["scope"] == "overall")
        overall["p_adjusted"] = overall["p_value"]
        for b in block:
            sig = bool(np.isfinite(b["p_adjusted"]) and b["p_adjusted"] < alpha)
            b["significant"] = sig
            b["winner"] = (c if b["_better"] == "y" else ref if b["_better"] == "x" else "tie") if sig else "not significant"
            b.pop("_better")
        rows += block
    return {"rows": rows}


# ============================================================================
# TABLES
# ============================================================================

def _tables(code, regimes, labels, objective_names, per_instance_hv, per_instance_realized, fronts, derived, tests,
            across, instance_size_classes, best_params_by_regime, report_cfg, dirs) -> Dict[str, Any]:
    ref = regimes[0]
    class_of = _class_lookup(instance_size_classes)
    ids = sorted(per_instance_hv)
    scopes = _scopes(instance_size_classes, ids)
    obj_keys = [re.sub(r"[^0-9a-zA-Z]+", "_", n).strip("_").lower() for n in objective_names]

    # ---- Data/
    rows = []
    for i in ids:
        R = max(len(per_instance_hv[i][r]) for r in regimes)
        for rep in range(R):
            row = {"instance_id": i, "size_class": class_of.get(i, ""), "replication": rep}
            for r in regimes:
                vals = per_instance_hv[i][r]
                row[f"hv_{r}"] = vals[rep] if rep < len(vals) else float("nan")
            rows.append(row)
    write_table(rows, [("instance_id", "Instance"), ("size_class", "Class"), ("replication", "Replication")]
                + [(f"hv_{r}", f"HV {labels[r]}") for r in regimes], dirs["data"], "per_instance_hv_raw")

    for kind, source in (("planned", None), ("realized", per_instance_realized)):
        path = os.path.join(dirs["data"], f"{kind}_fronts_long.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["instance_id", "size_class", "regime"] + (["replication"] if source else []) + ["point"] + obj_keys)
            for i in ids:
                for r in regimes:
                    if source is None:
                        for p, vec in enumerate(fronts[r][i].objectives):
                            writer.writerow([i, class_of.get(i, ""), r, p] + [f"{v:.8g}" for v in vec])
                    else:
                        for rep, front in enumerate(source[i][r]):
                            for p, vec in enumerate(front):
                                writer.writerow([i, class_of.get(i, ""), r, rep, p] + [f"{v:.8g}" for v in vec])

    rows = []
    for r in regimes:
        for i in sorted(best_params_by_regime.get(r, {})):
            for pname, pval in best_params_by_regime[r][i].items():
                rows.append({"instance_id": i, "size_class": class_of.get(i, ""), "regime": r, "param": pname, "value": pval})
    write_table(rows, [("instance_id", "Instance"), ("size_class", "Class"), ("regime", "Regime"), ("param", "Parameter"),
                       ("value", "Value")], dirs["data"], "tuned_params")

    # ---- Metrics/: hypervolume per instance, per class, overall
    rows = []
    for i in ids:
        row = {"instance_id": i, "size_class": class_of.get(i, "")}
        for r in regimes:
            vals = np.asarray(per_instance_hv[i][r], dtype=float)
            vals = vals[np.isfinite(vals)]
            mean, lo, hi = bootstrap_ci(vals)
            row.update({f"n_{r}": len(vals), f"mean_hv_{r}": mean, f"ci_lo_{r}": lo, f"ci_hi_{r}": hi})
        for c in regimes[1:]:
            t = tests["pairs"][c][i]
            row[f"gap_pct_{c}"] = _hv_gap(row[f"mean_hv_{ref}"], row[f"mean_hv_{c}"])
            row[f"winner_{c}"] = t["winner"]
        rows.append(row)
    cols = [("instance_id", "Instance"), ("size_class", "Class")]
    for r in regimes:
        cols += [(f"mean_hv_{r}", f"HV {labels[r]}"), (f"ci_lo_{r}", "CI low"), (f"ci_hi_{r}", "CI high")]
    for c in regimes[1:]:
        cols += [(f"gap_pct_{c}", f"Gap {labels[c]} (%)"), (f"winner_{c}", "Significant winner")]
    write_table(rows, cols, dirs["metrics"], "per_instance_summary")

    summary_rows = []
    for scope, sids in scopes:
        row = {"scope": scope, "n_instances": len(sids)}
        for r in regimes:
            row[f"mean_hv_{r}"] = _finite_mean([derived["hv_mean"][i][r] for i in sids])
        for c in regimes[1:]:
            gaps = [_hv_gap(derived["hv_mean"][i][ref], derived["hv_mean"][i][c]) for i in sids]
            diffs = [derived["hv_mean"][i][c] - derived["hv_mean"][i][ref] for i in sids]
            row[f"mean_gap_pct_{c}"] = _finite_mean(gaps)
            row[f"wins_{c}"] = sum(1 for d in diffs if np.isfinite(d) and d >= TIE_EPS)
            row[f"losses_{c}"] = sum(1 for d in diffs if np.isfinite(d) and d <= -TIE_EPS)
            row[f"ties_{c}"] = len(sids) - row[f"wins_{c}"] - row[f"losses_{c}"]
            res = [tests["pairs"][c][i] for i in sids]
            row[f"sig_better_{c}"] = sum(1 for t in res if t["winner"] == c)
            row[f"sig_worse_{c}"] = sum(1 for t in res if t["winner"] == ref)
            row[f"mean_cliff_{c}"] = _finite_mean([t["cliff_delta"] for t in res])
            if across:
                ov = next(a for a in across["rows"] if a["challenger"] == c and a["scope"] == scope)
                row[f"across_p_{c}"] = ov["p_adjusted"]
                row[f"across_winner_{c}"] = ov["winner"]
        summary_rows.append(row)
    cols = [("scope", "Scope"), ("n_instances", "Instances")] + [(f"mean_hv_{r}", f"HV {labels[r]}") for r in regimes]
    for c in regimes[1:]:
        cols += [(f"mean_gap_pct_{c}", f"Mean gap {labels[c]} (%)"), (f"wins_{c}", "Wins"), (f"ties_{c}", "Ties"),
                 (f"losses_{c}", "Losses"), (f"sig_better_{c}", "Sig. better"), (f"sig_worse_{c}", "Sig. worse"),
                 (f"mean_cliff_{c}", "Mean delta")]
        if across:
            cols += [(f"across_p_{c}", "p (across instances)"), (f"across_winner_{c}", "Winner (across instances)")]
    write_table(summary_rows, cols, dirs["metrics"], "per_size_class_summary")
    overall = summary_rows[-1]
    write_table([overall], cols, dirs["metrics"], "overall_summary")

    if report_cfg.get("promise_vs_delivery"):
        # ---- Metrics/: realized objective values
        rows, agg = [], []
        for i in ids:
            for point, store in (("knee", derived["knee"]), ("best", derived["best"])):
                row = {"instance_id": i, "size_class": class_of.get(i, ""), "point": point}
                for r in regimes:
                    for k, ok in enumerate(obj_keys):
                        row[f"{ok}_{r}"] = float(store[i][r][k])
                for c in regimes[1:]:
                    for k, ok in enumerate(obj_keys):
                        row[f"{ok}_change_pct_{c}"] = _rel_change(store[i][c][k], store[i][ref][k])
                rows.append(row)
        cols = [("instance_id", "Instance"), ("size_class", "Class"), ("point", "Point")]
        for k, ok in enumerate(obj_keys):
            cols += [(f"{ok}_{r}", f"{objective_names[k]} {labels[r]}") for r in regimes]
            cols += [(f"{ok}_change_pct_{c}", f"{objective_names[k]} change {labels[c]} (%)") for c in regimes[1:]]
        write_table(rows, cols, dirs["metrics"], "realized_objectives_per_instance")
        for scope, sids in scopes:
            for point in ("knee", "best"):
                row = {"scope": scope, "point": point}
                for k, ok in enumerate(obj_keys):
                    for r in regimes:
                        row[f"{ok}_{r}"] = _finite_mean([x[f"{ok}_{r}"] for x in rows if x["instance_id"] in sids and x["point"] == point])
                    for c in regimes[1:]:
                        row[f"{ok}_change_pct_{c}"] = _finite_mean([x[f"{ok}_change_pct_{c}"] for x in rows
                                                                    if x["instance_id"] in sids and x["point"] == point])
                agg.append(row)
        write_table(agg, [("scope", "Scope"), ("point", "Point")] + cols[3:], dirs["metrics"], "realized_objectives_summary")

        # ---- Metrics/: promise vs delivery
        rows = []
        for i in ids:
            row = {"instance_id": i, "size_class": class_of.get(i, "")}
            for r in regimes:
                pr = derived["promise"][i][r]
                row[f"planned_hv_{r}"] = pr["planned_hv"]
                row[f"realized_hv_{r}"] = pr["realized_hv"]
                row[f"delivery_{r}"] = pr["realized_hv"] / pr["planned_hv"] if pr["planned_hv"] and np.isfinite(pr["planned_hv"]) else float("nan")
                for k, ok in enumerate(obj_keys):
                    row[f"{ok}_overpromise_pct_{r}"] = _rel_change(pr["realized_best"][k], pr["planned_best"][k])
            rows.append(row)
        cols = [("instance_id", "Instance"), ("size_class", "Class")]
        for r in regimes:
            cols += [(f"planned_hv_{r}", f"Planned HV {labels[r]}"), (f"realized_hv_{r}", f"Realized HV {labels[r]}"),
                     (f"delivery_{r}", f"Delivery {labels[r]}")]
            cols += [(f"{ok}_overpromise_pct_{r}", f"{objective_names[k]} realized vs planned {labels[r]} (%)")
                     for k, ok in enumerate(obj_keys)]
        write_table(rows, cols, dirs["metrics"], "promise_vs_delivery_per_instance")
        agg = []
        for scope, sids in scopes:
            row = {"scope": scope}
            for key, _ in cols[2:]:
                row[key] = _finite_mean([x[key] for x in rows if x["instance_id"] in sids])
            agg.append(row)
        write_table(agg, [("scope", "Scope")] + cols[2:], dirs["metrics"], "promise_vs_delivery_summary")

    # ---- Statistical Tests/
    rows = []
    for c in regimes[1:]:
        for i in ids:
            t = tests["pairs"][c][i]
            rows.append({"challenger": c, "reference": ref, "instance_id": i, "size_class": class_of.get(i, ""),
                         "n_replications": t["n"], "mean_difference": t["mean_diff"], "cliff_delta": t["cliff_delta"],
                         "delta_ci_lo": t["delta_lo"], "delta_ci_hi": t["delta_hi"],
                         "effect": cliff_interpretation(t["cliff_delta"]) if np.isfinite(t["cliff_delta"]) else "",
                         "p_value": t["p_value"], "p_adjusted": t["p_adjusted"], "significant": t["significant"],
                         "winner": t["winner"]})
    write_table(rows, [("challenger", "Challenger"), ("reference", "Reference"), ("instance_id", "Instance"),
                       ("size_class", "Class"), ("n_replications", "R"), ("mean_difference", "Mean HV diff."),
                       ("cliff_delta", "Cliff delta"), ("delta_ci_lo", "CI low"), ("delta_ci_hi", "CI high"),
                       ("effect", "Effect"), ("p_value", "p"), ("p_adjusted", "p adj."), ("significant", "Significant"),
                       ("winner", "Winner")], dirs["stats"], "per_instance_tests")
    if across:
        write_table(across["rows"], [("indicator", "Indicator"), ("challenger", "Challenger"), ("reference", "Reference"),
                                     ("scope", "Scope"), ("n_instances", "Instances"), ("mean_reference", "Mean ref."),
                                     ("mean_challenger", "Mean chall."), ("mean_difference", "Mean diff."),
                                     ("diff_ci_lo", "CI low"), ("diff_ci_hi", "CI high"),
                                     ("mean_relative_gap_pct", "Mean gap (%)"), ("cliff_delta", "Cliff delta"),
                                     ("delta_ci_lo", "Delta CI low"), ("delta_ci_hi", "Delta CI high"), ("effect", "Effect"),
                                     ("p_value", "p"), ("p_adjusted", "p adj."), ("significant", "Significant"),
                                     ("winner", "Winner")], dirs["stats"], "across_instance_tests")

    return {"summary_rows": summary_rows, "overall": overall, "scopes": scopes}


# ============================================================================
# FIGURES
# ============================================================================

def _colors(regimes):
    return {r: _REGIME_PALETTE[k % len(_REGIME_PALETTE)] for k, r in enumerate(regimes)}


def _markers(regimes):
    return {r: style.MARKERS[k % len(style.MARKERS)] for k, r in enumerate(regimes)}


def _instances_by_class(instance_size_classes, ids):
    class_of = _class_lookup(instance_size_classes)
    out = []
    for cls in _ordered_classes(instance_size_classes):
        out += [(cls, i) for i in sorted(ids) if class_of.get(i) == cls]
    return out


def plot_hv_realized(code, regimes, labels, per_instance_hv, out_dir, figures_cfg):
    ids = sorted(per_instance_hv)
    col, mk = _colors(regimes), _markers(regimes)
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(ids)), 5))
    for r in regimes:
        stats_ = [bootstrap_ci(per_instance_hv[i][r]) for i in ids]
        means = [s[0] for s in stats_]
        ax.plot(ids, means, marker=mk[r], color=col[r], label=labels[r])
        ax.fill_between(ids, [s[1] for s in stats_], [s[2] for s in stats_], color=col[r], alpha=0.2)
    ax.set_xlabel("Instance")
    ax.set_ylabel("Mean realized hypervolume (95% bootstrap CI)")
    ax.set_xticks(ids)
    ax.set_title(f"Realized Hypervolume per Instance -- {_pretty_problem_name(code)}")
    ax.legend()
    _save_all_formats(fig, out_dir, f"{code}_hv_realized_lineplot", figures_cfg)


def plot_hv_by_size_class(code, regimes, labels, derived, instance_size_classes, out_dir, figures_cfg):
    ids = sorted(derived["hv_mean"])
    scopes = _scopes(instance_size_classes, ids)
    col = _colors(regimes)
    k = len(regimes)
    width = 0.8 / k
    fig, ax = plt.subplots(figsize=(1.6 * len(scopes) * max(1, k / 2) + 2, 5))
    for j, r in enumerate(regimes):
        data = [[derived["hv_mean"][i][r] for i in sids if np.isfinite(derived["hv_mean"][i][r])] for _, sids in scopes]
        pos = np.arange(len(scopes)) + (j - (k - 1) / 2) * width
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True, showmeans=True)
        for box in bp["boxes"]:
            box.set_facecolor(col[r])
        ax.plot([], [], color=col[r], linewidth=8, label=labels[r])
    ax.set_xticks(np.arange(len(scopes)))
    ax.set_xticklabels([s.capitalize() for s, _ in scopes])
    ax.set_xlabel("Size class")
    ax.set_ylabel("Per-instance mean realized hypervolume")
    ax.set_title(f"Realized Hypervolume by Size Class -- {_pretty_problem_name(code)}")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    _save_all_formats(fig, out_dir, f"{code}_hv_by_size_class_boxplot", figures_cfg)


def _pairs_grid(n_obj):
    pairs = list(itertools.combinations(range(n_obj), 2))
    ncols = min(3, len(pairs))
    return pairs, ncols, int(np.ceil(len(pairs) / ncols))



def plot_realized_fronts(code, regimes, labels, objective_names, fronts, per_instance_realized, instance_size_classes,
                         out_dir, figures_cfg):
    """D7: per instance, the median (solid) and the 25%/75% (dashed) empirical
    attainment surfaces of every regime's realized fronts over the R
    replications, with its planned front (hollow markers)."""
    if len(objective_names) < 2:
        return []
    col, mk = _colors(regimes), _markers(regimes)
    pairs, ncols, nrows = _pairs_grid(len(objective_names))
    plotted = []
    for cls, i in _instances_by_class(instance_size_classes, per_instance_realized.keys()):
        plotted.append(i)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows), squeeze=False)
        for k, (a, b) in enumerate(pairs):
            ax = axes[k // ncols][k % ncols]
            for r in regimes:
                runs = [np.asarray(f, dtype=float)[:, [a, b]] for f in per_instance_realized[i][r] if f]
                for level, ls, lw in ((0.5, "-", 1.8), (0.25, "--", 0.9), (0.75, "--", 0.9)):
                    surf = IND.attainment_surface(runs, level)
                    if len(surf):
                        ax.step(surf[:, 0], surf[:, 1], where="post", color=col[r], linestyle=ls, linewidth=lw,
                                label=f"{labels[r]} realized (median)" if level == 0.5 else None)
                planned = np.asarray(fronts[r][i].objectives, dtype=float)
                if planned.size:
                    ax.scatter(planned[:, a], planned[:, b], facecolors="none", edgecolors=col[r], marker=mk[r], s=40,
                               linewidths=1.1, label=f"{labels[r]} planned")
            ax.set_xlabel(objective_names[a])
            ax.set_ylabel(objective_names[b])
            if k == 0:
                ax.legend(fontsize=7)
        for k in range(len(pairs), nrows * ncols):
            axes[k // ncols][k % ncols].axis("off")
        fig.suptitle(f"Realized Fronts (attainment surfaces) -- {_pretty_problem_name(code)}, instance {i} ({cls})")
        fig.tight_layout()
        _save_all_formats(fig, out_dir, f"{code}_realized_fronts_{cls}_instance{i}", figures_cfg)
    return plotted


def plot_promise_vs_delivery(code, regimes, labels, derived, instance_size_classes, out_dir, figures_cfg):
    """D2: per instance and regime, the planned front's hypervolume (hollow) and
    the mean realized hypervolume (filled), in one normalisation per instance."""
    col, mk = _colors(regimes), _markers(regimes)
    order = _instances_by_class(instance_size_classes, derived["promise"].keys())
    k = len(regimes)
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(order)), 5))
    for j, r in enumerate(regimes):
        xs = np.arange(len(order)) + (j - (k - 1) / 2) * 0.25
        planned = [derived["promise"][i][r]["planned_hv"] for _, i in order]
        realized = [derived["promise"][i][r]["realized_hv"] for _, i in order]
        for x, p, q in zip(xs, planned, realized):
            ax.plot([x, x], [p, q], color=col[r], linewidth=1.2, alpha=0.8)
        ax.scatter(xs, planned, facecolors="none", edgecolors=col[r], marker=mk[r], s=45, label=f"{labels[r]} planned")
        ax.scatter(xs, realized, color=col[r], marker=mk[r], s=45, label=f"{labels[r]} realized")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f"{i}\n({cls[:3]})" for cls, i in order], fontsize=8)
    ax.set_xlabel("Instance (size class)")
    ax.set_ylabel("Hypervolume (one normalisation per instance)")
    ax.set_title(f"Promise vs Delivery -- {_pretty_problem_name(code)}")
    ax.legend(fontsize=8, ncol=2)
    _save_all_formats(fig, out_dir, f"{code}_promise_vs_delivery", figures_cfg)


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def generate_report(
    code: str,
    objective_names: List[str],
    regimes: List[str],
    regime_labels: Dict[str, str],
    per_instance_hv: Dict[int, Dict[str, List[float]]],
    per_instance_realized: Dict[int, Dict[str, list]],
    instance_size_classes: Dict[str, List[int]],
    fronts: Dict[str, Dict[int, Any]],
    best_params_by_regime: Dict[str, Dict[int, Dict[str, Any]]],
    out_dir: str,
    figures_cfg: Optional[Dict[str, Any]] = None,
    stats_cfg: Optional[Dict[str, Any]] = None,
    report_cfg: Optional[Dict[str, Any]] = None,
    eps: float = 1e-3,
) -> Dict[str, Any]:
    """Writes the report of one chapter's run under `out_dir`; returns the overall
    summary row. `report_cfg`: what the chapter reports (see config.py)."""
    figures_cfg = figures_cfg or {"formats": ["png", "svg"], "dpi": 400}
    stats_cfg = stats_cfg or {"alpha": 0.05, "correction": "holm", "bootstrap_samples": 2000}
    report_cfg = report_cfg or {}
    if len(regimes) != 2:
        raise ValueError(f"{code}: the comparison has two regimes, the reference and the challenger; got {regimes}")
    labels = {r: regime_labels.get(r, r.replace("_", " ").capitalize()) for r in regimes}
    ref, chall = regimes
    dirs = {"data": os.path.join(out_dir, "Data"), "metrics": os.path.join(out_dir, "Metrics"),
            "stats": os.path.join(out_dir, "Statistical Tests")}
    fig = {name: os.path.join(out_dir, "Figures", name) for name in ("HV Realized", "HV by Size Class")}
    if report_cfg.get("promise_vs_delivery"):
        fig["Promise vs Delivery"] = os.path.join(out_dir, "Figures", "Promise vs Delivery")
    if report_cfg.get("realized_fronts"):
        fig["Realized Fronts"] = os.path.join(out_dir, "Figures", "Realized Fronts")
    for d in list(dirs.values()) + list(fig.values()):
        os.makedirs(d, exist_ok=True)

    derived = compute_derived(regimes, objective_names, per_instance_hv, per_instance_realized, fronts, eps,
                              promise=bool(report_cfg.get("promise_vs_delivery")))
    tests = per_instance_tests(regimes, per_instance_hv, stats_cfg)
    across = (across_instance_tests(regimes, derived, instance_size_classes, stats_cfg)
              if report_cfg.get("across_instances") else None)
    tables = _tables(code, regimes, labels, objective_names, per_instance_hv, per_instance_realized, fronts, derived,
                     tests, across, instance_size_classes, best_params_by_regime, report_cfg, dirs)
    corr = stats_cfg.get("correction", "holm")
    alpha = stats_cfg.get("alpha", 0.05)

    _write_caption(dirs["data"], "Data -- raw exports", [
        "per_instance_hv_raw: realized hypervolume of both regimes at every replication of every instance (one "
        "normalisation shared by the two regimes per instance and replication).",
        "planned_fronts_long.csv: the non-dominated front of each regime and instance as planned (evaluation phase); "
        "realized_fronts_long.csv: the non-dominated front of the realized objective vectors, per replication.",
        "tuned_params: the parameters retained by each regime's tuning on every instance.",
        "Every table also exists as a LaTeX (booktabs) .tex file."])
    metrics = [
        f"Reference regime: {labels[ref]}; challenger: {labels[chall]}.",
        "per_instance_summary / per_size_class_summary / overall_summary: mean realized hypervolume per regime "
        "(percentile bootstrap 95% CI over the replications), relative gap 100 (HV_challenger - HV_reference) / "
        "HV_reference, wins/ties/losses by mean, and the number of instances whose own test is significant in each "
        "direction" + (", with the across-instance test of the scope (Statistical Tests/across_instance_tests)."
                       if across else ".")]
    if report_cfg.get("promise_vs_delivery"):
        metrics += [
            "realized_objectives_per_instance / _summary (objective units): for every replication, the realized front's "
            "knee point (closest to the ideal point once both regimes' realized fronts of the replication are normalised "
            "together) and the best realized value of every objective, averaged over the replications; change (%) = "
            "100 (challenger - reference) / |reference| (negative = lower = better, all objectives are minimised).",
            "promise_vs_delivery_per_instance / _summary: hypervolume of the planned front and mean hypervolume of the "
            "realized fronts, in ONE normalisation per instance (every planned and realized front of both regimes), and "
            "delivery = realized / planned (1 = the plan delivered what it promised); per objective, the best realized "
            "value against the best planned value (%; positive = worse than promised)."]
    _write_caption(dirs["metrics"], "Metrics -- how each table was computed", metrics)
    stats_text = [
        f"per_instance_tests: for every instance, a paired Wilcoxon signed-rank test over the realization replications "
        f"(the same realization is applied to both regimes at a replication), Cliff's delta with a bootstrap 95% CI; "
        f"p-values {corr}-adjusted over the instances; significant when the adjusted p < {alpha}."]
    if across:
        stats_text.append(
            "across_instance_tests: for each scope (each size class, overall), a paired Wilcoxon test on the "
            "per-instance mean hypervolumes (the instances are the blocks), the mean difference and Cliff's delta with "
            "bootstrap 95% CIs (delta > 0: the challenger's values are larger). The overall test is the primary one; "
            f"the class-level p-values are {corr}-adjusted together (with 5 instances per class, the smallest two-sided "
            "Wilcoxon p-value is 0.0625: a class alone cannot reach significance).")
    _write_caption(dirs["stats"], "Statistical Tests -- how these results were computed", stats_text)

    plot_hv_realized(code, regimes, labels, per_instance_hv, fig["HV Realized"], figures_cfg)
    _write_caption(fig["HV Realized"], "HV Realized", [
        "Mean realized hypervolume per instance with a 95% bootstrap band over the replications, one line per regime."])
    plot_hv_by_size_class(code, regimes, labels, derived, instance_size_classes, fig["HV by Size Class"], figures_cfg)
    _write_caption(fig["HV by Size Class"], "HV by Size Class", [
        "Box plot of the per-instance mean realized hypervolume, per size class and overall, one box per regime."])
    if report_cfg.get("realized_fronts"):
        realized_ids = plot_realized_fronts(code, regimes, labels, objective_names, fronts, per_instance_realized,
                                            instance_size_classes, fig["Realized Fronts"], figures_cfg)
        _write_caption(fig["Realized Fronts"], "Realized Fronts", [
            "For every instance, the empirical attainment surfaces of each regime's realized fronts over the "
            "replications: solid = median (attained in at least half of the replications), dashed = 25% and 75%; "
            "hollow markers = the planned front.", f"Instances: {realized_ids}."])
    if report_cfg.get("promise_vs_delivery"):
        plot_promise_vs_delivery(code, regimes, labels, derived, instance_size_classes, fig["Promise vs Delivery"],
                                 figures_cfg)
        _write_caption(fig["Promise vs Delivery"], "Promise vs Delivery", [
            "Per instance and regime: hypervolume of the planned front (hollow) and mean hypervolume of the realized "
            "fronts (filled), in one normalisation per instance; a long downward segment = a plan that over-promised."])
    return tables["overall"]
