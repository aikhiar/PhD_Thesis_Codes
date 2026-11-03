"""
Reporting: figures, tables, and statistical-validation summaries for the
surrogate training pipeline -- one call per objective (`report_objective`)
plus one end-to-end summary per problem run (`report_end_to_end`), plus
execution-time analysis (`benchmark_inference_time` + `report_execution_time`):
how long each candidate took to tune+fit, how long each fitted model takes
to score one candidate schedule at each instance-size class, AND how that
compares to the real (expensive Monte-Carlo) objective function's own
evaluation cost -- the actual speedup a surrogate buys a live search.
Figures use the same methods_comparison-styled theme as the rest of the thesis
(`style.py`, physically copied from `methods_comparison/figures/style.py`).

Every output folder groups its contents under clearly named subfolders
(Metrics/, Statistical Tests/, Figures/<category>/<FORMAT>/) with a plain-
English `_HOW_IT_IS_COMPUTED.txt` caption per section, mirroring the
convention already used in methods_comparison's own Results Analysis output.
"""

import collections
import csv
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import style
from .progress import ProgressReporter, display_model_name
from .stats import cliff_interpretation, friedman_test, wilcoxon_pairs
from dependency.trainer.trainer import _PARAM_DISTRIBUTIONS, instance_level_errors

style.apply_thesis_style()


def _pretty_problem_name(code: str) -> str:
    """Professional figure-title form of a problem code, e.g. 'chapter11' ->
    'Chapter 11'. Falls back to a title-cased, underscore-stripped version
    for any code that doesn't match the 'word+digits' pattern."""
    if re.fullmatch(r"[A-Z]\d+", code):        # paper codes (Chapters 5 and 6) stay as they are
        return code
    m = re.match(r"^([a-zA-Z]+?)(\d+)$", code)
    if m:
        return f"{m.group(1).capitalize()} {m.group(2)}"
    return code.replace("_", " ").title()


def _write_caption(out_dir: str, title: str, paragraphs: List[str]) -> None:
    """Writes `_HOW_IT_IS_COMPUTED.txt` under out_dir -- a short, plain-
    English explanation of what a section's tables/figures show and how
    they were computed, so the folder is self-explanatory without having to
    cross-reference the thesis text or the code."""
    os.makedirs(out_dir, exist_ok=True)
    lines = [title, "=" * len(title), ""]
    for p in paragraphs:
        lines.append(p)
        lines.append("")
    with open(os.path.join(out_dir, "_HOW_IT_IS_COMPUTED.txt"), "w") as f:
        f.write("\n".join(lines))


# ============================================================================
# PER-OBJECTIVE REPORT
# ============================================================================

def report_objective(
    training_result, out_dir: str, figures_cfg: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Writes, under `out_dir`:

      Metrics/
        - leaderboard.csv      : one consolidated row per model -- CV score,
          validation RMSE/MAE/R^2, Friedman average rank, best-model flag,
          and (for alternatives) the outcome of its statistical comparison
          against the most accurate model, and the two-stage selection.
        - metrics.csv          : per-model CV score / val RMSE / MAE / R^2
        - cv_fold_scores.csv   : raw per-fold tuning scores (model x fold)
        - validation_errors.csv: raw per-validation-sample absolute error
          (model x sample) -- the exact material the statistical tests below
          are computed from.
        - hyperparameters.csv / .tex: every candidate's search space and the
          values the randomized search retained (deployed model flagged).
      Statistical Tests/
        - significance_tests.csv (+ caption): Friedman omnibus test across
          all candidates, then pairwise Wilcoxon/Cliff's-delta of the best
          model against every alternative, paired by held-out validation
          sample (Demsar, 2006 protocol -- see `_validation_significance_tests`).
      Figures/
        - Predictions/{pred_vs_actual, residuals}.<ext> : deployed model only.
        - Model Comparison/model_comparison.<ext> : validation RMSE of all
          candidates.
      summary.json              : machine-readable summary of everything above.

    Returns the summary dict also written to summary.json."""
    os.makedirs(out_dir, exist_ok=True)
    figures_cfg = figures_cfg or {"formats": ["png", "svg"], "dpi": 400}

    metrics_dir = os.path.join(out_dir, "Metrics")
    stats_dir = os.path.join(out_dir, "Statistical Tests")
    fig_pred_dir = os.path.join(out_dir, "Figures", "Predictions")
    fig_cmp_dir = os.path.join(out_dir, "Figures", "Model Comparison")
    for d in (metrics_dir, stats_dir, fig_pred_dir, fig_cmp_dir):
        os.makedirs(d, exist_ok=True)

    significance = _validation_significance_tests(training_result)

    _write_metrics_csv(training_result, os.path.join(metrics_dir, "metrics.csv"))
    _write_cv_fold_scores_csv(training_result, os.path.join(metrics_dir, "cv_fold_scores.csv"))
    _write_validation_errors_csv(training_result, os.path.join(metrics_dir, "validation_errors.csv"))
    _write_leaderboard_csv(training_result, significance, os.path.join(metrics_dir, "leaderboard.csv"))
    _write_hyperparameters(training_result, metrics_dir)
    _write_caption(
        metrics_dir, "Metrics -- how each table was computed",
        [
            "leaderboard.csv: one row per candidate model. cv_score/cv_score_std "
            "are the mean/std of the k-fold cross-validation score used to pick "
            "each model's own hyperparameters during tuning (higher = better, "
            "since the scorer is neg_root_mean_squared_error). val_rmse/val_mae/"
            "val_r2 are computed once, after tuning, on the held-out validation "
            "split every candidate shares. friedman_avg_rank is this model's "
            "average rank (1 = best) across every held-out validation sample in "
            "the Friedman test described in 'Statistical Tests/'.",
            "hyperparameters.csv / .tex: for every candidate, each tuned "
            "hyperparameter, its search space (randomized search) and the value "
            "retained by the cross-validation over folds of whole instances; the "
            "deployed model (two-stage rule) is flagged.",
            "metrics.csv / cv_fold_scores.csv / validation_errors.csv: the same "
            "numbers broken out into plain per-model and per-fold/per-sample "
            "matrices, for independent re-analysis outside this pipeline.",
        ],
    )

    _plot_pred_vs_actual(training_result, fig_pred_dir, figures_cfg)
    _plot_residuals(training_result, fig_pred_dir, figures_cfg)
    _write_caption(
        fig_pred_dir, "Predictions -- how these figures were computed",
        [
            "pred_vs_actual: the DEPLOYED model's prediction (two-stage rule) on "
            "every held-out validation sample, plotted against the true value; the dashed "
            "line is y = x (a perfect model). residuals: the same predictions' "
            "signed error (actual - predicted) against the predicted value, "
            "which surfaces any systematic bias (e.g. under-predicting at the "
            "high end) that a single RMSE number would hide.",
        ],
    )

    _plot_model_comparison(training_result, fig_cmp_dir, figures_cfg)
    _write_caption(
        fig_cmp_dir, "Model Comparison -- how this figure was computed",
        [
            "model_comparison: held-out validation RMSE for every candidate "
            "(lower is better); the deployed model's bar is highlighted and the most "
            "accurate one is outlined when it is a different model.",
        ],
    )

    _write_significance_csv(significance, os.path.join(stats_dir, "significance_tests.csv"))
    _write_caption(
        stats_dir, "Statistical Tests -- how these results were computed",
        [
            "Non-parametric statistical validation, following Demsar (2006), "
            "'Statistical Comparisons of Classifiers over Multiple Data Sets' "
            "(JMLR 7:1-30). Every candidate predicts on the SAME held-out "
            "validation instances; each candidate is summarised on every validation "
            "INSTANCE by its mean absolute error over that instance's schedules, and "
            "the instances are the blocks of the tests (the schedules of one instance "
            "are strongly correlated, so treating them as independent observations "
            "would inflate the significance).",
            "Friedman test: an omnibus rank-based test across ALL candidates "
            "jointly -- 'are these models equivalent?'.",
            "Pairwise Wilcoxon signed-rank test (only run if Friedman finds a "
            "significant difference is not assumed -- both are always reported): "
            "the most accurate model against every alternative, on the same paired "
            "per-instance mean absolute errors, with a Bonferroni-corrected significance "
            "threshold (alpha / number of pairwise comparisons across the full "
            "candidate set) and Cliff's delta as a non-parametric effect size "
            "alongside every p-value.",
        ],
    )

    summary = {
        "objective_name": training_result.objective_name,
        "best_model": training_result.best_model_name,
        "deployed_model": training_result.deployed_model_name or training_result.best_model_name,
        "selection": training_result.selection,
        "split": ("by instance, stratified by size class" if training_result.split_mode == "instance"
                  else "by row"),
        "n_train": int(len(training_result.y_train)),
        "n_val": int(len(training_result.y_val)),
        "n_train_instances": (int(len(set(training_result.groups_train.tolist())))
                              if training_result.groups_train is not None else None),
        "n_val_instances": (int(len(set(training_result.groups_val.tolist())))
                            if training_result.groups_val is not None else None),
        "models": {
            name: {
                "cv_score": r.cv_score,
                "cv_fold_scores": r.cv_fold_scores.tolist(),
                "val_rmse": r.val_rmse,
                "val_mae": r.val_mae,
                "val_r2": r.val_r2,
                "fit_params": r.fit_params,
            }
            for name, r in training_result.results.items()
        },
        "significance": significance,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _validation_significance_tests(training_result, alpha: float = 0.05) -> Dict[str, Any]:
    """Non-parametric statistical validation (Demsar, 2006): a Friedman omnibus
    test across ALL candidates, then pairwise Wilcoxon signed-rank tests
    (Bonferroni-corrected over every pair of candidates) with Cliff's delta as
    effect size, reported for the most accurate model against every alternative.

    The unit (block) is the VALIDATION INSTANCE: each candidate is summarised on
    every validation instance by its mean absolute error over that instance's
    schedules. The schedules logged on one instance are strongly correlated, so
    treating them as independent paired observations (one block per row) would
    inflate the significance; with the split made by instance, the validation
    instances are the independent units. When the split was made row by row the
    blocks fall back to the rows (and the p-values must be read with that caveat)."""
    blocks, errors = instance_level_errors(training_result)
    errors = {name: errs.tolist() for name, errs in errors.items()}

    friedman = friedman_test(errors, alpha=alpha, higher_is_better=False)
    pairwise = wilcoxon_pairs(errors, alpha=alpha, bonferroni=True, higher_is_better=False)

    best_name = training_result.best_model_name
    names = pairwise["method_names"]
    best_idx = names.index(best_name)
    corrected_alpha = pairwise["corrected_alpha"]

    vs_best: Dict[str, Dict[str, Any]] = {}
    for j, name in enumerate(names):
        if name == best_name:
            continue
        p = pairwise["p_matrix"][best_idx][j]
        delta = pairwise["cliff_delta_matrix"][best_idx][j]
        vs_best[name] = {
            "wilcoxon_pvalue": p,
            "cliff_delta": delta,
            "interpretation": cliff_interpretation(delta),
            "significant": bool(np.isfinite(p) and p < corrected_alpha),
        }

    grouped = training_result.split_mode == "instance"
    return {
        "n_val": int(len(training_result.y_val)),
        "n_blocks": int(len(blocks)),
        "unit": "validation instance" if grouped else "validation row",
        "alpha": alpha,
        "corrected_alpha": corrected_alpha,
        "friedman": friedman,
        "vs_best": vs_best,
    }


def select_deployed_model(training_result, significance: Dict[str, Any],
                          inference_rows: List[Dict[str, Any]],
                          selection_cfg: Dict[str, Any],
                          real_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Two-stage rule, fixed before any result (config.SELECTION):
    stage 1 keeps the candidates practically equivalent in accuracy to the most
    accurate one: |Cliff's delta| between their per-validation-instance mean
    absolute errors and the most accurate one's at most `max_abs_cliff_delta`
    (the most accurate always qualifies); stage 2 deploys, among them, the one
    with the lowest end-to-end scoring cost for this objective (feature
    extraction + prediction, ms per schedule, mean over the size classes of the
    benchmark). The speed against the real Monte Carlo objective is reported for
    every candidate, with a warning when the deployed model is not faster."""
    from .stats import cliff_delta
    threshold = float(selection_cfg.get("max_abs_cliff_delta", 0.33))
    best = training_result.best_model_name
    _, inst_errors = instance_level_errors(training_result)
    cost: Dict[str, List[float]] = collections.defaultdict(list)
    for row in inference_rows:
        if row["objective"] == training_result.objective_name:
            cost[row["model"]].append(row["total_ms"])
    mean_cost = {m: float(np.mean(v)) for m, v in cost.items()}
    real_ms = float(np.mean([r["real_objective_ms"] for r in (real_rows or [])])) if real_rows else float("nan")
    table = []
    for name, res in training_result.results.items():
        delta = 0.0 if name == best else cliff_delta(inst_errors[best], inst_errors[name])
        equivalent = name == best or (np.isfinite(delta) and abs(delta) <= threshold)
        e2e = mean_cost.get(name, float("nan"))
        table.append({"model": name, "val_rmse": res.val_rmse, "cliff_delta_vs_most_accurate": float(delta),
                      "practically_equivalent": bool(equivalent), "end_to_end_ms": e2e,
                      "real_objective_ms": real_ms,
                      "speedup_vs_real_objective": (real_ms / e2e) if (np.isfinite(real_ms) and e2e > 0) else float("nan")})
    stage1 = [r for r in table if r["practically_equivalent"]]
    timed = [r for r in stage1 if np.isfinite(r["end_to_end_ms"])]
    deployed = min(timed, key=lambda r: r["end_to_end_ms"])["model"] if timed else best
    for r in table:
        r["deployed"] = r["model"] == deployed
        r["most_accurate"] = r["model"] == best
    dep = next(r for r in table if r["deployed"])
    warning = None
    if np.isfinite(dep["speedup_vs_real_objective"]) and dep["speedup_vs_real_objective"] <= 1.0:
        warning = (f"the deployed model '{deployed}' for '{training_result.objective_name}' is NOT faster than the "
                   f"real Monte Carlo objective ({dep['end_to_end_ms']:.3g} ms vs {real_ms:.3g} ms per schedule)")
        print(f"  [selection] WARNING: {warning}", flush=True)
    return {"rule": f"|Cliff's delta| <= {threshold} vs the most accurate (per validation instance), "
                    f"then lowest end-to-end scoring time",
            "max_abs_cliff_delta": threshold, "most_accurate": best, "deployed": deployed,
            "stage1_candidates": [r["model"] for r in stage1], "warning": warning, "table": table}


def _deployed_name(training_result) -> str:
    return training_result.deployed_model_name or training_result.best_model_name


def _bar_colors(training_result, names: List[str]) -> List[str]:
    deployed = _deployed_name(training_result)
    return [style.PALETTE[1] if n == deployed else style.PALETTE[4] for n in names]


def _write_leaderboard_csv(training_result, significance: Dict[str, Any], path: str) -> None:
    """One row per candidate: CV score, validation metrics, Friedman average
    rank, the comparison with the most accurate model, and the two stages of the
    deployment rule."""
    best_name = training_result.best_model_name
    avg_ranks = significance["friedman"].get("average_ranks", {})
    vs_best = significance["vs_best"]
    sel = {r["model"]: r for r in (training_result.selection or {}).get("table", [])}

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "cv_score", "cv_score_std", "val_rmse", "val_mae", "val_r2",
            "friedman_avg_rank", "most_accurate", "wilcoxon_pvalue_vs_most_accurate",
            "cliff_delta_vs_most_accurate", "effect", "significant",
            "practically_equivalent", "end_to_end_ms", "deployed",
        ])
        ordered = [best_name] + sorted(
            (n for n in training_result.results if n != best_name),
            key=lambda n: training_result.results[n].val_rmse,
        )
        for name in ordered:
            r = training_result.results[name]
            is_best = name == best_name
            sig = vs_best.get(name)
            s = sel.get(name, {})
            writer.writerow([
                name, f"{r.cv_score:.6g}", f"{float(np.std(r.cv_fold_scores)):.6g}",
                f"{r.val_rmse:.6g}", f"{r.val_mae:.6g}", f"{r.val_r2:.6g}",
                f"{avg_ranks.get(name, float('nan')):.3f}",
                "yes" if is_best else "",
                "" if is_best else f"{sig['wilcoxon_pvalue']:.4g}",
                "" if is_best else f"{sig['cliff_delta']:+.3f}",
                "" if is_best else sig["interpretation"],
                "" if is_best else ("yes" if sig["significant"] else "no"),
                "yes" if s.get("practically_equivalent") else "no",
                f"{s.get('end_to_end_ms', float('nan')):.4g}",
                "yes" if s.get("deployed") else "",
            ])


def _write_metrics_csv(training_result, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "cv_score", "cv_score_std", "val_rmse", "val_mae", "val_r2", "is_best",
        ])
        for name, r in training_result.results.items():
            writer.writerow([
                name, f"{r.cv_score:.6g}", f"{float(np.std(r.cv_fold_scores)):.6g}",
                f"{r.val_rmse:.6g}", f"{r.val_mae:.6g}", f"{r.val_r2:.6g}",
                "yes" if name == training_result.best_model_name else "",
            ])


def _describe_space(space) -> str:
    """Readable form of one search-space entry: a list of values or a frozen
    scipy distribution (randint, uniform, loguniform)."""
    if isinstance(space, (list, tuple)):
        return "{" + ", ".join(_format_value(v) for v in space) + "}"
    name = getattr(getattr(space, "dist", None), "name", "")
    args = tuple(space.args) + tuple(space.kwds.values())
    if name == "randint":
        return f"integer in [{int(args[0])}, {int(args[1]) - 1}]"
    if name == "uniform":
        loc, scale = (list(args) + [0.0, 1.0])[:2]
        return f"uniform on [{loc:g}, {loc + scale:g}]"
    if name in ("loguniform", "reciprocal"):
        return f"log-uniform on [{args[0]:g}, {args[1]:g}]"
    return str(space)


def _format_value(value) -> str:
    """4 significant digits; a whole float keeps its decimal point (1.0 = every
    feature for max_features, not 1 feature)."""
    if isinstance(value, float):
        return f"{value:.1f}" if value.is_integer() and abs(value) < 1e4 else f"{value:.4g}"
    return "None" if value is None else str(value)


def _write_hyperparameters(training_result, metrics_dir: str) -> None:
    """hyperparameters.csv and .tex: each candidate's tuned hyperparameters,
    their search space and the retained value (Amine's remark A2)."""
    deployed = _deployed_name(training_result)
    rows = []
    for name, r in training_result.results.items():
        space = _PARAM_DISTRIBUTIONS.get(name, {})
        for key, dist in space.items():
            rows.append({
                "model": display_model_name(name),
                "hyperparameter": key.replace("model__", ""),
                "search_space": _describe_space(dist),
                "retained_value": _format_value((r.fit_params or {}).get(key)),
                "deployed": "yes" if name == deployed else "",
            })
    fields = ["model", "hyperparameter", "search_space", "retained_value", "deployed"]
    with open(os.path.join(metrics_dir, "hyperparameters.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def tex(text: str) -> str:
        return text.replace("_", r"\_").replace("{", r"\{").replace("}", r"\}")

    lines = [r"\begin{tabular}{lllr}", r"\toprule",
             r"Model & Hyperparameter & Search space & Retained \\", r"\midrule"]
    previous = None
    for row in rows:
        model = row["model"] + (r" (deployed)" if row["deployed"] else "")
        lines.append(" & ".join([tex(model) if model != previous else "", tex(row["hyperparameter"]),
                                 tex(row["search_space"]), tex(row["retained_value"])]) + r" \\")
        previous = model
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(metrics_dir, "hyperparameters.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_cv_fold_scores_csv(training_result, path: str) -> None:
    """Raw per-fold CV scores for every candidate model, in matrix form
    (model x fold) -- the tuning-phase scores used to pick each model's
    hyperparameters, exposed here as a plain spreadsheet-friendly table for
    independent re-analysis. NOT what the significance test is computed
    from (see `_write_validation_errors_csv` for that)."""
    n_folds = max((len(r.cv_fold_scores) for r in training_result.results.values()), default=0)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + [f"fold_{i}" for i in range(n_folds)])
        for name, r in training_result.results.items():
            writer.writerow([name] + [f"{s:.6g}" for s in r.cv_fold_scores])


def _write_validation_errors_csv(training_result, path: str) -> None:
    """Per-model, per-held-out-validation-sample absolute error, in matrix
    form (model x sample) -- the exact raw material
    `_validation_significance_tests` computes the Friedman/Wilcoxon tests
    from, exposed here as a plain spreadsheet-friendly table for
    independent re-analysis."""
    y_val = training_result.y_val
    names = list(training_result.results.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + [f"val_{i}" for i in range(len(y_val))])
        for name in names:
            errs = np.abs(y_val - training_result.val_predictions[name])
            writer.writerow([name] + [f"{e:.6g}" for e in errs])


def _write_significance_csv(significance: Dict[str, Any], path: str) -> None:
    """Statistical-validation results as a plain table -- the same numbers
    embedded in summary.json/report.md, exposed here for direct
    spreadsheet/stats-software consumption. First row is the Friedman
    omnibus result across all candidates; remaining rows are the best
    model's pairwise Wilcoxon/Cliff's-delta comparison against each
    alternative, on the same per-validation-sample paired errors."""
    fr = significance["friedman"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["friedman_statistic", "friedman_pvalue", "friedman_significant", "n_val", "corrected_alpha"])
        writer.writerow([
            f"{fr['statistic']:.6g}" if np.isfinite(fr["statistic"]) else "n/a",
            f"{fr['p_value']:.6g}" if np.isfinite(fr["p_value"]) else "n/a",
            fr["significant"], significance["n_val"], f"{significance['corrected_alpha']:.6g}",
        ])
        writer.writerow([])
        writer.writerow(["vs_model", "wilcoxon_pvalue", "cliff_delta", "interpretation", "significant"])
        for name, sig in significance["vs_best"].items():
            writer.writerow([
                name, f"{sig['wilcoxon_pvalue']:.6g}", f"{sig['cliff_delta']:.6g}",
                sig["interpretation"], sig["significant"],
            ])


def _save_all_formats(fig, out_dir: str, stem: str, figures_cfg: Dict[str, Any]) -> None:
    formats = figures_cfg.get("formats", ["png", "svg"])
    dpi = figures_cfg.get("dpi", 400)
    for fmt in formats:
        d = os.path.join(out_dir, fmt.upper())
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{stem}.{fmt}"), dpi=dpi, bbox_inches="tight", format=fmt)
    plt.close(fig)


def _plot_pred_vs_actual(training_result, out_dir: str, figures_cfg: Dict[str, Any]) -> None:
    name = _deployed_name(training_result)
    model = training_result.results[name]
    y_val = training_result.y_val
    y_pred = training_result.val_predictions[name]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_val, y_pred, s=16, color=style.PALETTE[0], alpha=0.6, edgecolors="white", linewidths=0.3)
    lo = min(np.min(y_val), np.min(y_pred))
    hi = max(np.max(y_val), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], color=style.PALETTE[5], linestyle="--", linewidth=1.2, label="y = x")
    ax.set_xlabel(f"Actual {training_result.objective_name}")
    ax.set_ylabel(f"Predicted {training_result.objective_name}")
    ax.set_title(f"Predicted vs. Actual -- {training_result.objective_name}\n"
                 f"(Deployed model: {display_model_name(name)}, Val. RMSE = {model.val_rmse:.4g})")
    ax.legend()
    _save_all_formats(fig, out_dir, "pred_vs_actual", figures_cfg)


def _plot_residuals(training_result, out_dir: str, figures_cfg: Dict[str, Any]) -> None:
    name = _deployed_name(training_result)
    y_pred = training_result.val_predictions[name]
    residuals = training_result.y_val - y_pred

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.scatter(y_pred, residuals, s=16, color=style.PALETTE[2], alpha=0.6, edgecolors="white", linewidths=0.3)
    ax.axhline(0.0, color=style.PALETTE[5], linestyle="--", linewidth=1.2)
    ax.set_xlabel(f"Predicted {training_result.objective_name}")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(f"Residuals -- {training_result.objective_name} (Deployed model: {display_model_name(name)})")
    _save_all_formats(fig, out_dir, "residuals", figures_cfg)


def _plot_model_comparison(training_result, out_dir: str, figures_cfg: Dict[str, Any]) -> None:
    """Validation RMSE of every candidate; the deployed model in orange, the most
    accurate one outlined when it is a different model."""
    names = list(training_result.results.keys())
    display_names = [display_model_name(n) for n in names]
    rmses = [training_result.results[n].val_rmse for n in names]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bars = ax.bar(display_names, rmses, color=_bar_colors(training_result, names))
    best = training_result.best_model_name
    if best != _deployed_name(training_result):
        k = names.index(best)
        bars[k].set_edgecolor("black")
        bars[k].set_linewidth(1.5)
        bars[k].set_label("most accurate")
        ax.legend(loc="best", fontsize=8)
    ax.set_ylabel("Validation RMSE")
    ax.set_title(f"Model Comparison -- {training_result.objective_name}")
    ax.tick_params(axis="x", labelrotation=20)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    _save_all_formats(fig, out_dir, "model_comparison", figures_cfg)


# ============================================================================
# END-TO-END (ALL OBJECTIVES, ONE PROBLEM) REPORT
# ============================================================================


# ============================================================================
# EXECUTION-TIME ANALYSIS (training time + inference time vs. instance scale,
# and the real (Monte-Carlo) objective vs. every surrogate model)
# ============================================================================


def benchmark_inference_time(
    problem_module,
    models_by_objective: Dict[str, Dict[str, Any]],
    benchmark_cfg: Optional[Dict[str, Any]] = None,
    progress: Optional[ProgressReporter] = None,
) -> "tuple[List[Dict[str, Any]], List[Dict[str, Any]]]":
    """Per-schedule scoring cost at every instance-size class, measured on
    `schedules_per_class` random schedules of the first training instance of the
    class. Every timed function is first called `warmup_calls` times UNTIMED, so
    that no one-off cost (Numba compilation, lazy imports, caches) is timed.

    Returns (rows, real_rows):
      rows      one per (size class, objective, model): feature_ms (feature
                extraction), predict_ms (the model's prediction for that one
                objective), total_ms = feature_ms + predict_ms (the end-to-end
                cost of scoring one objective of one schedule with that model);
      real_rows one per size class: real_objective_ms, the real objective
                (every objective at once, Monte Carlo with the problem's own
                default replication count -- its label count N_r) and
                exact_objective_ms (objectives computed exactly, paid by the
                surrogate path too)."""
    cfg = {"schedules_per_class": 5, "warmup_calls": 3, "predict_repeats": 200,
           "feature_repeats": 100, "real_repeats": 10, **(benchmark_cfg or {})}
    instances = problem_module.generate_training_instances()
    class_of = getattr(problem_module, "instance_class", lambda iid: str(iid).rsplit("_", 1)[0])
    representative: Dict[str, Any] = {}
    for instance_id, instance in instances:
        representative.setdefault(class_of(instance_id), instance)

    bench_key = "benchmark"
    if progress is not None:
        progress.start(bench_key, "Benchmarking scoring time", total=len(representative))

    def _time(fn, repeats: int) -> float:
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - t0) / repeats * 1000.0

    rows: List[Dict[str, Any]] = []
    real_rows: List[Dict[str, Any]] = []
    rng_state = np.random.get_state()
    np.random.seed(12345)
    try:
        for label, instance in representative.items():
            if progress is not None:
                progress.update(bench_key, description=f"Benchmarking scoring time ({label})")
            problem_params = problem_module.build_problem_params({}, instance)
            n_reps = int(problem_params[2]) if len(problem_params) > 2 else None
            schedules = [problem_module.Generate_Solution(problem_params) for _ in range(cfg["schedules_per_class"])]
            n_vehicles = int(problem_module.solution_shape(instance)[0])
            phis = [np.asarray(problem_module.feature_fn(s, instance), dtype=np.float64).reshape(1, -1)
                    for s in schedules]

            for _ in range(cfg["warmup_calls"]):                       # untimed warm-up
                problem_module.feature_fn(schedules[0], instance)
                problem_module.objective_function(schedules[0], problem_params)
                problem_module.exact_objective_values(instance, schedules[0])
                for models in models_by_objective.values():
                    for est in models.values():
                        try:
                            est.predict(phis[0])
                        except Exception:
                            pass

            feature_ms = float(np.mean([_time(lambda s=s: problem_module.feature_fn(s, instance),
                                              cfg["feature_repeats"]) for s in schedules]))
            real_ms = float(np.mean([_time(lambda s=s: problem_module.objective_function(s, problem_params),
                                           cfg["real_repeats"]) for s in schedules]))
            exact_ms = float(np.mean([_time(lambda s=s: problem_module.exact_objective_values(instance, s),
                                            cfg["feature_repeats"]) for s in schedules]))
            real_rows.append({"size_label": label, "n_vehicles": n_vehicles, "real_objective_ms": real_ms,
                              "exact_objective_ms": exact_ms, "real_objective_replications": n_reps})

            for objective_name, models in models_by_objective.items():
                for model_name, estimator in models.items():
                    try:
                        predict_ms = float(np.mean([_time(lambda p=p: estimator.predict(p), cfg["predict_repeats"])
                                                    for p in phis]))
                    except Exception as e:
                        print(f"  [benchmark] WARNING: '{model_name}' ({objective_name}, {label}) failed "
                              f"during .predict() and is being skipped -- {type(e).__name__}: {e}", flush=True)
                        continue
                    rows.append({"size_label": label, "n_vehicles": n_vehicles, "objective": objective_name,
                                 "model": model_name, "feature_extraction_ms": feature_ms,
                                 "predict_ms": predict_ms, "total_ms": feature_ms + predict_ms})
            if progress is not None:
                progress.update(bench_key, advance=1)
    finally:
        np.random.set_state(rng_state)
    if progress is not None:
        progress.finish(bench_key)
    return rows, real_rows


def _combine_real_vs_surrogate(
    rows: List[Dict[str, Any]], real_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """One row per (size class, model): the full surrogate-based evaluation of a
    schedule if that model served EVERY learned objective -- one feature
    extraction, the model's prediction for each learned objective, plus the
    exactly computed objectives -- against the real Monte Carlo objective."""
    real_by_size = {r["size_label"]: r for r in real_rows}
    by_model_size: Dict[Any, Dict[str, Any]] = {}
    for row in rows:
        key = (row["model"], row["size_label"])
        entry = by_model_size.setdefault(key, {
            "size_label": row["size_label"], "n_vehicles": row["n_vehicles"],
            "model": row["model"], "feature_extraction_ms": row["feature_extraction_ms"],
            "predict_total_ms": 0.0,
        })
        entry["predict_total_ms"] += row["predict_ms"]
    combined: List[Dict[str, Any]] = []
    for (model_name, size_label), entry in by_model_size.items():
        real = real_by_size.get(size_label, {})
        exact_ms = real.get("exact_objective_ms", 0.0)
        surrogate_total_ms = entry["feature_extraction_ms"] + entry["predict_total_ms"] + exact_ms
        real_ms = real.get("real_objective_ms", float("nan"))
        combined.append({
            "size_label": size_label, "n_vehicles": entry["n_vehicles"], "model": model_name,
            "feature_extraction_ms": entry["feature_extraction_ms"], "predict_total_ms": entry["predict_total_ms"],
            "real_objective_ms": real_ms, "real_objective_replications": real.get("real_objective_replications"),
            "surrogate_total_ms": surrogate_total_ms,
            "speedup_factor": (real_ms / surrogate_total_ms)
                              if surrogate_total_ms > 0 and np.isfinite(real_ms) else float("nan"),
        })
    return combined


def report_execution_time(
    problem_name: str,
    training_results: Dict[str, Any],
    inference_rows: List[Dict[str, Any]],
    out_dir: str,
    figures_cfg: Dict[str, Any] = None,
    real_rows: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Writes, under `out_dir`:
      Data/training_time.csv            tuning + fit wall-clock time per model and objective
      Data/inference_time.csv           per (size class, objective, model): feature
                                        extraction, prediction, end-to-end (ms)
      Data/real_vs_surrogate_time.csv   per (size class, model): full surrogate-based
                                        evaluation vs the real Monte Carlo objective
      Figures/Scoring Time vs Scale/inference_time_vs_scale       feature extraction + prediction,
                                                                  one objective
      Figures/Real Objective vs Surrogate/{real_vs_surrogate_time, speedup_factor}
                                        full surrogate evaluation (every learned objective)
                                        vs the real objective
    with a _HOW_IT_IS_COMPUTED.txt caption per figure folder."""
    os.makedirs(out_dir, exist_ok=True)
    figures_cfg = figures_cfg or {"formats": ["png", "svg"], "dpi": 400}
    data_dir = os.path.join(out_dir, "Data")
    fig_scale_dir = os.path.join(out_dir, "Figures", "Scoring Time vs Scale")
    fig_real_dir = os.path.join(out_dir, "Figures", "Real Objective vs Surrogate")
    for d in (data_dir, fig_scale_dir):
        os.makedirs(d, exist_ok=True)

    objectives = list(training_results.keys())
    model_names = sorted({m for r in training_results.values() for m in r.results.keys()})
    deployed = {obj: (r.deployed_model_name or r.best_model_name) for obj, r in training_results.items()}

    with open(os.path.join(data_dir, "training_time.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["objective", "model", "fit_time_seconds", "deployed"])
        for obj_name, result in training_results.items():
            for model_name, model_result in result.results.items():
                writer.writerow([obj_name, model_name, f"{model_result.fit_time_seconds:.6g}",
                                 "yes" if deployed[obj_name] == model_name else ""])

    with open(os.path.join(data_dir, "inference_time.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["size_label", "n_vehicles", "objective", "model",
                         "feature_extraction_ms", "predict_ms", "end_to_end_ms", "deployed"])
        for row in inference_rows:
            writer.writerow([row["size_label"], row["n_vehicles"], row["objective"], row["model"],
                             f"{row['feature_extraction_ms']:.6g}", f"{row['predict_ms']:.6g}",
                             f"{row['total_ms']:.6g}", "yes" if deployed.get(row["objective"]) == row["model"] else ""])

    def _per_model_curve(key: str):
        by = collections.defaultdict(list)
        for row in inference_rows:
            by[(row["model"], row["n_vehicles"])].append(row[key])
        models = sorted({k[0] for k in by})
        sizes = sorted({k[1] for k in by})
        return models, sizes, by

    for key, folder, stem, ylabel, title, caption in [
        ("total_ms", fig_scale_dir, "inference_time_vs_scale",
         "Scoring Time per Schedule (ms)\n(feature extraction + prediction, one objective)",
         "Surrogate Scoring Time vs. Instance Scale",
         "Mean end-to-end time of scoring ONE objective of one schedule with each candidate model: feature "
         "extraction plus the model's prediction, averaged over the objectives and over the benchmark "
         "schedules of each size class (all candidates are shown; the deployed models are listed in "
         "Data/inference_time.csv). Warm-up calls are excluded from the timing."),
    ]:
        if inference_rows:
            models, sizes, by = _per_model_curve(key)
            fig, ax = plt.subplots(figsize=(7, 5))
            for i, model_name in enumerate(models):
                ax.plot(sizes, [float(np.mean(by[(model_name, n)])) for n in sizes],
                        marker=style.MARKERS[i % len(style.MARKERS)], color=style.PALETTE[i % len(style.PALETTE)],
                        label=display_model_name(model_name))
            ax.set_xlabel("Instance Size (Number of Vehicles)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} -- {_pretty_problem_name(problem_name)}")
            ax.legend()
            _save_all_formats(fig, folder, stem, figures_cfg)
        _write_caption(folder, f"{title} -- how this figure was computed", [caption])

    if real_rows:
        os.makedirs(fig_real_dir, exist_ok=True)
        combined = _combine_real_vs_surrogate(inference_rows, real_rows)
        with open(os.path.join(data_dir, "real_vs_surrogate_time.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["size_label", "n_vehicles", "model", "feature_extraction_ms", "predict_total_ms",
                             "surrogate_total_ms", "real_objective_ms", "real_objective_replications",
                             "speedup_factor"])
            for row in sorted(combined, key=lambda r: (r["n_vehicles"], r["model"])):
                writer.writerow([row["size_label"], row["n_vehicles"], row["model"],
                                 f"{row['feature_extraction_ms']:.6g}", f"{row['predict_total_ms']:.6g}",
                                 f"{row['surrogate_total_ms']:.6g}",
                                 f"{row['real_objective_ms']:.6g}" if np.isfinite(row["real_objective_ms"]) else "n/a",
                                 row["real_objective_replications"],
                                 f"{row['speedup_factor']:.6g}" if np.isfinite(row["speedup_factor"]) else "n/a"])
        n_reps = next((r.get("real_objective_replications") for r in real_rows), None)
        if combined:
            models = sorted({r["model"] for r in combined})
            sizes = sorted({r["n_vehicles"] for r in combined})
            real_ms_by_n = {r["n_vehicles"]: r["real_objective_ms"] for r in real_rows
                            if np.isfinite(r["real_objective_ms"])}
            fig, ax = plt.subplots(figsize=(7.5, 5))
            if real_ms_by_n:
                ax.plot(sizes, [real_ms_by_n.get(n, np.nan) for n in sizes], marker="o", color="black",
                        linewidth=2.2, label=f"Real objective (Monte Carlo, {n_reps} replications)", zorder=10)
            for i, model_name in enumerate(models):
                by_n = {r["n_vehicles"]: r["surrogate_total_ms"] for r in combined if r["model"] == model_name}
                ax.plot(sizes, [by_n.get(n, np.nan) for n in sizes], marker=style.MARKERS[i % len(style.MARKERS)],
                        color=style.PALETTE[i % len(style.PALETTE)], label=display_model_name(model_name))
            ax.set_yscale("log")
            ax.set_xlabel("Instance Size (Number of Vehicles)")
            ax.set_ylabel("Full Evaluation Time per Schedule (ms, log scale)")
            ax.set_title(f"Real Objective vs. Surrogate-Based Evaluation Time\n{_pretty_problem_name(problem_name)}")
            ax.legend(fontsize=8)
            _save_all_formats(fig, fig_real_dir, "real_vs_surrogate_time", figures_cfg)

            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(models))
            width = 0.8 / max(len(sizes), 1)
            for i, n in enumerate(sizes):
                label = next((r["size_label"] for r in combined if r["n_vehicles"] == n), str(n))
                ys = []
                for m in models:
                    match = [r["speedup_factor"] for r in combined if r["model"] == m and r["n_vehicles"] == n]
                    ys.append(match[0] if match and np.isfinite(match[0]) else 0.0)
                ax.bar(x + i * width, ys, width=width, label=f"{label} (n={n})", color=style.PALETTE[i % len(style.PALETTE)])
            ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
            ax.set_yscale("log")
            ax.set_xticks(x + width * (len(sizes) - 1) / 2)
            ax.set_xticklabels([display_model_name(m) for m in models], rotation=30, ha="right")
            ax.set_ylabel("Speedup Factor (real objective time / surrogate time, log scale)")
            ax.set_title(f"Surrogate Speedup over the Real Objective -- {_pretty_problem_name(problem_name)}")
            ax.legend(fontsize=8)
            _save_all_formats(fig, fig_real_dir, "speedup_factor", figures_cfg)
        _write_caption(fig_real_dir, "Real Objective vs Surrogate -- how these figures were computed", [
            f"'Real objective' is the wall-clock cost of ONE call to the real objective function (every "
            f"objective at once, Monte Carlo with {n_reps} replications -- the label count N_r of the design "
            f"set). Each model's curve/bar is the FULL cost of the surrogate-based alternative for the same "
            f"schedule if that model served every learned objective: one feature extraction, its prediction "
            f"for each learned objective, plus every objective computed exactly. Every timed function was "
            f"first called untimed (warm-up). speedup_factor = real_objective_ms / surrogate_total_ms; the "
            f"dashed line marks 1 (no speedup).",
        ])
