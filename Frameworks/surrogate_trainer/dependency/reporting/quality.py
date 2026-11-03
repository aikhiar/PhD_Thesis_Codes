"""
quality.py -- the reports on what matters to the optimizer that will use the
surrogates (the fidelity of the deployed surrogates in Chapters 5 and 6),
beyond the regression metrics of reporting.py:

  Per learned objective (<objective>/Metrics):
    ranking_by_model       Kendall's tau between predicted and true values within
                           every validation instance (the optimizer compares
                           schedules of one instance, never across instances)
  Joint, all objectives (optimizer_quality/):
    Pareto-dominance agreement between the true objective vectors and those
    predicted by the deployed surrogates (exact objectives exact), the share of
    dominance relations preserved and reversed, precision/recall of the
    non-dominated set, per validation instance and summarised per size class.
  Training data (training_data/):
    records per size class and algorithm.
  Cost accounting (execution_time_analysis/):
    data generation + training cost, saving per evaluation, break-even number of
    evaluations and of optimizer runs.
"""

import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau

from . import style
from .progress import display_model_name
from .tables import write_table

style.apply_thesis_style()

SIZE_CLASS_ORDER = ["small", "medium", "moderate", "large"]


# ============================================================================
# HELPERS
# ============================================================================

def _save(fig, out_dir: str, stem: str, figures_cfg: Dict[str, Any]) -> None:
    for ext in figures_cfg.get("formats", ["png", "svg"]):
        d = os.path.join(out_dir, ext.upper())
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{stem}.{ext}"), dpi=figures_cfg.get("dpi", 400), bbox_inches="tight")
    plt.close(fig)


def _caption(out_dir: str, title: str, paragraphs: List[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "_HOW_IT_IS_COMPUTED.txt")
    existing = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
    block = "\n".join([title, "=" * len(title), ""] + [p + "\n" for p in paragraphs])
    if block in existing:                       # a resumed run rewrites its reports
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write((existing + "\n\n" if existing else "") + block)


def _classes_in_order(classes) -> List[str]:
    present = sorted(set(np.asarray(classes).tolist()))
    return [c for c in SIZE_CLASS_ORDER if c in present] + [c for c in present if c not in SIZE_CLASS_ORDER]


def nondominated_mask(points: np.ndarray) -> np.ndarray:
    """True for the non-dominated rows (minimisation)."""
    P = np.asarray(points, dtype=float)
    n = len(P)
    if n == 0:
        return np.zeros(0, dtype=bool)
    le = np.all(P[:, None, :] <= P[None, :, :], axis=2)
    lt = np.any(P[:, None, :] < P[None, :, :], axis=2)
    dominated = np.any(le & lt, axis=0)
    return ~dominated


# ============================================================================
# THE JOINT VALIDATION VIEW
# ============================================================================

def validation_view(training_results: Dict[str, Any], metadata: Dict[str, np.ndarray],
                    objective_names: List[str]) -> Dict[str, Any]:
    """The validation records (identical for every learned objective: same rows,
    same split) with their instance, class, the true value of every objective, and
    the deployed prediction of every learned objective (exact objectives: their
    true value)."""
    results = list(training_results.values())
    idx = results[0].val_index
    for r in results[1:]:
        if r.val_index is None or not np.array_equal(r.val_index, idx):
            raise ValueError("the validation rows differ between objectives")
    actual = metadata["objectives"][idx]
    predicted = actual.copy()
    for name, r in training_results.items():
        k = objective_names.index(name)
        predicted[:, k] = r.val_predictions[r.deployed_model_name or r.best_model_name]
    return {"index": idx, "groups": metadata["groups"][idx], "classes": metadata["classes"][idx],
            "actual": actual, "predicted": predicted}


# ============================================================================
# PER-OBJECTIVE QUALITY
# ============================================================================

def report_objective_quality(result, view: Dict[str, Any], out_dir: str, qcfg: Dict[str, Any]) -> Dict[str, Any]:
    """Within-instance ranking (Kendall's tau) of every candidate for one learned
    objective."""
    metrics_dir = os.path.join(out_dir, "Metrics")
    deployed = result.deployed_model_name or result.best_model_name
    y = result.y_val
    classes = view["classes"]
    groups = view["groups"]

    # within-instance ranking (Kendall tau), every candidate
    rng = np.random.default_rng(0)
    cap = int(qcfg.get("ranking_max_rows", 1500))
    tau_rows = []
    for g in sorted(set(groups.tolist())):
        rows_g = np.where(groups == g)[0]
        if len(rows_g) > cap:
            rows_g = np.sort(rng.choice(rows_g, size=cap, replace=False))
        for model, pred in result.val_predictions.items():
            tau = kendalltau(y[rows_g], pred[rows_g])[0] if len(rows_g) >= 3 else np.nan
            tau_rows.append({"instance": g, "size_class": classes[rows_g[0]], "model": model,
                             "kendall_tau": float(tau) if np.isfinite(tau) else float("nan")})
    rank_rows = []
    for model in result.val_predictions:
        taus = np.array([t["kendall_tau"] for t in tau_rows if t["model"] == model], dtype=float)
        taus = taus[np.isfinite(taus)]
        rank_rows.append({"model": model, "deployed": model == deployed, "n_instances": len(taus),
                          "mean_tau": float(taus.mean()) if len(taus) else float("nan"),
                          "median_tau": float(np.median(taus)) if len(taus) else float("nan"),
                          "min_tau": float(taus.min()) if len(taus) else float("nan")})
    write_table(rank_rows, [("model", "Model"), ("deployed", "Deployed"), ("n_instances", "Validation instances"),
                            ("mean_tau", "Mean tau"), ("median_tau", "Median tau"), ("min_tau", "Min tau")],
                metrics_dir, "ranking_by_model")
    write_table(tau_rows, [("instance", "Instance"), ("size_class", "Class"), ("model", "Model"),
                           ("kendall_tau", "Kendall tau")], metrics_dir, "ranking_per_instance")

    _caption(metrics_dir, "Optimizer-oriented accuracy", [
        "ranking_by_model / ranking_per_instance: Kendall's tau between the predicted and true values of the "
        f"schedules of every validation instance (at most {cap} schedules per instance, random subsample); the "
        "optimizer compares schedules of one instance only, so this is the accuracy it relies on."])
    return {"ranking": rank_rows}


# ============================================================================
# JOINT QUALITY FOR THE OPTIMIZER (all objectives)
# ============================================================================

def _relation(P: np.ndarray) -> np.ndarray:
    """R[i, j] = 1 if i dominates j, -1 if j dominates i, 0 otherwise."""
    le = np.all(P[:, None, :] <= P[None, :, :], axis=2)
    lt = np.any(P[:, None, :] < P[None, :, :], axis=2)
    dom = le & lt
    return dom.astype(int) - dom.T.astype(int)


def report_optimizer_quality(code: str, view: Dict[str, Any], objective_names: List[str],
                             surrogate_names: List[str], metadata: Dict[str, np.ndarray], out_dir: str,
                             figures_cfg: Dict[str, Any], qcfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per validation instance: agreement of the Pareto-dominance relation of
    every pair of schedules between true values and deployed predictions, share
    of true dominance relations preserved and reversed, precision/recall/F1 of
    the predicted non-dominated set; summaries per size class."""
    rng = np.random.default_rng(0)
    cap = int(qcfg.get("ranking_max_rows", 1500))
    rows = []
    for g in sorted(set(view["groups"].tolist())):
        idx = np.where(view["groups"] == g)[0]
        if len(idx) > cap:
            idx = np.sort(rng.choice(idx, size=cap, replace=False))
        A, P = view["actual"][idx], view["predicted"][idx]
        RA, RP = _relation(A), _relation(P)
        iu = np.triu_indices(len(idx), k=1)
        ra, rp = RA[iu], RP[iu]
        dominated_pairs = ra != 0
        true_front, pred_front = nondominated_mask(A), nondominated_mask(P)
        tp = int(np.sum(true_front & pred_front))
        precision = tp / pred_front.sum() if pred_front.sum() else float("nan")
        recall = tp / true_front.sum() if true_front.sum() else float("nan")
        rows.append({
            "instance": g, "size_class": view["classes"][idx[0]], "n_schedules": len(idx),
            "pair_agreement": float(np.mean(ra == rp)) if len(ra) else float("nan"),
            "dominance_preserved": float(np.mean(rp[dominated_pairs] == ra[dominated_pairs])) if dominated_pairs.any() else float("nan"),
            "dominance_reversed": float(np.mean(rp[dominated_pairs] == -ra[dominated_pairs])) if dominated_pairs.any() else float("nan"),
            "front_size_true": int(true_front.sum()), "front_size_predicted": int(pred_front.sum()),
            "front_precision": precision, "front_recall": recall,
            "front_f1": 2 * precision * recall / (precision + recall) if (np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0) else float("nan"),
        })
    cols = [("instance", "Instance"), ("size_class", "Class"), ("n_schedules", "Schedules"),
            ("pair_agreement", "Pair agreement"), ("dominance_preserved", "Dominance preserved"),
            ("dominance_reversed", "Dominance reversed"), ("front_size_true", "True front"),
            ("front_size_predicted", "Predicted front"), ("front_precision", "Front precision"),
            ("front_recall", "Front recall"), ("front_f1", "Front F1")]
    metrics_dir = os.path.join(out_dir, "Metrics")
    write_table(rows, cols, metrics_dir, "dominance_per_instance")
    summary = []
    for scope in _classes_in_order([r["size_class"] for r in rows]) + ["overall"]:
        sel = [r for r in rows if scope == "overall" or r["size_class"] == scope]
        summary.append({"scope": scope, "n_instances": len(sel),
                        **{k: float(np.nanmean([r[k] for r in sel])) for k, _ in cols[3:]}})
    write_table(summary, [("scope", "Scope"), ("n_instances", "Instances")] + cols[3:], metrics_dir, "dominance_summary")
    _caption(metrics_dir, "Quality of the deployed surrogates for the optimizer (all objectives)", [
        f"Learned objectives: {', '.join(surrogate_names)}; computed exactly: "
        f"{', '.join(n for n in objective_names if n not in surrogate_names) or '(none)'}.",
        "For every validation instance (at most "
        f"{cap} schedules, random subsample), every pair of schedules is compared by Pareto dominance on the true "
        "objective vectors and on the predicted ones (deployed models; exact objectives exact). pair_agreement: share "
        "of pairs with the same relation (i dominates j, j dominates i, or neither); dominance_preserved: share of "
        "the truly dominated pairs predicted in the same direction; dominance_reversed: share predicted in the "
        "opposite direction (the surrogates would prefer the dominated schedule). front_precision / front_recall / "
        "front_F1: of the schedules the surrogates find non-dominated, the share truly non-dominated, and of the "
        "truly non-dominated ones, the share the surrogates keep."])

    fig_dir = os.path.join(out_dir, "Figures", "Dominance")
    classes = _classes_in_order([r["size_class"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, keys, title in ((axes[0], ("pair_agreement", "dominance_preserved", "dominance_reversed"), "Pairs of schedules"),
                            (axes[1], ("front_precision", "front_recall", "front_f1"), "Non-dominated set")):
        width = 0.8 / len(keys)
        for j, key in enumerate(keys):
            data = [[r[key] for r in rows if r["size_class"] == c and np.isfinite(r[key])] or [np.nan] for c in classes]
            bp = ax.boxplot(data, positions=np.arange(len(classes)) + (j - (len(keys) - 1) / 2) * width,
                            widths=width * 0.9, patch_artist=True, showmeans=True)
            for box in bp["boxes"]:
                box.set_facecolor(style.PALETTE[j % 8])
            ax.plot([], [], color=style.PALETTE[j % 8], linewidth=8, label=key.replace("_", " "))
        ax.set_xticks(np.arange(len(classes)))
        ax.set_xticklabels([c.capitalize() for c in classes])
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle(f"Dominance Agreement of the Deployed Surrogates -- {code}")
    fig.tight_layout()
    _save(fig, fig_dir, "dominance_agreement", figures_cfg)
    _caption(fig_dir, "Dominance agreement", ["Per size class, one value per validation instance: the measures of "
                                              "Metrics/dominance_per_instance."])
    return summary


# ============================================================================
# TRAINING DATA
# ============================================================================

def report_training_data(code: str, metadata: Dict[str, np.ndarray], out_dir: str) -> None:
    """Records per size class and algorithm."""
    metrics_dir = os.path.join(out_dir, "Metrics")
    classes = _classes_in_order(metadata["classes"])
    algos = sorted(set(metadata["algorithms"].tolist()))
    rows = []
    for c in classes:
        mc = metadata["classes"] == c
        row = {"size_class": c, "n_instances": len(set(metadata["groups"][mc].tolist())), "n_records": int(mc.sum())}
        for a in algos:
            row[f"records_{a}"] = int(np.sum(mc & (metadata["algorithms"] == a)))
        rows.append(row)
    rows.append({"size_class": "overall", "n_instances": len(set(metadata["groups"].tolist())),
                 "n_records": len(metadata["groups"]),
                 **{f"records_{a}": int(np.sum(metadata["algorithms"] == a)) for a in algos}})
    write_table(rows, [("size_class", "Class"), ("n_instances", "Instances"), ("n_records", "Records")]
                + [(f"records_{a}", a) for a in algos], metrics_dir, "records_by_class_and_algorithm")

    _caption(metrics_dir, "Training data", [
        "records_by_class_and_algorithm: number of labelled schedules (and of training instances) per size class "
        "and algorithm."])


# ============================================================================
# COST ACCOUNTING
# ============================================================================

def report_cost_accounting(code: str, dataset_summary: Dict[str, Any], training_results: Dict[str, Any],
                           inference_rows: List[Dict[str, Any]], real_rows: List[Dict[str, Any]],
                           evaluations_per_run: Optional[int], out_dir: str) -> List[Dict[str, Any]]:
    """Up-front cost of the surrogates (data generation + tuning/fitting every
    candidate) against the saving per evaluation of the deployed surrogates
    (feature extraction once, every deployed prediction, exact objectives) over
    the real Monte Carlo objective, per size class: break-even number of
    evaluations and of optimizer runs."""
    datagen_s = float(dataset_summary.get("datagen_compute_seconds", 0.0)) + float(dataset_summary.get("datagen_warmup_seconds", 0.0))
    train_all_s = float(sum(m.fit_time_seconds for r in training_results.values() for m in r.results.values()))
    train_dep_s = float(sum(r.results[r.deployed_model_name or r.best_model_name].fit_time_seconds
                            for r in training_results.values()))
    upfront_s = datagen_s + train_all_s
    deployed = {o: (r.deployed_model_name or r.best_model_name) for o, r in training_results.items()}
    real_by = {r["size_label"]: r for r in real_rows}
    rows = []
    for label, real in real_by.items():
        sel = [r for r in inference_rows if r["size_label"] == label and deployed.get(r["objective"]) == r["model"]]
        if not sel:
            continue
        surrogate_ms = sel[0]["feature_extraction_ms"] + sum(r["predict_ms"] for r in sel) + real.get("exact_objective_ms", 0.0)
        saving_ms = real["real_objective_ms"] - surrogate_ms
        evals = upfront_s * 1000.0 / saving_ms if saving_ms > 0 else float("nan")
        rows.append({"size_class": label, "n_vehicles": real["n_vehicles"], "real_objective_ms": real["real_objective_ms"],
                     "surrogate_ms": surrogate_ms, "saving_ms": saving_ms, "pays_off": bool(saving_ms > 0),
                     "speedup": real["real_objective_ms"] / surrogate_ms if surrogate_ms > 0 else float("nan"),
                     "break_even_evaluations": evals,
                     "break_even_runs": evals / evaluations_per_run if (evaluations_per_run and np.isfinite(evals)) else float("nan")})
    rows.sort(key=lambda r: r["n_vehicles"])
    data_dir = os.path.join(out_dir, "Data")
    write_table(rows, [("size_class", "Class"), ("n_vehicles", "n"), ("real_objective_ms", "Real objective (ms)"),
                       ("surrogate_ms", "Surrogates (ms)"), ("saving_ms", "Saving (ms)"), ("speedup", "Speedup"),
                       ("pays_off", "Pays off"),
                       ("break_even_evaluations", "Break-even evaluations"),
                       ("break_even_runs", "Break-even optimizer runs")], data_dir, "cost_accounting")
    write_table([{"item": "data generation (compute, incl. warm-up)", "seconds": datagen_s},
                 {"item": "data generation (wall clock)", "seconds": float(dataset_summary.get("datagen_wall_seconds", 0.0))},
                 {"item": "tuning + fitting, every candidate", "seconds": train_all_s},
                 {"item": "tuning + fitting, deployed models only", "seconds": train_dep_s},
                 {"item": "up-front cost (data generation + every candidate)", "seconds": upfront_s}],
                [("item", "Item"), ("seconds", "Seconds")], data_dir, "upfront_cost")

    _caption(data_dir, "Cost accounting", [
        "upfront_cost: compute seconds of the generation of the training data (sum over the (instance, algorithm) "
        "tasks, incl. the JIT warm-up) and of the tuning and fitting of every candidate (and of the deployed ones).",
        "cost_accounting, per size class: real objective and deployed surrogates per evaluation (feature extraction "
        "once, every deployed prediction, exact objectives), saving per evaluation, speedup, break-even number of "
        "evaluations = up-front cost / saving, and break-even number of optimizer runs = that / evaluations per run "
        f"({evaluations_per_run} = population x (generations + 1) of the study)."])
    return rows
