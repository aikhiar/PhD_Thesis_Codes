"""
tables.py  —  per-(method, instance, run) scalar extraction + gap + CSVs.

Returns
-------
scores_obj  : {method: {inst_id: [float, ...]}}   HV (multi) | best_objective (mono)
scores_gap  : {method: {inst_id: [float, ...]}}   gap in %
scores_time : {method: {inst_id: [float, ...]}}   exec_time in s

Gap reference (per instance)
----------------------------
Mono  (minimise f):   f* = exact_solver value if available, else best (lowest) f
                      found by any method/run.   gap = 100 * (f - f*) / |f*|
Multi (maximise HV):  HV* = hypervolume of the empirical best Pareto front
                      (non-dominated UNION of every returned front), in the same
                      shared coordinate system as the per-run HVs.
                      gap = 100 * (HV* - HV) / HV*
"""

import os
import numpy as np

from ..utils.io import (write_csv, slug, BLOCK_OBJ_MULTI, BLOCK_OBJ_MONO,
                        BLOCK_GAP, BLOCK_TIME)
from ..utils.hypervolume import compute_hypervolume_batch, best_front_hypervolume


def extract_scores(problem, eval_results: dict, paths: dict,
                   exact_refs: dict = None, refs_out: dict = None) -> tuple:
    """`refs_out` (optional dict) receives {inst_id: gap reference} -- the
    f* / HV* actually used, so later analyses share it."""
    method_names = list(eval_results.keys())
    all_inst_ids = sorted({i for m in eval_results for i in eval_results[m]})

    scores_obj  = {m: {} for m in method_names}
    scores_time = {m: {} for m in method_names}
    obj_ref     = {}                       # inst_id -> reference value
    ref_source  = {}                       # inst_id -> "exact" | "union front" | "best found"

    if problem.is_multiobjective:
        _extract_multiobjective(eval_results, method_names, all_inst_ids,
                                scores_obj, scores_time, obj_ref, ref_source)
    else:
        _extract_monoobjective(eval_results, method_names, all_inst_ids,
                               scores_obj, scores_time)
        if exact_refs:
            info = getattr(problem, "exact_solver_info", None) or {}
            for i in all_inst_ids:
                if i in exact_refs and np.isfinite(exact_refs[i]):
                    obj_ref[i] = float(exact_refs[i])
                    st = str((info.get(i) or {}).get("status", "") or "").lower()
                    # a status other than optimal: the value is the solver's
                    # incumbent, not a proven optimum
                    ref_source[i] = ("exact" if st in ("", "optimal", "opt")
                                     else f"exact incumbent ({st})")

    scores_gap = _compute_gap(scores_obj, problem.is_multiobjective,
                              all_inst_ids, method_names, obj_ref, ref_source)

    # ── CSVs ─────────────────────────────────────────────────────────────────
    obj_block = BLOCK_OBJ_MULTI if problem.is_multiobjective else BLOCK_OBJ_MONO
    obj_label = "Hypervolume" if problem.is_multiobjective else "Objective"
    _write_data_csvs(scores_obj,  paths, slug(obj_block), method_names,
                     all_inst_ids, f"{obj_label}.csv", problem.code)
    _write_data_csvs(scores_gap,  paths, slug(BLOCK_GAP), method_names,
                     all_inst_ids, "Gap.csv", problem.code)
    _write_data_csvs(scores_time, paths, slug(BLOCK_TIME), method_names,
                     all_inst_ids, "Execution time.csv", problem.code)
    _write_reference_csv(paths, slug(BLOCK_GAP), all_inst_ids, obj_ref, ref_source, problem.code)
    if refs_out is not None:
        refs_out.update(obj_ref)

    return scores_obj, scores_gap, scores_time


# ── multi-objective ──────────────────────────────────────────────────────────
def _extract_multiobjective(eval_results, method_names, all_inst_ids,
                            scores_obj, scores_time, obj_ref, ref_source):
    for inst_id in all_inst_ids:
        fronts_per_method, times_per_method = {}, {}
        for mname in method_names:
            fronts, times = [], []
            for r in eval_results[mname].get(inst_id, []):
                if r.get("_crashed") or r.get("_timeout"):
                    fronts.append([]); times.append(np.nan)
                else:
                    fronts.append(r.get("pareto_objectives", []))
                    times.append(r.get("exec_time", np.nan))
            fronts_per_method[mname] = fronts
            times_per_method[mname]  = times

        hv_per_method = compute_hypervolume_batch(fronts_per_method)
        obj_ref[inst_id]    = best_front_hypervolume(fronts_per_method)
        ref_source[inst_id] = "union front"

        for mname in method_names:
            scores_obj[mname][inst_id]  = hv_per_method[mname]
            scores_time[mname][inst_id] = times_per_method[mname]


# ── mono-objective ───────────────────────────────────────────────────────────
def _extract_monoobjective(eval_results, method_names, all_inst_ids,
                           scores_obj, scores_time):
    for mname in method_names:
        for inst_id in all_inst_ids:
            objs, times = [], []
            for r in eval_results[mname].get(inst_id, []):
                if r.get("_crashed") or r.get("_timeout"):
                    objs.append(np.nan); times.append(np.nan)
                else:
                    v = r.get("best_objective", np.nan)
                    objs.append(float(v) if v is not None else np.nan)
                    times.append(r.get("exec_time", np.nan))
            scores_obj[mname][inst_id]  = objs
            scores_time[mname][inst_id] = times


# ── gap ───────────────────────────────────────────────────────────────────────
def _compute_gap(scores_obj, is_multi, all_inst_ids, method_names,
                 obj_ref, ref_source):
    scores_gap = {m: {} for m in method_names}
    for inst_id in all_inst_ids:
        all_vals = [v for m in method_names
                    for v in scores_obj[m].get(inst_id, []) if np.isfinite(v)]

        ref = obj_ref.get(inst_id, np.nan)
        if not np.isfinite(ref):
            if all_vals:
                ref = max(all_vals) if is_multi else min(all_vals)
                ref_source.setdefault(inst_id, "best found")
                obj_ref[inst_id] = ref            # propagate for the reference CSV
            else:
                ref = np.nan

        for m in method_names:
            gaps = []
            for v in scores_obj[m].get(inst_id, []):
                if not np.isfinite(v) or not np.isfinite(ref):
                    gaps.append(np.nan)
                elif is_multi:
                    gaps.append(100.0 * (ref - v) / ref if ref > 0 else np.nan)
                else:
                    denom = abs(ref)
                    gaps.append(100.0 * (v - ref) / denom if denom > 1e-12
                                else (0.0 if v == ref else np.nan))
            scores_gap[m][inst_id] = gaps
    return scores_gap


# ── per-run export (Appendix D: read by AnnexeD/complements.py) ──────────────
def write_tidy_csv(problem, eval_results, blocks, out_path):
    """One row per (method, instance, run). `blocks`: list of (column name,
    scores {m: {i: [per run]}})."""
    rows = []
    for m, per in eval_results.items():
        for i in sorted(per):
            for r, res in enumerate(per[i]):
                row = {"method": m, "label": problem.label(m), "instance": i, "run": r + 1,
                       "seed": res.get("_seed", ""),
                       "crashed": bool(res.get("_crashed")),
                       "timeout": bool(res.get("_timeout"))}
                for col, sc in blocks:
                    vals = sc.get(m, {}).get(i, []) if sc else []
                    v = vals[r] if r < len(vals) else np.nan
                    row[col] = "" if v is None or not np.isfinite(v) else round(float(v), 8)
                rows.append(row)
    if rows:
        write_csv(rows, out_path, header=list(rows[0].keys()))
    return out_path


# ── CSV writers ───────────────────────────────────────────────────────────────
def _write_data_csvs(scores, paths, s, method_names, all_inst_ids, summary_fname, code=None):
    data_dir = paths.get(f"{s}/data")
    if not data_dir:
        return
    for mname in method_names:
        rows = []
        for inst_id in all_inst_ids:
            vals = scores[mname].get(inst_id, [])
            clean = [v for v in vals if np.isfinite(v)]
            rows.append({
                "Instance": inst_id,
                "Mean":   _f(np.mean(clean)            if clean else np.nan),
                "Median": _f(np.median(clean)          if clean else np.nan),
                "Std":    _f(np.std(clean)             if clean else np.nan),
                "P2.5":   _f(np.percentile(clean, 2.5) if clean else np.nan),
                "P97.5":  _f(np.percentile(clean, 97.5) if clean else np.nan),
                "Min":    _f(np.min(clean)             if clean else np.nan),
                "Max":    _f(np.max(clean)             if clean else np.nan),
                "N valid": len(clean),
                "N crashed": len(vals) - len(clean),
            })
        write_csv(rows, os.path.join(data_dir, f"{mname}.csv"), code=code)

    rows = []
    for inst_id in all_inst_ids:
        row = {"Instance": inst_id}
        for mname in method_names:
            vals = scores[mname].get(inst_id, [])
            clean = [v for v in vals if np.isfinite(v)]
            row[mname] = _mean_std(clean)
        rows.append(row)
    if rows:
        write_csv(rows, os.path.join(data_dir, summary_fname),
                  header=list(rows[0].keys()), code=code)


def _write_reference_csv(paths, s, all_inst_ids, obj_ref, ref_source, code=None):
    data_dir = paths.get(f"{s}/data")
    if not data_dir:
        return
    rows = [{"Instance": i,
             "Reference": _f(obj_ref.get(i, np.nan)),
             "Source": ref_source.get(i, "best found")}
            for i in all_inst_ids]
    write_csv(rows, os.path.join(data_dir, "Reference values.csv"),
              header=["Instance", "Reference", "Source"], code=code)


def _f(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return round(float(v), 6)


def _mean_std(clean):
    """Combined 'mean (std)' cell; blank if no valid values."""
    if not clean:
        return ""
    return f"{np.mean(clean):.4g} ({np.std(clean):.3g})"
