"""
hypervolume.py  —  hypervolume for multi-objective benchmarking (minimisation).

Two levels of API
-----------------
1. ``compute_hypervolume_batch(fronts_per_method, eps)``  — HIGH-LEVEL.
   Takes every front of every method on ONE instance, builds a SHARED
   ideal/nadir over the union of all points, normalises consistently, and
   returns {method: [hv_run0, hv_run1, ...]}.  Values ARE comparable across
   methods on the same instance.  Use this in evaluation.

2. ``compute_hypervolume(front, reference_point, eps)``  — LOW-LEVEL.
   Self-normalised single front.  Values from different calls are NOT
   comparable.  Use only for within-method tuning ranking / per-generation
   progress.

3. ``best_front_hypervolume(fronts_per_method, eps)``    — REFERENCE.
   HV of the empirical best Pareto front (the non-dominated union of every
   returned front), computed in the SAME shared coordinate system as (1).
   This is the correct HV* reference for the multi-objective gap.

Normalisation (shared): for the union of all points on instance i,
    ideal_k = min_k,  nadir_k = max_k,   p_k_norm = (p_k - ideal_k)/(nadir_k - ideal_k)
Reference point after normalisation: r = [1 + eps]^K   →  HV in [0, (1+eps)^K].
Degenerate objective (nadir == ideal): coordinate set to 0, RuntimeWarning.

Library priority: pygmo → pymoo → built-in exact 2-D sweep (K = 2 only).
"""

import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple

_EPS = 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL API
# ─────────────────────────────────────────────────────────────────────────────
def compute_hypervolume_batch(
    fronts_per_method: Dict[str, List[List[List[float]]]],
    eps: float = _EPS,
) -> Dict[str, List[float]]:
    """
    Comparable HV per run for every method on one instance (shared normalisation).
    Crashed/empty runs must be passed as ``[]`` and receive NaN.
    """
    all_pts = _collect_points(fronts_per_method)
    if not all_pts:
        return {m: [float("nan")] * len(runs) for m, runs in fronts_per_method.items()}

    combined = np.vstack(all_pts)
    ideal, nadir = _shared_bounds(combined)
    n_obj = combined.shape[1]
    ref_norm = np.full(n_obj, 1.0 + eps)
    upper_hv = (1.0 + eps) ** n_obj

    result = {}
    for mname, runs in fronts_per_method.items():
        hvs = []
        for front in runs:
            hvs.append(_front_hv(front, ideal, nadir, ref_norm, upper_hv))
        result[mname] = hvs
    return result


def best_front_hypervolume(
    fronts_per_method: Dict[str, List[List[List[float]]]],
    eps: float = _EPS,
) -> float:
    """
    HV of the empirical best Pareto front = non-dominated union of ALL returned
    fronts, in the SAME shared coordinate system used by
    ``compute_hypervolume_batch``.  Returns NaN if there are no valid points.

    Note: the HV of a point *set* equals the HV of its non-dominated subset
    (dominated points contribute nothing), so it suffices to pool every point.
    """
    all_pts = _collect_points(fronts_per_method)
    if not all_pts:
        return float("nan")
    combined = np.vstack(all_pts)
    ideal, nadir = _shared_bounds(combined)
    n_obj = combined.shape[1]
    ref_norm = np.full(n_obj, 1.0 + eps)
    upper_hv = (1.0 + eps) ** n_obj
    return _front_hv(combined.tolist(), ideal, nadir, ref_norm, upper_hv)


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL API
# ─────────────────────────────────────────────────────────────────────────────
def compute_hypervolume(
    pareto_objectives: List[List[float]],
    reference_point: Optional[List[float]] = None,
    eps: float = _EPS,
) -> float:
    """Self-normalised HV of a single front.  NOT comparable across calls."""
    if not pareto_objectives:
        return 0.0
    pts = np.asarray(pareto_objectives, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return 0.0
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return 0.0
    n_obj = pts.shape[1]

    if reference_point is not None:
        ref = np.asarray(reference_point, dtype=float)
        combined = np.vstack([pts, ref.reshape(1, -1)])
        ideal, nadir = _shared_bounds(combined)
        pts_n = _normalise_shared(pts, ideal, nadir)
        ref_n = _normalise_shared(ref.reshape(1, -1), ideal, nadir)[0]
        upper = float("inf")
    else:
        ideal, nadir = _shared_bounds(pts)
        pts_n = _normalise_shared(pts, ideal, nadir)
        ref_n = np.full(n_obj, 1.0 + eps)
        upper = (1.0 + eps) ** n_obj
    return _compute_hv(pts_n, ref_n, upper)


# ─────────────────────────────────────────────────────────────────────────────
# internal helpers
# ─────────────────────────────────────────────────────────────────────────────
def _collect_points(fronts_per_method) -> List[np.ndarray]:
    pts = []
    for runs in fronts_per_method.values():
        for front in runs:
            if front:
                arr = np.asarray(front, dtype=float)
                if arr.ndim == 2 and arr.shape[1] > 0:
                    arr = arr[np.all(np.isfinite(arr), axis=1)]
                    if len(arr):
                        pts.append(arr)
    return pts


def _front_hv(front, ideal, nadir, ref_norm, upper_hv) -> float:
    if not front:
        return float("nan")
    arr = np.asarray(front, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if len(arr) == 0:
        return float("nan")
    pts_n = _normalise_shared(arr, ideal, nadir)
    return _compute_hv(pts_n, ref_norm, upper_hv)


def _shared_bounds(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ideal = pts.min(axis=0)
    nadir = pts.max(axis=0)
    degen = nadir == ideal
    if degen.any():
        dims = np.where(degen)[0].tolist()
        warnings.warn(
            f"[methods_comparison] Degenerate objective dimension(s) {dims}: all points "
            f"share one value; HV contribution there is 0.",
            RuntimeWarning, stacklevel=3,
        )
    return ideal, nadir


def _normalise_shared(pts, ideal, nadir) -> np.ndarray:
    rng = nadir - ideal
    degen = rng == 0
    rng_safe = np.where(degen, 1.0, rng)
    pts_n = (pts - ideal) / rng_safe
    pts_n[:, degen] = 0.0
    return np.clip(pts_n, 0.0, 1.0)


def _compute_hv(pts_n, ref_n, upper_hv) -> float:
    mask = np.all(pts_n < ref_n, axis=1)
    pts_f = pts_n[mask]
    if len(pts_f) == 0:
        return 0.0

    hv = None
    try:
        import pygmo as pg
        hv = float(pg.hypervolume(pts_f.tolist()).compute(ref_n.tolist()))
    except ImportError:
        pass
    if hv is None:
        try:
            from pymoo.indicators.hv import HV
            hv = float(HV(ref_point=ref_n)(pts_f))
        except ImportError:
            pass
    if hv is None:
        if pts_f.shape[1] == 2:
            hv = _hv2d_sweep(pts_f, ref_n)
        else:
            raise ImportError(
                "No hypervolume backend for K>2. Install one:\n"
                "  pip install pygmo    (recommended)\n"
                "  pip install pymoo    (alternative)"
            )

    if np.isfinite(upper_hv) and hv > upper_hv * 1.05:
        warnings.warn(
            f"[methods_comparison] HV={hv:.6g} exceeds theoretical bound {upper_hv:.6g}; "
            f"returning 0.", RuntimeWarning, stacklevel=3,
        )
        return 0.0
    return hv


def _hv2d_sweep(pts: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2-D HV via sweep line (minimisation, no external deps)."""
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev = ref[1]
    for p in pts:
        contrib = (ref[0] - p[0]) * (prev - p[1])
        if contrib > 0:
            hv += contrib
        prev = min(prev, p[1])
    return float(hv)


# ─────────────────────────────────────────────────────────────────────────────
# CONVERGENCE  —  per-generation HV against a FIXED per-instance reference
# ─────────────────────────────────────────────────────────────────────────────
def convergence_bounds(list_of_fronts, eps: float = _EPS):
    """
    Build the shared (ideal, nadir) over EVERY point seen across all methods,
    runs and generations of one instance.  Reuse for every generation so the HV
    series is monotone, concave and comparable across runs/methods.
    `list_of_fronts` is a flat iterable of fronts (each a list of points).
    Returns (ideal, nadir) or (None, None) if no finite points.
    """
    pts = []
    for fr in list_of_fronts:
        if fr:
            arr = np.asarray(fr, dtype=float)
            if arr.ndim == 2 and arr.shape[1] > 0:
                arr = arr[np.all(np.isfinite(arr), axis=1)]
                if len(arr):
                    pts.append(arr)
    if not pts:
        return None, None
    combined = np.vstack(pts)
    return _shared_bounds(combined)


def hv_series(per_generation_fronts, ideal, nadir, eps: float = _EPS):
    """
    HV at each generation against the fixed reference defined by (ideal, nadir).
    `per_generation_fronts` is a list (over generations) of fronts.
    Returns a list of HV values in the shared normalised space (in [0, ~1]).
    """
    if ideal is None or nadir is None:
        return [float("nan")] * len(per_generation_fronts)
    n_obj = len(ideal)
    ref_norm = np.full(n_obj, 1.0 + eps)
    upper = (1.0 + eps) ** n_obj
    out = []
    for front in per_generation_fronts:
        out.append(_front_hv(front, ideal, nadir, ref_norm, upper) if front else 0.0)
    return out
