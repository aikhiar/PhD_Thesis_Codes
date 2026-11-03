"""
indicators.py  —  multi-objective quality indicators (minimisation).

Hypervolume answers "how much objective space is dominated", but reviewers in
top-tier OR/EC venues expect convergence AND diversity indicators relative to a
common reference front.  These are pure-NumPy and computed against the empirical
best Pareto front (non-dominated union of all returned fronts) per instance.

Indicators (all lower-is-better)
--------------------------------
IGD+      : modified inverted generational distance (Ishibuchi et al. 2015);
            Pareto-compliant; average distance from each reference point to the
            nearest dominating-or-equal region of the approximation set.
epsilon   : additive unary epsilon indicator (Zitzler et al. 2003); the smallest
            shift that makes the approximation set dominate the reference set.
spacing   : Schott's spacing; uniformity of the approximation set (diversity).

Scale: by default (config["indicators"]["normalise"]) every point is first
mapped to [0, 1] per objective with the same per-instance ideal/nadir as the
hypervolume (union of all returned points), so no objective dominates the
distances because of its units.
"""

import numpy as np
from typing import Dict, List


def non_dominated(points: np.ndarray) -> np.ndarray:
    """Return the non-dominated subset (minimisation)."""
    if len(points) == 0:
        return points
    pts = np.asarray(points, dtype=float)
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        dominated = np.all(pts <= pts[i], axis=1) & np.any(pts < pts[i], axis=1)
        keep[dominated] = False
    return pts[keep]


def igd_plus(approx: np.ndarray, reference: np.ndarray) -> float:
    approx = np.asarray(approx, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(approx) == 0 or len(reference) == 0:
        return float("nan")
    dists = []
    for r in reference:
        diff = np.maximum(approx - r, 0.0)          # only worse-than-r coords count
        dists.append(np.min(np.sqrt(np.sum(diff ** 2, axis=1))))
    return float(np.mean(dists))


def epsilon_additive(approx: np.ndarray, reference: np.ndarray) -> float:
    approx = np.asarray(approx, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(approx) == 0 or len(reference) == 0:
        return float("nan")
    eps = -np.inf
    for r in reference:
        eps = max(eps, np.min(np.max(approx - r, axis=1)))
    return float(eps)


def spacing(approx: np.ndarray, ddof: int = 1) -> float:
    """Schott's spacing: sqrt( sum (d_i - mean d)^2 / (n - ddof) ), d_i the
    L1 distance from point i to its nearest neighbour. ddof=1 is Schott's
    (1995) definition."""
    approx = np.asarray(approx, dtype=float)
    if len(approx) < 2:
        return float("nan")
    d = []
    for i in range(len(approx)):
        others = np.delete(approx, i, axis=0)
        d.append(np.min(np.sum(np.abs(approx[i] - others), axis=1)))
    d = np.asarray(d)
    return float(np.sqrt(np.sum((d - d.mean()) ** 2) / (len(d) - ddof)))


def _scaler(points: np.ndarray):
    """x -> (x - ideal) / (nadir - ideal) per objective; a degenerate
    objective (nadir == ideal) maps to 0."""
    ideal = points.min(axis=0)
    span = points.max(axis=0) - ideal
    span = np.where(span > 0, span, 1.0)
    return lambda x: (np.asarray(x, dtype=float) - ideal) / span


def compute_indicators(eval_results: dict, method_names: List[str],
                       all_inst_ids: List[int], normalise: bool = True,
                       spacing_ddof: int = 1) -> Dict[str, dict]:
    """
    Returns {indicator: {method: {inst_id: [per-run values]}}} for IGD+, epsilon,
    spacing — each run's approximation set compared to the per-instance union
    reference front (normalised coordinates when `normalise`).
    """
    out = {"igd_plus": {m: {} for m in method_names},
           "epsilon":  {m: {} for m in method_names},
           "spacing":  {m: {} for m in method_names}}

    for inst_id in all_inst_ids:
        union = []
        for m in method_names:
            for r in eval_results[m].get(inst_id, []):
                if not r.get("_crashed") and not r.get("_timeout"):
                    union.extend(r.get("pareto_objectives", []) or [])
        if not union:
            for m in method_names:
                n = len(eval_results[m].get(inst_id, []))
                for key in out:
                    out[key][m][inst_id] = [np.nan] * n
            continue
        union = np.asarray(union, dtype=float)
        scale = _scaler(union) if normalise else (lambda x: np.asarray(x, dtype=float))
        ref = non_dominated(scale(union))

        for m in method_names:
            igd_l, eps_l, sp_l = [], [], []
            for r in eval_results[m].get(inst_id, []):
                if r.get("_crashed") or r.get("_timeout"):
                    igd_l.append(np.nan); eps_l.append(np.nan); sp_l.append(np.nan)
                    continue
                pts = r.get("pareto_objectives", []) or []
                if not pts:
                    igd_l.append(np.nan); eps_l.append(np.nan); sp_l.append(np.nan)
                    continue
                arr = scale(pts)
                igd_l.append(igd_plus(arr, ref))
                eps_l.append(epsilon_additive(arr, ref))
                sp_l.append(spacing(arr, spacing_ddof))
            out["igd_plus"][m][inst_id] = igd_l
            out["epsilon"][m][inst_id]  = eps_l
            out["spacing"][m][inst_id]  = sp_l
    return out
