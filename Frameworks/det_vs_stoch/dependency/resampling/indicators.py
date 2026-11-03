"""
indicators.py -- descriptors of realized fronts (minimisation) used by the
report of Chapter 6, in a normalised space shared by the fronts being compared
(ideal/nadir of their union), as the hypervolume does.

  normalise_together  shared ideal/nadir normalisation of several point sets
  knee_point          point of a front closest to the ideal point (normalised space)
  attainment_surface  2-D empirical attainment surface at a given level (Fonseca et al., 2001)
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np


def _as_array(front) -> np.ndarray:
    arr = np.asarray(front, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 0))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[np.all(np.isfinite(arr), axis=1)]


def normalise_together(fronts: Dict[str, Sequence]) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """({name: normalised points}, ideal, nadir) with one ideal/nadir for all."""
    arrays = {k: _as_array(v) for k, v in fronts.items()}
    pts = [a for a in arrays.values() if a.size]
    if not pts:
        return arrays, np.zeros(0), np.zeros(0)
    allp = np.vstack(pts)
    ideal, nadir = allp.min(axis=0), allp.max(axis=0)
    span = np.where(nadir > ideal, nadir - ideal, 1.0)
    return {k: ((a - ideal) / span if a.size else a) for k, a in arrays.items()}, ideal, nadir


def knee_point(front, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    """Raw objective values of the point of `front` closest (Euclidean) to the
    ideal point once normalised with (ideal, nadir)."""
    a = _as_array(front)
    if not len(a):
        return np.full(len(ideal), np.nan)
    span = np.where(nadir > ideal, nadir - ideal, 1.0)
    return a[int(np.argmin(np.sqrt((((a - ideal) / span) ** 2).sum(axis=1))))]


def attainment_surface(runs: List, level: float) -> np.ndarray:
    """2-D empirical attainment surface: the staircase of points attained by at
    least ceil(level * R) of the R runs (level 0.5 = median surface). Returns
    the staircase corners (x ascending, y descending); empty if never attained."""
    fronts = [_as_array(r)[:, :2] for r in runs]
    fronts = [f for f in fronts if len(f)]
    R = len(fronts)
    if R == 0:
        return np.zeros((0, 2))
    need = max(1, int(np.ceil(level * R)))
    xs = np.unique(np.concatenate([f[:, 0] for f in fronts]))
    corners = []
    for x in xs:
        m = np.sort([f[f[:, 0] <= x, 1].min() if np.any(f[:, 0] <= x) else np.inf for f in fronts])
        y = m[need - 1]
        if np.isfinite(y) and (not corners or y < corners[-1][1]):
            corners.append((x, y))
    return np.asarray(corners, dtype=float).reshape(-1, 2)
