"""
io.py  —  folder layout + pickle / JSON / CSV helpers.

The experiment folder name is DETERMINISTIC in (problem name, config) so that
re-running the same experiment reuses the same folder and therefore finds its
checkpoints and `instances.pkl` (genuine resume).  The run timestamp is stored
in the provenance manifest instead of the folder name.
"""

import os
import csv
import json
import pickle
import hashlib
import numpy as np
from datetime import datetime


# ── slug + canonical block names ─────────────────────────────────────────────
def slug(name: str) -> str:
    """Folder name → dict key (lowercase, underscores)."""
    return name.lower().replace(" ", "_").replace("'", "").replace("-", "_")


BLOCK_OBJ_MULTI = "Hypervolume Comparison"
BLOCK_OBJ_MONO  = "Objective Comparison"
BLOCK_GAP       = "Gap Comparison"
BLOCK_TIME      = "Execution Time Comparison"


# ── experiment root ──────────────────────────────────────────────────────────
def make_experiment_folder(output_dir: str, problem_name: str, cfg: dict = None,
                           code: str = None) -> str:
    suffix = ""
    if cfg is not None:
        h = hashlib.sha1(json.dumps(_hashable_cfg(cfg), sort_keys=True,
                                    default=str).encode()).hexdigest()[:8]
        suffix = f"_{h}"
    prefix = f"{code}_" if code else ""
    folder = os.path.join(output_dir, f"{prefix}{problem_name}{suffix}")
    os.makedirs(folder, exist_ok=True)
    return folder


def _hashable_cfg(cfg: dict) -> dict:
    """The sections of cfg that decide the runs (config.RUN_SECTIONS): a change
    of a report, statistics or figure setting keeps the same folder and its
    cached runs."""
    from ..config import RUN_SECTIONS
    return {k: cfg[k] for k in RUN_SECTIONS if k in cfg}


# ── CODE_ prefix (every framework-saved FILE begins with the problem's code) ──
def coded_path(path: str, code: str = None) -> str:
    if not code:
        return path
    d, base = os.path.split(path)
    if base.startswith(f"{code}_"):
        return path
    return os.path.join(d, f"{code}_{base}")


# ── analysis directory tree ──────────────────────────────────────────────────
def make_analysis_tree(main_folder: str, is_multiobjective: bool,
                       has_size_classes: bool) -> dict:
    """The paths of the Results Analysis/ tree, {key: absolute path}. A folder
    is created by the first file written into it, so that the tree holds only
    what the study reports."""
    ra = _d(main_folder, "Results Analysis")
    paths = {"root": ra}

    obj_block = BLOCK_OBJ_MULTI if is_multiobjective else BLOCK_OBJ_MONO
    for block in (obj_block, BLOCK_TIME):
        s = slug(block)
        base = _d(ra, block)

        paths[f"{s}/data"]         = _d(base, "Data")
        fig                        = _d(base, "Figures")
        paths[f"{s}/figures"]      = fig
        paths[f"{s}/box"]          = _d(fig, "Distribution", "Box Plots")
        paths[f"{s}/perfprofiles"] = _d(fig, "Performance Profiles")

        stat                       = _d(base, "Statistical Tests")
        paths[f"{s}/stats"]        = stat
        paths[f"{s}/friedman"]     = _d(stat, "Friedman Test")
        paths[f"{s}/nemenyi"]      = _d(stat, "Nemenyi Post-hoc")
        paths[f"{s}/wilcoxon"]     = _d(stat, "Wilcoxon Test")
        paths[f"{s}/cliffdelta"]   = _d(stat, "Cliff's Delta")
        paths[f"{s}/control"]      = _d(stat, "Against Control")
        paths[f"{s}/tables"]       = _d(base, "Tables")

        if has_size_classes:
            psd = _d(fig, "Per Size", "Distribution")
            paths[f"{s}/persize/box"]    = _d(psd, "Box Plots")
            paths[f"{s}/persize/violin"] = _d(psd, "Violin Plots")

    # the gap to the reference (best front, or exact value): data and table only
    s = slug(BLOCK_GAP)
    paths[f"{s}/data"]   = _d(ra, BLOCK_GAP, "Data")
    paths[f"{s}/tables"] = _d(ra, BLOCK_GAP, "Tables")

    paths["convergence"] = _d(ra, "Convergence Curves")
    paths["rank"]        = _d(ra, "Rank Heatmaps")
    paths["tradeoff"]    = _d(ra, "Quality vs Time Tradeoff")
    paths["summary"]     = _d(ra, "Summary")
    if is_multiobjective:
        paths["pareto"]     = _d(ra, "Pareto Fronts")
        paths["indicators"] = _d(ra, "Multi-objective Indicators")
    return paths


def _d(parent, *names):
    return os.path.join(parent, *names)


# ── file helpers ─────────────────────────────────────────────────────────────
def save_pickle(obj, path: str, code: str = None):
    """Atomic write (temp file + os.replace): an interruption cannot leave a
    truncated instances.pkl or results pkl behind."""
    path = coded_path(path, code)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)


def load_pickle(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_json(obj, path: str, code: str = None):
    path = coded_path(path, code)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(obj), fh, indent=2, default=str, allow_nan=False)


def _json_safe(o):
    """Recursively convert NaN/Inf to None so output is valid JSON."""
    import math
    if isinstance(o, np.generic):          # numpy scalars -> Python scalars
        o = o.item()
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(rows: list, path: str, header: list = None, code: str = None):
    path = coded_path(path, code)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        if rows and isinstance(rows[0], dict):
            writer = csv.DictWriter(fh, fieldnames=header or list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(fh)
            if header:
                writer.writerow(header)
            writer.writerows(rows)


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
