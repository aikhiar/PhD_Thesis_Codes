"""
provenance.py  —  write a reproducibility manifest for each experiment.

Captures everything needed to reproduce a run: Python/OS, key package
versions, git commit (if any), the fully-resolved config, the RNG scheme, and a
hash of every instance.  Written to <main_folder>/run_manifest.json.
"""

import os
import sys
import platform
import subprocess
import importlib

from .io import save_json, now_string
from .checkpoint import instance_signature


def _pkg_version(name: str):
    try:
        return importlib.import_module(name).__version__
    except Exception:
        return None


def _git_commit():
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL,
                                      cwd=os.path.dirname(os.path.abspath(__file__)))
        return out.decode().strip()
    except Exception:
        return None


def write_manifest(main_folder: str, cfg: dict, problem, instances,
                   exact_refs=None) -> str:
    manifest = {
        "timestamp": now_string(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: _pkg_version(p) for p in
                     ("numpy", "scipy", "matplotlib", "numba", "joblib")},
        "git_commit": _git_commit(),
        "rng_scheme": "numpy.random.SeedSequence([seed_base, phase, "
                      "method_idx, inst_idx, (combo_idx,) run])",
        "problem": {
            "name": problem.name,
            "is_multiobjective": problem.is_multiobjective,
            "n_objectives": problem.n_objectives,
            "methods": [problem.label(m) for m in problem.method_names],
            "has_exact_solver": problem.exact_solver is not None,
            "exact_solver_timelimit": problem.exact_solver_timelimit,
            "n_instances": len(instances),
        },
        "instance_hashes": {i + 1: instance_signature(inst)
                            for i, inst in enumerate(instances)},
        "exact_references": exact_refs or None,
        "config": cfg,
    }
    path = os.path.join(main_folder, "run_manifest.json")
    save_json(manifest, path, code=problem.code)
    return path
