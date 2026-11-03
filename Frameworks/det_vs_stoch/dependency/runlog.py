"""
runlog.py -- traceability of a run: a timestamped log of its stages
(<run>/run.log) and the environment it ran in (library versions, machine),
recorded in its manifest.
"""

import datetime
import os
import platform
import sys
import traceback
from importlib import metadata
from typing import Dict

_PACKAGES = ["numpy", "scipy", "numba", "joblib", "matplotlib", "rich", "pymoo", "scikit-learn", "xgboost",
             "catboost", "lightgbm"]


def environment() -> Dict[str, object]:
    """Python, library versions and machine (no personal information)."""
    versions = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    try:
        from joblib import cpu_count
        physical = cpu_count(only_physical_cores=True)
    except Exception:
        physical = None
    return {"python": sys.version.split()[0], "packages": versions, "platform": platform.platform(),
            "processor": platform.processor(), "logical_cores": os.cpu_count(), "physical_cores": physical}


class RunLog:
    """Appends timestamped lines to <run_dir>/run.log (resumed runs append)."""

    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "run.log")

    def __call__(self, message: str) -> None:
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{stamp}  {message}\n")

    def exception(self, where: str) -> None:
        self(f"FAILED in {where}:\n{traceback.format_exc()}")
