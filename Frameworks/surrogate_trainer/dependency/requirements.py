"""
requirements.py -- checks, at start-up, that every package this framework
imports is installed, and installs the missing ones with pip (in the Python
that runs the framework). Standard library only, so it runs before any
third-party import.

    REQUIREMENTS: (import name, pip name, minimum version, version installed
                   when missing -- the version the framework was tested with)

A package present but older than its minimum is upgraded to the tested
version. Set the environment variable NO_AUTO_INSTALL=1 to only check (the
framework then stops with the list of what is missing).
"""

import importlib
import importlib.metadata as md
import importlib.util
import os
import subprocess
import sys

REQUIREMENTS = [
    ("numpy", "numpy", "1.24", "2.2.6"),
    ("scipy", "scipy", "1.10", "1.15.3"),
    ("numba", "numba", "0.57", "0.61.2"),
    ("matplotlib", "matplotlib", "3.7", "3.10.9"),
    ("joblib", "joblib", "1.2", "1.5.1"),
    ("tqdm", "tqdm", "4.60", "4.67.1"),
    ("rich", "rich", "13.0", "14.0.0"),
    ("sklearn", "scikit-learn", "1.3", "1.7.2"),
    ("threadpoolctl", "threadpoolctl", "3.1", "3.6.0"),
    ("xgboost", "xgboost", "2.0", "3.1.1"),
    ("catboost", "catboost", "1.2", "1.2.8"),
]


def _version_tuple(v: str):
    out = []
    for part in v.replace("+", ".").split("."):
        num = "".join(ch for ch in part if ch.isdigit())
        if not num:
            break
        out.append(int(num))
    return tuple(out)


def _installed_version(pip_name: str):
    try:
        return md.version(pip_name)
    except md.PackageNotFoundError:
        return None


def ensure_requirements(requirements=None, verbose: bool = True) -> None:
    """Installs the missing (or too old) packages; raises if one still fails."""
    requirements = REQUIREMENTS if requirements is None else requirements
    todo = []
    for import_name, pip_name, minimum, tested in requirements:
        found = importlib.util.find_spec(import_name) is not None
        version = _installed_version(pip_name) if found else None
        if not found:
            todo.append((pip_name, tested, "missing"))
        elif version is not None and _version_tuple(version) < _version_tuple(minimum):
            todo.append((pip_name, tested, f"version {version} < {minimum}"))
    if not todo:
        return
    if os.environ.get("NO_AUTO_INSTALL") == "1":
        raise RuntimeError("missing packages (NO_AUTO_INSTALL=1): " +
                           ", ".join(f"{p} ({why})" for p, _, why in todo))
    for pip_name, tested, why in todo:
        if verbose:
            print(f"[requirements] {pip_name}: {why} -> pip install {pip_name}=={tested}", flush=True)
        cmd = [sys.executable, "-m", "pip", "install", f"{pip_name}=={tested}"]
        if subprocess.call(cmd) != 0:
            if verbose:
                print(f"[requirements] {pip_name}=={tested} unavailable here, trying the latest compatible version",
                      flush=True)
            if subprocess.call([sys.executable, "-m", "pip", "install", pip_name]) != 0:
                raise RuntimeError(f"could not install {pip_name}; install it manually")
    importlib.invalidate_caches()
    still = [p for imp, p, _, _ in requirements if importlib.util.find_spec(imp) is None]
    if still:
        raise RuntimeError(f"still missing after installation: {still}")
    if verbose:
        print("[requirements] every package is installed.", flush=True)
