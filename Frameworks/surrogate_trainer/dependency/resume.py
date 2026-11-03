"""
resume.py -- the task cache that lets an interrupted run resume where it
stopped (python main.py --resume runs/<run folder>).

Every expensive task (one (instance, algorithm) data-generation run, the
training of one objective) is written to <run>/cache/... as soon as it finishes, with
the settings it was run with. On resume a task is loaded instead of re-run
when its file exists AND its settings match the current ones; otherwise it is
run again. Every task is seeded from its own settings, so a resumed run gives
the same results as an uninterrupted one.

Files are written atomically (temporary file, then rename): a crash while
writing never leaves a truncated cache file behind.
"""

import os
from typing import Any, Dict, Optional

import joblib


def task_path(cache_dir: Optional[str], *parts: str) -> Optional[str]:
    if not cache_dir:
        return None
    path = os.path.join(cache_dir, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def load_task(path: Optional[str], settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The cached payload of a task, or None when absent, unreadable or run
    with other settings."""
    if not path or not os.path.exists(path):
        return None
    try:
        cached = joblib.load(path)
    except Exception:
        return None
    if cached.get("settings") != settings:
        return None
    return cached.get("payload")


def save_task(path: Optional[str], settings: Dict[str, Any], payload: Dict[str, Any]) -> None:
    if not path:
        return
    tmp = path + ".tmp"
    joblib.dump({"settings": settings, "payload": payload}, tmp)
    os.replace(tmp, path)
