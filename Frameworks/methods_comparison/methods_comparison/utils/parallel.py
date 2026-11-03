"""parallel.py  —  number of worker processes.

config["parallel"]["n_jobs"]:
    "auto"   physical cores of the machine running the study (psutil; logical
             cores / 2 without it) -- the default to prefer when execution times
             are measured, since two workers sharing a hyper-threaded core slow
             each other down;
    -1       every logical core (joblib convention);
    k >= 1   exactly k processes (1 or None: sequential path).
"""
import os


def physical_cores() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except ImportError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def resolve_n_jobs(value):
    if isinstance(value, str) and value.lower() == "auto":
        return physical_cores()
    return value
