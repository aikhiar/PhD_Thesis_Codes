"""
progress.py  —  lightweight progress reporting, no hard dependency.

Uses `tqdm` if it happens to be installed (nicer, live-updating bar);
otherwise falls back to plain interval-based print statements. Either way,
progress is shown without requiring an extra dependency.
"""

import sys

try:
    from tqdm import tqdm as _tqdm
    _TQDM_OK = True
except Exception:                        # tqdm not installed
    _TQDM_OK = False


def progress_iter(iterable, total=None, desc="", verbose=True, leave=False):
    """
    Wrap `iterable` with a progress indicator.

    verbose=False returns a plain iterator (no output at all) — used when
    the runner is constructed with verbose=False.
    """
    if not verbose:
        return iter(iterable)
    if _TQDM_OK:
        return _tqdm(iterable, total=total, desc=desc, leave=leave, file=sys.stdout)
    return _PlainProgress(iterable, total=total, desc=desc)


class _PlainProgress:
    """Minimal fallback used when tqdm isn't installed: one print per item."""

    def __init__(self, iterable, total=None, desc=""):
        self._it = iter(iterable)
        self._total = total
        self._desc = desc
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self._it)          # StopIteration propagates naturally
        self._i += 1
        tot = f"/{self._total}" if self._total else ""
        print(f"{self._desc}: {self._i}{tot}", flush=True)
        return item
