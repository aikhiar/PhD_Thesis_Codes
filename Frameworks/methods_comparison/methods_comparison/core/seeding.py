"""
seeding.py  —  reproducible, collision-free seeding.

Why this module exists
----------------------
The previous additive scheme (`base + m*1e6 + inst*1e4 + run`) overflowed its
digit bands for larger parameter grids and silently produced duplicate seeds.
Here we use ``numpy.random.SeedSequence`` with a structured key, which is
designed for exactly this (hierarchical, independent streams).

Numba caveat
------------
Numba keeps an RNG state that is *separate* from NumPy's Python-level state, so
`numpy.random.seed()` alone does NOT make `@njit` algorithms reproducible.
`seed_everything` therefore also seeds Numba's generator (when numba is
installed) by calling a tiny compiled helper.
"""

import random
import numpy as np

# ── optional numba seeding ───────────────────────────────────────────────────
_NUMBA_OK = False
try:
    from numba import njit

    @njit(cache=True)
    def _nb_seed(s):                     # seeds Numba's internal RNG state
        np.random.seed(s)

    _NUMBA_OK = True
except Exception:                        # numba not installed / failed to import
    _NUMBA_OK = False


def make_seed(seed_base: int, *keys: int) -> int:
    """
    Deterministic, independent 32-bit seed from a structured key.

    Example
    -------
    >>> make_seed(100, method_idx, inst_idx, run)
    """
    entropy = [int(seed_base)] + [int(k) for k in keys]
    ss = np.random.SeedSequence(entropy)
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def seed_everything(seed: int) -> None:
    """Seed Python `random`, NumPy, and (if available) Numba."""
    random.seed(seed)
    np.random.seed(seed)
    if _NUMBA_OK:
        try:
            _nb_seed(seed)
        except Exception:
            pass
