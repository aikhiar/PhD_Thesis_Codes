"""
seeding.py -- reproducible seeding across Python's `random`, NumPy's
Python-level RNG, AND numba's internal RNG. Physically copied from
methods_comparison's `methods_comparison/core/seeding.py` per this framework's "full standalone
copy" convention.

Why this matters here specifically
-----------------------------------
Every chapter's `realize_one_draw` is meant to be reproducible given a
fixed `seed` -- the resampling protocol's "common random numbers" design
(the SAME realization seed applied to both the deterministic-tuned and
stochastic-tuned fronts within one replication) depends on this. But
several chapters' single-draw realizers (`actualise_schedule_lognormal`,
`_realize_schedule_once_lognormal`, `_realize_schedule_once`) are `@njit`
-decorated and call `np.random.*` from WITHIN compiled code -- numba keeps
an RNG state that is *separate* from NumPy's Python-level state, so a bare
`np.random.seed(seed)` call does NOT make those `@njit` functions
reproducible (confirmed by an actual determinism check that failed before
this fix was added: two calls to chapter11's `realize_one_draw` with the
identical seed produced different results). `seed_everything` seeds both.
"""

import random
import numpy as np

_NUMBA_OK = False
try:
    from numba import njit

    @njit(cache=True)
    def _nb_seed(s):                     # seeds Numba's internal RNG state
        np.random.seed(s)

    _NUMBA_OK = True
except Exception:                        # numba not installed / failed to import
    _NUMBA_OK = False


def seed_everything(seed: int) -> None:
    """Seed Python `random`, NumPy's Python-level RNG, and (if available)
    Numba's separate internal RNG."""
    random.seed(seed)
    np.random.seed(seed)
    if _NUMBA_OK:
        try:
            _nb_seed(seed)
        except Exception:
            pass
