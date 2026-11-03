"""
seeding.py -- reproducible seeding across Python's `random`, NumPy's
Python-level RNG, AND numba's internal RNG. Physically copied from
methods_comparison's `methods_comparison/core/seeding.py` (same copy used by
det_vs_stoch/dependency/resampling/seeding.py) per this framework's
"full standalone copy" convention.

Why this matters here specifically
-----------------------------------
mono_algorithms/multi_algorithms' algorithm functions (GA, NSGA2, MOCS, ...)
each seed a local `rng = np.random.default_rng(seed)` for their own
Python-level orchestration, but the problem-specific `Generate_Solution`/
`Perturb`/`Crossover` operators they call every generation are `@njit`
-decorated and call `np.random.*` from WITHIN compiled code -- numba keeps
an RNG state that is *separate* from NumPy's Python-level state, so a bare
`np.random.default_rng(seed)`/`np.random.seed(seed)` call does NOT make
those `@njit` functions reproducible. Confirmed by an actual determinism
check: two `run_problem()` calls processing the SAME (instance, algorithm,
seed) tasks under different execution orders (sequential vs. parallel
dispatch, which changes what numba RNG state happens to already exist in
the process at call time) produced different logged training records and
therefore different trained models. `seed_everything` seeds both.
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
