"""
defaults.py  --  the default parameter values of the bundled solvers, as names
the solver modules can use in their signatures. No value is defined here: they
all come from CONFIG["solvers"] in comparison_config.py (main folder), the single
place where the defaults of the comparisons are set.
"""

from methods_comparison.config import DEFAULT_CONFIG

_S = DEFAULT_CONFIG["solvers"]
_A = _S["algorithms"]

DEFAULT_POPULATION_SIZE = _S["population_size"]
DEFAULT_GENERATIONS = _S["generations"]
# multi-objective solvers
DEFAULT_CROSSOVER_POOL_SIZE = _A["NSGA2"]["params"]["crossover_pool_size"]
DEFAULT_N_DIVISIONS = _A["NSGA3"]["params"]["n_divisions"]
DEFAULT_P_ABANDON = _A["MOCS"]["params"]["p_abandon"]
DEFAULT_ELITE_FRACTION = _A["MOCS"]["params"]["elite_fraction"]
DEFAULT_ARCHIVE_SIZE = _A["MOPSO"]["params"]["archive_size"]
DEFAULT_N_GRID = _A["MOPSO"]["params"]["n_grid"]
DEFAULT_BETA = _A["MOPSO"]["params"]["beta"]
