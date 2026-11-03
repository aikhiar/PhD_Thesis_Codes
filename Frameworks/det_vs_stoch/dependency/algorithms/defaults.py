"""
defaults.py -- default parameter values of the bundled multi-objective solvers
(the values of CONFIG["solvers"] in methods_comparison's comparison_config.py). They
are only used when a caller does not pass the parameter, which this framework
never does: every run passes its parameters explicitly.
"""

DEFAULT_POPULATION_SIZE = 100
DEFAULT_GENERATIONS = 200
DEFAULT_CROSSOVER_POOL_SIZE = 2
DEFAULT_N_DIVISIONS = 6
DEFAULT_P_ABANDON = 0.25
DEFAULT_ELITE_FRACTION = 0.3
DEFAULT_ARCHIVE_SIZE = 60
DEFAULT_N_GRID = 10
DEFAULT_BETA = 1.0
