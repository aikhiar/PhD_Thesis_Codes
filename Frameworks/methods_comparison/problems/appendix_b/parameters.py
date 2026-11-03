"""
parameters.py -- every parameter of Appendix B of the thesis (mono-objective
constant-power EV charging scheduling), with the names and values of the thesis
code.

Sections
    1. Instances
    2. Model constants
    3. Problem-level parameters of the operators
    4. Protocol of the study (population, generations, tuning grids)
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# 1. INSTANCES
# ══════════════════════════════════════════════════════════════════════════════
# 20 instances derived from the ACN-Data claimed sessions, 5 per size, m =
# ceil(n / 3) chargers; instance k (0-based) has the seed INSTANCE_SEED_BASE + k.
SIZE_CLASSES = [50, 100, 150, 200]           # numbers of vehicles n
CLASS_LABELS = ["small", "medium", "moderate", "large"]
INSTANCES_PER_CLASS = 5
VEHICLES_PER_CHARGER = 3                     # m = ceil(n / 3)
INSTANCE_SEED_BASE = 42
ACN_DATA_FILE = "acn_sessions.pkl"           # this folder: ACN-Data claimed sessions

# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_TAU = 5.0 / 60.0                     # slot length (h)
DEFAULT_CHARGING_POWERS = np.array(
    [3.3, 7.2, 7.7, 9.6, 11.0, 11.5, 19.2, 22.0, 50.0], dtype=np.float64
)
INFINITY = 123456789                         # sentinel: open-ended free window

# ══════════════════════════════════════════════════════════════════════════════
# 3. PROBLEM-LEVEL PARAMETERS OF THE OPERATORS
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_PERC_PERTURB = 0.05                  # share of the fleet re-placed by Perturb
PROBLEM_PARAM_GRID = {
    "perc_perturb": [0.03, 0.05, 0.1],
}

# ══════════════════════════════════════════════════════════════════════════════
# 4. PROTOCOL OF THE STUDY
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_POPULATION_SIZE = 80
DEFAULT_GENERATIONS = 150
DEFAULT_P_ABANDON = 0.25
DEFAULT_ELITE_FRACTION = 0.3
# the tuning grids of the thesis
ALGORITHM_GRIDS = {
    "GA": {"mutation_rate": [0.05, 0.10, 0.15]},
    "PSO": {"archive_size": [50, 60, 70], "mutation_rate": [0.05, 0.10, 0.15]},
    "GWO": {"mutation_rate": [0.05, 0.10, 0.15]},
    "CS": {"p_abandon": [0.25], "elite_fraction": [0.1, 0.2, 0.3]},
}
