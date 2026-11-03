"""
parameters.py -- every parameter of Chapter 3 of the thesis (chapter9.tex of the
manuscript: the model of Khiar et al., Omega 141 (2026) 103506, multi-objective
EV charging scheduling under stochastic charging durations).

The constants of sections 2 to 4 keep the names and the values of the code of the
thesis.

Sections
    1. Instances (the ACN-Data sessions; a family of simulated requests stays available)
    2. Model constants of the chapter (granularity, delivered energy, compatibility)
    3. Problem-level parameters of the operators and the Monte Carlo budget
    4. Solver defaults of the chapter and the methods_comparison protocol
"""

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# 1. INSTANCES
# ══════════════════════════════════════════════════════════════════════════════
# The manuscript uses 20 instances derived from the ACN-Data claimed sessions, 5
# per size; m = ceil(n / 3) chargers. FAMILIES = ("ACN", "SIM") adds 20 instances
# of simulated requests; the chargers, the compatibilities and the duration
# standard deviations are drawn as in model.simulate_instance for both families.
FAMILIES = ("ACN",)
SIZES = (50, 100, 150, 200)                  # numbers of vehicles n
CLASS_LABELS = ("small", "medium", "moderate", "large")
INSTANCES_PER_SIZE = 5                       # per family and size
VEHICLES_PER_CHARGER = 3                     # m = ceil(n / 3)
ACN_SEED_BASE = 42                           # ACN instance k (0-based): seed 42 + k (the chapter's seeds)
SIM_SEED_BASE = 1042                         # simulated instance k (0-based): seed 1042 + k
ACN_DATA_FILE = "acn_sessions.pkl"           # this folder: ACN-Data claimed sessions
                                             # (arrival h, requested departure h, requested energy kWh)
CHARGER_OPENING_RANGE_H = (0.0, 8.0)         # charger j available from U[0, 8] h
STD_VALUES = (0.05, 0.20, 10)                # std of every compatible pair from linspace(0.05, 0.2, 10)
KAPPA = 1.0                                  # charging efficiency factor
# Simulated requests: the protocol of Zaidi et al. and Golabi et al. (2025,
# Section 6), as in Appendix D: e ~ U[5.5, 66] kWh, a ~ U[0, 0.2 n] h,
# d = a + (1 + alpha) e / 11 with alpha ~ U[range of e / 11].
SIM_ENERGY_RANGE = (5.5, 66.0)
SIM_ARRIVAL_SPAN_PER_VEHICLE = 0.2
SIM_REFERENCE_POWER = 11.0
SIM_ALPHA = (((0.5, 1.0), (0.1, 1.0)), ((1.0, 2.0), (0.1, 0.9)), ((2.0, 3.0), (0.1, 0.8)),
             ((3.0, 4.0), (0.1, 0.7)), ((4.0, 5.0), (0.1, 0.6)), ((5.0, 6.0), (0.1, 0.5)))
SIM_ALPHA_DEFAULT = (0.1, 0.5)

# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL CONSTANTS OF CHAPTER 9
# ══════════════════════════════════════════════════════════════════════════════
INFINITY = 123456789                         # sentinel: open-ended free window
EPS = 1e-9
DEFAULT_CHARGING_POWERS = np.array(
    [3.3, 7.2, 7.7, 9.6, 11.0, 11.5, 19.2, 22.0, 50.0], dtype=np.float64
)
DEFAULT_MAX_PREEMPTIONS = 5                  # at most 5 charging sessions per vehicle (row width 4 + 2 x 5)
DEFAULT_MIN_PL = 5 / 60                      # m^(sl): minimum charging session (h)
DEFAULT_MIN_NPL = 5 / 60                     # m^(pl): minimum pause between sessions (h)
DEFAULT_MIN_DIST = 5 / 60                    # m^(d): minimum gap between two vehicles of a charger (h)
DEFAULT_MIN_PERCENTAGE = 50.0                # delivered energy >= 50 % of e_i; each vehicle compatible
                                             # with at least 50 % of the chargers

# ══════════════════════════════════════════════════════════════════════════════
# 3. PROBLEM-LEVEL PARAMETERS AND MONTE CARLO BUDGET (the chapter)
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_LAM = 1.0                            # scale of the open-ended window extension
DEFAULT_PERC_PERTURB = 0.05                  # share of the fleet re-placed by Perturb
DEFAULT_VARIANCE_FACTOR = 0.1                # carried in problem_params, unused by the model
DEFAULT_NUM_REPLICATIONS = 30                # N_r: Monte Carlo replications of the peak load
PROBLEM_PARAM_GRID = {                       # the chapter's problem_param_grids
    "perc_perturb": [0.03, 0.05, 0.1],
    "lam": [1.0],
    "num_of_replications": [30],
}

# ══════════════════════════════════════════════════════════════════════════════
# 4. SOLVER DEFAULTS OF CHAPTER 9 AND THE METHODS_COMPARISON PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_POPULATION_SIZE = 80
DEFAULT_GENERATIONS = 150
DEFAULT_SEED = 123
DEFAULT_CROSSOVER_POOL_SIZE = 2
DEFAULT_MUTATION_RATE = 0.1
DEFAULT_N_DIVISIONS = 6
DEFAULT_P_ABANDON = 0.25
DEFAULT_ELITE_FRACTION = 0.3
DEFAULT_ARCHIVE_SIZE = 60
DEFAULT_N_GRID = 10
DEFAULT_BETA = 1.0
# the chapter's param_grids (identical to the grids of comparison_config.py, except
# MOPSO, which also tunes n_grid)
ALGORITHM_GRIDS = {
    "NSGA2": {"mutation_rate": [0.05, 0.10, 0.15]},
    "NSGA3": {"n_divisions": [4, 6, 8], "mutation_rate": [0.05, 0.10, 0.15]},
    "MOCS": {"elite_fraction": [0.1, 0.2, 0.3]},
    "MOPSO": {"archive_size": [50, 60, 70], "mutation_rate": [0.05, 0.10, 0.15], "n_grid": [8, 10, 12]},
    "MOGWO": {"mutation_rate": [0.05, 0.10, 0.15]},
}


# tolerance of the schedule verifiers
VERIFIER_EPS = 1e-8
