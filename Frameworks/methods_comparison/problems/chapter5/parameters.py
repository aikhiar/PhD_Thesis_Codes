"""
parameters.py -- every PROBLEM-SPECIFIC parameter of the Chapter 5 study (Chapter 5:
Surrogate-assisted multi-objective optimization for preemptive electric vehicle
charging scheduling under stochastic charging durations; tri-objective).

This folder (problems/chapter5/: parameters.py, model.py, the ACN data and the
deployed surrogate models) is IDENTICAL in the two frameworks of the study:
    methods_comparison            the metaheuristics comparison (problem_chapter5.py)
    surrogate_trainer   training of the surrogates   (problem_chapter5.py)
so that both experiments run on exactly the same model. Change a value
here, then copy the folder to the other framework.

Sections
    1. Instances (ACN-Data claimed sessions)
    2. Schedule granularity and delivered energy
    3. Uncertainty model (stochastic charging durations)
    4. Monte Carlo budgets
    5. Problem-level parameters of the operators
    6. Surrogate models
"""

# ══════════════════════════════════════════════════════════════════════════════
# 1. INSTANCES
# ══════════════════════════════════════════════════════════════════════════════
SIZES = (50, 100, 150, 200)                  # numbers of vehicles n
SIZE_LABELS = ("small", "medium", "moderate", "large")
VEHICLES_PER_CHARGER = 3                     # m = ceil(n / 3)
CHARGER_POWERS_KW = (3.3, 7.2, 7.7, 9.6, 11.0, 11.5, 19.2, 22.0, 50.0)  # drawn with replacement
GRID_FRACTION = 0.7                          # P_max = GRID_FRACTION * sum of the charger powers
ACN_DATA_FILE = "acn_sessions.pkl"           # this folder: ACN-Data claimed sessions
                                             # (arrival h, requested departure h, requested energy kWh)
ACN_KDE_BATCH = 10                           # KDE draws per matching round
ACN_MAX_ATTEMPTS_PER_VEHICLE = 200

INSTANCES_PER_SIZE = 5                       # evaluation instances (ids 1..20)
EVAL_SEED_BASE = 42                          # evaluation instance k (0-based): seed EVAL_SEED_BASE + k
TRAINING_INSTANCES_PER_SIZE = 30             # surrogate training pool
TRAINING_SEED_BASE = 9000

# ══════════════════════════════════════════════════════════════════════════════
# 2. SCHEDULE GRANULARITY (hours) AND DELIVERED ENERGY
# ══════════════════════════════════════════════════════════════════════════════
MAX_PREEMPTIONS = 5
MIN_CHARGING_INTERVAL_H = 5 / 60             # m^(sl)
MIN_IDLE_INTERVAL_H = 5 / 60                 # m^(pl)
MIN_SEPARATION_H = 5 / 60                    # m^(d)
MIN_DELIVERED_FRACTION = 0.50                # delivered energy in [0.5 e_i, e_i]

# ══════════════════════════════════════════════════════════════════════════════
# 3. UNCERTAINTY MODEL
# ══════════════════════════════════════════════════════════════════════════════
# The realized length of every charging interval is log-normal with mean the
# planned length and standard deviation VARIANCE_FACTOR x the planned length.
VARIANCE_FACTOR = 0.10

# ══════════════════════════════════════════════════════════════════════════════
# 4. MONTE CARLO BUDGETS
# ══════════════════════════════════════════════════════════════════════════════
LABEL_REPLICATIONS = 30        # N_r: labels of the surrogate design set
REFINEMENT_REPLICATIONS = 100  # N_r^*: re-evaluation of the front returned by the
                               # surrogate-assisted search (methods_comparison)

# ══════════════════════════════════════════════════════════════════════════════
# 5. PROBLEM-LEVEL PARAMETERS OF THE OPERATORS
# ══════════════════════════════════════════════════════════════════════════════
PROBLEM_PARAM_GRID = {"perc_perturb": [0.03, 0.05, 0.10]}
UNTUNED_PERC_PERTURB = 0.05
FREE_REGION_RATE = 1.0         # lambda of the free-region selection (fixed)

# ══════════════════════════════════════════════════════════════════════════════
# 6. SURROGATE MODELS (methods_comparison; written by surrogate_trainer)
# ══════════════════════════════════════════════════════════════════════════════
SURROGATE_DIR = "surrogate_models"
SURROGATE_FILES = {             # undelivered energy is deterministic: computed exactly
    "Peak load (kW)": "peak_load__kw.joblib",
    "Total tardiness (h)": "total_tardiness__h.joblib",
}


# tolerance of the schedule verifiers
VERIFIER_EPS = 1e-8

# instance of methods_comparison's start-up self-check
SELF_CHECK_N = 50
SELF_CHECK_SEED = 42
