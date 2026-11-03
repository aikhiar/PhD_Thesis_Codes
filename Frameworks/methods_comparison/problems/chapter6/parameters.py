"""
parameters.py -- every PROBLEM-SPECIFIC parameter of the Chapter 6 study (Chapter 6:
Surrogate-Assisted Multi-Objective Scheduling of Preemptive Electric Vehicle
Charging under Joint Arrival and Energy Uncertainty).

This folder (problems/chapter6/: parameters.py, model.py, the ACN data and the
deployed surrogate models) is IDENTICAL in the three frameworks of the study:
    methods_comparison            the metaheuristics comparison (problem_chapter6.py)
    surrogate_trainer   training of the surrogates   (problem_chapter6.py)
    det_vs_stoch      deterministic vs stochastic planning (problem_chapter6.py)
so that the three experiments run on exactly the same model. Change a value
here, then copy the folder to the two other frameworks.

Everything general to a framework (methods, population, generations, solver
grids, runs, parallelism, statistics, figures) lives in that framework's own
configuration (comparison_config.py and the problem's STUDY_CONFIG for methods_comparison).

Sections
    1. Instances (ACN-Data claimed sessions)
    2. Schedule granularity
    3. Uncertainty model (joint arrival / energy error)
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
ACN_DATA_FILE = "acn_sessions.pkl"           # this folder: ACN-Data claimed sessions
                                             # (arrival h, requested departure h, requested energy kWh)
ACN_KDE_BATCH = 10                           # KDE draws per matching round
ACN_MAX_ATTEMPTS_PER_VEHICLE = 200

# Evaluation instances (metaheuristics comparison): ids 1..20, instance k (0-based)
# of the suite uses seed EVAL_SEED_BASE + k.
INSTANCES_PER_SIZE = 5
EVAL_SEED_BASE = 42
# Surrogate training pool (surrogate_trainer): distinct seeds, same session pool.
TRAINING_INSTANCES_PER_SIZE = 30
TRAINING_SEED_BASE = 9000
# Deterministic vs stochastic planning (det_vs_stoch): the first
# DET_STOCH_INSTANCES_PER_SIZE evaluation instances of each size class, here
# every evaluation instance (ids 1..20).
DET_STOCH_INSTANCES_PER_SIZE = INSTANCES_PER_SIZE

# ══════════════════════════════════════════════════════════════════════════════
# 2. SCHEDULE GRANULARITY (hours)
# ══════════════════════════════════════════════════════════════════════════════
MAX_PREEMPTIONS = 5                          # Pi^max
MIN_CHARGING_INTERVAL_H = 5 / 60             # m^(sl)
MIN_IDLE_INTERVAL_H = 5 / 60                 # m^(pl)
MIN_SEPARATION_H = 5 / 60                    # m^(d)

# ══════════════════════════════════════════════════════════════════════════════
# 3. UNCERTAINTY MODEL
# ══════════════════════════════════════════════════════════════════════════════
# kappa: Gaussian KDE fitted, per instance, on N_KDE_SAMPLES draws of
# (eps_a, |eps_e|) from a bivariate normal (sigma_a, sigma_e, rho); the draws are
# made with the instance's own seeded generator, so kappa is a deterministic
# function of the instance seed (identical in the three frameworks).
ERROR_SIGMA_ARRIVAL = 0.10
ERROR_SIGMA_ENERGY = 0.10
ERROR_CORRELATION = 0.30
N_KDE_SAMPLES = 1000
# Truncation of kappa to the admissible set D, implemented by rejection:
#   ONE_SIDED_ENERGY_ERROR = True  -> D = {1 + eps_a > 0, eps_e >= 0}: the realized
#                                     energy is never below the declared one;
#   ONE_SIDED_ENERGY_ERROR = False -> D = {1 + eps_a > 0, 1 + eps_e >= 0}.
ONE_SIDED_ENERGY_ERROR = True

# ══════════════════════════════════════════════════════════════════════════════
# 4. MONTE CARLO BUDGETS
# ══════════════════════════════════════════════════════════════════════════════
LABEL_REPLICATIONS = 30        # N_r: labels of the surrogate design set, and the
                               # objective of the stochastic regime (det_vs_stoch)
REFINEMENT_REPLICATIONS = 100  # N_r^*: re-evaluation of the front returned by the
                               # surrogate-assisted search (methods_comparison)

# ══════════════════════════════════════════════════════════════════════════════
# 5. PROBLEM-LEVEL PARAMETERS OF THE OPERATORS
# ══════════════════════════════════════════════════════════════════════════════
# perturbation intensity (share of the vehicles re-inserted by Perturb), tuned for
# every method; merged by methods_comparison with the solver grids
PROBLEM_PARAM_GRID = {"perc_perturb": [0.03, 0.05, 0.10]}
UNTUNED_PERC_PERTURB = 0.05    # value used when perc_perturb is not tuned
FREE_REGION_RATE = 1.0         # lambda of SelectFreeRegion (fixed)

# ══════════════════════════════════════════════════════════════════════════════
# 6. SURROGATE MODELS (methods_comparison; written by surrogate_trainer)
# ══════════════════════════════════════════════════════════════════════════════
SURROGATE_DIR = "surrogate_models"           # this folder
SURROGATE_FILES = {
    "Mean tardiness (min)": "mean_tardiness__min.joblib",
    "Peak load (kW)": "peak_load__kw.joblib",
}


# tolerance of the schedule verifiers
VERIFIER_EPS = 1e-8

# instance of methods_comparison's start-up self-check
SELF_CHECK_N = 50
SELF_CHECK_SEED = 42
