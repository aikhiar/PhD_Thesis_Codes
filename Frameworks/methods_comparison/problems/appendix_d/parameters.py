"""
parameters.py -- every PROBLEM-SPECIFIC parameter of the Appendix D study
(problems/problem_appendix_d.py), which defines no value of its own.

Everything that is general to methods_comparison (methods, population size, generations,
solver parameters and tuning grids, runs, tuning protocol, parallelism,
statistics, figures) is set in comparison_config.py, in the main folder.
This folder is the model; the study of methods_comparison is problems/problem_appendix_d.py.

Sections
    1. Time discretisation and bookings
    2. Charging infrastructure
    3. Instances: ACN and simulated
    4. Problem-level parameters of the operators
"""

# ══════════════════════════════════════════════════════════════════════════════
# 1. TIME DISCRETISATION AND BOOKINGS
# ══════════════════════════════════════════════════════════════════════════════
TAU = 0.1                    # slot length (h), as in Golabi et al. (2025) and Khiar et al. (SNCS)
BOOKING_VALUE = 1.0          # v_j for every booking: the objective counts accepted bookings
# Acceptance limit P_j of each vehicle (kW): the largest power it accepts, drawn among the
# values below with the given shares; None = no limit (effective levels = charger levels).
# The three values mirror the three charger types (a design choice, not a fleet statistic).
# Limits below 11 kW would make many simulated bookings unservable: their departure times
# are built so that 11 kW is just sufficient.
ACCEPTANCE_LIMITS_KW = (11.0, 22.0, None)
ACCEPTANCE_LIMIT_SHARES = (1 / 3, 1 / 3, 1 / 3)

# ══════════════════════════════════════════════════════════════════════════════
# 2. CHARGING INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
# Charger types: power set-points (kW) of each type. Multi-level configuration of
# Golabi et al. (2025): {11}, {11, 22}, {11, 22, 43}.
LEVEL_SETS = ((11.0,), (11.0, 22.0), (11.0, 22.0, 43.0))
TYPE_SHARES = (1 / 3, 1 / 3, 1 / 3)  # share of the chargers of each type
CHARGERS_PER_PEAK_VEHICLE = 0.9       # m = ceil(this * largest number of vehicles present at once)
# Nominal grid capacity W_G = GRID_FRACTION[source] * peak average power, i.e. the largest
# sum over a slot of e_j / (tau |S_j|) for the vehicles present (power needed if each spread
# its energy evenly over its stay). ACN vehicles stay long and book little energy, so that a
# fraction of 0.7 leaves their grid non-binding (every method then rejects exactly the
# bookings that the number of chargers forces to reject); 0.5 makes both resources bind.
GRID_FRACTION = {"ACN": 0.5, "SIM": 0.7}
# Both rules count only the bookings that some charger type can serve on its own
# (a booking without compatible type is rejected whatever the schedule).
# Time-varying grid: W_t = W_G * (1 - GRID_VARIATION * R_h), with one draw R_h ~ U(0, 1) per
# period of GRID_PERIOD_H hours (a contracted profile, or other loads on the connection): the
# capacity lies between (1 - GRID_VARIATION) W_G and W_G. GRID_VARIATION = 0: constant grid.
GRID_VARIATION = 0.3
GRID_PERIOD_H = 1.0

# ══════════════════════════════════════════════════════════════════════════════
# 3. INSTANCES: 40 = 20 ACN + 20 SIMULATED
# ══════════════════════════════════════════════════════════════════════════════
SIZES = (50, 100, 150, 200)  # numbers of bookings n
INSTANCES_PER_SIZE = 5       # per source and size
ACN_SEED_BASE = 42           # instance k of a source uses seed SEED_BASE + k
SIM_SEED_BASE = 1042
ACN_DATA_FILE = "acn_sessions.pkl"   # this folder: ACN-Data claimed sessions
                                     # (arrival h, departure h, energy kWh)
ACN_KDE_BATCH = 10           # KDE draws per matching round (Khiar et al., SNCS)
ACN_MAX_ATTEMPTS_PER_BOOKING = 200
# Simulated requests: the protocol of Zaidi et al. and Golabi et al. (2025, Section 6)
SIM_ENERGY_RANGE = (5.5, 66.0)       # e_j ~ U[5.5, 66] kWh
SIM_ARRIVAL_SPAN_PER_BOOKING = 0.2   # arrivals ~ U[0, 0.2 n] h
SIM_REFERENCE_POWER = 11.0           # p_1j = e_j / 11 (h)
SIM_ALPHA = (((0.5, 1.0), (0.1, 1.0)), ((1.0, 2.0), (0.1, 0.9)), ((2.0, 3.0), (0.1, 0.8)),
             ((3.0, 4.0), (0.1, 0.7)), ((4.0, 5.0), (0.1, 0.6)), ((5.0, 6.0), (0.1, 0.5)))
SIM_ALPHA_DEFAULT = (0.1, 0.5)       # d_j = r_j + (1 + alpha) p_1j, alpha ~ U[range of p_1j]

# ══════════════════════════════════════════════════════════════════════════════
# 4. PROBLEM-LEVEL PARAMETERS OF THE OPERATORS
# ══════════════════════════════════════════════════════════════════════════════
# Level rule of the stochastic insertion: in each visited slot of the stay, the level
# drawn is the largest one that fits the residual grid ("max") or a random one among
# those that fit ("random").
LEVEL_RULE = "max"
# pc, the perturbation intensity (share of the bookings removed and re-inserted by
# Perturb): tuned for every method with the grid below (Khiar et al., SNCS Table 2),
# merged by methods_comparison with the solver grids of comparison_config.py
PROBLEM_PARAM_GRID = {"pc": [0.01, 0.03, 0.05, 0.10]}
UNTUNED_PC = 0.05            # value used when pc is not tuned

# ══════════════════════════════════════════════════════════════════════════════
# 4b. EXACT REFERENCE: THE UNIT-FLOW METHOD (problems/appendix_d/unit_flow.py)
# ══════════════════════════════════════════════════════════════════════════════
# True: methods_comparison solves every instance once with the exact method before the tuning (a
# few seconds per instance, all cores), and the gaps of the metaheuristics are then
# measured against the proven optimum instead of the best value found. Needs gurobipy
# with a licence (the method is problems/appendix_d/unit_flow.py); without it the
# study runs as before (gap against the best value found).
EXACT_SOLVER = True
EXACT_TIME_LIMIT = 1800.0    # s per instance
# True: the exact method is also one of the METHODS of the study, compared with the
# metaheuristics in every table, test and figure (label below). It is run like them: the
# same number of independent runs, ONE thread per run (EXACT_METHOD_THREADS), runs in
# parallel on the cores; its reported time covers the model, the search, the schedule and
# its verification. It has nothing to tune.
EXACT_AS_METHOD = True
EXACT_METHOD_THREADS = 1
EXACT_METHOD_LABEL = "UF"


# tolerance of the schedule verifiers
VERIFIER_EPS = 1e-6

# instance of methods_comparison's start-up self-check
SELF_CHECK_N = 50
SELF_CHECK_SEED = 42
