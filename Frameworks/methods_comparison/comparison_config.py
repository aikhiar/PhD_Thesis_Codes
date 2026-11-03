"""
comparison_config.py  --  the protocol of the metaheuristics comparisons of the thesis.

    STUDIES  the eight studies, in the order of the manuscript
    RUN      what run_all.bat runs (which studies, per-study overrides)
    CONFIG   the protocol shared by every study: instances, tuning, evaluation,
             parallelism, statistics, report, figures, and the solvers
             (methods, population, generations, parameters, tuning grids)

Each study has one binding, problems/problem_<study>.py, with its instances, its
operators and a dict `STUDY_CONFIG` holding the settings in which the study
departs from CONFIG (population, generations, grids, and the statistics its
chapter reports). Precedence, for each study:
    CONFIG  <  STUDY_CONFIG of the binding  <  RUN["overrides"][study]

Run every study:   run_all.bat     (or: python -m methods_comparison.run_all)
"""

# ══════════════════════════════════════════════════════════════════════════════
# STUDIES: the comparisons of the thesis (binding: problems/problem_<study>.py)
# ══════════════════════════════════════════════════════════════════════════════
STUDIES = {
    "chapter2":   "Chapter 2, stochastic vehicle arrivals",
    "chapter3":   "Chapter 3, stochastic durations (first approach)",
    "chapter4":   "Chapter 4, stochastic durations (second approach)",
    "chapter5":   "Chapter 5, surrogate-assisted search, preemptive stochastic durations",
    "chapter6":   "Chapter 6, surrogate-assisted search, joint arrival and energy uncertainty",
    "appendix_b": "Appendix B, constant-power model",
    "appendix_c": "Appendix C, bi-objective deterministic model",
    "appendix_d": "Appendix D, heterogeneous discrete-level chargers",
}

# ══════════════════════════════════════════════════════════════════════════════
# RUN: what run_all.bat does
# ══════════════════════════════════════════════════════════════════════════════
RUN = {
    "output_dir": "experiments",
    # None = every study of STUDIES; or a list of study names, e.g. ["chapter2"].
    # Chapters 5 and 6 need their surrogates trained and deployed first (see the README of the repository).
    "only":       None,
    # per-study overrides of CONFIG, e.g. {"chapter6": {"evaluation": {"n_runs_per_instance": 10}}}
    "overrides":  {},
}

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG: the protocol of every study
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {

    # ── instances ────────────────────────────────────────────────────────────
    # Generated ONCE with a fixed seed and saved to instances.pkl; a re-run of
    # the same study reuses the same experiment folder and reloads them, so
    # cached runs always match the instances they were computed on.
    "instances": {
        "seed_base":      7,
        "reuse_existing": True,
    },

    # ── parameter tuning: grid search, one parameter set per instance ────────
    "tuning": {
        "enabled":          True,
        "n_runs_per_combo": 5,        # runs of each parameter combination
        "aggregation":      "mean",   # "mean" | "median" | "best"
        "seed_base":        42,
        "timeout":          None,     # s per solver call (None = no limit)
        # parameters used INSTEAD of tuning, per method (Appendix D: the exact method):
        #   None | {method: {param: value}} | {method: {"1": {...}, ...}} (per instance)
        "fixed_params":     None,
    },

    # ── evaluation (the runs that produce the reported results) ──────────────
    "evaluation": {
        "n_runs_per_instance": 30,
        # one extra, discarded run per (method, instance) absorbs the compilation
        # (numba) time, so that the kept runs' times are fair
        "warmup":              True,
        "seed_base":           100,
        "timeout":             None,  # s per run (None = no limit)
        # keep each run's solution in the results (False = objective values only)
        "save_solutions":      True,
    },

    # ── convergence: recorded during the evaluation runs themselves ──────────
    "convergence": {
        "enabled": True,
    },

    # ── multi-objective indicators (IGD+, additive epsilon, spacing) ─────────
    "indicators": {
        "normalise":    True,         # same ideal and nadir points as the hypervolume
        "spacing_ddof": 1,            # Schott (1995)
    },

    # ── parallelism ──────────────────────────────────────────────────────────
    "parallel": {
        # "auto" = physical cores of the machine (detected at run time),
        # -1 = all logical cores, k = k processes, 1 = sequential
        "n_jobs":       "auto",
        # how timeouts are enforced: "alarm" (POSIX only) | "process" (everywhere)
        "timeout_mode": "alarm",
    },

    # ── start-up check of every study ────────────────────────────────────────
    "validation": {
        "run_self_check":              True,   # problem_dict["self_check"]() at start-up
        "abort_on_self_check_failure": True,
    },

    # ── statistical tests (each study keeps the tests its chapter reports) ───
    # The unit of analysis is the instance: each method is summarised by its
    # mean over the runs of every instance.
    "statistics": {
        "alpha":               0.05,
        "tested_blocks":       ["objective"],  # "objective" (hypervolume or objective) and/or "time"
        "friedman":            True,           # Friedman test, average ranks, Kendall's W
        "nemenyi":             False,          # critical difference and its diagram
        "wilcoxon":            True,           # pairwise Wilcoxon signed-rank tests, across instances
        "pairwise_correction": "holm",         # "holm" | "bonferroni"
        "cliff_delta":         True,           # Cliff's delta of every pair
        "mann_whitney":        False,          # per instance, two-sided Mann-Whitney U test against the control
        "holm":                False,          # Friedman post-hoc and Wilcoxon against the control (Holm), A12
        "bayesian":            False,          # Bayesian signed-rank test against the control
        "bayes_rope":          0.01,
        "bayes_samples":       20000,
    },

    # ── report ───────────────────────────────────────────────────────────────
    "report": {
        "control":           None,     # None = the method with the best average rank
        "summary":           False,    # summary table of each block (Appendix D)
        "tidy_csv":          False,    # one row per run (Appendix D: read by AnnexeD/complements.py)
        "decimals":          2,
        "bootstrap_samples": 2000,
        "marker_alpha":      0.05,
    },

    # ── anytime analysis (mono-objective studies with an exact reference) ────
    "anytime": {
        "enabled": False,
        "targets": [0.0],             # gap targets (%) of the time-to-target curves
    },

    # ── scaling of a block with an instance feature ──────────────────────────
    "scaling": {
        "enabled":  False,
        "features": ["n"],
        "blocks":   ["time"],         # "objective" and/or "time"
    },

    # ── figures ──────────────────────────────────────────────────────────────
    "figures": {
        "formats":         ["png", "svg"],
        "dpi":             300,
        "figsize_default": (8, 5),
        "figsize_wide":    (10, 5),
        "figsize_square":  (6, 6),
        "palette": [                  # Wong (2011) colour-blind-safe palette
            "#0072B2", "#E69F00", "#009E73",
            "#CC79A7", "#56B4E9", "#D55E00",
            "#F0E442", "#000000",
        ],
    },

    # ── solvers (methods_comparison/solvers) ─────────────────────────────────
    # The methods of every study, built from the operators of its binding. For
    # each algorithm, "params" are the values used when the parameter is not
    # tuned, and "grid" the values tried by tuning (merged with the study's own
    # problem-level grid).
    "solvers": {
        "mono_methods":    ["GA", "ES", "PSO", "GWO", "CS"],
        "multi_methods":   ["NSGA2", "NSGA3", "MOCS", "MOPSO", "MOGWO"],
        "population_size": 100,
        "generations":     500,       # Appendix D; the other studies set theirs in STUDY_CONFIG
        "algorithms": {
            # mono-objective
            "GA":     {"label": "GA",
                       "params": {"crossover_pool_size": 2, "mutation_rate": 0.15},
                       "grid":   {"mutation_rate": [0.01, 0.05, 0.15, 0.20]}},
            "ES":     {"label": "ES",
                       "params": {"lambda_ratio": 1.0, "elite_fraction": 0.5,
                                  "crossover_pool_size": 2, "mutation_rate": 0.15},
                       "grid":   {"mutation_rate": [0.01, 0.05, 0.15, 0.20]}},
            "PSO":    {"label": "PSO",
                       "params": {"archive_size": 100, "mutation_rate": 0.15},
                       "grid":   {"mutation_rate": [0.01, 0.05, 0.15, 0.20]}},
            "GWO":    {"label": "GWO",
                       "params": {"mutation_rate": 0.15},
                       "grid":   {"mutation_rate": [0.01, 0.05, 0.15, 0.20]}},
            "CS":     {"label": "CSA",
                       "params": {"p_abandon": 0.25, "elite_fraction": 1 / 3},
                       "grid":   {"p_abandon": [0.10, 0.20, 0.25, 0.30]}},
            # multi-objective
            "NSGA2":  {"label": "NSGA-II",
                       "params": {"crossover_pool_size": 2, "mutation_rate": 0.10},
                       "grid":   {"mutation_rate": [0.05, 0.10, 0.15]}},
            "NSGA3":  {"label": "NSGA-III",
                       "params": {"n_divisions": 6, "crossover_pool_size": 2, "mutation_rate": 0.10},
                       "grid":   {"n_divisions": [4, 6, 8], "mutation_rate": [0.05, 0.10, 0.15]}},
            "MOCS":   {"label": "MOCS",
                       "params": {"p_abandon": 0.25, "elite_fraction": 0.3},
                       "grid":   {"elite_fraction": [0.1, 0.2, 0.3]}},
            "MOPSO":  {"label": "MOPSO",
                       "params": {"archive_size": 60, "mutation_rate": 0.10, "n_grid": 10, "beta": 1.0},
                       "grid":   {"archive_size": [50, 60, 70], "mutation_rate": [0.05, 0.10, 0.15]}},
            "MOGWO":  {"label": "MOGWO",
                       "params": {"archive_size": 60, "mutation_rate": 0.10, "n_grid": 10, "beta": 1.0},
                       "grid":   {"mutation_rate": [0.05, 0.10, 0.15]}},
        },
    },
}
