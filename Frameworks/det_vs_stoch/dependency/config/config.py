"""
Global configuration for the deterministic-vs-stochastic regime comparison.
Mirrors the shape of surrogate_trainer/dependency/config/config.py.
"""

RESAMPLING_CONFIG = {
    # Tuning: independent grid search per regime, single algorithm only
    # (BEST_ALGORITHM, declared per problem module) -- no cross-algorithm
    # comparison happens in this framework. If "enabled" is False, tuning is
    # skipped entirely and every regime/instance uses the fixed hyperparameters
    # returned by the problem module's default_params_for_best_algorithm()
    # (sourced from that module's own DEFAULT_* constants, defined near the
    # top of the file, right after the imports).
    "tuning": {
        "enabled": True,
        "n_runs_per_combo": 3,
        "seed_base": 1000,
    },
    # Evaluation: re-run the winning combo n_eval_runs times per regime per
    # instance; the non-dominated union of those runs (WITH schedules kept)
    # is "the" Pareto front for that (instance, regime), and every run is also
    # kept as an independent front for the run-level analysis. 10 runs make the
    # per-instance run-level test testable after Holm over 20 instances (with K
    # runs the smallest attainable p-value is 2 / C(2K, K): 0.0079 for K = 5,
    # 1.1e-5 for K = 10).
    "evaluation": {
        "n_eval_runs": 10,
        "seed_base": 2000,
        "warmup": True,
    },
    # Resampling protocol: R replications per instance, drawing one
    # realization and applying it to both regimes' schedules each time.
    "resampling": {
        "n_replications": 30,
        "seed_base": 3000,
    },
    "hypervolume": {
        "eps": 1e-3,
    },
    # Parallelism: the (seed, combo) tuning tasks, per-instance evaluation
    # tasks, and per-instance resampling tasks are each dispatched to
    # separate worker processes via joblib (numba/GIL-bound work benefits
    # from process-, not thread-, parallelism). -1 uses every logical core;
    # 1 falls back to the original sequential path with full live per-item
    # progress (parallel workers cannot stream progress back to the parent
    # process until each task completes). Tuning is by far the most
    # expensive phase (grid size x n_runs_per_combo x instances), so this is
    # the single biggest lever in this framework. Lower this on a shared
    # machine to leave headroom for other users.
    "parallel": {
        "n_jobs": -1,
    },
    # Statistics of the report: per-instance tests over the replications are
    # adjusted over the instances (the correction is the chapter's, in "report");
    # bootstrap resamples of every confidence interval.
    "statistics": {
        "alpha": 0.05,
        "bootstrap_samples": 2000,
    },
    # What a chapter reports; each problem module sets its own in REPORT
    # (Chapters 2, 3 and 4: these values; Chapter 6: everything, with Holm).
    "report": {
        "correction": "bonferroni",      # of the per-instance tests: "bonferroni" | "holm"
        "across_instances": False,       # tests on the per-instance means, per size class and overall
        "run_level": False,              # Run-level analysis/ (each evaluation run as a front)
        "promise_vs_delivery": False,    # planned vs realized hypervolume, realized objectives
        "realized_fronts": False,        # attainment surfaces of the realized fronts
    },
    "figures": {
        "formats": ["png", "svg"],
        "dpi": 400,
    },
    "random_state": 42,
}
