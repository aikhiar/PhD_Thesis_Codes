"""
General (problem-independent) configuration for the surrogate_trainer
framework -- everything ML-side that does not vary by problem: which models
are candidates, how much tuning budget each gets, how train/validation data
is split, how the deployed model is selected, how scoring cost is measured,
and how per-generation logging is filtered down to a training set.

Problem-SPECIFIC settings (which metaheuristics to run and with what fixed
parameters, the training-instance generator, the feature-extraction function)
live inside each `problems/problem_*.py` file instead -- see
`training_data_generator/generator.py`.
"""

from typing import Any, Dict

# ============================================================================
# LOGGING (how much of each generation is kept as training data)
# ============================================================================

# Mono-objective algorithms: fraction of the (fitness-sorted) population kept
# as elite records at every logged generation.
ELITE_LOG_FRACTION: float = 0.10

# Multi-objective algorithms: fraction of each logged generation's
# non-dominated front kept as records, evenly spread along the front.
PARETO_LOG_FRACTION: float = 0.10

# Only every LOG_GENERATION_STRIDE-th generation is logged (the initial and
# the final fronts always are). Together with the fractions above this bounds
# the size of the design set.
LOG_GENERATION_STRIDE: int = 5

# Parallelism of the (instance, algorithm) data-generation tasks (joblib
# processes); -1 = every logical core, 1 = sequential with live progress.
DATAGEN_N_JOBS: int = -1

# ============================================================================
# TRAIN / VALIDATION SPLIT -- BY INSTANCE
# ============================================================================
# The schedules logged on one training instance are strongly correlated (the
# same run logs near-identical fronts generation after generation). Splitting
# row by row would put schedules of the same instance on both sides of the
# split and overstate the generalisation accuracy. The split is therefore made
# BY INSTANCE: `validation_fraction` of the training instances of every size
# class (stratified) go to validation with ALL their schedules, the others to
# training; the k-fold cross-validation of the hyperparameter search uses
# folds of whole instances too (stratified by size class).
TRAIN_VAL_SPLIT: Dict[str, Any] = {
    "validation_fraction": 0.30,
    "random_state": 42,
    "group_by_instance": True,     # False restores the row-wise split (not recommended)
    "stratify_by_class": True,
}

# ============================================================================
# MODEL SELECTION / TUNING
# ============================================================================

CANDIDATE_MODELS = [
    "random_forest",
    "xgboost", "catboost", "hist_gradient_boosting",
    "gaussian_process", "svr", "rbf",
]

# Cross-validation folds (of whole instances) of the hyperparameter search.
CV_FOLDS: int = 5

# Randomized-search budget per tunable model (parameter combinations drawn).
SEARCH_N_ITER: int = 20

# sklearn scoring string of the cross-validation.
SCORING: str = "neg_root_mean_squared_error"

RANDOM_STATE: int = 42

# Gaussian process, SVR and kernel ridge (RBF) scale quadratically to cubically
# with the number of rows: they are tuned and fitted on a random subsample of at
# most SLOW_MODEL_TRAIN_CAP training rows (the validation split is never
# subsampled, so every candidate is scored on the same validation rows).
SLOW_MODEL_TRAIN_CAP: int = 1000

# n_jobs of cross_val_score when a model is tuned outside trainer.py (trainer.py
# forces 1 internally to avoid nested parallelism).
N_JOBS: int = -1

# Parallelism ACROSS candidate models (one joblib process per model); 1 =
# sequential with per-draw live progress.
TRAIN_N_JOBS: int = -1

# ============================================================================
# DEPLOYED-MODEL SELECTION -- TWO-STAGE RULE, FIXED BEFORE ANY RESULT
# ============================================================================
# Stage 1 (accuracy): the candidates whose held-out accuracy is practically
#   equivalent to the most accurate one (lowest validation RMSE): Cliff's delta
#   between their absolute validation errors and the best one's, computed on the
#   per-validation-instance mean absolute errors, at most `max_abs_cliff_delta`
#   in absolute value (0.33 = at most a "small" effect, Romano et al. 2006).
#   The most accurate candidate always qualifies.
# Stage 2 (cost): among them, the one with the lowest end-to-end scoring cost
#   per schedule (feature extraction + prediction), averaged over the size
#   classes of the execution-time benchmark, is deployed.
SELECTION: Dict[str, Any] = {
    "max_abs_cliff_delta": 0.33,
}

# ============================================================================
# EXECUTION-TIME BENCHMARK (per-schedule scoring cost)
# ============================================================================
# Measured on `schedules_per_class` random schedules of the first training
# instance of every size class, after `warmup_calls` untimed calls of every
# timed function (the Numba compilation is never timed). Reported separately:
# feature extraction, prediction of each model (one objective), the end-to-end
# surrogate evaluation, and the real Monte Carlo objective with the problem's
# label replication count (N_r).
BENCHMARK: Dict[str, Any] = {
    "schedules_per_class": 5,
    "warmup_calls": 3,
    "predict_repeats": 200,
    "feature_repeats": 100,
    "real_repeats": 10,
}

# ============================================================================
# REPORTS OF THE SURROGATE'S QUALITY FOR THE OPTIMIZER
# ============================================================================
# ranking_max_rows: at most this many validation records per instance enter the
#   within-instance ranking and dominance analysis (random subsample, fixed seed).
QUALITY_REPORT: Dict[str, Any] = {
    "ranking_max_rows": 1500,
}

ML_CONFIG: Dict[str, Any] = {
    "elite_log_fraction": ELITE_LOG_FRACTION,
    "pareto_log_fraction": PARETO_LOG_FRACTION,
    "log_generation_stride": LOG_GENERATION_STRIDE,
    "datagen_n_jobs": DATAGEN_N_JOBS,
    "train_val_split": TRAIN_VAL_SPLIT,
    "candidate_models": CANDIDATE_MODELS,
    "cv_folds": CV_FOLDS,
    "search_n_iter": SEARCH_N_ITER,
    "scoring": SCORING,
    "random_state": RANDOM_STATE,
    "slow_model_train_cap": SLOW_MODEL_TRAIN_CAP,
    "n_jobs": N_JOBS,
    "train_n_jobs": TRAIN_N_JOBS,
    "selection": SELECTION,
    "benchmark": BENCHMARK,
    "quality_report": QUALITY_REPORT,
}
