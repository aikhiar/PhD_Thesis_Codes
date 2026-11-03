"""
Tuning, training, and validation for the surrogate ML models. One call,
`train_and_validate`, handles a single objective: split, cheap randomized
hyperparameter search (a small, fixed budget -- no long-running tuning),
k-fold CV scoring, and held-out validation metrics, for every candidate model
in `config.CANDIDATE_MODELS`. The caller (training_data_generator / main.py)
repeats this once per objective.
"""

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import threadpoolctl
from joblib import Parallel, delayed
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    ParameterSampler,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from dependency.reporting.progress import MODEL_DISPLAY_NAMES, ProgressReporter

from catboost import CatBoostRegressor

# The GP kernel optimizer legitimately lands near a bound on some restarts
# (harmless -- L-BFGS just reports it found a local optimum there); this
# warning fires often enough across many small per-fold fits during
# RandomizedSearchCV to be pure noise rather than a signal of a real problem.
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.gaussian_process")
from xgboost import XGBRegressor

# ============================================================================
# CANDIDATE MODEL REGISTRY
# ============================================================================
# Random Forest (bagged trees, robust, little tuning, feature importance);
# XGBoost / CatBoost / HistGradientBoosting (three independent gradient-
# boosted-tree implementations -- typically the most accurate family on
# tabular data, and genuinely different enough from each other -- histogram
# binning strategy, ordered boosting -- to be worth comparing rather than
# picking one and assuming it represents the family). HistGradientBoosting
# (scikit-learn's own native, histogram-based implementation) replaces the
# LightGBM slot in this three-way GBT comparison: it uses the same
# histogram-binning strategy LightGBM does and is comparably fast, but
# ships inside scikit-learn itself -- no separate compiled DLL with its own
# threading/build-toolchain to go incompatible with a given machine, which
# is exactly the class of problem that made LightGBM unusable (see below);
# LightGBM itself was dropped from this roster: it reproducibly
# crashed the native LGBM_DatasetSetField call with a Windows access
# violation on every single tuning draw on at least one real deployment
# machine, even fully single-threaded and isolated in its own worker
# process (ruling out the nested-parallelism issue this framework already
# guards against elsewhere) -- a broken/incompatible install on that
# machine, not something fixable from this side, and burning large amounts
# of wall-clock time for zero usable results while doing so;
# Gaussian Process Regression / Kriging (gives a predictive uncertainty
# estimate alongside the point prediction, well suited to the small-to-medium
# design sets this framework targets); Support Vector Regression (handles
# nonlinear structure via the RBF kernel, also small-to-medium-data-suited);
# and a Kernel-Ridge-with-RBF-kernel model standing in for the "Radial Basis
# Function surrogate" family -- literal RBF *interpolation* fits every design
# point exactly, which overfits noise, and our labels are noisy R-replication
# Monte Carlo estimates, not exact function values; Kernel Ridge is the same
# RBF-kernel machinery with an explicit regularization term, so it behaves
# like an RBF surrogate without fitting the noise.
#
# Gaussian Process, SVR, and Kernel-Ridge-RBF all have training cost that
# grows roughly quadratically-to-cubically with the number of training rows
# and no sparse/inducing-point approximation here, so -- unlike the tree
# ensembles, which scale fine -- they are fit on a capped, subsampled slice of
# the training split (`_SLOW_MODEL_TRAIN_CAP` rows) rather than the full
# design set. The held-out validation split is never subsampled, so every
# model's reported metrics remain comparable.

_SLOW_MODELS = {"gaussian_process", "svr", "rbf"}
_NEEDS_SCALING = {"gaussian_process", "svr", "rbf"}
# Cubic-cost models (GP/SVR/RBF) are dominated by this cap, not by
# search_n_iter/cv_folds -- 1000 rows keeps tuning fast (each fit is ~27x
# cheaper than at the old 3000-row cap) while still giving GP a reasonably
# well-distributed training sample; tree models are unaffected (full split).
_SLOW_MODEL_TRAIN_CAP = 1000


def _to_native_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """`ParameterSampler` draws numeric hyperparameters as numpy scalars
    (e.g. `np.float64` from `scipy.stats.loguniform`/`uniform`), not native
    Python `float`/`int`. On some catboost/scikit-learn version pairings
    this reproducibly breaks `sklearn.base.clone()`'s post-construction
    equality check ("Cannot clone object ... as the constructor either does
    not set or modifies parameter X") for numpy-scalar-valued params --
    confirmed as a real, version-dependent incompatibility class (does not
    reproduce on every catboost/sklearn pairing, but does on at least one
    real deployment). Casting to native types before `set_params()` is
    strictly safer everywhere (every estimator accepts a native float/int
    wherever it accepts a numpy scalar) and eliminates this failure mode
    at the source rather than merely tolerating it after the fact."""
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()}


def _model_factory(name: str, random_state: int, n_jobs: int = -1):
    if name == "random_forest":
        # scikit-learn's regressor defaults to max_features=1.0 (every
        # split considers ALL features) -- unlike RandomForestClassifier's
        # "sqrt" default, and unlike Breiman's original RF algorithm, whose
        # whole decorrelation mechanism IS restricting each split to a
        # random feature subset. "sqrt" is both more faithful to the
        # algorithm this candidate is meant to represent and measurably
        # cheaper per split (confirmed: ~2.4x faster fit on a 4k-row/10-
        # feature synthetic benchmark, unrelated to n_jobs/threading).
        est = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs, max_features="sqrt")
    elif name == "xgboost":
        est = XGBRegressor(random_state=random_state, n_jobs=n_jobs, verbosity=0)
    elif name == "catboost":
        # CatBoost's own name for this is `thread_count`, not `n_jobs` --
        # easy to miss, and its default (-1, all cores) means an unset
        # thread_count silently ignores whatever oversubscription guard the
        # caller intended (e.g. the parallel-per-model worker path below,
        # which relies on every model actually respecting a 1-thread cap).
        est = CatBoostRegressor(random_state=random_state, thread_count=n_jobs,
                                verbose=False, allow_writing_files=False)
    elif name == "hist_gradient_boosting":
        # No n_jobs parameter on this estimator -- its OpenMP thread count
        # is controlled globally via threadpoolctl, not per-instance (see
        # the threadpool_limits(1) wrap around the parallel-per-model
        # worker path in _train_one_model_task, which this relies on to
        # avoid oversubscribing the machine the same way n_jobs=1 does for
        # every other tree model here).
        est = HistGradientBoostingRegressor(random_state=random_state)
    elif name == "gaussian_process":
        # WhiteKernel's lower bound is set wide (1e-8) since objectives here
        # span very different scales (e.g. a near-zero mean tardiness vs.
        # peak load ~1e1-1e2 kW) -- normalize_y rescales the target, but the
        # noise level found by the optimizer for a near-noiseless small-scale
        # objective can still sit close to a too-tight lower bound otherwise.
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1.0, (1e-8, 1e2))
        est = GaussianProcessRegressor(kernel=kernel, random_state=random_state, normalize_y=True)
    elif name == "svr":
        est = SVR(kernel="rbf")
    elif name == "rbf":
        est = KernelRidge(kernel="rbf")
    else:
        raise ValueError(f"Unknown candidate model '{name}'. Add it to trainer._model_factory.")

    if name in _NEEDS_SCALING:
        return Pipeline([("scaler", StandardScaler()), ("model", est)])
    return est


# Deliberately small, cheap-to-search spaces -- this is meant to run in
# minutes per objective, not hours. Pipeline-wrapped models (scaled ones)
# need the "model__" step prefix on every hyperparameter name.
_PARAM_DISTRIBUTIONS: Dict[str, Dict[str, Any]] = {
    "random_forest": {
        "n_estimators": randint(100, 400),
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": randint(1, 8),
        "max_features": ["sqrt", 1.0],
    },
    "xgboost": {
        "n_estimators": randint(100, 400),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.02, 0.28),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
    },
    "catboost": {
        "iterations": randint(100, 400),
        "depth": randint(3, 10),
        "learning_rate": uniform(0.02, 0.28),
        "l2_leaf_reg": loguniform(1e-1, 1e1),
    },
    "hist_gradient_boosting": {
        "max_iter": randint(100, 400),
        "max_leaf_nodes": randint(15, 63),
        "max_depth": [None, 5, 10, 20],
        "learning_rate": uniform(0.02, 0.28),
        "l2_regularization": loguniform(1e-1, 1e1),
        # This estimator's default (20) requires 20+ samples in a leaf
        # before it will split at all -- same small-training-set failure
        # mode documented for Random Forest's min_samples_leaf above (and
        # previously for LightGBM's identically-named/defaulted parameter):
        # left at 20, a small fold can silently produce single-leaf,
        # never-split trees for every boosting round.
        "min_samples_leaf": randint(2, 21),
    },
    "gaussian_process": {
        "model__alpha": loguniform(1e-10, 1e-2),
    },
    "svr": {
        "model__C": loguniform(1e-1, 1e3),
        "model__epsilon": loguniform(1e-3, 1.0),
        "model__gamma": loguniform(1e-4, 1e0),
    },
    "rbf": {
        "model__alpha": loguniform(1e-3, 1e2),
        "model__gamma": loguniform(1e-4, 1e1),
    },
}


# MODEL_DISPLAY_NAMES lives in dependency.reporting.progress (imported above)
# -- re-exported here under its old name so any existing
# `from dependency.trainer.trainer import MODEL_DISPLAY_NAMES` keeps working.


# ============================================================================
# RESULT CONTAINERS
# ============================================================================


@dataclass
class ModelResult:
    name: str
    estimator: Any
    cv_score: float  # mean CV score under ml_config["scoring"] (higher = better)
    cv_fold_scores: np.ndarray  # per-fold scores for the CHOSEN hyperparameters -- statistical-validation raw material
    val_rmse: float
    val_mae: float
    val_r2: float
    fit_time_seconds: float  # wall-clock for the full tune+fit step (RandomizedSearchCV, or direct fit when untunable)
    fit_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectiveTrainingResult:
    objective_name: str
    results: Dict[str, ModelResult]
    best_model_name: str          # most accurate candidate (lowest validation RMSE)
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    val_predictions: Dict[str, np.ndarray]
    groups_train: Optional[np.ndarray] = None   # training-instance id of every row
    groups_val: Optional[np.ndarray] = None
    classes_val: Optional[np.ndarray] = None
    classes_train: Optional[np.ndarray] = None
    train_index: Optional[np.ndarray] = None    # row indices (into the X given to train_and_validate)
    val_index: Optional[np.ndarray] = None
    split_mode: str = "row"                     # "instance" (grouped) or "row"
    deployed_model_name: Optional[str] = None   # set by the two-stage selection (main.py)
    selection: Dict[str, Any] = field(default_factory=dict)

    @property
    def best(self) -> ModelResult:
        return self.results[self.best_model_name]

    @property
    def deployed(self) -> ModelResult:
        return self.results[self.deployed_model_name or self.best_model_name]


def instance_level_errors(result: "ObjectiveTrainingResult") -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Mean absolute validation error of every candidate on every validation
    INSTANCE (the unit of the statistical tests: the schedules of one instance
    are not independent). Falls back to one block per validation row when the
    split was made row by row."""
    names = list(result.results)
    abs_err = {m: np.abs(result.y_val - result.val_predictions[m]) for m in names}
    if result.groups_val is None or result.split_mode != "instance":
        return [str(i) for i in range(len(result.y_val))], abs_err
    blocks = sorted(set(result.groups_val.tolist()))
    idx = {g: np.where(result.groups_val == g)[0] for g in blocks}
    return blocks, {m: np.array([abs_err[m][idx[g]].mean() for g in blocks]) for m in names}


def split_by_instance(groups: np.ndarray, classes: np.ndarray, validation_fraction: float,
                      random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """(train_mask, val_mask): in every class, round(validation_fraction x its
    number of instances) instances (at least one, at most all but one) go to
    validation with all their rows."""
    rng = np.random.default_rng(random_state)
    val_groups = set()
    for c in sorted(set(classes.tolist())):
        members = sorted(set(groups[classes == c].tolist()))
        rng.shuffle(members)
        k = int(round(validation_fraction * len(members)))
        k = min(max(k, 1), len(members) - 1) if len(members) >= 2 else 0
        val_groups.update(members[:k])
    val = np.array([g in val_groups for g in groups.tolist()])
    if not val.any() or val.all():
        raise ValueError("split_by_instance: need at least two training instances to split by instance")
    return ~val, val


def group_folds(groups: np.ndarray, classes: np.ndarray, n_folds: int, random_state: int):
    """Cross-validation folds of whole instances, stratified by class: a list of
    (train_idx, test_idx) usable as `cv` by scikit-learn."""
    rng = np.random.default_rng(random_state)
    fold_of, offset = {}, 0
    for c in sorted(set(classes.tolist())):
        members = sorted(set(groups[classes == c].tolist()))
        rng.shuffle(members)
        for k, g in enumerate(members):
            fold_of[g] = (k + offset) % n_folds
        offset += len(members)
    f = np.array([fold_of[g] for g in groups.tolist()])
    folds = [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(n_folds)]
    folds = [(tr, te) for tr, te in folds if len(tr) and len(te)]
    if len(folds) < 2:
        raise ValueError("group_folds: fewer than two non-empty folds of instances")
    return folds


def _cv_for(ml_config: Dict[str, Any], groups_fit: Optional[np.ndarray], classes_fit: Optional[np.ndarray]):
    """The cross-validation splitter of the hyperparameter search: folds of whole
    instances when the split is grouped, else shuffled k-fold of rows."""
    if groups_fit is not None and len(set(groups_fit.tolist())) >= 2:
        n_folds = min(ml_config["cv_folds"], len(set(groups_fit.tolist())))
        return group_folds(groups_fit, classes_fit, n_folds, ml_config["random_state"])
    return KFold(n_splits=ml_config["cv_folds"], shuffle=True, random_state=ml_config["random_state"])


# ============================================================================
# TRAIN + VALIDATE (single objective, all candidate models)
# ============================================================================


def _training_subset_for(name: str, X_train: np.ndarray, y_train: np.ndarray, random_state: int,
                         cap: int = _SLOW_MODEL_TRAIN_CAP):
    """Full training split for tree ensembles; a capped random subsample for
    the O(n^2)-O(n^3) models (`_SLOW_MODELS`), which have no sparse/inducing-
    point approximation available here. Returns (X, y, row indices)."""
    if name not in _SLOW_MODELS or len(X_train) <= cap:
        return X_train, y_train, np.arange(len(X_train))
    rng = np.random.default_rng(random_state)
    idx = np.sort(rng.choice(len(X_train), size=cap, replace=False))
    return X_train[idx], y_train[idx], idx


def _tune_one_model(
    name: str,
    base_estimator: Any,
    param_dist: Dict[str, Any],
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    cv: KFold,
    ml_config: Dict[str, Any],
    objective_name: str,
    progress: Optional[ProgressReporter],
):
    """Manual randomized search: draws `search_n_iter` hyperparameter
    combinations via `ParameterSampler` (same sampling scheme, same
    `random_state`, as `sklearn.model_selection.RandomizedSearchCV` would
    use internally) and scores each via k-fold `cross_val_score`, keeping
    the best. Written out by hand, rather than calling `RandomizedSearchCV`
    directly, specifically so each draw is a real, individually-timed
    iteration a `ProgressReporter` can report against -- true per-iteration
    elapsed/ETA, not a single opaque black-box call."""
    n_iter = ml_config["search_n_iter"]
    sampler = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=ml_config["random_state"]))

    task_key = f"tune::{objective_name}::{name}"
    if progress is not None:
        progress.start(task_key, f"  Tuning [bold]{name}[/bold] ({objective_name})", total=len(sampler))

    best_score = -np.inf
    best_params: Dict[str, Any] = {}
    best_fold_scores = None

    for i, params in enumerate(sampler):
        try:
            candidate = clone(base_estimator).set_params(**_to_native_params(params))
            fold_scores = np.asarray(
                cross_val_score(
                    candidate, X_fit, y_fit, scoring=ml_config["scoring"], cv=cv,
                    n_jobs=ml_config["n_jobs"],
                ),
                dtype=np.float64,
            )
            mean_score = float(np.mean(fold_scores))
        except Exception as e:
            # Some model families (catboost has a documented, version-
            # dependent quirk here: sklearn.base.clone() can fail to
            # round-trip certain parameter combinations, e.g. l2_leaf_reg,
            # through get_params()/__init__ -- "Cannot clone object ...").
            # Treat this ONE draw as a failure (worst possible score) and
            # keep searching the remaining draws, rather than aborting the
            # whole model over a single unlucky hyperparameter combination
            # -- most draws for a flaky-but-not-fundamentally-broken model
            # will still succeed.
            print(
                f"  [tune::{objective_name}::{name}] WARNING: draw {i + 1}/{len(sampler)} "
                f"({params}) failed and is being skipped -- {type(e).__name__}: {e}",
                flush=True,
            )
            mean_score = -np.inf
            fold_scores = None

        if mean_score > best_score:
            best_score = mean_score
            best_params = _to_native_params(params)
            best_fold_scores = fold_scores

        if progress is not None:
            progress.update(task_key, completed=i + 1)

    if progress is not None:
        progress.finish(task_key)

    # If EVERY draw failed, best_params/best_fold_scores are still {}/None
    # -- let this raise naturally (clone().fit() on default params should
    # normally succeed; if even that fails, the caller's own per-model
    # try/except skips this model entirely rather than the pipeline dying).
    best_estimator = clone(base_estimator).set_params(**best_params)
    best_estimator.fit(X_fit, y_fit)

    if best_fold_scores is None:
        # Every single draw's cross_val_score() call failed (observed for
        # real: catboost's clone()-in-CV quirk is not always intermittent --
        # on at least one machine it reproduces on 100% of draws, not just
        # some). best_estimator above still fit fine on default params (only
        # the internal clone() inside cross_val_score is broken, not fit()
        # itself), so this model is NOT dropped from results -- but
        # cv_fold_scores must never be None going forward: reporting.py
        # unconditionally calls np.std()/subtracts/iterates over it for
        # every model in results. Try one direct cross_val_score on the
        # final default-params estimator to recover real fold scores;
        # if that ALSO fails the same way, fall back to an all-NaN array
        # (same fold count) so downstream CSV/JSON/stats code degrades
        # gracefully (NaN prints/serializes fine) instead of crashing.
        try:
            best_fold_scores = np.asarray(
                cross_val_score(
                    clone(best_estimator), X_fit, y_fit, scoring=ml_config["scoring"], cv=cv,
                    n_jobs=ml_config["n_jobs"],
                ),
                dtype=np.float64,
            )
            best_score = float(np.mean(best_fold_scores))
        except Exception as e:
            print(
                f"  [tune::{objective_name}::{name}] WARNING: all {len(sampler)} draws AND the "
                f"final default-params CV scoring failed -- keeping the fitted model with "
                f"NaN cv_fold_scores -- {type(e).__name__}: {e}",
                flush=True,
            )
            n_folds = getattr(cv, "n_splits", ml_config.get("cv_folds", 5))
            best_fold_scores = np.full(n_folds, np.nan, dtype=np.float64)
            best_score = float("nan")

    return best_estimator, best_score, best_fold_scores, best_params


def _fit_one_model(
    name: str, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    ml_config: Dict[str, Any], model_n_jobs: int, groups_train: Optional[np.ndarray],
    classes_train: Optional[np.ndarray], objective_name: str, progress: Optional[ProgressReporter],
) -> Tuple["ModelResult", np.ndarray]:
    """Tune (randomized search, CV of whole instances when grouped), fit and
    validate ONE candidate. Shared by the sequential and parallel paths."""
    base_estimator = _model_factory(name, ml_config["random_state"], n_jobs=model_n_jobs)
    param_dist = _PARAM_DISTRIBUTIONS.get(name, {})
    cap = int(ml_config.get("slow_model_train_cap", _SLOW_MODEL_TRAIN_CAP))
    X_fit, y_fit, idx = _training_subset_for(name, X_train, y_train, ml_config["random_state"], cap)
    groups_fit = groups_train[idx] if groups_train is not None else None
    classes_fit = classes_train[idx] if classes_train is not None else None
    cv = _cv_for(ml_config, groups_fit, classes_fit)
    no_fold_parallel_config = dict(ml_config, n_jobs=1)

    t_fit_start = time.perf_counter()
    if param_dist:
        best_estimator, cv_score, cv_fold_scores, fit_params = _tune_one_model(
            name, base_estimator, param_dist, X_fit, y_fit, cv, no_fold_parallel_config, objective_name, progress,
        )
    else:
        cv_fold_scores = np.asarray(cross_val_score(
            base_estimator, X_fit, y_fit, scoring=ml_config["scoring"], cv=cv, n_jobs=1,
        ), dtype=np.float64)
        cv_score = float(np.mean(cv_fold_scores))
        best_estimator = clone(base_estimator)
        best_estimator.fit(X_fit, y_fit)
        fit_params = {}
    fit_time_seconds = time.perf_counter() - t_fit_start
    y_pred = best_estimator.predict(X_val)
    result = ModelResult(
        name=name,
        estimator=best_estimator,
        cv_score=cv_score,
        fit_time_seconds=fit_time_seconds,
        cv_fold_scores=cv_fold_scores,
        val_rmse=float(np.sqrt(mean_squared_error(y_val, y_pred))),
        val_mae=float(mean_absolute_error(y_val, y_pred)),
        val_r2=float(r2_score(y_val, y_pred)),
        fit_params=fit_params,
    )
    return result, y_pred


def _train_one_model_task(
    name: str, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    ml_config: Dict[str, Any], per_worker_n_jobs: int = 1,
    groups_train: Optional[np.ndarray] = None, classes_train: Optional[np.ndarray] = None,
) -> Tuple[str, "ModelResult", np.ndarray]:
    """Process-safe tune+fit+validate of ONE candidate -- the unit of work of
    train_and_validate's parallel path. Runs under
    threadpool_limits(per_worker_n_jobs) so that no model claims more than its
    share of the machine (see train_and_validate)."""
    try:
        with threadpoolctl.threadpool_limits(limits=per_worker_n_jobs):
            result, y_pred = _fit_one_model(name, X_train, y_train, X_val, y_val, ml_config,
                                            per_worker_n_jobs, groups_train, classes_train, "", None)
    except Exception as e:
        # a single flaky/crashing candidate must not take down the whole dispatch
        print(f"  [train] WARNING: model '{name}' failed and is being skipped -- {type(e).__name__}: {e}",
              flush=True)
        return name, None, None
    return name, result, y_pred


def train_and_validate(
    X: np.ndarray,
    y: np.ndarray,
    objective_name: str,
    ml_config: Dict[str, Any],
    progress: Optional[ProgressReporter] = None,
    groups: Optional[np.ndarray] = None,
    classes: Optional[np.ndarray] = None,
) -> ObjectiveTrainingResult:
    """Splits (X, y) into train/validation per `ml_config["train_val_split"]` --
    BY INSTANCE when `groups` (training-instance id of every row) are given and
    `group_by_instance` is set, stratified by `classes` -- then for every model
    in `ml_config["candidate_models"]`: tunes it by a randomized search with a
    k-fold cross-validation (folds of whole instances when grouped), and
    evaluates the tuned estimator on the held-out validation split (RMSE, MAE,
    R^2). The most accurate model (lowest validation RMSE) is exposed as `.best`;
    the deployed model is chosen afterwards by the two-stage rule (main.py).
    Models in `_SLOW_MODELS` are tuned/fit on a capped subsample of the training
    split; the validation split is never subsampled."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    split_cfg = ml_config["train_val_split"]
    grouped = groups is not None and split_cfg.get("group_by_instance", True)

    if grouped:
        groups = np.asarray(groups)
        classes = np.asarray(classes) if (classes is not None and split_cfg.get("stratify_by_class", True)) \
            else np.zeros(len(groups), dtype=int)
        train_mask, val_mask = split_by_instance(groups, classes, split_cfg["validation_fraction"],
                                                 split_cfg["random_state"])
        X_train, X_val, y_train, y_val = X[train_mask], X[val_mask], y[train_mask], y[val_mask]
        groups_train, groups_val = groups[train_mask], groups[val_mask]
        classes_train, classes_val = classes[train_mask], classes[val_mask]
        idx_train, idx_val = np.where(train_mask)[0], np.where(val_mask)[0]
    else:
        idx_train, idx_val = train_test_split(
            np.arange(len(y)), test_size=split_cfg["validation_fraction"],
            random_state=split_cfg["random_state"], shuffle=True,
        )
        X_train, X_val, y_train, y_val = X[idx_train], X[idx_val], y[idx_train], y[idx_val]
        groups_train = groups_val = classes_train = classes_val = None

    results: Dict[str, ModelResult] = {}
    val_predictions: Dict[str, np.ndarray] = {}

    models_key = f"train::{objective_name}"
    if progress is not None:
        progress.start(models_key, f"Training models for [bold]{objective_name}[/bold]",
                       total=len(ml_config["candidate_models"]))

    train_n_jobs = ml_config.get("train_n_jobs", 1)
    if train_n_jobs in (1, None):
        # one model at a time: the model may use every core (n_jobs=-1), the CV
        # folds run sequentially (no nested parallelism)
        for name in ml_config["candidate_models"]:
            try:
                result, y_pred = _fit_one_model(name, X_train, y_train, X_val, y_val, ml_config, -1,
                                                groups_train, classes_train, objective_name, progress)
            except Exception as e:
                print(f"  [train::{objective_name}] WARNING: model '{name}' failed and is being "
                      f"skipped -- {type(e).__name__}: {e}", flush=True)
                if progress is not None:
                    progress.update(models_key, advance=1)
                continue
            results[name] = result
            val_predictions[name] = y_pred
            if progress is not None:
                progress.update(models_key, advance=1)
    else:
        # one worker process per candidate; the cores of the machine are shared
        # between the workers (os.cpu_count() // number of candidates threads each)
        n_models = len(ml_config["candidate_models"])
        per_worker_n_jobs = max(1, (os.cpu_count() or 1) // n_models)
        task_iter = (
            delayed(_train_one_model_task)(name, X_train, y_train, X_val, y_val, ml_config, per_worker_n_jobs,
                                           groups_train, classes_train)
            for name in ml_config["candidate_models"]
        )
        parallel_kwargs = dict(n_jobs=train_n_jobs, backend="loky")
        try:
            result_iter = Parallel(return_as="generator", **parallel_kwargs)(task_iter)
            incremental = True
        except TypeError:
            result_iter = Parallel(verbose=(10 if progress is None else 0), **parallel_kwargs)(task_iter)
            incremental = False
        for name, result, y_pred in result_iter:
            if result is not None:
                results[name] = result
                val_predictions[name] = y_pred
            if incremental and progress is not None:
                progress.update(models_key, advance=1)
        if progress is not None and not incremental:
            progress.update(models_key, completed=len(ml_config["candidate_models"]))

    if progress is not None:
        progress.finish(models_key)

    if not results:
        raise RuntimeError(
            f"train_and_validate('{objective_name}'): every candidate model in "
            f"{ml_config['candidate_models']} failed -- see the WARNING lines above."
        )
    best_model_name = min(results, key=lambda n: results[n].val_rmse)

    return ObjectiveTrainingResult(
        objective_name=objective_name,
        results=results,
        best_model_name=best_model_name,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        val_predictions=val_predictions,
        groups_train=groups_train,
        groups_val=groups_val,
        classes_val=classes_val,
        classes_train=classes_train,
        train_index=np.asarray(idx_train),
        val_index=np.asarray(idx_val),
        split_mode="instance" if grouped else "row",
    )


