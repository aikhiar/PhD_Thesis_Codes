"""
model.py -- the Chapter 6 model, shared verbatim by the three frameworks of the study
(methods_comparison, surrogate_trainer, det_vs_stoch; see
parameters.py): instances, feasibility-preserving operators, the recourse
policy, the Monte Carlo objectives, the surrogate feature map, and independent
verifiers. It imports only NumPy, SciPy
and Numba (no framework), so the three experiments run on the same model.

Problem      n vehicles (estimated arrival a_i, declared departure d_i, estimated
             energy e_i) on m constant-power chargers (w_j kW); preemptive
             sessions: at most MAX_PREEMPTIONS interruptions, charging and idle
             intervals of at least m^(sl) and m^(pl), and m^(d) between two
             consecutive vehicles of a charger.
Uncertainty  one pair (eps_a, eps_e) per vehicle, drawn from kappa: the
             instance's Gaussian KDE truncated to the admissible set D by
             rejection (parameters.py, ONE_SIDED_ENERGY_ERROR); realized
             a~ = a (1 + eps_a), e~ = e (1 + eps_e); independent across vehicles.
Recourse     charger by charger, in planned start order: start = max(planned
             start, realized arrival, predecessor's realized end + m^(d)); the
             charging intervals are repaired to the realized charging time
             e~ / w by the tail (extended, or shortened from the last one
             backwards, never below m^(sl) per charging interval), so the
             realized occupancy of a vehicle is
                 B_i = max(e~_i / w, (p_i + 1) m^(sl)) + Delta_i
             (p_i preemptions, Delta_i planned idle time), checked by
             check_realized_schedule_feasibility.
Objectives   (mean tardiness in minutes, peak load in kW), both minimised, as
             expectations over kappa estimated by Monte Carlo. The tardiness of
             a vehicle is max(0, t~_e - d) (times in hours, reported in
             minutes): the same weight for every vehicle and every time of day.

Solution     float64 array (n, 1 + 2 (2 MAX_PREEMPTIONS + 1)):
                 row[i, 0]  = charger index in {1, ..., m}
                 row[i, 1:] = (start, end) pairs, alternating charging, idle,
                              charging, ..., charging; zero-padded.
problem_params (every solver-facing operator) = (perc_perturb, lam, n_replications, instance)

The njit operators and the recourse policy are those of the experiments of the
paper (Algorithms 1-11), unchanged.
"""

import math
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from numba import njit
from scipy.stats import gaussian_kde

from . import parameters as P

CHAPTER6_DIR = Path(__file__).resolve().parent
INFINITY = 123456789
EPS = 1e-9
MIN_PREEMPTIONS = 0
OBJECTIVE_NAMES = ["Mean tardiness (min)", "Peak load (kW)"]
MINUTES_PER_HOUR = 60.0                  # instance times are in hours, the tardiness in minutes
_ACN = None


def check_parameters() -> None:
    """Validates parameters.py at import: a wrong value stops the study at once
    with a clear message instead of failing inside a run."""
    err = []

    def need(cond, msg):
        if not cond:
            err.append(msg)

    need(len(P.SIZES) >= 1 and all(isinstance(n, int) and n >= 2 for n in P.SIZES), "SIZES: integers >= 2")
    need(len(P.SIZE_LABELS) == len(P.SIZES), "SIZE_LABELS: one label per size")
    need(isinstance(P.VEHICLES_PER_CHARGER, int) and P.VEHICLES_PER_CHARGER >= 1, "VEHICLES_PER_CHARGER >= 1")
    need(len(P.CHARGER_POWERS_KW) >= 1 and min(P.CHARGER_POWERS_KW) > 0, "CHARGER_POWERS_KW: positive powers")
    need((CHAPTER6_DIR / P.ACN_DATA_FILE).exists(), f"missing data file chapter6/{P.ACN_DATA_FILE}")
    for key in ("INSTANCES_PER_SIZE", "TRAINING_INSTANCES_PER_SIZE", "DET_STOCH_INSTANCES_PER_SIZE"):
        need(isinstance(getattr(P, key), int) and getattr(P, key) >= 1, f"{key}: integer >= 1")
    need(P.DET_STOCH_INSTANCES_PER_SIZE <= P.INSTANCES_PER_SIZE, "DET_STOCH_INSTANCES_PER_SIZE <= INSTANCES_PER_SIZE")
    need(isinstance(P.MAX_PREEMPTIONS, int) and P.MAX_PREEMPTIONS >= 0, "MAX_PREEMPTIONS: integer >= 0")
    need(min(P.MIN_CHARGING_INTERVAL_H, P.MIN_IDLE_INTERVAL_H, P.MIN_SEPARATION_H) > 0, "minimum intervals > 0")
    need(P.ERROR_SIGMA_ARRIVAL > 0 and P.ERROR_SIGMA_ENERGY > 0, "error standard deviations > 0")
    need(-1 < P.ERROR_CORRELATION < 1, "ERROR_CORRELATION in (-1, 1)")
    need(isinstance(P.N_KDE_SAMPLES, int) and P.N_KDE_SAMPLES >= 10, "N_KDE_SAMPLES >= 10")
    need(isinstance(P.LABEL_REPLICATIONS, int) and P.LABEL_REPLICATIONS >= 1, "LABEL_REPLICATIONS >= 1")
    need(isinstance(P.REFINEMENT_REPLICATIONS, int) and P.REFINEMENT_REPLICATIONS >= P.LABEL_REPLICATIONS,
         "REFINEMENT_REPLICATIONS >= LABEL_REPLICATIONS")
    need(set(P.PROBLEM_PARAM_GRID) == {"perc_perturb"} and len(P.PROBLEM_PARAM_GRID["perc_perturb"]) >= 1
         and all(0 < x <= 1 for x in P.PROBLEM_PARAM_GRID["perc_perturb"]),
         "PROBLEM_PARAM_GRID: perc_perturb values in (0, 1]")
    need(0 < P.UNTUNED_PERC_PERTURB <= 1, "UNTUNED_PERC_PERTURB in (0, 1]")
    need(P.FREE_REGION_RATE > 0, "FREE_REGION_RATE > 0")
    need(set(P.SURROGATE_FILES) == set(OBJECTIVE_NAMES), f"SURROGATE_FILES: one file per objective {OBJECTIVE_NAMES}")
    if err:
        raise ValueError("problems/chapter6/parameters.py:\n  " + "\n  ".join(err))


check_parameters()


# ═══════════════════════════════════════════════════════════════════════════════
# INSTANCES
# ═══════════════════════════════════════════════════════════════════════════════

class Chapter6Instance:
    """One instance: the data, the 10-field tuple read by the njit operators
    (`arrays`) and the error law kappa (`kde`, fitted on `kde_sample`)."""

    def __init__(self, name, seed, arrivals, departures, energy, powers, kde_sample):
        self.name, self.seed = str(name), int(seed)
        self.a = np.asarray(arrivals, dtype=np.float64)
        self.d = np.asarray(departures, dtype=np.float64)
        self.e = np.asarray(energy, dtype=np.float64)
        self.w = np.asarray(powers, dtype=np.float64)
        self.n, self.m = len(self.a), len(self.w)
        self.kde_sample = np.asarray(kde_sample, dtype=np.float64)   # (2, N): (eps_a, |eps_e|)
        self.kde = gaussian_kde(self.kde_sample)
        self.arrays = (self.a, self.d, self.e, self.w,
                       float(P.MIN_CHARGING_INTERVAL_H), float(P.MIN_IDLE_INTERVAL_H),
                       float(P.MIN_SEPARATION_H), int(self.n), int(self.m), int(P.MAX_PREEMPTIONS))

    def features(self) -> Dict[str, float]:
        """Numeric descriptors of the instance (size n, ...)."""
        return {"n": float(self.n), "m": float(self.m),
                "energy_per_charger_kwh": float(self.e.sum() / self.m),
                "mean_power_kw": float(self.w.mean()),
                "arrival_span_h": float(self.a.max() - self.a.min())}

    def __repr__(self):
        return f"Chapter6Instance({self.name}: n={self.n}, m={self.m})"


def load_acn() -> np.ndarray:
    """ACN-Data claimed sessions: (arrival h, requested departure h, requested energy kWh)."""
    global _ACN
    if _ACN is None:
        with open(CHAPTER6_DIR / P.ACN_DATA_FILE, "rb") as fh:
            _ACN = np.asarray(pickle.load(fh), dtype=np.float64)
    return _ACN


def _sample_sessions(n: int, requests: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """n distinct sessions whose arrival times follow the KDE of the ACN
    arrivals: draw arrivals from the KDE, keep the session whose connection time
    is the closest, without repetition within the instance."""
    arrivals = requests[:, 0]
    kde = gaussian_kde(arrivals)
    chosen, seen, attempts = [], set(), 0
    max_attempts = n * P.ACN_MAX_ATTEMPTS_PER_VEHICLE
    while len(chosen) < n and attempts < max_attempts:
        sample = kde.resample(P.ACN_KDE_BATCH, seed=rng.integers(int(1e9)))[0]
        for i in np.argmin(np.abs(arrivals[:, None] - sample[None, :]), axis=0):
            if int(i) not in seen:
                seen.add(int(i))
                chosen.append(int(i))
                if len(chosen) == n:
                    break
        attempts += P.ACN_KDE_BATCH
    if len(chosen) < n:
        raise RuntimeError(f"could not draw {n} distinct ACN sessions")
    return np.array(chosen, dtype=np.int64)


def _error_sample(rng: np.random.Generator) -> np.ndarray:
    """The sample kappa is fitted on: (eps_a, |eps_e|) for N_KDE_SAMPLES draws of
    a bivariate normal (sigma_a, sigma_e, rho)."""
    sa, se, rho = P.ERROR_SIGMA_ARRIVAL, P.ERROR_SIGMA_ENERGY, P.ERROR_CORRELATION
    L = np.linalg.cholesky(np.array([[sa ** 2, rho * sa * se], [rho * sa * se, se ** 2]]))
    xy = L @ rng.standard_normal((2, P.N_KDE_SAMPLES))
    return np.vstack([xy[0], np.abs(xy[1])])


def n_chargers(n: int) -> int:
    return -(-int(n) // P.VEHICLES_PER_CHARGER)


def build_instance(n: int, seed: int, name: str = None, m: int = None) -> Chapter6Instance:
    """Instance of n vehicles drawn from the ACN sessions with its own seeded
    generator: sessions, then charger powers, then the sample of kappa."""
    rng = np.random.default_rng(seed)
    requests = load_acn()
    data = requests[_sample_sessions(n, requests, rng)]
    a = data[:, 0]
    d = np.maximum(data[:, 1], a)
    e = data[:, 2]
    m = n_chargers(n) if m is None else int(m)
    powers = rng.choice(np.asarray(P.CHARGER_POWERS_KW, dtype=np.float64), size=m, replace=True)
    return Chapter6Instance(name or f"n{n}-s{seed}", seed, a, d, e, powers, _error_sample(rng))


def evaluation_instances() -> List[Chapter6Instance]:
    """The suite of the metaheuristics comparison: INSTANCES_PER_SIZE instances per
    size, ids 1..len, instance k (0-based) with seed EVAL_SEED_BASE + k."""
    out, k = [], 0
    for n, label in zip(P.SIZES, P.SIZE_LABELS):
        for j in range(P.INSTANCES_PER_SIZE):
            out.append(build_instance(n, P.EVAL_SEED_BASE + k, name=f"{label}-{j + 1}"))
            k += 1
    return out


def size_classes() -> Dict[str, List[int]]:
    """Ids (1-based, order of evaluation_instances) of each size class."""
    out, k = {}, 0
    for label in P.SIZE_LABELS:
        out[label] = list(range(k + 1, k + 1 + P.INSTANCES_PER_SIZE))
        k += P.INSTANCES_PER_SIZE
    return out


def instance_class(inst: Chapter6Instance) -> str:
    return inst.name.rsplit("-", 1)[0]


def training_instances() -> List[Tuple[str, Chapter6Instance]]:
    """The surrogate training pool: (id, instance) pairs, ids '<label>_<j>',
    seeds TRAINING_SEED_BASE + k (disjoint from the evaluation seeds)."""
    out, k = [], 0
    for n, label in zip(P.SIZES, P.SIZE_LABELS):
        for j in range(P.TRAINING_INSTANCES_PER_SIZE):
            out.append((f"{label}_{j}", build_instance(n, P.TRAINING_SEED_BASE + k, name=f"train-{label}-{j}")))
            k += 1
    return out


def det_stoch_ids() -> List[int]:
    """Evaluation ids of the deterministic-vs-stochastic comparison: the first
    DET_STOCH_INSTANCES_PER_SIZE instances of each size class."""
    return [i for ids in size_classes().values() for i in ids[:P.DET_STOCH_INSTANCES_PER_SIZE]]


def evaluation_instance(inst_id: int) -> Chapter6Instance:
    """Evaluation instance `inst_id` (1-based), built alone."""
    k = int(inst_id) - 1
    c = k // P.INSTANCES_PER_SIZE
    return build_instance(P.SIZES[c], P.EVAL_SEED_BASE + k,
                          name=f"{P.SIZE_LABELS[c]}-{k % P.INSTANCES_PER_SIZE + 1}")


# ═══════════════════════════════════════════════════════════════════════════════
# THE ERROR LAW kappa AND ONE REALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

_ENERGY_ERROR_FLOOR = 0.0 if P.ONE_SIDED_ENERGY_ERROR else -1.0


def _outside_D(s: np.ndarray) -> np.ndarray:
    return (s[0] <= -1.0) | (s[1] < _ENERGY_ERROR_FLOOR)


def sample_errors(inst: Chapter6Instance, size: int, rng: np.random.Generator) -> np.ndarray:
    """`size` draws (2, size) of kappa: the instance KDE truncated to D by
    rejection (a draw outside D is replaced by a fresh one)."""
    s = inst.kde.resample(int(size), seed=rng)
    bad = _outside_D(s)
    while np.any(bad):
        s[:, bad] = inst.kde.resample(int(bad.sum()), seed=rng)
        bad = _outside_D(s)
    return s


def sample_realization(inst: Chapter6Instance) -> Tuple[np.ndarray, np.ndarray]:
    """One joint realization (realized arrivals, realized energies). Its
    generator is seeded from NumPy's global RNG, which the frameworks seed
    before every run, so a run is reproducible."""
    rng = np.random.default_rng(int(np.random.randint(0, 2 ** 31 - 1)))
    s = sample_errors(inst, inst.n, rng)
    return np.maximum(inst.a * (1.0 + s[0]), 0.0), np.maximum(inst.e * (1.0 + s[1]), 0.0)


@njit
def _nb_seed(seed):
    np.random.seed(seed)


def seed_all(seed: int) -> None:
    """Seeds Python's random, NumPy's global RNG and Numba's own RNG (the njit
    operators draw from the latter)."""
    random.seed(seed)
    np.random.seed(seed)
    _nb_seed(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# NUMBA OPERATORS AND RECOURSE POLICY (the paper's Algorithms 1-11, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

@njit
def _pad_intervals(intervals: np.ndarray, max_rows: int) -> np.ndarray:
    n_rows, n_cols = intervals.shape
    padded = np.zeros((max_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        for j in range(n_cols):
            padded[i, j] = intervals[i, j]
    return padded


@njit
def _get_row_start_time(row: np.ndarray) -> float:
    for col in range(1, row.shape[0] - 1, 2):
        s = row[col]
        e = row[col + 1]
        if s == 0.0 and e == 0.0:
            break
        return s
    return 0.0


@njit
def _get_row_end_time(row: np.ndarray) -> float:
    end_time = 0.0
    for col in range(1, row.shape[0] - 1, 2):
        s = row[col]
        e = row[col + 1]
        if s == 0.0 and e == 0.0:
            break
        end_time = e
    return end_time


@njit
def _count_active_intervals(row: np.ndarray) -> int:
    n_intervals = 0
    for col in range(1, row.shape[0] - 1, 2):
        s = row[col]
        e = row[col + 1]
        if s == 0.0 and e == 0.0:
            break
        n_intervals += 1
    return n_intervals


@njit
def _get_total_charging_time(row: np.ndarray) -> float:
    total = 0.0
    interval_idx = 0
    for col in range(1, row.shape[0] - 1, 2):
        s = row[col]
        e = row[col + 1]
        if s == 0.0 and e == 0.0:
            break
        if interval_idx % 2 == 0:
            total += (e - s)
        interval_idx += 1
    return total


@njit
def _shift_row_to_start(row: np.ndarray, new_start: float) -> np.ndarray:
    shifted = row.copy()
    old_start = _get_row_start_time(row)
    delta = new_start - old_start
    for col in range(1, row.shape[0] - 1, 2):
        s = row[col]
        e = row[col + 1]
        if s == 0.0 and e == 0.0:
            break
        shifted[col] = s + delta
        shifted[col + 1] = e + delta
    return shifted


@njit
def _left_compact_row(row: np.ndarray) -> np.ndarray:
    compacted = row.copy()
    current = _get_row_start_time(compacted)
    for col in range(1, compacted.shape[0] - 1, 2):
        s = compacted[col]
        e = compacted[col + 1]
        if s == 0.0 and e == 0.0:
            break
        length = e - s
        compacted[col] = current
        compacted[col + 1] = current + length
        current = compacted[col + 1]
    return compacted


@njit
def _get_charger_free_intervals(
    schedule: np.ndarray,
    charger_idx: int,
    distance_between_two_sessions: float,
) -> np.ndarray:
    delta = distance_between_two_sessions
    max_sessions = schedule.shape[0]
    occupied = np.empty((max_sessions, 2), dtype=np.float64)
    n_occupied = 0

    for i in range(max_sessions):
        if schedule[i, 0] != charger_idx:
            continue
        row = schedule[i, 1:]
        n_slots = row.shape[0]
        session_start = -1.0
        session_end = -1.0
        for k in range(0, n_slots - 1, 2):
            s = row[k]
            e = row[k + 1]
            if s == 0.0 and e == 0.0:
                break
            if session_start < 0.0:
                session_start = s
            session_end = e
        if session_start >= 0.0:
            occupied[n_occupied, 0] = session_start - delta
            occupied[n_occupied, 1] = session_end + delta
            n_occupied += 1

    occupied = occupied[:n_occupied]

    for i in range(1, n_occupied):
        key_s = occupied[i, 0]
        key_e = occupied[i, 1]
        j = i - 1
        while j >= 0 and occupied[j, 0] > key_s:
            occupied[j + 1, 0] = occupied[j, 0]
            occupied[j + 1, 1] = occupied[j, 1]
            j -= 1
        occupied[j + 1, 0] = key_s
        occupied[j + 1, 1] = key_e

    free = np.empty((n_occupied + 1, 2), dtype=np.float64)
    n_free = 0
    cursor = 0.0
    for i in range(n_occupied):
        occ_start = max(occupied[i, 0], 0.0)
        occ_end = occupied[i, 1]
        if cursor < occ_start:
            free[n_free, 0] = cursor
            free[n_free, 1] = occ_start
            n_free += 1
        cursor = max(cursor, occ_end)
    free[n_free, 0] = cursor
    free[n_free, 1] = INFINITY
    n_free += 1

    return free[:n_free]


@njit
def _select_free_region(
    free_intervals: np.ndarray,
    min_start: float,
    min_length: float,
    lam: float,
) -> tuple:
    max_k = free_intervals.shape[0]
    idx = np.empty(max_k, dtype=np.int64)
    n_elig = 0
    for i in range(max_k):
        free_start = free_intervals[i, 0]
        free_end = free_intervals[i, 1]
        eff_start = max(free_start, min_start)
        if free_end == INFINITY or (
            eff_start < free_end and (free_end - eff_start) >= min_length
        ):
            idx[n_elig] = i
            n_elig += 1

    chosen = idx[np.random.randint(0, n_elig)]
    free_start = free_intervals[chosen, 0]
    free_end = free_intervals[chosen, 1]
    eff_start = max(free_start, min_start)

    if free_end == INFINITY:
        # inverse-CDF: usable length beyond the demand, mean 1/lam
        extra = -np.log(1.0 - np.random.uniform(0.0, 1.0)) / lam
        usable = min_length + extra
    else:
        usable = free_end - eff_start

    return eff_start, usable


@njit
def _sample_preemptive_schedule(
    charging_start: float,
    charging_end: float,
    n_preemptions: int,
    min_charging_interval: float,
    min_non_charging_interval: float,
    total_charging_time: float,
) -> tuple:
    n_c = n_preemptions + 1
    n_nc = n_preemptions
    n = n_c + n_nc
    total = charging_end - charging_start

    charging_slack = total_charging_time - n_c * min_charging_interval
    total_non_charging_time = total - total_charging_time
    non_charging_slack = total_non_charging_time - n_nc * min_non_charging_interval

    intervals = np.zeros((n, 2))
    if charging_slack < -EPS or non_charging_slack < -EPS:
        return intervals, 0

    gamma_c = np.empty(n_c)
    for i in range(n_c):
        gamma_c[i] = np.random.gamma(1.0, 1.0)
    gamma_c_sum = gamma_c.sum()
    charging_lengths = np.empty(n_c)
    for i in range(n_c):
        charging_lengths[i] = min_charging_interval + (gamma_c[i] / gamma_c_sum) * charging_slack

    if n_nc > 0:
        gamma_nc = np.empty(n_nc)
        for i in range(n_nc):
            gamma_nc[i] = np.random.gamma(1.0, 1.0)
        gamma_nc_sum = gamma_nc.sum()
        non_charging_lengths = np.empty(n_nc)
        for i in range(n_nc):
            non_charging_lengths[i] = min_non_charging_interval + (gamma_nc[i] / gamma_nc_sum) * non_charging_slack

    cursor = charging_start
    c_idx = 0
    nc_idx = 0
    for i in range(n):
        if i % 2 == 0:
            length = charging_lengths[c_idx]
            c_idx += 1
        else:
            length = non_charging_lengths[nc_idx]
            nc_idx += 1
        intervals[i, 0] = cursor
        intervals[i, 1] = cursor + length
        cursor = intervals[i, 1]

    return intervals, 1


@njit
def _sample_n_preemptions(
    total_charging_time: float,
    interval_length: float,
    min_charging_interval: float,
    min_non_charging_interval: float,
    max_preemptions: int,
) -> int:
    max_p_charging = int(total_charging_time / min_charging_interval) - 1
    max_p_non_charging = int((interval_length - total_charging_time) / min_non_charging_interval)
    p_max = min(max_preemptions, max_p_charging, max_p_non_charging)
    p_max = max(p_max, 0)
    lo = MIN_PREEMPTIONS if MIN_PREEMPTIONS <= p_max else p_max
    return np.random.randint(lo, p_max + 1)


@njit
def _stochastic_insertion_heuristic(
    schedule: np.ndarray,
    vehicle_index: int,
    lam: float,
    Instance: tuple,
) -> np.ndarray:
    (estimated_arrivals, declared_departures, estimated_energy, powers,
     min_charging_interval, min_non_charging_interval,
     distance_between_two_sessions, n, m, max_preemptions) = Instance

    schedule = schedule.copy()
    # InsertVehicle (Algorithm sae:alg:insert) presumes vehicle_index is not
    # yet present in `schedule` (true for Construct/Crossover, which only
    # ever call this on still-zero rows); Perturb (Algorithm sae:alg:perturb)
    # explicitly "removes i from S'" before re-inserting an ALREADY-assigned
    # vehicle, so clear its old row here too -- otherwise, if the freshly
    # drawn charger happens to match the vehicle's previous charger,
    # FreeRegions would wrongly treat the vehicle's own old occupancy as
    # space occupied by someone else.
    schedule[vehicle_index, :] = 0.0
    charger_idx = np.random.randint(m) + 1

    free_intervals = _get_charger_free_intervals(schedule, charger_idx, distance_between_two_sessions)

    min_start = estimated_arrivals[vehicle_index]
    total_charging_time = estimated_energy[vehicle_index] / powers[charger_idx - 1]
    # check_schedule_feasibility (and _set_realized_energy_on_row's realized-
    # schedule counterpart) enforce min_charging_interval as a hard floor on
    # every charging interval, with no exception for low-energy vehicles, so
    # the region search must reserve enough room for that floor even when
    # total_charging_time itself is smaller than it.
    session_floor = max(total_charging_time, min_charging_interval)

    eff_start, usable_length = _select_free_region(free_intervals, min_start, session_floor, lam)

    if total_charging_time < min_charging_interval:
        # A lone (non-preemptible) session can't be shorter than
        # min_charging_interval, so a vehicle whose own energy need is below
        # that floor still gets a full min_charging_interval-long session --
        # over-delivering energy -- rather than a session sized to its exact
        # (too-small) need. check_schedule_feasibility's CHARGE-TIME check is
        # floor-aware to match (mirrors the realized-schedule checker's
        # existing floor_bound handling in check_realized_schedule_feasibility).
        charging_start = np.random.uniform(eff_start, eff_start + usable_length - min_charging_interval)
        charging_intervals = np.zeros((1, 2))
        charging_intervals[0, 0] = charging_start
        charging_intervals[0, 1] = charging_start + min_charging_interval
    else:
        n_preemptions = _sample_n_preemptions(
            total_charging_time, usable_length,
            min_charging_interval, min_non_charging_interval, max_preemptions,
        )

        interval_length = total_charging_time + n_preemptions * min_non_charging_interval
        charging_start = np.random.uniform(eff_start, eff_start + usable_length - interval_length)
        charging_end = charging_start + interval_length

        charging_intervals, _ = _sample_preemptive_schedule(
            charging_start, charging_end, n_preemptions,
            min_charging_interval, min_non_charging_interval, total_charging_time,
        )

    schedule[vehicle_index, 0] = charger_idx
    schedule[vehicle_index, 1:] = _pad_intervals(charging_intervals, 2 * max_preemptions + 1).flatten()
    return schedule


@njit
def _generate_random_schedule(lam: float, Instance: tuple) -> np.ndarray:
    (estimated_arrivals, declared_departures, estimated_energy, powers,
     min_charging_interval, min_non_charging_interval,
     distance_between_two_sessions, n, m, max_preemptions) = Instance

    schedule = np.zeros((n, 1 + 2 * (2 * max_preemptions + 1)), dtype=np.float64)
    vehicles = np.arange(n)
    np.random.shuffle(vehicles)
    for vehicle_index in vehicles:
        schedule = _stochastic_insertion_heuristic(schedule, vehicle_index, lam, Instance)
    return schedule


@njit
def _generalized_crossover(solutions: np.ndarray, lam: float, Instance: tuple) -> np.ndarray:
    (estimated_arrivals, declared_departures, estimated_energy, powers,
     min_charging_interval, min_non_charging_interval,
     distance_between_two_sessions, n, m, max_preemptions) = Instance

    delta = distance_between_two_sessions
    N = solutions.shape[0]
    schedule_width = 1 + 2 * (2 * max_preemptions + 1)
    offspring = np.zeros((n, schedule_width), dtype=np.float64)

    vehicles_indexes = np.arange(n)
    np.random.shuffle(vehicles_indexes)

    placed_flags = np.zeros(n, dtype=np.int8)
    not_placed = np.empty(n, dtype=np.int64)
    n_not_placed = 0

    for vi in range(n):
        i = vehicles_indexes[vi]

        if vi == 0:
            s = np.random.randint(N)
            offspring[i] = solutions[s, i].copy()
            placed_flags[i] = 1
            continue

        placable = np.ones(N, dtype=np.int8)
        for s in range(N):
            cand_charger = solutions[s, i, 0]
            cand_start = -1.0
            cand_end = -1.0
            row_s = solutions[s, i, 1:]
            n_slots = row_s.shape[0]
            for k in range(0, n_slots - 1, 2):
                sv = row_s[k]
                ev = row_s[k + 1]
                if sv == 0.0 and ev == 0.0:
                    break
                if cand_start < 0.0:
                    cand_start = sv
                cand_end = ev

            if cand_start < 0.0:
                placable[s] = 0
                continue

            for j in range(n):
                if placed_flags[j] == 0:
                    continue
                if offspring[j, 0] != cand_charger:
                    continue

                placed_start = -1.0
                placed_end = -1.0
                row_j = offspring[j, 1:]
                for k in range(0, n_slots - 1, 2):
                    sv = row_j[k]
                    ev = row_j[k + 1]
                    if sv == 0.0 and ev == 0.0:
                        break
                    if placed_start < 0.0:
                        placed_start = sv
                    placed_end = ev

                if placed_start < 0.0:
                    continue

                no_overlap = (
                    cand_end + delta <= placed_start
                    or placed_end + delta <= cand_start
                )
                if not no_overlap:
                    placable[s] = 0
                    break

        n_placable = 0
        for s in range(N):
            if placable[s] == 1:
                n_placable += 1

        if n_placable > 0:
            chosen_idx = np.random.randint(n_placable)
            count = 0
            for s in range(N):
                if placable[s] == 1:
                    if count == chosen_idx:
                        offspring[i] = solutions[s, i].copy()
                        placed_flags[i] = 1
                        break
                    count += 1
        else:
            not_placed[n_not_placed] = i
            n_not_placed += 1

    for k in range(n_not_placed):
        i = not_placed[k]
        offspring = _stochastic_insertion_heuristic(offspring, i, lam, Instance)

    return offspring


@njit
def _perturb(schedule: np.ndarray, perc_perturb: float, lam: float, Instance: tuple) -> np.ndarray:
    (estimated_arrivals, declared_departures, estimated_energy, powers,
     min_charging_interval, min_non_charging_interval,
     distance_between_two_sessions, n, m, max_preemptions) = Instance

    schedule = schedule.copy()
    candidate_vehicles = np.arange(n)
    np.random.shuffle(candidate_vehicles)

    for i in candidate_vehicles:
        if np.random.rand() < perc_perturb:
            schedule = _stochastic_insertion_heuristic(schedule, i, lam, Instance)

    return schedule


# ============================================================================
# STOCHASTIC REALIZATION (RECOURSE POLICY)
# ============================================================================

@njit
def _set_realized_energy_on_row(
    row: np.ndarray,
    realized_charging_time: float,
    min_charging_interval: float,
) -> np.ndarray:
    """Tail-based energy repair (Algorithm sae:alg:repair): extend the last
    charging interval if more charging is needed, else shrink charging
    intervals from the last backwards (never below min_charging_interval)
    and left-compact to remove the resulting holes."""
    updated = row.copy()
    current_charging_time = _get_total_charging_time(updated)
    n_intervals = _count_active_intervals(updated)
    if n_intervals == 0:
        return updated

    diff = realized_charging_time - current_charging_time
    if abs(diff) <= EPS:
        return updated

    last_charging_interval_idx = 0
    for interval_idx in range(n_intervals):
        if interval_idx % 2 == 0:
            last_charging_interval_idx = interval_idx
    last_col = 1 + 2 * last_charging_interval_idx

    if diff > 0.0:
        updated[last_col + 1] += diff
        return updated

    reduction = -diff
    for interval_idx in range(n_intervals - 1, -1, -1):
        if interval_idx % 2 != 0:
            continue
        col = 1 + 2 * interval_idx
        s = updated[col]
        e = updated[col + 1]
        curr_len = e - s
        removable = max(0.0, curr_len - min_charging_interval)
        take = min(removable, reduction)
        updated[col + 1] = e - take
        reduction -= take
        if reduction <= EPS:
            break

    updated = _left_compact_row(updated)
    return updated


@njit
def _enforce_gap_from_previous(previous_row: np.ndarray, current_row: np.ndarray, min_gap: float) -> np.ndarray:
    prev_end = _get_row_end_time(previous_row)
    curr_start = _get_row_start_time(current_row)
    required_start = prev_end + min_gap
    if curr_start + EPS < required_start:
        return _shift_row_to_start(current_row, required_start)
    return current_row.copy()


@njit
def _realize_vehicle_row(
    planned_row: np.ndarray,
    realized_arrival: float,
    realized_energy: float,
    charger_power: float,
    min_charging_interval: float,
) -> np.ndarray:
    """Own-vehicle realization step (Algorithm sae:alg:realize): start =
    max(planned_start, realized_arrival); the required charging time is
    recomputed from the realized energy; the planned interval pattern is
    shifted then energy-repaired."""
    planned_start = _get_row_start_time(planned_row)
    realized_start = max(planned_start, realized_arrival)
    realized_charging_time = realized_energy / charger_power

    row = _shift_row_to_start(planned_row, realized_start)
    row = _set_realized_energy_on_row(row, realized_charging_time, min_charging_interval)
    return row


@njit
def _get_charger_vehicle_order(schedule: np.ndarray, m: int) -> tuple:
    n = schedule.shape[0]
    counts = np.zeros(m, dtype=np.int64)
    for i in range(n):
        charger = int(schedule[i, 0])
        if charger > 0:
            counts[charger - 1] += 1

    max_count = 0
    for c in range(m):
        if counts[c] > max_count:
            max_count = counts[c]

    orders = -np.ones((m, max_count), dtype=np.int64)
    fill = np.zeros(m, dtype=np.int64)
    for i in range(n):
        charger = int(schedule[i, 0])
        if charger > 0:
            c = charger - 1
            orders[c, fill[c]] = i
            fill[c] += 1

    for c in range(m):
        k = counts[c]
        for i in range(1, k):
            key = orders[c, i]
            key_start = _get_row_start_time(schedule[key])
            j = i - 1
            while j >= 0 and _get_row_start_time(schedule[orders[c, j]]) > key_start:
                orders[c, j + 1] = orders[c, j]
                j -= 1
            orders[c, j + 1] = key

    return orders, counts


@njit
def _apply_realization_policy_numba(
    planned_schedule: np.ndarray,
    realized_arrivals: np.ndarray,
    realized_energies: np.ndarray,
    powers: np.ndarray,
    min_charging_interval: float,
    distance_between_two_sessions: float,
    m: int,
) -> np.ndarray:
    """Charger-wise, chronological recourse policy (Algorithm sae:alg:realize):
    each charger's vehicles are realized in *planned* start order (Lemma
    sae:lem:order guarantees this is also the realized order); each row is
    first realized independently (own arrival + own energy repair), then
    forward-shifted -- never earlier -- to respect the m^(d) gap from its
    now-realized predecessor."""
    realized_schedule = planned_schedule.copy()
    orders, counts = _get_charger_vehicle_order(planned_schedule, m)

    for charger_zero_based in range(m):
        k = counts[charger_zero_based]
        if k == 0:
            continue
        charger_power = powers[charger_zero_based]

        for pos in range(k):
            i = orders[charger_zero_based, pos]
            row = _realize_vehicle_row(
                planned_schedule[i], realized_arrivals[i], realized_energies[i],
                charger_power, min_charging_interval,
            )
            if pos > 0:
                prev_i = orders[charger_zero_based, pos - 1]
                row = _enforce_gap_from_previous(realized_schedule[prev_i], row, distance_between_two_sessions)
            realized_schedule[i] = row

    return realized_schedule


@njit
def apply_realization_policy(
    planned_schedule: np.ndarray,
    realized_arrivals: np.ndarray,
    realized_energies: np.ndarray,
    Instance: tuple,
) -> np.ndarray:
    """Njit. `Instance`'s trailing element (instance_id) is a plain int64,
    so the full tuple is Numba-typeable and can be passed as-is."""
    powers = Instance[3]
    min_charging_interval = Instance[4]
    distance_between_two_sessions = Instance[6]
    m = Instance[8]
    return _apply_realization_policy_numba(
        np.asarray(planned_schedule, dtype=np.float64), realized_arrivals, realized_energies,
        powers, min_charging_interval, distance_between_two_sessions, m,
    )


@njit
def _compute_mean_tardiness(schedule: np.ndarray, declared_departures: np.ndarray) -> float:
    """(1/n) sum_i max(0, t_i^(e) - d_i), in minutes."""
    n = schedule.shape[0]
    total = 0.0
    for i in range(n):
        end_time = _get_row_end_time(schedule[i])
        if end_time > declared_departures[i]:
            total += end_time - declared_departures[i]
    return MINUTES_PER_HOUR * total / n


@njit
def _compute_peak_load(schedule: np.ndarray, powers: np.ndarray) -> float:
    max_events = schedule.shape[0] * (schedule.shape[1] - 1)
    events = np.empty((max_events, 2), dtype=np.float64)
    n_events = 0

    for i in range(schedule.shape[0]):
        charger = int(schedule[i, 0])
        if charger == 0:
            continue
        power = powers[charger - 1]
        col = 1
        sub_idx = 0
        while col + 1 < schedule.shape[1]:
            s = schedule[i, col]
            e = schedule[i, col + 1]
            if s == 0.0 and e == 0.0:
                break
            if sub_idx % 2 == 0:
                events[n_events, 0] = s
                events[n_events, 1] = power
                n_events += 1
                events[n_events, 0] = e
                events[n_events, 1] = -power
                n_events += 1
            col += 2
            sub_idx += 1

    for i in range(1, n_events):
        kt = events[i, 0]
        kp = events[i, 1]
        j = i - 1
        while j >= 0 and (events[j, 0] > kt or (events[j, 0] == kt and events[j, 1] > kp)):
            events[j + 1, 0] = events[j, 0]
            events[j + 1, 1] = events[j, 1]
            j -= 1
        events[j + 1, 0] = kt
        events[j + 1, 1] = kp

    peak = 0.0
    load = 0.0
    for i in range(n_events):
        load += events[i, 1]
        if load > peak:
            peak = load
    return peak


@njit
def _objective_deterministic(schedule: np.ndarray, Instance: tuple) -> np.ndarray:
    """Zero-error realization (kappa = delta_(0,0)), matching the degenerate
    law used in the NP-hardness reduction (Proposition sae:prop:hardness):
    estimates are realized exactly as declared. Njit -- `Instance`'s
    trailing instance_id is a plain int64, so the full tuple is
    Numba-typeable and can be passed as-is."""
    schedule = np.asarray(schedule, dtype=np.float64)
    estimated_arrivals, declared_departures = Instance[0], Instance[1]
    estimated_energy, powers = Instance[2], Instance[3]
    realized = apply_realization_policy(schedule, estimated_arrivals, estimated_energy, Instance)
    tardiness = _compute_mean_tardiness(realized, declared_departures)
    peak = _compute_peak_load(realized, powers)
    return np.array([tardiness, peak])


# ═══════════════════════════════════════════════════════════════════════════════
# OBJECTIVES
# ═══════════════════════════════════════════════════════════════════════════════

def deterministic_objective(schedule: np.ndarray, inst: Chapter6Instance) -> np.ndarray:
    """Zero-error realization (kappa = delta_(0,0)): the objectives of the plan
    under the declared arrivals and energies."""
    return _objective_deterministic(np.asarray(schedule, dtype=np.float64), inst.arrays)


def objective_mc(schedule: np.ndarray, n_replications: int, inst: Chapter6Instance) -> np.ndarray:
    """(mean tardiness in minutes, peak load): sample means over n_replications
    realizations of kappa (Algorithm 10); n_replications <= 0 gives the
    deterministic objective."""
    schedule = np.asarray(schedule, dtype=np.float64)
    if int(n_replications) <= 0:
        return deterministic_objective(schedule, inst)
    tardiness = peak = 0.0
    for _ in range(int(n_replications)):
        ra, re = sample_realization(inst)
        realized = apply_realization_policy(schedule, ra, re, inst.arrays)
        tardiness += _compute_mean_tardiness(realized, inst.d)
        peak += _compute_peak_load(realized, inst.w)
    r = float(n_replications)
    return np.array([tardiness / r, peak / r])


def realize(schedule: np.ndarray, inst: Chapter6Instance) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One realization of a planned schedule: (realized schedule, realized
    arrivals, realized energies)."""
    ra, re = sample_realization(inst)
    return apply_realization_policy(np.asarray(schedule, dtype=np.float64), ra, re, inst.arrays), ra, re


def score_realized(realized: np.ndarray, inst: Chapter6Instance) -> np.ndarray:
    """Objectives of an already realized schedule (no further realization)."""
    return np.array([_compute_mean_tardiness(realized, inst.d), _compute_peak_load(realized, inst.w)])


# ═══════════════════════════════════════════════════════════════════════════════
# SOLVER-FACING OPERATORS   problem_params = (perc_perturb, lam, n_replications, instance)
# ═══════════════════════════════════════════════════════════════════════════════

def build_problem_params(params: Dict[str, Any], inst: Chapter6Instance) -> tuple:
    return (float(params.get("perc_perturb", P.UNTUNED_PERC_PERTURB)),
            float(params.get("lam", P.FREE_REGION_RATE)),
            int(params.get("num_of_replications", P.LABEL_REPLICATIONS)),
            inst)


def Generate_Solution(problem_params: tuple) -> np.ndarray:
    """Construct (Algorithm 5)."""
    return _generate_random_schedule(float(problem_params[1]), problem_params[3].arrays)


def Perturb(schedule: np.ndarray, problem_params: tuple) -> np.ndarray:
    """PerturbSchedule (Algorithm 7)."""
    return _perturb(np.asarray(schedule, dtype=np.float64), float(problem_params[0]),
                    float(problem_params[1]), problem_params[3].arrays)


def Crossover(parents_array: np.ndarray, problem_params: tuple) -> np.ndarray:
    """Multi-parent conflict-free crossover (Algorithm 6)."""
    return _generalized_crossover(np.asarray(parents_array, dtype=np.float64),
                                  float(problem_params[1]), problem_params[3].arrays)


def objective_function(schedule: np.ndarray, problem_params: tuple) -> np.ndarray:
    """The Monte Carlo objective with problem_params' replication count."""
    return objective_mc(schedule, problem_params[2], problem_params[3])


def solution_shape(inst: Chapter6Instance) -> tuple:
    return (inst.n, 1 + 2 * (2 * int(inst.arrays[9]) + 1))


# ═══════════════════════════════════════════════════════════════════════════════
# SURROGATES: FEATURE MAP AND DEPLOYED MODELS
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "det_tardiness", "det_peak_load",
    "n_vehicles", "n_chargers",
    "total_charging_time", "mean_charging_time", "charging_time_std", "mean_estimated_energy",
    "mean_preemptions", "max_preemptions_used", "preemptions_std",
    "mean_session_length", "max_session_length", "mean_gap_length",
    "charger_load_std", "charger_load_max",
    "energy_heterogeneity", "arrival_heterogeneity", "charger_power_heterogeneity",
]


def _intervals(row: np.ndarray, eps: float) -> List[Tuple[float, float]]:
    out = []
    for col in range(1, row.shape[0] - 1, 2):
        s, e = float(row[col]), float(row[col + 1])
        if abs(s) <= eps and abs(e) <= eps:
            break
        out.append((s, e))
    return out


def feature_fn(solution: np.ndarray, inst: Chapter6Instance) -> np.ndarray:
    """phi(S): the zero-error replay of the recourse policy (deterministic proxy
    of both objectives), structural statistics of the schedule and their
    dispersion, and instance heterogeneity -- no Monte Carlo draw. Generation,
    algorithm and front rank are deliberately excluded (they describe how a
    schedule was found, not the schedule)."""
    solution = np.asarray(solution, dtype=np.float64)
    n, m = inst.n, inst.m
    det_tardiness, det_peak = deterministic_objective(solution, inst)
    charging_times = np.array([_get_total_charging_time(solution[i]) for i in range(n)])
    n_intervals = np.array([_count_active_intervals(solution[i]) for i in range(n)], dtype=np.float64)
    preemptions = (n_intervals - 1.0) / 2.0
    sessions, gaps = [], []
    for i in range(n):
        for k, (s, e) in enumerate(_intervals(solution[i], EPS)):
            (sessions if k % 2 == 0 else gaps).append(e - s)
    counts = np.zeros(m)
    for i in range(n):
        c = int(round(solution[i, 0]))
        if 1 <= c <= m:
            counts[c - 1] += 1
    return np.array([
        det_tardiness, det_peak,
        float(n), float(m),
        float(np.sum(charging_times)), float(np.mean(charging_times)), float(np.std(charging_times)),
        float(np.mean(inst.e)),
        float(np.mean(preemptions)), float(np.max(preemptions)), float(np.std(preemptions)),
        float(np.mean(sessions)) if sessions else 0.0, float(np.max(sessions)) if sessions else 0.0,
        float(np.mean(gaps)) if gaps else 0.0,
        float(np.std(counts)), float(np.max(counts)),
        float(np.std(inst.e)), float(np.std(inst.a)), float(np.std(inst.w)),
    ], dtype=np.float64)


_SURROGATES: Dict[str, Any] = {}


def surrogate_paths() -> Dict[str, Path]:
    return {name: CHAPTER6_DIR / P.SURROGATE_DIR / fname for name, fname in P.SURROGATE_FILES.items()}


_VERSION_CHECKED = []


def _check_library_versions() -> None:
    """Warns (once per process) when a library the surrogates were trained with
    (model_card.json written next to them by surrogate_trainer) is installed
    here in another version: a model unpickled under another scikit-learn,
    XGBoost or CatBoost version may fail or predict differently."""
    if _VERSION_CHECKED:
        return
    _VERSION_CHECKED.append(True)
    import json
    import warnings
    from importlib import metadata
    card_path = CHAPTER6_DIR / P.SURROGATE_DIR / "model_card.json"
    if not card_path.exists():
        warnings.warn(f"{card_path} not found: the library versions of the surrogates cannot be checked "
                      "(copy model_card.json from surrogate_export/best_models/ with the models)", stacklevel=3)
        return
    trained = json.loads(card_path.read_text()).get("environment", {}).get("packages", {})
    mismatched = []
    for pkg in ("scikit-learn", "xgboost", "catboost", "lightgbm", "numpy", "scipy"):
        if pkg in trained:
            try:
                here = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                here = "not installed"
            if here != trained[pkg]:
                mismatched.append(f"{pkg} {trained[pkg]} (training) vs {here} (here)")
    if mismatched:
        warnings.warn("the surrogates were trained with other library versions: " + "; ".join(mismatched),
                      stacklevel=3)


def load_surrogates() -> Dict[str, Any]:
    """The deployed regressors (one per objective), loaded once per process."""
    if _SURROGATES:
        return _SURROGATES
    missing = [str(p) for p in surrogate_paths().values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Chapter 6 surrogate models not found: " + ", ".join(missing) + ". Train them with "
            "surrogate_trainer (python main.py), then copy "
            "everything in runs/surrogate_chapter6_<key>/surrogate_export/best_models/ (the .joblib files and "
            "model_card.json) into problems/chapter6/"
            f"{P.SURROGATE_DIR}/ of methods_comparison.")
    import joblib
    _check_library_versions()
    models = {name: joblib.load(path) for name, path in surrogate_paths().items()}
    for name, model in models.items():
        k = getattr(model, "n_features_in_", len(FEATURE_NAMES))
        if k != len(FEATURE_NAMES):
            raise ValueError(f"surrogate '{name}' expects {k} features, the feature map gives {len(FEATURE_NAMES)}")
    _SURROGATES.update(models)
    return _SURROGATES


def surrogate_objective(schedule: np.ndarray, problem_params: tuple) -> np.ndarray:
    """Both objectives predicted by the deployed surrogates (the search
    objective of the surrogate-assisted search, Algorithm 13, line 2)."""
    models = load_surrogates()
    phi = feature_fn(schedule, problem_params[3]).reshape(1, -1)
    return np.array([float(models[name].predict(phi)[0]) for name in OBJECTIVE_NAMES], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# INDEPENDENT VERIFIERS (plain Python, not the operators' code)
# ═══════════════════════════════════════════════════════════════════════════════

def check_schedule_feasibility(schedule, inst: Chapter6Instance, label="schedule", eps=None):
    """Every clause of Definition 3.1 on a planned schedule; raises AssertionError
    describing the first violation. A vehicle whose demand is below m^(sl) must
    receive exactly one charging interval of length m^(sl) (waiver of clause iv)."""
    a, d, e, w, msl, mpl, md, n, m, pmax = inst.arrays
    eps = P.VERIFIER_EPS if eps is None else eps
    schedule = np.asarray(schedule, dtype=np.float64)
    assert schedule.shape == solution_shape(inst), f"[{label}] SHAPE FAIL {schedule.shape}"
    for i in range(n):
        charger = int(round(schedule[i, 0]))
        assert 1 <= charger <= m, f"[{label}] CHARGER FAIL vehicle {i} on charger {charger}"
        iv = _intervals(schedule[i], eps)
        assert iv, f"[{label}] EMPTY FAIL vehicle {i}"
        assert len(iv) % 2 == 1, f"[{label}] PARITY FAIL vehicle {i}: {len(iv)} intervals"
        assert len(iv) <= 2 * pmax + 1, f"[{label}] COUNT FAIL vehicle {i}: {len(iv)} intervals"
        assert iv[0][0] >= a[i] - eps, f"[{label}] ARRIVAL FAIL vehicle {i} starts {iv[0][0]} < {a[i]}"
        total = 0.0
        for k, (s, t) in enumerate(iv):
            assert np.isfinite(s) and np.isfinite(t) and t > s + eps, f"[{label}] LENGTH FAIL vehicle {i} interval {k}"
            if k > 0:
                assert abs(s - iv[k - 1][1]) <= 1e-6, f"[{label}] CONTIGUITY FAIL vehicle {i} interval {k}"
            if k % 2 == 1:
                assert t - s >= mpl - 1e-6, f"[{label}] IDLE FAIL vehicle {i} interval {k} shorter than m^(pl)"
            else:
                assert t - s >= msl - 1e-6, f"[{label}] CHARGE-LENGTH FAIL vehicle {i} interval {k} shorter than m^(sl)"
                total += t - s
        tau = e[i] / w[charger - 1]
        expected = msl if tau < msl else tau
        if tau < msl:
            assert len(iv) == 1, f"[{label}] SHORT-DEMAND FAIL vehicle {i}: preempted below the floor"
        assert abs(total - expected) <= 1e-5, \
            f"[{label}] CHARGE-TIME FAIL vehicle {i}: charging {total}, expected {expected}"
        assert not np.any(np.abs(schedule[i, 1 + 2 * len(iv):]) > eps), f"[{label}] TAIL FAIL vehicle {i}"
    for c in range(1, m + 1):
        occ = sorted((_intervals(schedule[i], eps)[0][0], _intervals(schedule[i], eps)[-1][1], i)
                     for i in range(n) if int(round(schedule[i, 0])) == c)
        for (s1, e1, i1), (s2, e2, i2) in zip(occ, occ[1:]):
            assert e1 + md <= s2 + eps, f"[{label}] OVERLAP FAIL charger {c}: vehicles {i1}, {i2}"
