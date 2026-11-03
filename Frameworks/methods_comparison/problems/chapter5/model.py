"""
model.py -- the Chapter 5 model, shared verbatim by the two frameworks of the study
(methods_comparison, surrogate_trainer; see
parameters.py): instances, feasibility-preserving operators, the stochastic
realization, the objectives, the surrogate feature map, and independent
verifiers. It imports only NumPy, SciPy
and Numba (no framework), so both experiments run on the same model.

Problem      n vehicles (arrival a_i, departure d_i, requested energy e_i) on m
             constant-power chargers; each vehicle receives a delivered energy
             E_i in [MIN_DELIVERED_FRACTION e_i, e_i] (decision) through a
             preemptive session (at most MAX_PREEMPTIONS interruptions; charging
             and idle intervals of at least m^(sl), m^(pl); m^(d) between two
             consecutive vehicles of a charger).
Uncertainty  every planned charging interval has a log-normal realized length
             (mean the planned length, standard deviation VARIANCE_FACTOR x it);
             realized intervals are shifted forward along each charger.
Objectives   (expected peak load kW, expected total tardiness h, undelivered
             energy kWh), all minimised; the first two by Monte Carlo, the third
             is deterministic (computed exactly).

Solution     float64 array (n, 2 + 2 (2 MAX_PREEMPTIONS + 1)):
                 row[i, 0]  = charger index in {1, ..., m}
                 row[i, 1]  = delivered energy E_i
                 row[i, 2:] = (start, end) pairs, alternating charging, pause,
                              charging, ..., charging; zero-padded.
problem_params (every solver-facing operator) = (perc_perturb, lam, n_replications, instance)

The njit operators and the realization are those of the experiments of the
paper, unchanged.
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

CHAPTER5_DIR = Path(__file__).resolve().parent
INFINITY = 123456789
EPS = 1e-9
OBJECTIVE_NAMES = ["Peak load (kW)", "Total tardiness (h)", "Undelivered energy (kWh)"]
SURROGATE_OBJECTIVE_NAMES = ["Peak load (kW)", "Total tardiness (h)"]     # the third is exact
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
    need(P.GRID_FRACTION > 0, "GRID_FRACTION > 0")
    need((CHAPTER5_DIR / P.ACN_DATA_FILE).exists(), f"missing data file chapter5/{P.ACN_DATA_FILE}")
    for key in ("INSTANCES_PER_SIZE", "TRAINING_INSTANCES_PER_SIZE"):
        need(isinstance(getattr(P, key), int) and getattr(P, key) >= 1, f"{key}: integer >= 1")
    need(isinstance(P.MAX_PREEMPTIONS, int) and P.MAX_PREEMPTIONS >= 0, "MAX_PREEMPTIONS: integer >= 0")
    need(min(P.MIN_CHARGING_INTERVAL_H, P.MIN_IDLE_INTERVAL_H, P.MIN_SEPARATION_H) > 0, "minimum intervals > 0")
    need(0 < P.MIN_DELIVERED_FRACTION <= 1, "MIN_DELIVERED_FRACTION in (0, 1]")
    need(P.VARIANCE_FACTOR > 0, "VARIANCE_FACTOR > 0")
    need(isinstance(P.LABEL_REPLICATIONS, int) and P.LABEL_REPLICATIONS >= 1, "LABEL_REPLICATIONS >= 1")
    need(isinstance(P.REFINEMENT_REPLICATIONS, int) and P.REFINEMENT_REPLICATIONS >= P.LABEL_REPLICATIONS,
         "REFINEMENT_REPLICATIONS >= LABEL_REPLICATIONS")
    need(set(P.PROBLEM_PARAM_GRID) == {"perc_perturb"} and len(P.PROBLEM_PARAM_GRID["perc_perturb"]) >= 1
         and all(0 < x <= 1 for x in P.PROBLEM_PARAM_GRID["perc_perturb"]),
         "PROBLEM_PARAM_GRID: perc_perturb values in (0, 1]")
    need(0 < P.UNTUNED_PERC_PERTURB <= 1, "UNTUNED_PERC_PERTURB in (0, 1]")
    need(P.FREE_REGION_RATE > 0, "FREE_REGION_RATE > 0")
    need(set(P.SURROGATE_FILES) == set(SURROGATE_OBJECTIVE_NAMES),
         f"SURROGATE_FILES: one file per learned objective {SURROGATE_OBJECTIVE_NAMES}")
    if err:
        raise ValueError("problems/chapter5/parameters.py:\n  " + "\n  ".join(err))


check_parameters()


# ═══════════════════════════════════════════════════════════════════════════════
# INSTANCES
# ═══════════════════════════════════════════════════════════════════════════════

class Chapter5Instance:
    """One instance: the data and the 15-field tuple read by the njit operators
    (`arrays`, the tuple of the original model)."""

    def __init__(self, name, seed, arrivals, departures, energy, powers):
        self.name, self.seed = str(name), int(seed)
        self.a = np.asarray(arrivals, dtype=np.float64)
        self.d = np.asarray(departures, dtype=np.float64)
        self.e = np.asarray(energy, dtype=np.float64)
        self.w = np.asarray(powers, dtype=np.float64)
        self.n, self.m = len(self.a), len(self.w)
        self.variance_factor = float(P.VARIANCE_FACTOR)
        full_r_ij = self.e[:, None] / self.w[None, :]
        p_max = P.GRID_FRACTION * float(np.sum(self.w))
        self.arrays = (self.a, self.d, self.e, full_r_ij, self.w, float(p_max),
                       float(np.min(self.a)), float(np.max(self.d)), int(self.n), int(self.m),
                       int(P.MAX_PREEMPTIONS), float(P.MIN_CHARGING_INTERVAL_H), float(P.MIN_IDLE_INTERVAL_H),
                       float(P.MIN_SEPARATION_H), float(P.MIN_DELIVERED_FRACTION))

    def features(self) -> Dict[str, float]:
        """Numeric descriptors of the instance (size n, ...)."""
        return {"n": float(self.n), "m": float(self.m),
                "energy_per_charger_kwh": float(self.e.sum() / self.m),
                "mean_power_kw": float(self.w.mean()),
                "arrival_span_h": float(self.a.max() - self.a.min())}

    def __repr__(self):
        return f"Chapter5Instance({self.name}: n={self.n}, m={self.m})"


def load_acn() -> np.ndarray:
    """ACN-Data claimed sessions: (arrival h, requested departure h, requested energy kWh)."""
    global _ACN
    if _ACN is None:
        with open(CHAPTER5_DIR / P.ACN_DATA_FILE, "rb") as fh:
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


def n_chargers(n: int) -> int:
    return -(-int(n) // P.VEHICLES_PER_CHARGER)


def build_instance(n: int, seed: int, name: str = None, m: int = None) -> Chapter5Instance:
    """Instance of n vehicles drawn from the ACN sessions with its own seeded
    generator: sessions, then charger powers."""
    rng = np.random.default_rng(seed)
    requests = load_acn()
    data = requests[_sample_sessions(n, requests, rng)]
    a = data[:, 0]
    d = np.maximum(data[:, 1], a)
    e = data[:, 2]
    m = n_chargers(n) if m is None else int(m)
    powers = rng.choice(np.asarray(P.CHARGER_POWERS_KW, dtype=np.float64), size=m, replace=True)
    return Chapter5Instance(name or f"n{n}-s{seed}", seed, a, d, e, powers)


def evaluation_instances() -> List[Chapter5Instance]:
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


def instance_class(inst: Chapter5Instance) -> str:
    return inst.name.rsplit("-", 1)[0]


def training_instances() -> List[Tuple[str, Chapter5Instance]]:
    """The surrogate training pool: (id, instance) pairs, ids '<label>_<j>',
    seeds TRAINING_SEED_BASE + k (disjoint from the evaluation seeds)."""
    out, k = [], 0
    for n, label in zip(P.SIZES, P.SIZE_LABELS):
        for j in range(P.TRAINING_INSTANCES_PER_SIZE):
            out.append((f"{label}_{j}", build_instance(n, P.TRAINING_SEED_BASE + k, name=f"train-{label}-{j}")))
            k += 1
    return out


def evaluation_instance(inst_id: int) -> Chapter5Instance:
    """Evaluation instance `inst_id` (1-based), built alone."""
    k = int(inst_id) - 1
    c = k // P.INSTANCES_PER_SIZE
    return build_instance(P.SIZES[c], P.EVAL_SEED_BASE + k,
                          name=f"{P.SIZE_LABELS[c]}-{k % P.INSTANCES_PER_SIZE + 1}")


@njit
def _nb_seed(seed):
    np.random.seed(seed)


def seed_all(seed: int) -> None:
    """Seeds Python's random, NumPy's global RNG and Numba's own RNG (the njit
    operators and the log-normal realization draw from the latter)."""
    random.seed(seed)
    np.random.seed(seed)
    _nb_seed(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# NUMBA OPERATORS, REALIZATION AND OBJECTIVES (the paper's, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

@njit
def _schedule_width(max_preemptions: int) -> int:
    return 2 + 2 * (2 * max_preemptions + 1)


# Pure-Python, list-of-tuples interval extractor. Kept only for the
# structural test helpers below (check_schedule_feasibility and friends),
# which are not on the hot path and are already verified against this exact
# implementation. The construction/objective hot path uses the @njit,
# fixed-size-array sibling _extract_intervals_arr defined right after it --
# same semantics, no Python list/sort-by-key (which Numba nopython mode
# cannot compile).
def _extract_active_intervals(row: np.ndarray, eps: float = EPS):
    intervals = []
    for col in range(2, row.shape[0] - 1, 2):
        s = float(row[col])
        e = float(row[col + 1])
        if abs(s) <= eps and abs(e) <= eps:
            break
        intervals.append((s, e))
    return intervals


@njit
def _extract_intervals_arr(row: np.ndarray):
    """Njit, array-based sibling of _extract_active_intervals: returns
    (starts, ends, count) instead of a list of tuples."""
    max_pairs = (row.shape[0] - 2) // 2
    starts = np.empty(max_pairs, dtype=np.float64)
    ends = np.empty(max_pairs, dtype=np.float64)
    count = 0
    for col in range(2, row.shape[0] - 1, 2):
        s = row[col]
        e = row[col + 1]
        if abs(s) <= EPS and abs(e) <= EPS:
            break
        starts[count] = s
        ends[count] = e
        count += 1
    return starts[:count], ends[:count], count


@njit
def _sample_uniform_simplex(k: int) -> np.ndarray:
    """Dirichlet(1,...,1) (uniform on the k-simplex), drawn without
    `np.random.dirichlet` (unsupported by Numba nopython mode): k i.i.d.
    Exponential(1) draws normalised to sum to 1 are exactly
    Dirichlet(1,...,1)-distributed."""
    e = np.empty(k, dtype=np.float64)
    total = 0.0
    for j in range(k):
        e[j] = np.random.exponential(1.0)
        total += e[j]
    for j in range(k):
        e[j] = e[j] / total
    return e


@njit
def _row_start_time(row: np.ndarray) -> float:
    for col in range(2, row.shape[0] - 1, 2):
        s = float(row[col])
        e = float(row[col + 1])
        if abs(s) <= EPS and abs(e) <= EPS:
            break
        return s
    return 0.0


@njit
def _row_end_time(row: np.ndarray) -> float:
    end_time = 0.0
    for col in range(2, row.shape[0] - 1, 2):
        s = float(row[col])
        e = float(row[col + 1])
        if abs(s) <= EPS and abs(e) <= EPS:
            break
        end_time = e
    return end_time


@njit
def _fill_row_from_intervals(
    row: np.ndarray,
    charger_idx: int,
    delivered_energy: float,
    starts: np.ndarray,
    ends: np.ndarray,
    count: int,
) -> np.ndarray:
    row[:] = 0.0
    row[0] = float(charger_idx)
    row[1] = float(delivered_energy)
    cursor = 2
    for idx in range(count):
        row[cursor] = starts[idx]
        row[cursor + 1] = ends[idx]
        cursor += 2
    return row


# ============================================================================
# CONSTRUCTION HELPERS
# ============================================================================

@njit
def _get_charger_free_intervals(
    schedule: np.ndarray,
    charger_idx: int,
    distance_between_two_sessions: float,
):
    n = schedule.shape[0]
    occ_s = np.empty(n, dtype=np.float64)
    occ_e = np.empty(n, dtype=np.float64)
    occ_count = 0
    for i in range(n):
        if int(round(schedule[i, 0])) != charger_idx:
            continue
        s = _row_start_time(schedule[i])
        e = _row_end_time(schedule[i])
        if e <= 0.0 and s <= 0.0:
            continue
        occ_s[occ_count] = max(s - distance_between_two_sessions, 0.0)
        occ_e[occ_count] = e + distance_between_two_sessions
        occ_count += 1

    order = np.argsort(occ_s[:occ_count])

    free_s = np.empty(occ_count + 1, dtype=np.float64)
    free_e = np.empty(occ_count + 1, dtype=np.float64)
    free_count = 0
    cursor = 0.0
    for k in range(occ_count):
        idx = order[k]
        s = occ_s[idx]
        e = occ_e[idx]
        if cursor < s:
            free_s[free_count] = cursor
            free_e[free_count] = s
            free_count += 1
        cursor = max(cursor, e)
    free_s[free_count] = cursor
    free_e[free_count] = INFINITY
    free_count += 1
    return free_s[:free_count], free_e[:free_count], free_count


@njit
def _sample_delivered_energy(
    requested_energy: float,
    charger_power: float,
    min_delivered_fraction: float,
    min_charging_interval: float,
) -> float:
    lower = min_delivered_fraction * requested_energy
    lower = max(lower, charger_power * min_charging_interval)
    if lower > requested_energy:
        return float(requested_energy)
    if abs(lower - requested_energy) <= EPS:
        return float(requested_energy)
    return float(np.random.uniform(lower, requested_energy))


@njit
def _sample_n_preemptions(
    total_charging_time: float,
    usable_length: float,
    max_preemptions: int,
    min_charging_interval: float,
    min_non_charging_interval: float,
) -> int:
    max_p_charging = int(total_charging_time / min_charging_interval) - 1
    max_p_non_charging = int((usable_length - total_charging_time) / min_non_charging_interval)
    p_max = min(max_preemptions, max_p_charging, max_p_non_charging)
    p_max = max(p_max, 0)
    return int(np.random.randint(0, p_max + 1))


@njit
def _sample_preemptive_operation(
    charging_start: float,
    total_window_length: float,
    total_charging_time: float,
    n_preemptions: int,
    min_charging_interval: float,
    min_non_charging_interval: float,
):
    n_charge = n_preemptions + 1
    n_pause = n_preemptions

    charge_slack = total_charging_time - n_charge * min_charging_interval
    total_pause_time = total_window_length - total_charging_time
    pause_slack = total_pause_time - n_pause * min_non_charging_interval

    if charge_slack < -EPS or pause_slack < -EPS:
        starts = np.empty(1, dtype=np.float64)
        ends = np.empty(1, dtype=np.float64)
        starts[0] = charging_start
        ends[0] = charging_start + total_charging_time
        return starts, ends, 1

    if n_charge == 1:
        charging_lengths = np.empty(1, dtype=np.float64)
        charging_lengths[0] = total_charging_time
    else:
        charging_lengths = (
            min_charging_interval
            + _sample_uniform_simplex(n_charge) * max(charge_slack, 0.0)
        )

    if n_pause == 0:
        pause_lengths = np.empty(0, dtype=np.float64)
    else:
        pause_lengths = (
            min_non_charging_interval
            + _sample_uniform_simplex(n_pause) * max(pause_slack, 0.0)
        )

    count = n_charge + n_pause
    starts = np.empty(count, dtype=np.float64)
    ends = np.empty(count, dtype=np.float64)
    idx = 0
    cursor = charging_start
    for k in range(n_charge):
        c_len = charging_lengths[k]
        starts[idx] = cursor
        ends[idx] = cursor + c_len
        cursor = cursor + c_len
        idx += 1
        if k < n_pause:
            p_len = pause_lengths[k]
            starts[idx] = cursor
            ends[idx] = cursor + p_len
            cursor = cursor + p_len
            idx += 1
    return starts, ends, idx


@njit
def _construct_operation(
    schedule: np.ndarray,
    vehicle_index: int,
    charger_idx: int,
    lam: float,
    Instance: tuple,
) -> np.ndarray:
    (
        arrivals,
        departures,
        requested_energy,
        full_r_ij,
        powers,
        P_max,
        t_min,
        t_max,
        n,
        m,
        max_preemptions,
        min_charging_interval,
        min_non_charging_interval,
        distance_between_two_sessions,
        min_delivered_fraction,
    ) = Instance

    schedule = np.asarray(schedule, dtype=np.float64).copy()
    schedule[vehicle_index, :] = 0.0

    delivered = _sample_delivered_energy(
        requested_energy[vehicle_index],
        powers[charger_idx - 1],
        min_delivered_fraction,
        min_charging_interval,
    )
    total_charging_time = delivered / powers[charger_idx - 1]

    free_s, free_e, free_count = _get_charger_free_intervals(
        schedule,
        charger_idx,
        distance_between_two_sessions,
    )

    min_start = float(arrivals[vehicle_index])
    elig_s = np.empty(free_count, dtype=np.float64)
    elig_e = np.empty(free_count, dtype=np.float64)
    elig_count = 0
    for k in range(free_count):
        s = free_s[k]
        e = free_e[k]
        eff_s = max(s, min_start)
        if e == INFINITY or (eff_s < e and e - eff_s >= total_charging_time - EPS):
            elig_s[elig_count] = eff_s
            elig_e[elig_count] = e
            elig_count += 1

    if elig_count == 0:
        eff_start = min_start
        usable_length = total_charging_time + max(0.0, np.random.exponential(1.0 / max(lam, EPS)))
    else:
        chosen = np.random.randint(elig_count)
        eff_start = elig_s[chosen]
        raw_end = elig_e[chosen]
        if raw_end == INFINITY:
            usable_length = total_charging_time + np.random.exponential(1.0 / max(lam, EPS))
        else:
            usable_length = raw_end - eff_start

    n_preemptions = _sample_n_preemptions(
        total_charging_time,
        usable_length,
        max_preemptions,
        min_charging_interval,
        min_non_charging_interval,
    )

    min_total_window = total_charging_time + n_preemptions * min_non_charging_interval
    extra_window = max(0.0, usable_length - min_total_window)
    total_window = min_total_window + (
        np.random.uniform(0.0, extra_window) if extra_window > EPS else 0.0
    )

    lo = eff_start
    hi = eff_start + max(usable_length - total_window, 0.0)
    charging_start = np.random.uniform(lo, hi) if hi > lo + EPS else lo

    starts, ends, count = _sample_preemptive_operation(
        charging_start,
        total_window,
        total_charging_time,
        n_preemptions,
        min_charging_interval,
        min_non_charging_interval,
    )

    _fill_row_from_intervals(schedule[vehicle_index], charger_idx, delivered, starts, ends, count)
    return schedule


# ============================================================================
# OBJECTIVES
# ============================================================================

@njit
def _peak_load(schedule: np.ndarray, powers: np.ndarray) -> float:
    n = schedule.shape[0]
    max_pairs = (schedule.shape[1] - 2) // 2
    event_t = np.empty(n * max_pairs * 2, dtype=np.float64)
    event_d = np.empty(n * max_pairs * 2, dtype=np.float64)
    ev_count = 0
    for i in range(n):
        charger = int(round(schedule[i, 0]))
        if charger <= 0:
            continue
        power = float(powers[charger - 1])
        starts, ends, count = _extract_intervals_arr(schedule[i])
        for idx in range(count):
            if idx % 2 == 0:
                event_t[ev_count] = starts[idx]
                event_d[ev_count] = power
                ev_count += 1
                event_t[ev_count] = ends[idx] - EPS
                event_d[ev_count] = -power
                ev_count += 1

    if ev_count == 0:
        return 0.0

    order = np.argsort(event_t[:ev_count])
    current = 0.0
    peak = 0.0
    for k in range(ev_count):
        current += event_d[order[k]]
        if current > peak:
            peak = current
    return float(peak)


@njit
def _total_tardiness(schedule: np.ndarray, departures: np.ndarray) -> float:
    total = 0.0
    for i in range(schedule.shape[0]):
        total += max(_row_end_time(schedule[i]) - departures[i], 0.0)
    return float(total)


@njit
def _undelivered_energy(schedule: np.ndarray, requested_energy: np.ndarray) -> float:
    """Total energy shortfall (requested - delivered), MINIMISED -- the
    complement of the previously-maximised total delivered energy, so every
    objective follows methods_comparison's minimisation convention with no obj_signs
    flip. (delivered energy is a property of the PLANNED schedule and is not
    affected by the stochastic MC realisation, same as before.)"""
    return float(np.sum(requested_energy) - np.sum(schedule[:, 1]))


@njit
def _sample_lognormal(mean: float, variance_factor: float) -> float:
    if mean <= EPS:
        return 0.0
    variance = (variance_factor * mean) ** 2
    mu = np.log((mean ** 2) / np.sqrt(mean ** 2 + variance))
    sigma2 = np.log(1.0 + variance / (mean ** 2))
    return float(np.random.lognormal(mu, np.sqrt(sigma2)))


@njit
def _realize_schedule_once(
    schedule: np.ndarray,
    variance_factor: float,
    Instance: tuple,
) -> np.ndarray:
    (
        arrivals,
        departures,
        requested_energy,
        full_r_ij,
        powers,
        P_max,
        t_min,
        t_max,
        n,
        m,
        max_preemptions,
        min_charging_interval,
        min_non_charging_interval,
        distance_between_two_sessions,
        min_delivered_fraction,
    ) = Instance

    realized = np.zeros_like(schedule)
    realized[:, 0] = schedule[:, 0]
    realized[:, 1] = schedule[:, 1]

    for charger_idx in range(1, m + 1):
        veh_idx = np.empty(n, dtype=np.int64)
        veh_start = np.empty(n, dtype=np.float64)
        assigned_count = 0
        for i in range(n):
            if int(round(schedule[i, 0])) == charger_idx:
                veh_idx[assigned_count] = i
                veh_start[assigned_count] = _row_start_time(schedule[i])
                assigned_count += 1
        order = np.argsort(veh_start[:assigned_count])

        prev_vehicle_end = 0.0
        has_prev_vehicle_end = False
        for a in range(assigned_count):
            i = veh_idx[order[a]]
            starts, ends, count = _extract_intervals_arr(schedule[i])
            n_charging = (count + 1) // 2

            realized_starts = np.empty(n_charging, dtype=np.float64)
            realized_ends = np.empty(n_charging, dtype=np.float64)
            prev_end_within_vehicle = 0.0

            for k in range(n_charging):
                ps = starts[2 * k]
                pe = ends[2 * k]
                planned_mean = pe - ps
                realized_length = _sample_lognormal(planned_mean, variance_factor)

                if k == 0:
                    if not has_prev_vehicle_end:
                        rs = ps
                    else:
                        rs = max(ps, prev_vehicle_end + distance_between_two_sessions)
                else:
                    rs = max(ps, prev_end_within_vehicle + min_non_charging_interval)

                re = rs + realized_length
                realized_starts[k] = rs
                realized_ends[k] = re
                prev_end_within_vehicle = re

            prev_vehicle_end = realized_ends[n_charging - 1]
            has_prev_vehicle_end = True

            out_starts = np.empty(count, dtype=np.float64)
            out_ends = np.empty(count, dtype=np.float64)
            idx = 0
            for k in range(n_charging):
                out_starts[idx] = realized_starts[k]
                out_ends[idx] = realized_ends[k]
                idx += 1
                if k < n_charging - 1:
                    out_starts[idx] = realized_ends[k]
                    out_ends[idx] = realized_starts[k + 1]
                    idx += 1

            _fill_row_from_intervals(
                realized[i],
                charger_idx,
                schedule[i, 1],
                out_starts,
                out_ends,
                idx,
            )

    return realized


@njit
def _objective_deterministic(schedule: np.ndarray, Instance: tuple) -> np.ndarray:
    arrivals, departures, requested_energy = Instance[0], Instance[1], Instance[2]
    return np.array(
        [
            _peak_load(schedule, Instance[4]),
            _total_tardiness(schedule, departures),
            _undelivered_energy(schedule, requested_energy),
        ],
        dtype=np.float64,
    )


@njit
def _objective_mc(
    schedule: np.ndarray,
    variance_factor: float,
    num_of_replications: int,
    Instance: tuple,
) -> np.ndarray:
    schedule = np.asarray(schedule, dtype=np.float64).copy()

    if num_of_replications <= 0:
        return _objective_deterministic(schedule, Instance)

    x = np.zeros(int(num_of_replications), dtype=np.float64)
    y = np.zeros(int(num_of_replications), dtype=np.float64)

    for k in range(int(num_of_replications)):
        realized = _realize_schedule_once(schedule, variance_factor, Instance)
        x[k] = _peak_load(realized, Instance[4])
        y[k] = _total_tardiness(realized, Instance[1])

    return np.array(
        [
            float(np.mean(x)),
            float(np.mean(y)),
            _undelivered_energy(schedule, Instance[2]),
        ],
        dtype=np.float64,
    )


@njit
def _generate_schedule(lam: float, Instance: tuple) -> np.ndarray:
    n = Instance[8]
    m = Instance[9]
    max_preemptions = Instance[10]

    schedule = np.zeros((n, _schedule_width(max_preemptions)), dtype=np.float64)
    vehicles = np.arange(n)
    np.random.shuffle(vehicles)

    for idx in range(n):
        v = vehicles[idx]
        charger_idx = int(np.random.randint(m)) + 1
        schedule = _construct_operation(schedule, int(v), charger_idx, float(lam), Instance)
    return schedule


@njit
def _crossover_schedules(parents_array: np.ndarray, lam: float, Instance: tuple) -> np.ndarray:
    n = Instance[8]
    m = Instance[9]
    max_preemptions = Instance[10]
    distance_between_two_sessions = Instance[13]

    parents_array = np.asarray(parents_array, dtype=np.float64)
    N = parents_array.shape[0]
    child = np.zeros((n, _schedule_width(max_preemptions)), dtype=np.float64)

    vehicles = np.arange(n)
    np.random.shuffle(vehicles)
    placed = np.zeros(n, dtype=np.int8)
    not_placed = np.empty(n, dtype=np.int64)
    not_placed_count = 0

    for idx in range(n):
        i = vehicles[idx]
        if idx == 0:
            s = int(np.random.randint(N))
            child[i] = parents_array[s, i].copy()
            placed[i] = 1
            continue

        feasible_sources = np.empty(N, dtype=np.int64)
        feasible_count = 0
        for s in range(N):
            cand = parents_array[s, i]
            cand_charger = int(round(cand[0]))
            cand_start = _row_start_time(cand)
            cand_end = _row_end_time(cand)

            if cand_charger <= 0 or cand_end <= 0.0:
                continue

            ok = True
            for j in range(n):
                if placed[j] == 0:
                    continue
                if int(round(child[j, 0])) != cand_charger:
                    continue

                placed_start = _row_start_time(child[j])
                placed_end = _row_end_time(child[j])

                no_overlap = (
                    cand_end + distance_between_two_sessions <= placed_start + EPS
                    or placed_end + distance_between_two_sessions <= cand_start + EPS
                )
                if not no_overlap:
                    ok = False
                    break

            if ok:
                feasible_sources[feasible_count] = s
                feasible_count += 1

        if feasible_count > 0:
            s = feasible_sources[np.random.randint(feasible_count)]
            child[i] = parents_array[s, i].copy()
            placed[i] = 1
        else:
            not_placed[not_placed_count] = i
            not_placed_count += 1

    for k in range(not_placed_count):
        i = not_placed[k]
        charger_idx = int(np.random.randint(m)) + 1
        child = _construct_operation(child, i, charger_idx, float(lam), Instance)

    return child


@njit
def _perturb_schedule(schedule: np.ndarray, perc_perturb: float, lam: float, Instance: tuple) -> np.ndarray:
    n = Instance[8]
    m = Instance[9]

    schedule = np.asarray(schedule, dtype=np.float64).copy()
    vehicles = np.arange(n)
    np.random.shuffle(vehicles)

    n_changes = int(np.ceil(float(perc_perturb) * n))
    n_changes = max(0, min(n_changes, n))

    for k in range(n_changes):
        i = int(vehicles[k])
        charger_idx = int(np.random.randint(m)) + 1
        schedule = _construct_operation(schedule, i, charger_idx, float(lam), Instance)

    return schedule


# ═══════════════════════════════════════════════════════════════════════════════
# OBJECTIVES
# ═══════════════════════════════════════════════════════════════════════════════

def deterministic_objective(schedule: np.ndarray, inst: Chapter5Instance) -> np.ndarray:
    """(peak load, total tardiness, undelivered energy) of the plan itself
    (no duration noise)."""
    return _objective_deterministic(np.asarray(schedule, dtype=np.float64), inst.arrays)


def objective_mc(schedule: np.ndarray, n_replications: int, inst: Chapter5Instance) -> np.ndarray:
    """(E[peak load], E[total tardiness]) as sample means over n_replications
    realizations, and the exact undelivered energy; n_replications <= 0 gives
    the deterministic objective."""
    return _objective_mc(np.asarray(schedule, dtype=np.float64), float(inst.variance_factor),
                         int(n_replications), inst.arrays)


def realize(schedule: np.ndarray, inst: Chapter5Instance) -> np.ndarray:
    """One realization of a planned schedule (draws from Numba's RNG)."""
    return _realize_schedule_once(np.asarray(schedule, dtype=np.float64), float(inst.variance_factor), inst.arrays)


def score_realized(realized: np.ndarray, inst: Chapter5Instance) -> np.ndarray:
    """Objectives of an already realized schedule (no further realization)."""
    return _objective_deterministic(np.asarray(realized, dtype=np.float64), inst.arrays)


def undelivered_energy(schedule: np.ndarray, inst: Chapter5Instance) -> float:
    return float(_undelivered_energy(np.asarray(schedule, dtype=np.float64), inst.e))


# ═══════════════════════════════════════════════════════════════════════════════
# SOLVER-FACING OPERATORS   problem_params = (perc_perturb, lam, n_replications, instance)
# ═══════════════════════════════════════════════════════════════════════════════

def build_problem_params(params: Dict[str, Any], inst: Chapter5Instance) -> tuple:
    return (float(params.get("perc_perturb", P.UNTUNED_PERC_PERTURB)),
            float(params.get("lam", P.FREE_REGION_RATE)),
            int(params.get("num_of_replications", P.LABEL_REPLICATIONS)),
            inst)


def Generate_Solution(problem_params: tuple) -> np.ndarray:
    return _generate_schedule(float(problem_params[1]), problem_params[3].arrays)


def Perturb(schedule: np.ndarray, problem_params: tuple) -> np.ndarray:
    return _perturb_schedule(np.asarray(schedule, dtype=np.float64), float(problem_params[0]),
                             float(problem_params[1]), problem_params[3].arrays)


def Crossover(parents_array: np.ndarray, problem_params: tuple) -> np.ndarray:
    return _crossover_schedules(np.asarray(parents_array, dtype=np.float64), float(problem_params[1]),
                                problem_params[3].arrays)


def objective_function(schedule: np.ndarray, problem_params: tuple) -> np.ndarray:
    """The Monte Carlo objective with problem_params' replication count."""
    return objective_mc(schedule, problem_params[2], problem_params[3])


def solution_shape(inst: Chapter5Instance) -> tuple:
    return (inst.n, _schedule_width(int(inst.arrays[10])))


# ═══════════════════════════════════════════════════════════════════════════════
# SURROGATES: FEATURE MAP AND DEPLOYED MODELS
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "det_peak_load", "det_tardiness", "det_undelivered",
    "n_vehicles", "n_chargers", "P_max",
    "total_delivered_energy", "mean_delivered_fraction", "delivered_fraction_std",
    "mean_preemptions", "max_preemptions_used", "preemptions_std",
    "mean_session_length", "max_session_length", "mean_gap_length",
    "charger_load_std", "charger_load_max",
    "horizon_utilisation",
    "energy_heterogeneity", "arrival_heterogeneity", "charger_power_heterogeneity",
]


def feature_fn(solution: np.ndarray, inst: Chapter5Instance) -> np.ndarray:
    """phi(S): the noise-free objectives of the plan, structural statistics of
    the schedule and their dispersion, and instance heterogeneity -- no Monte
    Carlo draw. Generation, algorithm and front rank are deliberately excluded."""
    solution = np.asarray(solution, dtype=np.float64)
    (arrivals, departures, requested_energy, full_r_ij, powers, P_max, t_min, t_max,
     n, m, max_preemptions, msl, mpl, md, min_frac) = inst.arrays
    det_peak, det_tardiness, det_undelivered = deterministic_objective(solution, inst)
    delivered = solution[:, 1]
    total_delivered = float(np.sum(delivered))
    fraction = delivered / np.maximum(requested_energy, 1e-9)
    preemptions = np.empty(n, dtype=np.float64)
    sessions, gaps = [], []
    for i in range(n):
        starts, ends, count = _extract_intervals_arr(solution[i])
        preemptions[i] = (count - 1) / 2.0
        for k in range(count):
            (sessions if k % 2 == 0 else gaps).append(float(ends[k] - starts[k]))
    counts = np.zeros(m)
    for i in range(n):
        c = int(round(solution[i, 0]))
        if 1 <= c <= m:
            counts[c - 1] += 1
    span = max(t_max - t_min, 1e-9)
    return np.array([
        det_peak, det_tardiness, det_undelivered,
        float(n), float(m), float(P_max),
        total_delivered, float(np.mean(fraction)), float(np.std(fraction)),
        float(np.mean(preemptions)), float(np.max(preemptions)), float(np.std(preemptions)),
        float(np.mean(sessions)) if sessions else 0.0, float(np.max(sessions)) if sessions else 0.0,
        float(np.mean(gaps)) if gaps else 0.0,
        float(np.std(counts)), float(np.max(counts)),
        total_delivered / (span * float(np.sum(powers)) + 1e-9),
        float(np.std(requested_energy)), float(np.std(arrivals)), float(np.std(powers)),
    ], dtype=np.float64)


_SURROGATES: Dict[str, Any] = {}


def surrogate_paths() -> Dict[str, Path]:
    return {name: CHAPTER5_DIR / P.SURROGATE_DIR / fname for name, fname in P.SURROGATE_FILES.items()}


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
    card_path = CHAPTER5_DIR / P.SURROGATE_DIR / "model_card.json"
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
    """The deployed regressors (one per learned objective), loaded once per process."""
    if _SURROGATES:
        return _SURROGATES
    missing = [str(p) for p in surrogate_paths().values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Chapter 5 surrogate models not found: " + ", ".join(missing) + ". Train them with "
            "surrogate_trainer (python main.py), then copy "
            "everything in runs/surrogate_chapter5_<key>/surrogate_export/best_models/ (the .joblib files and "
            "model_card.json) into problems/chapter5/"
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
    """Search objective of the surrogate-assisted search: the two Monte Carlo
    objectives predicted by the deployed surrogates, the undelivered energy
    computed exactly."""
    models = load_surrogates()
    inst = problem_params[3]
    phi = feature_fn(schedule, inst).reshape(1, -1)
    return np.array([float(models["Peak load (kW)"].predict(phi)[0]),
                     float(models["Total tardiness (h)"].predict(phi)[0]),
                     undelivered_energy(schedule, inst)], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# INDEPENDENT VERIFIERS (plain Python, not the operators' code)
# ═══════════════════════════════════════════════════════════════════════════════

def check_schedule_feasibility(schedule, inst: Chapter5Instance, label="schedule", eps=None):
    """Every constraint of a planned schedule; raises AssertionError describing
    the first violation."""
    (a, d, e, r_ij, w, P_max, t_min, t_max, n, m, pmax, msl, mpl, md, min_frac) = inst.arrays
    eps = P.VERIFIER_EPS if eps is None else eps
    schedule = np.asarray(schedule, dtype=np.float64)
    assert schedule.shape == solution_shape(inst), f"[{label}] SHAPE FAIL {schedule.shape}"
    for i in range(n):
        charger = int(round(schedule[i, 0]))
        delivered = float(schedule[i, 1])
        assert 1 <= charger <= m, f"[{label}] CHARGER FAIL vehicle {i} on charger {charger}"
        assert min_frac * e[i] - 1e-6 <= delivered <= e[i] + 1e-6, \
            f"[{label}] ENERGY FAIL vehicle {i}: delivered {delivered} not in [{min_frac * e[i]}, {e[i]}]"
        iv = _extract_active_intervals(schedule[i], eps)
        assert iv, f"[{label}] EMPTY FAIL vehicle {i}"
        assert len(iv) % 2 == 1, f"[{label}] PARITY FAIL vehicle {i}"
        assert len(iv) <= 2 * pmax + 1, f"[{label}] COUNT FAIL vehicle {i}: {len(iv)} intervals"
        assert iv[0][0] >= a[i] - eps, f"[{label}] ARRIVAL FAIL vehicle {i} starts {iv[0][0]} < {a[i]}"
        total = 0.0
        for k, (s, t) in enumerate(iv):
            assert np.isfinite(s) and np.isfinite(t) and t >= s + eps, f"[{label}] LENGTH FAIL vehicle {i} interval {k}"
            if k > 0:
                assert abs(s - iv[k - 1][1]) <= 1e-6, f"[{label}] CONTIGUITY FAIL vehicle {i} interval {k}"
                if k % 2 == 1:
                    assert t - s >= mpl - 1e-6, f"[{label}] PAUSE FAIL vehicle {i} interval {k} shorter than m^(pl)"
            if k % 2 == 0:
                total += t - s
                # a lone interval may be shorter than m^(sl) only when the requested
                # energy itself takes less than m^(sl) (delivered exactly)
                if t - s < msl - 1e-6:
                    assert len(iv) == 1 and abs(delivered - e[i]) <= 1e-6, \
                        f"[{label}] CHARGE-LENGTH FAIL vehicle {i} interval {k} shorter than m^(sl)"
        assert abs(total - delivered / w[charger - 1]) <= 1e-5, \
            f"[{label}] CHARGE-TIME FAIL vehicle {i}: {total} != delivered/power {delivered / w[charger - 1]}"
        assert not np.any(np.abs(schedule[i, 2 + 2 * len(iv):]) > eps), f"[{label}] TAIL FAIL vehicle {i}"
    for c in range(1, m + 1):
        occ = sorted((_row_start_time(schedule[i]), _row_end_time(schedule[i]), i)
                     for i in range(n) if int(round(schedule[i, 0])) == c)
        for (s1, e1, i1), (s2, e2, i2) in zip(occ, occ[1:]):
            assert e1 + md <= s2 + eps, f"[{label}] OVERLAP FAIL charger {c}: vehicles {i1}, {i2}"
