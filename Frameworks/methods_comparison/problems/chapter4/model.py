"""
model.py -- Chapter 4 of the thesis (chapter10.tex of the manuscript):
bi-objective EV charging scheduling under stochastic charging durations, Monte
Carlo lognormal simulation.

The operators, the objectives, the instance sampling and the verifiers are the
code of the thesis, unchanged. What is adapted to the current frameworks is the
last section (the instance suite as a function, the interface of methods_comparison and of
det_vs_stoch) and the data file (acn_sessions.pkl: the same sessions as
requests_data.pkl). The parameters are in parameters.py. This folder is
identical in every framework that uses it.

Objectives   (expected peak load kW, expected total tardiness), Monte Carlo over
             the lognormal charging durations with N_r replications.
"""

import math
import pickle
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from numba import njit
from scipy.stats import gaussian_kde

from . import parameters as P
from .parameters import *  # noqa: F401,F403  (the constants, under their names in the thesis code)

PROBLEM_NAME = "chapter4"
CODE = "chapter4"
CHAPTER4_DIR = Path(__file__).resolve().parent
OBJECTIVE_NAMES = ["Peak load (kW)", "Total tardiness"]
REQUESTS_DATA = None


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL & INSTANCE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def _load_requests_data() -> np.ndarray:
    global REQUESTS_DATA
    if REQUESTS_DATA is not None:
        return REQUESTS_DATA

    candidates = []
    if "__file__" in globals():
        candidates.extend([
            Path(__file__).with_name("acn_sessions.pkl"),
            Path(__file__).resolve().parent / "acn_sessions.pkl",
        ])

    candidates.extend([
        Path.cwd() / "acn_sessions.pkl",
        Path.cwd() / "problems" / "acn_sessions.pkl",
    ])

    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                REQUESTS_DATA = pickle.load(f)
            return REQUESTS_DATA

    raise FileNotFoundError(
        "acn_sessions.pkl not found next to the dependency file or in the project root."
    )


def _load_or_mock_requests_hours(seed: int = 123) -> np.ndarray:
    global REQUESTS_DATA
    if REQUESTS_DATA is not None:
        return np.asarray(REQUESTS_DATA, dtype=np.float64).copy()

    REQUESTS_DATA = _load_requests_data()
    return np.asarray(REQUESTS_DATA, dtype=np.float64).copy()


def _sample_from_requests(n: int, requests: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    arrivals = requests[:, 0]
    kde = gaussian_kde(arrivals)

    chosen_indices = []
    k = 0
    max_attempts = n * 200
    attempts = 0

    while k < n and attempts < max_attempts:
        sample = kde.resample(10, seed=rng.integers(int(1e9)))[0]
        distances = np.abs(arrivals[:, None] - sample[None, :])
        idxs = np.argmin(distances, axis=0)
        for i in idxs:
            if i not in chosen_indices:
                chosen_indices.append(int(i))
                k += 1
                if k == n:
                    break
        attempts += 10

    if k < n:
        raise RuntimeError(f"Could not find {n} distinct indices after {max_attempts} attempts.")

    return np.array(chosen_indices)


@njit
def compute_rij(required_energy, powers):
    n, m = len(required_energy), len(powers)
    r = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            r[i, j] = required_energy[i] / powers[j]
    return r


def simulate_instance(
    requests=None,
    n=20,
    m=10,
    chargers_powers=None,
    distance_between_two_sessions=0.1,
    seed=None,
):
    if requests is None:
        requests = _load_requests_data()

    if chargers_powers is None:
        chargers_powers = DEFAULT_CHARGING_POWERS
    chargers_powers = np.asarray(chargers_powers, dtype=np.float64)

    rng = np.random.default_rng(seed)
    idx = _sample_from_requests(n, requests, rng)
    data = requests[idx]

    arrivals_raw = np.asarray(data[:, 0], dtype=np.float64)
    departures_raw = np.asarray(data[:, 1], dtype=np.float64)
    required_energy = np.asarray(data[:, 2], dtype=np.float64)

    departures_raw = np.maximum(departures_raw, arrivals_raw)

    arrivals = arrivals_raw
    departures = departures_raw

    powers = np.asarray(rng.choice(chargers_powers, size=m, replace=True), dtype=np.float64)
    r_ij = compute_rij(required_energy, powers)
    max_r = float(np.max(r_ij))
    P_max = 0.7 * np.sum(powers)
    t_min = float(np.min(arrivals))
    t_max = float(np.max(departures))

    Instance = (
        arrivals, departures, required_energy, r_ij, powers,
        P_max, t_min, t_max, n, m, max_r, float(distance_between_two_sessions),
    )
    return Instance


# ====================================================================
# NUMBA HELPERS
# ====================================================================

@njit
def find_availability(j, schedule, distance_between_two_sessions):
    sub_schedule = schedule[schedule[:, 0] == j + 1]
    sub_schedule = sub_schedule[np.argsort(sub_schedule[:, 1])]
    N = len(sub_schedule)
    availability = np.zeros((N + 1, 2), dtype=np.float64)
    if N > 0:
        availability[0] = np.array([0.0, sub_schedule[0, 1] - distance_between_two_sessions])
        for k in range(1, N):
            availability[k] = np.array([
                sub_schedule[k - 1, 2] + distance_between_two_sessions,
                sub_schedule[k, 1] - distance_between_two_sessions,
            ])
        availability[N] = np.array([sub_schedule[N - 1, 2] + distance_between_two_sessions, INFINITY])
    else:
        availability[0] = np.array([0.0, INFINITY])
    return availability


@njit
def find_candidate_charging_periods(i, j, availability, Instance):
    arrivals, departures, requested_energies, r_ij, powers, P_max, t_min, t_max, n, m, max_r, distance_between_two_sessions = Instance
    candidate_charging_periods = []
    N = len(availability)
    for k in range(N):
        av_s = availability[k, 0]
        av_e = availability[k, 1]
        if av_e - max(av_s, arrivals[i]) >= r_ij[i, j]:
            candidate_charging_periods.append(k)
    return np.array(candidate_charging_periods)


@njit
def allocate_charger(schedule, i, j, Instance):
    arrivals, departures, requested_energies, r_ij, powers, P_max, t_min, t_max, n, m, max_r, distance_between_two_sessions = Instance
    schedule = schedule.copy()

    availability = find_availability(j, schedule, distance_between_two_sessions)
    candidate_charging_periods = find_candidate_charging_periods(i, j, availability, Instance)

    if len(candidate_charging_periods) == 0:
        ts = max(0.0, arrivals[i])
        te = ts + r_ij[i, j]
        schedule[i] = [j + 1, ts, te]
        return schedule

    k = candidate_charging_periods[0]
    av_s = availability[k, 0]
    av_e = availability[k, 1]
    lower = max(av_s, arrivals[i])

    if av_e < INFINITY:
        upper = av_e - r_ij[i, j]
        if upper <= lower + 1e-12:
            ts = lower
        else:
            ts = np.random.uniform(lower, upper)
    else:
        ts = lower + np.abs(np.random.randn())

    te = ts + r_ij[i, j]
    schedule[i] = [j + 1, ts, te]
    return schedule


@njit
def generate_schedule(Instance):
    arrivals, departures, requested_energies, r_ij, powers, P_max, t_min, t_max, n, m, max_r, distance_between_two_sessions = Instance
    schedule = np.zeros((n, 3), dtype=np.float64)
    vehicles_idx = np.arange(n)
    np.random.shuffle(vehicles_idx)
    for idx in range(n):
        i = vehicles_idx[idx]
        j = np.random.randint(m)
        schedule = allocate_charger(schedule, i, j, Instance)
    return schedule


@njit
def peak_load(schedule, powers):
    n = len(schedule)
    events = np.zeros((2 * n, 2), dtype=np.float64)
    for i in range(n):
        j = int(schedule[i, 0]) - 1
        if j >= 0:
            events[2 * i, 0] = schedule[i, 1]
            events[2 * i, 1] = powers[j]
            events[2 * i + 1, 0] = schedule[i, 2]
            events[2 * i + 1, 1] = -powers[j]
    sorted_indexes = np.argsort(events[:, 0])
    events = events[sorted_indexes]
    cumsum = np.cumsum(events[:, 1])
    return np.max(cumsum)


@njit
def total_tardiness(schedule, departures):
    x = np.maximum(schedule[:, 2] - departures, 0.0)
    return np.sum(x)


@njit
def _objective_deterministic(schedule, Instance):
    arrivals, departures, requested_energies, r_ij, powers, P_max, t_min, t_max, n, m, max_r, distance_between_two_sessions = Instance
    return np.array([
        peak_load(schedule, powers),
        total_tardiness(schedule, departures),
    ])


@njit
def _realize_schedule_once_lognormal(schedule, Instance, variance_factor):
    """Draws ONE Monte Carlo realisation of `schedule`: independently
    resamples each vehicle's charging duration from a lognormal centred on
    its planned duration r_ij, then re-sequences same-charger vehicles
    (in their planned start-time order) so later vehicles are pushed back by
    distance_between_two_sessions where needed. This is the exact
    per-replication step _objective_MC_lognormal averages over; factored out
    so run_end_to_end_tests can call it directly to check every single
    realisation, not just the aggregated mean objective."""
    arrivals, departures, requested_energies, r_ij, powers, P_max, t_min, t_max, n, m, max_r, distance_between_two_sessions = Instance

    charging_durations = np.zeros(n)
    for i in range(n):
        j = int(schedule[i, 0]) - 1
        mean = r_ij[i, j]
        variance = (variance_factor * mean) ** 2
        mu = np.log((mean ** 2) / np.sqrt(mean ** 2 + variance))
        sigma2 = np.log(1.0 + variance / (mean ** 2))
        charging_durations[i] = np.random.lognormal(mu, np.sqrt(sigma2))

    updated_schedule = schedule.copy()
    for jj in range(m):
        mask = schedule[:, 0] == jj + 1
        indexes = np.arange(n)[mask]
        order = np.argsort(schedule[indexes, 1])
        indexes = indexes[order]

        N = len(indexes)
        if N > 0:
            updated_schedule[indexes[0], 2] = updated_schedule[indexes[0], 1] + charging_durations[indexes[0]]
        if N >= 2:
            for ii in range(1, N):
                updated_schedule[indexes[ii], 1] = max(
                    updated_schedule[indexes[ii], 1],
                    updated_schedule[indexes[ii - 1], 2] + distance_between_two_sessions,
                )
                updated_schedule[indexes[ii], 2] = updated_schedule[indexes[ii], 1] + charging_durations[indexes[ii]]

    return updated_schedule, charging_durations


@njit
def _objective_MC_lognormal(schedule, Instance, variance_factor, num_of_replications):
    arrivals, departures, requested_energies, r_ij, powers, P_max, t_min, t_max, n, m, max_r, distance_between_two_sessions = Instance
    schedule = schedule.copy()

    if num_of_replications <= 0:
        return _objective_deterministic(schedule, Instance)

    x = np.zeros(num_of_replications)
    y = np.zeros(num_of_replications)

    for k in range(num_of_replications):
        updated_schedule, _durations = _realize_schedule_once_lognormal(schedule, Instance, variance_factor)
        x[k] = peak_load(updated_schedule, powers)
        y[k] = total_tardiness(updated_schedule, departures)

    mean_peak_load = np.mean(x)
    mean_total_tardiness = np.mean(y)
    return np.array([mean_peak_load, mean_total_tardiness])


@njit
def _generalized_crossover(solutions, Instance):
    arrivals, departures, requested_energies, r_ij, powers, P_max, t_min, t_max, n, m, max_r, distance_between_two_sessions = Instance
    N = len(solutions)

    new_solution = np.zeros((n, 3), dtype=np.float64)
    vehicles_indexes = np.arange(n)
    np.random.shuffle(vehicles_indexes)

    placed = np.zeros(n, dtype=np.int8)
    not_placed = np.zeros(n, dtype=np.int64)
    n_not_placed = 0

    for vi in range(n):
        i = vehicles_indexes[vi]

        placable = np.zeros(N, dtype=np.int8)
        for s in range(N):
            candidate = solutions[s][i]
            charger = int(candidate[0])

            if charger < 1 or charger > m:
                continue

            cand_ts = candidate[1]
            cand_te = candidate[2]
            ok = True

            for jj in range(n):
                if placed[jj] == 0:
                    continue
                if int(new_solution[jj, 0]) != charger:
                    continue

                placed_ts = new_solution[jj, 1]
                placed_te = new_solution[jj, 2]
                separated = (
                    cand_te + distance_between_two_sessions <= placed_ts
                    or placed_te + distance_between_two_sessions <= cand_ts
                )
                if not separated:
                    ok = False
                    break

            if ok:
                placable[s] = 1

        count_placable = 0
        for s in range(N):
            if placable[s] == 1:
                count_placable += 1

        if count_placable > 0:
            chosen = np.random.randint(count_placable)
            count = 0
            for s in range(N):
                if placable[s] == 1:
                    if count == chosen:
                        new_solution[i] = solutions[s][i].copy()
                        placed[i] = 1
                        break
                    count += 1
        else:
            not_placed[n_not_placed] = i
            n_not_placed += 1

    for k in range(n_not_placed):
        i = not_placed[k]
        j = np.random.randint(m)
        new_solution = allocate_charger(new_solution, i, j, Instance)

    return new_solution


@njit
def _FN(schedule, Instance, perc_perturb):
    arrivals, departures, requested_energies, r_ij, powers, P_max, t_min, t_max, n, m, max_r, distance_between_two_sessions = Instance
    schedule = schedule.copy()

    candidate_vehicles = np.arange(n)
    np.random.shuffle(candidate_vehicles)
    number_of_changed = int(np.ceil(perc_perturb * n))
    if number_of_changed < 0:
        number_of_changed = 0
    if number_of_changed > n:
        number_of_changed = n

    for idx in range(number_of_changed):
        i = candidate_vehicles[idx]
        j = np.random.randint(m)
        schedule = allocate_charger(schedule, i, j, Instance)

    return schedule


# ====================================================================
# SOLVER-FACING FUNCTIONS
# ====================================================================

@njit
def objective_function(solution, problem_params):
    perc_perturb, lam, variance_factor, num_of_replications, Instance = problem_params
    return _objective_MC_lognormal(
        np.asarray(solution, dtype=np.float64),
        Instance,
        float(variance_factor),
        int(num_of_replications),
    )


@njit
def Generate_Solution(problem_params):
    _, _, _, _, Instance = problem_params
    return generate_schedule(Instance)


@njit
def Perturb(solution, problem_params):
    perc_perturb, _, _, _, Instance = problem_params
    return _FN(
        np.asarray(solution, dtype=np.float64),
        Instance,
        float(perc_perturb),
    )


@njit
def Crossover(parents_array, problem_params):
    _, _, _, _, Instance = problem_params
    return _generalized_crossover(
        np.asarray(parents_array, dtype=np.float64),
        Instance,
    )


def check_schedule_feasibility(schedule, Instance, label="schedule", eps=None):
    arrivals, departures, requested_energies, r_ij, powers, P_max, t_min, t_max, n, m, max_r, distance_between_two_sessions = Instance
    eps = VERIFIER_EPS if eps is None else eps

    schedule = np.asarray(schedule, dtype=np.float64)

    assert schedule.shape == (n, 3), (
        f"[{label}] SHAPE FAIL -- expected {(n, 3)}, got {schedule.shape}."
    )

    for i in range(n):
        charger = int(round(schedule[i, 0]))
        ts = float(schedule[i, 1])
        te = float(schedule[i, 2])

        if charger < 1 or charger > m:
            raise AssertionError(
                f"[{label}] CHARGER FAIL -- vehicle {i} charger {charger} outside [1,{m}]."
            )

        if not np.isfinite(ts) or not np.isfinite(te):
            raise AssertionError(
                f"[{label}] FINITE FAIL -- vehicle {i} has non-finite interval [{ts},{te}]."
            )

        if ts < -eps:
            raise AssertionError(
                f"[{label}] START FAIL -- vehicle {i} starts at {ts} < 0."
            )

        if ts < arrivals[i] - eps:
            raise AssertionError(
                f"[{label}] ARRIVAL FAIL -- vehicle {i} starts at {ts} before arrival {arrivals[i]}."
            )

        if te < ts - eps:
            raise AssertionError(
                f"[{label}] INTERVAL FAIL -- vehicle {i} has te={te} < ts={ts}."
            )

        expected_duration = float(r_ij[i, charger - 1])
        actual_duration = float(te - ts)
        if abs(actual_duration - expected_duration) > 1e-6:
            raise AssertionError(
                f"[{label}] DURATION FAIL -- vehicle {i} duration {actual_duration} "
                f"!= r_ij[{i},{charger - 1}]={expected_duration}."
            )

    for c in range(1, m + 1):
        assigned = []
        for i in range(n):
            if int(round(schedule[i, 0])) == c:
                assigned.append((float(schedule[i, 1]), float(schedule[i, 2]), i))
        assigned.sort(key=lambda x: x[0])

        for a in range(len(assigned)):
            s1, e1, i1 = assigned[a]
            for b in range(a + 1, len(assigned)):
                s2, e2, i2 = assigned[b]
                separated = (
                    e1 + distance_between_two_sessions <= s2 + eps
                    or e2 + distance_between_two_sessions <= s1 + eps
                )
                if not separated:
                    raise AssertionError(
                        f"[{label}] OVERLAP FAIL -- charger {c}: vehicles {i1},{i2} "
                        f"intervals [{s1},{e1}] and [{s2},{e2}] violate required distance "
                        f"{distance_between_two_sessions}."
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# THE INSTANCE SUITE AND THE INTERFACE OF THE FRAMEWORKS
# ═══════════════════════════════════════════════════════════════════════════════

def n_chargers(n: int) -> int:
    return math.ceil(n / P.VEHICLES_PER_CHARGER)


def suite_instance(inst_id: int) -> tuple:
    """Instance `inst_id` (1-based) of the suite (the same under both regimes of
    the deterministic-versus-stochastic comparison)."""
    i = inst_id - 1
    n = P.SIZE_CLASSES[i // P.INSTANCES_PER_CLASS]
    return simulate_instance(requests=_load_or_mock_requests_hours(seed=99), n=n, m=n_chargers(n),
                             seed=P.INSTANCE_SEED_BASE + i)


def generate_instances() -> List[tuple]:
    """The suite of the thesis: 20 instances, 5 per size, m = ceil(n / 3)."""
    return [suite_instance(k) for k in range(1, len(P.SIZE_CLASSES) * P.INSTANCES_PER_CLASS + 1)]


def instance_class(Instance: tuple) -> str:
    return P.CLASS_LABELS[P.SIZE_CLASSES.index(int(Instance[8]))]


def size_classes() -> Dict[str, List[int]]:
    """Ids (1-based, order of generate_instances) of every size class, without
    generating the instances."""
    out, k = {}, 0
    for label in P.CLASS_LABELS[:len(P.SIZE_CLASSES)]:
        out[label] = list(range(k + 1, k + 1 + P.INSTANCES_PER_CLASS))
        k += P.INSTANCES_PER_CLASS
    return out


def instance_features(Instance: tuple) -> Dict[str, float]:
    """Numeric descriptors of the instance (size n, ...)."""
    return {"n": float(Instance[8]), "m": float(Instance[9])}


def build_problem_params(params: Dict[str, Any], Instance: Any) -> tuple:
    perc_perturb = float(params.get("perc_perturb", DEFAULT_PERTURB))
    variance_factor = float(params.get("variance_factor", DEFAULT_VARIANCE_FACTOR))
    num_of_replications = int(params.get("num_of_replications", DEFAULT_NUM_REPLICATIONS))
    return (perc_perturb, DEFAULT_LAM, variance_factor, num_of_replications, Instance)


def solution_shape(Instance: tuple) -> tuple:
    return (int(Instance[8]), 3)


def evaluate_objective(instance: Any, solution: Any) -> np.ndarray:
    """The objectives of a schedule, in a single call to `objective_function`."""
    return objective_function(np.asarray(solution, dtype=np.float64), build_problem_params({}, instance))
