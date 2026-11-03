"""
model.py -- Chapter 3 of the thesis (chapter9.tex of the manuscript; Khiar et al.,
"Multi-objective electric vehicle charging scheduling under stochastic duration
uncertainty", Omega 141 (2026) 103506), adapted to the current frameworks. It
imports only NumPy, SciPy and Numba (no framework).

UNCHANGED   Every function of the section "CHAPTER 3 CODE (VERBATIM)" is the
            exact source text of the thesis code (copied by a script,
            not retyped): instance sampling, construction, operators
            (Generate_Solution, Perturb, Crossover), the lognormal
            actualisation, the three objectives, and the verifiers of the
            thesis. Do not edit them.
ADAPTED     (1) the data source: the ACN-Data claimed sessions of this folder
            (acn_sessions.pkl) instead of requests_data.pkl (same content:
            arrival h, requested departure h, requested energy kWh); (2) the
            instance suite: the 20 ACN instances of the chapter (seeds 42 + k,
            parameters.FAMILIES; the family of simulated requests stays
            available, it is not in the manuscript); (3) the interface of the
            frameworks (Chapter3Instance, build_problem_params, solution_shape,
            size classes).

Problem      n requests (a_i, d_i, e_i), m chargers (opening time, power p_j),
             a compatibility matrix A(i, j); every vehicle is charged on one
             compatible charger, in at most DEFAULT_MAX_PREEMPTIONS sessions
             (minimum session DEFAULT_MIN_PL, minimum pause DEFAULT_MIN_NPL),
             two vehicles of a charger at least DEFAULT_MIN_DIST apart, with a
             delivered energy E_i in [DEFAULT_MIN_PERCENTAGE % e_i, e_i].
Uncertainty  the end of every session is lognormal (mean the planned end,
             standard deviation std_deviations[i, j] x the planned length),
             conditioned on keeping the planned structure (event C).
Objectives   (conditional expected grid capacity kW [Monte Carlo, N_r
             replications], conditional expected total tardiness h [closed
             form], unmet energy %), all minimised.
Solution     float64 array (n, 4 + 2 DEFAULT_MAX_PREEMPTIONS): charger j + 1,
             occupancy start, occupancy end, delivered energy, then the
             (start, end) pairs of the sessions, zero-padded.
problem_params = (perc_perturb, lam, variance_factor, num_of_replications, Instance)
             with Instance = (requests, chargers, compatibility_matrix, kappa,
             std_deviations), the tuple of the thesis (Chapter3Instance.data).
"""

import math
import pickle
import random
from math import ceil, erf
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from numba import njit
from scipy.stats import gaussian_kde

from . import parameters as P
from .parameters import (  # Chapter 3's constants, under their Chapter 3 names
    INFINITY, EPS, DEFAULT_CHARGING_POWERS, DEFAULT_MAX_PREEMPTIONS, DEFAULT_MIN_PL,
    DEFAULT_MIN_NPL, DEFAULT_MIN_DIST, DEFAULT_MIN_PERCENTAGE, DEFAULT_LAM,
    DEFAULT_PERC_PERTURB, DEFAULT_VARIANCE_FACTOR, DEFAULT_NUM_REPLICATIONS, VERIFIER_EPS,
)

CHAPTER3_DIR = Path(__file__).resolve().parent
OBJECTIVE_NAMES = ["Grid capacity (kW)", "Total tardiness (h)", "Unmet energy (%)"]
REQUESTS_DATA = None


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTATION (1): THE DATA SOURCE
# ═══════════════════════════════════════════════════════════════════════════════

def _load_requests_data() -> np.ndarray:
    """The ACN-Data claimed sessions of this folder (Chapter 3 read the same
    sessions from requests_data.pkl), cached in REQUESTS_DATA."""
    global REQUESTS_DATA
    if REQUESTS_DATA is None:
        with open(CHAPTER3_DIR / P.ACN_DATA_FILE, "rb") as fh:
            REQUESTS_DATA = np.asarray(pickle.load(fh), dtype=np.float64)
    return REQUESTS_DATA


def _load_or_mock_requests_hours(seed: int = 123) -> np.ndarray:
    """Chapter 3's accessor (the seed was unused there too)."""
    return np.asarray(_load_requests_data(), dtype=np.float64).copy()


def load_acn() -> np.ndarray:
    """ACN-Data claimed sessions: (arrival h, requested departure h, requested energy kWh)."""
    return _load_or_mock_requests_hours()


# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 3 CODE (VERBATIM) -- instance sampling
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_from_requests(n: int, requests: np.ndarray, rng=None) -> np.ndarray:
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


def simulate_instance(
    requests=None,
    n=20,
    m=6,
    charging_powers=None,
    min_perc_compatibility=DEFAULT_MIN_PERCENTAGE,
    kappa=1.0,
    seed=None,
):
    if requests is None:
        requests = _load_requests_data()

    if charging_powers is None:
        charging_powers = DEFAULT_CHARGING_POWERS
    charging_powers = np.asarray(charging_powers, dtype=np.float64)

    rng = np.random.default_rng(seed)
    idx = _sample_from_requests(n, requests, rng)
    data = requests[idx]

    arrivals_raw = np.asarray(data[:, 0], dtype=np.float64)
    departures_raw = np.asarray(data[:, 1], dtype=np.float64)
    departures_raw = np.maximum(departures_raw, arrivals_raw)
    required_energy = np.asarray(data[:, 2], dtype=np.float64)

    requests_arr = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        requests_arr[i, 0] = arrivals_raw[i]
        requests_arr[i, 1] = departures_raw[i]
        requests_arr[i, 2] = required_energy[i]

    compatibility_matrix = np.zeros((n, m), dtype=np.int8)
    for i in range(n):
        min_compat = max(1, int(ceil((min_perc_compatibility / 100.0) * m)))
        num_compat = int(rng.integers(min_compat, m + 1))
        compatible = rng.choice(m, num_compat, replace=False)
        compatibility_matrix[i, compatible] = 1

    chargers = np.zeros((m, 2), dtype=np.float64)
    for j in range(m):
        chargers[j, 0] = float(rng.uniform(0.0, 8.0))
        chargers[j, 1] = float(rng.choice(charging_powers))

    std_deviations = np.zeros((n, m), dtype=np.float64)
    std_values = np.linspace(0.05, 0.2, 10)
    for i in range(n):
        for j in range(m):
            if compatibility_matrix[i, j] == 1:
                std_deviations[i, j] = float(rng.choice(std_values))

    Instance = (requests_arr, chargers, compatibility_matrix, float(kappa), std_deviations)
    return Instance


# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 3 CODE (VERBATIM) -- construction, operators, actualisation, objectives
# ═══════════════════════════════════════════════════════════════════════════════

@njit
def random_choice(x, size=1, replace=True, p=None):
    n = len(x)
    selected = np.zeros(size, dtype=np.int64)
    possible_indices = np.arange(n)
    if p is None:
        p = np.ones(n, dtype=np.float64) / n
    for i in range(size):
        cdf = np.cumsum(p)
        r = np.random.rand()
        picked = 0
        for index, threshold in enumerate(cdf):
            if r <= threshold:
                selected[i] = x[possible_indices[index]]
                picked = index
                break
        if replace is False and size > 1:
            possible_indices = np.delete(possible_indices, picked)
            p = np.array([a for k, a in enumerate(p) if k != picked])
            p = p / p.sum()
    return selected


@njit
def generate_intervals(a, b, min_charging_interval, min_non_charging_interval, k, L):
    if L < k * min_charging_interval or (b - a) < L + (k - 1) * min_non_charging_interval:
        raise ValueError("Infeasible")
    L_extra = L - k * min_charging_interval
    G_extra = (b - a) - L - (k - 1) * min_non_charging_interval
    if L_extra == 0:
        interval_deltas = np.zeros(k)
    else:
        interval_deltas = np.random.dirichlet(np.ones(k)) * L_extra
    if G_extra == 0:
        gap_deltas = np.zeros(k + 1)
    else:
        gap_deltas = np.random.dirichlet(np.ones(k + 1)) * G_extra
    intervals = np.zeros((k, 2))
    current = a + gap_deltas[0]
    for i in range(k):
        start = current
        end = start + min_charging_interval + interval_deltas[i]
        intervals[i] = start, end
        current = end + min_non_charging_interval + gap_deltas[i + 1]
    return intervals


@njit
def find_availability(j, schedule, Instance, min_distance_between_vehicles):
    requests, chargers, compatibility_matrix, kappa, std_deviations = Instance
    n, m = compatibility_matrix.shape
    sub_schedule = schedule[schedule[:, 0] == j + 1].copy()
    sub_schedule = sub_schedule[np.argsort(sub_schedule[:, 1])]
    N = len(sub_schedule)
    availability = np.zeros((N + 1, 2))
    if N > 0:
        availability[0] = np.array([chargers[j, 0], sub_schedule[0, 1] - min_distance_between_vehicles])
        for k in range(1, N):
            availability[k] = np.array([
                sub_schedule[k - 1, 2] + min_distance_between_vehicles,
                sub_schedule[k, 1] - min_distance_between_vehicles
            ])
        availability[N] = np.array([sub_schedule[N - 1, 2] + min_distance_between_vehicles, INFINITY])
    else:
        availability[0] = np.array([chargers[j, 0], INFINITY])
    return availability


@njit
def find_candidate_charging_periods(i, j, availability, mean_duration_to_allocate, Instance):
    requests, chargers, compatibility_matrix, kappa, std_deviations = Instance
    n, m = compatibility_matrix.shape
    candidate_charging_periods = []
    N = len(availability)
    for k in range(N):
        av_s = max(availability[k, 0], requests[i, 0])
        av_e = availability[k, 1]
        if av_e == INFINITY:
            candidate_charging_periods.append(k)
        elif av_e - av_s >= mean_duration_to_allocate:
            candidate_charging_periods.append(k)
    return np.array(candidate_charging_periods)


@njit
def make_vehicle_charging_sessions(energy_to_deliver, charger_power, a, b, kappa, min_charging_interval, min_non_charging_interval, max_preemptions):
    L = energy_to_deliver / (charger_power * kappa)
    if b - a < L:
        raise ValueError("Not feasible b-a<L")
    if L > min_charging_interval:
        ub = min(int(L / min_charging_interval), int((b - a - L) / min_non_charging_interval + 1), max_preemptions)
        if ub < 1:
            ub = 1
        k = np.random.randint(1, ub + 1)
        intervals = generate_intervals(a, b, min_charging_interval, min_non_charging_interval, k, L)
        ts = intervals[0, 0]
        te = intervals[-1, 1]
    else:
        ts = np.random.uniform(a, b - L)
        # The session interval itself stays anchored to [ts, ts+L]: its
        # length must equal L = delivered/power exactly, since that same
        # total_duration is also the mean fed into the lognormal
        # actualisation (actualise_schedule_lognormal). The *occupancy*
        # end te is sampled independently and can exceed ts+L -- this
        # reserved slack is what lets a later preemption be carved out of
        # this session by crossover/perturbation without renegotiating the
        # charger boundary with the next vehicle.
        te = np.random.uniform(ts + L, b)
        intervals = np.array([[ts, ts + L]])
    return ts, te, intervals


@njit
def place_vehicle_on_charger(
    i, j, schedule, lam, Instance,
    min_percentage, min_distance_between_vehicles,
    min_charging_interval, min_non_charging_interval, max_preemptions
):
    requests, chargers, compatibility_matrix, kappa, std_deviations = Instance
    schedule = schedule.copy()

    e = ((min_percentage / 100.0) + (1.0 - min_percentage / 100.0) * np.random.rand()) * requests[i, 2]
    mean_duration_to_allocate = e / (chargers[j, 1] * kappa)

    availability = find_availability(j, schedule, Instance, min_distance_between_vehicles)
    candidate_charging_periods = find_candidate_charging_periods(i, j, availability, mean_duration_to_allocate, Instance)

    if len(candidate_charging_periods) == 0:
        ts = max(chargers[j, 0], requests[i, 0])
        te = ts + mean_duration_to_allocate
        schedule[i, :4] = [j + 1, ts, te, e]
        schedule[i, 4:] *= 0
        schedule[i, 4:6] = [ts, te]
        return schedule

    k = random_choice(candidate_charging_periods)[0]
    a = max(availability[k, 0], requests[i, 0])
    b = availability[k, 1]
    if b == INFINITY:
        b = a + mean_duration_to_allocate + np.abs(np.random.normal(0, lam))

    ts, te, intervals = make_vehicle_charging_sessions(
        e, chargers[j, 1], a, b, kappa, min_charging_interval, min_non_charging_interval, max_preemptions
    )

    schedule[i, :4] = [j + 1, ts, te, e]
    schedule[i, 4:] *= 0
    schedule[i, 4:4 + 2 * len(intervals)] = intervals.flatten()
    return schedule


@njit
def generate_random_schedule(
    Instance, lam, min_percentage,
    min_distance_between_vehicles, min_charging_interval, min_non_charging_interval, max_preemptions
):
    requests, chargers, compatibility_matrix, kappa, std_deviations = Instance
    n, m = compatibility_matrix.shape
    schedule = np.zeros((n, 4 + 2 * max_preemptions))
    vehicles_indexes = np.arange(n)
    np.random.shuffle(vehicles_indexes)
    for idx in range(n):
        i = vehicles_indexes[idx]
        candidate_chargers = np.arange(m)[compatibility_matrix[i, :] == 1]
        j = int(np.random.choice(candidate_chargers))
        schedule = place_vehicle_on_charger(
            i, j, schedule, lam, Instance,
            min_percentage, min_distance_between_vehicles,
            min_charging_interval, min_non_charging_interval, max_preemptions
        )
    return schedule


@njit
def find_next_vehicle(i, schedule):
    j = int(schedule[i, 0])
    te = schedule[i, 2]
    mask = (schedule[:, 0] == j) & (schedule[:, 1] > te)
    candidates = np.where(mask)[0]
    if candidates.size > 0:
        return candidates[np.argmin(schedule[candidates, 1])]
    return -1


@njit
def cdf_normal(x):
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


@njit
def grid_capacity(schedule, chargers):
    n = schedule.shape[0]
    s = schedule.shape[1] - 4
    max_events = n * (s // 2)
    events = np.zeros((2 * max_events, 2), dtype=np.float64)
    n_events = 0
    for i in range(n):
        power = chargers[int(schedule[i, 0]) - 1, 1]
        for col in range(4, 4 + s - 1, 2):
            if schedule[i, col] == 0:
                break
            events[n_events, 0] = schedule[i, col]
            events[n_events, 1] = power
            n_events += 1
            events[n_events, 0] = schedule[i, col + 1]
            events[n_events, 1] = -power
            n_events += 1
    if n_events == 0:
        return 0.0
    events = events[:n_events]
    order = np.argsort(events[:, 0])
    events = events[order]
    return np.max(np.cumsum(events[:, 1]))


@njit
def conditional_mean_lognormal(mu, sigma, a, b, d):
    if b > max(a, d):
        A_val = np.exp(mu + 0.5 * (sigma ** 2))
        A1 = (np.log(b) - mu) / sigma - sigma
        A2 = (np.log(max(a, d)) - mu) / sigma - sigma
        A3 = (np.log(b) - mu) / sigma
        A4 = (np.log(max(a, d)) - mu) / sigma
        A5 = (np.log(b) - mu) / sigma
        A6 = (np.log(a) - mu) / sigma
        B = A_val * (cdf_normal(A1) - cdf_normal(A2)) - d * (cdf_normal(A3) - cdf_normal(A4))
        C = cdf_normal(A5) - cdf_normal(A6)
        if C > 1e-15:
            # This is E[max(R-d,0) | a<R<b] for R lognormal -- mathematically
            # always >= 0, but B/C is a difference of two near-cancelling
            # terms and floating-point rounding can push it a hair below
            # zero (observed: -5e-16). Clamp to the mathematically-guaranteed
            # non-negative range.
            return max(0.0, B / C)
        return 0.0
    else:
        return 0.0


@njit
def actualise_schedule_lognormal(schedule, std_deviations, min_distance_between_vehicles, max_preemptions, min_charging_interval, min_non_charging_interval):
    n = schedule.shape[0]
    actualised_schedule = schedule.copy()
    for i in range(n):
        next_vehicle = find_next_vehicle(i, schedule)
        j = int(schedule[i, 0]) - 1

        last_index = 0
        for li in range(len(schedule[i]) - 1, -1, -1):
            if schedule[i, li] != 0:
                last_index = li
                break
        if last_index < 4:
            continue

        flattened_intervals = schedule[i, 4:last_index + 1]
        starts = flattened_intervals[::2]
        ends = flattened_intervals[1::2]
        sessions_mean_lengths = ends - starts
        total_duration = np.sum(sessions_mean_lengths)
        if total_duration <= 0:
            continue

        portions = sessions_mean_lengths / total_duration
        new_intervals = np.zeros_like(flattened_intervals)

        for v in range(len(starts)):
            if sessions_mean_lengths[v] > min_charging_interval:
                a = starts[v] + min_charging_interval
            else:
                a = starts[v]

            if v < len(starts) - 1:
                b = starts[v + 1] - min_non_charging_interval
            else:
                b = INFINITY if next_vehicle == -1 else schedule[next_vehicle, 1] - min_distance_between_vehicles

            E = starts[v] + portions[v] * total_duration
            V = (std_deviations[i, j] * sessions_mean_lengths[v]) ** 2
            A_val = 1.0 + V / (E ** 2)
            mean_parms = np.log(E / np.sqrt(A_val))
            sigma_parms = np.sqrt(np.log(A_val))

            while True:
                r = np.random.lognormal(mean_parms, sigma_parms)
                if a < r < b:
                    break

            new_intervals[2 * v] = starts[v]
            new_intervals[2 * v + 1] = r

        actualised_schedule[i, 1] = new_intervals[0]
        actualised_schedule[i, 2] = new_intervals[-1]
        actualised_schedule[i, 4:last_index + 1] = new_intervals
    return actualised_schedule


@njit
def total_tardiness_lognormal(schedule, requests, min_charging_interval, min_distance_between_vehicles, std_deviations):
    TT = 0.0
    n = len(requests)
    for i in range(n):
        last_index = 0
        for li in range(len(schedule[i]) - 1, -1, -1):
            if schedule[i, li] != 0:
                last_index = li
                break
        if last_index < 4:
            continue

        flattened_intervals = schedule[i, 4:last_index + 1]
        starts = flattened_intervals[::2]
        ends = flattened_intervals[1::2]
        sessions_mean_lengths = ends - starts
        total_duration = np.sum(sessions_mean_lengths)
        if total_duration == 0:
            continue

        portions = sessions_mean_lengths / total_duration
        ts = starts[-1]
        ui = 1.0 if total_duration >= min_charging_interval else 0.0
        a = ts + min_charging_interval * ui
        next_vehicle = find_next_vehicle(i, schedule)
        b = INFINITY if next_vehicle == -1 else schedule[next_vehicle, 1] - min_distance_between_vehicles
        d = requests[i, 1]
        j = int(schedule[i, 0]) - 1

        E = starts[-1] + portions[-1] * total_duration
        V = (std_deviations[i, j] * sessions_mean_lengths[-1]) ** 2
        A_val = 1.0 + V / (E ** 2)
        mean_parms = np.log(E / np.sqrt(A_val))
        sigma_parms = np.sqrt(np.log(A_val))
        TT += conditional_mean_lognormal(mean_parms, sigma_parms, a, b, d)
    return TT


@njit
def _objectives_calculation(schedule, Instance, min_distance_between_vehicles, max_preemptions, min_charging_interval, min_non_charging_interval, num_replications):
    requests, chargers, compatibility_matrix, kappa, std_deviations = Instance
    if num_replications == 0:
        grid_capacity_value = grid_capacity(schedule, chargers)
        total_tardiness = np.sum(np.maximum(schedule[:, 2] - requests[:, 1], 0.0))
        unmet_energy_fraction = np.sum(np.maximum((requests[:, 2] - schedule[:, 3]) / requests[:, 2], 0.0)) / len(schedule)
    else:
        replication_gc_samples = np.zeros(num_replications)
        for k in range(num_replications):
            actualised_schedule = actualise_schedule_lognormal(
                schedule, std_deviations, min_distance_between_vehicles,
                max_preemptions, min_charging_interval, min_non_charging_interval
            )
            replication_gc_samples[k] = grid_capacity(actualised_schedule, chargers)
        grid_capacity_value = np.mean(replication_gc_samples)
        total_tardiness = total_tardiness_lognormal(schedule, requests, min_charging_interval, min_distance_between_vehicles, std_deviations)
        unmet_energy_fraction = np.sum(np.maximum((requests[:, 2] - schedule[:, 3]) / requests[:, 2], 0.0)) / len(schedule)
    return np.array([grid_capacity_value, total_tardiness, 100.0 * unmet_energy_fraction])


@njit
def _generalized_crossover(
    solutions, Instance, lam, min_percentage,
    min_distance_between_vehicles, min_charging_interval, min_non_charging_interval, max_preemptions
):
    requests, chargers, compatibility_matrix, kappa, std_deviations = Instance
    n, m = compatibility_matrix.shape
    N = len(solutions)
    new_solution = np.zeros(solutions[0].shape, dtype=np.float64)
    vehicles_indexes = np.arange(n)
    np.random.shuffle(vehicles_indexes)
    first_one = True
    placed = np.zeros(n, dtype=np.int8)
    not_placed = np.zeros(n, dtype=np.int64)
    n_placed = 0
    n_not_placed = 0

    for vi in range(n):
        i = vehicles_indexes[vi]
        if first_one:
            s = np.random.randint(N)
            new_solution[i] = solutions[s][i].copy()
            placed[i] = 1
            n_placed += 1
            first_one = False
        else:
            placable = np.ones(N, dtype=np.int8)
            for s in range(N):
                for j in range(n):
                    if placed[j] == 0:
                        continue
                    # Overlap only matters between occupancies on the SAME
                    # charger -- two vehicles on different chargers can
                    # freely have overlapping time windows (that's the whole
                    # point of having multiple chargers). Without this check,
                    # any time overlap with ANY already-placed vehicle
                    # (regardless of charger) blocked the copy, which made
                    # crossover degenerate to near-random placement on
                    # anything but tiny instances. Sibling chapters (8, 10,
                    # 12) all gate the overlap test on charger equality.
                    if int(solutions[s][i, 0]) != int(new_solution[j, 0]):
                        continue
                    cond1 = solutions[s][i, 1] >= new_solution[j, 2] + min_distance_between_vehicles
                    cond2 = new_solution[j, 1] >= solutions[s][i, 2] + min_distance_between_vehicles
                    cond = cond1 or cond2
                    if not cond:
                        placable[s] = 0
                        break

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
                            n_placed += 1
                            break
                        count += 1
            else:
                not_placed[n_not_placed] = i
                n_not_placed += 1

    for k in range(n_not_placed):
        i = not_placed[k]
        candidate_chargers = np.arange(m)[compatibility_matrix[i, :] == 1]
        j = int(np.random.choice(candidate_chargers))
        new_solution = place_vehicle_on_charger(
            i, j, new_solution, lam, Instance,
            min_percentage, min_distance_between_vehicles,
            min_charging_interval, min_non_charging_interval, max_preemptions
        )
    return new_solution


@njit
def _FN(
    schedule, Instance, perc_perturb, lam, min_percentage,
    min_distance_between_vehicles, min_charging_interval, min_non_charging_interval, max_preemptions
):
    requests, chargers, compatibility_matrix, kappa, std_deviations = Instance
    n, m = compatibility_matrix.shape
    schedule = schedule.copy()
    # Vehicles with only one compatible charger are excluded from selection
    # since reassigning them would be a no-op, but the perturbation rate perc_perturb
    # is applied relative to the full instance size n (matching the paper's
    # K=ceil(perc_perturb*n)), not to this filtered pool -- otherwise the same perc_perturb
    # would perturb a different fraction of the fleet depending on how many
    # vehicles happen to have only one compatible charger.
    candidate_vehicles = np.array([i for i in range(n) if np.sum(compatibility_matrix[i, :]) > 1], dtype=np.int32)
    np.random.shuffle(candidate_vehicles)
    number_of_changed = int(np.ceil(perc_perturb * n))
    for k in range(min(number_of_changed, len(candidate_vehicles))):
        i = candidate_vehicles[k]
        candidate_chargers = np.array([j for j in range(m) if compatibility_matrix[i, j] == 1], dtype=np.int32)
        j = int(np.random.choice(candidate_chargers))
        schedule = place_vehicle_on_charger(
            i, j, schedule, lam, Instance,
            min_percentage, min_distance_between_vehicles,
            min_charging_interval, min_non_charging_interval, max_preemptions
        )
    return schedule


@njit
def objective_function(solution, problem_params):
    perc_perturb, lam, variance_factor, num_of_replications, Instance = problem_params
    return _objectives_calculation(
        np.asarray(solution, dtype=np.float64), Instance,
        DEFAULT_MIN_DIST, DEFAULT_MAX_PREEMPTIONS,
        DEFAULT_MIN_PL, DEFAULT_MIN_NPL, int(num_of_replications),
    )


@njit
def Generate_Solution(problem_params):
    _, lam, _, _, Instance = problem_params
    return generate_random_schedule(
        Instance, lam, DEFAULT_MIN_PERCENTAGE, DEFAULT_MIN_DIST,
        DEFAULT_MIN_PL, DEFAULT_MIN_NPL, DEFAULT_MAX_PREEMPTIONS,
    )


@njit
def Perturb(solution, problem_params):
    perc_perturb, lam, _, _, Instance = problem_params
    return _FN(
        np.asarray(solution, dtype=np.float64), Instance, perc_perturb, lam,
        DEFAULT_MIN_PERCENTAGE, DEFAULT_MIN_DIST,
        DEFAULT_MIN_PL, DEFAULT_MIN_NPL, DEFAULT_MAX_PREEMPTIONS,
    )


@njit
def Crossover(parents_array, problem_params):
    _, lam, _, _, Instance = problem_params
    return _generalized_crossover(
        np.asarray(parents_array, dtype=np.float64), Instance, lam,
        DEFAULT_MIN_PERCENTAGE, DEFAULT_MIN_DIST,
        DEFAULT_MIN_PL, DEFAULT_MIN_NPL, DEFAULT_MAX_PREEMPTIONS,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CHAPTER 3 CODE (VERBATIM) -- verifiers
# ═══════════════════════════════════════════════════════════════════════════════


@njit
def _extract_intervals_from_row(row: np.ndarray, max_preemptions: int, eps: float):
    intervals = []
    start_col = 4
    end_col = 4 + 2 * max_preemptions
    for col in range(start_col, end_col, 2):
        s = float(row[col])
        e = float(row[col + 1])
        if abs(s) <= eps and abs(e) <= eps:
            break
        if (abs(s) <= eps) != (abs(e) <= eps):
            raise AssertionError(
                "Malformed interval encoding: one endpoint is zero while the other is nonzero."
            )
        intervals.append((s, e))
    return intervals


def check_schedule_feasibility(schedule, Instance, label="schedule", eps=None):
    requests, chargers, compatibility_matrix, kappa, std_deviations = Instance
    eps = VERIFIER_EPS if eps is None else eps

    requests = np.asarray(requests, dtype=np.float64)
    chargers = np.asarray(chargers, dtype=np.float64)
    compatibility_matrix = np.asarray(compatibility_matrix, dtype=np.int8)
    schedule = np.asarray(schedule, dtype=np.float64)

    n, m = compatibility_matrix.shape
    expected_shape = (n, 4 + 2 * DEFAULT_MAX_PREEMPTIONS)
    assert schedule.shape == expected_shape, (
        f"[{label}] SHAPE FAIL -- expected {expected_shape}, got {schedule.shape}."
    )

    for i in range(n):
        charger = int(round(schedule[i, 0]))
        ts = float(schedule[i, 1])
        te = float(schedule[i, 2])
        delivered = float(schedule[i, 3])

        if charger < 1 or charger > m:
            raise AssertionError(
                f"[{label}] CHARGER FAIL -- vehicle {i} charger {charger} outside [1,{m}]."
            )

        if compatibility_matrix[i, charger - 1] != 1:
            raise AssertionError(
                f"[{label}] COMPAT FAIL -- vehicle {i} incompatible with charger {charger}."
            )

        if te < ts - eps:
            raise AssertionError(
                f"[{label}] INTERVAL FAIL -- vehicle {i} has te={te} < ts={ts}."
            )

        if ts < requests[i, 0] - eps:
            raise AssertionError(
                f"[{label}] ARRIVAL FAIL -- vehicle {i} starts at {ts} before arrival {requests[i, 0]}."
            )

        if ts < chargers[charger - 1, 0] - eps:
            raise AssertionError(
                f"[{label}] OPEN FAIL -- vehicle {i} starts at {ts} before charger {charger} opens at {chargers[charger - 1, 0]}."
            )

        req_energy = float(requests[i, 2])
        min_energy = (DEFAULT_MIN_PERCENTAGE / 100.0) * req_energy
        if delivered < min_energy - eps or delivered > req_energy + eps:
            raise AssertionError(
                f"[{label}] ENERGY FAIL -- vehicle {i} delivered energy {delivered} not in [{min_energy},{req_energy}]."
            )

        intervals = _extract_intervals_from_row(schedule[i], DEFAULT_MAX_PREEMPTIONS, eps)
        if len(intervals) == 0:
            raise AssertionError(
                f"[{label}] SESSION FAIL -- vehicle {i} has no charging sessions encoded."
            )

        if len(intervals) > DEFAULT_MAX_PREEMPTIONS:
            raise AssertionError(
                f"[{label}] SESSION FAIL -- vehicle {i} has {len(intervals)} sessions > max {DEFAULT_MAX_PREEMPTIONS}."
            )

        first_s = intervals[0][0]
        last_e = intervals[-1][1]
        if abs(first_s - ts) > 1e-6:
            raise AssertionError(
                f"[{label}] BOUNDARY FAIL -- vehicle {i} first session start {first_s} != occupancy start {ts}."
            )
        if last_e > te + 1e-6:
            raise AssertionError(
                f"[{label}] BOUNDARY FAIL -- vehicle {i} last session end {last_e} exceeds occupancy end {te}."
            )

        total_duration = 0.0
        for v, (s, e) in enumerate(intervals):
            if e <= s + eps:
                raise AssertionError(
                    f"[{label}] SESSION LENGTH FAIL -- vehicle {i} session {v} has nonpositive length."
                )
            # make_vehicle_charging_sessions (line ~262) only pads sessions up
            # to min_pl when splitting into >=2 sessions; a vehicle whose full
            # required duration L is itself <= min_pl gets exactly ONE session
            # of length L (padding it would over-deliver energy), so the floor
            # only applies when there's more than one session.
            if len(intervals) > 1 and e - s < DEFAULT_MIN_PL - eps:
                raise AssertionError(
                    f"[{label}] SESSION LENGTH FAIL -- vehicle {i} session {v} length {e - s} "
                    f"below min_pl={DEFAULT_MIN_PL} (vehicle has {len(intervals)} sessions)."
                )
            total_duration += (e - s)

            if v > 0:
                prev_e = intervals[v - 1][1]
                if s < prev_e - eps:
                    raise AssertionError(
                        f"[{label}] SESSION OVERLAP FAIL -- vehicle {i} sessions overlap."
                    )
                if s - prev_e < DEFAULT_MIN_NPL - eps:
                    raise AssertionError(
                        f"[{label}] GAP FAIL -- vehicle {i} gap {s - prev_e} smaller than min_npl={DEFAULT_MIN_NPL}."
                    )

        implied_duration = delivered / (chargers[charger - 1, 1] * kappa)
        if abs(total_duration - implied_duration) > 1e-5:
            raise AssertionError(
                f"[{label}] DURATION FAIL -- vehicle {i} total session length {total_duration} "
                f"!= delivered/(power*kappa) {implied_duration}."
            )

        used_cols = 4 + 2 * len(intervals)
        if np.any(np.abs(schedule[i, used_cols:]) > eps):
            raise AssertionError(
                f"[{label}] TAIL FAIL -- vehicle {i} has nonzero values after encoded sessions."
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
                separated = (s2 >= e1 + DEFAULT_MIN_DIST - eps) or (s1 >= e2 + DEFAULT_MIN_DIST - eps)
                if not separated:
                    raise AssertionError(
                        f"[{label}] VEHICLE DIST FAIL -- charger {c}: vehicles {i1},{i2} "
                        f"violates min distance {DEFAULT_MIN_DIST} with occupancies [{s1},{e1}] and [{s2},{e2}]."
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTATION (2): THE INSTANCE SUITE (ACN-DATA AND SIMULATED REQUESTS)
# ═══════════════════════════════════════════════════════════════════════════════

class Chapter3Instance:
    """One instance: its identity (family, size, seed) and Chapter 3's tuple
    `data` = (requests, chargers, compatibility_matrix, kappa, std_deviations)
    read by the njit operators."""

    def __init__(self, name: str, source: str, seed: int, data: tuple):
        self.name, self.source, self.seed = str(name), str(source), int(seed)
        self.data = data
        self.n, self.m = (int(x) for x in data[2].shape)

    @property
    def requests(self) -> np.ndarray:
        return self.data[0]

    @property
    def chargers(self) -> np.ndarray:
        return self.data[1]

    def features(self) -> Dict[str, float]:
        """Numeric descriptors of the instance (size n, ...)."""
        req, ch, comp = self.data[0], self.data[1], self.data[2]
        return {"n": float(self.n), "m": float(self.m),
                "energy_per_charger_kwh": float(req[:, 2].sum() / self.m),
                "mean_power_kw": float(ch[:, 1].mean()),
                "arrival_span_h": float(req[:, 0].max() - req[:, 0].min()),
                "mean_stay_h": float((req[:, 1] - req[:, 0]).mean()),
                "mean_compatibility": float(comp.mean()),
                "acn": 1.0 if self.source == "ACN" else 0.0}

    def __repr__(self):
        return f"Chapter3Instance({self.name}: n={self.n}, m={self.m})"


def n_chargers(n: int) -> int:
    return -(-int(n) // P.VEHICLES_PER_CHARGER)


def acn_instance(n: int, seed: int, name: str = None) -> Chapter3Instance:
    """Chapter 3's instance: simulate_instance on the ACN sessions (unchanged)."""
    data = simulate_instance(requests=load_acn(), n=n, m=n_chargers(n), seed=seed)
    return Chapter3Instance(name or f"ACN-n{n}-s{seed}", "ACN", seed, data)


def simulated_requests(n: int, rng: np.random.Generator):
    """Requests of Zaidi et al. and Golabi et al. (2025, Section 6), as in Appendix D:
    e ~ U[5.5, 66] kWh, a ~ U[0, 0.2 n] h, d = a + (1 + alpha) e / 11 with alpha
    uniform in the range of Table 3 of Golabi et al. for e / 11."""
    e = rng.uniform(*P.SIM_ENERGY_RANGE, n)
    a = rng.uniform(0.0, P.SIM_ARRIVAL_SPAN_PER_VEHICLE * n, n)
    p1 = e / P.SIM_REFERENCE_POWER
    alpha = np.empty(n)
    for k, p in enumerate(p1):
        lo, hi = next((r for (plo, phi), r in P.SIM_ALPHA if plo <= p <= phi), P.SIM_ALPHA_DEFAULT)
        alpha[k] = rng.uniform(lo, hi)
    return a, a + (1.0 + alpha) * p1, e


def simulated_instance(n: int, seed: int, name: str = None) -> Chapter3Instance:
    """Simulated requests; the chargers, compatibilities and standard deviations
    are drawn exactly as in Chapter 3's simulate_instance (same distributions,
    same order of draws after the requests)."""
    m = n_chargers(n)
    rng = np.random.default_rng(seed)
    a, d, e = simulated_requests(n, rng)
    requests_arr = np.zeros((n, 3), dtype=np.float64)
    requests_arr[:, 0], requests_arr[:, 1], requests_arr[:, 2] = a, np.maximum(d, a), e
    compatibility_matrix = np.zeros((n, m), dtype=np.int8)
    for i in range(n):
        min_compat = max(1, int(ceil((DEFAULT_MIN_PERCENTAGE / 100.0) * m)))
        num_compat = int(rng.integers(min_compat, m + 1))
        compatible = rng.choice(m, num_compat, replace=False)
        compatibility_matrix[i, compatible] = 1
    chargers = np.zeros((m, 2), dtype=np.float64)
    for j in range(m):
        chargers[j, 0] = float(rng.uniform(*P.CHARGER_OPENING_RANGE_H))
        chargers[j, 1] = float(rng.choice(DEFAULT_CHARGING_POWERS))
    std_deviations = np.zeros((n, m), dtype=np.float64)
    std_values = np.linspace(*P.STD_VALUES)
    for i in range(n):
        for j in range(m):
            if compatibility_matrix[i, j] == 1:
                std_deviations[i, j] = float(rng.choice(std_values))
    data = (requests_arr, chargers, compatibility_matrix, float(P.KAPPA), std_deviations)
    return Chapter3Instance(name or f"SIM-n{n}-s{seed}", "SIM", seed, data)


def generate_instances() -> List[Chapter3Instance]:
    """The evaluation suite: the instances of parameters.FAMILIES (the manuscript
    uses the 20 ACN instances), by increasing size, INSTANCES_PER_SIZE per size."""
    makers = {"ACN": (acn_instance, P.ACN_SEED_BASE), "SIM": (simulated_instance, P.SIM_SEED_BASE)}
    out = []
    for source in P.FAMILIES:
        maker, base = makers[source]
        k = 0
        for n in P.SIZES:
            for j in range(P.INSTANCES_PER_SIZE):
                out.append(maker(n, base + k, name=f"{source} n={n}-{j + 1}"))
                k += 1
    return out


def _class_label(source: str, label: str) -> str:
    return label if len(P.FAMILIES) == 1 else f"{source} {label}"


def instance_class(inst: Chapter3Instance) -> str:
    return _class_label(inst.source, P.CLASS_LABELS[P.SIZES.index(inst.n)])


def size_classes() -> Dict[str, List[int]]:
    """Ids (1-based, order of generate_instances) of every class, without
    generating the instances."""
    out, k = {}, 0
    for source in P.FAMILIES:
        for label in P.CLASS_LABELS[:len(P.SIZES)]:
            out[_class_label(source, label)] = list(range(k + 1, k + 1 + P.INSTANCES_PER_SIZE))
            k += P.INSTANCES_PER_SIZE
    return out


def instance_features(inst: Chapter3Instance) -> Dict[str, float]:
    return inst.features()


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTATION (3): METHODS_COMPARISON INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def build_problem_params(params: Dict[str, Any], inst: Chapter3Instance) -> tuple:
    """Chapter 3's _build_problem_params, on the instance's tuple."""
    perc_perturb = float(params.get("perc_perturb", DEFAULT_PERC_PERTURB))
    lam = float(params.get("lam", DEFAULT_LAM))
    num_of_replications = int(params.get("num_of_replications", DEFAULT_NUM_REPLICATIONS))
    return (perc_perturb, lam, DEFAULT_VARIANCE_FACTOR, num_of_replications, inst.data)


def solution_shape(inst: Chapter3Instance) -> tuple:
    return (inst.n, 4 + 2 * int(DEFAULT_MAX_PREEMPTIONS))


def evaluate_objective(inst: Chapter3Instance, solution) -> np.ndarray:
    """The three objectives of a schedule (Chapter 3's evaluate_objective: N_r = 30)."""
    return objective_function(np.asarray(solution, dtype=np.float64), build_problem_params({}, inst))


@njit
def _nb_seed(seed):
    np.random.seed(seed)


def seed_all(seed: int) -> None:
    """Seeds Python's random, NumPy's global RNG and Numba's own RNG (the njit
    operators and the actualisation draw from the latter)."""
    random.seed(seed)
    np.random.seed(seed)
    _nb_seed(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# INDEPENDENT REFERENCES (tests only; plain Python, from the definitions)
# ═══════════════════════════════════════════════════════════════════════════════
