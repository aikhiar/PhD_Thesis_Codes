"""
model.py -- Appendix C of the thesis (appendix_biobjective.tex of the
manuscript): bi-objective EV charging scheduling for grid stability and user
satisfaction.

The operators, the objectives, the instance sampling and the verifiers are the
code of the thesis, unchanged. What is adapted to the current frameworks is the
last section (the instance suite as a function, the interface of methods_comparison and of
det_vs_stoch) and the data file (acn_sessions.pkl: the same sessions as
requests_data.pkl). The parameters are in parameters.py. This folder is
identical in every framework that uses it.

Objectives   (peak load kW, total completion time in slots), deterministic.
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

PROBLEM_NAME = "appendix_c"
CODE = "appendix_c"
APPENDIX_C_DIR = Path(__file__).resolve().parent
OBJECTIVE_NAMES = ["Peak load (kW)", "Total completion time (slots)"]
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


def _sample_request_rows(n: int, requests: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw n DISTINCT rows whose arrival-time distribution matches the full
    requests population (KDE-resample, then snap each draw to its nearest
    real row) rather than a uniform draw over rows."""
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

    return requests[np.array(chosen_indices)]


def simulate_instance(
    requests: Optional[np.ndarray] = None,
    n: int = 20,
    m: int = 6,
    charging_powers: Optional[np.ndarray] = None,
    delta: Optional[np.ndarray] = None,
    tau: float = 1.0,
    seed: Optional[int] = None,
):
    """
    tau is in HOURS per slot.
    Request arrival/departure times are assumed to be in HOURS too.
    """
    if requests is None:
        requests = _load_requests_data()
    if charging_powers is None:
        charging_powers = DEFAULT_CHARGING_POWERS

    rng = np.random.default_rng(seed)
    data = _sample_request_rows(n, requests, rng)

    arrival_departure = np.zeros((n, 2), dtype=np.int32)
    required_energy = np.zeros(n, dtype=np.float64)
    for i in range(n):
        arrival = int(np.ceil(float(data[i, 0]) / tau))
        departure = int(np.ceil(float(data[i, 1]) / tau))
        departure = max(departure, arrival)
        arrival_departure[i, 0] = arrival
        arrival_departure[i, 1] = departure
        required_energy[i] = float(data[i, 2])

    chargers = np.zeros((m, 2), dtype=np.float64)
    for j in range(m):
        chargers[j, 0] = float(np.ceil(rng.uniform(0.0, 8.0) / tau))
        chargers[j, 1] = float(rng.choice(charging_powers))

    if delta is None:
        delta = np.ones((n, m), dtype=np.int8)
    delta = np.asarray(delta, dtype=np.int8)

    requests_temp = np.array(arrival_departure, dtype=np.int32)
    chargers_temp = np.array(chargers, dtype=np.float32)
    scheduling_start_slot = int(min(np.min(requests_temp[:, 0]), np.min(chargers_temp[:, 0])))

    slot_budget = np.zeros((n, m), dtype=np.int16)
    for i in range(n):
        for j in range(m):
            slot_budget[i, j] = ceil(required_energy[i] / (tau * chargers[j, 1]))

    return (
        requests_temp,
        required_energy,
        chargers_temp,
        delta,
        float(tau),
        int(n),
        int(m),
        scheduling_start_slot,
        slot_budget,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# OPERATORS (numba-compiled core + Generate/Perturb/Crossover/objective)
# ═══════════════════════════════════════════════════════════════════════════════

@njit
def random_choice(x, size=1, replace=True, p=None):
    n = len(x)
    selected = np.zeros(size, dtype=np.int32)
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
def find_availability(schedule, chargers_temp, j):
    sub_schedule = schedule[schedule[:, 0] == j + 1]
    sub_schedule = sub_schedule[np.argsort(sub_schedule[:, 1])]
    N = len(sub_schedule)
    availability = np.zeros((N + 1, 2), dtype=np.int32)
    if N > 0:
        availability[0] = np.array([int(chargers_temp[j, 0]), sub_schedule[0, 1] - 1])
        for k in range(1, N):
            availability[k] = np.array([sub_schedule[k - 1, 2] + 1, sub_schedule[k, 1] - 1])
        availability[N] = np.array([sub_schedule[N - 1, 2] + 1, INFINITY])
    else:
        availability[0] = np.array([int(chargers_temp[j, 0]), INFINITY])
    return availability


@njit
def find_candidate_charging_periods(i, j, availability, requests_temp, slot_budget):
    candidate_charging_periods = []
    for k in range(len(availability)):
        if availability[k, 1] == INFINITY:
            candidate_charging_periods.append(k)
        elif availability[k, 1] - max(availability[k, 0], requests_temp[i, 0]) + 1 >= slot_budget[i, j]:
            candidate_charging_periods.append(k)
    return np.array(candidate_charging_periods)


@njit
def place_vehicle_on_charger(i, j, schedule, requests_temp, chargers_temp, slot_budget, sigma):
    schedule = schedule.copy()
    availability = find_availability(schedule, chargers_temp, j)
    candidate_charging_periods = find_candidate_charging_periods(i, j, availability, requests_temp, slot_budget)

    N = len(candidate_charging_periods)
    probabilities = (N - np.arange(N)) / (N * (N + 1) / 2)
    k = random_choice(candidate_charging_periods, p=probabilities)[0]

    A = max(availability[k, 0], requests_temp[i, 0])
    if availability[k, 1] == INFINITY:
        ts = A + int(np.abs(sigma * np.random.randn()))
    else:
        ts = np.random.choice(np.arange(A, availability[k, 1] - slot_budget[i, j] + 2))

    schedule[i] = [j + 1, ts, ts + slot_budget[i, j] - 1]
    return schedule


@njit
def generate_random_schedule(delta, requests_temp, chargers_temp, slot_budget, sigma=5):
    n = len(requests_temp)
    m = len(chargers_temp)
    schedule = np.zeros((n, 3), dtype=np.int32)
    vehicles_indexes = np.arange(n)
    np.random.shuffle(vehicles_indexes)
    for i in vehicles_indexes:
        candidate_chargers = np.arange(m)[delta[i, :] == 1]
        j = np.random.choice(candidate_chargers)
        schedule = place_vehicle_on_charger(i, j, schedule, requests_temp, chargers_temp, slot_budget, sigma)
    return schedule


@njit
def _generalized_crossover(solutions, delta, requests_temp, chargers_temp, slot_budget, sigma):
    # N-parent, vehicle-centric crossover (matches the pattern used in
    # chapter9_dependency._generalized_crossover): for each vehicle, inherit
    # its assignment from a uniformly-drawn parent among those that don't
    # collide with what has already been placed in the child; vehicles with
    # no conflict-free parent are reinserted via PlaceVehicleOnCharger. Works
    # for any N >= 2, so the same routine serves NSGA-II/NSGA-III's 2-parent
    # recombination and MOPSO/MOGWO's 3-parent leader-guided moves alike --
    # unlike the previous exchange_schedules(schedule1, schedule2), which
    # only ever looked at the first two array entries and silently dropped
    # the third parent (MOPSO's guide, MOGWO's delta leader) whenever it was
    # called with 3.
    N = len(solutions)
    n = solutions[0].shape[0]
    m = len(chargers_temp)
    new_solution = np.zeros(solutions[0].shape, dtype=np.int32)

    vehicles_indexes = np.arange(n)
    np.random.shuffle(vehicles_indexes)

    placed = np.zeros(n, dtype=np.int8)
    not_placed = np.zeros(n, dtype=np.int64)
    n_not_placed = 0
    first_one = True

    for vi in range(n):
        i = vehicles_indexes[vi]

        if first_one:
            s = np.random.randint(N)
            new_solution[i] = solutions[s][i]
            placed[i] = 1
            first_one = False
            continue

        placable = np.ones(N, dtype=np.int8)
        for s in range(N):
            cj = solutions[s][i, 0]
            cs = solutions[s][i, 1]
            ce = solutions[s][i, 2]
            for j in range(n):
                if placed[j] == 0:
                    continue
                if new_solution[j, 0] != cj:
                    continue
                if cs > new_solution[j, 2] or new_solution[j, 1] > ce:
                    continue
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
                        new_solution[i] = solutions[s][i]
                        placed[i] = 1
                        break
                    count += 1
        else:
            not_placed[n_not_placed] = i
            n_not_placed += 1

    for k in range(n_not_placed):
        i = not_placed[k]
        candidate_chargers = np.arange(m)[delta[i, :] == 1]
        j = np.random.choice(candidate_chargers)
        new_solution = place_vehicle_on_charger(i, j, new_solution, requests_temp, chargers_temp, slot_budget, sigma)

    return new_solution


@njit
def _reinsert_random_vehicle(schedule, requests_temp, chargers_temp, delta, slot_budget, sigma, exclude_current):
    # Shared core of Mutation (Algorithm 2, exclude_current=True: forces a
    # DIFFERENT compatible charger) and FindNeighbour (Algorithm 3,
    # exclude_current=False: the current charger is allowed, since it is a
    # milder cuckoo-style re-randomization of session/timing rather than a
    # forced relocation).
    n, m = slot_budget.shape
    schedule = schedule.copy()

    candidate_vehicles = np.array([i for i in range(n) if np.sum(delta[i, :]) > 1], dtype=np.int32)
    if len(candidate_vehicles) == 0:
        return schedule

    i = np.random.choice(candidate_vehicles)
    if exclude_current:
        candidate_chargers = np.array(
            [j for j in range(m) if delta[i, j] == 1 and j != schedule[i, 0] - 1],
            dtype=np.int32,
        )
    else:
        candidate_chargers = np.array(
            [j for j in range(m) if delta[i, j] == 1],
            dtype=np.int32,
        )
    if len(candidate_chargers) == 0:
        return schedule

    j = np.random.choice(candidate_chargers)
    return place_vehicle_on_charger(i, j, schedule, requests_temp, chargers_temp, slot_budget, sigma)


@njit
def change_vehicle_charger(schedule, requests_temp, chargers_temp, delta, slot_budget, sigma):
    return _reinsert_random_vehicle(schedule, requests_temp, chargers_temp, delta, slot_budget, sigma, True)


@njit
def find_neighbour_move(schedule, requests_temp, chargers_temp, delta, slot_budget, sigma):
    return _reinsert_random_vehicle(schedule, requests_temp, chargers_temp, delta, slot_budget, sigma, False)


@njit
def objective_function(solution: np.ndarray, problem_params: tuple) -> np.ndarray:
    _, _, _, _, instance = problem_params
    requests_temp, required_energy, chargers_temp, delta, tau, n, m, scheduling_start_slot, slot_budget = instance

    schedule = np.asarray(solution, dtype=np.int32)
    T1 = int(np.min(schedule[:, 1]))
    T2 = int(np.max(schedule[:, 2]))
    Wg = np.zeros(T2 - T1 + 1, dtype=np.float64)

    for i in range(len(schedule)):
        for t in range(schedule[i, 1], schedule[i, 2] + 1):
            Wg[t - T1] += chargers_temp[schedule[i, 0] - 1, 1]

    peak_load = np.max(Wg)
    total_completion = np.sum(schedule[:, 2])
    return np.array([peak_load, total_completion], dtype=np.float64)


@njit
def Generate_Solution(problem_params: tuple) -> np.ndarray:
    _, lam, _, _, instance = problem_params
    requests_temp, required_energy, chargers_temp, delta, tau, n, m, scheduling_start_slot, slot_budget = instance
    sigma = max(1.0, float(lam))
    return generate_random_schedule(delta, requests_temp, chargers_temp, slot_budget, sigma=sigma)


@njit
def Perturb(solution: np.ndarray, problem_params: tuple) -> np.ndarray:
    """Mutation variant of \\textsc{Perturb} (Algorithm~\\ref{alg:perturb_gco}
    in appendix_biobjective.tex, exclude_current=True): each move relocates a vehicle to
    a DIFFERENT compatible charger. Used as the `mutate` hook for
    NSGA-II/NSGA-III/MOGWO/MOPSO."""
    perc_perturb, lam, _, _, instance = problem_params
    requests_temp, required_energy, chargers_temp, delta, tau, n, m, scheduling_start_slot, slot_budget = instance

    sigma = max(1.0, float(lam))
    n_moves = max(1, int(np.ceil(float(perc_perturb) * len(solution))))
    new_solution = np.asarray(solution, dtype=np.int32).copy()
    for _ in range(n_moves):
        new_solution = change_vehicle_charger(new_solution, requests_temp, chargers_temp, delta, slot_budget, sigma)
    return new_solution


@njit
def FindNeighbour(solution: np.ndarray, problem_params: tuple) -> np.ndarray:
    """Cuckoo variant of \\textsc{Perturb} (Algorithm~\\ref{alg:perturb_gco}
    in appendix_biobjective.tex, exclude_current=False): each move re-inserts a vehicle
    on a uniformly drawn compatible charger, INCLUDING its current one (a
    milder re-randomization than Mutation). Used as the `perturb` hook for
    MOCS's nest replacement and MOPSO's non-guided move."""
    perc_perturb, lam, _, _, instance = problem_params
    requests_temp, required_energy, chargers_temp, delta, tau, n, m, scheduling_start_slot, slot_budget = instance

    sigma = max(1.0, float(lam))
    n_moves = max(1, int(np.ceil(float(perc_perturb) * len(solution))))
    new_solution = np.asarray(solution, dtype=np.int32).copy()
    for _ in range(n_moves):
        new_solution = find_neighbour_move(new_solution, requests_temp, chargers_temp, delta, slot_budget, sigma)
    return new_solution


@njit
def Crossover(parents_array: np.ndarray, problem_params: tuple) -> np.ndarray:
    _, lam, _, _, instance = problem_params
    parents_array = np.asarray(parents_array, dtype=np.int32)
    if len(parents_array) == 1:
        return parents_array[0].copy()

    requests_temp, required_energy, chargers_temp, delta, tau, n, m, scheduling_start_slot, slot_budget = instance
    sigma = max(1.0, float(lam))
    return _generalized_crossover(parents_array, delta, requests_temp, chargers_temp, slot_budget, sigma)


def check_schedule_feasibility(schedule, instance, label="schedule", eps=None):
    requests_temp, required_energy, chargers_temp, delta, tau, n, m, scheduling_start_slot, slot_budget = instance
    eps = VERIFIER_EPS if eps is None else eps

    schedule = np.asarray(schedule, dtype=np.int32)

    assert schedule.shape == (n, 3), (
        f"[{label}] SHAPE FAIL -- expected {(n, 3)}, got {schedule.shape}."
    )

    for i in range(n):
        charger = int(schedule[i, 0])
        start = int(schedule[i, 1])
        end = int(schedule[i, 2])
        arrival = int(requests_temp[i, 0])

        if charger < 1 or charger > m:
            raise AssertionError(
                f"[{label}] CHARGER FAIL -- vehicle {i} charger {charger} outside [1,{m}]."
            )

        if delta[i, charger - 1] != 1:
            raise AssertionError(
                f"[{label}] COMPAT FAIL -- vehicle {i} incompatible with charger {charger}."
            )

        if start < arrival:
            raise AssertionError(
                f"[{label}] ARRIVAL FAIL -- vehicle {i} starts at {start} before arrival {arrival}."
            )

        if end < start:
            raise AssertionError(
                f"[{label}] INTERVAL FAIL -- vehicle {i} has end {end} < start {start}."
            )

        expected_len = int(slot_budget[i, charger - 1])
        actual_len = int(end - start + 1)
        if actual_len != expected_len:
            raise AssertionError(
                f"[{label}] LENGTH FAIL -- vehicle {i} length {actual_len} != slot_budget[{i},{charger - 1}]={expected_len}."
            )

        charger_open = int(chargers_temp[charger - 1, 0])
        if start < charger_open:
            raise AssertionError(
                f"[{label}] OPEN FAIL -- vehicle {i} starts at {start} before charger {charger} opens at {charger_open}."
            )

    for c in range(1, m + 1):
        assigned = []
        for i in range(n):
            if int(schedule[i, 0]) == c:
                assigned.append((int(schedule[i, 1]), int(schedule[i, 2]), i))
        assigned.sort(key=lambda x: x[0])

        for a in range(len(assigned)):
            s1, e1, i1 = assigned[a]
            for b in range(a + 1, len(assigned)):
                s2, e2, i2 = assigned[b]
                if not (e1 < s2 or e2 < s1):
                    raise AssertionError(
                        f"[{label}] OVERLAP FAIL -- charger {c}: vehicles {i1},{i2} "
                        f"intervals [{s1},{e1}] and [{s2},{e2}] overlap."
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# THE INSTANCE SUITE AND THE INTERFACE OF THE FRAMEWORKS
# ═══════════════════════════════════════════════════════════════════════════════

def n_chargers(n: int) -> int:
    return math.ceil(n / P.VEHICLES_PER_CHARGER)


def _load_or_mock_requests() -> np.ndarray:
    requests = REQUESTS_DATA
    if requests is None:
        requests = _load_requests_data()
    return np.asarray(requests, dtype=np.float64).copy()


def suite_instance(inst_id: int) -> tuple:
    """Instance `inst_id` (1-based) of the suite."""
    i = inst_id - 1
    n = P.SIZE_CLASSES[i // P.INSTANCES_PER_CLASS]
    return simulate_instance(requests=_load_or_mock_requests(), n=n, m=n_chargers(n), tau=P.DEFAULT_TAU,
                             seed=P.INSTANCE_SEED_BASE + i)


def generate_instances() -> List[tuple]:
    """The suite of the thesis: 20 instances, 5 per size, m = ceil(n / 3)."""
    return [suite_instance(k) for k in range(1, len(P.SIZE_CLASSES) * P.INSTANCES_PER_CLASS + 1)]


def instance_class(Instance: tuple) -> str:
    return P.CLASS_LABELS[P.SIZE_CLASSES.index(int(Instance[5]))]


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
    return {"n": float(Instance[5]),
            "m": float(Instance[6]),
            "energy_per_charger_kwh": float(np.sum(Instance[1]) / Instance[6]),
            "mean_power_kw": float(np.mean(Instance[2][:, 1]))}


def build_problem_params(params: Dict[str, Any], Instance: Any) -> tuple:
    """The `problem_params` tuple the operators expect, reading the tunable
    problem-level parameters out of `params`."""
    perc_perturb = float(params.get("perc_perturb", DEFAULT_PERC_PERTURB))
    lam = float(params.get("lam", DEFAULT_LAM))
    return (perc_perturb, lam, 0.0, 1, Instance)


def solution_shape(Instance: tuple) -> tuple:
    return (int(Instance[0].shape[0]), 3)


def evaluate_objective(instance: Any, solution: Any) -> np.ndarray:
    """The objectives of a schedule, in a single call to `objective_function`."""
    return objective_function(np.asarray(solution), build_problem_params({}, instance))
