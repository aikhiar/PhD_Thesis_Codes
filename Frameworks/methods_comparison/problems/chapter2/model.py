"""
model.py -- Chapter 2 of the thesis (chapter8.tex of the manuscript):
bi-objective EV charging scheduling under stochastic vehicle arrivals.

The operators, the objectives, the instance sampling and the verifiers are the
code of the thesis, unchanged. What is adapted to the current frameworks is the
last section (the instance suite as a function, the interface of methods_comparison and of
det_vs_stoch) and the data file (acn_sessions.pkl: the same sessions as
requests_data.pkl). The parameters are in parameters.py. This folder is
identical in every framework that uses it.

Objectives   (peak load kW, mean relative tardiness), expectations over the
             attendance of the vehicles (Monte Carlo with N_r replications for
             the peak load) on a stochastic instance.
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

PROBLEM_NAME = "chapter2"
CODE = "chapter2"
CHAPTER2_DIR = Path(__file__).resolve().parent
OBJECTIVE_NAMES = ["Peak load (kW)", "Avg tardiness"]
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
def compute_rij(required_energy: np.ndarray, chargers: np.ndarray, tau: float) -> np.ndarray:
    n, m = len(required_energy), len(chargers)
    r = np.zeros((n, m), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            r[i, j] = int(np.ceil(required_energy[i] / (tau * chargers[j])))
    return r


def simulate_instance(
    requests: Optional[np.ndarray] = None,
    n: int = 20,
    m: int = 10,
    chargers_powers: Optional[np.ndarray] = None,
    tau: float = 0.1,
    seed: Optional[int] = None,
    stochastic: bool = False,
) -> tuple:
    if requests is None:
        requests = _load_requests_data()

    if chargers_powers is None:
        chargers_powers = DEFAULT_CHARGING_POWERS
    chargers_powers = np.asarray(chargers_powers, dtype=np.float64)

    rng = np.random.default_rng(seed)
    idx = _sample_from_requests(n, requests, rng)
    data = requests[idx]

    arrival_time_raw = np.asarray(data[:, 0], dtype=np.float64)
    departure_time_raw = np.asarray(data[:, 1], dtype=np.float64)
    required_energy = np.asarray(data[:, 2], dtype=np.float64)

    arrival_time = np.int32(np.ceil(arrival_time_raw / tau))
    departure_time = np.int32(np.ceil(departure_time_raw / tau))
    departure_time = np.maximum(departure_time, arrival_time)

    chargers = np.zeros(m, dtype=np.float64)
    chargers[:] = rng.choice(chargers_powers, size=m, replace=True)

    P_max = 0.7 * np.sum(chargers)
    r_ij = compute_rij(required_energy, chargers, tau)
    t_min = int(np.min(arrival_time))
    t_max = int(np.max(departure_time))
    max_r = int(np.max(r_ij))

    # rho[i]: probability vehicle i actually shows up. Deterministic mode
    # (stochastic=False) is exact full-attendance (rho=1), matching the
    # non-Monte-Carlo _objective_deterministic path; stochastic mode uses a
    # synthetic constant (see SYNTHETIC_RHO) until real per-vehicle
    # probabilities are calibrated.
    rho = np.full(n, SYNTHETIC_RHO if stochastic else 1.0, dtype=np.float64)

    Instance = (
        arrival_time, departure_time, required_energy, r_ij, chargers,
        P_max, t_min, t_max, n, m, tau, max_r, rho, stochastic,
    )
    return Instance


# ============================================================================
# NUMBA HELPER FUNCTIONS
# ============================================================================

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
def find_availability(j, schedule, Instance):
    arrival_time, departure_time, required_energy, r_ij, chargers, P_max, t_min, t_max, n, m, tau, max_r, rho, stochastic = Instance
    sub_schedule = schedule[schedule[:, 0] == j + 1]
    sub_schedule = sub_schedule[np.argsort(sub_schedule[:, 1])]
    N = len(sub_schedule)
    availability = np.zeros((N + 1, 2), dtype=np.int64)
    if N > 0:
        availability[0] = np.array([1, sub_schedule[0, 1] - 1])
        for k in range(1, N):
            availability[k] = np.array([sub_schedule[k - 1, 2] + 1, sub_schedule[k, 1] - 1])
        availability[N] = np.array([sub_schedule[N - 1, 2] + 1, INFINITY])
    else:
        availability[0] = np.array([1, INFINITY])
    return availability


@njit
def find_candidate_charging_periods(i, j, availability, Instance):
    arrival_time, departure_time, required_energy, r_ij, chargers, P_max, t_min, t_max, n, m, tau, max_r, rho, stochastic = Instance
    candidate_charging_periods = []
    for k in range(len(availability)):
        cond1 = availability[k, 1] == INFINITY
        cond2 = availability[k, 1] - max(availability[k, 0], arrival_time[i]) + 1 >= r_ij[i, j]
        if cond1 or cond2:
            candidate_charging_periods.append(k)
    return np.array(candidate_charging_periods)


@njit
def place_vehicle_on_charger(i, j, schedule, lam, Instance):
    arrival_time, departure_time, required_energy, r_ij, chargers, P_max, t_min, t_max, n, m, tau, max_r, rho, stochastic = Instance
    schedule = schedule.copy()

    availability = find_availability(j, schedule, Instance)
    candidate_charging_periods = find_candidate_charging_periods(i, j, availability, Instance)

    N = len(candidate_charging_periods)
    if N == 0:
        ts = max(1, arrival_time[i])
        te = ts + r_ij[i, j] - 1
        schedule[i] = [j + 1, ts, te]
        return schedule

    probabilities = (N - np.arange(N)) / (N * (N + 1) / 2)
    k = int(random_choice(candidate_charging_periods, p=probabilities)[0])

    A = max(availability[k, 0], arrival_time[i])
    if availability[k, 1] == INFINITY:
        ts = A + int(np.abs(lam * np.random.randn()))
    else:
        upper = int(availability[k, 1]) - r_ij[i, j] + 2 - int(A)
        if upper <= 0:
            upper = 1
        ts = int(A) + np.random.randint(0, upper)

    schedule[i] = [j + 1, ts, ts + r_ij[i, j] - 1]
    return schedule


@njit
def generate_random_schedule(lam, Instance):
    arrival_time, departure_time, required_energy, r_ij, chargers, P_max, t_min, t_max, n, m, tau, max_r, rho, stochastic = Instance
    schedule = np.zeros((n, 3), dtype=np.int32)
    vehicles_indexes = np.arange(n)
    np.random.shuffle(vehicles_indexes)
    for idx in range(n):
        i = vehicles_indexes[idx]
        j = np.random.randint(m)
        schedule = place_vehicle_on_charger(i, j, schedule, lam, Instance)
    return schedule


@njit
def _FN(schedule, perc_perturb, lam, Instance):
    arrival_time, departure_time, required_energy, r_ij, chargers, P_max, t_min, t_max, n, m, tau, max_r, rho, stochastic = Instance
    schedule = schedule.copy()
    vehicles_idx = np.arange(n)
    np.random.shuffle(vehicles_idx)
    for idx in range(min(int(ceil(perc_perturb * n)), n)):
        i = vehicles_idx[idx]
        j = np.random.randint(m)
        schedule = place_vehicle_on_charger(i, j, schedule, lam, Instance)
    return schedule


@njit
def _crossover(solutions, lam, Instance):
    arrival_time, departure_time, required_energy, r_ij, chargers, P_max, t_min, t_max, n, m, tau, max_r, rho, stochastic = Instance
    N = len(solutions)
    child = np.zeros((n, 3), dtype=np.int32)
    not_placed = np.zeros(n, dtype=np.int64)
    n_not_placed = 0
    for i in range(n):
        k = np.random.randint(N)
        if i == 0:
            child[i] = solutions[k][i].copy()
        else:
            sol_ts = solutions[k][i, 1]
            sol_te = solutions[k][i, 2]
            place_it = True
            for v in range(n):
                child_ts = child[v, 1]
                child_te = child[v, 2]
                cond1 = v != i
                cond2 = child[v, 0] == solutions[k][i, 0]
                cond3 = sol_ts <= child_ts <= sol_te or child_ts <= sol_ts <= child_te
                if cond1 and cond2 and cond3:
                    place_it = False
                    not_placed[n_not_placed] = i
                    n_not_placed += 1
                    break
            if place_it:
                child[i] = solutions[k][i].copy()

    for idx in range(n_not_placed):
        i = not_placed[idx]
        j = np.random.randint(m)
        child = place_vehicle_on_charger(i, j, child, lam, Instance)
    return child


@njit
def _objective_deterministic(schedule, Instance):
    arrival_time, departure_time, required_energy, r_ij, chargers, P_max, t_min, t_max, n, m, tau, max_r, rho, stochastic = Instance
    T1 = int(np.min(schedule[:, 1]))
    T2 = int(np.max(schedule[:, 2]))
    Wg = np.zeros(T2 - T1 + 1, dtype=np.float64)

    mean_tardiness = 0.0
    for i in range(n):
        j = int(schedule[i, 0]) - 1
        ts = int(schedule[i, 1])
        te = int(schedule[i, 2])
        mean_tardiness += max((te - departure_time[i]) / departure_time[i], 0.0)
        for t in range(ts, te + 1):
            Wg[t - T1] += chargers[j]

    mean_tardiness /= n
    return np.array([np.max(Wg), mean_tardiness])


@njit
def _objective_stochastic(schedule, probas, number_of_replications, Instance):
    arrival_time, departure_time, required_energy, r_ij, chargers, P_max, t_min, t_max, n, m, tau, max_r, rho, stochastic = Instance
    T1 = int(np.min(schedule[:, 1]))
    T2 = int(np.max(schedule[:, 2]))
    expected_peak_load = 0.0
    for _ in range(number_of_replications):
        Wg = np.zeros(T2 - T1 + 1, dtype=np.float64)
        active = np.random.rand(n) < probas
        for i in range(n):
            if active[i]:
                j = int(schedule[i, 0]) - 1
                ts = int(schedule[i, 1])
                te = int(schedule[i, 2])
                for t in range(ts, te + 1):
                    Wg[t - T1] += chargers[j]
        expected_peak_load += np.max(Wg)
    expected_peak_load /= number_of_replications

    expected_tardiness = 0.0
    for i in range(n):
        expected_tardiness += probas[i] * max((schedule[i, 2] - departure_time[i]) / departure_time[i], 0.0)
    expected_tardiness /= n

    return np.array([expected_peak_load, expected_tardiness])


# ============================================================================
# SOLVER-FACING FUNCTIONS
# ============================================================================

@njit
def objective_function(solution: np.ndarray, problem_params: tuple) -> np.ndarray:
    _, lam, _, number_of_replications, Instance = problem_params
    schedule = np.asarray(solution, dtype=np.int32)

    is_stochastic = bool(Instance[13])
    if is_stochastic:
        rho = np.asarray(Instance[12], dtype=np.float64)
        reps = max(1, int(number_of_replications))
        return _objective_stochastic(schedule, rho, reps, Instance)

    return _objective_deterministic(schedule, Instance)


@njit
def Generate_Solution(problem_params: tuple) -> np.ndarray:
    _, lam, _, _, Instance = problem_params
    return generate_random_schedule(lam, Instance)


@njit
def Perturb(solution: np.ndarray, problem_params: tuple) -> np.ndarray:
    perc_perturb, lam, _, _, Instance = problem_params
    return _FN(np.asarray(solution, dtype=np.int32), perc_perturb, lam, Instance)


@njit
def Crossover(parents_array: np.ndarray, problem_params: tuple) -> np.ndarray:
    _, lam, _, _, Instance = problem_params
    return _crossover(np.asarray(parents_array, dtype=np.int32), lam, Instance)


def check_schedule_feasibility(schedule, Instance, label="schedule", eps=None):
    arrival_time, departure_time, required_energy, r_ij, chargers, P_max, t_min, t_max, n, m, tau, max_r, rho, stochastic = Instance
    eps = VERIFIER_EPS if eps is None else eps

    schedule = np.asarray(schedule, dtype=np.int32)

    assert schedule.shape == (n, 3), (
        f"[{label}] SHAPE FAIL -- expected {(n, 3)}, got {schedule.shape}."
    )

    for i in range(n):
        charger = int(schedule[i, 0])
        start = int(schedule[i, 1])
        end = int(schedule[i, 2])

        if charger < 1 or charger > m:
            raise AssertionError(
                f"[{label}] CHARGER FAIL -- vehicle {i} charger {charger} outside [1,{m}]."
            )

        if end < start:
            raise AssertionError(
                f"[{label}] INTERVAL FAIL -- vehicle {i} has end {end} < start {start}."
            )

        if start < arrival_time[i]:
            raise AssertionError(
                f"[{label}] ARRIVAL FAIL -- vehicle {i} starts at {start} before arrival {arrival_time[i]}."
            )

        expected_len = int(r_ij[i, charger - 1])
        actual_len = int(end - start + 1)
        if actual_len != expected_len:
            raise AssertionError(
                f"[{label}] LENGTH FAIL -- vehicle {i} length {actual_len} != r_ij[{i},{charger - 1}]={expected_len}."
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


def suite_instance(inst_id: int, stochastic: bool = True) -> tuple:
    """Instance `inst_id` (1-based) of the suite. The two regimes of the
    deterministic-versus-stochastic comparison share the requests, the chargers
    and r_ij (same seed); only rho and the stochastic flag differ."""
    i = inst_id - 1
    n = P.SIZE_CLASSES[i // P.INSTANCES_PER_CLASS]
    return simulate_instance(requests=_load_or_mock_requests_hours(seed=99), n=n, m=n_chargers(n),
                             tau=P.DEFAULT_TAU, seed=P.INSTANCE_SEED_BASE + i, stochastic=stochastic)


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
    return {"n": float(Instance[8]),
            "m": float(Instance[9]),
            "energy_per_charger_kwh": float(np.sum(Instance[2]) / Instance[9]),
            "mean_power_kw": float(np.mean(Instance[4]))}


def build_problem_params(params: Dict[str, Any], Instance: Any) -> tuple:
    perc_perturb = float(params.get("perc_perturb", DEFAULT_PERC_PERTURB))
    lam = float(params.get("lam", DEFAULT_LAM))
    num_of_replications = int(params.get("num_of_replications", DEFAULT_NUM_REPLICATIONS))
    return (perc_perturb, lam, 0.0, num_of_replications, Instance)


def solution_shape(Instance: tuple) -> tuple:
    return (int(Instance[8]), 3)


def evaluate_objective(instance: Any, solution: Any) -> np.ndarray:
    """The objectives of a schedule, in a single call to `objective_function`."""
    return objective_function(np.asarray(solution, dtype=np.int32), build_problem_params({}, instance))
