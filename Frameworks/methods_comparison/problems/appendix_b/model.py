"""
model.py -- Appendix B of the thesis (appendix_constant_power.tex of the
manuscript): mono-objective constant-power EV charging scheduling.

The operators, the objectives, the instance sampling and the verifiers are the
code of the thesis, unchanged. What is adapted to the current frameworks is the
last section (the instance suite as a function, the interface of methods_comparison and of
det_vs_stoch) and the data file (acn_sessions.pkl: the same sessions as
requests_data.pkl). The parameters are in parameters.py. This folder is
identical in every framework that uses it.

Objectives   the unmet energy fraction, sum_i max(1 - tau p_j served_slots_i /
             e_i, 0), minimised.
"""

from __future__ import annotations

import math
import pickle
import warnings
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numba import njit
from scipy.stats import gaussian_kde

from . import parameters as P
from .parameters import *  # noqa: F401,F403  (the constants, under their names in the thesis code)

try:
    from pulp import (
        LpBinary,
        LpContinuous,
        LpMinimize,
        LpProblem,
        LpVariable,
        PULP_CBC_CMD,
        lpSum,
    )
except Exception:
    LpBinary = LpContinuous = LpMinimize = LpProblem = LpVariable = PULP_CBC_CMD = lpSum = None

PROBLEM_NAME = "appendix_b"
CODE = "appendix_b"
APPENDIX_B_DIR = Path(__file__).resolve().parent
OBJECTIVE_NAMES = ["Unmet charging fraction"]
REQUESTS_DATA = None


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL & INSTANCE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_requests_data() -> np.ndarray:
    global REQUESTS_DATA
    if REQUESTS_DATA is not None:
        return REQUESTS_DATA

    candidates = [
        Path(__file__).with_name("acn_sessions.pkl"),
        Path(__file__).resolve().parent.parent / "acn_sessions.pkl",
        Path.cwd() / "acn_sessions.pkl",
        Path.cwd() / "problems" / "acn_sessions.pkl",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                REQUESTS_DATA = pickle.load(f)
            return REQUESTS_DATA

    raise FileNotFoundError(
        "acn_sessions.pkl not found next to model.py, "
        "in the project root, or in ./problems/"
    )


def sample_request_rows(n: int, requests: np.ndarray, rng: np.random.Generator) -> np.ndarray:
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
    charger_powers: Optional[np.ndarray] = None,
    tau: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    if requests is None:
        requests = load_requests_data()
    else:
        requests = np.asarray(requests, dtype=np.float64).copy()

    if charger_powers is None:
        charger_powers = DEFAULT_CHARGING_POWERS

    rng = np.random.default_rng(seed)
    data = sample_request_rows(n, requests, rng)

    arrivals = np.array(np.ceil(data[:, 0] / tau), dtype=np.int32)
    departures = np.array(np.ceil(data[:, 1] / tau), dtype=np.int32)
    required_energy = np.asarray(data[:, 2], dtype=np.float64)

    departures = np.maximum(departures, arrivals)
    powers = rng.choice(np.asarray(charger_powers, dtype=np.float64), size=m, replace=True)
    r_ij = compute_rij(required_energy, powers, tau)

    max_r = int(np.max(r_ij))
    P_max = float(0.7 * np.sum(powers))
    t_min = int(np.min(arrivals))
    t_max = int(np.max(departures))

    return (
        arrivals,
        departures,
        required_energy,
        r_ij,
        powers,
        P_max,
        t_min,
        t_max,
        int(n),
        int(m),
        float(tau),
        int(max_r),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# OPERATORS (numba-compiled core + Generate/Perturb/Crossover/objective)
# ═══════════════════════════════════════════════════════════════════════════════

@njit
def energy_consumption(schedule: np.ndarray, instance: tuple) -> np.ndarray:
    arrivals, departures, required_energy, r_ij, powers, P_max, t_min, t_max, n, m, tau, max_r = instance
    P_t = np.zeros(t_max - t_min + 1, dtype=np.float64)

    for i in range(n):
        if schedule[i, 0] > 0:
            j = schedule[i, 0] - 1
            for k in range(3, schedule.shape[1]):
                t = schedule[i, k]
                if t == 0:
                    break
                if t_min <= t <= t_max:
                    P_t[t - t_min] += powers[j]
    return P_t


@njit
def find_availability(j: int, schedule: np.ndarray) -> np.ndarray:
    sub_schedule = schedule[schedule[:, 0] == j + 1]
    sub_schedule = sub_schedule[np.argsort(sub_schedule[:, 1])]
    N = len(sub_schedule)

    availability = np.zeros((N + 1, 2), dtype=np.int32)
    if N > 0:
        availability[0] = np.array([1, sub_schedule[0, 1] - 1], dtype=np.int32)
        for k in range(1, N):
            availability[k] = np.array([sub_schedule[k - 1, 2] + 1, sub_schedule[k, 1] - 1], dtype=np.int32)
        availability[N] = np.array([sub_schedule[N - 1, 2] + 1, INFINITY], dtype=np.int32)
    else:
        availability[0] = np.array([1, INFINITY], dtype=np.int32)

    return availability


@njit
def find_candidate_charging_periods(i: int, j: int, availability: np.ndarray, instance: tuple) -> np.ndarray:
    arrivals, departures, required_energy, r_ij, powers, P_max, t_min, t_max, n, m, tau, max_r = instance
    candidate = []
    for k in range(len(availability)):
        av_s, av_e = availability[k]
        if av_s <= departures[i] <= av_e:
            candidate.append(k)
    return np.array(candidate, dtype=np.int32)


@njit
def allocate_charging(
    schedule: np.ndarray,
    P_t: np.ndarray,
    i: int,
    j: int,
    candidate_periods: np.ndarray,
    availability: np.ndarray,
    instance: tuple,
) -> tuple:
    arrivals, departures, required_energy, r_ij, powers, P_max, t_min, t_max, n, m, tau, max_r = instance

    schedule = schedule.copy()
    P_t = P_t.copy()

    current_charger = schedule[i, 0] - 1
    if current_charger >= 0:
        for k in range(3, schedule.shape[1]):
            t = schedule[i, k]
            if t == 0:
                break
            P_t[t - t_min] -= powers[current_charger]

    schedule[i, :] = 0

    k0 = candidate_periods[0]
    av_s, av_e = availability[k0]
    av_s = max(av_s, arrivals[i])
    av_e = min(av_e, departures[i])

    considered = False
    write_k = 3
    to_allocate = min(r_ij[i, j], av_e - av_s + 1)
    t_idx = np.arange(av_s, av_e + 1)

    while to_allocate > 0:
        feasible_idx = t_idx[P_t[t_idx - t_min] + powers[j] <= P_max]
        if len(feasible_idx) > 0:
            considered = True
            t = feasible_idx[np.random.randint(len(feasible_idx))]
            schedule[i, write_k] = t
            write_k += 1
            P_t[t - t_min] += powers[j]
            t_idx = t_idx[t_idx != t]
            to_allocate -= 1
        else:
            break

    if considered:
        active = np.sort(schedule[i, 3:write_k])
        schedule[i, 0] = j + 1
        schedule[i, 1] = active[0]
        schedule[i, 2] = departures[i]
        schedule[i, 3:3 + len(active)] = active

    return schedule, P_t


@njit
def place_vehicle(i: int, schedule: np.ndarray, instance: tuple) -> np.ndarray:
    arrivals, departures, required_energy, r_ij, powers, P_max, t_min, t_max, n, m, tau, max_r = instance
    schedule = schedule.copy()

    # Algorithm 1 (StochasticInsertion), step 1: S_i <- 0. This must happen
    # BEFORE idle intervals are identified, otherwise a vehicle already
    # occupying charger j blocks its own re-insertion onto that same charger
    # (its own old window is seen as "occupied" by find_availability), which
    # would spuriously reject re-placing a perturbed vehicle back onto a
    # charger that is, in fact, idle once its own prior slots are freed.
    if schedule[i, 0] > 0:
        schedule[i, :] = 0

    P_t = energy_consumption(schedule, instance)

    charger_idx = np.arange(m)
    np.random.shuffle(charger_idx)

    for j in charger_idx:
        availability = find_availability(j, schedule)
        candidate = find_candidate_charging_periods(i, j, availability, instance)
        if len(candidate) > 0:
            schedule, _ = allocate_charging(schedule, P_t, i, j, candidate, availability, instance)
            break

    return schedule


@njit
def generate_schedule(instance: tuple) -> np.ndarray:
    arrivals, departures, required_energy, r_ij, powers, P_max, t_min, t_max, n, m, tau, max_r = instance

    schedule = np.zeros((n, 3 + max_r), dtype=np.int32)
    vehicles = np.arange(n)
    np.random.shuffle(vehicles)

    for i in vehicles:
        schedule = place_vehicle(i, schedule, instance)
    return schedule


@njit
def FN(schedule: np.ndarray, instance: tuple, perc_perturb: float) -> np.ndarray:
    arrivals, departures, required_energy, r_ij, powers, P_max, t_min, t_max, n, m, tau, max_r = instance
    schedule = schedule.copy()

    vehicles = np.arange(n)
    np.random.shuffle(vehicles)
    n_moves = max(1, int(np.ceil(perc_perturb * n)))

    for i in vehicles[:n_moves]:
        schedule = place_vehicle(i, schedule, instance)
    return schedule


@njit
def generalized_crossover_from_schedules(parents: np.ndarray, instance: tuple) -> np.ndarray:
    arrivals, departures, required_energy, r_ij, powers, P_max, t_min, t_max, n, m, tau, max_r = instance

    N = len(parents)
    child = np.zeros((n, 3 + max_r), dtype=np.int32)
    P_t = np.zeros(t_max - t_min + 1, dtype=np.float64)

    vehicles = np.arange(n)
    np.random.shuffle(vehicles)
    not_placed = []

    for i in vehicles:
        k = np.random.randint(N)
        current = parents[k]

        if current[i, 0] > 0:
            sol_start, sol_end = current[i, 1], current[i, 2]
            can_place = True

            for v2 in range(n):
                if child[v2, 0] > 0 and v2 != i:
                    child_start, child_end = child[v2, 1], child[v2, 2]
                    same_charger = child[v2, 0] == current[i, 0]
                    overlap = (sol_start <= child_start <= sol_end) or (child_start <= sol_start <= child_end)
                    if same_charger and overlap:
                        can_place = False
                        not_placed.append(i)
                        break

            if can_place:
                P_t_candidate = P_t.copy()
                for c in range(3, current.shape[1]):
                    t = current[i, c]
                    if t == 0:
                        break
                    P_t_candidate[t - t_min] += powers[current[i, 0] - 1]
                if np.max(P_t_candidate) > P_max:
                    can_place = False
                    not_placed.append(i)

            if can_place:
                child[i] = current[i].copy()
                for c in range(3, current.shape[1]):
                    t = current[i, c]
                    if t == 0:
                        break
                    P_t[t - t_min] += powers[current[i, 0] - 1]
        else:
            not_placed.append(i)

    for i in not_placed:
        child = place_vehicle(i, child, instance)

    return child


def check_schedule(instance: tuple, schedule: np.ndarray) -> None:
    arrivals, departures, required_energy, r_ij, powers, P_max, t_min, t_max, n, m, tau, max_r = instance
    P_t = energy_consumption(schedule, instance)

    if schedule.shape != (n, 3 + max_r):
        raise ValueError(f"Wrong schedule shape {schedule.shape}, expected {(n, 3 + max_r)}")

    for i in range(n):
        if schedule[i, 0] > 0:
            j, ts, te = schedule[i, :3]

            if not (1 <= j <= m):
                raise ValueError(f"Invalid charger index for EV {i + 1}")

            if ts < arrivals[i] or te > departures[i]:
                raise ValueError(f"Availability not respected for EV {i + 1}")

            for i2 in range(i + 1, n):
                if schedule[i2, 0] > 0:
                    j2, ts2, te2 = schedule[i2, :3]
                    if j == j2 and ((ts2 <= ts <= te2) or (ts <= ts2 <= te)):
                        raise ValueError(f"Overlapping periods {i + 1} / {i2 + 1}")

            served = int(np.sum(schedule[i, 3:] > 0))
            if served > r_ij[i, j - 1]:
                raise ValueError(f"Too many charging slots for EV {i + 1}")

            if schedule[i, 2] != departures[i]:
                raise ValueError(f"Unplug time differs from departure for EV {i + 1}")
        else:
            if np.any(schedule[i, 1:] != 0):
                raise ValueError(f"Unassigned EV {i + 1} has nonzero fields")

    if np.max(P_t) > P_max + 1e-9:
        raise ValueError("Grid overload")


@njit
def objective_function(schedule: np.ndarray, instance: tuple) -> float:
    arrivals, departures, required_energy, r_ij, powers, P_max, t_min, t_max, n, m, tau, max_r = instance
    total_unmet_fraction = 0.0
    for i in range(n):
        if schedule[i, 0] > 0:
            j = schedule[i, 0] - 1
            served = 0
            for c in range(3, schedule.shape[1]):
                if schedule[i, c] > 0:
                    served += 1
            delivered_energy = tau * powers[j] * served
            unmet = 1.0 - delivered_energy / required_energy[i]
            total_unmet_fraction += unmet if unmet > 0.0 else 0.0
        else:
            total_unmet_fraction += 1.0
    return float(total_unmet_fraction)


@njit
def GenerateSolution(problem_params: tuple) -> np.ndarray:
    perc_perturb, instance = problem_params
    return generate_schedule(instance)


@njit
def Perturb(solution: np.ndarray, problem_params: tuple) -> np.ndarray:
    perc_perturb, instance = problem_params
    return FN(np.asarray(solution, dtype=np.int32), instance, float(perc_perturb))


@njit
def Crossover(parents_array: np.ndarray, problem_params: tuple) -> np.ndarray:
    perc_perturb, instance = problem_params
    parents_array = np.asarray(parents_array, dtype=np.int32)
    if len(parents_array) == 1:
        return parents_array[0].copy()
    return generalized_crossover_from_schedules(parents_array, instance)


@njit
def objectivefunction(solution: np.ndarray, problem_params: tuple) -> float:
    perc_perturb, instance = problem_params
    return objective_function(np.asarray(solution, dtype=np.int32), instance)


def ev_charging_milp(instance: tuple, timelimit: int = 60) -> dict:
    if LpProblem is None:
        raise ImportError("pulp is required for the exact MILP solver.")

    arrivals, departures, required_energy, r_ij, powers, P_max, t_min, t_max, n, m, tau, max_r = instance

    I = range(n)
    J = range(m)
    H = range(int(np.min(arrivals)), int(np.max(departures)) + 1)
    A = {i: range(arrivals[i], departures[i] + 1) for i in I}

    prob = LpProblem("EV_Charging_Scheduling_MILP_Linear", LpMinimize)

    x = LpVariable.dicts(
        "x",
        ((i, j, t) for i in I for j in J for t in H),
        lowBound=0,
        upBound=1,
        cat=LpBinary,
    )

    # Epigraph linearization of the chapter's true (non-linear, per-vehicle
    # capped) objective f_i = max(1 - tau*p_j*K_i/e_i, 0), K_i = sum_t x_{i,j,t}
    # for whichever charger j vehicle i ends up on. u_i >= 0 together with
    # u_i >= 1 - sum_j (tau*p_j/e_i) sum_t x_{i,j,t} forces u_i == f_i exactly
    # at optimality (minimizing sum u_i always drives each u_i down to the
    # larger of the two bounds) -- exact for every instance, not just when
    # e_i happens to be a multiple of tau*p_j (unlike the old 1/r_ij
    # surrogate, which only coincided with f_i in that non-generic case).
    u = LpVariable.dicts("u", (i for i in I), lowBound=0, upBound=1, cat=LpContinuous)

    prob += lpSum(u[i] for i in I)

    for i in I:
        prob += u[i] >= 1.0 - lpSum(
            (tau * powers[j] / required_energy[i]) * x[(i, j, t)]
            for j in J for t in A[i]
        )

    for j in J:
        for t in H:
            prob += lpSum(x[(i, j, t)] for i in I) <= 1

    for i in I:
        for t in H:
            prob += lpSum(x[(i, j, t)] for j in J) <= 1

    for j in J:
        for i in I:
            for i2 in I:
                if i2 != i:
                    for t in A[i]:
                        for t2 in A[i]:
                            if t2 >= t:
                                prob += x[(i, j, t)] + x[(i2, j, t2)] <= 1

    for i in I:
        for j in J:
            for j2 in J:
                if j2 != j:
                    for t in A[i]:
                        for t2 in A[i]:
                            if t2 > t:
                                prob += x[(i, j, t)] + x[(i, j2, t2)] <= 1

    for i in I:
        for j in J:
            prob += lpSum(x[(i, j, t)] for t in A[i]) <= r_ij[i][j]

    for t in H:
        prob += lpSum(powers[j] * x[(i, j, t)] for i in I for j in J) <= P_max

    for i in I:
        for j in J:
            for t in H:
                if t not in A[i]:
                    prob += x[(i, j, t)] == 0

    solver = PULP_CBC_CMD(msg=False, timeLimit=timelimit)
    prob.solve(solver)

    from pulp import LpStatus
    if LpStatus[prob.status] != "Optimal":
        warnings.warn(
            f"ev_charging_milp: CBC returned status '{LpStatus[prob.status]}' "
            f"(not 'Optimal') within timelimit={timelimit}s -- the returned "
            f"schedule/objective may not be provably optimal, and the raw "
            f"relaxation values reported by the solver at cutoff can even be "
            f"fractional despite x/u being declared binary/bounded. Increase "
            f"timelimit for a trustworthy result on this instance size."
        )

    schedule = np.zeros((n, 3 + max_r), dtype=np.int32)

    for i in I:
        chosen_j = -1
        charging_slots = []

        for j in J:
            temp = []
            for t in H:
                v = x[(i, j, t)].varValue
                if v is not None and v > 0.5:
                    temp.append(t)
            if len(temp) > 0:
                chosen_j = j
                charging_slots = temp
                break

        if chosen_j >= 0:
            schedule[i, 0] = chosen_j + 1
            schedule[i, 1] = int(np.min(charging_slots))
            schedule[i, 2] = int(departures[i])
            schedule[i, 3:3 + len(charging_slots)] = np.array(charging_slots, dtype=np.int32)

    # Recomputed from the thresholded (>0.5) extracted schedule via the
    # chapter's own objective_function, rather than trusting value(prob.objective)
    # directly -- the two agree whenever CBC reaches a genuine integer-optimal
    # solution, but at a hard timeLimit cutoff CBC can report "Optimal" while
    # x/u still hold fractional relaxation values, silently desynchronizing
    # the solver's raw objective from the schedule actually returned.
    best_objective = float(objective_function(schedule, instance))

    return {
        "best_solution": schedule,
        "best_objective": best_objective,
    }


def _load_or_mock_requests_hours(seed=None):
    return load_requests_data()


def check_schedule_feasibility(schedule, instance, label="schedule"):
    schedule = np.asarray(schedule, dtype=np.int32)

    if schedule.ndim != 2:
        raise ValueError(f"{label}: schedule must be 2D, got shape {schedule.shape}")

    check_schedule(instance, schedule)

    arrivals, departures, required_energy, r_ij, powers, P_max, t_min, t_max, n, m, tau, max_r = instance

    if schedule.shape != (n, 3 + max_r):
        raise ValueError(f"{label}: wrong shape {schedule.shape}, expected {(n, 3 + max_r)}")

    P_t = energy_consumption(schedule, instance)
    if len(P_t) != (t_max - t_min + 1):
        raise ValueError(f"{label}: invalid load profile length")

    if np.any(P_t < -1e-9):
        raise ValueError(f"{label}: negative power values in load profile")

    if np.max(P_t) > P_max + 1e-9:
        raise ValueError(f"{label}: load exceeds P_max")

    for i in range(n):
        row = schedule[i]
        charger = row[0]

        if charger == 0:
            if np.any(row[1:] != 0):
                raise ValueError(f"{label}: unassigned EV {i + 1} has nonzero fields")
            continue

        if not (1 <= charger <= m):
            raise ValueError(f"{label}: invalid charger index for EV {i + 1}")

        ts = row[1]
        te = row[2]
        slots = row[3:][row[3:] > 0]

        if len(slots) == 0:
            raise ValueError(f"{label}: assigned EV {i + 1} has no charging slots")

        if ts != np.min(slots):
            raise ValueError(f"{label}: EV {i + 1} start slot inconsistent with slots")

        if te != departures[i]:
            raise ValueError(f"{label}: EV {i + 1} departure slot inconsistent")

        if np.any(slots < arrivals[i]) or np.any(slots > departures[i]):
            raise ValueError(f"{label}: EV {i + 1} charges outside availability window")

        if len(np.unique(slots)) != len(slots):
            raise ValueError(f"{label}: duplicate charging slots for EV {i + 1}")

        if len(slots) > r_ij[i, charger - 1]:
            raise ValueError(f"{label}: too many charging slots for EV {i + 1}")


# ═══════════════════════════════════════════════════════════════════════════════
# THE INSTANCE SUITE AND THE INTERFACE OF THE FRAMEWORKS
# ═══════════════════════════════════════════════════════════════════════════════

def n_chargers(n: int) -> int:
    return math.ceil(n / P.VEHICLES_PER_CHARGER)


def suite_instance(inst_id: int) -> tuple:
    """Instance `inst_id` (1-based) of the suite."""
    i = inst_id - 1
    n = P.SIZE_CLASSES[i // P.INSTANCES_PER_CLASS]
    return simulate_instance(n=n, m=n_chargers(n), tau=P.DEFAULT_TAU, charger_powers=None,
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
    return {"n": float(Instance[8]),
            "m": float(Instance[9]),
            "energy_per_charger_kwh": float(np.sum(Instance[2]) / Instance[9]),
            "mean_power_kw": float(np.mean(Instance[4]))}


def build_problem_params(params: Dict[str, Any], instance: tuple) -> tuple:
    return (float(params.get("perc_perturb", DEFAULT_PERC_PERTURB)), instance)


def solution_shape(Instance: tuple) -> tuple:
    return (int(Instance[8]), int(3 + Instance[11]))


def evaluate_objective(instance: tuple, solution: np.ndarray) -> tuple:
    """The objective of a schedule, as a 1-tuple (mono-objective, but the contract
    is uniform across problems)."""
    return (float(objective_function(np.asarray(solution, dtype=np.int32), instance)),)
