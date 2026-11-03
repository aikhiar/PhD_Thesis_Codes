"""
Verbatim copy of the instance generator and of the algorithms of the earlier version of this work
(its original notebook, cells 1 and 4). Used only by faithfulness/check.py, to check that
the Numba re-implementations of G, RR, RLP, SA and the hybrids behave like the original code.
Do not edit (except this header).
"""
# flake8: noqa
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit
from pulp import (GUROBI, LpMinimize, LpProblem, LpStatus, LpVariable, PULP_CBC_CMD, lpSum, value)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ============================================================================
# INSTANCE GENERATION
# ============================================================================

def partition_set_completely_random(n: int, m: int) -> np.ndarray:
    """
    Randomly partition n tasks into m machines, guaranteeing every machine
    receives at least one task.

    Args:
        n: Number of tasks
        m: Number of machines

    Returns:
        partition: Array of size n with values in [1, m], dtype int32

    Raises:
        ValueError: If m > n or if n or m are non-positive
    """
    if m > n:
        raise ValueError(
            f"Number of machines (m={m}) cannot exceed number of tasks (n={n})")
    if m <= 0 or n <= 0:
        raise ValueError(
            f"Both n and m must be positive (n={n}, m={m})")

    partition = np.zeros(n, dtype=np.int32)
    partition[:m] = np.arange(1, m + 1, dtype=np.int32)
    partition[m:] = np.random.randint(1, m + 1, size=n - m, dtype=np.int32)
    return partition[np.random.permutation(n)]


def generate_instance(n: int = 10, T: int = 100, m: int = 10,
                      seed: Optional[int] = None) -> Tuple:
    """
    Generate a random PPMSP instance with machine constraints.
    Charger powers are drawn from standard Level 2 EV charger ratings
    (SAE J1772 North America + IEC 62196 Type 2 Europe).
    Processing times are derived from realistic EV battery energy demands.
    The SOC deficit is drawn adaptively to ensure p_i <= floor(0.8 * window_size)
    for every EV i, guaranteeing at least 20% scheduling slack per window.
    P_max is set to 70% of the total installed charging power.

    Args:
        n:    Number of tasks (EVs)
        T:    Time horizon (number of time slots)
        m:    Number of chargers
        seed: Random seed for reproducibility

    Returns:
        Tuple: (partition, start_time, end_time, processing_time,
                power_requirement, n, m, T, P_max)
    """
    LEVEL2_POWERS = np.array([3.3, 6.2, 7.2, 9.6, 11.5,
                               12.0, 14.4, 19.2, 22.0])          # kW
    EV_BATTERY_CAPACITIES_KWH = np.array([
        24.0, 30.0, 40.0, 52.0, 58.0, 62.0,
        64.0, 75.0, 77.0, 82.0, 90.0, 100.0, 105.0
    ])
    SLOT_DURATION_H = 24.0 / T

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    start_time        = np.zeros(n, dtype=np.int32)
    end_time          = np.zeros(n, dtype=np.int32)
    processing_time   = np.zeros(n, dtype=np.int32)
    power_requirement = np.zeros(n, dtype=np.float32)

    partition = partition_set_completely_random(n, m)

    machine_power = LEVEL2_POWERS[
        np.random.randint(0, len(LEVEL2_POWERS), size=m)
    ]

    for j in range(1, m + 1):
        machine_tasks    = np.where(partition == j)[0]
        N                = len(machine_tasks)
        charger_power_kw = machine_power[j - 1]

        if N == 0:
            continue

        if 2 * N > T:
            step = max(1, T // N)
            for k, i in enumerate(machine_tasks):
                s = k * step + 1
                e = min(s + step - 1, T)
                start_time[i], end_time[i] = s, e
        else:
            times = np.sort(
                np.random.choice(np.arange(1, T + 1), 2 * N, replace=False)
            ).reshape(N, 2)
            for k, i in enumerate(machine_tasks):
                start_time[i], end_time[i] = times[k]

        for i in machine_tasks:
            window_size          = end_time[i] - start_time[i] + 1
            battery_kwh          = np.random.choice(EV_BATTERY_CAPACITIES_KWH)
            target_slots         = max(1, int(np.floor(0.8 * window_size)))
            phi_max              = min(0.95,
                                       (target_slots * charger_power_kw
                                        * SLOT_DURATION_H) / battery_kwh)
            phi_min              = min(0.10, phi_max)
            soc_deficit          = np.random.uniform(phi_min, phi_max)
            energy_needed_kwh    = battery_kwh * soc_deficit
            raw_slots            = energy_needed_kwh / (charger_power_kw * SLOT_DURATION_H)
            processing_time[i]   = int(np.clip(int(np.ceil(raw_slots)), 1, target_slots))
            power_requirement[i] = charger_power_kw

    P_max = 0.7 * float(np.sum(machine_power))

    return (partition, start_time, end_time, processing_time,
            power_requirement, n, m, T, P_max)



# ============================================================================
# SCHEDULE GENERATION AND EVALUATION
# ============================================================================

@njit
def generate_schedule(instance: Tuple) -> np.ndarray:
    """
    Generate a random feasible schedule by independently sampling exactly
    p_i time slots uniformly at random from the feasibility window of each task.

    Args:
        instance: Problem instance tuple

    Returns:
        x: Feasible schedule matrix (n x T), dtype int32
    """
    partition, start_time, end_time, processing_time, \
        power_requirement, n, m, T, P_max = instance

    x = np.zeros((n, T), dtype=np.int32)
    for i in range(n):
        available = np.arange(start_time[i] - 1, end_time[i])  # 0-indexed
        p = processing_time[i]
        if len(available) >= p:
            chosen = np.random.choice(available, size=p, replace=False)
            x[i, chosen] = 1
        else:
            x[i, available] = 1   # fallback: fill entire window
    return x


@njit
def peak_load_computation(x: np.ndarray, instance: Tuple) -> Tuple:
    """
    Compute power consumption per slot and peak load.

    Args:
        x:        Schedule matrix (n x T), dtype int32
        instance: Problem instance tuple

    Returns:
        power_per_slot: (T,) array of power values, dtype float64
        peak_load:      scalar maximum (float)
    """
    partition, start_time, end_time, processing_time, \
        power_requirement, n, m, T, P_max = instance

    n              = x.shape[0]
    power_per_slot = np.zeros(T, dtype=np.float64)

    for t in range(T):
        for i in range(n):
            if x[i, t] == 1:
                power_per_slot[t] += power_requirement[i]

    peak_load = 0.0
    for t in range(T):
        if power_per_slot[t] > peak_load:
            peak_load = power_per_slot[t]

    return power_per_slot, peak_load


def evaluate_schedule(instance: Tuple, x: np.ndarray) -> Dict:
    """
    Comprehensive evaluation of a schedule against all PPMSP constraints.

    Feasibility requires simultaneously:
      (1) Each task scheduled for exactly p_i slots        [processing time]
      (2) No task scheduled outside its window [s_i, e_i] [time window]
      (3) No two tasks on the same machine active at once  [machine conflict]
      (4) No time slot exceeds P_max                       [power capacity]

    Args:
        instance: Problem instance tuple
        x:        Schedule matrix (n x T), dtype int32

    Returns:
        metrics: Dictionary of performance metrics
    """
    partition, start_time, end_time, processing_time, \
        power_requirement, n, m, T, P_max = instance

    # Power profile
    power_per_slot = np.sum(power_requirement[:, np.newaxis] * x, axis=0)
    peak_power     = float(np.max(power_per_slot)) if T > 0 else 0.0
    avg_power      = float(np.mean(power_per_slot)) if T > 0 else 0.0

    # Power violations
    violations    = int(np.sum(power_per_slot > P_max))
    max_violation = float(np.max(power_per_slot - P_max)) \
                    if violations > 0 else 0.0

    # Processing time
    total_scheduled = np.sum(x, axis=1)
    completion_rate = float(np.sum(total_scheduled == processing_time)) / n \
                      if n > 0 else 0.0

    # Time window violations
    allowed = np.zeros((n, T), dtype=np.int32)
    for i in range(n):
        allowed[i, start_time[i] - 1: end_time[i]] = 1
    window_violations = int(np.sum(x * (1 - allowed)))

    # Machine conflicts (vectorized)
    conflicts = 0
    if partition is not None and m > 0:
        for j in range(1, m + 1):
            mt = np.where(partition == j)[0]
            if len(mt) > 1:
                active_per_slot = np.sum(x[mt, :], axis=0)
                conflicts += int(np.sum(active_per_slot > 1))

    # Utilization
    total_work     = float(np.sum(x))
    utilization    = (total_work / (n * T) * 100.0) if n * T > 0 else 0.0

    is_feasible = (completion_rate   == 1.0 and
                   window_violations == 0   and
                   conflicts         == 0   and
                   violations        == 0)

    return {
        'power_per_slot':    power_per_slot,
        'peak_power':        peak_power,
        'avg_power':         avg_power,
        'violations':        violations,
        'max_violation':     max_violation,
        'window_violations': window_violations,
        'completion_rate':   completion_rate,
        'conflicts':         conflicts,
        'utilization':       utilization,
        'is_feasible':       is_feasible,
    }


# ============================================================================
# SCHEDULE VISUALIZATION
# ============================================================================

def plot_schedule(instance: Tuple, x: Optional[np.ndarray] = None,
                  figsize: Tuple[int, int] = (20, 15),
                  show_task_labels: bool = True,
                  show_power_values: bool = True,
                  show_machine_legend: bool = True,
                  filename: Optional[str] = None) -> Dict:
    """
    Comprehensive visualization of a PPMSP schedule:
    Gantt chart, power profile, completion status, and machine utilization.

    Args:
        instance:            Problem instance tuple
        x:                   Schedule (n x T); generates random if None
        figsize:             Figure size in inches
        show_task_labels:    Show task labels on Gantt chart
        show_power_values:   Annotate power violation values on profile
        show_machine_legend: Render the machine legend panel
        filename:            If provided, save figure to this path (PNG)

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    partition, start_time, end_time, processing_time, \
        power_requirement, n, m, T, P_max = instance

    if x is None:
        x = generate_schedule(instance)

    metrics = evaluate_schedule(instance, x)

    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(3, 3, height_ratios=[2.5, 1.5, 0.8],
                           hspace=0.35, wspace=0.25,
                           left=0.08, right=0.95, top=0.93, bottom=0.05)
    ax1    = fig.add_subplot(gs[0, :2])   # Gantt chart
    ax_leg = fig.add_subplot(gs[0, 2])    # Machine legend
    ax2    = fig.add_subplot(gs[1, :])    # Power profile
    ax3    = fig.add_subplot(gs[2, 0])    # Completion pie
    ax4    = fig.add_subplot(gs[2, 1])    # Metrics table
    ax5    = fig.add_subplot(gs[2, 2])    # Machine utilization

    colors = plt.cm.tab10(np.linspace(0, 1, max(m, 1)))

    # Build machine info
    machine_info = {}
    if partition is not None and m > 0:
        for j in range(1, m + 1):
            mt = np.where(partition == j)[0]
            if len(mt) > 0:
                machine_info[j] = {
                    'power':   power_requirement[mt[0]],
                    'n_tasks': len(mt),
                    'tasks':   mt,
                    'color':   colors[j - 1]
                }

    # ── Gantt Chart ──────────────────────────────────────────────────────────
    current_y = 0
    if partition is not None and m > 0:
        for j in range(1, m + 1):
            if j not in machine_info:
                continue
            info            = machine_info[j]
            mc              = info['color']
            machine_start_y = current_y

            for idx, i in enumerate(info['tasks']):
                y_pos = current_y + idx
                ax1.barh(y_pos, end_time[i] - start_time[i] + 1,
                         left=start_time[i], height=0.8,
                         color=mc, alpha=0.15, edgecolor='gray', linewidth=0.5)
                for t in range(start_time[i], end_time[i] + 1):
                    if x[i, t - 1] == 1:
                        ax1.barh(y_pos, 1, left=t, height=0.8,
                                 color=mc, alpha=0.9,
                                 edgecolor='black', linewidth=1.5)
                if show_task_labels:
                    ax1.text(-3, y_pos,
                             f'T{i:02d} (p={processing_time[i]:2d})',
                             va='center', ha='right',
                             fontsize=8, fontweight='bold')

            machine_end_y    = current_y + len(info['tasks'])
            machine_center_y = (machine_start_y + machine_end_y - 1) / 2

            if j > 1:
                ax1.axhline(y=machine_start_y - 0.4, color='black',
                            linewidth=2.5, alpha=0.7, zorder=10)

            ax1.text(-10, machine_center_y, f'M{j}',
                     va='center', ha='center',
                     fontsize=11, fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor=mc,
                               edgecolor='black', linewidth=2, alpha=0.9))
            ax1.text(-16, machine_center_y, f'{info["power"]:.1f} kW',
                     va='center', ha='right',
                     fontsize=9, fontweight='bold', color=mc)
            # ── fixed: \n not \\n ────────────────────────────────────────────
            ax1.text(T + 2, machine_center_y,
                     f'M{j}\n{info["power"]:.1f} kW\n{info["n_tasks"]} tasks',
                     va='center', ha='left',
                     fontsize=9, fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor=mc,
                               edgecolor='black', linewidth=1.5, alpha=0.8))
            current_y += len(info['tasks']) + 0.5
    else:
        for i in range(n):
            ax1.barh(i, end_time[i] - start_time[i] + 1,
                     left=start_time[i], height=0.8,
                     color='steelblue', alpha=0.15,
                     edgecolor='gray', linewidth=0.5)
            for t in range(start_time[i], end_time[i] + 1):
                if x[i, t - 1] == 1:
                    ax1.barh(i, 1, left=t, height=0.8,
                             color='steelblue', alpha=0.9,
                             edgecolor='black', linewidth=1.5)
            if show_task_labels:
                ax1.text(-3, i, f'T{i} (p={processing_time[i]})',
                         va='center', ha='right', fontsize=8, fontweight='bold')
        current_y = n

    ax1.set_xlabel('Time Slot', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Tasks (Grouped by Machine)', fontsize=13, fontweight='bold')
    ax1.set_title('Task Scheduling Gantt Chart', fontsize=15,
                  fontweight='bold', pad=15)
    ax1.set_xlim(-20, T + 14)
    ax1.set_ylim(-0.5, current_y)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_yticks([])

    # ── Machine Legend ───────────────────────────────────────────────────────
    ax_leg.axis('off')
    if show_machine_legend and machine_info:
        ax_leg.text(0.5, 0.97, 'MACHINES', ha='center', va='top',
                    fontsize=13, fontweight='bold',
                    transform=ax_leg.transAxes)
        ax_leg.plot([0.05, 0.95], [0.91, 0.91], 'k-',
                    linewidth=2, transform=ax_leg.transAxes)
        n_machines = len(machine_info)
        row_h      = min(0.12, 0.85 / max(n_machines, 1))
        y          = 0.88
        for j in sorted(machine_info.keys()):
            info = machine_info[j]
            rect = plt.Rectangle((0.05, y - 0.6 * row_h), 0.18, 0.75 * row_h,
                                  facecolor=info['color'], edgecolor='black',
                                  linewidth=2, transform=ax_leg.transAxes,
                                  clip_on=False)
            ax_leg.add_patch(rect)
            ax_leg.text(0.14, y - 0.12 * row_h, f'M{j}',
                        ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white',
                        transform=ax_leg.transAxes)
            ax_leg.text(0.30, y - 0.12 * row_h,
                        f'{info["power"]:.1f} kW  |  {info["n_tasks"]} tasks',
                        ha='left', va='center', fontsize=9,
                        transform=ax_leg.transAxes)
            y -= row_h + 0.02

    # ── Power Profile ────────────────────────────────────────────────────────
    power_per_slot = metrics['power_per_slot']
    bar_colors     = ['#d62728' if p > P_max else '#2ca02c'
                      for p in power_per_slot]
    ax2.bar(range(T), power_per_slot, color=bar_colors,
            alpha=0.75, edgecolor='black', linewidth=0.4)
    ax2.axhline(y=P_max, color='#1f77b4', linestyle='--', linewidth=3,
                # ── fixed: single backslash for LaTeX in f-string ────────────
                label=f'$P_{{\\max}}$ = {P_max:.1f} kW', zorder=10)

    if show_power_values:
        violation_slots = [t for t in range(T) if power_per_slot[t] > P_max]
        if len(violation_slots) <= 20:
            # ── fixed: np.max instead of built-in max on numpy array ─────────
            y_offset = 0.02 * float(np.max(power_per_slot)) \
                       if len(power_per_slot) > 0 else 1.0
            for t in violation_slots:
                ax2.text(t, power_per_slot[t] + y_offset,
                         f'{power_per_slot[t]:.1f}',
                         ha='center', va='bottom',
                         fontsize=7, fontweight='bold', color='red')

    ax2.set_xlabel('Time Slot', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Power (kW)', fontsize=13, fontweight='bold')
    status_text  = 'FEASIBLE ✓' if metrics['is_feasible'] else 'INFEASIBLE ✗'
    status_color = 'green'      if metrics['is_feasible'] else 'red'
    ax2.set_title(
        f'Power Profile  |  {status_text}  |  '
        f'Violations: {metrics["violations"]}/{T}  |  '
        f'Peak: {metrics["peak_power"]:.2f} kW',
        fontsize=13, fontweight='bold', color=status_color, pad=12)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_xlim(-0.5, T - 0.5)

    # ── Completion Pie ───────────────────────────────────────────────────────
    total_sched = np.sum(x, axis=1)
    completed   = int(np.sum(total_sched == processing_time))
    partial     = int(np.sum((total_sched > 0) & (total_sched < processing_time)))
    unscheduled = int(np.sum(total_sched == 0))
    nonzero     = [(s, l, c) for s, l, c in
                   zip([completed, partial, unscheduled],
                       ['Completed', 'Partial', 'Unscheduled'],
                       ['#2ca02c',   '#ff7f0e', '#d62728']) if s > 0]
    if nonzero:
        s_, l_, c_ = zip(*nonzero)
        ax3.pie(s_, labels=l_, autopct='%1.1f%%', colors=c_, startangle=90,
                textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax3.set_title('Task Completion Status', fontsize=12, fontweight='bold')

    # ── Metrics Table ────────────────────────────────────────────────────────
    ax4.axis('off')
    metrics_data = [
        ['Metric',      'Value'],
        ['Tasks',       f'{n}'],
        ['Machines',    f'{m}'],
        ['Horizon',     f'{T} slots'],
        ['P_max',       f'{P_max:.2f} kW'],
        ['Peak Power',  f'{metrics["peak_power"]:.2f} kW'],
        ['Avg Power',   f'{metrics["avg_power"]:.2f} kW'],
        ['Completion',  f'{metrics["completion_rate"]*100:.1f}%'],
        ['Violations',  f'{metrics["violations"]}'],
        ['Conflicts',   f'{metrics["conflicts"]}'],
        ['Utilization', f'{metrics["utilization"]:.1f}%'],
    ]
    tbl = ax4.table(cellText=metrics_data, cellLoc='left', loc='center',
                    colWidths=[0.6, 0.4], bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for col in range(2):
        tbl[(0, col)].set_facecolor('#4CAF50')
        tbl[(0, col)].set_text_props(weight='bold', color='white')
    for row in range(1, len(metrics_data)):
        bg = '#f0f0f0' if row % 2 == 0 else 'white'
        for col in range(2):
            tbl[(row, col)].set_facecolor(bg)
    ax4.set_title('Overall Metrics', fontsize=12, fontweight='bold', pad=10)

    # ── Machine Utilization Table ────────────────────────────────────────────
    ax5.axis('off')
    if machine_info:
        util_data = [['M', 'kW', 'Tasks', 'Util%']]
        for j in sorted(machine_info.keys()):
            info = machine_info[j]
            used = sum(int(np.sum(x[i, :])) for i in info['tasks'])
            util = (used / T) * 100 if T > 0 else 0.0
            util_data.append([f'M{j}', f'{info["power"]:.1f}',
                               f'{info["n_tasks"]}', f'{util:.1f}%'])
        tbl2 = ax5.table(cellText=util_data, cellLoc='center', loc='center',
                         colWidths=[0.25] * 4, bbox=[0, 0, 1, 1])
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(9)
        tbl2.scale(1, 1.6)
        for col in range(4):
            tbl2[(0, col)].set_facecolor('#FF9800')
            tbl2[(0, col)].set_text_props(weight='bold', color='white')
        for row in range(1, len(util_data)):
            r, g, b, _ = list(machine_info.values())[row - 1]['color']
            for col in range(4):
                tbl2[(row, col)].set_facecolor((r, g, b, 0.25))
        ax5.set_title('Machine Utilization', fontsize=12, fontweight='bold', pad=10)
    else:
        # ── fixed: \n not \\n ────────────────────────────────────────────────
        ax5.text(0.5, 0.5, 'No Machine\nInformation',
                 ha='center', va='center', fontsize=12,
                 transform=ax5.transAxes)

    # ── Finalize ─────────────────────────────────────────────────────────────
    plt.suptitle(
        f'PPMSP Dashboard  |  {n} tasks  ·  {m} chargers  ·  {T} time slots',
        fontsize=16, fontweight='bold', y=0.98)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved → {filename}")

    plt.show()
    plt.close(fig)   # safe after show(); prevents memory leak in batch runs

    return metrics

# ============================================================================
# GREEDY CONSTRUCTIVE HEURISTIC
# ============================================================================

@njit
def greedy_constructive_heuristic(instance: Tuple, 
                                  given_P_max: Optional[float] = None,
                                  seed: Optional[int] = None) -> Tuple:
    """
    Greedy constructive heuristic for PPMSP.
    
    Prioritizes high-power tasks and assigns them to minimum-load slots.
    
    Args:
        instance: Problem instance tuple
        given_P_max: Target peak power constraint (uses P_max from instance if None)
        seed: Random seed for tie-breaking
        
    Returns:
        x: Schedule matrix (n x T)
        peak_load: Achieved peak power
        history: Convergence history (single value)
        initial_peak: Same as peak_load (no refinement)
        status: 1 if peak_load <= given_P_max, 0 otherwise
        improvement: 0.0 (no iterative improvement)
    """
    if seed is not None:
        np.random.seed(seed)
    
    partition, start_time, end_time, processing_time, power_requirement, n, m, T, P_max = instance
    
    # Determine target constraint
    used_P_max = given_P_max if given_P_max is not None else P_max
    
    # Initialize
    x = np.zeros((n, T), dtype=np.int32)
    P_t = np.zeros(T, dtype=np.float32)
    
    # Sort tasks by power requirement (descending)
    sorted_indices = np.argsort(-power_requirement)
    
    # Schedule each task greedily
    for i in sorted_indices:
        R = processing_time[i]
        available_slots = np.arange(start_time[i] - 1, end_time[i])
        
        while R > 0 and len(available_slots) > 0:
            # Find minimum power among available slots
            min_power = np.inf
            min_count = 0
            
            # First pass: find minimum power
            for t in available_slots:
                if P_t[t] < min_power:
                    min_power = P_t[t]
            
            # Second pass: count slots with minimum power
            for t in available_slots:
                if P_t[t] == min_power:
                    min_count += 1
            
            # Allocate array for minimum slots
            min_slots = np.empty(min_count, dtype=available_slots.dtype)
            idx = 0
            for t in available_slots:
                if P_t[t] == min_power:
                    min_slots[idx] = t
                    idx += 1
            
            # Random tie-breaking
            if len(min_slots) > 0:
                t_star = min_slots[np.random.randint(len(min_slots))]
                
                # Assign
                x[i, int(t_star)] = 1
                P_t[int(t_star)] += power_requirement[i]
                
                # Remove from available
                new_available = np.empty(len(available_slots) - 1, dtype=available_slots.dtype)
                new_idx = 0
                for t in available_slots:
                    if t != t_star:
                        new_available[new_idx] = t
                        new_idx += 1
                available_slots = new_available
                
                R -= 1
            else:
                break
    
    peak_load = np.max(P_t) if len(P_t) > 0 else 0.0
    
    # Determine status based on constraint satisfaction
    status = 1 if peak_load <= used_P_max else 0
    
    # Unified return format
    history = np.array([peak_load])
    initial_peak = peak_load
    improvement = 0.0  # No iterative improvement
    
    return x, peak_load, history, initial_peak, status, improvement


# ============================================================================
# MOSER-TARDOS INSPIRED ALGORITHM
# ============================================================================
# @njit
# def moser_tardos_inspired_algorithm(instance: Tuple,
#                                     N: int = 10000,
#                                     alpha: float = 0.7,
#                                     given_P_max: Optional[float] = None,
#                                     seed: Optional[int] = None) -> Tuple:
#     """
#     Moser-Tardos inspired randomized resampling algorithm.

#     Args:
#         instance:     Problem instance tuple
#         N:            Maximum number of iterations
#         alpha:        Grid capacity ratio — used_P_max = alpha * sum(charger powers)
#                       when given_P_max is None. Ignored if given_P_max is provided.
#         given_P_max:  Override target peak power. If None, uses alpha * sum(powers).
#                       If 0.0, runs in pure optimization mode (attacks current peak).
#         seed:         Random seed for reproducibility

#     Returns:
#         x_best:       Best schedule found (n x T)
#         f_best:       Best peak load achieved
#         history:      Convergence history (best value at each iteration)
#         initial_peak: Peak load of the initial random schedule
#         status:       1 if f_best <= used_P_max, 0 otherwise
#         improvement:  Improvement (%) from initial_peak to f_best
#     """
#     if seed is not None:
#         np.random.seed(seed)

#     partition, start_time, end_time, processing_time, \
#         power_requirement, n, m, T, P_max = instance

#     # ── Determine effective P_max target ─────────────────────────────────────
#     if given_P_max is not None:
#         used_P_max = given_P_max          # explicit override (0 = pure optim)
#     else:
#         used_P_max = alpha * float(np.sum(power_requirement))  # alpha * sum(w_j)

#     # ── Initialize random schedule ────────────────────────────────────────────
#     x = generate_schedule(instance)
#     history = []

#     _, initial_peak = peak_load_computation(x, instance)
#     f_best = initial_peak
#     x_best = x.copy()
#     status = 0

#     # ── Main resampling loop ──────────────────────────────────────────────────
#     for k in range(N):
#         power_per_slot, peak_load = peak_load_computation(x, instance)

#         if peak_load < f_best:
#             f_best = peak_load
#             x_best = x.copy()

#         history.append(f_best)

#         # Adaptive threshold
#         P_threshold = peak_load if used_P_max == 0.0 else used_P_max

#         # Find violated slots
#         exceeded_indices = np.where(power_per_slot > P_threshold)[0]

#         # Early termination
#         if len(exceeded_indices) == 0:
#             status = 1
#             break

#         # Pick a random violated slot
#         t_idx = np.random.choice(exceeded_indices)

#         # Resample all tasks active at t*
#         tasks_at_t = np.where(x[:, t_idx] == 1)[0]
#         for i in tasks_at_t:
#             x[i, :] = 0
#             available = np.arange(start_time[i] - 1, end_time[i])
#             new_slots = np.random.choice(available,
#                                          size=processing_time[i],
#                                          replace=False)
#             x[i, new_slots] = 1

#     # ── Final evaluation ──────────────────────────────────────────────────────
#     _, peak_load = peak_load_computation(x, instance)
#     if peak_load < f_best:
#         f_best = peak_load
#         x_best = x.copy()
#     history.append(f_best)

#     if used_P_max == 0.0:
#         status = 1
#     else:
#         status = 1 if f_best <= used_P_max else 0

#     improvement = ((initial_peak - f_best) / initial_peak * 100.0
#                    if initial_peak > 0 else 0.0)

#     return x_best, f_best, np.array(history), initial_peak, status, improvement


# @njit
# def moser_tardos_inspired_algorithm(instance, N=10000, given_P_max=0.0, seed=None):
#     if seed is not None: np.random.seed(seed)
#     partition, start_time, end_time, processing_time, power_requirement, n, m, T, P_max = instance
#     used_P_max = given_P_max 
    
#     x = generate_schedule(instance)
#     _, initial_peak = peak_load_computation(x, instance)
#     f_best = initial_peak
#     x_best = x.copy()

#     for k in range(N):
#         power_per_slot, peak_load = peak_load_computation(x, instance)
#         if peak_load < f_best:
#             f_best = peak_load
#             x_best = x.copy()
            
#         P_threshold = peak_load if used_P_max == 0.0 else used_P_max
        
#         # 🚨 THE BUG FIX: Use >= instead of > so it catches the actual peak!
#         exceeded_indices = np.where(power_per_slot >= P_threshold)[0]
        
#         if len(exceeded_indices) == 0: 
#             break
        
#         # Pick a random violated slot
#         t_idx = np.random.choice(exceeded_indices)
#         tasks_at_t = np.where(x[:, t_idx] == 1)[0]
        
#         if len(tasks_at_t) == 0:
#             continue
            
#         # IMPROVEMENT 1: THE SCALPEL (Resample only one task, not all of them)
#         i = np.random.choice(tasks_at_t)
        
#         # Remove it from the schedule
#         x[i, :] = 0
        
#         # Fast recalculation of the current power profile WITHOUT task i
#         current_P_t = np.zeros(T, dtype=np.float64)
#         for t_slot in range(T):
#             for j in range(n):
#                 if x[j, t_slot] == 1:
#                     current_P_t[t_slot] += power_requirement[j]
        
#         # IMPROVEMENT 2: VALLEY FILLING (Place it in the lowest load slots)
#         available = np.arange(start_time[i] - 1, end_time[i])
#         loads_on_available = np.zeros(len(available), dtype=np.float64)
        
#         for idx, t_val in enumerate(available):
#             loads_on_available[idx] = current_P_t[t_val]
            
#         sorted_indices = np.argsort(loads_on_available)
        
#         p_i = processing_time[i]
#         best_slots = available[sorted_indices[:p_i]]
        
#         x[i, best_slots] = 1

#     _, peak_load = peak_load_computation(x, instance)
#     if peak_load < f_best:
#         f_best = peak_load
#         x_best = x.copy()
        
#     return x_best, f_best

@njit
def moser_tardos_inspired_algorithm(instance, N=10000, given_P_max=0.0, seed=None):
    if seed is not None: np.random.seed(seed)
    partition, start_time, end_time, processing_time, power_requirement, n, m, T, P_max = instance
    used_P_max = given_P_max 
    
    x = generate_schedule(instance)
    history = []  # Initialize history tracking
    
    _, initial_peak = peak_load_computation(x, instance)
    f_best = initial_peak
    x_best = x.copy()
    status = 0

    for k in range(N):
        power_per_slot, peak_load = peak_load_computation(x, instance)
        
        if peak_load < f_best:
            f_best = peak_load
            x_best = x.copy()
            
        history.append(f_best)  # Track best objective at each iteration
            
        P_threshold = peak_load if used_P_max == 0.0 else used_P_max
        
        # 🚨 THE BUG FIX: Use >= instead of > so it catches the actual peak!
        exceeded_indices = np.where(power_per_slot >= P_threshold)[0]
        
        if len(exceeded_indices) == 0: 
            break
        
        # Pick a random violated slot
        t_idx = np.random.choice(exceeded_indices)
        tasks_at_t = np.where(x[:, t_idx] == 1)[0]
        
        if len(tasks_at_t) == 0:
            continue
            
        # IMPROVEMENT 1: THE SCALPEL (Resample only one task, not all of them)
        i = np.random.choice(tasks_at_t)
        
        # Remove it from the schedule
        x[i, :] = 0
        
        # Fast recalculation of the current power profile WITHOUT task i
        current_P_t = np.zeros(T, dtype=np.float64)
        for t_slot in range(T):
            for j in range(n):
                if x[j, t_slot] == 1:
                    current_P_t[t_slot] += power_requirement[j]
        
        # IMPROVEMENT 2: VALLEY FILLING (Place it in the lowest load slots)
        available = np.arange(start_time[i] - 1, end_time[i])
        loads_on_available = np.zeros(len(available), dtype=np.float64)
        
        for idx, t_val in enumerate(available):
            loads_on_available[idx] = current_P_t[t_val]
            
        sorted_indices = np.argsort(loads_on_available)
        
        p_i = processing_time[i]
        best_slots = available[sorted_indices[:p_i]]
        
        x[i, best_slots] = 1

    # Final evaluation check
    _, peak_load = peak_load_computation(x, instance)
    if peak_load < f_best:
        f_best = peak_load
        x_best = x.copy()
        
    history.append(f_best)  # Append final best
    
    # Calculate constraint satisfaction status
    if used_P_max == 0.0:
        status = 1
    else:
        status = 1 if f_best <= used_P_max else 0
        
    # Calculate percentage improvement
    improvement = ((initial_peak - f_best) / initial_peak * 100.0) if initial_peak > 0 else 0.0

    return x_best, f_best, np.array(history), initial_peak, status, improvement


# ============================================================================
# LP-BASED RANDOMIZED ROUNDING ALGORITHM
# ============================================================================

# def solve_lp_relaxation(instance: Tuple) -> Tuple[np.ndarray, float]:
#     """
#     Solve LP relaxation of PPMSP.
    
#     Replace binary constraints x_{i,t} ∈ {0,1} with 0 ≤ x_{i,t} ≤ 1
#     while maintaining all other constraints.
    
#     Args:
#         instance: Problem instance
        
#     Returns:
#         x_tilde: Fractional solution [n x T] with x_{i,t} ∈ [0,1]
#         P_LP: Optimal LP objective value (lower bound)
#     """
#     from scipy.optimize import linprog
    
#     partition, start_time, end_time, processing_time, power_requirement, n, m, T, P_max = instance
    
#     # Decision variables: [x_{1,1}, ..., x_{1,T}, x_{2,1}, ..., x_{n,T}, P]
#     # Total: n*T + 1 variables
#     num_vars = n * T + 1
    
#     # Objective: minimize P (last variable)
#     c = np.zeros(num_vars)
#     c[-1] = 1.0  # Minimize P
    
#     # Constraints
#     A_eq = []
#     b_eq = []
#     A_ub = []
#     b_ub = []
    
#     # Charging duration constraints (equality): sum_{t in T_i} x_{i,t} = p_i
#     for i in range(n):
#         row = np.zeros(num_vars)
#         for t in range(start_time[i] - 1, end_time[i]):
#             row[i * T + t] = 1.0
#         A_eq.append(row)
#         b_eq.append(processing_time[i])
    
#     # Peak power constraints (inequality): sum_i w_i * x_{i,t} <= P for all t
#     for t in range(T):
#         row = np.zeros(num_vars)
#         for i in range(n):
#             if start_time[i] - 1 <= t < end_time[i]:
#                 row[i * T + t] = power_requirement[i]
#         row[-1] = -1.0  # -P
#         A_ub.append(row)
#         b_ub.append(0.0)
    
#     # Time window constraints: x_{i,t} = 0 for t not in [s_i, e_i]
#     for i in range(n):
#         for t in range(T):
#             if t < start_time[i] - 1 or t >= end_time[i]:
#                 row = np.zeros(num_vars)
#                 row[i * T + t] = 1.0
#                 A_eq.append(row)
#                 b_eq.append(0.0)
    
#     # Bounds: 0 <= x_{i,t} <= 1 for all i,t; P >= 0
#     bounds = [(0, 1) for _ in range(n * T)] + [(0, None)]
    
#     # Solve LP
#     A_eq = np.array(A_eq) if A_eq else None
#     b_eq = np.array(b_eq) if b_eq else None
#     A_ub = np.array(A_ub) if A_ub else None
#     b_ub = np.array(b_ub) if b_ub else None
    
#     result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
#                     bounds=bounds, method='highs')
    
#     if not result.success:
#         # Fallback: uniform fractional solution
#         x_tilde = np.zeros((n, T))
#         for i in range(n):
#             available = end_time[i] - (start_time[i] - 1)
#             if available > 0:
#                 prob = processing_time[i] / available
#                 for t in range(start_time[i] - 1, end_time[i]):
#                     x_tilde[i, t] = min(prob, 1.0)
#         P_LP = np.max([np.sum(power_requirement * x_tilde[:, t]) for t in range(T)])
#         return x_tilde, P_LP
    
#     # Extract solution
#     x_flat = result.x[:n * T]
#     x_tilde = x_flat.reshape((n, T))
#     P_LP = result.x[-1]
    
#     return x_tilde, P_LP


def solve_lp_relaxation(instance: Tuple) -> Tuple[np.ndarray, float]:
    """
    Solve LP relaxation of PPMSP.

    Replace binary constraints x_{i,t} ∈ {0,1} with 0 ≤ x_{i,t} ≤ 1
    while maintaining all other constraints.

    Args:
        instance: Problem instance

    Returns:
        x_tilde: Fractional solution [n x T] with x_{i,t} ∈ [0,1]
        P_LP:    Optimal LP objective value (lower bound)
    """
    partition, start_time, end_time, processing_time, power_requirement, n, m, T, P_max = instance

    prob = LpProblem("PPMSP_LP_Relaxation", LpMinimize)

    # ── Decision variables ───────────────────────────────────────────────────
    x = [[LpVariable(f"x_{i}_{t}", lowBound=0.0, upBound=1.0)
          for t in range(T)]
         for i in range(n)]

    P = LpVariable("P", lowBound=0.0)

    # ── Objective ────────────────────────────────────────────────────────────
    prob += P

    # ── Charging duration constraints ────────────────────────────────────────
    for i in range(n):
        prob += (
            lpSum(x[i][t] for t in range(start_time[i] - 1, end_time[i]))
            == processing_time[i],
            f"duration_{i}"
        )

    # ── Time window: x_{i,t} = 0 outside [s_i, e_i) ────────────────────────
    for i in range(n):
        for t in range(T):
            if t < start_time[i] - 1 or t >= end_time[i]:
                prob += (x[i][t] == 0.0, f"window_{i}_{t}")

    # ── Peak power constraints ───────────────────────────────────────────────
    for t in range(T):
        active = [i for i in range(n)
                  if start_time[i] - 1 <= t < end_time[i]]
        if active:
            prob += (
                lpSum(power_requirement[i] * x[i][t] for i in active) <= P,
                f"peak_{t}"
            )

    # ── Solve with Gurobi ────────────────────────────────────────────────────
    solver = GUROBI(
        msg=False,
        Method=2,        # barrier method — fastest for LP relaxations
        MIPGap=1e-6,       # tight optimality gap
        Threads=os.cpu_count()          # adjust to your machine
    )
    prob.solve(solver)

    # ── Extract solution ─────────────────────────────────────────────────────
    if LpStatus[prob.status] != 'Optimal':
        x_tilde = np.zeros((n, T))
        for i in range(n):
            available = end_time[i] - (start_time[i] - 1)
            if available > 0:
                frac = processing_time[i] / available
                for t in range(start_time[i] - 1, end_time[i]):
                    x_tilde[i, t] = min(frac, 1.0)
        P_LP = float(np.max(
            [np.sum(power_requirement * x_tilde[:, t]) for t in range(T)]
        ))
        return x_tilde, P_LP

    x_tilde = np.array([[value(x[i][t]) or 0.0
                         for t in range(T)]
                        for i in range(n)])
    P_LP = value(P)

    return x_tilde, P_LP



@njit
def weighted_sample_slots(slots: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    """
    Weighted sampling without replacement using accept-reject.
    
    Args:
        slots: Available slots
        weights: Weight for each slot (higher = more likely)
        k: Number of slots to sample
        
    Returns:
        selected: Array of k selected slots
    """
    if len(slots) == 0 or k == 0:
        return np.empty(0, dtype=slots.dtype)
    
    k = min(k, len(slots))
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Sample without replacement
    selected = np.empty(k, dtype=slots.dtype)
    available = slots.copy()
    available_weights = weights.copy()
    
    for idx in range(k):
        # Normalize remaining weights
        available_weights = available_weights / np.sum(available_weights)
        
        # Cumulative distribution
        cumsum = np.cumsum(available_weights)
        
        # Sample
        r = np.random.rand()
        for i in range(len(available)):
            if r <= cumsum[i]:
                selected[idx] = available[i]
                # Remove selected element
                available = np.concatenate((available[:i], available[i+1:]))
                available_weights = np.concatenate((available_weights[:i], available_weights[i+1:]))
                break
    
    return selected


def lp_based_rounding_algorithm(instance: Tuple,
                                num_trials: int = 50,
                                given_P_max: Optional[float] = None,
                                seed: Optional[int] = None) -> Tuple:
    """
    Randomized LP-based rounding algorithm.
    
    Solves LP relaxation, performs probabilistic rounding, and repairs
    violations using peak-aware weighted sampling.
    
    Args:
        instance: Problem instance
        num_trials: Number of independent rounding trials
        given_P_max: Target peak power constraint
        seed: Random seed
        
    Returns:
        x_best: Best feasible schedule found
        f_best: Best peak load achieved
        history: Convergence history (best per trial)
        initial_peak: LP lower bound P^LP
        status: 1 if f_best <= given_P_max, 0 otherwise
        improvement: Improvement over LP bound (%)
    """
    if seed is not None:
        np.random.seed(seed)
    
    partition, start_time, end_time, processing_time, power_requirement, n, m, T, P_max = instance
    
    used_P_max = given_P_max if given_P_max is not None else P_max
    
    # Line 3: Solve LP relaxation
    x_tilde, P_LP = solve_lp_relaxation(instance)
    initial_peak = P_LP  # LP lower bound
    
    x_best = None
    f_best = np.inf
    history = []
    
    # Multi-start: perform multiple rounding trials
    for trial in range(num_trials):
        # Line 4: Randomized rounding
        x = np.zeros((n, T), dtype=np.int32)
        for i in range(n):
            for t in range(start_time[i] - 1, end_time[i]):
                if np.random.rand() < x_tilde[i, t]:
                    x[i, t] = 1
        
        # Lines 5-15: Repair phase
        for i in range(n):
            # Line 6: Count assigned slots
            r_i = np.sum(x[i, start_time[i]-1:end_time[i]])
            
            # Line 7: Compute current peak loads
            P_t = np.array([np.sum(power_requirement * x[:, t]) for t in range(T)])
            
            if r_i < processing_time[i]:
                # Lines 8-10: Underassignment - add slots with low peak
                deficit = processing_time[i] - r_i
                
                # Find unassigned slots in time window
                available_range = np.arange(start_time[i] - 1, end_time[i])
                unassigned = available_range[x[i, available_range] == 0]
                
                if len(unassigned) > 0:
                    # Weight ∝ 1/(P_t + w_i) - prefer low-load slots
                    weights = 1.0 / (P_t[unassigned] + power_requirement[i] + 1e-6)
                    
                    # Sample deficit slots
                    slots_to_add = weighted_sample_slots(unassigned, weights, deficit)
                    x[i, slots_to_add] = 1
                    
            elif r_i > processing_time[i]:
                # Lines 11-13: Overassignment - remove slots with high peak
                surplus = r_i - processing_time[i]
                
                # Find assigned slots in time window
                available_range = np.arange(start_time[i] - 1, end_time[i])
                assigned = available_range[x[i, available_range] == 1]
                
                if len(assigned) > 0:
                    # Weight ∝ P_t - prefer high-load slots for removal
                    weights = P_t[assigned] + 1e-6
                    
                    # Sample surplus slots to remove
                    slots_to_remove = weighted_sample_slots(assigned, weights, surplus)
                    x[i, slots_to_remove] = 0
        
        # Line 16: Compute final peak load
        P_t_final = np.array([np.sum(power_requirement * x[:, t]) for t in range(T)])
        peak_load = np.max(P_t_final) if len(P_t_final) > 0 else 0.0
        
        # Track best solution
        if peak_load < f_best:
            f_best = peak_load
            x_best = x.copy()
        
        history.append(f_best)
    
    # Determine status
    status = 1 if f_best <= used_P_max else 0
    
    # Calculate improvement over LP bound
    improvement = (f_best - P_LP) / P_LP * 100.0 if P_LP > 0 else 0.0
    
    # Line 17: Return best solution
    return x_best, f_best, np.array(history), initial_peak, status, improvement


@njit
def perturb_partial_slot_reassignment(x: np.ndarray, 
                                     instance: Tuple, 
                                     alpha_p: float) -> np.ndarray:
    """
    Perturbation operator: partial slot reassignment.
    
    Removes k = ⌈α_p · p_i*⌉ slots and adds k new slots for randomly selected task i*.
    
    Args:
        x: Current schedule (n x T)
        instance: Problem instance
        alpha_p: Perturbation rate ∈ (0, 1]
        
    Returns:
        x_prime: Perturbed schedule
    """
    partition, start_time, end_time, processing_time, power_requirement, n, m, T, P_max = instance
    
    # Line 3: Random vehicle selection
    i_star = np.random.randint(n)
    
    # Line 4: Copy current schedule
    x_prime = x.copy()
    
    # Line 5: Number of slots to perturb
    k = max(1, int(np.ceil(alpha_p * processing_time[i_star])))
    
    # Line 6: Find currently assigned slots
    assigned_slots = np.where(x_prime[i_star, :] == 1)[0]
    
    # Line 7: Randomly select k slots to remove
    if len(assigned_slots) < k:
        k = len(assigned_slots)  # Can't remove more than assigned
    
    slots_to_remove = np.random.choice(assigned_slots, size=k, replace=False)
    
    # Line 8: Remove selected slots
    x_prime[i_star, slots_to_remove] = 0
    
    # Line 9: Find available slots (unassigned within time window)
    available_range = np.arange(start_time[i_star] - 1, end_time[i_star])
    available_slots = available_range[x_prime[i_star, available_range] == 0]
    
    # Line 10: Randomly select k slots to add
    if len(available_slots) < k:
        k = len(available_slots)  # Can't add more than available
    
    slots_to_add = np.random.choice(available_slots, size=k, replace=False)
    
    # Line 11: Add selected slots
    x_prime[i_star, slots_to_add] = 1
    
    return x_prime


# ============================================================================
# SIMULATED ANNEALING
# ============================================================================

@njit
def simulated_annealing(instance: Tuple, 
                       T_0: float = 100.0, 
                       T_min: float = 0.1, 
                       alpha_c: float = 0.95, 
                       L: int = 100, 
                       N: int = 10000,
                       alpha_p: float = 0.3,
                       given_P_max: Optional[float] = None,
                       seed: Optional[int] = None) -> Tuple:
    """
    Simulated Annealing with random initialization.
    
    Args:
        instance: Problem instance
        T_0: Initial temperature (θ_0 in algorithm)
        T_min: Terminal temperature (θ_min)
        alpha_c: Cooling rate (α_c)
        L: Epoch length
        N: Maximum iterations
        alpha_p: Perturbation rate for partial slot reassignment
        given_P_max: Target peak power constraint
        seed: Random seed
        
    Returns:
        x_best: Best schedule found
        f_best: Best peak load
        history: Convergence history
        initial_peak: Initial peak load
        status: 1 if f_best <= given_P_max, 0 otherwise
        improvement: Improvement percentage
    """
    if seed is not None:
        np.random.seed(seed)
    
    partition, start_time, end_time, processing_time, power_requirement, n, m, T, P_max = instance
    
    used_P_max = given_P_max if given_P_max is not None else P_max
    
    # Line 3: Random initialization
    x_current = generate_schedule(instance)
    
    # Line 4: Initial evaluation and initialize {P_t}
    P_t, f_current = peak_load_computation(x_current, instance)
    initial_peak = f_current
    
    # Line 5: Initialize best solution, temperature, iteration counter
    x_best = x_current.copy()
    f_best = f_current
    theta = T_0
    k = 0
    
    history = [f_current]
    
    # Line 6: Main loop
    while theta > T_min and k < N:
        # Line 7: Epoch loop
        for _ in range(L):
            # Line 8: Generate neighbor via partial perturbation
            x_neighbor = perturb_partial_slot_reassignment(x_current, instance, alpha_p)
            
            # Line 9: Incremental evaluation (recompute for simplicity, or implement incremental)
            # Note: True incremental would update only affected slots
            _, f_neighbor = peak_load_computation(x_neighbor, instance)
            delta_f = f_neighbor - f_current
            
            # Line 10: Random uniform sample
            u = np.random.rand()
            
            # Lines 11-12: Acceptance criterion
            if delta_f < 0 or u < np.exp(-delta_f / theta):
                x_current = x_neighbor.copy()
                f_current = f_neighbor
                # Line 12: Update {P_t} (handled implicitly next iteration)
            
            # Lines 14-15: Update best
            if f_current < f_best:
                x_best = x_current.copy()
                f_best = f_current
            
            history.append(f_best)
            
            # Line 17: Increment counter
            k += 1
            
            # Lines 18-19: Check iteration limit
            if k >= N:
                break
        
        # Line 21: Cool down
        theta = alpha_c * theta
    
    # Determine status
    status = 1 if f_best <= used_P_max else 0
    
    # Calculate improvement
    improvement = (initial_peak - f_best) / initial_peak * 100.0 if initial_peak > 0 else 0.0
    
    # Line 23: Return best solution
    return x_best, f_best, np.array(history), initial_peak, status, improvement


# ============================================================================
# HYBRID SIMULATED ANNEALING VARIANTS
# ============================================================================

@njit
def hybrid_sa_greedy(instance: Tuple, 
                     T_0: float = 100.0, 
                     T_min: float = 0.1, 
                     alpha_c: float = 0.95, 
                     L: int = 100, 
                     N: int = 10000,
                     alpha_p: float = 0.3,
                     given_P_max: Optional[float] = None,
                     seed: Optional[int] = None) -> Tuple:
    """
    Hybrid Simulated Annealing with Greedy initialization.
    
    Initializes from greedy heuristic, then refines with SA.
    
    Args:
        instance: Problem instance
        T_0: Initial temperature (θ_0 in algorithm)
        T_min: Terminal temperature (θ_min)
        alpha_c: Cooling rate (α_c)
        L: Epoch length
        N: Maximum iterations
        alpha_p: Perturbation rate for partial slot reassignment
        given_P_max: Target peak power constraint
        seed: Random seed
        
    Returns:
        x_best: Best schedule found
        f_best: Best peak load
        history: Convergence history
        initial_peak: Initial peak load from greedy
        status: 1 if f_best <= given_P_max, 0 otherwise
        improvement: Improvement percentage
    """
    if seed is not None:
        np.random.seed(seed)
    
    partition, start_time, end_time, processing_time, power_requirement, n, m, T, P_max = instance
    
    used_P_max = given_P_max if given_P_max is not None else P_max
    
    # Line 3: GREEDY initialization (instead of random)
    x_current, initial_peak, _, _, _, _ = greedy_constructive_heuristic(instance, given_P_max, seed)
    
    # Line 4: Initial evaluation and initialize {P_t}
    P_t, f_current = peak_load_computation(x_current, instance)
    
    # Line 5: Initialize best solution, temperature, iteration counter
    x_best = x_current.copy()
    f_best = f_current
    theta = T_0
    k = 0
    
    history = [f_current]
    
    # Lines 6-22: Identical SA loop
    while theta > T_min and k < N:
        for _ in range(L):
            x_neighbor = perturb_partial_slot_reassignment(x_current, instance, alpha_p)
            _, f_neighbor = peak_load_computation(x_neighbor, instance)
            delta_f = f_neighbor - f_current
            u = np.random.rand()
            
            if delta_f < 0 or u < np.exp(-delta_f / theta):
                x_current = x_neighbor.copy()
                f_current = f_neighbor
            
            if f_current < f_best:
                x_best = x_current.copy()
                f_best = f_current
            
            history.append(f_best)
            k += 1
            
            if k >= N:
                break
        
        theta = alpha_c * theta
    
    status = 1 if f_best <= used_P_max else 0
    improvement = (initial_peak - f_best) / initial_peak * 100.0 if initial_peak > 0 else 0.0
    
    return x_best, f_best, np.array(history), initial_peak, status, improvement


@njit
def hybrid_sa_resampling(instance: Tuple, 
                         N_init: int = 10000,
                         T_0: float = 100.0, 
                         T_min: float = 0.1, 
                         alpha_c: float = 0.95, 
                         L: int = 100, 
                         N: int = 10000,
                         alpha_p: float = 0.3,
                         given_P_max: Optional[float] = 0.0,
                         alpha: float = 0.7,
                         seed: Optional[int] = None) -> Tuple:
    """
    Hybrid Simulated Annealing with Resampling initialization.
    
    Initializes from randomized resampling algorithm, then refines with SA.
    
    Args:
        instance: Problem instance
        N_init: Max iterations for resampling initialization
        T_0: Initial temperature (θ_0 in algorithm)
        T_min: Terminal temperature (θ_min)
        alpha_c: Cooling rate (α_c)
        L: Epoch length
        N: Maximum iterations for SA refinement
        alpha_p: Perturbation rate for partial slot reassignment
        given_P_max: Target peak power constraint
        seed: Random seed
        
    Returns:
        x_best: Best schedule found
        f_best: Best peak load
        history: Convergence history
        initial_peak: Initial peak load from resampling
        status: 1 if f_best <= given_P_max, 0 otherwise
        improvement: Improvement percentage
    """
    if seed is not None:
        np.random.seed(seed)
    
    partition, start_time, end_time, processing_time, power_requirement, n, m, T, P_max = instance
    
    used_P_max = given_P_max if given_P_max is not None else 0.0
    
    # Line 3: RESAMPLING initialization (instead of random)
    x_current, initial_peak, _, _, _, _ = moser_tardos_inspired_algorithm(
        instance, N=N_init, given_P_max=given_P_max, seed=seed
    )
    
    
    # Line 4: Initial evaluation and initialize {P_t}
    P_t, f_current = peak_load_computation(x_current, instance)
    
    # Line 5: Initialize best solution, temperature, iteration counter
    x_best = x_current.copy()
    f_best = f_current
    theta = T_0
    k = 0
    
    history = [f_current]
    
    # Lines 6-22: Identical SA loop
    while theta > T_min and k < N:
        for _ in range(L):
            x_neighbor = perturb_partial_slot_reassignment(x_current, instance, alpha_p)
            _, f_neighbor = peak_load_computation(x_neighbor, instance)
            delta_f = f_neighbor - f_current
            u = np.random.rand()
            
            if delta_f < 0 or u < np.exp(-delta_f / theta):
                x_current = x_neighbor.copy()
                f_current = f_neighbor
            
            if f_current < f_best:
                x_best = x_current.copy()
                f_best = f_current
            
            history.append(f_best)
            k += 1
            
            if k >= N:
                break
        
        theta = alpha_c * theta
    
    status = 1 if f_best <= used_P_max else 0
    improvement = (initial_peak - f_best) / initial_peak * 100.0 if initial_peak > 0 else 0.0
    
    return x_best, f_best, np.array(history), initial_peak, status, improvement


def hybrid_sa_lp(instance: Tuple, 
                 num_trials_lp: int = 50,
                 T_0: float = 100.0, 
                 T_min: float = 0.1, 
                 alpha_c: float = 0.95, 
                 L: int = 100, 
                 N: int = 10000,
                 alpha_p: float = 0.3,
                 given_P_max: Optional[float] = None,
                 seed: Optional[int] = None) -> Tuple:
    """
    Hybrid Simulated Annealing with LP-based rounding initialization.
    
    Initializes from LP-based randomized rounding, then refines with SA.
    
    Note: NOT decorated with @njit because LP solving uses scipy (not Numba-compatible).
    
    Args:
        instance: Problem instance
        num_trials_lp: Number of LP rounding trials for initialization
        T_0: Initial temperature (θ_0 in algorithm)
        T_min: Terminal temperature (θ_min)
        alpha_c: Cooling rate (α_c)
        L: Epoch length
        N: Maximum iterations for SA refinement
        alpha_p: Perturbation rate for partial slot reassignment
        given_P_max: Target peak power constraint
        seed: Random seed
        
    Returns:
        x_best: Best schedule found
        f_best: Best peak load
        history: Convergence history
        initial_peak: Initial peak load from LP rounding
        status: 1 if f_best <= given_P_max, 0 otherwise
        improvement: Improvement percentage
    """
    if seed is not None:
        np.random.seed(seed)
    
    partition, start_time, end_time, processing_time, power_requirement, n, m, T, P_max = instance
    
    used_P_max = given_P_max if given_P_max is not None else P_max
    
    # Line 3: LP-BASED ROUNDING initialization (instead of random)
    x_current, initial_peak, _, _, _, _ = lp_based_rounding_algorithm(
        instance, num_trials=num_trials_lp, given_P_max=given_P_max, seed=seed
    )
    
    # Line 4: Initial evaluation and initialize {P_t}
    P_t, f_current = peak_load_computation(x_current, instance)
    
    # Line 5: Initialize best solution, temperature, iteration counter
    x_best = x_current.copy()
    f_best = f_current
    theta = T_0
    k = 0
    
    history = [f_current]
    
    # Lines 6-22: Identical SA loop
    while theta > T_min and k < N:
        for _ in range(L):
            x_neighbor = perturb_partial_slot_reassignment(x_current, instance, alpha_p)
            _, f_neighbor = peak_load_computation(x_neighbor, instance)
            delta_f = f_neighbor - f_current
            u = np.random.rand()
            
            if delta_f < 0 or u < np.exp(-delta_f / theta):
                x_current = x_neighbor.copy()
                f_current = f_neighbor
            
            if f_current < f_best:
                x_best = x_current.copy()
                f_best = f_current
            
            history.append(f_best)
            k += 1
            
            if k >= N:
                break
        
        theta = alpha_c * theta
    
    status = 1 if f_best <= used_P_max else 0
    improvement = (initial_peak - f_best) / initial_peak * 100.0 if initial_peak > 0 else 0.0
    
    return x_best, f_best, np.array(history), initial_peak, status, improvement
