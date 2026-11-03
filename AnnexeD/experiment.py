"""
experiment.py -- the exact method against Gurobi on the time-indexed formulations.

The unit-flow formulation (UF) is compared, through the computing time, with Gurobi applied
to the formulation (P) of the paper and to its covering form (C). The three are solved by
the same solver under the same conditions: time limit, threads, gap tolerance, machine.

    python experiment.py [--name appendix_d_exact_methods] [--limit 1800] [--ids 1-40] [--methods UF,C,P]
                         [--extra 300,400,500] [--extra-methods UF] [--threads all] [--report]

    --name           results folder: results/NAME (one JSON per method and instance: resumable)
    --limit          time limit per run (s), the same for every formulation
    --ids            instances of the study, e.g. 1-10,21-30 (1 to 20: ACN, 21 to 40: simulated)
    --methods        UF (unit-flow formulation), P (formulation (P), per-type form, milp.py),
                     C (covering form of (P), covering.py),
                     UF0 (unit-flow formulation with the plain rounding constraints: ablation)
    --extra          larger instances: one ACN and one simulated instance for each size given
    --extra-methods  formulations run on the larger instances
    --threads        Gurobi threads per run: "all" (logical cores) or a number
    --report         tables, tests and figures only, from the cached runs

Every schedule is checked by the independent verifier of the Appendix D problem, and the formulations
are cross-checked: when two of them prove optimality, their values coincide, and no incumbent
exceeds the bound of another formulation.

Outputs in results/NAME (copied as such next to the paper):
    runs/*.json                          one file per formulation and instance
    Tables/appendix_d_instances.tex              characteristics of the instances
    Tables/appendix_d_exact_methods.tex          per instance
    Tables/appendix_d_exact_summary.tex          per class of instances
    Tables/appendix_d_exact_tests.tex            statistical tests on the computing times (analysis.py)
    Tables/appendix_d_exact_large.tex            larger instances
    Statistical Tests/appendix_d_exact_tests.json
    Figures/PNG, Figures/SVG             see analysis.py
    Data/appendix_d_exact_methods.csv, Data/appendix_d_instances.csv
    report.txt                           summary, tests, cross-checks, machine
"""
import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import problems.appendix_d.model as J                           # noqa: E402
from problems.appendix_d import unit_flow as UF             # noqa: E402
import analysis as A                                 # noqa: E402

CODE = "appendix_d"
MIP_GAP = 1e-6            # relative gap tolerance of every formulation (integral objective: exact)
EXTRA_SEED = 9000         # seeds of the larger instances: EXTRA_SEED + n (ACN), + n + 1 (simulated)
LABEL = {"UF": r"$(\mathrm{UF})$", "UF0": r"$(\mathrm{UF})$, plain rounding", "C": r"Gurobi on $(\mathrm{C})$",
         "P": r"Gurobi on $(\mathrm{P})$"}
SHORT = {"UF": r"$(\mathrm{UF})$", "UF0": r"$(\mathrm{UF}_0)$", "C": r"$(\mathrm{C})$", "P": r"$(\mathrm{P})$"}


def machine() -> dict:
    out = {"platform": platform.platform(), "processor": platform.processor(), "logical_cores": os.cpu_count()}
    try:
        import gurobipy as gp
        out["gurobi"] = ".".join(map(str, gp.gurobi.version()))
    except ImportError:
        pass
    return out


def parse_ids(text):
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def instances(ids, extra):
    """(instances of the study [(key, id, instance)], larger instances [(key, None, instance)])"""
    study = J.generate_instances() if ids else []
    main = [(f"inst{i:02d}", i, study[i - 1]) for i in ids]
    larger = []
    for n in extra:
        larger.append((f"ACN-n{n}", None, J.acn_instance(n, EXTRA_SEED + n)))
        larger.append((f"SIM-n{n}", None, J.simulated_instance(n, EXTRA_SEED + n + 1)))
    return main, larger


def run_one(method, inst, limit, threads) -> dict:
    """One formulation on one instance. "time" covers the construction of the model, the
    search and the construction of the schedule; "solver_time" the search of Gurobi alone."""
    t0 = time.perf_counter()
    if method in ("UF", "UF0"):
        r = UF.solve(inst, limit, threads=threads, mip_gap=MIP_GAP, tight=(method == "UF"))
    elif method == "C":
        import covering
        r = covering.solve(inst, limit, threads=threads, mip_gap=MIP_GAP)
    else:
        import milp
        r = milp.solve(inst, limit, threads=threads, model="per_type", mip_gap=MIP_GAP)
        r["solver_time"] = r["time"]
        r["n_int"] = r["n_bin"]
    wall = time.perf_counter() - t0
    sol = r.pop("solution")
    out = {"method": method, "status": r["status"], "accepted": r["accepted"], "bound": r["bound"],
           "bound_floor": r.get("bound_floor"), "time": wall,
           "solver_time": r.get("solver_time"), "nodes": r.get("nodes"),
           "n_vars": r["n_vars"], "n_bin": r["n_bin"], "n_int": r.get("n_int"), "n_constrs": r["n_constrs"],
           "n_special": r.get("n_special"), "specials_used": r.get("specials_used"),
           "bonus_units": r.get("bonus_units"), "trajectory": r.get("trajectory"),
           "limit": limit, "threads": threads, "mip_gap": MIP_GAP, "machine": machine(), "verified": None}
    if sol is not None:
        try:
            J.check_solution_feasibility(inst, sol, f"{method} on {inst.name}")
            assert abs(float(inst.v[sol[:, 0] > 0].sum()) - r["accepted"]) < 1e-6, "value differs from schedule"
            out["verified"] = True
        except AssertionError as exc:
            out["verified"], out["verification_error"] = False, str(exc)
    return out


def run(out, plan, limit, threads, log=print):
    """plan: [(key, id, instance, methods)]. One run at a time: the times are measured."""
    rdir = Path(out) / "runs"
    rdir.mkdir(parents=True, exist_ok=True)
    failed = []
    for key, _, inst, methods in plan:
        for m in methods:
            f = rdir / f"{m}_{key}.json"
            if f.exists():
                continue
            log(f"[{time.strftime('%H:%M:%S')}] {m:2s} on {key} ({inst.name}, n={inst.n}, m={inst.m}, H={inst.H}) ...")
            try:
                r = run_one(m, inst, limit, threads)
            except Exception as exc:                          # the other runs go on
                log(f"   FAILED: {type(exc).__name__}: {exc}")
                failed.append(f"{m} on {key}: {type(exc).__name__}: {exc}")
                continue
            r.update({"key": key, "name": inst.name, "source": inst.source, "n": inst.n, "m": inst.m, "H": inst.H})
            tmp = str(f) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(r, fh)
            os.replace(tmp, f)
            log(f"   {r['status']}, accepted {r['accepted']}, bound {r['bound_floor']}, {r['time']:.1f} s, "
                f"{r['n_bin']} binaries, verified {r['verified']}")
    return failed


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _f(x, d=1):
    return "--" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def _geo(x):
    return float(np.exp(np.mean(np.log(np.maximum(x, 1e-12))))) if len(x) else float("nan")


def _cls_tex(cl) -> str:
    """'ACN n=50' -> 'ACN, $n=50$'"""
    src, _, size = cl.partition(" ")
    return f"{src}, ${size}$" if size else cl


def _nominal_grid(inst) -> float:
    """Nominal grid capacity W_G of an instance (the available power W_t varies below it)."""
    return float(getattr(inst, "meta", {}).get("grid_nominal", inst.W.max()))


def _write_csv(rows, path, first=()):
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in first, first.index(k) if k in first else k))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_rows(out, plan, issues, limit=None, setting=None):
    """One row per instance, with the results of every formulation run on it.
    limit: time limit of the experiment; setting: dict shared by the calls, in which the
    threads and the machine of the cached runs are collected. A cached run made with
    another limit, or runs made with different threads or on different machines, would
    make the comparison of the times meaningless: they are reported as issues."""
    rows = []
    setting = {} if setting is None else setting
    for key, i, inst, methods in plan:
        row = {"key": key, "id": i, "name": inst.name, "class": f"{inst.source} n={inst.n}", "source": inst.source,
               "n": inst.n, "m": inst.m, "H": inst.H}
        done = []
        for m in methods:
            f = Path(out) / "runs" / f"{m}_{key}.json"
            if not f.exists():
                issues.append(f"{m} on {key}: no result (run missing or failed)")
                continue
            with open(f, encoding="utf-8") as fh:
                r = json.load(fh)
            done.append(m)
            if limit is not None and abs(float(r.get("limit", limit)) - limit) > 1e-9:
                issues.append(f"{m} on {key}: cached run made with a limit of {r['limit']:g} s, not {limit:g} s "
                              f"(delete runs/{m}_{key}.json to run it again)")
            mach = r.get("machine") or {}
            where = (r.get("threads"), mach.get("platform"), mach.get("processor"), mach.get("logical_cores"),
                     mach.get("gurobi"))
            setting.setdefault(where, f"{m} on {key}")
            ub = r["bound_floor"] if r["bound_floor"] is not None else r["bound"]
            acc = r["accepted"]
            row.update({f"{m}_status": r["status"], f"{m}_accepted": acc, f"{m}_bound": ub,
                        # a bound below the incumbent by solver round-off (1e-7) counts as a zero gap
                        f"{m}_gap": (max(0.0, 100.0 * (ub - acc) / ub) if acc is not None and ub else
                                     (0.0 if r["status"] == "optimal" else None)),
                        f"{m}_time": r["time"], f"{m}_solver": r.get("solver_time"), f"{m}_bin": r["n_bin"],
                        f"{m}_vars": r["n_vars"], f"{m}_rows": r["n_constrs"], f"{m}_nodes": r["nodes"],
                        f"{m}_verified": r["verified"]})
            if r["verified"] is False:
                issues.append(f"{m} on {key}: schedule NOT verified: {r.get('verification_error')}")
        proven = [m for m in done if row[f"{m}_status"] == "optimal"]
        if len({round(row[f"{m}_accepted"], 6) for m in proven}) > 1:
            issues.append(f"{key}: optimal values differ {[(m, row[f'{m}_accepted']) for m in proven]}")
        for a in done:                                        # a proven bound below another incumbent
            for b in done:
                if a != b and row.get(f"{a}_bound") is not None and row.get(f"{b}_accepted") is not None \
                        and row[f"{b}_accepted"] > row[f"{a}_bound"] + 1e-6:
                    issues.append(f"{key}: incumbent of {b} above the bound of {a}")
        vals = [row[f"{m}_accepted"] for m in done if row[f"{m}_accepted"] is not None]
        row["best"] = max(vals) if vals else None
        row["proven"] = bool(proven)
        for ref in done:
            if ref != "UF" and "UF" in done:
                row[f"speed_up_{ref}"] = row[f"{ref}_time"] / max(row["UF_time"], 1e-9)
        rows.append(row)
    return rows


def instances_table(plan, tdir, ddir):
    """Characteristics of the instances of the study (tab:instances)."""
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Characteristics of the instances.}", r"\label{tab:instances}", r"\small",
         r"\begin{tabular}{rlrrrrrr}", r"\toprule",
         r"Id & Source & $n$ & $m$ & $H$ & $W_G$ & $\bar n$ & Energy/grid \\", r"\midrule"]
    data = []
    for key, i, inst, _ in plan:
        ok = inst.CT.any(axis=1)                 # bookings with a compatible type (sizing rule)
        pres = np.zeros(inst.H + 2)
        np.add.at(pres, inst.r[ok], 1)
        np.add.at(pres, inst.d[ok] + 1, -1)
        f = inst.features()
        peak = int(np.cumsum(pres).max())
        L.append(f"{i if i else inst.name} & {inst.source} & {inst.n} & {inst.m} & {inst.H} & "
                 f"{_nominal_grid(inst):.1f} & {peak} & {f['energy_to_grid']:.2f} \\\\")
        data.append({"id": i, "key": key, "name": inst.name, "source": inst.source, "n": inst.n, "m": inst.m,
                     "H": inst.H, "W_G": _nominal_grid(inst), "W_mean": float(inst.W.mean()),
                     "W_min": float(inst.W.min()), "peak_vehicles": peak, **f})
    L += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
          r"{\footnotesize Number of bookings $n$, of chargers $m$ and of slots $H$; nominal grid "
          r"capacity $W_G$ (kW); largest number of vehicles with a compatible charger present at once "
          r"$\bar n$; ratio of the booked energy to the grid energy over the occupied slots.}", r"\end{table}"]
    (tdir / f"{CODE}_instances.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    _write_csv(data, ddir / f"{CODE}_instances.csv", first=("id", "key", "name", "source", "n", "m", "H"))


def _time_cell(r, m, limit):
    if f"{m}_status" not in r:
        return "--"
    return _f(r[f"{m}_time"]) if r[f"{m}_status"] == "optimal" else rf"$>${limit:.0f}"


def instance_table(rows, methods, limit, path):
    """Per instance (tab:exact_methods)."""
    refs = [m for m in methods if m != "UF"] if "UF" in methods else []
    T = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{The unit-flow formulation against Gurobi on the time-indexed formulations, per instance.}",
         r"\label{tab:exact_methods}", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         rf"\begin{{tabular}}{{rrrr{'rrr' * len(methods)}{'r' * len(refs)}}}", r"\toprule",
         " & & & " + "".join(rf"& \multicolumn{{3}}{{c}}{{{LABEL[m]}}} " for m in methods)
         + (rf"& \multicolumn{{{len(refs)}}}{{c}}{{Speed-up}} " if refs else "") + r"\\",
         "Id & $n$ & $m$ & $z^\\star$ " + "& Time & Gap & Nodes " * len(methods)
         + "".join(f"& {SHORT[m]} " for m in refs) + r"\\", r"\midrule"]
    for k, r in enumerate(rows):
        if k and r["class"] != rows[k - 1]["class"]:
            T.append(r"\addlinespace")
        best = "--" if r["best"] is None else f"{r['best']:.0f}" + ("" if r["proven"] else r"$^{\dagger}$")
        line = f"{r['id'] if r['id'] else r['name']} & {r['n']} & {r['m']} & {best} "
        for m in methods:
            line += (f"& {_time_cell(r, m, limit)} & {_f(r.get(f'{m}_gap'), 2)} & {_f(r.get(f'{m}_nodes'), 0)} ")
        for m in refs:
            line += f"& {_f(r.get(f'speed_up_{m}'), 1)} "
        T.append(line + r"\\")
    T += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
          rf"{{\footnotesize Time limit {limit:.0f}~s. $z^\star$: optimal number of accepted bookings "
          r"($\dagger$: best value found, optimality not proven); Time (s): construction of the model, search and "
          r"construction of the schedule; Gap (\%) between the incumbent and the upper bound; Nodes of the "
          r"branch-and-bound tree; Speed-up: time of the formulation over the time of $(\mathrm{UF})$, a lower "
          r"bound when the formulation reaches the limit.}", r"\end{table}"]
    Path(path).write_text("\n".join(T) + "\n", encoding="utf-8")


def summary_table(rows, methods, limit, path):
    """Per class of instances (tab:exact_summary). Returns the lines of the text report."""
    refs = [m for m in methods if m != "UF"] if "UF" in methods else []
    classes = []
    for r in rows:
        if r["class"] not in classes:
            classes.append(r["class"])
    head = f"{'class':12s} {'inst':>4s}"
    for m in methods:
        head += f" | {m:>3s}: {'opt':>3s} {'mean t':>8s} {'med t':>8s} {'max t':>8s} {'gap%':>5s} {'binaries':>9s}"
    for m in refs:
        head += f" | {'x ' + m + ' (geo)':>10s}"
    L = [head]
    S = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{The unit-flow formulation against Gurobi on the time-indexed formulations, per class of "
         r"instances.}", r"\label{tab:exact_summary}", r"\small", r"\setlength{\tabcolsep}{4pt}",
         rf"\begin{{tabular}}{{lr{'rrrrr' * len(methods)}{'r' * len(refs)}}}", r"\toprule",
         " & " + "".join(rf"& \multicolumn{{5}}{{c}}{{{LABEL[m]}}} " for m in methods)
         + (rf"& \multicolumn{{{len(refs)}}}{{c}}{{Speed-up}} " if refs else "") + r"\\",
         "Class & Inst. " + "& Opt. & Mean & Max. & Gap & Bin. " * len(methods)
         + "".join(f"& {SHORT[m]} " for m in refs) + r"\\", r"\midrule"]
    for cl in classes + ["All"]:
        sel = rows if cl == "All" else [r for r in rows if r["class"] == cl]
        line, tex = f"{cl:12s} {len(sel):4d}", f"{_cls_tex(cl)} & {len(sel)} "
        for m in methods:
            d = [r for r in sel if f"{m}_status" in r]
            if not d:
                line += " |  (not run)" + " " * 40
                tex += "& -- & -- & -- & -- & -- "
                continue
            ts = [r[f"{m}_time"] for r in d]
            gaps = [r[f"{m}_gap"] if r[f"{m}_gap"] is not None else 100.0 for r in d]
            nopt = sum(r[f"{m}_status"] == "optimal" for r in d)
            nbin = np.mean([r[f"{m}_bin"] for r in d])
            line += (f" | {m:>3s}: {nopt:3d} {np.mean(ts):8.1f} {np.median(ts):8.1f} {np.max(ts):8.1f} "
                     f"{np.mean(gaps):5.2f} {nbin:9.0f}")
            tex += f"& {nopt} & {np.mean(ts):.1f} & {np.max(ts):.1f} & {np.mean(gaps):.2f} & {nbin:.0f} "
        for m in refs:
            g = _geo([r[f"speed_up_{m}"] for r in sel if f"speed_up_{m}" in r])
            line += f" | {g:10.1f}"
            tex += f"& {_f(g, 1)} "
        if cl == "All":
            S.append(r"\midrule")
        L.append(line)
        S.append(tex + r"\\")
    S += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
          rf"{{\footnotesize Time limit {limit:.0f}~s. Opt.: instances solved to proven optimality; Mean, Max.: "
          r"mean and largest time (s), an unsolved instance counting for the time used; Gap: mean gap (\%); Bin.: "
          r"mean number of binary variables; Speed-up: geometric mean of the time of the formulation over the "
          r"time of $(\mathrm{UF})$.}", r"\end{table}"]
    Path(path).write_text("\n".join(S) + "\n", encoding="utf-8")
    return L


def large_table(rows, methods, limit, path):
    """Larger instances (tab:exact_large)."""
    T = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{The unit-flow formulation on larger instances.}", r"\label{tab:exact_large}", r"\small",
         rf"\begin{{tabular}}{{lrrr{'rrrrrr' * len(methods)}}}", r"\toprule"]
    if len(methods) > 1:
        T.append(" & & & " + "".join(rf"& \multicolumn{{6}}{{c}}{{{LABEL[m]}}} " for m in methods) + r"\\")
    T += ["Source & $n$ & $m$ & $H$ " + "& Acc. & UB & Gap & Time & Nodes & Bin. " * len(methods) + r"\\",
          r"\midrule"]
    for r in rows:
        line = f"{r['source']} & {r['n']} & {r['m']} & {r['H']} "
        for m in methods:
            if f"{m}_status" not in r:
                line += "& -- & -- & -- & -- & -- & -- "
                continue
            line += (f"& {_f(r[f'{m}_accepted'], 0)} & {_f(r[f'{m}_bound'], 0)} & {_f(r[f'{m}_gap'], 2)} "
                     f"& {_time_cell(r, m, limit)} & {_f(r[f'{m}_nodes'], 0)} & {r[f'{m}_bin']} ")
        T.append(line + r"\\")
    T += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
          rf"{{\footnotesize Time limit {limit:.0f}~s. Acc.: accepted bookings of the incumbent; UB: upper bound; "
          r"Gap (\%); Time (s), construction of the model and of the schedule included; Nodes of the "
          r"branch-and-bound tree; Bin.: binary variables.}", r"\end{table}"]
    Path(path).write_text("\n".join(T) + "\n", encoding="utf-8")


def report(out, main, larger, methods, extra_methods, limit, log=print, figures=True):
    """Tables, statistical tests, figures and text report from the cached runs.
    Returns (rows of the instances of the study, issues)."""
    out = Path(out)
    tdir, ddir, sdir = out / "Tables", out / "Data", out / "Statistical Tests"
    for d in (tdir, ddir, sdir):
        d.mkdir(parents=True, exist_ok=True)
    issues = []
    setting = {}
    rows = load_rows(out, [(k, i, inst, methods) for k, i, inst in main], issues, limit, setting)
    big = load_rows(out, [(k, i, inst, extra_methods) for k, i, inst in larger], issues, limit, setting)
    if len(setting) > 1:
        issues.append("the cached runs were NOT all made with the same threads, machine and Gurobi version: "
                      + "; ".join(f"{w} (first: {first})" for w, first in setting.items()))
    # the machine of the runs (recorded in each of them), not the one that writes the report
    where = sorted(setting, key=str)
    runs_on = ("; ".join(f"threads {w[0]}, {w[1]}, {w[2]}, {w[3]} logical cores, Gurobi {w[4]}" for w in where)
               if where else str(machine()))
    L = [f"Exact method against Gurobi on the time-indexed formulations: time limit {limit:.0f} s, "
         f"{len(rows)} instance(s), formulations {', '.join(methods)}", f"runs made on: {runs_on}", ""]
    if rows:
        instances_table([(k, i, inst, methods) for k, i, inst in main], tdir, ddir)
        _write_csv(rows + big, ddir / f"{CODE}_exact_methods.csv",
                   first=("key", "id", "name", "class", "source", "n", "m", "H", "best", "proven"))
        instance_table(rows, methods, limit, tdir / f"{CODE}_exact_methods.tex")
        L += summary_table(rows, methods, limit, tdir / f"{CODE}_exact_summary.tex")
        tests = A.paired_tests(rows, methods, key="time")
        tests["solver_time"] = A.paired_tests(rows, methods, key="solver")
        A.write_tests(tests, sdir / f"{CODE}_exact_tests.json")
        if tests["comparisons"]:
            (tdir / f"{CODE}_exact_tests.tex").write_text(A.tests_table(tests, limit, "tab:exact_tests"),
                                                          encoding="utf-8")
            L += ["", "Statistical tests on the times (model, search and schedule):"] + A.tests_text(tests)
            L += ["", "Same tests on the time of the Gurobi search alone:"] + A.tests_text(tests["solver_time"])
        if figures:
            A.figures(rows, methods, limit, out / "Figures", code=CODE, larger=big)
    if big:
        large_table(big, extra_methods, limit, tdir / f"{CODE}_exact_large.tex")
        L += ["", "Larger instances:"]
        for r in big:
            for m in extra_methods:
                if f"{m}_status" in r:
                    L.append(f"  {r['key']:10s} {m:3s} {r[f'{m}_status']:10s} accepted {_f(r[f'{m}_accepted'], 0)}, "
                             f"bound {_f(r[f'{m}_bound'], 0)}, {r[f'{m}_time']:.1f} s, "
                             f"{_f(r[f'{m}_nodes'], 0)} nodes, {r[f'{m}_bin']} binaries")
    L += ["", "cross-checks: " + ("all schedules verified, optimal values and bounds consistent" if not issues
                                  else "ISSUES: " + "; ".join(issues))]
    (out / "report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    for line in L:
        log(line)
    return rows, issues


def main(argv=None):
    ap = argparse.ArgumentParser(description="Exact method (UF) against Gurobi on the time-indexed formulations")
    ap.add_argument("--name", default="appendix_d_exact_methods")
    ap.add_argument("--limit", type=float, default=1800.0)
    ap.add_argument("--ids", default="1-40")
    ap.add_argument("--methods", default="UF,C,P")
    ap.add_argument("--extra", default="")
    ap.add_argument("--extra-methods", default="UF")
    ap.add_argument("--threads", default="all")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    methods = [m for m in a.methods.split(",") if m]
    extra_methods = [m for m in a.extra_methods.split(",") if m]
    if set(methods + extra_methods) - set(LABEL):
        raise ValueError(f"methods: among {sorted(LABEL)}")
    threads = (os.cpu_count() or 1) if a.threads == "all" else int(a.threads)
    main_insts, larger = instances(parse_ids(a.ids), [int(x) for x in a.extra.split(",") if x])
    out = HERE / "results" / a.name
    failed = []
    if not a.report:
        failed = run(out, [(k, i, inst, methods) for k, i, inst in main_insts]
                     + [(k, i, inst, extra_methods) for k, i, inst in larger], a.limit, threads)
    _, issues = report(out, main_insts, larger, methods, extra_methods, a.limit)
    print(f"\nresults -> {out}")
    return 1 if (failed or issues) else 0


if __name__ == "__main__":
    sys.exit(main())
