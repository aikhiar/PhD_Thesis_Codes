"""
complements.py -- three tables of the paper computed from the raw output of the methods_comparison
study (experiment 2), which the report of methods_comparison does not include.

    python complements.py <methods_comparison study folder> [--out results/appendix_d_complements]
    (the study folder: absolute, or relative to the folder you run from)

    Tables/appendix_d_loss_by_class.tex   bookings rejected in excess of the optimum, per class of
                                  instances and method, and computing times
    Tables/appendix_d_plateau.tex         convergence of the metaheuristics: excess rejections after
                                  100, 200, 300 and 500 generations, improvement made after
                                  generation 250, runs that reach the optimum
    Tables/appendix_d_tuning.tex          parameter values selected by the tuning (number of instances)
    Data/appendix_d_complements.json      every number of the three tables
"""
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from problems.appendix_d.model import AppendixDInstance  # noqa: E402,F401  (to unpickle the results)

METHODS = ["GA", "ES", "PSO", "GWO", "CS"]
LABEL = {"GA": "GA", "ES": "ES", "PSO": "PSO", "GWO": "GWO", "CS": "CSA"}
ORDER = ["ACN n=50", "ACN n=100", "ACN n=150", "ACN n=200", "SIM n=50", "SIM n=100", "SIM n=150", "SIM n=200"]


def cls_tex(c):
    src, _, size = c.partition(" ")
    return f"{'ACN' if src == 'ACN' else 'Simulated'}, ${size}$"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    study = Path(argv[0]).resolve()
    out = Path(argv[argv.index("--out") + 1]).resolve() if "--out" in argv else HERE / "results" / "appendix_d_complements"
    (out / "Tables").mkdir(parents=True, exist_ok=True)
    (out / "Data").mkdir(parents=True, exist_ok=True)
    with open(study / "appendix_d_instances.pkl", "rb") as fh:
        insts = pickle.load(fh)
    insts = list(insts.values()) if isinstance(insts, dict) else list(insts)
    assert all(isinstance(x, AppendixDInstance) for x in insts), "not a Appendix D study"
    refs = {int(k): float(v) for k, v in json.load(open(study / "appendix_d_run_manifest.json"))["exact_references"].items()}
    cls = {i: f"{x.source} n={x.n}" for i, x in enumerate(insts, 1)}
    d = pd.read_csv(study / "Results Analysis" / "appendix_d_results_tidy.csv")
    d = d[~d.crashed.astype(bool)]
    d["extra"] = d.objective - d.instance.map(refs)
    d["cls"] = d.instance.map(cls)
    d["n"] = d.instance.map(lambda i: insts[i - 1].n)
    d["lost_pct"] = 100.0 * d.extra / (d.n - d.instance.map(refs))
    mh = d[d.method != "UF"]
    data = {}

    # ── 1. excess rejections per class ───────────────────────────────────────
    ex = mh.groupby(["cls", "method"]).extra.mean().unstack()
    t_uf = d[d.method == "UF"].groupby("cls").time.mean()
    t_mh = mh.groupby(["cls", "method"]).time.mean().unstack()
    rej = d[d.method == "UF"].groupby("cls").objective.mean()
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Bookings rejected by the metaheuristics in excess of the optimum, per class of instances.}",
         r"\label{tab:loss_by_class}", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lr" + "r" * len(METHODS) + "rrr}", r"\toprule",
         r" & Optimum & \multicolumn{" + str(len(METHODS)) + r"}{c}{Excess rejected bookings} & "
         r"\multicolumn{3}{c}{Time per run (s)} \\",
         r"Class & rejected " + "".join(f"& {LABEL[m]} " for m in METHODS) + r"& UF & \multicolumn{2}{c}{metaheuristics} \\",
         r" & " + "& " * len(METHODS) + r"& & min. & max. \\", r"\midrule"]
    rows = {}
    for c in ORDER + ["All"]:
        sel_ex = ex.mean() if c == "All" else ex.loc[c]
        tu = d[d.method == "UF"].time.mean() if c == "All" else t_uf[c]
        tm = mh.groupby("method").time.mean() if c == "All" else t_mh.loc[c]
        r0 = d[d.method == "UF"].objective.mean() if c == "All" else rej[c]
        if c == "All":
            L.append(r"\midrule")
        best = min(sel_ex[m] for m in METHODS)
        cells = "".join((rf"& \textbf{{{sel_ex[m]:.2f}}} " if abs(sel_ex[m] - best) < 1e-12 else f"& {sel_ex[m]:.2f} ")
                        for m in METHODS)
        L.append(f"{'All' if c == 'All' else cls_tex(c)} & {r0:.1f} {cells}& {tu:.1f} & {tm.min():.1f} & {tm.max():.1f} \\\\")
        rows[c] = {"optimal_rejected": float(r0), "excess": {m: float(sel_ex[m]) for m in METHODS},
                   "time_uf": float(tu), "time_mh": {m: float(tm[m]) for m in METHODS}}
    allost = mh.groupby("method").lost_pct.mean()
    L.append(r"\midrule")
    L.append(r"Lost (\%) & " + "".join(f"& {allost[m]:.2f} " for m in METHODS) + r"& & & \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
          r"{\footnotesize Optimum rejected: mean optimal number of rejected bookings of the class; excess rejected "
          r"bookings: mean over the 30 runs and the five instances of the class of the number of bookings rejected "
          r"beyond the optimum (best method in bold); Lost: bookings accepted at the optimum but rejected, in percent "
          r"of the optimal number of accepted bookings, mean over all runs; time per run: mean computing time of a run "
          r"on one thread, for the exact method (UF) and for the fastest and slowest metaheuristics of the class.}",
          r"\end{table}"]
    (out / "Tables" / "appendix_d_loss_by_class.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    data["loss_by_class"] = rows
    data["lost_pct"] = {m: float(allost[m]) for m in METHODS}

    # ── 2. convergence plateau ───────────────────────────────────────────────
    marks = (100, 200, 300, 500)
    plateau = {}
    for m in METHODS:
        with open(study / "Evaluation Results" / f"appendix_d_{m}_results.pkl", "rb") as fh:
            res = pickle.load(fh)["results"]
        at = {g: [] for g in marks}
        share_after, flat, gained = [], 0, []
        runs = opt_runs = 0
        inst_opt = set()
        for iid, lst in res.items():
            for r in lst:
                if r.get("_crashed") or not r.get("progress"):
                    continue
                p = np.asarray(r["progress"], float)
                runs += 1
                for g in marks:
                    at[g].append(p[min(g, len(p)) - 1] - refs[int(iid)])
                gain = p[0] - p[-1]
                share_after.append((p[249] - p[-1]) / gain if gain > 0 else 0.0)
                flat += p[249] == p[-1]
                gained.append(p[249] - p[-1])
                if abs(p[-1] - refs[int(iid)]) < 1e-9:
                    opt_runs += 1
                    inst_opt.add(int(iid))
        plateau[m] = {"excess_at": {g: float(np.mean(at[g])) for g in marks},
                      "improvement_after_250_pct": float(100 * np.mean(share_after)),
                      "runs_flat_after_250_pct": float(100 * flat / runs), "gain_after_250": float(np.mean(gained)),
                      "runs_optimal_pct": float(100 * opt_runs / runs), "instances_with_optimal_run": sorted(inst_opt),
                      "runs": runs}
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Convergence of the metaheuristics over the generations.}", r"\label{tab:plateau}", r"\small",
         r"\begin{tabular}{l" + "r" * len(marks) + "rrrr}", r"\toprule",
         r" & \multicolumn{" + str(len(marks)) + r"}{c}{Excess rejected bookings after $g$ generations} & "
         r"Improvement & Runs not & \multicolumn{2}{c}{Optimum reached} \\",
         r"Method " + "".join(f"& $g={g}$ " for g in marks) + r"& after 250 (\%) & improved & runs (\%) & instances \\",
         r"\midrule"]
    for m in METHODS:
        q = plateau[m]
        L.append(f"{LABEL[m]} " + "".join(f"& {q['excess_at'][g]:.2f} " for g in marks)
                 + f"& {q['improvement_after_250_pct']:.1f} & {q['runs_flat_after_250_pct']:.0f}\\% "
                 f"& {q['runs_optimal_pct']:.1f} & {len(q['instances_with_optimal_run'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
          r"{\footnotesize Excess rejected bookings: best schedule of the run after $g$ generations, minus the optimum, "
          r"mean over the 1200 runs (30 per instance); improvement after 250: share of the improvement of a run, from "
          r"its initial population to generation 500, made after generation 250, mean over the runs; runs not "
          r"improved: runs whose best schedule does not change after generation 250; optimum reached: share of the "
          r"runs that end at the optimum, and number of instances (of 40) on which at least one run does.}",
          r"\end{table}"]
    (out / "Tables" / "appendix_d_plateau.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    data["plateau"] = plateau

    # ── 3. tuned parameter values ────────────────────────────────────────────
    grids = {"pc": [0.01, 0.03, 0.05, 0.10], "mutation_rate": [0.01, 0.05, 0.15, 0.20],
             "p_abandon": [0.10, 0.20, 0.25, 0.30]}
    chosen = {}
    for m in METHODS:
        with open(study / "Parameter Tuning" / f"appendix_d_{m}_best_params.pkl", "rb") as fh:
            bp = pickle.load(fh)
        chosen[m] = {k: Counter(round(float(p[k]), 4) for p in bp.values() if k in p) for k in grids}
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Parameter values selected by the tuning: number of instances, out of forty.}",
         r"\label{tab:tuning_choices}", r"\small",
         r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r" & \multicolumn{4}{c}{Perturbation intensity $p_c$} & \multicolumn{4}{c}{$p_m$ (GA, ES, PSO, GWO) or $p_a$ (CSA)} \\",
         r"Method & 0.01 & 0.03 & 0.05 & 0.10 & 0.01 / 0.10 & 0.05 / 0.20 & 0.15 / 0.25 & 0.20 / 0.30 \\", r"\midrule"]
    for m in METHODS:
        second = "p_abandon" if m == "CS" else "mutation_rate"
        L.append(f"{LABEL[m]} " + "".join(f"& {chosen[m]['pc'].get(v, 0)} " for v in grids["pc"])
                 + "".join(f"& {chosen[m][second].get(v, 0)} " for v in grids[second]) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
          r"{\footnotesize Values of the grid of Table~\ref{tab:tuning}; in the last four columns, the value of "
          r"$p_m$ is given first and that of $p_a$ second.}", r"\end{table}"]
    (out / "Tables" / "appendix_d_tuning.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    data["tuning"] = {m: {k: {str(v): c for v, c in sorted(cnt.items())} for k, cnt in chosen[m].items()} for m in METHODS}

    with open(out / "Data" / "appendix_d_complements.json", "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    print(f"tables -> {out / 'Tables'}")
    for m in METHODS:
        q = plateau[m]
        print(f"  {LABEL[m]:4s} excess {[round(q['excess_at'][g], 2) for g in marks]}, after 250: "
              f"{q['improvement_after_250_pct']:.1f}%, not improved {q['runs_flat_after_250_pct']:.0f}%, "
              f"optimal runs {q['runs_optimal_pct']:.1f}% on instances {q['instances_with_optimal_run']}")


if __name__ == "__main__":
    main()
