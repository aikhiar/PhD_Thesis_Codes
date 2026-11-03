"""
Faithfulness of the re-implemented earlier methods (G, RR, RLP, SA and the hybrids HSG, HSR, HSL).

For K instances of the omega generator and each method, R runs of the original code
(faithfulness/reference_old/old_code.py, verbatim) and R runs of the framework are compared:
Mann-Whitney U test on the peaks (Holm over all instance-method pairs), difference of the means.
"RLP-sameLP" feeds the framework's rounding kernel with the LP solution of the original code,
which isolates the rounding and repair logic from the choice of the optimal LP vertex.

    python -m faithfulness.check --instances 6 --runs 20          (in parallel, writes a table)
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cri.analysis.stats import holm  # noqa: E402
from cri.core.registry import make_method  # noqa: E402
from cri.problems.ppmsp import context as CX  # noqa: E402
from cri.problems.ppmsp import kernels as K  # noqa: E402
from cri.problems.ppmsp.instance import UNIT, Instance  # noqa: E402

SA_PRM = dict(T_0=100.0, T_min=0.01, alpha_c=0.95, L=50, alpha_p=0.3)


def _old():
    from faithfulness.reference_old import old_code as O
    return O


def to_instance(t, name):
    part, st, en, p, w, n, m, T, pmax = t
    return Instance(np.asarray(st) - 1, np.asarray(en) - 1, np.asarray(p), np.asarray(w, float), int(T), name=name,
                    family="omega")


def r01(v):
    return np.round(np.asarray(v, float) * UNIT) / UNIT


def run_pair(O, tup, inst, ctx, method, R, N, seed0=0):
    old, new = [], []
    sa_new = {"T0": SA_PRM["T_0"], "Tmin": SA_PRM["T_min"], "alpha_c": SA_PRM["alpha_c"], "L": SA_PRM["L"], "N": N,
              "alpha_p": SA_PRM["alpha_p"], "stop": "iterations"}
    for k in range(R):
        s = seed0 + k
        if method == "G":
            old.append(O.greedy_constructive_heuristic(tup, 0.0, s)[1])
            new.append(make_method({"class": "G"}).run(inst, ctx, None, s)["objective"])
        elif method == "RR":
            old.append(O.moser_tardos_inspired_algorithm(tup, N, 0.0, s)[1])
            new.append(make_method({"class": "RR", "params": {"N": N}}).run(inst, ctx, None, s)["objective"])
        elif method == "RLP":
            old.append(O.lp_based_rounding_algorithm(tup, num_trials=10, given_P_max=0.0, seed=s)[1])
            new.append(make_method({"class": "RLP", "params": {"trials": 10}}).run(inst, ctx, None, s)["objective"])
        elif method == "RLP-sameLP":
            old.append(O.lp_based_rounding_algorithm(tup, num_trials=10, given_P_max=0.0, seed=s)[1])
            if k == 0:
                alpha_old, _ = O.solve_lp_relaxation(tup)
                alpha_old = np.clip(np.asarray(alpha_old, float), 0, 1)
            K.seed_numba(s)
            best = min(K.rlp_trial(alpha_old, inst.s, inst.e, inst.p, inst.wu, float(UNIT))[1].max() for _ in range(10))
            new.append(best / UNIT)
        elif method == "SA":
            old.append(O.simulated_annealing(tup, N=N, given_P_max=0.0, seed=s, **SA_PRM)[1])
            new.append(make_method({"class": "SA", "params": sa_new}).run(inst, ctx, None, s)["objective"])
        elif method == "HSG":
            old.append(O.hybrid_sa_greedy(tup, N=N, given_P_max=0.0, seed=s, **SA_PRM)[1])
            new.append(make_method({"class": "SA", "params": dict(sa_new, init="G")}).run(inst, ctx, None, s)["objective"])
        elif method == "HSR":
            old.append(O.hybrid_sa_resampling(tup, N_init=1000, N=N, given_P_max=0.0, seed=s, **SA_PRM)[1])
            new.append(make_method({"class": "SA", "params": dict(sa_new, init="RR", init_spec={
                "class": "RR", "params": {"N": 1000}})}).run(inst, ctx, None, s)["objective"])
        elif method == "HSL":
            old.append(O.hybrid_sa_lp(tup, num_trials_lp=5, N=N, given_P_max=0.0, seed=s, **SA_PRM)[1])
            new.append(make_method({"class": "SA", "params": dict(sa_new, init="RLP", init_spec={
                "class": "RLP", "params": {"trials": 5}})}).run(inst, ctx, None, s)["objective"])
    return r01(old), r01(new)


METHODS = ["G", "RR", "RLP", "RLP-sameLP", "SA", "HSG", "HSR", "HSL"]


def instance_rows(k, R, N, sizes, methods, seed, verbose):
    """All methods on instance k (its own seeds: the result does not depend on how instances are spread
    over processes)."""
    import gurobipy as gp
    O = _old()
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.setParam("Threads", 1); env.start()
    n, T = sizes[k % len(sizes)]
    tup = O.generate_instance(n=n, T=T, m=int(np.ceil(n / 3)), seed=seed + k)
    inst = to_instance(tup, f"old_n{n}_T{T}_{k}")
    ctx = CX.make_context(inst, env, {})
    rows = []
    for m in methods:
        t0 = time.time()
        a, b = run_pair(O, tup, inst, ctx, m, R, N, seed0=100 * k)
        if np.allclose(a, a[0]) and np.allclose(b, b[0]) and abs(a[0] - b[0]) < 1e-9:
            p = 1.0
        else:
            p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        rows.append({"instance": inst.name, "n": n, "T": T, "method": m, "runs": R, "mean_old": a.mean(),
                     "mean_new": b.mean(), "rel_diff_pct": 100 * (b.mean() - a.mean()) / a.mean(),
                     "best_old": a.min(), "best_new": b.min(), "p_MWU": p, "seconds": time.time() - t0})
        if verbose:
            print(f"{inst.name:18s} {m:11s} old {a.mean():8.3f}  new {b.mean():8.3f}  p {p:.3g}", flush=True)
    return rows


def check(n_inst=6, R=20, N=10000, sizes=((20, 48), (40, 72), (75, 96)), methods=None, seed=2561, verbose=True,
          workers=1):
    """workers > 1: the instances run in parallel processes (spawn), one instance per process at a time."""
    methods = methods or METHODS
    args = [(k, R, N, sizes, methods, seed, verbose) for k in range(n_inst)]
    if workers > 1 and n_inst > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(workers, n_inst), mp_context=mp.get_context("spawn")) as ex:
            parts = list(ex.map(instance_rows, *zip(*args)))
    else:
        parts = [instance_rows(*a) for a in args]
    df = pd.DataFrame([r for part in parts for r in part])
    df["p_holm"] = holm(df.p_MWU.values)
    return df


def _auto_workers(n_inst):
    try:
        import psutil
        cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except Exception:
        cores = os.cpu_count() or 1
    return max(1, min(n_inst, cores - 1, 61 if os.name == "nt" else 10 ** 6))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=6)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--N", type=int, default=10000)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "04_faithfulness_check"))
    ap.add_argument("--workers", type=int, default=0, help="parallel processes (default: physical cores - 1)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    df = check(a.instances, a.runs, a.N, workers=a.workers or _auto_workers(a.instances))
    os.makedirs(a.out, exist_ok=True)
    df.to_pickle(os.path.join(a.out, "faithfulness.pkl"))
    df.to_csv(os.path.join(a.out, "faithfulness.csv"), index=False)
    from cri.analysis.tables import to_markdown
    summ = df.groupby("method").agg(pairs=("instance", "size"), mean_rel_diff_pct=("rel_diff_pct", "mean"),
                                    max_abs_rel_diff_pct=("rel_diff_pct", lambda x: x.abs().max()),
                                    min_p_holm=("p_holm", "min"))
    with open(os.path.join(a.out, "faithfulness.md"), "w", encoding="utf-8") as f:
        f.write("# Faithfulness of the re-implemented earlier methods\n\n")
        f.write(to_markdown(summ, 3) + "\n\n" + to_markdown(df.set_index(["instance", "method"]), 3) + "\n")
    print(summ.to_string())
    print("significant differences (Holm p < 0.01):", int((df.p_holm < 0.01).sum()))
