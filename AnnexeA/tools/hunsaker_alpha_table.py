"""Precompute alpha_T, the optimal competitive ratio of the ORMP (Hunsaker et al. 2003, Corollary 4.13),
for T = 1..Tmax, into cri/problems/ppmsp/data/hunsaker_alpha.json (used by the HUNSAKER baseline).

    python tools/hunsaker_alpha_table.py [Tmax]

LP with prefix sums S(t) = a(1) + ... + a(t), so that every allocation constraint has three terms:
    min sum_t x(t)  s.t.  S(T) = 1,  (T - i + 1) x(t) - S(t) + S(i - 1) >= 0  (1 <= i <= t <= T),  a, x >= 0
and alpha_T = 1 / optimum. Checked against Table 4.1 of the paper (T = 2, 3, 10, 50, 100, 200).
"""

import json
import os
import sys
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

TABLE_4_1 = {2: 4 / 3, 3: 1.5, 4: 44 / 27, 5: 1.71329, 10: 1.93576, 25: 2.14951, 50: 2.26470, 100: 2.35061,
             200: 2.41585, 300: 2.44663}


def alpha(T):
    # variables: a(1..T) -> 0..T-1, S(0..T) -> T..2T, x(1..T) -> 2T+1..3T
    na, nS = T, T + 1
    ix_a = lambda t: t - 1                       # noqa: E731
    ix_S = lambda t: na + t                      # noqa: E731
    ix_x = lambda t: na + nS + t - 1             # noqa: E731
    nv = na + nS + T
    # equalities: S(0) = 0, S(t) - S(t-1) - a(t) = 0, S(T) = 1
    er, ec, ev, eb = [0], [ix_S(0)], [1.0], [0.0]
    for t in range(1, T + 1):
        r = len(eb)
        er += [r, r, r]; ec += [ix_S(t), ix_S(t - 1), ix_a(t)]; ev += [1.0, -1.0, -1.0]; eb.append(0.0)
    r = len(eb); er.append(r); ec.append(ix_S(T)); ev.append(1.0); eb.append(1.0)
    Aeq = coo_matrix((ev, (er, ec)), shape=(len(eb), nv)).tocsr()
    # inequalities in <= form: S(t) - S(i-1) - (T - i + 1) x(t) <= 0
    rr, cc, vv = [], [], []
    k = 0
    for t in range(1, T + 1):
        for i in range(1, t + 1):
            rr += [k, k, k]; cc += [ix_S(t), ix_S(i - 1), ix_x(t)]; vv += [1.0, -1.0, -float(T - i + 1)]
            k += 1
    Aub = coo_matrix((vv, (rr, cc)), shape=(k, nv)).tocsr()
    c = np.zeros(nv); c[na + nS:] = 1.0
    res = linprog(c, A_ub=Aub, b_ub=np.zeros(k), A_eq=Aeq, b_eq=np.array(eb), bounds=[(0, None)] * nv,
                  method="highs")
    if res.status != 0:
        raise RuntimeError(f"T={T}: {res.message}")
    return 1.0 / res.fun


if __name__ == "__main__":
    Tmax = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    out_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "cri", "problems", "ppmsp", "data", "hunsaker_alpha.json")
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    table = {}
    t0 = time.time()
    for T in range(1, Tmax + 1):
        table[T] = alpha(T)
        if T in TABLE_4_1:
            ok = abs(table[T] - TABLE_4_1[T]) < 1e-4
            print(f"T={T:4d} alpha={table[T]:.6f}  Table 4.1 {TABLE_4_1[T]:.6f}  {'OK' if ok else 'DIFFERS'}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({"source": "Hunsaker et al. (2003), Corollary 4.13, solved with HiGHS",
                   "alpha": {str(k): v for k, v in table.items()}}, f, indent=0)
    print("written", out_p)
