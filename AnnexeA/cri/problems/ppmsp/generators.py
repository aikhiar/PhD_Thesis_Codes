"""
Instance families. Every generator has the signature (n, T, rng, **params) -> Instance.

  omega      the generator used with the earlier methods (chargers with disjoint windows per charger,
             Level-2 powers, durations from battery energy demand)
  staggered  short windows (3-10 slots by default) at random positions: sparse dependency
  overnight  long windows: arrival in the first quarter of the horizon, departure in the last
  pcmax      unit durations and a common window (the P||Cmax special case, Proposition 3.2)
  single     one power level (Corollary 3.4: polynomial case)
  csv        instances read from a CSV file (columns s, e, p, w; 1-based slots), e.g. built from
             ACN-Data; the file is given by params['path']
New families are added by decorating a function with @family("name").
"""

import numpy as np

from .instance import Instance

LEVEL2 = np.array([3.3, 6.2, 7.2, 9.6, 11.5, 12.0, 14.4, 19.2, 22.0])          # kW
BATT = np.array([24.0, 30.0, 40.0, 52.0, 58.0, 62.0, 64.0, 75.0, 77.0, 82.0, 90.0, 100.0, 105.0])  # kWh

FAMILIES = {}


def family(name):
    def deco(f):
        FAMILIES[name] = f
        return f
    return deco


@family("omega")
def gen_omega(n, T, rng, chargers_ratio=1 / 3, powers=None, **_):
    powers = LEVEL2 if powers is None else np.asarray(powers, float)
    m = max(1, int(np.ceil(n * chargers_ratio)))
    part = np.zeros(n, int); part[:m] = np.arange(1, m + 1); part[m:] = rng.integers(1, m + 1, n - m)
    part = part[rng.permutation(n)]
    mp = powers[rng.integers(0, len(powers), m)]
    s = np.zeros(n, int); e = np.zeros(n, int); p = np.zeros(n, int); w = np.zeros(n)
    dt = 24.0 / T
    for j in range(1, m + 1):
        tasks = np.where(part == j)[0]; N = len(tasks)
        if N == 0:
            continue
        if 2 * N > T:
            step = max(1, T // N)
            for k, i in enumerate(tasks):
                s[i] = k * step + 1; e[i] = min(s[i] + step - 1, T)
        else:
            tt = np.sort(rng.choice(np.arange(1, T + 1), 2 * N, replace=False)).reshape(N, 2)
            for k, i in enumerate(tasks):
                s[i], e[i] = tt[k]
        for i in tasks:
            L = e[i] - s[i] + 1; b = rng.choice(BATT); tgt = max(1, int(0.8 * L))
            phmax = min(0.95, tgt * mp[j - 1] * dt / b); phmin = min(0.10, phmax)
            soc = rng.uniform(phmin, phmax)
            p[i] = int(np.clip(int(np.ceil(b * soc / (mp[j - 1] * dt))), 1, tgt)); w[i] = mp[j - 1]
    return Instance(s - 1, e - 1, p, w, T, family="omega", meta={"chargers": m})


@family("staggered")
def gen_staggered(n, T, rng, min_window=3, max_window=10, max_fill=0.8, powers=None, **_):
    powers = LEVEL2 if powers is None else np.asarray(powers, float)
    L = rng.integers(min_window, min(max_window, T) + 1, n)
    s = np.array([rng.integers(0, T - l + 1) for l in L])
    e = s + L - 1
    p = np.array([rng.integers(1, max(1, int(max_fill * l)) + 1) for l in L])
    w = powers[rng.integers(0, len(powers), n)]
    return Instance(s, e, p, w, T, family="staggered")


@family("overnight")
def gen_overnight(n, T, rng, arrive_frac=0.25, depart_frac=0.75, max_fill=0.4, powers=None, **_):
    powers = LEVEL2 if powers is None else np.asarray(powers, float)
    s = rng.integers(0, max(1, int(T * arrive_frac)), n)
    e = rng.integers(int(T * depart_frac) - 1, T, n)
    L = e - s + 1
    p = np.array([rng.integers(1, max(1, int(max_fill * l)) + 1) for l in L])
    w = powers[rng.integers(0, len(powers), n)]
    return Instance(s, e, p, w, T, family="overnight")


@family("pcmax")
def gen_pcmax(n, T, rng, powers=None, **_):
    powers = LEVEL2 if powers is None else np.asarray(powers, float)
    w = powers[rng.integers(0, len(powers), n)]
    return Instance(np.zeros(n, int), np.full(n, T - 1), np.ones(n, int), w, T, family="pcmax")


@family("single")
def gen_single(n, T, rng, power=7.2, min_window=3, max_window=12, **_):
    L = rng.integers(min_window, min(max_window, T) + 1, n)
    s = np.array([rng.integers(0, T - l + 1) for l in L]); e = s + L - 1
    p = np.array([rng.integers(1, l + 1) for l in L])
    return Instance(s, e, p, np.full(n, float(power)), T, family="single")


@family("csv")
def gen_csv(n, T, rng, path=None, **_):
    import pandas as pd
    d = pd.read_csv(path)
    if n and n < len(d):
        d = d.iloc[rng.choice(len(d), n, replace=False)]
    return Instance(d.s.values - 1, d.e.values - 1, d.p.values, d.w.values, T, family="csv", meta={"path": path})


def build_set(spec, base_seed=0):
    """spec: dict(name, family, n, T, count, seed, params). Returns a list of named instances."""
    fam = FAMILIES[spec["family"]]
    seed = int(spec.get("seed", 0)) + int(base_seed)
    rng = np.random.default_rng(seed)
    out = []
    for k in range(int(spec.get("count", 1))):
        inst = fam(int(spec["n"]), int(spec["T"]), rng, **spec.get("params", {}))
        inst.name = f"{spec['name']}_{spec['family']}_n{spec['n']}_T{spec['T']}_{k:03d}"
        inst.meta.update({"set": spec["name"], "rep": k, "seed": seed})
        out.append(inst)
    return out
