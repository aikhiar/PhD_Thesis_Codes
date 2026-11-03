"""
Experiment runner (problem-agnostic).

Pipeline
  1. build the instance sets of the configuration (deterministic seeds); save them
  2. run the exact references on the sets that ask for one
  3. run every experiment block in configuration order:
       tasks = instances x methods x budgets x seeds  (deterministic methods: first seed only)
     tasks are grouped by instance, so that a worker computes the instance context (relaxations,
     lower bounds, certificates) once and every method is charged the recorded cost of the
     artifacts it uses; contexts are also cached on disk for the next blocks
  4. tuning blocks (kind: tuning) evaluate a parameter grid on their own instance set and, with
     apply: true, write the chosen parameters into the defaults used by the later blocks
  5. aggregate every run into tables (pickled DataFrames) for the analysis and the report

Every run is saved as one pickle as soon as it ends, so an interrupted campaign resumes where it
stopped (run.resume: true). Times are wall-clock seconds.
"""

import copy
import hashlib
import itertools
import json
import os
import pickle
import re
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .config import deep_merge
from .resources import describe_environment, environment, machine, plan_workers, single_thread_env
from .registry import METHODS, PROBLEMS, make_method, set_defaults
from .results import DATA, ENVIRONMENT, FAITHFULNESS, LOGS, RAW, REPORT, ResultStore, RunResult

_ENV = None          # solver environment of the worker process
_PROBLEM = None
_CACHE_MAX_N = 3000  # contexts of larger instances are not written to the cache (set from run.cache_max_n)
_HOST = __import__("platform").node()   # machine that produced a run (campaigns may be split across devices)


def write_results_readme(out):
    """README.md of a results folder: what each folder holds."""
    text = f"""# Results of the campaign

| folder | content |
|---|---|
| `{RAW}/` | the source of truth: one pickle per (experiment block, instance) job, a list of runs with their peak, charged time, trace, guarantee, lower bound and status; `_references/` holds the exact references and `_contexts/` the bound summaries of each instance |
| `{DATA}/` | tables rebuilt from `{RAW}/` by `python -m cri report` (runs, enriched runs with every gap measure as .pkl and .csv, traces, Gurobi bound traces, solutions, references, instances, tuning tables, tuned defaults) |
| `{REPORT}/` | `report.html` / `report.md` (open these first), `index.md` (every table and figure with its caption) and one folder per section and experiment block: `tables/` (.csv, .md, .tex), `statistics/` (tests, ranks, effect sizes, their diagrams) and `figures/` (.png, .pdf) |
| `{FAITHFULNESS}/` | the original code against the re-implementation of the earlier methods |
| `{LOGS}/` | `run_log.txt` (every block, job and worker), `config_used.json`, `{ENVIRONMENT}` (machine, processor, cores, RAM and software versions of every start of `run`; recorded by this version of the framework), `audit.md`, `theory_verification.csv`/`.md` (step 3 of `run_all.bat`), the step markers of `run_all.bat` |

Nothing in `{DATA}/` or `{REPORT}/` needs to be kept: both are rebuilt from `{RAW}/`.
"""
    try:
        with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        pass


class LowDiskSpace(RuntimeError):
    pass


class SolverUnavailable(RuntimeError):
    """The solver licence cannot be obtained (a Gurobi WLS licence needs the network)."""


def free_gb(path):
    import shutil
    try:
        return shutil.disk_usage(path).free / 2 ** 30
    except OSError:
        return float("inf")


# ── helpers ─────────────────────────────────────────────────────────────────────

def keep_awake(on=True):
    """Ask Windows not to idle-sleep while this process runs (SetThreadExecutionState; released when the
    process exits). Wall-clock budgets are meaningless across a sleep. No effect elsewhere."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
        return True
    except Exception:
        return False


def _safe(s, maxlen=48):
    t = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(s)).strip("_")
    if len(t) > maxlen:
        t = t[:maxlen - 11] + "_" + hashlib.md5(t.encode()).hexdigest()[:10]
    return t


def run_key(inst_name, method_id, budget, seed):
    return f"{_safe(inst_name, 64)}__{_safe(method_id)}__b{_safe(budget)}__s{seed}"


def _hash(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def campaign_id(out, store):
    """Identifier of the campaign of a results folder (05_logs/campaign_id.txt); it names the folder of its
    context cache. A new results folder gets a new identifier (date, time, process), so a fresh run never
    reuses the contexts of another campaign; a resumed one finds its own again. A results folder made by an
    earlier version of the framework (runs stored, no identifier) keeps the shared cache folder ('')."""
    p = os.path.join(out, LOGS, "campaign_id.txt")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read().strip()
    legacy = os.path.isdir(store.raw_root) and any(os.scandir(store.raw_root))
    cid = "" if legacy else time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(cid + "\n")
    return cid


def resolve_path(cfg, p):
    """Paths of the config: ~ and environment variables are expanded; relative paths are relative to the
    configuration file."""
    p = os.path.expandvars(os.path.expanduser(p))
    base = os.path.dirname(cfg.get("_config_path", os.getcwd()))
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def expand_grid(spec, grid):
    """One spec per point of the parameter grid; ids such as SA[L=50,alpha_c=0.95]."""
    keys = sorted(grid)
    out = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        s = copy.deepcopy(spec)
        s.setdefault("params", {}).update(dict(zip(keys, vals)))
        s["id"] = f"{spec.get('id', spec['class'])}[" + ",".join(f"{k}={v}" for k, v in zip(keys, vals)) + "]"
        s["tuning_point"] = dict(zip(keys, vals))
        out.append(s)
    return out


def method_specs(cfg, names):
    """Experiment method entries are ids of the `methods` library of the config or inline specs."""
    lib = {}
    for m in cfg.get("methods", []):
        lib[m.get("id", m["class"])] = m
    out = []
    for e in names:
        if isinstance(e, str):
            if e not in lib:
                if e in METHODS:
                    out.append({"id": e, "class": e})
                    continue
                raise KeyError(f"unknown method '{e}' (not in the methods library nor a registered class)")
            out.append(copy.deepcopy(lib[e]))
        else:
            out.append(copy.deepcopy(e))
    for s in out:
        s.setdefault("id", s["class"])
    return out


def expand_sets(cfg):
    """Instance-set entries may give lists for n and T: one sub-set per (n, T) pair."""
    out = []
    for spec in cfg["instance_sets"]:
        ns = spec["n"] if isinstance(spec["n"], list) else [spec["n"]]
        Ts = spec["T"] if isinstance(spec["T"], list) else [spec["T"]]
        k = 0
        for n in ns:
            for T in Ts:
                s = dict(spec); s["n"], s["T"] = n, T
                s["seed"] = int(spec.get("seed", 0)) + 1000 * k
                k += 1
                out.append(s)
    return out


# ── instances ───────────────────────────────────────────────────────────────────

def build_instances(cfg, problem, store, log):
    path = os.path.join(store.data_root, "instances.pkl")
    sig = _hash([cfg["instance_sets"], cfg["run"].get("seed", 0)])
    if os.path.exists(path):
        with open(path, "rb") as f:
            saved = pickle.load(f)
        if saved.get("signature") == sig:
            sets = {k: [problem.instance_from_dict(d) for d in v] for k, v in saved["sets"].items()}
            log(f"instances: loaded {sum(map(len, sets.values()))} instances in {len(sets)} sets")
            return sets
    sets = {}
    for spec in expand_sets(cfg):
        insts = problem.build_set(spec, cfg["run"].get("seed", 0))
        sets.setdefault(spec["name"], []).extend(insts)
    rows = []
    for name, insts in sets.items():
        for inst in insts:
            rows.append({"set": name, "instance": inst.name, "family": inst.family, **inst.features()})
    store.save_table("instances", {"signature": sig, "sets": {k: [problem.instance_to_dict(i) for i in v]
                                                               for k, v in sets.items()}})
    store.save_table("instance_features", pd.DataFrame(rows))
    log(f"instances: built {len(rows)} instances in {len(sets)} sets")
    return sets


# ── worker side ─────────────────────────────────────────────────────────────────

def _init_worker(problem_name, run_cfg, defaults):
    global _ENV, _PROBLEM, _CACHE_MAX_N
    import warnings
    warnings.filterwarnings("ignore")
    _import_problem(problem_name)
    _CACHE_MAX_N = int(run_cfg.get("cache_max_n", 3000))
    _PROBLEM = PROBLEMS[problem_name]
    _ENV = _PROBLEM.make_env(run_cfg)
    if hasattr(_PROBLEM, "warmup") and run_cfg.get("warmup", True):
        try:
            _PROBLEM.warmup(_ENV)
        except Exception:
            pass
    set_defaults(defaults)


def _load_context(inst, ctx_cfg, cache_dir):
    p = os.path.join(cache_dir, f"{_safe(inst.name, 80)}__{_hash(ctx_cfg)}.pkl")
    if cache_dir and os.path.exists(p):
        try:
            with open(p, "rb") as f:
                ctx = pickle.load(f)
            ctx.env = _ENV
            ctx.inst = inst
            return ctx, p, set(ctx._val)
        except Exception:
            pass
    return _PROBLEM.make_context(inst, _ENV, ctx_cfg), p, set()


def _save_context(ctx, path, before):
    if not path or set(ctx._val) == before:
        return
    if getattr(ctx.inst, "n", 0) > _CACHE_MAX_N:          # very large contexts: used once, not cached
        return
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(ctx, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)


def _run_one(inst, ctx, spec, budget, seed, exp, keep_solution, stop_at_lb):
    m = make_method(spec)
    lb = ctx.get("lb") if stop_at_lb else None
    t0 = time.time()
    try:
        r = m.run(inst, ctx, budget, seed, lb)
    except Exception as ex:                                   # a failing method never stops a campaign
        r = {"status": "failed", "info": {"error": repr(ex), "traceback": traceback.format_exc()[-4000:]}}
    wall = time.time() - t0
    status = r.get("status", "ok")
    if budget and wall > 2.0 * float(budget) + 30.0:
        status = "suspect_timing"          # e.g. the machine slept during the run: re-run on resume
    x = r.get("solution")
    obj, feas = _PROBLEM.evaluate(inst, x) if x is not None else (np.nan, False)
    if x is not None and abs(obj - r.get("objective", obj)) > 1e-6:
        status = "inconsistent"
    info = dict(r.get("info", {}) or {})
    info["wall"] = wall
    info["class"] = spec["class"]
    info["host"] = _HOST
    if "tuning_point" in spec:
        info["tuning_point"] = json.dumps(spec["tuning_point"], sort_keys=True)
    return RunResult(experiment=exp, instance=inst.name, method=spec["id"], budget=budget, seed=seed,
                     objective=obj if x is not None else np.nan, time=r.get("time", np.nan),
                     own_time=r.get("own_time", np.nan), artifact_time=r.get("artifact_time", 0.0),
                     trace=list(r.get("trace", [])), guarantee=r.get("guarantee", np.nan),
                     lower_bound=r.get("lower_bound", np.nan), status=status,
                     feasible=bool(feas) and r.get("feasible", True), info=info,
                     solution=x if keep_solution else None)


def _work(job):
    """Run all tasks of one (experiment, instance) job in this worker. The runs are kept in one job
    file, rewritten at most every `flush` seconds and at the end (an interruption loses at most that)."""
    store = ResultStore(job["out"])
    inst = _PROBLEM.instance_from_dict(job["inst"])
    ctx, cpath, before = _load_context(inst, job["ctx_cfg"], job["cache_dir"])
    # unique even when one worker runs two parts of an instance within one clock tick (Windows clocks tick
    # every 0.5-16 ms; two parts with the same name would overwrite each other)
    fname = f"{_safe(inst.name, 64)}__{time.time_ns()}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    results, done = [], []
    last = time.time()
    for spec, budget, seed in job["tasks"]:
        rr = _run_one(inst, ctx, spec, budget, seed, job["exp"], job["keep_solution"], job["stop_at_lb"])
        results.append(rr)
        done.append((spec["id"], budget, seed, rr.status, rr.objective, rr.time))
        if set(ctx._val) != before:
            _save_context(ctx, cpath, before)
            before = set(ctx._val)
        if time.time() - last > job.get("flush", 30.0):
            store.save(job["exp"], fname, results)
            last = time.time()
    if job.get("record_lb", True):
        try:
            ctx.get("lb")                     # every instance gets its certified lower bound (for the metrics)
        except Exception:
            pass
    _save_context(ctx, cpath, before)
    store.save(job["exp"], fname, results)
    try:                                       # small summary of the context, stored with the results
        store.save("_contexts", f"{_safe(inst.name, 80)}__{_hash(job['ctx_cfg'])}", context_summary(ctx))
    except Exception:
        pass
    return {"exp": job["exp"], "instance": inst.name, "done": done, "costs": {k: float(v) for k, v in ctx._own.items()}}


def context_summary(ctx):
    """Bounds and costs of a context (a few hundred bytes; the full context stays in the cache)."""
    d = {"instance": ctx.inst.name}
    d.update({f"cost_{k}": float(v) for k, v in ctx._own.items()})
    v = ctx._val
    if "lb" in v:
        d["LB"] = float(v["lb"])
    if "lp" in v:
        d["P_LP"] = float(v["lp"][0])
    if "lb_cut" in v:
        d["LB_cut"] = float(v["lb_cut"][0])
    if "lb_conf" in v:
        d["LB_conf"] = float(v["lb_conf"][0]); d["conf_finished"] = bool(v["lb_conf"][2])
    return d


def _ref_work(job):
    inst = _PROBLEM.instance_from_dict(job["inst"])
    ctx, cpath, before = _load_context(inst, job["ctx_cfg"], job["cache_dir"])
    try:
        r = _PROBLEM.reference(inst, ctx, _ENV, job["ref_cfg"])
    except Exception as ex:
        r = {"opt": np.nan, "best": np.nan, "bound": np.nan, "proven": False, "time": np.nan,
             "status": "failed: " + repr(ex)}
    r.update({"instance": inst.name, "set": job["set"], "lb_certified": float(ctx.get("lb"))})
    _save_context(ctx, cpath, before)
    try:
        ResultStore(job["out"]).save("_contexts", f"{_safe(inst.name, 80)}__{_hash(job['ctx_cfg'])}",
                                     context_summary(ctx))
    except Exception:
        pass
    ResultStore(job["out"]).save("_references", _safe(inst.name, 80), r)
    return r


# ── master side ─────────────────────────────────────────────────────────────────

class Runner:
    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.problem_name = cfg.get("problem", "ppmsp")
        _import_problem(self.problem_name)
        self.problem = PROBLEMS[self.problem_name]
        self.out = resolve_path(cfg, cfg["run"].get("output", "results"))
        self.store = ResultStore(self.out)
        cache = cfg["run"].get("cache")
        base = resolve_path(cfg, cache) if cache else os.path.join(self.out, "cache")
        self.campaign_id = campaign_id(self.out, self.store)
        self.cache_dir = os.path.join(base, self.campaign_id, "contexts") if self.campaign_id \
            else os.path.join(base, "contexts")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.defaults = copy.deepcopy(cfg.get("defaults", {}))
        self._done = {}
        self.machine = machine()
        os.makedirs(os.path.join(self.out, LOGS), exist_ok=True)
        self._logf = open(os.path.join(self.out, LOGS, "run_log.txt"), "a", encoding="utf-8")
        write_results_readme(self.out)
        self._print = log

    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        self._print(line)
        self._logf.write(line + "\n"); self._logf.flush()

    # -- planning --
    def plan(self, only=None):
        sets = self.sets
        out = []
        for name, ex in self.cfg["experiments"].items():
            if only and name not in only:
                continue
            if not ex.get("enabled", True):
                continue
            specs = method_specs(self.cfg, ex["methods"])
            if ex.get("kind") == "tuning":
                grid = ex.get("grid", {})
                specs = [g for s in specs for g in (expand_grid(s, grid[s["id"]]) if s["id"] in grid else [s])]
            seeds = ex.get("seeds", [0])
            tasks = []
            reps = ex.get("reps")
            for set_name in ex["sets"]:
                pool_ = [i for i in sets[set_name] if reps is None or i.meta.get("rep") in reps]
                for inst in pool_[: ex.get("max_instances", None)]:
                    for spec in specs:
                        cls = METHODS[spec["class"]]
                        sd = seeds if getattr(cls, "stochastic", True) else seeds[:1]
                        for b in ex.get("budgets", [None]):
                            for s in sd:
                                tasks.append((set_name, inst, spec, b, s))
            out.append((name, ex, tasks))
        return out

    def completed(self, name):
        if name not in self._done:
            retry = self.cfg["run"].get("retry_failed", True)
            self._done[name] = {run_key(r.instance, r.method, r.budget, r.seed) for r in self.store.load_all(name)
                                if not (retry and r.status in ("failed", "suspect_timing"))}
        return self._done[name]

    def estimate(self, plan):
        rows = []
        for name, ex, tasks in plan:
            done = self.completed(name) if self.cfg["run"].get("resume", True) else set()
            todo = [t for t in tasks if run_key(t[1].name, t[2]["id"], t[3], t[4]) not in done]
            ub = getattr(self.problem, "uses_budget", None)
            cpu = sum((t[3] or 0.0) for t in todo if ub is None or ub(t[2], self.defaults))
            w, _ = self.block_workers(ex, [t[2] for t in tasks])
            rows.append({"experiment": name, "tasks": len(tasks), "remaining": len(todo), "workers": w,
                         "budget_seconds": cpu, "est. hours": cpu / max(1, w) / 3600})
        return pd.DataFrame(rows)

    def threads_per_task(self, specs):
        """Largest number of solver threads a task of these specs uses (Gurobi variants with threads > 1)."""
        t = 1
        for sp in specs:
            prm = dict(self.defaults.get(sp["class"], {}))
            prm.update(sp.get("params", {}) or {})
            try:
                t = max(t, int(prm.get("threads", 1) or 1))
            except (TypeError, ValueError):
                pass
        return t

    def block_workers(self, ex, specs, fresh=False):
        m = machine() if fresh else self.machine
        return plan_workers(self.cfg["run"], ex, self.threads_per_task(specs), m)

    @property
    def workers(self):
        return plan_workers(self.cfg["run"], {}, 1, self.machine)[0]

    def _pool(self, defaults, workers=None):
        import multiprocessing as mp
        single_thread_env()                          # BLAS / OpenMP: 1 thread per worker process
        # spawn on every platform: each worker starts clean (no forked solver environment or thread pools)
        return ProcessPoolExecutor(max_workers=int(workers or self.workers), initializer=_init_worker,
                                   initargs=(self.problem_name, self.cfg["run"], defaults),
                                   mp_context=mp.get_context("spawn"))

    def warm_master(self):
        """Compile the Numba kernels once in the master (the workers then load them from the cache instead
        of compiling them concurrently)."""
        if not hasattr(self.problem, "warmup"):
            return
        t0 = time.time()
        try:
            env = self.problem.make_env(self.cfg["run"])
            self.problem.warmup(env)
            self.log(f"kernels compiled / loaded in {time.time() - t0:.0f} s")
        except Exception as ex:
            self.log(f"warm-up skipped: {ex!r}")

    def record_environment(self):
        """Append the machine and the software versions to 05_logs/environment.jsonl (one line per start of
        `run`; a campaign split across devices or resumed elsewhere lists every machine)."""
        env = environment()
        self.log("environment: " + describe_environment(env))
        with open(os.path.join(self.out, LOGS, ENVIRONMENT), "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "campaign": self.campaign_id, **env},
                               default=str) + "\n")
        return env

    # -- execution --
    def run(self, only=None, dry=False, skip_references=False):
        t_start = time.time()
        if not dry and self.cfg["run"].get("keep_awake", True) and keep_awake(True):
            self.log("keep-awake requested for the duration of the campaign (idle sleep only; a closed lid "
                     "or a manual sleep still suspends the machine)")
        self.log(f"config {self.cfg.get('_config_path')} -> output {self.out}")
        self.log(f"campaign {self.campaign_id or '(legacy results folder)'}, context cache {self.cache_dir}")
        m = self.machine
        self.log(f"machine: {m['physical_cores']} physical / {m['logical_cores']} logical cores, "
                 + (f"{m['mem_available_gb']:.1f} of {m['mem_total_gb']:.1f} GB RAM available" if m['mem_total_gb']
                    else "RAM unknown"))
        self.sets = build_instances(self.cfg, self.problem, self.store, self.log)
        with open(os.path.join(self.out, LOGS, "config_used.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=1, default=str)
        if not dry:
            self.record_environment()
        plan = self.plan(only)
        est = self.estimate(plan)
        self.log("plan (est. hours = budgets of the methods that use their whole budget / workers; Certify "
                 "stages and single-run methods come on top):\n" + est.to_string(index=False)
                 + f"\ntotal est. hours: {est['est. hours'].sum():.1f}")
        if dry:
            return est
        self.warm_master()
        if not skip_references:
            self.run_references()
        tuned = self._load_tuned()
        for name, ex, tasks in plan:
            defaults = deep_merge(self.defaults, tuned.get("classes", {}))
            if ex.get("kind") != "tuning":
                tasks = [(a, i, _apply_tuned(sp, tuned.get("ids", {})), b, s) for a, i, sp, b, s in tasks]
            self.run_block(name, ex, tasks, defaults)
            if ex.get("kind") == "tuning":
                chosen = self.choose_tuned(name, ex)
                if ex.get("apply", True):
                    tuned.setdefault("ids", {}).update(chosen)
                    for mid, cls in (ex.get("apply_to_class") or {}).items():
                        if mid in chosen:
                            tuned.setdefault("classes", {}).setdefault(cls, {}).update(chosen[mid])
                    self._save_tuned(tuned)
        self.aggregate()
        self.log(f"campaign finished in {(time.time() - t_start) / 60:.1f} min")
        return est

    def run_block(self, name, ex, tasks, defaults):
        ctx_cfg = deep_merge(self.cfg.get("context", {}), ex.get("context", {}))
        by_inst = {}
        done = self.completed(name) if self.cfg["run"].get("resume", True) else set()
        for set_name, inst, spec, b, s in tasks:
            if run_key(inst.name, spec["id"], b, s) in done:
                continue
            by_inst.setdefault(inst.name, (inst, []))[1].append((spec, b, s))
        if not by_inst:
            self.log(f"{name}: nothing to do ({len(tasks)} runs already stored)")
            return
        self.check_disk(name)
        self.check_solver(name)
        n_tasks = sum(len(v[1]) for v in by_inst.values())
        self.log(f"{name}: {n_tasks} runs on {len(by_inst)} instances ({ex.get('description', '')})")
        jobs = []
        for inst, tl in by_inst.values():
            # long budgets first, so that the pool ends together
            tl.sort(key=lambda t: -(t[1] or 0))
            chunk = int(ex.get("tasks_per_job", 0)) or len(tl)
            for k in range(0, len(tl), chunk):
                jobs.append({"exp": name, "inst": self.problem.instance_to_dict(inst), "tasks": tl[k:k + chunk],
                             "ctx_cfg": ctx_cfg, "cache_dir": self.cache_dir, "out": self.out,
                             "keep_solution": bool(ex.get("store_solutions", False)),
                             "stop_at_lb": bool(ex.get("stop_at_lb", False)),
                             "flush": float(self.cfg["run"].get("flush_seconds", 30))})
        workers, why = self.block_workers(ex, [t[0] for j in jobs for t in j["tasks"]], fresh=True)
        jobs = self._split_jobs(jobs, workers, ex)
        jobs.sort(key=lambda j: -sum((t[1] or 0) for t in j["tasks"]))
        self.log(f"{name}: {len(jobs)} jobs on {workers} workers ({why})")
        t0 = time.time()
        errors, failed = self._run_jobs(name, jobs, workers, defaults)
        for attempt in range(int(self.cfg["run"].get("block_retries", 2))):
            if not errors and not failed:
                break
            # a worker died (e.g. out of memory, or no licence at start-up) or runs failed (e.g. the licence
            # was lost during a solve): once the solver is available, run the missing and failed tasks again,
            # with fewer workers if a worker died while the licence was there
            waited = self.check_solver(name)
            self._done.pop(name, None)
            done = self.completed(name)
            left = [dict(j, tasks=[t for t in j["tasks"] if run_key(self.problem.instance_from_dict(j["inst"]).name,
                                                                     t[0]["id"], t[1], t[2]) not in done])
                    for j in jobs]
            left = [j for j in left if j["tasks"]]
            if not left:
                break
            if errors and not waited:
                workers = max(1, workers // 2)
            self.log(f"{name}: {errors} worker error(s), {failed} failed run(s); retrying "
                     f"{sum(len(j['tasks']) for j in left)} missing runs with {workers} workers (attempt {attempt + 1})")
            jobs = left
            errors, failed = self._run_jobs(name, left, workers, defaults)
        self._done.pop(name, None)
        self.log(f"{name}: done in {(time.time() - t0) / 60:.1f} min")

    def _run_jobs(self, name, jobs, workers, defaults):
        """Run jobs in a fresh pool; returns (jobs that ended with a worker error, runs with status failed)."""
        k, errors, failed = 0, 0, 0
        with self._pool(defaults, workers) as pool:
            futs = [pool.submit(_work, j) for j in jobs]
            for f in as_completed(futs):
                k += 1
                if free_gb(self.out) < self.min_free_gb:
                    for g in futs:
                        g.cancel()
                    self.log(f"{name}: free disk space below {self.min_free_gb} GB - pending jobs cancelled; the runs "
                             "already stored are kept. Free some space and run the same command again to resume.")
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise LowDiskSpace(f"free disk space below {self.min_free_gb} GB")
                try:
                    r = f.result()
                    bad = [d for d in r["done"] if d[3] not in ("ok", "not_available", "timeout")]
                    msg = f"{name} [{k}/{len(jobs)}] {r['instance']}: {len(r['done'])} runs"
                    if bad:
                        msg += f"  ({len(bad)} with status {sorted(set(d[3] for d in bad))})"
                    failed += sum(1 for d in bad if d[3] == "failed")
                    self.log(msg)
                except Exception as ex2:
                    errors += 1
                    self.log(f"{name} [{k}/{len(jobs)}] worker error: {ex2!r}")
        return errors, failed

    def _split_jobs(self, jobs, workers, ex):
        """Load balance: with fewer than split_factor x workers jobs (e.g. E3, a few instances with four 600 s
        runs each), split the jobs of instances whose context is already cached, longest first. Their tasks
        then run in parallel without recomputing the context; jobs with an uncached context stay whole, so a
        context is never computed twice at the same time."""
        target = int(self.cfg["run"].get("split_factor", 3)) * workers
        if workers <= 1 or len(jobs) >= target or not self.cfg["run"].get("split_jobs", True)                 or ex.get("tasks_per_job"):
            return jobs
        out, extra = [], target - len(jobs)
        for j in sorted(jobs, key=lambda j: -sum((t[1] or 1) for t in j["tasks"])):
            inst = self.problem.instance_from_dict(j["inst"])
            cached = os.path.exists(os.path.join(self.cache_dir, f"{_safe(inst.name, 80)}__{_hash(j['ctx_cfg'])}.pkl"))
            parts = min(len(j["tasks"]), 1 + extra) if cached else 1
            if parts <= 1:
                out.append(j)
                continue
            extra -= parts - 1
            for q in range(parts):
                jj = dict(j); jj["tasks"] = j["tasks"][q::parts]
                out.append(jj)
        return out

    @property
    def min_free_gb(self):
        return float(self.cfg["run"].get("min_free_gb", 3.0))

    def check_solver(self, what):
        """Every worker opens its own solver environment; with a Gurobi WLS licence this needs the network.
        Check that an environment starts before a block (or a retry); while it does not, wait and try again
        every run.licence_poll_s seconds, and stop cleanly after run.licence_wait_min minutes (the runs done
        so far are stored; the same command resumes). Returns True if it had to wait."""
        make = getattr(self.problem, "make_env", None)
        if make is None:
            return False
        wait_min = float(self.cfg["run"].get("licence_wait_min", 30))
        poll = float(self.cfg["run"].get("licence_poll_s", 60))
        t0, waited = time.time(), False
        while True:
            try:
                env = make(self.cfg["run"])
                if env is not None and hasattr(env, "dispose"):
                    env.dispose()
                if waited:
                    self.log(f"{what}: solver licence available again after {(time.time() - t0) / 60:.1f} min")
                return waited
            except Exception as ex:
                if time.time() - t0 >= 60 * wait_min:
                    self.log(f"{what}: solver licence still unavailable after {wait_min:g} min ({ex!r}): stopping. "
                             "Everything finished so far is stored; run the same command again to resume.")
                    raise SolverUnavailable(repr(ex))
                if not waited:
                    self.log(f"{what}: solver licence unavailable ({ex!r}) - network cut? Waiting up to "
                             f"{wait_min:g} min, checking every {poll:g} s")
                    waited = True
                time.sleep(poll)

    def check_disk(self, what):
        f = free_gb(self.out)
        if f < self.min_free_gb:
            self.log(f"{what}: only {f:.1f} GB free on the results disk (minimum {self.min_free_gb} GB): stopping before "
                     "it starts. Free some space (Windows also grows its page file under memory pressure) and resume.")
            raise LowDiskSpace(f"free disk space {f:.1f} GB < {self.min_free_gb} GB")

    def run_references(self):
        rc = self.cfg.get("references", {})
        if not rc.get("sets"):
            return
        ctx_cfg = self.cfg.get("context", {})
        jobs = []
        for set_name in rc["sets"]:
            for inst in self.sets[set_name]:
                if self.store.exists("_references", _safe(inst.name, 80)):
                    continue
                jobs.append({"inst": self.problem.instance_to_dict(inst), "set": set_name, "ctx_cfg": ctx_cfg,
                             "cache_dir": self.cache_dir, "out": self.out, "ref_cfg": rc})
        if not jobs:
            self.log("references: all stored")
            return
        self.check_disk("references")
        self.check_solver("references")
        self.log(f"references: {len(jobs)} exact solves (time limit {rc.get('time_limit', 600)} s)")
        workers, why = plan_workers(self.cfg["run"], rc, int(rc.get("threads", 1)), machine())
        self.log(f"references: {workers} workers ({why})")
        with self._pool(self.defaults, workers) as pool:
            for k, f in enumerate(as_completed([pool.submit(_ref_work, j) for j in jobs]), 1):
                try:
                    r = f.result()
                    self.log(f"references [{k}/{len(jobs)}] {r['instance']}: {r['status']} best {r['best']} "
                             f"bound {r['bound']:.2f} ({r['time']:.1f} s)")
                except Exception as ex:
                    self.log(f"references [{k}/{len(jobs)}] error {ex!r}")

    # -- tuning --
    def _tuned_path(self):
        return os.path.join(self.store.data_root, "tuned_defaults.json")

    def _load_tuned(self):
        p = self._tuned_path()
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_tuned(self, tuned):
        os.makedirs(os.path.dirname(self._tuned_path()), exist_ok=True)
        with open(self._tuned_path(), "w", encoding="utf-8") as f:
            json.dump(tuned, f, indent=1)

    def choose_tuned(self, name, ex):
        """Best grid point per tuned method: lowest mean relative gap to the best value found on each
        tuning instance (ties: lowest mean time). Returns {class: params}."""
        runs = pd.DataFrame([r.to_row() for r in self.store.load_all(name)])
        if runs.empty or "info_tuning_point" not in runs:
            return {}
        runs = runs[runs.status.isin(["ok", "timeout"])]
        best = runs.groupby("instance").objective.min()
        runs["rel"] = (runs.objective - runs.instance.map(best)) / runs.instance.map(best)
        out, rows = {}, []
        runs["base_id"] = runs.method.str.replace(r"\[.*\]$", "", regex=True)
        for cls, g in runs[runs.info_tuning_point.notna()].groupby("base_id"):
            sc = g.groupby("info_tuning_point").agg(rel=("rel", "mean"), time=("time", "mean")).reset_index()
            sc = sc.sort_values(["rel", "time"])
            point = json.loads(sc.iloc[0].info_tuning_point)
            out[cls] = point
            for _, r in sc.iterrows():
                rows.append({"method": cls, "point": r.info_tuning_point, "mean_rel_gap": r.rel, "mean_time": r.time,
                             "chosen": r.info_tuning_point == sc.iloc[0].info_tuning_point})
        self.store.save_table(f"tuning_{name}", pd.DataFrame(rows))
        self.log(f"{name}: chosen parameters {out}")
        return out

    # -- aggregation --
    def rebuild_data(self):
        """Rebuild every derived table of results/data from the raw job files (results/raw) and the
        configuration alone: instances and their features, tuning tables, runs, traces, bound traces,
        solutions, references and context summaries. The report command calls it first, so a report can
        be regenerated from the pkl files of results/raw (for example merged from several machines).
        tuned_defaults.json records what was applied to the later blocks and is kept; a recomputed
        choice that differs from it is reported."""
        self.sets = build_instances(self.cfg, self.problem, self.store, self.log)
        applied = self._load_tuned().get("ids", {})
        for name, ex in self.cfg["experiments"].items():
            if ex.get("kind") == "tuning" and os.path.isdir(os.path.join(self.store.raw_root, name)):
                chosen = self.choose_tuned(name, ex)
                for mid, point in chosen.items():
                    if mid in applied and applied[mid] != point:
                        self.log(f"note: {name} now selects {point} for {mid}; the later blocks used {applied[mid]}")
        return self.aggregate()

    def aggregate(self):
        rows, traces, bounds, sols = [], {}, {}, {}
        for ex in self.store.experiments():
            if ex.startswith("_"):
                continue
            latest = {}
            for r in self.store.load_all(ex):           # a re-run replaces a failed / suspect run
                k = run_key(r.instance, r.method, r.budget, r.seed)
                if k not in latest or latest[k].status in ("failed", "suspect_timing"):
                    latest[k] = r
            for r in latest.values():
                rows.append(r.to_row())
                key = (r.experiment, r.instance, r.method, r.budget, r.seed)
                traces[key] = r.trace
                if r.info and r.info.get("bound_trace"):
                    bounds[key] = r.info["bound_trace"]
                if r.solution is not None:
                    sols[key] = np.asarray(r.solution, np.int8)
        from ..analysis.metrics import charge_wall_time
        runs = charge_wall_time(pd.DataFrame(rows))
        refs = pd.DataFrame(self.store.load_all("_references")) if os.path.isdir(
            os.path.join(self.store.raw_root, "_references")) else pd.DataFrame()
        ctx_rows = list(self.store.load_all("_contexts")) if os.path.isdir(
            os.path.join(self.store.raw_root, "_contexts")) else []
        have = {d["instance"] for d in ctx_rows}
        for p in (os.listdir(self.cache_dir) if os.path.isdir(self.cache_dir) else []):
            if not p.endswith(".pkl"):
                continue
            if p.rsplit("__", 1)[0] in {_safe(i, 80) for i in have}:
                continue                                 # summary already stored with the results
            try:
                with open(os.path.join(self.cache_dir, p), "rb") as f:
                    ctx = pickle.load(f)
            except Exception:
                continue
            ctx_rows.append(context_summary(ctx))
        contexts = pd.DataFrame(ctx_rows)
        self.store.save_table("runs", runs)
        self.store.save_table("traces", traces)
        self.store.save_table("bound_traces", bounds)
        self.store.save_table("solutions", sols)
        self.store.save_table("references", refs)
        self.store.save_table("contexts", contexts)
        self.log(f"aggregated {len(runs)} runs, {len(refs)} references, {len(contexts)} contexts "
                 f"into {self.store.data_root}")
        return runs


def forget_runs(cfg, blocks, methods=None):
    """Remove the stored runs of the given blocks (and methods) from their job files (atomic rewrite)."""
    import glob
    out = resolve_path(cfg, cfg["run"].get("output", "results"))
    n = 0
    for blk in blocks:
        for p in glob.glob(os.path.join(ResultStore(out).raw_root, blk, "*.pkl")):
            with open(p, "rb") as f:
                runs = pickle.load(f)
            runs = runs if isinstance(runs, list) else [runs]
            keep = [r for r in runs if methods is not None and r.method not in methods]
            n += len(runs) - len(keep)
            if keep:
                with open(p + ".tmp", "wb") as f:
                    pickle.dump(keep, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(p + ".tmp", p)
            else:
                os.remove(p)
    return n


def _apply_tuned(spec, tuned_ids):
    key = spec.get("tuned_as", spec["id"])
    if key not in tuned_ids:
        return spec
    s = copy.deepcopy(spec)
    s.setdefault("params", {}).update(tuned_ids[key])
    return s


def _import_problem(name):
    import importlib
    importlib.import_module(f"cri.problems.{name}")


def run_campaign(cfg, only=None, dry=False, skip_references=False):
    r = Runner(cfg)
    return r.run(only=only, dry=dry, skip_references=skip_references)


if __name__ == "__main__":
    print("use: python -m cri run <config.yaml>", file=sys.stderr)
