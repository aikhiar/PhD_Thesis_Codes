"""
Machine resources and the number of parallel workers (works on laptops, servers and virtual machines).

  cores      CPUs usable by this process: the affinity mask (Linux), capped by a container CPU quota
             (cgroup v1/v2); 'physical' counts one per physical core when the OS reports it (on most
             virtual machines the reported cores are the vCPUs)
  memory     available and total RAM (psutil)
  environment  host, processor model, cores, RAM, OS and software versions of the machine that runs a
             campaign (recorded in 05_logs/environment.jsonl at every start of `run`)

Worker rule for a block (run.workers: auto, or an integer):
  by_cpu = (cores - reserve_cores) // threads_per_task     threads_per_task = largest Gurobi thread count
                                                           of the block's methods (1 for everything else)
  by_mem = (available RAM - reserve_mem_gb) // mem_gb_per_worker
  workers = max(1, min(by_cpu, by_mem, max_workers))
One worker per physical core by default: hyperthreads share a core, and wall-clock budgets must be
comparable between runs.
"""

import math
import os


def _cgroup_cpus():
    """CPU quota of a container (cgroup v2 cpu.max or v1 cfs quota), or None."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            q, p = f.read().split()[:2]
            if q != "max":
                return max(1, int(math.floor(int(q) / int(p))))
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            q = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            p = int(f.read())
        if q > 0 and p > 0:
            return max(1, q // p)
    except (OSError, ValueError):
        pass
    return None


def machine():
    logical = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            logical = len(os.sched_getaffinity(0))
        except OSError:
            pass
    physical = None
    total = avail = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        vm = psutil.virtual_memory()
        total, avail = vm.total / 2 ** 30, vm.available / 2 ** 30
    except Exception:
        pass
    if not physical:
        physical = max(1, logical // 2) if logical > 1 else 1
    physical = min(physical, logical)
    quota = _cgroup_cpus()
    if quota:
        logical, physical = min(logical, quota), min(physical, quota)
    return {"logical_cores": logical, "physical_cores": physical, "mem_total_gb": total, "mem_available_gb": avail}


def cpu_name():
    """Model name of the processor (its brand string; platform.processor() gives only the family on
    Windows)."""
    import platform
    try:
        if os.name == "nt":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if platform.system() == "Darwin":
            import subprocess
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                 timeout=5).stdout.strip()
            if out:
                return out
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"


def environment():
    """The machine and the software versions a campaign runs with: host, processor, cores, RAM, OS, Python,
    numpy, scipy, numba, pandas and Gurobi (None when a package is missing)."""
    import platform
    env = {"host": platform.node(), "os": f"{platform.system()} {platform.release()}", "cpu": cpu_name()}
    env.update(machine())
    env["python"] = platform.python_version()
    for mod in ("numpy", "scipy", "numba", "pandas"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:
        import gurobipy
        env["gurobi"] = ".".join(map(str, gurobipy.gurobi.version()))
    except Exception:
        env["gurobi"] = None
    return env


def describe_environment(env):
    """One line for the run log and the report: host, processor, cores, RAM, OS and versions."""
    mem = f", {env['mem_total_gb']:.0f} GB RAM" if env.get("mem_total_gb") else ""
    sw = ", ".join(f"{k} {env[k]}" for k in ("python", "numpy", "scipy", "numba", "pandas", "gurobi") if env.get(k))
    return (f"{env.get('host')}: {env.get('cpu')}, {env.get('physical_cores')} physical / "
            f"{env.get('logical_cores')} logical cores{mem}, {env.get('os')}; {sw}")


_POOL_MAX = 61 if os.name == "nt" else 10 ** 6      # ProcessPoolExecutor accepts at most 61 workers on Windows


def plan_workers(run_cfg, block_cfg=None, threads_per_task=1, m=None):
    """Number of worker processes for a block, and the reason (for the log)."""
    block_cfg = block_cfg or {}
    m = m or machine()
    w = block_cfg.get("workers", run_cfg.get("workers", "auto"))
    if w not in (None, "auto"):
        return min(max(1, int(w)), _POOL_MAX), f"fixed ({w})" + ("" if int(w) <= _POOL_MAX else f", Windows cap {_POOL_MAX}")
    cores = m["physical_cores"] if run_cfg.get("count_physical_cores", True) else m["logical_cores"]
    reserve = int(run_cfg.get("reserve_cores", 1 if cores > 2 else 0))
    by_cpu = max(1, (cores - reserve) // max(1, int(threads_per_task)))
    mem_per = float(block_cfg.get("mem_gb_per_worker", run_cfg.get("mem_gb_per_worker", 1.0)))
    if m["mem_available_gb"] is not None:
        by_mem = max(1, int((m["mem_available_gb"] - float(run_cfg.get("reserve_mem_gb", 1.0))) // mem_per))
    else:
        by_mem = by_cpu
    cap = run_cfg.get("max_workers")
    n = min(by_cpu, by_mem, int(cap) if cap else by_cpu, _POOL_MAX)
    why = (f"auto: {cores} {'physical' if run_cfg.get('count_physical_cores', True) else 'logical'} cores - {reserve} "
           f"reserved, {threads_per_task} thread(s)/task -> {by_cpu}; "
           + (f"{m['mem_available_gb']:.1f} GB available, {mem_per:g} GB/worker -> {by_mem}" if m["mem_available_gb"]
              is not None else "memory unknown") + (f"; cap {cap}" if cap else ""))
    return max(1, n), why


def single_thread_env():
    """Pin BLAS / OpenMP thread pools to one thread in the worker processes (no oversubscription: the
    parallelism is across processes). Set before the pool is created, inherited by the workers."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(k, "1")
