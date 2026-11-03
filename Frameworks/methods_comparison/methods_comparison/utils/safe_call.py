"""
safe_call.py  —  reproducible, crash-isolated, optionally time-limited solver call.

Design choice: SIMPLE, single-process execution (no ProcessPool, no threads).

Seeding
-------
Before every call we seed Python `random`, NumPy and (if installed) Numba via
``core.seeding.seed_everything`` so the run is reproducible.

Timeout (best effort, POSIX)
----------------------------
When a timeout is set and we are on the main thread of a POSIX system, a
``signal.SIGALRM`` timer is armed.  The alarm raises between Python bytecode
boundaries, which for a typical metaheuristic (a Python `for generation in ...`
loop calling compiled kernels) means it interrupts at generation boundaries.
A single uninterruptible C/compiled call longer than the timeout cannot be
pre-empted this way — that requires process isolation, which is intentionally
out of scope for the simple runner.  On Windows / non-main-thread the timeout
is skipped (a warning-free no-op) and the call runs to completion.

Timeout (process mode)
----------------------
With mode="process" (config['parallel']['timeout_mode']) the call runs in a
reusable loky worker process, killed when the timeout expires: enforced on
every platform and for compiled code. Seeding and timing happen inside the
worker, so results equal the in-process ones.

Every returned dict carries: exec_time, _crashed, _timeout, _error.
"""

import signal
import threading
import traceback
from timeit import default_timer as timer

from ..core.seeding import seed_everything


class _Timeout(Exception):
    pass


def _alarm_available() -> bool:
    return (hasattr(signal, "SIGALRM")
            and threading.current_thread() is threading.main_thread())


def safe_call(method_fn, instance, params, progress_record,
              seed: int = None, timeout: float = None, mode: str = "alarm") -> dict:
    """mode="alarm" (default): in-process, SIGALRM timeout where available.
    mode="process": the call runs in a reusable worker process (loky) that is
    killed when `timeout` expires -- enforced on every platform, including
    Windows and long compiled kernels. Used only when a timeout is set."""
    if mode == "process" and timeout is not None:
        res = _call_in_process(method_fn, instance, params, progress_record, seed, timeout)
    else:
        res = _call_here(method_fn, instance, params, progress_record, seed, timeout)
    res["_seed"] = seed
    return res


# ── process-isolated call (config['parallel']['timeout_mode'] = "process") ───
_STARTUP_LIMIT = 300.0       # s allowed for a fresh worker to start (imports, JIT)


def _child_call(method_fn, instance, params, progress_record, seed):
    """Runs inside the worker process: seeding + timing as in-process."""
    return _call_here(method_fn, instance, params, progress_record, seed, None)


def _noop():
    return True


def _call_in_process(method_fn, instance, params, progress_record, seed, timeout):
    from concurrent.futures import TimeoutError as _FutTimeout
    from joblib.externals.loky import get_reusable_executor
    t0 = timer()
    ex = get_reusable_executor(max_workers=1, timeout=600)
    try:
        # make sure the worker is up, so its start-up time is not charged to
        # the method's time limit (nor to the recorded exec_time)
        ex.submit(_noop).result(timeout=_STARTUP_LIMIT)
        t0 = timer()
        fut = ex.submit(_child_call, method_fn, instance, params, progress_record, seed)
        return fut.result(timeout=float(timeout))
    except _FutTimeout:
        ex.shutdown(wait=False, kill_workers=True)     # kill the runaway call
        return {"exec_time": timer() - t0, "_crashed": False,
                "_timeout": True, "_error": f"Timeout after {timeout}s (process killed)"}
    except Exception:
        return {"exec_time": timer() - t0, "_crashed": True,
                "_timeout": False, "_error": traceback.format_exc()}


def _call_here(method_fn, instance, params, progress_record, seed, timeout) -> dict:
    if seed is not None:
        seed_everything(seed)

    t0 = timer()
    use_alarm = timeout is not None and _alarm_available()
    old_handler = None
    if use_alarm:
        def _handler(signum, frame):
            raise _Timeout()
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, float(timeout))

    try:
        result = method_fn(instance, params, progress_record)
        if not isinstance(result, dict):
            result = {}
        result.setdefault("exec_time", timer() - t0)
        result["_crashed"] = False
        result["_timeout"] = False
        result["_error"] = None
        return result
    except _Timeout:
        return {"exec_time": timer() - t0, "_crashed": False,
                "_timeout": True, "_error": f"Timeout after {timeout}s"}
    except Exception:
        return {"exec_time": timer() - t0, "_crashed": True,
                "_timeout": False, "_error": traceback.format_exc()}
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
