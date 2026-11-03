"""Problem-agnostic run record and result store (one pickle per run, aggregated tables as pickles)."""

import glob
import os
import pickle
import time
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class RunResult:
    experiment: str
    instance: str
    method: str
    budget: float
    seed: int
    objective: float = np.nan          # best objective found (peak power, kW)
    time: float = np.nan               # charged time: own work + artifacts used (s)
    own_time: float = np.nan
    artifact_time: float = 0.0
    trace: list = field(default_factory=list)     # [(t, best objective)] anytime trace
    guarantee: float = np.nan          # a priori guarantee on the objective, when the method has one
    lower_bound: float = np.nan        # certified lower bound known to the method (if any)
    status: str = "ok"                 # ok | not_available | failed | timeout
    feasible: bool = True
    info: dict = field(default_factory=dict)
    solution: object = None            # stored only if requested (memory)

    def to_row(self):
        d = asdict(self)
        d.pop("solution", None)
        d.pop("trace", None)
        info = d.pop("info", {}) or {}
        for k, v in info.items():
            if np.isscalar(v) or v is None:
                d["info_" + k] = v
        return d


class Deadline:
    """Wall-clock budget, with an initial charge (artifacts already paid for)."""

    def __init__(self, budget, already=0.0):
        self.t0 = time.time() - already
        self.budget = np.inf if budget is None else float(budget)

    def elapsed(self):
        return time.time() - self.t0

    def remaining(self):
        return self.budget - self.elapsed()

    def expired(self):
        return self.remaining() <= 0


# Layout of a results folder (names say what each folder holds; README.md in it explains them)
RAW = "01_raw_runs"                    # one pickle per (block, instance) job: the source of truth
DATA = "02_aggregated_data"            # tables rebuilt from 01 by `python -m cri report`
REPORT = "03_report"                   # report.md / report.html and one folder per section and block
FAITHFULNESS = "04_faithfulness_check"  # original code against the re-implementation (faithfulness/check.py)
LOGS = "05_logs"                       # run log, configuration used, step markers of run_all
ENVIRONMENT = "environment.jsonl"      # in 05_logs: machine and software versions of every start of `run`
LEGACY = {RAW: "raw", DATA: "data", REPORT: "report", FAITHFULNESS: "faithfulness"}   # names of the older layout


def results_dir(root, kind):
    """Folder `kind` of a results folder; the names of the older layout are still read when the new
    folder does not exist yet (to resume older results)."""
    new = os.path.join(root, kind)
    old = os.path.join(root, LEGACY.get(kind, kind))
    return old if (kind in LEGACY and not os.path.exists(new) and os.path.exists(old)) else new


class ResultStore:
    """01_raw_runs/<experiment>/<job>.pkl hold the list of runs of one job (one instance); a job file is
    rewritten while the job advances, so an interrupted campaign keeps its finished runs.
    02_aggregated_data/<name>.pkl hold the aggregated tables."""

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.raw_root = results_dir(root, RAW)
        self.data_root = results_dir(root, DATA)

    def _dir(self, experiment):
        d = os.path.join(self.raw_root, experiment)
        os.makedirs(d, exist_ok=True)
        return d

    def path(self, experiment, key):
        return os.path.join(self._dir(experiment), key + ".pkl")

    def exists(self, experiment, key):
        return os.path.exists(self.path(experiment, key))

    def save(self, experiment, key, obj):
        p = self.path(experiment, key)
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)

    def load(self, experiment, key):
        with open(self.path(experiment, key), "rb") as f:
            return pickle.load(f)

    def load_all(self, experiment):
        """Every stored object of an experiment; job files (lists of runs) are flattened."""
        out = []
        d = os.path.join(self.raw_root, experiment)
        for p in sorted(glob.glob(os.path.join(d, "*.pkl"))):
            try:
                with open(p, "rb") as f:
                    obj = pickle.load(f)
            except Exception:
                continue
            if isinstance(obj, list):
                out.extend(obj)
            else:
                out.append(obj)
        return out

    def experiments(self):
        d = self.raw_root
        return sorted(e for e in os.listdir(d) if os.path.isdir(os.path.join(d, e))) if os.path.isdir(d) else []

    def save_table(self, name, obj):
        d = self.data_root
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, name + ".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, os.path.join(d, name + ".pkl"))

    def load_table(self, name):
        p = os.path.join(self.data_root, name + ".pkl")
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return pickle.load(f)
