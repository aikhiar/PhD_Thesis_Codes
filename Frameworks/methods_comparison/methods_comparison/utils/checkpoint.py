"""
checkpoint.py  —  incremental result saving with stale-cache protection.

Every cached run is keyed by a tag that embeds a short hash of the *parameters*
and the *instance content*.  Changing the param grid, the tuned parameters, or
the instances therefore yields a different tag, so a stale result can never be
silently reused.  Writes are atomic (temp file + os.replace) so an interrupt
cannot leave a half-written checkpoint.

Layout inside <main_folder>/checkpoints/:
    {method}_{phase}_inst{id}_run{r}_{sig}.pkl     phase ∈ {tune, eval, evalc}
    exact_inst{id}_{instance sig}.pkl

`evalc` (vs plain `eval`) is the evaluation phase's tag when
config["convergence"]["enabled"] is True: those runs also record
per-generation progress, so they occupy a separate cache namespace from
plain `eval` runs -- a progress-less checkpoint can never be reused as a
convergence source.
"""

import os
import glob
import json
import pickle
import hashlib
from .io import load_pickle


def instance_signature(instance) -> str:
    """Short stable hash of an instance's content."""
    try:
        blob = pickle.dumps(instance)
    except Exception:
        blob = repr(instance).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


def params_signature(params: dict, inst_sig: str) -> str:
    blob = json.dumps([params, inst_sig], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def run_tag(method: str, phase: str, inst_id: int, run: int, sig: str) -> str:
    return f"{method}_{phase}_inst{inst_id}_run{run}_{sig}"


def exact_tag(inst_id: int, inst_sig: str) -> str:
    """Keyed by the instance content, like every other checkpoint."""
    return f"exact_inst{inst_id}_{inst_sig}"


def _path(checkpoint_dir: str, tag: str) -> str:
    return os.path.join(checkpoint_dir, f"{tag}.pkl")


def save_run(checkpoint_dir: str, tag: str, result: dict) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    final = _path(checkpoint_dir, tag)
    tmp = final + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(result, fh)
    os.replace(tmp, final)              # atomic on the same filesystem


def load_run(checkpoint_dir: str, tag: str):
    p = _path(checkpoint_dir, tag)
    if os.path.exists(p):
        try:
            return load_pickle(p)
        except Exception:
            return None
    return None


def count_checkpoints(checkpoint_dir: str) -> int:
    if not os.path.isdir(checkpoint_dir):
        return 0
    return len(glob.glob(os.path.join(checkpoint_dir, "*.pkl")))
