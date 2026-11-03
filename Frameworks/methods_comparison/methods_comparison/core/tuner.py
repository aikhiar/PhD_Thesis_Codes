"""
tuner.py  —  parameter tuning: grid search, one parameter set per instance.

Every combination of a method's grid is run `n_runs_per_combo` times on every
evaluation instance; on each instance the combination with the best aggregated
score is retained (mono-objective: the objective, lower is better;
multi-objective: the hypervolume, in one normalisation shared by every run of
every combination on that instance, higher is better).
"""

import numpy as np
from joblib import Parallel, delayed

from .seeding import make_seed
from ..utils.checkpoint import (save_run, load_run, run_tag, instance_signature,
                                params_signature)
from ..utils.safe_call import safe_call
from ..utils.progress import progress_iter

_PH_TUNE = 1


# ── one (instance, combo) evaluation, process-safe ───────────────────────────
def tune_one_combo(method_fn, mname, inst, inst_id, inst_idx, combo, combo_idx,
                   n_runs, seed_base, m_idx, timeout, ckpt_dir, timeout_mode="alarm"):
    """n_runs runs of one combo on one instance (checkpointed per run);
    returns (inst_idx, combo_idx, valid runs stripped to the tuning keys, combo)."""
    from .runner import _strip_for_tuning_checkpoint
    sig = params_signature(combo, instance_signature(inst))
    valid = []
    for run in range(n_runs):
        tag = run_tag(mname, "tune", inst_id, run, sig)
        res = load_run(ckpt_dir, tag)
        if res is None:
            seed = make_seed(seed_base, _PH_TUNE, m_idx, inst_idx, combo_idx, run)
            res = safe_call(method_fn, inst, combo, False, seed, timeout, timeout_mode)
            res = _strip_for_tuning_checkpoint(res)
            save_run(ckpt_dir, tag, res)
        if not res.get("_crashed") and not res.get("_timeout"):
            valid.append(_strip_for_tuning_checkpoint(res))
    return inst_idx, combo_idx, valid, combo


class Tuner:
    def __init__(self, runner, prob, ckpt):
        self.R, self.prob, self.ckpt = runner, prob, ckpt
        self.cfg = runner.cfg
        t = self.cfg["tuning"]
        self.n_runs, self.seed_base = t["n_runs_per_combo"], t["seed_base"]
        self.timeout, self.agg = t.get("timeout"), t.get("aggregation", "mean")
        from ..utils.parallel import resolve_n_jobs
        self.n_jobs = resolve_n_jobs(self.cfg["parallel"].get("n_jobs", -1))
        self.timeout_mode = self.cfg["parallel"].get("timeout_mode", "alarm")

    def _eval_pairs(self, mname, m_idx, pairs, full, instances, desc):
        """pairs: [(inst_idx, combo_idx)] -> {(inst_idx, combo_idx): valid runs}."""
        fn = self.prob.get_method(mname)
        out = {}
        if self.n_jobs in (1, None):
            it = progress_iter(pairs, total=len(pairs), desc=desc, verbose=self.R.verbose)
            for ii, ci in it:
                _, _, valid, _ = tune_one_combo(
                    fn, mname, instances[ii], ii + 1, ii, full[ci], ci, self.n_runs,
                    self.seed_base, m_idx, self.timeout, self.ckpt, self.timeout_mode)
                out[(ii, ci)] = valid
            return out
        tasks = (delayed(tune_one_combo)(fn, mname, instances[ii], ii + 1, ii, full[ci], ci,
                                         self.n_runs, self.seed_base, m_idx, self.timeout,
                                         self.ckpt, self.timeout_mode)
                 for ii, ci in pairs)
        kw = dict(n_jobs=self.n_jobs, backend="loky")
        try:
            res = Parallel(return_as="generator", **kw)(tasks)
        except TypeError:
            res = Parallel(**kw)(tasks)
        for ii, ci, valid, _ in progress_iter(res, total=len(pairs), desc=desc,
                                              verbose=self.R.verbose):
            out[(ii, ci)] = valid
        return out

    def run(self, mname, m_idx, full, instances):
        """-> (best {str(inst id): params}, trace [{instance_id, score, **combo}])."""
        from .runner import _score_combos
        prob = self.prob
        cands = list(range(len(full)))
        pairs = [(ii, ci) for ii in range(len(instances)) for ci in cands]
        runs = self._eval_pairs(mname, m_idx, pairs, full, instances,
                                f"    [Tuning:{prob.label(mname)}]")
        best, trace = {}, []
        for ii in range(len(instances)):
            cs = _score_combos({ci: (full[ci], runs.get((ii, ci), [])) for ci in cands},
                               self.agg, prob.is_multiobjective)
            sc = {ci: v[0] for ci, v in cs.items()}
            for ci in sorted(sc):
                trace.append({"instance_id": ii + 1, "score": sc[ci], **full[ci]})
            if sc:
                pick = (max if prob.is_multiobjective else min)(sc, key=lambda k: sc[k])
                best[str(ii + 1)] = full[pick]
        return best, trace
