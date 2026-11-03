"""
runner.py  —  ExperimentRunner: runs one study of the thesis.

Pipeline
--------
1. validate the binding's problem_dict and the merged configuration
2. experiment folder, DETERMINISTIC in the configuration (an interrupted run
   resumes from its checkpoints)
3. instances: reload instances.pkl if present, else generate (seeded) and save
4. exact reference (Appendix D): the exact method once per instance
5. tuning: grid search, one parameter set per instance
6. evaluation: warm-up (+1 discarded run for the compilation) then the n_runs
   kept runs; when config["convergence"]["enabled"] is True these SAME runs
   record the per-generation progress of the convergence curves
7. analysis: tables, statistical tests, figures, multi-objective indicators,
   anytime and scaling analyses where the study reports them
8. reproducibility manifest

The solving-method contract:
    method(instance, params: dict, progress_record: bool) -> dict
Reproducibility: NumPy, random and Numba are seeded before every call (see
utils.safe_call and core.seeding).
"""

import os
import copy
import time
import numpy as np
from joblib import Parallel, delayed

from ..config import DEFAULT_CONFIG
from ..solvers.bridge import attach_operator_methods
from .problem import ProblemDescriptor
from .seeding import make_seed, seed_everything

from ..utils.validate_config import validate_config, validate_problem_dict
from ..utils.io import (make_experiment_folder, make_analysis_tree, save_pickle,
                        load_pickle, save_json, slug, coded_path, BLOCK_OBJ_MULTI,
                        BLOCK_OBJ_MONO, BLOCK_GAP, BLOCK_TIME)
from ..utils.checkpoint import (save_run, load_run, run_tag, exact_tag,
                                instance_signature, params_signature,
                                count_checkpoints)
from ..utils.safe_call import safe_call
from ..utils.hypervolume import compute_hypervolume, compute_hypervolume_batch
from ..utils.provenance import write_manifest
from ..utils.progress import progress_iter
from ..utils.parallel import resolve_n_jobs

from ..analysis.tables import extract_scores
from ..analysis.statistics import run_all_statistical_tests
from ..analysis.indicators import compute_indicators

from ..figures.style import apply_thesis_style
from ..figures import plots
from .tuner import Tuner


# phase ids keep seed streams independent even if seed_base values collide
_PH_INST, _PH_EVAL, _PH_WARM, _PH_EXACT = 0, 2, 7, 9


def _evaluate_one_instance(
    method_fn, mname, inst, inst_id, params, n_runs, seed_base,
    m_idx, inst_idx, warmup, timeout, record_progress, phase, sig, ckpt_dir,
    save_solutions, timeout_mode="alarm",
):
    """Process-safe worker of the parallel evaluation (see
    `ExperimentRunner._run_evaluation`). `method_fn` is shipped by joblib's
    `loky` backend with cloudpickle, by reference: the worker re-imports the
    binding's module, so the numba-compiled operators are compiled there. Each
    worker pays its own compilation (numba's cache is per process)."""

    tags = [run_tag(mname, phase, inst_id, r, sig) for r in range(n_runs)]
    cached = [load_run(ckpt_dir, t) for t in tags]

    if warmup and any(c is None for c in cached):
        wseed = make_seed(seed_base, _PH_WARM, m_idx, inst_idx, 999999)
        safe_call(method_fn, inst, params, False, wseed, timeout, timeout_mode)

    runs, crashes = [], 0
    for r in range(n_runs):
        res = cached[r]
        if res is None:
            seed = make_seed(seed_base, _PH_EVAL, m_idx, inst_idx, r)
            res = safe_call(method_fn, inst, params, record_progress, seed, timeout,
                            timeout_mode)
            res = _strip_solutions_from_eval_result(res, save_solutions)
            save_run(ckpt_dir, tags[r], res)
        else:
            res = _strip_solutions_from_eval_result(res, save_solutions)
        if res.get("_crashed") or res.get("_timeout"):
            crashes += 1
        runs.append(res)
    return inst_id, runs, crashes


class ExperimentRunner:
    def __init__(self, problem_dict: dict, output_dir: str = "experiments",
                 config: dict = None, verbose: bool = True):
        self.cfg = copy.deepcopy(DEFAULT_CONFIG)
        if config:
            _deep_update(self.cfg, config)
        validate_config(self.cfg)
        # a binding that provides "operators" runs the solvers of cfg["solvers"]
        # built from them (methods_comparison/solvers/bridge.py)
        problem_dict = attach_operator_methods(problem_dict, self.cfg)
        validate_problem_dict(problem_dict)
        self.problem = ProblemDescriptor(problem_dict)
        # "_code" is internal: every saved file and folder is prefixed with the study's code
        self.cfg["_code"] = self.problem.code
        self.output_dir = output_dir
        self.verbose = verbose

    # ── results pkls ─────────────────────────────────────────────────────────
    @staticmethod
    def _scan_eval_pkls(eval_dir: str):
        """Every '*_results.pkl' of `eval_dir`, loaded, as (filename, saved dict) pairs."""
        pkl_paths = sorted(
            p for p in os.listdir(eval_dir) if p.endswith("_results.pkl")
        ) if os.path.isdir(eval_dir) else []
        return [(fname, load_pickle(os.path.join(eval_dir, fname))) for fname in pkl_paths]

    @staticmethod
    def _results_dict_from_loaded(loaded, code: str):
        """{method: results} from `_scan_eval_pkls`, the 'CODE_' prefix of the file
        names (see `utils.io.coded_path`) stripped back off."""
        prefix = f"{code}_"
        results = {}
        for fname, saved in loaded:
            mname = fname[: -len("_results.pkl")]
            if mname.startswith(prefix):
                mname = mname[len(prefix):]
            results[mname] = saved["results"]
        return results

    # ── public ────────────────────────────────────────────────────────────────
    def run(self):
        apply_thesis_style()
        prob = self.problem
        self._log(f"\n{'='*60}\n  {prob.code}  —  {prob.name}\n"
                  f"  {'multi' if prob.is_multiobjective else 'mono'}-objective, "
                  f"{len(prob.method_names)} method(s)\n{'='*60}")

        self._print_setup_summary(prob)
        self._run_self_check(prob)

        main = make_experiment_folder(self.output_dir, prob.name, self.cfg, code=prob.code)
        ckpt = os.path.join(main, "checkpoints"); os.makedirs(ckpt, exist_ok=True)
        self.main_folder, self.ckpt_dir = main, ckpt
        self._log(f"  Folder: {main}")
        if count_checkpoints(ckpt):
            self._log(f"  Resuming — {count_checkpoints(ckpt)} cached run(s) found.")

        instances = self._get_instances(prob, main)
        self._describe_instances(prob, instances)
        exact_refs = self._run_exact(instances, prob, ckpt)
        prob.exact_reference_values = exact_refs or None
        best_params = self._run_tuning(instances, prob, ckpt)
        eval_dir = self._run_evaluation(instances, prob, best_params, ckpt, main)
        # reloaded from the pkls `_run_evaluation` just wrote: the per-generation
        # fronts of every method are never all in memory during the evaluation
        loaded = self._scan_eval_pkls(eval_dir)
        eval_results = self._results_dict_from_loaded(loaded, prob.code)
        conv_results = eval_results if self.cfg["convergence"].get("enabled", True) else {}
        self._run_analysis(prob, eval_results, conv_results, exact_refs, main)

        write_manifest(main, self.cfg, prob, instances, exact_refs)
        self._log(f"\n  Done. Results in: {main}\n")
        return main

    # ── setup summary & self-check ──────────────────────────────────────────────
    def _print_setup_summary(self, prob):
        """The resolved protocol and every method's tuning grid, printed before
        any computation."""
        if not self.verbose:
            return
        c = self.cfg
        st = c["statistics"]
        tests = [name for name in ("friedman", "nemenyi", "wilcoxon", "cliff_delta",
                                   "mann_whitney", "holm", "bayesian") if st.get(name)]
        L = ["",
             "  --- Setup ---",
             f"  Study              : {prob.code} ({prob.name})",
             f"  Type               : "
             f"{'multi-objective' if prob.is_multiobjective else 'mono-objective'} "
             f"({prob.n_objectives} objective(s): {', '.join(prob.objective_names)})",
             f"  Methods            : {', '.join(prob.label(m) for m in prob.method_names)}",
             f"  Instances          : generate_instances() of the binding "
             f"(seed_base={c['instances']['seed_base']}, "
             f"reuse_existing={c['instances']['reuse_existing']})",
             f"  Exact reference    : "
             + (f"yes (time limit {prob.exact_solver_timelimit} s)" if prob.exact_solver
                else "no -- the gap reference is the best value found"),
             f"  Size classes       : {list(prob.size_classes.keys()) or '(none)'}",
             "",
             f"  Tuning             : enabled={c['tuning']['enabled']}, grid search per instance, "
             f"aggregation={c['tuning']['aggregation']}, "
             f"n_runs_per_combo={c['tuning']['n_runs_per_combo']}",
             f"  Evaluation         : n_runs_per_instance={c['evaluation']['n_runs_per_instance']}, "
             f"warmup={c['evaluation']['warmup']}",
             f"  Fixed parameters   : "
             + (", ".join(prob.label(m) for m in self._resolve_fixed_params(prob, 0))
                if c['tuning'].get('fixed_params') else "none"),
             f"  Convergence        : enabled={c['convergence']['enabled']} "
             f"(recorded during the evaluation runs)",
             f"  Statistics         : alpha={st['alpha']}, blocks={st['tested_blocks']}, "
             f"{', '.join(tests)}"
             + (f" ({st['pairwise_correction']})" if st.get("wilcoxon") else ""),
             f"  Anytime analysis   : {'yes' if c['anytime']['enabled'] else 'no'}",
             f"  Scaling analysis   : "
             + (f"{c['scaling']['blocks']} against {c['scaling']['features']}"
                if c['scaling']['enabled'] else "no"),
             f"  Figures            : formats={c['figures']['formats']}, dpi={c['figures']['dpi']}",
             f"  Worker processes   : n_jobs={c['parallel'].get('n_jobs', -1)} -> "
             f"{resolve_n_jobs(c['parallel'].get('n_jobs', -1))}",
             ""]

        if c["tuning"].get("enabled", True):
            L.append("  Tunable parameters per method "
                     "(problem-level + algorithm-level, merged):")
            for mname in prob.method_names:
                grid = prob.get_param_grid(mname)
                if not grid:
                    L.append(f"    {prob.label(mname):20s} (none -- runs with defaults)")
                    continue
                problem_keys = sorted(prob.problem_param_grids.keys() & grid.keys())
                n_combos = 1
                for v in grid.values():
                    n_combos *= len(v)
                tag = (f"  [problem-level: {', '.join(problem_keys)}]"
                       if problem_keys else "")
                L.append(f"    {prob.label(mname):20s} {grid}  "
                         f"({n_combos} combo(s)){tag}")
        else:
            L.append("  Tuning disabled -- every method runs with the untuned values of "
                     "CONFIG['solvers'] (comparison_config.py).")

        from ..utils.safe_call import _alarm_available
        if ((c["tuning"].get("timeout") or c["evaluation"].get("timeout")) and not _alarm_available()
                and c["parallel"].get("timeout_mode", "alarm") == "alarm"):
            L.append("  WARNING: timeouts are configured but cannot be enforced on this "
                     "platform (SIGALRM is POSIX-only); runs execute to completion. Set "
                     "config['parallel']['timeout_mode'] = 'process' to enforce them.")
        L.append("  " + "-" * 12)
        self._log("\n".join(L))

    def _run_self_check(self, prob):
        vcfg = self.cfg.get("validation", {})
        if not vcfg.get("run_self_check", True) or prob.self_check is None:
            return
        self._log("  [Self-check] verifying declared params are actually "
                  "wired into the model...")
        try:
            prob.self_check()
        except Exception as e:
            self._log(f"  [Self-check] FAILED: {e}")
            if vcfg.get("abort_on_self_check_failure", True):
                raise
            self._log("  [Self-check] continuing anyway "
                      "(config['validation']['abort_on_self_check_failure']=False)")
        else:
            self._log("  [Self-check] OK")

    # ── instances ──────────────────────────────────────────────────────────────
    def _get_instances(self, prob, main):
        path = coded_path(os.path.join(main, "instances.pkl"), prob.code)
        if self.cfg["instances"].get("reuse_existing", True) and os.path.exists(path):
            instances = load_pickle(path)
            self._log(f"  [Instances] reusing instances.pkl ({len(instances)} instance(s))")
            return instances
        seed_everything(make_seed(self.cfg["instances"]["seed_base"], _PH_INST))
        instances = list(prob.generate_instances())
        save_pickle(instances, path)
        self._log(f"  [Instances] generated {len(instances)} (seeded) -> instances.pkl")
        return instances

    def _describe_instances(self, prob, instances):
        """Numeric features of every instance (scaling analysis), kept in the
        results metadata."""
        if prob.instance_features is None:
            return
        fv = {}
        for k, inst in enumerate(instances):
            try:
                fv[k + 1] = {str(a): float(b) for a, b in dict(prob.instance_features(inst)).items()}
            except Exception as e:
                self._log(f"      ! instance_features failed on instance {k + 1}: {e}")
        prob.instance_feature_values = fv or None

    # ── exact reference (once per instance, Appendix D) ─────────────────────────
    def _run_exact(self, instances, prob, ckpt):
        if prob.exact_solver is None:
            return None
        self._log("  [Exact] running exact solver once per instance")
        refs, info = {}, {}
        for inst_idx, inst in enumerate(instances):
            inst_id = inst_idx + 1
            tag = exact_tag(inst_id, instance_signature(inst))
            cached = load_run(ckpt, tag)
            if cached is not None and np.isfinite(cached.get("best_objective", float("nan"))):
                refs[inst_id] = float(cached["best_objective"])
                info[inst_id] = dict(cached); continue
            seed_everything(make_seed(self.cfg["instances"]["seed_base"],
                                      _PH_EXACT, inst_idx))
            t0 = time.perf_counter()
            rec = {}
            try:
                out = prob.exact_solver(inst, prob.exact_solver_timelimit)
                if isinstance(out, dict):
                    # the solver's details (bound, status, gap, time) are kept
                    rec = {k: (v.item() if isinstance(v, np.generic) else v)
                           for k, v in out.items()
                           if isinstance(v, (int, float, str, np.generic)) and not isinstance(v, bool)}
                    f_star = float(out["best_objective"])
                else:
                    f_star = float(out)
            except Exception as e:
                f_star = float("nan")
                self._log(f"      ! exact solver failed on instance {inst_id}: {e}")
            rec["best_objective"] = f_star
            rec["wall_time"] = time.perf_counter() - t0
            if np.isfinite(f_star):         # failures are not cached: retried on resume
                save_run(ckpt, tag, rec)
                info[inst_id] = rec
            refs[inst_id] = f_star
        prob.exact_solver_info = info or None
        return refs

    # ── tuning ──────────────────────────────────────────────────────────────────
    def _run_tuning(self, instances, prob, ckpt):
        """Grid search per instance (see core.tuner). Returns
        {method: {str(inst id): params}}."""
        tcfg = self.cfg["tuning"]
        names = prob.method_names

        fixed = self._resolve_fixed_params(prob, len(instances))
        if fixed:
            self._log(f"  [Tuning] fixed parameters for: "
                      f"{', '.join(prob.label(m) for m in fixed)} (not tuned)")
            fixed_dir = os.path.join(self.main_folder, "Parameter Tuning")
            os.makedirs(fixed_dir, exist_ok=True)
            for m, per_inst in fixed.items():
                save_pickle(per_inst, os.path.join(fixed_dir, f"{m}_best_params.pkl"),
                            code=prob.code)

        if not tcfg.get("enabled", True):
            return {m: fixed.get(m, {}) for m in names}

        n_runs = tcfg["n_runs_per_combo"]; agg = tcfg.get("aggregation", "mean")
        n_jobs = resolve_n_jobs(self.cfg["parallel"].get("n_jobs", -1))
        self._log(f"\n  [Tuning] grid search per instance, {agg} over {n_runs} runs/combo, "
                  f"on the {len(instances)} instance(s)"
                  f"{f' -- {n_jobs} worker process(es)' if n_jobs not in (1, None) else ''}")

        tune_dir = os.path.join(self.main_folder, "Parameter Tuning")
        os.makedirs(tune_dir, exist_ok=True)
        tuner = Tuner(self, prob, ckpt)
        best_params = {}

        for m_idx, mname in enumerate(names):
            if mname in fixed:
                best_params[mname] = fixed[mname]; continue
            full_grid = prob.get_param_grid(mname)
            if not full_grid:
                self._log(f"    {prob.label(mname)}: no tunable params -- "
                          f"running with defaults")
                best_params[mname] = {}; continue
            full = _expand_grid(full_grid)
            problem_keys = sorted(prob.problem_param_grids.keys() & full_grid.keys())
            algo_keys = sorted(set(full_grid.keys()) - set(problem_keys))
            self._log(f"    {prob.label(mname)}: {len(full)} combo(s) "
                      f"x {n_runs} run(s) x {len(instances)} instance(s)"
                      f" = {len(full) * n_runs * len(instances)} call(s)")
            if problem_keys:
                self._log(f"      problem-level params : "
                          f"{ {k: full_grid[k] for k in problem_keys} }")
            if algo_keys:
                self._log(f"      algorithm params      : "
                          f"{ {k: full_grid[k] for k in algo_keys} }")

            best, trace = tuner.run(mname, m_idx, full, instances)
            best_params[mname] = best
            save_pickle(best, os.path.join(tune_dir, f"{mname}_best_params.pkl"),
                        code=prob.code)
            save_pickle(trace, os.path.join(tune_dir, f"{mname}_tuning_trace.pkl"),
                        code=prob.code)
        return best_params

    def _resolve_fixed_params(self, prob, n_instances):
        """config['tuning']['fixed_params'] -> {method: {str(inst_id): params}}
        for the methods it covers: {method: params} (every instance) or
        {method: {"1": params, ...}} (per instance)."""
        fp = self.cfg["tuning"].get("fixed_params")
        if not fp:
            return {}
        unknown = sorted(set(fp) - set(prob.method_names))
        if unknown:
            raise ValueError(f"config['tuning']['fixed_params'] names unknown "
                             f"method(s) {unknown}; known: {prob.method_names}")
        ids = [str(i + 1) for i in range(n_instances)]
        out = {}
        for m, v in fp.items():
            if _is_per_instance(v):
                v = {str(k): dict(p) for k, p in v.items()}
                missing = [i for i in ids if i not in v]
                if missing:
                    raise ValueError(f"fixed_params[{m!r}] has no parameters for "
                                     f"instance(s) {missing}")
                out[m] = {i: v[i] for i in ids}
            else:
                out[m] = {i: dict(v) for i in ids}
        return out

    # ── evaluation (with the warm-up run; also the convergence source) ──────────
    def _run_evaluation(self, instances, prob, best_params, ckpt, main):
        ecfg = self.cfg["evaluation"]
        n_runs = ecfg["n_runs_per_instance"]; seed_base = ecfg["seed_base"]
        timeout = ecfg.get("timeout"); warmup = ecfg.get("warmup", True)
        # with convergence enabled, these SAME runs record the per-generation
        # progress: the algorithms are never re-run to trace convergence
        record_progress = self.cfg["convergence"].get("enabled", True)
        phase = "evalc" if record_progress else "eval"
        n_jobs = resolve_n_jobs(self.cfg["parallel"].get("n_jobs", -1))
        save_solutions = ecfg.get("save_solutions", True)
        timeout_mode = self.cfg["parallel"].get("timeout_mode", "alarm")
        self._log(f"\n  [Evaluation] {n_runs} run(s)/instance"
                  f"{' (+1 warm-up discarded)' if warmup else ''}"
                  f"{' -- recording per-generation progress for convergence' if record_progress else ''}"
                  f"{f' -- {n_jobs} worker process(es)' if n_jobs not in (1, None) else ''}"
                  f"{'' if save_solutions else ' -- solutions NOT saved (objectives only)'}")

        eval_dir = os.path.join(main, "Evaluation Results"); os.makedirs(eval_dir, exist_ok=True)
        for m_idx, mname in enumerate(prob.method_names):
            mres, crashes = {}, 0

            if n_jobs in (1, None):
                # Sequential path. One warm-up per METHOD: the discarded run only
                # absorbs numba's one-time compilation of the study's operators,
                # which does not depend on the instance size.
                already_warmed = False
                inst_iter = progress_iter(
                    list(enumerate(instances)), total=len(instances),
                    desc=f"    [Eval:{prob.label(mname)}]", verbose=self.verbose)
                for inst_idx, inst in inst_iter:
                    inst_id = inst_idx + 1
                    params = dict(best_params.get(mname, {}).get(str(inst_id), {}))
                    sig = params_signature(params, instance_signature(inst))
                    tags = [run_tag(mname, phase, inst_id, r, sig) for r in range(n_runs)]
                    cached = [load_run(ckpt, t) for t in tags]

                    if warmup and not already_warmed and any(c is None for c in cached):
                        wseed = make_seed(seed_base, _PH_WARM, m_idx, inst_idx, 999999)
                        safe_call(prob.get_method(mname), inst, params, False, wseed, timeout,
                                  timeout_mode)
                        already_warmed = True

                    runs = []
                    for r in range(n_runs):
                        res = cached[r]
                        if res is None:
                            seed = make_seed(seed_base, _PH_EVAL, m_idx, inst_idx, r)
                            res = safe_call(prob.get_method(mname), inst, params,
                                            record_progress, seed, timeout, timeout_mode)
                            res = _strip_solutions_from_eval_result(res, save_solutions)
                            save_run(ckpt, tags[r], res)
                        else:
                            res = _strip_solutions_from_eval_result(res, save_solutions)
                        if res.get("_crashed") or res.get("_timeout"):
                            crashes += 1
                        runs.append(res)
                    mres[inst_id] = runs
            else:
                # Parallel path: one worker process per instance (see
                # _evaluate_one_instance); the parent ticks one step per instance.
                method_fn = prob.get_method(mname)

                def _tasks(mname=mname, m_idx=m_idx, method_fn=method_fn):
                    for inst_idx, inst in enumerate(instances):
                        inst_id = inst_idx + 1
                        params = dict(best_params.get(mname, {}).get(str(inst_id), {}))
                        sig = params_signature(params, instance_signature(inst))
                        yield delayed(_evaluate_one_instance)(
                            method_fn, mname, inst, inst_id, params, n_runs, seed_base,
                            m_idx, inst_idx, warmup, timeout, record_progress, phase, sig, ckpt,
                            save_solutions, timeout_mode,
                        )

                parallel_kwargs = dict(n_jobs=n_jobs, backend="loky")
                try:
                    result_iter = Parallel(return_as="generator", **parallel_kwargs)(_tasks())
                except TypeError:  # older joblib without return_as=
                    result_iter = Parallel(verbose=(10 if self.verbose else 0),
                                           **parallel_kwargs)(_tasks())

                result_iter = progress_iter(
                    result_iter, total=len(instances),
                    desc=f"    [Eval:{prob.label(mname)}]", verbose=self.verbose)
                for inst_id, runs, inst_crashes in result_iter:
                    mres[inst_id] = runs
                    crashes += inst_crashes

            # one self-contained file per method: its runs, the study's metadata and
            # the resolved configuration
            save_pickle(
                {"results": mres, "meta": prob.report_metadata(), "config": self.cfg},
                os.path.join(eval_dir, f"{mname}_results.pkl"),
                code=prob.code,
            )
            if crashes:
                self._log(f"      {prob.label(mname)}: {crashes} crashed/timed-out run(s)")
        return eval_dir

    # ── analysis ────────────────────────────────────────────────────────────────
    def _safe_plot(self, label: str, fn, *args, **kwargs):
        """Runs one table, figure or analysis in isolation: a failure is reported
        and skipped, never aborts the rest of the report (the runs themselves are
        checkpointed). Returns the call's result, or None when it failed."""
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(
                f"  [analysis] WARNING: '{label}' failed and is being skipped -- "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            return None

    def _resolve_control(self, prob, scores, hib):
        """Internal name of the control method: config report.control (name
        or label), else the best average per-instance rank on `scores`."""
        names = list(scores.keys())
        if not names:
            return None
        c = self.cfg.get("report", {}).get("control")
        if c:
            for m in names:
                if c in (m, prob.label(m)):
                    return m
            self._log(f"  [Report] WARNING: control {c!r} is not one of the methods; "
                      f"using the best average rank instead")
        from scipy.stats import rankdata
        ids = sorted({i for m in names for i in scores[m]})
        R = []
        for i in ids:
            col = []
            for m in names:
                v = [x for x in scores[m].get(i, []) if x is not None and np.isfinite(x)]
                col.append(float(np.mean(v)) if v else np.nan)
            col = np.asarray(col, dtype=float)
            if np.all(np.isfinite(col)):
                R.append(rankdata(-col if hib else col))
        if not R:
            return names[0]
        return names[int(np.argmin(np.mean(R, axis=0)))]

    def _run_analysis(self, prob, eval_results, conv_results, exact_refs, main):
        self._log("  [Analysis] tables, statistics, figures")
        # the whole report is regenerated from the saved runs: the old one is
        # cleared first, so that no output of an earlier setting lingers
        old = os.path.join(main, "Results Analysis")
        if os.path.isdir(old):
            import shutil
            shutil.rmtree(old, ignore_errors=True)
        cfg = self.cfg
        mono = not prob.is_multiobjective
        paths = make_analysis_tree(main, prob.is_multiobjective, bool(prob.size_classes))

        refs = {}
        scores_obj, scores_gap, scores_time = extract_scores(
            prob, eval_results, paths, exact_refs, refs_out=refs)

        # anytime analysis (mono-objective, against the exact reference)
        anyres = None
        any_dir = os.path.join(paths["root"], "Anytime Performance")
        if mono and cfg["anytime"].get("enabled", False):
            from ..analysis.anytime import compute_anytime
            anyres = self._safe_plot("compute_anytime", compute_anytime, prob, eval_results,
                                     refs, cfg, any_dir, self.verbose)

        obj_block = BLOCK_OBJ_MULTI if prob.is_multiobjective else BLOCK_OBJ_MONO
        obj_label = "Hypervolume" if prob.is_multiobjective else "Objective value"
        obj_hib = prob.is_multiobjective
        control = self._resolve_control(prob, scores_obj, obj_hib)
        self._log(f"  [Report] control method: {prob.label(control) if control else 'none'}")

        # (key, block name, scores, label, higher is better)
        blocks = [("objective", obj_block, scores_obj, obj_label, obj_hib),
                  ("time", BLOCK_TIME, scores_time, "Execution time (s)", False)]
        tested = cfg["statistics"].get("tested_blocks") or []
        stats_results = run_all_statistical_tests(
            prob, [(b, sc, lab, hib) for key, b, sc, lab, hib in blocks if key in tested],
            paths, cfg["statistics"], self.verbose, control=control)

        from ..analysis.report_tables import block_tables, block_summary
        for key, block, scores, ylabel, hib in blocks:
            s = slug(block)
            self._safe_plot("plot_comparison_block", plots.plot_comparison_block,
                            scores, prob, cfg, paths, s, ylabel, self.verbose)
            if key == "objective":
                self._safe_plot("plot_rank_heatmap", plots.plot_rank_heatmap,
                                scores, prob, cfg, paths.get("rank"),
                                hib, ylabel, self.verbose, stem=f"rank_{s}")
            # performance profiles: of the execution time, and of the objective of the
            # mono-objective studies
            if key == "time" or mono:
                self._safe_plot("plot_performance_profiles", plots.plot_performance_profiles,
                                scores, prob, cfg, paths.get(f"{s}/perfprofiles"),
                                hib, ylabel, self.verbose)
            sres = stats_results.get(s, {})
            if sres.get("nemenyi"):
                self._safe_plot("plot_critical_difference", plots.plot_critical_difference,
                                sres["friedman"]["average_ranks"],
                                sres["nemenyi"]["critical_difference"], cfg,
                                paths.get(f"{s}/nemenyi"),
                                f"Critical difference: {ylabel}", self.verbose)
            if sres.get("wilcoxon", {}).get("p_matrix"):
                self._safe_plot("plot_significance_heatmap", plots.plot_significance_heatmap,
                                sres["wilcoxon"], cfg, paths.get(f"{s}/wilcoxon"),
                                f"Pairwise significance: {ylabel}", self.verbose)
            info = self._safe_plot("block_tables", block_tables, scores, prob, cfg,
                                   paths.get(f"{s}/tables"), block, ylabel, hib, control, sres)
            if info and cfg["report"].get("summary", False):
                self._safe_plot("block_summary", block_summary, scores, prob, cfg,
                                paths.get("summary"), block, ylabel, hib, control, info, sres)

        # gap to the reference (best front, or exact value): its table only
        gap_label = "Gap to the best front (%)" if prob.is_multiobjective else "Gap (%)"
        self._safe_plot("block_tables", block_tables, scores_gap, prob, cfg,
                        paths.get(f"{slug(BLOCK_GAP)}/tables"), BLOCK_GAP, gap_label, False,
                        control, {}, markers=False)

        self._safe_plot("plot_quality_vs_time", plots.plot_quality_vs_time,
                        scores_obj, scores_time, prob, cfg,
                        paths.get("tradeoff"), self.verbose)

        if prob.is_multiobjective:
            self._safe_plot("plot_pareto_aggregated", plots.plot_pareto_aggregated,
                            eval_results, prob, cfg,
                            paths.get("pareto"), self.verbose)
            self._safe_plot("_multiobjective_indicators", self._multiobjective_indicators,
                            prob, eval_results, paths)

        if conv_results:
            self._safe_plot("plot_convergence_curves", plots.plot_convergence_curves,
                            conv_results, prob, cfg,
                            paths.get("convergence"), self.verbose)

        if anyres:
            from ..figures import extra as xplots
            self._safe_plot("plot_anytime_curves", xplots.plot_anytime_curves, anyres, refs,
                            prob, cfg, os.path.join(any_dir, "Convergence vs Time"), self.verbose)
            self._safe_plot("plot_time_to_target", xplots.plot_time_to_target, anyres, prob,
                            cfg, os.path.join(any_dir, "Time to Target"), self.verbose)

        # the runs, one row each (Appendix D: read by AnnexeD/complements.py)
        if cfg["report"].get("tidy_csv", False):
            from ..analysis.tables import write_tidy_csv
            cols = [("hypervolume" if prob.is_multiobjective else "objective", scores_obj),
                    ("gap", scores_gap), ("time", scores_time)]
            self._safe_plot("write_tidy_csv", write_tidy_csv, prob, eval_results, cols,
                            os.path.join(paths["root"], f"{prob.code}_results_tidy.csv"))

        scfg = cfg["scaling"]
        if scfg.get("enabled", False) and getattr(prob, "instance_feature_values", None):
            from ..analysis.features import scaling_analysis
            chosen = [(b, sc, lab, hib) for key, b, sc, lab, hib in blocks
                      if key in (scfg.get("blocks") or [])]
            self._safe_plot("scaling_analysis", scaling_analysis, prob, chosen, cfg,
                            os.path.join(paths["root"], "Scaling"), self.verbose)

    def _multiobjective_indicators(self, prob, eval_results, paths):
        out = paths.get("indicators")
        if not out:
            return
        import matplotlib.pyplot as plt
        from ..utils.io import write_csv
        from ..figures.style import get_method_styles, save_figure
        names = list(eval_results.keys())
        ids = sorted({i for m in eval_results for i in eval_results[m]})
        icfg = self.cfg.get("indicators", {})
        ind = compute_indicators(eval_results, names, ids,
                                 normalise=icfg.get("normalise", True),
                                 spacing_ddof=icfg.get("spacing_ddof", 1))
        styles = get_method_styles(names, self.cfg["figures"]["palette"])

        for key, label in [("igd_plus", "IGD+"), ("epsilon", "Epsilon"),
                           ("spacing", "Spacing")]:
            scores = ind[key]
            rows = []
            for i in ids:
                row = {"Instance": i}
                for m in names:
                    c = [v for v in scores[m].get(i, []) if np.isfinite(v)]
                    row[prob.label(m)] = (f"{np.mean(c):.4g} ({np.std(c):.3g})"
                                          if c else "")
                rows.append(row)
            if rows:
                write_csv(rows, os.path.join(out, f"{label}.csv"),
                          header=list(rows[0].keys()), code=prob.code)
            pooled = {m: [v for i in ids for v in scores[m].get(i, []) if np.isfinite(v)]
                      for m in names}
            if not any(pooled.values()):
                continue
            fig, ax = plt.subplots(figsize=self.cfg["figures"]["figsize_default"])
            data = [pooled[m] if pooled[m] else [np.nan] for m in names]
            bp = plots._boxplot(ax, data, [prob.label(m) for m in names],
                                patch_artist=True, showmeans=True)
            for patch, m in zip(bp["boxes"], names):
                patch.set_facecolor(styles[m]["color"]); patch.set_alpha(0.6)
            ax.set_ylabel(f"{label} (lower is better)")
            ax.set_title(f"{label}: distribution (all instances)")
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
            save_figure(fig, out, f"{label.lower().replace('+','plus')}_box", self.cfg)

    # ── logging ──────────────────────────────────────────────────────────────────
    def _log(self, msg):
        if self.verbose:
            print(msg)


# ══════════════════════════════════════════════════════════════════════════════
#  module-level helpers
# ══════════════════════════════════════════════════════════════════════════════
def _deep_update(base: dict, override: dict):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _is_per_instance(v: dict) -> bool:
    """True for {inst_id: params} (keys all integer-like, values dicts)."""
    return bool(v) and all(str(k).isdigit() and isinstance(p, dict) for k, p in v.items())


def _expand_grid(grid: dict):
    if not grid:
        return [{}]
    import itertools
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*[grid[k] for k in keys])]


# What tuning reads back from a cached run: the score and the crash flags. The
# solutions are dropped from the tuning checkpoints (they can be two to three
# orders of magnitude larger than the objectives, e.g. Chapter 6's schedules).
_TUNING_CHECKPOINT_KEYS = (
    "pareto_objectives", "best_objective", "exec_time",
    "_crashed", "_timeout", "_error",
)


def _strip_for_tuning_checkpoint(res: dict) -> dict:
    """Keeps only `_TUNING_CHECKPOINT_KEYS` before a tuning run is pickled."""
    return {k: res[k] for k in _TUNING_CHECKPOINT_KEYS if k in res}


# the solution-shaped fields of a result (multi: one solution per front point;
# mono: one array), dropped from the evaluation results when
# cfg['evaluation']['save_solutions'] is False
_EVAL_SOLUTION_KEYS = ("pareto_front", "best_solution")


def _strip_solutions_from_eval_result(res: dict, save_solutions: bool) -> dict:
    """No-op when `save_solutions` is True (the protocol of the thesis)."""
    if save_solutions:
        return res
    return {k: v for k, v in res.items() if k not in _EVAL_SOLUTION_KEYS}


def _score_combos(combo_runs: dict, agg: str, is_multi: bool) -> dict:
    """{combo_idx: (combo, [valid runs])} for ONE instance ->
    {combo_idx: (aggregated score, combo)}; combos with no valid run are
    dropped.

    Mono-objective: score = best_objective (lower is better).
    Multi-objective: every run's hypervolume is measured in ONE normalisation
    shared by all runs of all combos on this instance
    (`compute_hypervolume_batch`), so combos are compared on the same scale."""
    combo_runs = {ci: v for ci, v in combo_runs.items() if v[1]}
    if not combo_runs:
        return {}
    if is_multi:
        fronts = {ci: [r.get("pareto_objectives", []) or [] for r in runs]
                  for ci, (_, runs) in combo_runs.items()}
        hv = compute_hypervolume_batch(fronts)
        # an empty front dominates nothing: HV 0, not NaN
        per_run = {ci: [0.0 if not np.isfinite(h) else h for h in hv[ci]]
                   for ci in combo_runs}
    else:
        per_run = {ci: [_tuning_score(r, is_multi) for r in runs]
                   for ci, (_, runs) in combo_runs.items()}
    return {ci: (_aggregate(per_run[ci], agg, is_multi), combo_runs[ci][0])
            for ci in combo_runs}


def _tuning_score(result: dict, is_multi: bool) -> float:
    if is_multi:
        pts = result.get("pareto_objectives", [])
        return compute_hypervolume(pts) if pts else 0.0
    v = result.get("best_objective", None)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("inf")
    return v if np.isfinite(v) else float("inf")   # NaN would break min()


def _aggregate(vals, agg: str, is_multi: bool) -> float:
    if agg == "median":
        return float(np.median(vals))
    if agg == "best":
        return float(np.max(vals) if is_multi else np.min(vals))
    return float(np.mean(vals))
