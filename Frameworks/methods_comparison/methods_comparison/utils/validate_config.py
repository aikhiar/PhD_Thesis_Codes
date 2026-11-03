"""
validate_config.py  —  start-up validation of the configuration and of the
problem_dict of a binding, with complete error messages.
"""

VALID_AGGREGATIONS = {"mean", "median", "best"}
VALID_FORMATS      = {"svg", "png", "pdf"}
VALID_BLOCKS       = {"objective", "time"}
_REQUIRED_SECTIONS = ["instances", "tuning", "evaluation", "convergence", "indicators",
                      "parallel", "validation", "statistics", "report", "anytime",
                      "scaling", "figures", "solvers"]


def validate_config(cfg: dict):
    errors = []

    for section in _REQUIRED_SECTIONS:
        if section not in cfg:
            errors.append(f"Missing top-level section: '{section}'")
    if errors:
        raise ValueError(_fmt(errors))

    # positive-integer run counts
    _pos_int(cfg, ["tuning", "n_runs_per_combo"], errors)
    _pos_int(cfg, ["evaluation", "n_runs_per_instance"], errors)
    _pos_int(cfg, ["instances", "seed_base"], errors, allow_zero=True)

    _bool(cfg, ["instances", "reuse_existing"], errors)
    _bool(cfg, ["tuning", "enabled"], errors)
    _bool(cfg, ["evaluation", "warmup"], errors)
    _bool(cfg, ["evaluation", "save_solutions"], errors)
    _bool(cfg, ["convergence", "enabled"], errors)
    _bool(cfg, ["validation", "run_self_check"], errors)
    _bool(cfg, ["validation", "abort_on_self_check_failure"], errors)

    agg = cfg["tuning"].get("aggregation", "mean")
    if agg not in VALID_AGGREGATIONS:
        errors.append(f"config['tuning']['aggregation'] must be one of "
                      f"{VALID_AGGREGATIONS} — got '{agg}'")

    for section in ("tuning", "evaluation"):
        t = cfg[section].get("timeout", None)
        if t is not None and (not isinstance(t, (int, float)) or t <= 0):
            errors.append(f"config['{section}']['timeout'] must be > 0 or None — got {t}")

    fp = cfg["tuning"].get("fixed_params", None)
    if fp is not None and not isinstance(fp, dict):
        errors.append("config['tuning']['fixed_params'] must be None or a dict {method: params}")
    elif isinstance(fp, dict):
        for m, v in fp.items():
            if not isinstance(v, dict):
                errors.append(f"config['tuning']['fixed_params'][{m!r}] must be a dict")

    ind = cfg["indicators"]
    if ind.get("normalise", True) not in (True, False):
        errors.append("config['indicators']['normalise'] must be True or False")
    if ind.get("spacing_ddof", 1) not in (0, 1):
        errors.append("config['indicators']['spacing_ddof'] must be 0 or 1")

    n_jobs = cfg["parallel"].get("n_jobs", -1)
    if n_jobs is not None and n_jobs != "auto" and not (
            isinstance(n_jobs, int) and (n_jobs == -1 or n_jobs > 0)):
        errors.append(f"config['parallel']['n_jobs'] must be None, 'auto' (physical cores), "
                      f"-1 (all logical cores), or a positive integer — got {n_jobs!r}")
    tm = cfg["parallel"].get("timeout_mode", "alarm")
    if tm not in ("alarm", "process"):
        errors.append(f"config['parallel']['timeout_mode'] must be 'alarm' or 'process' — got {tm!r}")

    st = cfg["statistics"]
    alpha = st.get("alpha", 0.05)
    if not (0 < alpha < 1):
        errors.append(f"config['statistics']['alpha'] must be in (0,1) — got {alpha}")
    tb = st.get("tested_blocks", [])
    if not isinstance(tb, (list, tuple)) or any(b not in VALID_BLOCKS for b in tb):
        errors.append(f"config['statistics']['tested_blocks'] must be a list of {VALID_BLOCKS}")
    for key in ("friedman", "nemenyi", "wilcoxon", "cliff_delta", "mann_whitney",
                "holm", "bayesian"):
        _bool(cfg, ["statistics", key], errors)
    if st.get("pairwise_correction", "holm") not in ("holm", "bonferroni"):
        errors.append("config['statistics']['pairwise_correction'] must be 'holm' or "
                      f"'bonferroni' — got {st.get('pairwise_correction')!r}")
    rope = st.get("bayes_rope", 0.01)
    if not isinstance(rope, (int, float)) or rope < 0:
        errors.append(f"config['statistics']['bayes_rope'] must be >= 0 — got {rope!r}")
    _pos_int(cfg, ["statistics", "bayes_samples"], errors)

    r = cfg["report"]
    if r.get("control", None) is not None and not isinstance(r.get("control"), str):
        errors.append("config['report']['control'] must be None or a method name")
    _bool(cfg, ["report", "summary"], errors)
    _bool(cfg, ["report", "tidy_csv"], errors)
    _pos_int(cfg, ["report", "decimals"], errors, allow_zero=True)
    _pos_int(cfg, ["report", "bootstrap_samples"], errors)
    ma = r.get("marker_alpha", 0.05)
    if not isinstance(ma, (int, float)) or not 0 < ma < 1:
        errors.append(f"config['report']['marker_alpha'] must be in (0,1) — got {ma!r}")

    _bool(cfg, ["anytime", "enabled"], errors)
    tg = cfg["anytime"].get("targets", [])
    if not isinstance(tg, (list, tuple)) or not all(
            isinstance(v, (int, float)) and v >= 0 for v in tg):
        errors.append("config['anytime']['targets'] must be a list of gaps >= 0 (in %)")

    sc = cfg["scaling"]
    _bool(cfg, ["scaling", "enabled"], errors)
    if not (isinstance(sc.get("features", []), (list, tuple))
            and all(isinstance(x, str) for x in sc.get("features", []))):
        errors.append("config['scaling']['features'] must be a list of feature names")
    if any(b not in VALID_BLOCKS for b in sc.get("blocks", [])):
        errors.append(f"config['scaling']['blocks'] must be a list of {VALID_BLOCKS}")

    fmts = cfg["figures"].get("formats", [])
    bad = [f for f in fmts if f not in VALID_FORMATS]
    if bad:
        errors.append(f"config['figures']['formats'] unknown: {bad}. "
                      f"Valid: {VALID_FORMATS}")
    if not isinstance(cfg["figures"].get("palette", []), list) or \
            not cfg["figures"]["palette"]:
        errors.append("config['figures']['palette'] must be a non-empty list")

    _validate_solvers(cfg, errors)
    if errors:
        raise ValueError(_fmt(errors))
    _warn_unknown_keys(cfg)


def _validate_solvers(cfg: dict, errors: list):
    """CONFIG['solvers']: the solvers run on the operators of a binding."""
    s = cfg.get("solvers")
    if not isinstance(s, dict):
        errors.append("config['solvers'] must be a dict")
        return
    for key in ("population_size", "generations"):
        v = s.get(key)
        if not (isinstance(v, int) and v >= 1):
            errors.append(f"config['solvers']['{key}'] must be a positive integer, got {v!r}")
    algos = s.get("algorithms", {})
    if not isinstance(algos, dict):
        errors.append("config['solvers']['algorithms'] must be a dict {algorithm: spec}")
        return
    for key in ("mono_methods", "multi_methods"):
        names = s.get(key, [])
        if not isinstance(names, (list, tuple)) or not names:
            errors.append(f"config['solvers']['{key}'] must be a non-empty list")
            continue
        for a in names:
            if a not in algos:
                errors.append(f"config['solvers']['{key}']: {a!r} has no entry in "
                              f"config['solvers']['algorithms']")
    for a, spec in algos.items():
        if not isinstance(spec, dict):
            errors.append(f"config['solvers']['algorithms'][{a!r}] must be a dict")
            continue
        for part in ("params", "grid"):
            if not isinstance(spec.get(part, {}), dict):
                errors.append(f"config['solvers']['algorithms'][{a!r}]['{part}'] must be a dict")
        for k, vals in spec.get("grid", {}).items():
            if not isinstance(vals, (list, tuple)) or not vals:
                errors.append(f"config['solvers']['algorithms'][{a!r}]['grid'][{k!r}] must be "
                              f"a non-empty list")


def _warn_unknown_keys(cfg: dict):
    """Unknown keys are ignored; a typo (e.g. 'n_run_per_instance') would
    silently fall back to the default, hence the warning."""
    from ..config import DEFAULT_CONFIG
    unknown = []
    for sec, val in cfg.items():
        if sec.startswith("_"):
            continue
        if sec not in DEFAULT_CONFIG:
            unknown.append(sec)
        elif isinstance(val, dict) and isinstance(DEFAULT_CONFIG[sec], dict):
            unknown += [f"{sec}.{k}" for k in val if k not in DEFAULT_CONFIG[sec]]
    if unknown:
        print(f"  [config] WARNING: unknown key(s) ignored: {', '.join(unknown)}",
              flush=True)


def validate_problem_dict(problem_dict: dict):
    errors = []
    for key in ("generate_instances", "objective_function", "solving_methods"):
        if key not in problem_dict:
            errors.append(f"problem_dict missing required key: '{key}'")

    obj_fn = problem_dict.get("objective_function")
    if obj_fn is not None and not callable(obj_fn):
        errors.append("'objective_function' must be a callable(instance, solution) "
                      "-> tuple of all objective values")

    names = problem_dict.get("objective_names")
    if not names or not isinstance(names, (list, tuple)):
        errors.append("'objective_names' must be a non-empty list of strings "
                      "(n_objectives is derived from its length)")

    code = problem_dict.get("code")
    if not code or not isinstance(code, str):
        errors.append("'code' must be a non-empty string -- every folder/file "
                      "the framework saves is prefixed with it")

    methods = problem_dict.get("solving_methods")
    if methods is not None:
        if not isinstance(methods, (list, tuple)) or len(methods) == 0:
            errors.append("'solving_methods' must be a non-empty list of callables")
        else:
            for i, fn in enumerate(methods):
                if not callable(fn):
                    errors.append(f"'solving_methods[{i}]' is not callable")

    gen_fn = problem_dict.get("generate_instances")
    if gen_fn is not None and not callable(gen_fn):
        errors.append("'generate_instances' must be a callable")

    ex = problem_dict.get("exact_solver")
    if ex is not None:
        if not callable(ex):
            errors.append("'exact_solver' must be a callable(instance, timelimit)")
        if names is not None and isinstance(names, (list, tuple)) and len(names) > 1:
            errors.append("'exact_solver' is set but the study is multi-objective; "
                          "the exact reference applies to mono-objective studies only.")

    ppg = problem_dict.get("problem_param_grids")
    if ppg is not None:
        if not isinstance(ppg, dict):
            errors.append("'problem_param_grids' must be a dict of "
                          "{param_name: [values...]}")
        else:
            for k, v in ppg.items():
                if not isinstance(v, (list, tuple)) or len(v) == 0:
                    errors.append(f"'problem_param_grids[{k!r}]' must be a "
                                  f"non-empty list of values")

    for key, sig in (("self_check", "callable() -> None"),
                     ("instance_features", "callable(instance) -> {feature: number}")):
        v = problem_dict.get(key)
        if v is not None and not callable(v):
            errors.append(f"'{key}' must be a {sig} (or omitted)")

    if errors:
        raise ValueError(_fmt(errors))


# ── helpers ──────────────────────────────────────────────────────────────────
def _pos_int(cfg, path, errors, allow_zero=False):
    node = cfg
    for k in path[:-1]:
        node = node.get(k, {})
    val = node.get(path[-1], None)
    if val is None:
        return
    lo = 0 if allow_zero else 1
    if not isinstance(val, int) or val < lo:
        key = "".join(f"['{p}']" for p in path)
        errors.append(f"config{key} must be an integer >= {lo} — got {val!r}")


def _bool(cfg, path, errors):
    node = cfg
    for k in path[:-1]:
        node = node.get(k, {})
    if path[-1] in node and node[path[-1]] not in (True, False):
        key = "".join(f"['{p}']" for p in path)
        errors.append(f"config{key} must be True or False — got {node[path[-1]]!r}")


def _fmt(errors):
    joined = "\n  - ".join(errors)
    return (f"\n[methods_comparison] Configuration error(s):\n  - {joined}\n"
            f"Fix the above before running the study.")
