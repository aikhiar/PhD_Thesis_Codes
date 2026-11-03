"""Toy run of one study of methods_comparison, through run_all as run_all.bat runs it (from
Frameworks/methods_comparison): one instance per size class, one run per tuning combination, two
values per tuning parameter, a small population. The sizes are set here, at run time, as the
overrides of RUN in comparison_config.py would set them: the code of the study is not changed.
Results in experiments/toy/<study>.

Besides the runs, it checks the report: every output the chapter reports is there, and no output
of a test the chapter does not report.

    python toy_methods_comparison.py chapter2
"""
import contextlib
import copy
import glob
import importlib
import io
import math
import os
import pickle
import shutil
import sys

sys.path.insert(0, ".")

from methods_comparison import run_all  # noqa: E402
from methods_comparison.config import DEFAULT_CONFIG, RUN_SETTINGS  # noqa: E402
from methods_comparison.run_all import merge  # noqa: E402

GRID_VALUES = 2
TOY = {
    "evaluation": {"n_runs_per_instance": 3},
    "tuning": {"n_runs_per_combo": 1},
    "parallel": {"n_jobs": "auto"},
    "figures": {"formats": ["png"], "dpi": 40},
}
GENERATIONS = 8
# NSGA-III with 3 objectives needs at least 28 individuals (n_divisions = 6)
POPULATION = {"chapter2": 30, "chapter3": 30, "chapter4": 30, "chapter5": 30, "chapter6": 10,
              "appendix_b": 30, "appendix_c": 30, "appendix_d": 10}
# complements.py of AnnexeD reads the progress of the runs up to generation 250
GENERATIONS_OF = {"appendix_d": 260}


def shrink(parameters) -> None:
    """One instance per size class, two values per problem-level tuning parameter."""
    name = "INSTANCES_PER_CLASS" if hasattr(parameters, "INSTANCES_PER_CLASS") else "INSTANCES_PER_SIZE"
    setattr(parameters, name, 1)
    parameters.PROBLEM_PARAM_GRID = {k: v[:GRID_VALUES] for k, v in parameters.PROBLEM_PARAM_GRID.items()}


def prepare(study: str, out: str):
    """The toy settings of the study as RUN["overrides"][study] and RUN["output_dir"], and its binding
    imported again with them (as run_all.bat imports it); returns the binding and the configuration
    of the run."""
    binding = importlib.import_module(f"problems.problem_{study}")
    shrink(binding.P)
    cfg = binding.effective_config()
    algorithms = copy.deepcopy(cfg["solvers"]["algorithms"])
    for spec in algorithms.values():
        spec["grid"] = {k: v[:GRID_VALUES] for k, v in spec.get("grid", {}).items()}
    solvers = {"population_size": POPULATION[study], "generations": GENERATIONS_OF.get(study, GENERATIONS),
               "algorithms": algorithms}
    RUN_SETTINGS["overrides"][study] = merge(TOY, {"solvers": solvers})
    RUN_SETTINGS["output_dir"] = out
    binding = importlib.reload(binding)          # its problem_dict, built with the toy settings
    full = merge(DEFAULT_CONFIG, run_all.study_config(binding, study))
    return binding, full


def check_exact(stored: dict, methods: list) -> None:
    """Appendix D: the exact method solved every instance, and no run is below its optimum."""
    meta = stored[methods[0]]["meta"]
    refs = {int(k): float(v) for k, v in (meta.get("exact_reference_values") or {}).items()}
    info = meta.get("exact_solver_info") or {}
    assert len(refs) == len(stored[methods[0]]["results"]) and all(math.isfinite(v) for v in refs.values()), \
        f"exact references missing ({refs})"
    assert len(info) == len(refs) and all(info[k]["status"] == "optimal" for k in info), \
        "an instance was not solved to optimality by the exact method"
    for name, data in stored.items():
        for instance, runs in data["results"].items():
            for run in runs:
                assert run["best_objective"] >= refs[int(instance)] - 1e-9, f"{name} below the optimum on {instance}"
                assert name in methods or abs(run["best_objective"] - refs[int(instance)]) <= 1e-9, \
                    f"exact method (one thread) differs from the reference on {instance}"


def check_report(study_dir: str, full: dict, mono: bool, exact: bool) -> None:
    """The outputs the chapter reports are there; the others are not."""
    ra = os.path.join(study_dir, "Results Analysis")
    st, rep = full["statistics"], full["report"]
    obj = "Objective Comparison" if mono else "Hypervolume Comparison"
    tested = {"objective": obj, "time": "Execution Time Comparison"}

    def there(*parts):
        return glob.glob(os.path.join(ra, *parts))

    required = [(obj, "Tables", "*table.tex"), ("Execution Time Comparison", "Tables", "*table.tex"),
                ("Gap Comparison", "Tables", "*table.tex"), ("Gap Comparison", "Data", "*Gap.csv"),
                (obj, "Figures", "Distribution", "Box Plots", "PNG", "*box_overall.png"),
                ("Execution Time Comparison", "Figures", "Performance Profiles", "PNG", "*performance_profile.png"),
                ("Rank Heatmaps", "PNG", "*rank_*.png"), ("Quality vs Time Tradeoff", "PNG", "*quality_vs_time.png"),
                ("Convergence Curves", "PNG", "*convergence_inst*.png")]
    if not mono:
        required += [("Pareto Fronts", "PNG", "Aggregated", "*instance_*.png"),
                     ("Multi-objective Indicators", "*IGD+.csv"),
                     ("Multi-objective Indicators", "PNG", "*igdplus_box.png")]
    for key, block in tested.items():
        if key not in st["tested_blocks"]:
            assert not there(block, "Statistical Tests"), f"tests on {block}, not reported by the chapter"
            continue
        required.append((block, "Statistical Tests", "*Statistical Tests Report.txt"))
        for flag, folder in (("friedman", "Friedman Test"), ("nemenyi", "Nemenyi Post-hoc"),
                             ("wilcoxon", "Wilcoxon Test"), ("cliff_delta", "Cliff's Delta")):
            if st[flag]:
                required.append((block, "Statistical Tests", folder, "*.json"))
            else:
                assert not there(block, "Statistical Tests", folder), f"{folder}: not reported by the chapter"
        if st["holm"] or st["bayesian"]:
            required.append((block, "Statistical Tests", "Against Control", "*tests_vs_control.json"))
        else:
            assert not there(block, "Statistical Tests", "Against Control"), "tests against the control: not reported"
        if st["nemenyi"]:
            required.append((block, "Statistical Tests", "Nemenyi Post-hoc", "PNG", "*critical_difference.png"))
    if rep["summary"]:
        required.append(("Summary", "*summary.tex"))
    else:
        assert not there("Summary"), "summary tables: not reported by the chapter"
    if rep["tidy_csv"]:
        required.append(("*_results_tidy.csv",))
    else:
        assert not there("*_results_tidy.csv"), "per-run export: only for AnnexeD/complements.py"
    if full["anytime"]["enabled"] and mono and exact:
        required += [("Anytime Performance", "Convergence vs Time", "PNG", "*anytime_all_instances.png"),
                     ("Anytime Performance", "Time to Target", "PNG", "*ttt_target_0.png")]
    else:
        assert not there("Anytime Performance"), "anytime analysis: not reported by the chapter"
    if full["scaling"]["enabled"]:
        required.append(("Scaling", "PNG", "*scaling_execution_time_comparison_n.png"))
    else:
        assert not there("Scaling"), "scaling analysis: not reported by the chapter"
    missing = [os.path.join(*r) for r in required if not there(*r)]
    assert not missing, f"missing outputs: {missing}"
    # the markers of the per-instance Mann-Whitney tests, only where the chapter reports them
    with open(there(obj, "Tables", "*table.tex")[0], encoding="utf-8") as fh:
        assert ("Mann-Whitney" in fh.read()) == st["mann_whitney"], "Mann-Whitney markers"
    # nothing but the report of the thesis
    other = [p for p in glob.glob(os.path.join(study_dir, "**", "*"), recursive=True)
             if p.endswith((".html", ".zip")) or os.path.basename(p) in
             ("Per Instance", "Violin Plots") and "Per Size" not in p]
    assert not other, f"outputs outside the thesis: {other[:3]}"


def check(binding, study: str, study_dir: str, full: dict, pdict: dict) -> None:
    # single-objective studies (Appendices B and D) run the mono-objective methods
    mono = len(pdict["objective_names"]) == 1
    methods = list(full["solvers"]["mono_methods" if mono else "multi_methods"])
    extra = [m.__name__ for m in binding.extra_methods()] if hasattr(binding, "extra_methods") else []
    everything = methods + extra
    stored = {}
    for name in everything:
        with open(os.path.join(study_dir, "Evaluation Results", f"{study}_{name}_results.pkl"), "rb") as fh:
            stored[name] = pickle.load(fh)
        runs = [x for v in stored[name]["results"].values() for x in v]
        assert runs and not any(x.get("_crashed") or x.get("_timeout") for x in runs), f"crashed runs for {name}"
        if not mono:
            assert all(x.get("pareto_objectives") for x in runs), f"empty front for {name}"
        elif name in methods:
            assert all("progress_evals" in x for x in runs), f"no progress for {name}"
    assert len(glob.glob(os.path.join(study_dir, "Parameter Tuning", "*best_params.pkl"))) == len(everything), \
        "tuning files"
    exact = bool(extra) and binding.exact_available()
    if exact:
        check_exact(stored, methods)
    check_report(study_dir, full, mono, exact)


def run(study: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    out = os.path.join("experiments", "toy", study)
    shutil.rmtree(out, ignore_errors=True)
    binding, full = prepare(study, out)
    pdict = binding.problem_dict
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            ok = run_all.run_studies([study])          # what run_all.bat --only <study> runs
    finally:
        print(buffer.getvalue(), flush=True)
    assert ok, f"run_all: the study {study} failed"
    # every message of the run but run_all's closing summary ("ran 1 [...], failed 0 [-]")
    bad = [line for line in buffer.getvalue().splitlines() if not line.startswith("==== done")
           and any(word in line for word in ("WARNING", "Traceback", "crashed", "failed", "Error"))]
    assert not bad, f"messages of the run: {bad[:5]}"
    folders = glob.glob(os.path.join(out, f"{study}_*"))
    assert len(folders) == 1, f"one study folder expected in {out}, found {folders}"
    study_dir = folders[0]
    check(binding, study, study_dir, full, pdict)
    print(f"Toy run passed ! ({len(pdict['generate_instances']())} instances, {study_dir})")


if __name__ == "__main__":
    run(sys.argv[1])
