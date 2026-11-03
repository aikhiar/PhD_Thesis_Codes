"""Toy run of the det_vs_stoch pipeline (from Frameworks/det_vs_stoch): one instance per size class,
one run per tuning combination, two values per tuning parameter, a small population. The sizes are
set here, at run time: the code of the chapter is not changed. Results in runs/toy/.

Besides the run, it checks the report: every output the chapter reports is there, and no output the
chapter does not report.

    python pilot_det_vs_stoch.py chapter2
"""
import glob
import importlib
import os
import sys

sys.path.insert(0, ".")
import main  # noqa: E402

slug = sys.argv[1]
mod = importlib.import_module(f"problems.problem_{slug}")
main.RESAMPLING_CONFIG["tuning"]["n_runs_per_combo"] = 1
grid = dict(mod.param_grid_for_best_algorithm())
grid = {k: v[:2] if isinstance(v, list) else v for k, v in grid.items()}
grid.update({"population_size": [20], "generations": [6]})
override = {r: dict(grid) for r in mod.REGIMES}
study = set(mod.generate_instance_seeds())
seeds = [min(set(ids) & study) for ids in mod.INSTANCE_SIZE_CLASSES.values() if set(ids) & study]
assert len(seeds) == len(mod.INSTANCE_SIZE_CLASSES), seeds     # an empty list would run the whole study
print("toy instances:", seeds, flush=True)
out = main.run_problem(mod, reporting_root=os.path.join(main.REPORTING_ROOT, "toy"), instance_seeds_override=seeds,
                       param_grid_override=override, verbose=True, show_progress=False)
print("report:", out, flush=True)

# ── the report of the chapter, and nothing else ─────────────────────────────────
_, report = main._report_settings(mod)
code = mod.CODE


def there(*parts):
    return glob.glob(os.path.join(out, *parts))


required = [("Data", "per_instance_hv_raw.csv"), ("Data", "planned_fronts_long.csv"),
            ("Data", "realized_fronts_long.csv"), ("Metrics", "per_instance_summary.tex"),
            ("Metrics", "per_size_class_summary.tex"), ("Statistical Tests", "per_instance_tests.tex"),
            ("Figures", "HV Realized", "PNG", f"{code}_hv_realized_lineplot.png"),
            ("Figures", "HV by Size Class", "PNG", f"{code}_hv_by_size_class_boxplot.png"),
            ("manifest.json",)]
optional = {"across_instances": [("Statistical Tests", "across_instance_tests.tex")],
            "run_level": [("Run-level analysis", "*")],
            "promise_vs_delivery": [("Metrics", "promise_vs_delivery_summary.tex"),
                                    ("Metrics", "realized_objectives_summary.tex"),
                                    ("Figures", "Promise vs Delivery", "PNG", "*.png")],
            "realized_fronts": [("Figures", "Realized Fronts", "PNG", "*.png")]}
for key, outputs in optional.items():
    if report[key]:
        required += outputs
    else:
        present = [os.path.join(*o) for o in outputs if there(*o)]
        assert not present, f"{key}: not reported by the chapter, yet written: {present}"
missing = [os.path.join(*r) for r in required if not there(*r)]
assert not missing, f"missing outputs: {missing}"
figures = {os.path.basename(p) for p in there("Figures", "*")}
assert figures <= {"HV Realized", "HV by Size Class", "Promise vs Delivery", "Realized Fronts"}, figures
assert not there("report.md"), "report.md: not part of the thesis"
with open(there("Statistical Tests", "_HOW_IT_IS_COMPUTED.txt")[0], encoding="utf-8") as fh:
    assert f"{report['correction']}-adjusted" in fh.read(), "correction of the per-instance tests"
print(f"Pilot run passed ! ({len(seeds)} instances, {out})")
