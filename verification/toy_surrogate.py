"""Toy run of the surrogate_trainer pipeline (from Frameworks/surrogate_trainer): a few small real
instances per size class, every portfolio algorithm at a small population, and a reduced tuning budget.
It goes through every stage: data generation, split by instance, tuning with folds of instances, benchmark,
two-stage selection, reports and export. Results in runs/toy/. The models it exports are toy models, never
to be deployed in a real study.

Besides the run, it checks the report: every output the chapter reports is there, and no other.

    python toy_surrogate.py chapter5
"""
import copy
import glob
import importlib
import os
import sys

sys.path.insert(0, ".")

from dependency.config.config import ML_CONFIG  # noqa: E402
from main import REPORTING_ROOT, _slugify, run_problem  # noqa: E402

# (class label, n vehicles): several instances per class, so that the split by
# instance has instances on both sides in every class
TOY_CLASSES = [("small", 10), ("medium", 20)]
INSTANCES_PER_CLASS = 3
SEED_BASE = 555


def toy_instances(problem_module):
    out, k = [], 0
    for label, n in TOY_CLASSES:
        for j in range(INSTANCES_PER_CLASS):
            out.append((f"{label}_{j}", problem_module.build_instance(n, SEED_BASE + k, name=f"tiny-{label}-{j}")))
            k += 1
    return out


def toy_algorithms(base_algorithms, population_size=30, generations=8):
    out = {}
    for name, spec in base_algorithms.items():
        solve_kwargs = dict(spec["solve_kwargs"], population_size=population_size, generations=generations)
        out[name] = {"kind": spec["kind"], "params": dict(spec["params"]), "solve_kwargs": solve_kwargs}
    return out


def toy_ml_config():
    cfg = copy.deepcopy(ML_CONFIG)
    cfg.update({"search_n_iter": 2, "cv_folds": 2, "log_generation_stride": 2,
                "benchmark": {**cfg["benchmark"], "schedules_per_class": 2, "predict_repeats": 20,
                              "feature_repeats": 10, "real_repeats": 2},
                "figures": {"formats": ["png"], "dpi": 80}})
    return cfg


def check(problem_module, out_dir: str) -> None:
    """The outputs of Chapters 5 and 6, and no other."""
    def there(*parts):
        return glob.glob(os.path.join(out_dir, *parts))

    learned = getattr(problem_module, "SURROGATE_OBJECTIVE_NAMES", problem_module.objective_names)
    required = [("training_data", "Metrics", "records_by_class_and_algorithm.tex"),
                ("optimizer_quality", "Metrics", "dominance_summary.tex"),
                ("optimizer_quality", "Figures", "*", "PNG", "dominance_agreement.png"),
                ("execution_time_analysis", "Data", "training_time.tex"),
                ("execution_time_analysis", "Data", "inference_time.tex"),
                ("execution_time_analysis", "Data", "cost_accounting.tex"),
                ("surrogate_export", "best_models", "model_card.json"), ("dataset_summary.json",)]
    for name in learned:
        o = _slugify(name)
        required += [(o, "Metrics", "leaderboard.tex"), (o, "Metrics", "hyperparameters.tex"),
                     (o, "Metrics", "ranking_by_model.tex"), (o, "Statistical Tests", "significance_tests.tex"),
                     (o, "Figures", "Predictions", "PNG", "pred_vs_actual.png"),
                     (o, "Figures", "Predictions", "PNG", "residuals.png"),
                     (o, "Figures", "Model Comparison", "PNG", "model_comparison.png")]
        for folder in ("Feature Importance", "Critical Difference", "Error by Generation", "Accuracy by Class",
                       "Learning Curve"):
            assert not there(o, "Figures", folder), f"{o}/Figures/{folder}: not reported by the chapter"
        assert not there(o, "Metrics", "accuracy_by_class*"), "accuracy by class: not reported by the chapter"
    missing = [os.path.join(*r) for r in required if not there(*r)]
    assert not missing, f"missing outputs: {missing}"
    for part in (("end_to_end",), ("optimizer_quality", "Figures", "Objective Space Coverage"),
                 ("training_data", "Figures"), ("execution_time_analysis", "Figures", "Break-even")):
        assert not there(*part), f"{os.path.join(*part)}: not reported by the chapter"


def run(code):
    problem_module = importlib.import_module(f"problems.problem_{code}")
    out_dir = run_problem(
        problem_module,
        reporting_root=os.path.join(REPORTING_ROOT, "toy"),
        instances_override=toy_instances(problem_module),
        algorithms_override=toy_algorithms(problem_module.ALGORITHMS),
        verbose=True,
        show_progress=False,
        ml_config=toy_ml_config(),
    )
    check(problem_module, out_dir)
    print(f"Toy training done: {code}, {out_dir}")


if __name__ == "__main__":
    run(sys.argv[1])
