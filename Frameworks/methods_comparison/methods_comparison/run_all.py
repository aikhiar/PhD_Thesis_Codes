"""
run_all.py  --  runs the metaheuristics comparisons of the thesis (called by run_all.bat).

The studies are those of STUDIES in comparison_config.py, each bound by
problems/problem_<study>.py. Every study runs the full pipeline (instances ->
tuning -> evaluation -> analysis); an interrupted run resumes from its checkpoints,
and a study that fails does not stop the next one.

Configuration of each study:
    CONFIG (comparison_config.py) < the binding's STUDY_CONFIG < RUN["overrides"][study]

Usage (from the main folder):
    run_all.bat                      every study
    run_all.bat --only chapter6      one study (several names allowed)
    run_all.bat --estimate           expected duration of each study on this machine
"""

import argparse
import copy
import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods_comparison import ExperimentRunner                         # noqa: E402
from methods_comparison.config import DEFAULT_CONFIG, RUN_SETTINGS, STUDIES   # noqa: E402


def merge(a: dict, b: dict) -> dict:
    """Deep merge: values of b override those of a (sections merged key by key)."""
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def study_config(module, study: str) -> dict:
    """Overrides of CONFIG for one study: its STUDY_CONFIG, then RUN["overrides"]."""
    own = getattr(module, "STUDY_CONFIG", None) or {}
    return merge(own, RUN_SETTINGS.get("overrides", {}).get(study, {}))


def selected_studies(only=None) -> list:
    """The studies to run, in the order of STUDIES; `only` accepts study names
    (chapter2) or binding names (problem_chapter2)."""
    only = only if only is not None else RUN_SETTINGS.get("only")
    if not only:
        return list(STUDIES)
    names = [o[len("problem_"):] if o.startswith("problem_") else o for o in only]
    unknown = [n for n in names if n not in STUDIES]
    if unknown:
        raise SystemExit(f"unknown stud{'y' if len(unknown) == 1 else 'ies'} {unknown}; "
                         f"the studies are {list(STUDIES)}")
    return [s for s in STUDIES if s in names]


def load_study(study: str):
    module = importlib.import_module(f"problems.problem_{study}")
    return module, module.problem_dict


def run_studies(only=None):
    studies = selected_studies(only)
    out_dir = str(ROOT / RUN_SETTINGS["output_dir"])
    print(f"Studies: {', '.join(studies)}")
    ran, failed = [], []
    for study in studies:
        print(f"\n########## {study}: {STUDIES[study]} ##########")
        try:
            module, pdict = load_study(study)
            ExperimentRunner(pdict, output_dir=out_dir, config=study_config(module, study),
                             verbose=True).run()
            ran.append(study)
        except Exception as exc:                               # run error -> next study
            print(f"[error] {study}: run failed ({exc})")
            traceback.print_exc()
            failed.append(study)
    print(f"\n==== done -- ran {len(ran)} [{', '.join(ran) or '-'}], "
          f"failed {len(failed)} [{', '.join(failed) or '-'}] ====")
    return not failed


def run_estimate(only=None):
    """Each binding's estimate(): short timed runs of every method, one instance per
    size class, extrapolated to the whole study."""
    for study in selected_studies(only):
        print(f"\n[estimate] {study}: {STUDIES[study]}")
        try:
            module, _ = load_study(study)
            module.estimate()
        except Exception as exc:                               # e.g. Chapter 5 before its surrogates
            print(f"[estimate] {study}: not available ({type(exc).__name__}: {exc})")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Metaheuristics comparisons of the thesis.")
    ap.add_argument("--only", nargs="*", default=None, help="studies to run, e.g. chapter2 appendix_d")
    ap.add_argument("--estimate", action="store_true", help="print the expected durations only")
    a = ap.parse_args(argv)
    if a.estimate:
        run_estimate(a.only)
        return
    if not run_studies(a.only):
        sys.exit(1)


if __name__ == "__main__":
    main()
