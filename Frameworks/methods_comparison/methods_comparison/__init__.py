"""
methods_comparison — the metaheuristics comparisons of the thesis (Chapters 2 to 6,
Appendices B, C and D): instances, tuning, evaluation runs, statistical tests and
figures, for the studies listed in comparison_config.py.

Convention used throughout the framework
----------------------------------------
*Every objective is MINIMISED.*  Consequently:
  * mono-objective    : lower `best_objective` is better;
  * multi-objective   : the Pareto front lives in the lower-left, the
                        hypervolume reference point is the (normalised) upper
                        corner, so a HIGHER hypervolume is better;
  * gap and exec-time : lower is better.
The ranking direction of every statistical test and figure is derived from
this single convention.
"""

from .requirements import ensure_requirements as _ensure_requirements

_ensure_requirements()                   # checks, and installs if missing, the required packages

from .core.runner import ExperimentRunner

__all__ = ["ExperimentRunner"]
