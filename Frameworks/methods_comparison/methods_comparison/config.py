"""
config.py  --  loads comparison_config.py, the protocol of the comparisons.

Every default value lives in ONE file, `comparison_config.py`, in the main folder
(next to run_all.bat); this module only loads it. All keys are validated at
start-up by ``utils.validate_config.validate_config``.
"""

import copy
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "comparison_config.py"


def _load_user_config():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"configuration file not found: {CONFIG_FILE}")
    spec = importlib.util.spec_from_file_location("comparison_config", CONFIG_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("STUDIES", "RUN", "CONFIG"):
        if not isinstance(getattr(mod, name, None), dict):
            raise ValueError(f"{CONFIG_FILE} must define a dict {name}")
    return mod


_USER = _load_user_config()

# the studies of the thesis, in the order of the manuscript
STUDIES = dict(_USER.STUDIES)
# the protocol of every study (sections instances ... solvers)
DEFAULT_CONFIG = copy.deepcopy(_USER.CONFIG)
# what run_all runs
RUN_SETTINGS = copy.deepcopy(_USER.RUN)

# Sections that decide the runs. Only they enter the experiment-folder hash:
# changing a report, statistics or figure setting re-uses every cached run.
RUN_SECTIONS = ("instances", "tuning", "evaluation", "convergence", "solvers")
