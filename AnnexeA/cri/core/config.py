"""Configuration: YAML files with inheritance ("base: other.yaml") and deep merge."""

import copy
import os

import yaml


def deep_merge(a, b):
    """Recursive merge of b into a; lists are replaced; a dict of b with "__replace__: true"
    replaces the dict of a instead of being merged into it."""
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and v.get("__replace__"):
            out[k] = {kk: copy.deepcopy(vv) for kk, vv in v.items() if kk != "__replace__"}
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path, overrides=None):
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(load_config(os.path.join(os.path.dirname(path), base)), cfg)
    cfg = deep_merge(cfg, overrides or {})
    cfg["_config_path"] = path
    validate(cfg)
    return cfg


def validate(cfg):
    for key in ("run", "instance_sets", "experiments"):
        if key not in cfg:
            raise ValueError(f"config: missing section '{key}'")
    sets = {s["name"] for s in cfg["instance_sets"]}
    for name, ex in cfg["experiments"].items():
        for s in ex.get("sets", []):
            if s not in sets:
                raise ValueError(f"experiment {name}: unknown instance set '{s}'")
        if not ex.get("methods"):
            raise ValueError(f"experiment {name}: no methods")
    return True
