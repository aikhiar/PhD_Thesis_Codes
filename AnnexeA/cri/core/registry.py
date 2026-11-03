"""Registries of problems and methods. A method is a class registered under a name, built from a spec
{"id": display name, "class": registered name, "params": {...}} and run through
    method.run(instance, context, budget, seed, lower_bound=None) -> RunResult-compatible dict
"""

METHODS = {}
PROBLEMS = {}
DEFAULTS = {}       # per-class default parameters, set from the config by the runner


def set_defaults(d):
    DEFAULTS.clear()
    DEFAULTS.update(d or {})


def register_method(name):
    def deco(cls):
        cls.registry_name = name
        METHODS[name] = cls
        return cls
    return deco


def register_problem(name):
    def deco(obj):
        PROBLEMS[name] = obj
        return obj
    return deco


def make_method(spec, defaults=None):
    """spec: {"id", "class", "params"}; defaults: per-class default parameters from the config."""
    cls = METHODS[spec["class"]]
    params = dict((DEFAULTS if defaults is None else defaults).get(spec["class"], {}))
    params.update(spec.get("params", {}) or {})
    m = cls(**params)
    m.id = spec.get("id", spec["class"])
    m.spec = spec
    return m
