"""
param_wiring_check.py  —  generic sanity check that declared problem-level
tunable parameters actually influence a problem's solve pipeline.

Motivation
----------
Auditing the EV-charging chapters found several `problem_params` fields
(e.g. `lam` in chapters 8/11, `variance_factor` in chapters 6/7/9/10) that
were threaded through every operator call but never actually *read* by any
of them — dead parameters. Declaring one of these in `problem_param_grids`
would waste the entire tuning budget searching a dimension that has zero
effect on the outcome, and would silently misreport "we tuned X" when
nothing was tuned at all.

This module gives every problem wrapper a cheap, generic way to catch that
class of bug BEFORE an experiment runs, without the framework knowing
anything about what a particular parameter means. The trick: if a parameter
is genuinely unused, then re-running the exact same
Generate -> Perturb (xN) -> Crossover -> objective pipeline with the *same*
random seed but a *different* value for that parameter must produce
bit-identical output (the code path never consumes the value, so nothing
about its execution can change, on ANY seed). If it's wired in, different
values will produce different output on at least one seed.

Why multiple seeds: some parameters only matter on a probabilistically rare
code path (e.g. a jitter magnitude that's only applied when a solution
lands in a specific branch). A single unlucky seed can make a genuinely
wired parameter look dead by chance. Trying several seeds and requiring a
difference on just ONE of them keeps the false-positive rate low while a
truly dead parameter still can never differ, no matter how many seeds are
tried -- the check stays a reliable test for "is this parameter read at
all", not a statistical test of effect size.

Usage (from a problem wrapper's `self_check` hook)
----------------------------------------------------
    from methods_comparison.utils.param_wiring_check import verify_param_wiring

    def self_check():
        verify_param_wiring(
            build_problem_params=lambda combo, inst: (
                combo.get("perc_perturb", 0.1), combo.get("lam", 1.0), 0.0, 1, inst
            ),
            instance=some_small_fixed_instance,
            generate_solution_fn=chd.Generate_Solution,
            perturb_fn=chd.Perturb,
            objective_fn=chd.objective_function,
            crossover_fn=chd.Crossover,          # optional
            param_grids=problem_dict["problem_param_grids"],
            baseline={"perc_perturb": 0.1, "lam": 1.0},
        )

    problem_dict["self_check"] = self_check
"""

import numpy as np

from ..core.seeding import seed_everything


def verify_param_wiring(
    build_problem_params,
    instance,
    generate_solution_fn,
    perturb_fn,
    objective_fn,
    param_grids: dict,
    baseline: dict,
    seed: int = 12345,
    n_seeds: int = 10,
    n_perturbations: int = 3,
    crossover_fn=None,
) -> None:
    """
    For each parameter in `param_grids`, vary it across its declared values
    (holding every other parameter at `baseline`) and confirm the resulting
    (solution, objective) pair is not identical, on at least one of
    `n_seeds` tried seeds, to the same-seed baseline run.

    Raises AssertionError naming the first parameter found to have no
    detectable effect on ANY seed tried. Returns None on success.
    """

    def _run(combo: dict, s: int):
        # seed_everything (not bare np.random.seed) so @njit-compiled operators
        # -- which keep a RNG state separate from NumPy's -- are reproducible
        # too; otherwise every call looks "different" regardless of the
        # parameter under test, making this check useless.
        seed_everything(s)
        pp = build_problem_params(combo, instance)
        sol = generate_solution_fn(pp)
        for _ in range(n_perturbations):      # more chances to hit rare branches
            sol = perturb_fn(sol, pp)
        if crossover_fn is not None:
            sol = crossover_fn(np.stack([sol, sol]), pp)
        obj = np.asarray(objective_fn(sol, pp), dtype=float)
        return np.asarray(sol, dtype=float).tobytes() + obj.tobytes()

    for pname, values in param_grids.items():
        if len(values) < 2:
            continue                      # nothing to vary against
        differed = False
        for s in range(seed, seed + n_seeds):
            base_result = _run(baseline, s)
            for v in values:
                combo = dict(baseline)
                combo[pname] = v
                if _run(combo, s) != base_result:
                    differed = True
                    break
            if differed:
                break
        if not differed:
            raise AssertionError(
                f"Parameter '{pname}' is declared in problem_param_grids with "
                f"values {list(values)}, but varying it produced IDENTICAL "
                f"output (same instance) for every value tested against the "
                f"baseline {baseline}, across {n_seeds} different seeds. It "
                f"is very likely not wired into "
                f"Generate/Perturb/Crossover/objective_function — tuning it "
                f"would waste the entire tuning budget on a dimension with "
                f"no effect. Check the dependency file before including it "
                f"in problem_param_grids."
            )
