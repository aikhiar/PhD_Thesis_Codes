# Deterministic versus stochastic planning

Measures what planning under uncertainty buys. For one chapter, it builds schedules two ways, from
the planned values (the deterministic regime) and from the simulated stochastic objective (the
stochastic regime), then subjects both to the same drawn realizations and scores them on equal
terms. It supports Chapters 2, 3, 4 and 6 of the thesis; Chapter 5 has no such study.

## What it does

1. **Per regime**: build the instances, run an independent tuning pass of MOCS, and evaluate the
   tuned combination to obtain a non-dominated front, schedules included, for every instance.
2. **Per instance**: run the R-replication resampling protocol (R = 30). One realization is drawn
   and applied to the schedules of both regimes, which are scored against it; the fronts are rebuilt
   and a comparable hypervolume is computed.
3. **Report**: what the chapter reports, as set by the `REPORT` of its binding.

| Chapter | Report |
|---|---|
| Chapters 2, 3 and 4 | realized hypervolume per instance (95% bootstrap band) and per size class; per instance, a paired Wilcoxon test over the realizations, Bonferroni-corrected across the instances, with Cliff's delta |
| Chapter 6 | the same, Holm-adjusted; the across-instance tests (overall and per size class, with bootstrap intervals); the run-level tests (Mann-Whitney per instance, Holm-adjusted, and Wilcoxon across instances); promise against delivery and the realized objectives; the realized fronts and their attainment surfaces |

Scoring both regimes on the same draw is what makes their difference attributable to the planning
regime rather than to the draw.

## Running it

```bash
python main.py                  # Chapters 2, 3, 4 and 6
python main.py chapter2         # one chapter
python main.py --estimate       # expected duration, without running
python main.py --resume runs/det_stoch_chapter2_ab12cd34
```

Output goes to `runs/det_stoch_<code>_<key>/`, a new folder per run. Every finished tuning and
evaluation task is cached there, so `--resume` restarts an interrupted run with identical results.

| Chapter | Instances | Evaluation runs per regime |
|---|---|---|
| Chapter 2 | 20 | 5 |
| Chapter 3 | 8 (the first 2 of each size class) | 5 |
| Chapter 4 | 8 (the first 2 of each size class) | 5 |
| Chapter 6 | 20 | 10 |

## Layout

```
main.py                    entry point
dependency/
  config/                  the settings of the protocol
  algorithms/              the multi-objective solvers
  tuning/                  per-regime parameter search
  evaluation/              front construction per instance
  resampling/              realization draws, hypervolume, statistics, seeding
  reporting/               tables, figures, run-level analysis
problems/
  problem_<chapter>.py     the regime contract and the report of the chapter
  <chapter>/               the model, identical to the folder in methods_comparison
```

The framework is standalone: everything it needs is a physical copy under `dependency/` and
`problems/`. Nothing is imported from `methods_comparison`.
