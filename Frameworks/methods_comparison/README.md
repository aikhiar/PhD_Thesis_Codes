# Comparison of the metaheuristics

Runs the comparisons of metaheuristics of the thesis: Chapters 2 to 6 and Appendices B, C and D.
For each study it generates the instances, tunes every method by grid search on every instance, runs
the evaluation, and writes the tables, statistical tests and figures that the chapter reports.

## The studies

| Study | Methods | Statistical tests of the chapter |
|---|---|---|
| `chapter2`, `chapter3`, `chapter4`, `appendix_c` | NSGA-II, NSGA-III, MOCS, MOPSO, MOGWO | Friedman, pairwise Wilcoxon (Bonferroni), Cliff's delta |
| `appendix_b` | GA, PSO, GWO, CS | Friedman, pairwise Wilcoxon (Bonferroni), Cliff's delta |
| `chapter5` | NSGA-II, NSGA-III, MOCS, MOPSO, MOGWO (surrogate-assisted) | Friedman and Kendall's W, pairwise Wilcoxon (Holm), Cliff's delta, Nemenyi, against the best method: Friedman post-hoc (Holm) and per-instance Mann-Whitney |
| `chapter6` | NSGA-II, NSGA-III, MOCS, MOPSO, MOGWO (surrogate-assisted) | Friedman and Kendall's W, pairwise Wilcoxon (Holm), Cliff's delta, per-instance Mann-Whitney against the best method |
| `appendix_d` | GA, ES, PSO, GWO, CSA, and the exact method | on the objective and the time: Friedman and Kendall's W, Nemenyi, pairwise Wilcoxon (Holm), against the best method: Friedman post-hoc and Wilcoxon (Holm, A12), Bayesian signed-rank, per-instance Mann-Whitney |

Every study also reports the objective (or hypervolume) and the execution time per instance and per
size class, the gap to the reference, the rank heatmap, the performance profiles, the quality-time
trade-off and the convergence curves; the multi-objective studies, the Pareto fronts and the IGD+,
additive epsilon and spacing indicators; Appendix D, its summary tables, the anytime analysis
against the exact optimum and the scaling of the time with the number of bookings.

## Running it

```bat
run_all.bat                      every study, in the order of the manuscript
run_all.bat --only chapter6      one study (several names allowed)
run_all.bat --estimate           expected duration of each study on this machine
```

Each study writes `experiments/<study>_metaheuristics_<hash>/`. The name is a hash of the settings
that decide the runs: rerunning the same study resumes from its checkpoints.

## Configuration

- `comparison_config.py`: the list of the studies and the protocol shared by all of them
  (instances, tuning, evaluation, parallelism, statistics, figures, the solvers and their grids).
- `problems/problem_<study>.py`: the binding of a study (its instances, its operators, its
  `problem_dict`) and `STUDY_CONFIG`, the settings in which the study departs from the shared
  protocol (population, generations, grids, and the tests of its chapter).
- `problems/<study>/`: the model and its parameters.

Precedence: `CONFIG` of `comparison_config.py`, then `STUDY_CONFIG`, then `RUN["overrides"][study]`.

## Layout

```
run_all.bat                 launcher
comparison_config.py        studies and protocol
methods_comparison/
  run_all.py                runs the studies
  core/                     the pipeline of a study (instances, tuning, evaluation, analysis)
  solvers/                  GA, ES, PSO, GWO, CS; NSGA-II, NSGA-III, MOCS, MOPSO, MOGWO
  analysis/                 scores, statistical tests, tables, indicators, anytime and scaling
  figures/                  figures
  utils/                    hypervolume, statistics, checkpoints, parallelism, validation
problems/
  problem_<study>.py        the binding of a study
  <study>/                  its model, parameters and data
```
