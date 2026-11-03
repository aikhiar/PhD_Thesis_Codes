# Surrogate model trainer

Trains the regressors that replace the Monte Carlo objectives inside the search of the two
surrogate-assisted chapters, Chapters 5 and 6 of the thesis, and selects the model to deploy.

## What it does

1. **Training data**: each metaheuristic of the study, with fixed parameters, on the training
   instances of the problem; the logged schedules are labelled by the Monte Carlo objective.
2. **Training**: every candidate model per objective, with a split by instance and a hyperparameter
   search over folds of whole instances.
3. **Selection**: the deployed model of each objective, by the two-stage rule of `config.SELECTION`
   (accuracy practically equivalent to the most accurate candidate, then lowest scoring cost).
4. **Reports**, those of Chapters 5 and 6: the training data; per objective, the candidate models
   (cross-validation and validation errors, scoring cost, selection), the search spaces and retained
   hyperparameters, the Friedman, Wilcoxon and Cliff's delta tests against the most accurate model,
   the within-instance ranking (Kendall's tau), the predictions of the deployed model and the
   comparison of the candidates; the Pareto-dominance agreement of the deployed surrogates; the
   training and scoring times, the speedup over the Monte Carlo objective and the break-even point.
   Every table also as LaTeX.
5. **Export**: the deployed models and `model_card.json`, in
   `runs/surrogate_<code>_<key>/surrogate_export/best_models/`.

## Running it

```bash
python main.py                  # Chapter 5 then Chapter 6
python main.py chapter5         # one chapter
python main.py --estimate       # expected duration of the data generation
python main.py --resume runs/surrogate_chapter5_ab12cd34
```

To deploy, copy everything in `surrogate_export/best_models/` into
`Frameworks/methods_comparison/problems/<chapter>/surrogate_models/`. The models of Chapter 6 are
already there; those of Chapter 5 must be trained first.

## Layout

```
main.py                     the pipeline
dependency/
  config/                   candidate models, search, selection rule, figures
  training_data_generator/  the logged schedules and their labels
  trainer/                  training and validation
  multi_algorithms/         copy of methods_comparison's multi-objective solvers
  mono_algorithms/          copy of methods_comparison's mono-objective solvers
  reporting/                tables, figures, statistics
problems/
  problem_<chapter>.py      the training contract of the chapter
  <chapter>/                the model, identical to the folder in methods_comparison
```

The framework is standalone: its solvers are a physical copy of methods_comparison's, and nothing is
imported from methods_comparison.
