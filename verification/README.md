# Verification of the repository

`verify_all.bat`, at the root of the repository, runs every experiment of the thesis in a toy
configuration, to show that every code runs without error on the machine. The numbers it produces
mean nothing and are not to be compared with the manuscript.

## Running it

Double-click `verify_all.bat`, or from a terminal at the root of the repository:

```bat
verify_all.bat
verify_all.bat --only "appendix d"
verify_all.bat --only "chapter 3,chapter 4"
```

`--only` runs the steps whose name contains one of the texts. On a laptop with 4 cores the whole
verification takes about two hours, most of it in the quick campaign of Appendix A. Run nothing else
on the machine meanwhile.

## What it checks, in order

1. **Environment**: Python, the packages of `requirements.txt`, cores, memory, disk, Gurobi licence.
   When a package is missing, the verification stops and prints the installation command.
2. **Files**: the key files are there, and every Python file compiles.
3. **Experiments**: `methods_comparison` (the eight studies, their duration estimate and the
   launcher), `det_vs_stoch` (Chapters 2, 3, 4 and 6), `surrogate_trainer` (Chapters 5 and 6), then
   the studies of Chapters 5 and 6 on the toy surrogates, `AnnexeD` and `AnnexeA`.

A step passes only if its process ends with exit code 0, its log contains the expected success
message and no error. The toy scripts also check the report of each study: every output its chapter
reports is there, and no output that the chapter does not report (for instance, no Nemenyi test in
Chapter 2, no attainment analysis in the metaheuristics comparisons). A failing step prints the
problem and the end of its log, and the verification goes on with the next steps. Nothing is skipped
silently: without Gurobi, the steps that need it fail.

## Result

On screen and in `verification/VERIFICATION_REPORT.txt`; full logs in `verification/logs/`.

The toy outputs go to a `toy` folder of each framework (`experiments/toy`, `runs/toy`,
`results/toy*`) and to `AnnexeA/results_quick`; each verification first removes those of the
previous one. Results of real runs are never touched. When the path of `AnnexeA` is too long, its
toy results go to the temporary folder of Windows (`cri_toy`): its audit needs a short results path.

## Toy configuration

No file of the repository is changed: the toy settings are applied at run time by the scripts of
this folder.

| Script | Settings |
|---|---|
| `toy_methods_comparison.py` | through `run_all`, as `run_all.bat --only <study>` runs it, the toy sizes given as `RUN["overrides"]`: one instance per size class (real sizes 50 to 200), one run per tuning combination, two values per tuning parameter, 3 runs per instance, population 10 to 30, 8 generations (260 for Appendix D, which `complements.py` needs), every physical core |
| `pilot_det_vs_stoch.py` | one instance per size class, one run per tuning combination, two values per parameter, population 20, 6 generations |
| `toy_surrogate.py` | 2 size classes, 3 instances per class, population 30, 8 generations, a search of 2 draws over 2 folds |
| AnnexeD | `experiment.py` and `run.bat` on 2 instances, at most 60 s per run |
| AnnexeA | the real `run_all.bat` with `config_quick.yaml` of this folder, the quick check of the theory and a faithfulness check on 2 instances and 3 runs |

The toy surrogates of Chapters 5 and 6 are copied into
`Frameworks/methods_comparison/problems/<chapter>/surrogate_models/` for this run only, to check the
chain from `surrogate_trainer` to `methods_comparison`; they are removed at the end. Models you have
deployed there are set aside during the run and put back afterwards.

## Compiled but not run

- `AnnexeA/tools/paper_artifacts.py`: needs the results of the full campaign.
- `AnnexeA/tools/hunsaker_alpha_table.py`: would rewrite a data file of the repository.
