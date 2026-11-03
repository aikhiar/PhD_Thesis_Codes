# Electric Vehicle Charging Scheduling Under Uncertainty: code of the thesis

Code of the experiments of the PhD thesis of Aimen Khiar, *Electric Vehicle Charging Scheduling
Under Uncertainty: Stochastic Multi-Objective Models and Metaheuristic Solution Methods* (University
of Haute-Alsace, IRIMAS; CESI LINEACT). The repository holds code and input data only: no trained
model and no result. Every experiment of the thesis is run on your machine, from the training of the
surrogates to the last statistical test, and everything a run produces is written under the
framework that runs it.

```
ev-charging-thesis
├── README.md                this file
├── requirements.txt         every package, with the tested versions
├── verify_all.bat           end-to-end check of the repository, in toy configuration
├── verification/            the scripts of verify_all.bat
├── Frameworks
│   ├── methods_comparison   comparison of the metaheuristics, every study of the thesis
│   │   └── problems         chapter2 to chapter6, appendix_b, appendix_c, appendix_d
│   ├── det_vs_stoch         deterministic versus stochastic planning
│   │   └── problems         chapter2, chapter3, chapter4, chapter6
│   └── surrogate_trainer    surrogate models of the two surrogate-assisted chapters
│       └── problems         chapter5, chapter6
├── AnnexeA                  Appendix A: certificates and the CRI framework
└── AnnexeD                  Appendix D: exact unit-flow method
```

## Where each part of the manuscript is

| Manuscript | Metaheuristics (`methods_comparison`) | Det. vs stoch. (`det_vs_stoch`) | Surrogates (`surrogate_trainer`) | Own folder |
|---|---|---|---|---|
| Chapter 2 | `chapter2` | `chapter2` | | |
| Chapter 3 | `chapter3` | `chapter3` | | |
| Chapter 4 | `chapter4` | `chapter4` | | |
| Chapter 5 | `chapter5` | | `chapter5` | |
| Chapter 6 | `chapter6` | `chapter6` | `chapter6` | |
| Appendix A | | | | `AnnexeA` |
| Appendix B | `appendix_b` | | | |
| Appendix C | `appendix_c` | | | |
| Appendix D | `appendix_d` | | | `AnnexeD` |

Every study is one binding file and one folder, `problems/problem_<study>.py` and
`problems/<study>/` (`model.py`, `parameters.py`, the ACN-Data sessions). The folder of a study is
identical in every framework that uses it; in `methods_comparison`, Chapters 5 and 6 also have a
`surrogate_models/` folder, where the surrogates you train are deployed.

Each framework writes the analyses that the manuscript reports for the study, and no other: the
statistical tests, tables and figures of each chapter are set in its binding.

## Getting started

```bash
python -m pip install -r requirements.txt
```

On Windows, clone the repository into a short path, for instance `C:\repos\ev-charging-thesis`: its
deepest file lies 91 characters below the root, and Windows limits paths to 260 characters unless
long paths are enabled.

Gurobi with a licence is needed for `AnnexeA`, `AnnexeD` and the exact method of Appendix D;
without it, the rest works.

The frameworks install any missing package with pip at start-up. Set `NO_AUTO_INSTALL=1` to make
them stop with the list of what is missing instead.

## Checking the installation

```bat
verify_all.bat
```

It runs every experiment of the thesis in a toy configuration (a few small instances, short runs) and
reports, step by step, whether each one runs and writes the outputs of its chapter. The numbers it
produces mean nothing. About two hours on a 4-core laptop; see `verification/README.md`.

## Running the experiments, from A to Z

The steps below reproduce every experiment of the thesis. Steps 1 and 2 must come before the
studies of Chapters 5 and 6 in step 3; everything else can run in any order. The three frameworks
print their expected duration on the machine with `--estimate`.

### 1. Train the surrogates of Chapters 5 and 6

From `Frameworks/surrogate_trainer` (results in `runs/`):

```bash
python main.py --estimate
python main.py chapter5
python main.py chapter6
```

Each training generates the data (schedules of the metaheuristics, labelled by Monte Carlo), trains
and validates the seven candidate regressors, selects the deployed model of each objective by the
two-stage rule of the thesis (accuracy first, then scoring cost), and writes the reports of its
chapter. The selected models are exported to `runs/surrogate_<study>_<key>/surrogate_export/best_models/`
(the `.joblib` files and `model_card.json`).

The training is meant for a machine with many cores. On a 4-core laptop the data generation of
Chapter 5 takes about 4 hours, and the random forest search runs on one thread, an estimated 8 to 16
hours per objective.

### 2. Deploy them in the metaheuristics

Copy everything in each `best_models/` folder into
`Frameworks/methods_comparison/problems/<study>/surrogate_models/` (`chapter5` or `chapter6`). The
surrogate-assisted metaheuristics of that study then search on these models, and the final fronts are
re-evaluated by Monte Carlo.

Any regressor trained on the same features can be plugged in the same way: retrain with other
settings, or another model, export it under the file names of `SURROGATE_FILES` in
`problems/<study>/parameters.py`, and copy it there. At start-up the study checks the number of
features and compares the library versions recorded in `model_card.json` with those installed.
Without deployed models, the study stops at start-up with this instruction.

### 3. Compare the metaheuristics

From `Frameworks/methods_comparison` (protocol in `comparison_config.py`, results in `experiments/`):

```bat
run_all.bat --estimate
run_all.bat --only chapter2
run_all.bat
```

`run_all.bat` alone runs the eight studies (Chapters 2 to 6, Appendices B, C and D) one after the
other, resumes from its checkpoints when interrupted, and moves on to the next study if one fails.
The metaheuristics of Appendix D run here too (study `appendix_d`, with the exact method as a sixth
method).

### 4. Compare deterministic and stochastic planning

From `Frameworks/det_vs_stoch` (results in `runs/`), Chapters 2, 3, 4 and 6; no surrogate is needed:

```bash
python main.py --estimate
python main.py chapter2
python main.py
```

### 5. Appendices A and D

See `AnnexeA/README.md` and `AnnexeD/HOW_TO_RUN.txt`.

Copy `AnnexeA` to a short local path before a campaign, for instance `C:\runs\AnnexeA`: its audit
needs a results folder path of at most 109 characters, and a synchronized folder (OneDrive, Dropbox)
can lock files during a run.

## Data

The bookings derive from the claimed sessions of ACN-Data (Z. J. Lee, T. Li and S. H. Low,
"ACN-Data: Analysis and applications of an open EV charging dataset", Proceedings of the Tenth ACM
International Conference on Future Energy Systems, 2019, https://ev.caltech.edu/dataset), stored as
`acn_sessions.pkl`.
