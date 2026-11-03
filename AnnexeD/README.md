# AnnexeD: the exact method of Appendix D (standalone framework)

Appendix D of the thesis (paper J07, version with CS and GWO): EV charging
scheduling with heterogeneous discrete-level chargers. This folder holds the
exact unit-flow method (UF) against Gurobi on the time-indexed formulations, and
the complements of the methods_comparison study. It imports nothing
from methods_comparison: it carries its own physical copy of the problem model.

```
AnnexeD/
├── problems/
│   ├── appendix_d/          the model folder, IDENTICAL to Frameworks/methods_comparison/problems/appendix_d/
│   │   ├── parameters.py    instances, operators, exact-method settings
│   │   ├── model.py         instances, njit operators, objective, verifier
│   │   ├── unit_flow.py     the unit-flow formulation (UF), Gurobi
│   │   └── acn_sessions.pkl ACN-Data claimed sessions
├── covering.py, milp.py     the reference formulations (C) and (P)
├── experiment.py            experiment 1: UF against (C) and (P), resumable -> results/<name>/
├── analysis.py              tables, statistical tests and figures of experiment.py
├── complements.py           complements of the methods_comparison study (experiment 2)
├── run.bat                  run.bat | run.bat --report | run.bat --name X ...
├── HOW_TO_RUN.txt           the two experiments of Appendix D, step by step
└── requirements.txt         the versions on which every test passed
```

The metaheuristics (GA, ES, PSO, GWO, CS) and the exact method as a sixth method
are experiment 2, in `Frameworks/methods_comparison` (`problems/problem_appendix_d.py`).

Requirements: Python with numpy, scipy, numba, pandas, matplotlib and gurobipy
(a Gurobi licence; the full experiment needs an unrestricted one).

```
run.bat                                           # experiment 1 -> results\appendix_d_exact_methods
python complements.py <methods_comparison study folder>
```

The model folder `problems/appendix_d/` is shared with `methods_comparison` by COPY:
change a file in one framework, then copy the folder to the other (both must stay
byte-identical). Results are written under `results/` (not part of the code).
