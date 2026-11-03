# theory

The verified implementation of Sections 3 and 4 of the paper, and its numerical verification.
The framework in `../cri/` has a faster version of the same Certify code.

- `verify_theory.py`: checks every statement of Sections 3 and 4 numerically, on a pool of small
  instances of three families (omega, staggered, overnight) with a proven optimum. Each check is
  tied to one statement and reports its number of tests and of failures; exact optima come from
  the class-aggregated model, and the closed forms are recomputed from the formulas of the paper,
  independently of `certificates.py`. The Monte Carlo checks are seeded and reproducible, and use
  z-scores of 4.5 standard errors (5 for the correlation tests) at levels where the normal
  approximation holds, or an exact binomial test where it does not, so that a false alarm is
  negligible over all the tests. This is step 3 of `../run_all.bat`.
- `bounds.py`: instances, the LP relaxation (vertex by primal simplex), exact slot laws, the
  non-localized bounds (Corollary 4.12), the localized closed forms, per-slot thresholds with the union
  bound and the asymmetric local lemma, the averaging bound, and derandomized rounding (method of
  conditional expectations).
- `certificates.py`: the full set of certificates and the algorithms that attain them:
  - union, componentwise union and asymmetric local lemma, with exact tails or with Hoeffding,
    Kearns-Saul, Bernstein and Bennett tails; averaging and componentwise averaging; closed forms;
    rounding to attainable loads;
  - componentwise derandomization, iterative LP rounding (P* <= P_LP + w_max), Moser-Tardos
    resampling;
  - the cover-cut lower bound (LB_cut) and the configuration LP (LB_conf, column generation with
    0/1-knapsack pricing);
  - dependent rounding by power class;
  - the exact dynamic program on the fractional support of an LP vertex (exact probability, exact
    expected peak, existence threshold and optimal rounding).
- `exact.py`: the exact class-aggregated MILP (integer class counts, one transportation problem per
  class), used as the reference solver.
- `validate.py`: the instance families and the validation suite that `verify_theory.py` builds on.

Needs numpy, pandas and gurobipy. Run from this folder:

```
python verify_theory.py [--quick] [--workers N] [--out DIR]
```

`--quick` uses a smaller pool (a few minutes); `--workers 1` runs sequentially, with the same
results. The pool keeps the instances that the exact model solves to proven optimality within its
time limit, so on a slower machine it can hold a few instances fewer, and the number of tests
varies slightly (for example 12,757,841 tests on 41 instances on a 4-core laptop, 12,758,086 on 44
on a 32-core machine); every check must still report 0 failures. The table of checks is written to `DIR/theory_verification.md` and `.csv` (default:
`results/` here; `run_all.bat` writes it to the campaign's `05_logs/`).
