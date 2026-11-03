# Preemptive Peak Power Scheduling with A Priori Certificates

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Solver: Gurobi](https://img.shields.io/badge/solver-Gurobi-red.svg)](https://www.gurobi.com/)

> **Thesis note.** This folder is Appendix A of the thesis (paper J10). It is the code of the
> repository without its results: `paper_results/` and every `results/` folder are left out, so the
> sections below that mention `paper_results/` apply to the public repository only.

Code of the paper *Preemptive Peak Power Scheduling with A Priori Certificates: Bounds, Certified
Algorithms and Large-Scale Experiments*, by A. Khiar, M. E. A. Brahmia, A. Oulamara, M. Golabi,
A. Azerine and L. Idoumghar (submitted to *Omega*).

The repository holds everything behind the computational study of the paper:

- the **CRI framework** (Certify-Round-Improve): the a priori certificates on the optimal peak, the
  certified roundings, the local searches, the exact models, and every baseline of the paper;
- the **numerical verification of the theory**: every statement of Sections 3 and 4 checked on
  instances with a proven optimum (`theory/`);
- the **experiments**: one configuration file drives the instances, the methods, their parameters
  and tuning, and every experiment block;
- the **report**: tables, figures and statistical tests, rebuilt from the raw runs, and the tables
  and figures of Section 6 of the paper;
- the **results of the paper's campaign** (`paper_results/`): the raw runs of all 41,801 runs, the
  full report, the audit and the theory verification, from which every table and figure of the
  paper is rebuilt.

## Quick start (Windows)

1. Get the code, with `git clone https://github.com/aikhiar/ppmsp-certificates.git` or the
   *Download ZIP* button, into a local folder outside any synchronized folder (see
   [Running a campaign on another machine](#running-a-campaign-on-another-machine)). Install
   Python 3.10 or later, then, in that folder: `pip install -r requirements.txt`.
2. Install a Gurobi licence (academic licences are free). The relaxations, the exact models and
   the Gurobi baselines need it.
3. Double-click **`run_all.bat`**: the full campaign of the paper into `results/` (about 7 hours
   on 32 cores). Open `results/03_report/report.html` at the end.

`run_all.bat` is the only launcher. Extra arguments go to the campaign, for example
`run_all.bat --workers 8`. On Linux or macOS, run the same steps
with the commands of [Command line](#command-line).

## What `run_all.bat` does

| step | what | time | once per machine |
|---|---|---|---|
| 1/4 | numerical verification of the theory (`theory/verify_theory.py`); stops if any test fails | 15 min to 1 h | yes |
| 2/4 | faithfulness check against the original code of the earlier methods (`faithfulness/check.py`) | ~10 min | yes |
| 3/4 | the campaign: every block of `config.yaml`, resumed from the stored runs | hours | resumes |
| 4/4 | the report, rebuilt from the raw pkl files (`results/03_report/report.html`), then its audit (`results/05_logs/audit.md`) | ~3 min | every time |

- The campaign does not start if a verification step fails.
- A step that passes writes a marker (`results/05_logs/step_*.txt`) and is skipped the next time.
  Delete a marker to repeat its step.
- If anything stops the run (licence, disk space, a restart), run `run_all.bat` again: finished
  steps and stored runs are skipped, so nothing is redone.
- A results folder without step markers or stored runs is a clean run from A to Z: every step runs,
  and the campaign gets its own, empty context cache.
- Settings can be changed with environment variables; the full run is used when they are not
  set: `CFG`, `OUT`, `PY`, `THEORY_ARGS` and `FAITH_ARGS`.
- Keep the console window open while it runs: closing it stops the campaign.

## Command line

From this folder (Windows, Linux or macOS):

```
cd theory && python verify_theory.py --out ../results/05_logs && cd ..   # step 1
python -m faithfulness.check --instances 6 --runs 20 --out results/04_faithfulness_check   # step 2
python -m cri all config.yaml                        # steps 3 and 4: campaign + report
python -m cri audit config.yaml                      # audit of the results folder

python -m cri run config.yaml --only E4a_equal_budgets,E3_long_runs
python -m cri report config.yaml                     # rebuild the data tables and the report
python -m cri plan config.yaml                       # tasks, budget-hours and workers per block
python -m cri list                                   # problems, families, methods
python -m cri forget config.yaml --only E2b_head_to_head_budgets --methods CRI-R   # recompute some runs
python -m cri all config.yaml --workers 8            # fix the number of workers (default: automatic)
python -m cri all config.yaml --cache D:/cri_cache   # put the context cache elsewhere (default: run.cache)
```

## Repository layout

```
run_all.bat          everything end to end (Windows)
config.yaml          the campaign of the paper: instances, methods, parameters, blocks, report
requirements.txt     Python packages (tested versions)
LICENSE              MIT
CITATION.cff         how to cite (GitHub's "Cite this repository")
paper_results/       the results of the paper's campaign: raw runs, report, audit, logs (README there)
cri/
  core/        problem-agnostic machinery
    config.py      YAML with inheritance (base: other.yaml) and deep merge (__replace__: true replaces a dict)
    registry.py    registries of problems and methods; make_method(spec)
    results.py     RunResult record, Deadline (budget with prepaid time), ResultStore (pickles)
    runner.py      instances -> exact references -> blocks (parallel, resumable) -> tuning -> aggregation
    resources.py   cores (affinity, container quota), RAM and the machine's environment -> workers per block
  analysis/    problem-agnostic analysis
    metrics.py     LB*, BKS, OPT per instance; gaps, hits, time to target, primal integral, anytime curves
    stats.py       Friedman + Iman-Davenport, Nemenyi CD, Wilcoxon + Holm/BH, sign test, Cliff's delta,
                   Vargha-Delaney A12, rank-biserial r, bootstrap CIs, win/tie/loss
    tables.py      Markdown / LaTeX (booktabs) / CSV writers, standard summaries
    figures.py     anytime curves, performance profiles, boxplots, CD diagrams, p-value heatmaps,
                   scaling plots, certificate tightness, load profiles, solver progress
    report.py      the report (report.md, report.html, one folder per section and block)
    audit.py       the audit of a results folder (python -m cri audit)
  problems/ppmsp/  the PPMSP plug-in
    instance.py    Instance (powers kept exactly in 0.1 kW units)
    generators.py  families omega, staggered, overnight, pcmax, single, csv
    kernels.py     Numba kernels of every method of the paper
    laws.py, certificates.py   slot laws, LP, every lower and upper certificate of Sections 3-4
    context.py     Certify artifacts of an instance, computed once, timed, cached, charged
    exact.py       Gurobi on the standard and aggregated models (anytime traces, bounds, MIP gaps)
    methods.py     G, RR, RLP, SA (+hybrids), CE, IR, MT, LLS, WLNS, CRI, GUROBI, ASAP, GNP, CERTIFY
    sota.py        baselines from the literature: BK, NFDH, ZAIDI_H, ZAIDI_ILS, HUNSAKER, AFSHAR_BB, AN_HYBRID
theory/        the verified implementation of Sections 3-4 and its numerical verification
  verify_theory.py   every statement of Sections 3-4 checked numerically (step 1 of run_all.bat)
  bounds.py, certificates.py, exact.py, validate.py   what it relies on
faithfulness/
  check.py             earlier methods against their original code (reference_old, verbatim)
tools/
  paper_artifacts.py   the tables and figures of Section 6 of the paper, from a results folder
  hunsaker_alpha_table.py   the table of Hunsaker et al.'s optimal online multiples (cri/problems/ppmsp/data)
```

## The methods

| id | what it is | paper |
|---|---|---|
| G, RR, RLP, SA | earlier methods, faithful re-implementations, original stopping rules | Algorithms 3-7 |
| HSG, HSR, HSL | SA started from G, from RR (N = 1000) and from RLP (5 trials): the former hybrids | Sec. 5 |
| G-b, RR-b, RLP-b, SA-b, HS*-b | the same methods using the whole budget (restarts, trials, re-heating) | equal-budget comparisons |
| CE-{lp,cut,conf}-{excess,union,comp} | derandomized rounding at α*, α_cut, α_conf | Algorithm 1, Thm 4.25(b) |
| IR | iterative LP rounding, peak ≤ P_LP + w_max | Thm 4.21 |
| MT-{lp,cut,conf}(-b) | Moser-Tardos resampling at U_LLL | Algorithm 2, Thm 4.25(a) |
| LLS, WLNS | load-leveling iterated local search; window large-neighborhood search on the aggregated model | Algorithms 8-9 |
| CRI, CRI+C, CRI-* | Certify-Round-Improve (+ exact completion) and its ablations | Algorithm 10 |
| GUROBI-std / -agg (-4t) | Gurobi, standard model (1)-(6) / aggregated model (Thm 3.3), 1 or 4 threads | |
| ASAP, GNP | uncontrolled charging; greedy non-preemptive charging | Sec. 6.8 |
| BK | Burgess-Killebrew sum-of-squares leveling (classical), adapted to preemption | Sec. 6.6 |
| NFDH | Next-Fit Decreasing Height, preemptive (Ranjan et al. 2014); exact on a common window, window-aware adaptation otherwise | Sec. 6.6 |
| ZAIDI_H, ZAIDI_ILS(-b) | heuristic and iterated local search (SA local search, their Table 4) of Zaidi et al. (2024), fixed charger assignment; -b restarts until the budget | Sec. 6.6 |
| HUNSAKER | optimal online α-policy of Hunsaker et al. (2003), α_T from their LP (table in `cri/problems/ppmsp/data/`), integral EDF | Sec. 6.6 |
| AFSHAR_BB | depth-first branch and bound of Afshar-Nadjafi et al. (2013), peak objective; exact when it completes | Sec. 6.6 |
| AN_HYBRID, AN-PAR, AN-dev | hybrid metaheuristics of Aristotelous and Nearchou (2024): SER/LS2, PAR/LS2, SER/LS2 with their leveling objective | Sec. 6.6 |
| CERTIFY | records every lower and upper certificate, with costs | Secs. 3-4, 6.2 |

All methods of the paper run on Numba-compiled kernels. Gurobi is a commercial solver. Every method
runs single-threaded, except the `-4t` Gurobi variants.

## Fair time accounting

- Each instance has a **context**: the LP vertex, LB_cut, LB_conf (time-limited), the certificate
  objects and iterative rounding. Each artifact is computed once, records its own time and is
  cached on disk.
- A method that uses an artifact is **charged** its time, together with the time of everything
  the artifact depends on, each counted once. The method's clock starts with that time already
  spent, so a method built on the LP pays for the LP inside its budget.
- Anytime traces start at the charged time. A method has no curve before its first schedule
  exists.
- **CRI's Certify stage is time-limited.** The steps run in order within half the budget
  (`certify_share`):
  1. the relaxation;
  2. the cover-cut bound and α_cut;
  3. iterative rounding;
  4. the configuration LP and α_conf.

  A step that cannot complete within the rest of the share is abandoned. Its time is lost and
  the later steps are skipped. Without the relaxation, CRI uses greedy rounding and the
  elementary bound. Every step is deterministic, computed once per instance and timed, so the
  rule is applied exactly to the recorded times. Every second is charged, abandoned steps
  included.
- Each worker loads every Numba kernel and opens Gurobi before its first measured run, so no run
  pays for compilation (`run.warmup: true`).
- **Sleep.** Budgets are wall-clock, so the machine must not sleep during a campaign. The
  runner asks Windows to block idle sleep while it runs (`run.keep_awake: true`, released when
  the process exits). A closed lid or a manual sleep still suspends it. A run whose wall time
  exceeds twice its budget + 30 s is stored as `suspect_timing`; it is re-run on the next resume
  and replaced in the tables.
- Gurobi gets the same wall-clock budget as its time limit, which includes building the model.
  It is charged its whole wall time: building, solve, and the read-out of the schedule (one LP
  for the aggregated model). With `stop_at_lb: false` (the default), no method is told the lower
  bound unless it computes the bound itself, as CRI does.
- **CRI's completion stage (CRI+C)** gives Gurobi 90% of the remaining time. It stops building
  the model at that deadline and reads the schedule only if Gurobi improved on CRI's.
- **Every run is evaluated at its budget.** A run is credited only with what it had found by
  its budget + 10% + 0.05 s (`metrics.evaluate_at_budget`, applied by the report and the audit).
  A run that improved only later, in an indivisible last step at the shortest budgets, gets the
  incumbent it had at that time; its trace is cut there. `objective_final` keeps the value it
  ended with, and `evaluated_at_budget` marks it. A run with no schedule by then keeps its first
  one.
- **Budget overruns.** A run overruns when its charged time exceeds the budget by more than
  10% + 0.05 s. The report lists every overrun per block, method and budget, with its reason
  (`budget_overruns` table, `03_validity_checks`), and for each one the number of runs whose
  credited result still comes after the budget, with no schedule by the budget, or evaluated at
  the budget. The audit accepts three kinds:
  - the single-shot methods (CE, MT, IR), whose certificate artifacts alone can exceed a
    1 s budget;
  - runs, of any method, whose credited result was found by the budget: the extra time changed
    neither the peak nor the anytime curves;
  - the methods listed in `report.documented_overruns` with their reason.

  Any other overrun fails the audit.

## Experiment blocks (`config.yaml`)

| block | content | paper |
|---|---|---|
| references | exact aggregated model (300 s) on small, omega_published, medium | Sec. 6.1 (LB*) |
| E0_sanity | every method once on a few small instances | |
| E1_certificates | every certificate against the optimum or best known peak: validity, tightness, cost, a priori optimality | Sec. 6.2 |
| E8a / E8b | tuning of SA in its original form (243-point grid) and of the anytime methods, on a separate tuning set; the chosen values are applied to the later blocks | Sec. 6.1 |
| E2_head_to_head | earlier methods against their certified counterparts with their own rules: RR vs MT, RLP vs CE/IR, G vs CE, SA vs hybrids (10 seeds) | Sec. 6.3 |
| E2b_head_to_head_budgets | the same pairs at equal budgets of 1 s and 10 s | Sec. 6.3 |
| E4a / E4b | every method and Gurobi (both models) at 1, 10 and 60 s; anytime curves | Sec. 6.4 |
| E3_long_runs | 600 s, CRI, CRI+C and Gurobi, incumbent and dual bound over time | Sec. 6.5 |
| E3x_long_runs_xlarge | the same 600 s runs on the xlarge instances (n = 5,000-10,000, T up to 288) | Sec. 6.5 |
| E6_scalability | up to n = 10,000, T = 288, at 120 s | Sec. 6.5 |
| E7_gurobi_threads | Gurobi with 4 threads against single-thread CRI | Sec. 6.5 |
| E10a / E10b | every baseline from the literature at 10 s (3 seeds) and the anytime ones at 60 s, against CRI and Gurobi (reuses the E4a/E4b runs) | Sec. 6.6 |
| E5 / E5b | ablation: stages, matrices, rounding families, SA initialization | Sec. 6.7 |
| E9_managerial | peak reduction against ASAP and non-preemptive GNP; load profiles | Sec. 6.8 |

## Runtime

`python -m cri plan config.yaml` prints the budget-hours of each block and the workers it will use
on the current machine.

- **Full campaign:** 74.6 budget-hours on one worker, counting only the methods that use their
  whole budget. E6 and E3x need 8 GB per worker and E7 4 cores per worker, so they get fewer
  workers; the rest divides by the number of workers. On top of that come the 130 exact
  references (up to 300 s each, divided by the workers), the single-run blocks (E1, E8a, E2), the
  Certify stages and the verification steps of `run_all.bat` (30 min to 1.5 h). The paper's
  campaign took about 7 hours on 32 cores.

The campaign resumes from what is stored. Delete `results/01_raw_runs/<block>` to recompute a block,
or `results/` to start over. Tuned parameters are stored in
`results/02_aggregated_data/tuned_defaults.json`.

## Parallel computing

- **Process level.** Each block runs in a pool of worker processes, started with spawn on every
  platform. A worker loads the Numba kernels, opens its own Gurobi environment and runs one job at
  a time: all the tasks of one instance in the block, so that the instance's context is computed
  or loaded once.
- **Job order.** Jobs are submitted longest first.
- **Job splitting (load balance).** When a block has fewer than `split_factor` (3) × workers jobs,
  the jobs of instances whose context is already cached are split, longest first. Jobs with an
  uncached context stay whole, so a context is never computed twice at once.
- **Threads.** BLAS/OpenMP thread pools are pinned to one thread per worker. Every Gurobi model of
  a worker uses one thread (`run.gurobi_threads: 1`): the LP relaxation, the cover-cut and
  configuration LPs, iterative rounding, schedule recovery and the W-LNS subproblems. Only the
  `-4t` variants raise it, to 4. With one thread, the LP vertex is also deterministic.
- **Windows** accepts at most 61 worker processes per pool; the count is capped there.
- **Number of workers** (`run.workers: auto`, `cri/core/resources.py`). At the start of each
  block, the runner reads the cores (with the CPU affinity and any container quota) and the free
  RAM, then takes

  ```
  workers = min( (physical cores - reserve_cores) // threads per task,
                 (available RAM - reserve_mem_gb) // mem_gb_per_worker,
                 max_workers )
  ```

  Physical cores, because hyperthreads would distort the wall-clock budgets; on most virtual
  machines the reported cores are the vCPUs. `mem_gb_per_worker` is a conservative estimate per
  block: 1-2 GB for most blocks, 3 GB for E3 and E7, 8 GB for E6. Fix the count with
  `--workers N` or `run.workers: N`, or per block with `workers:`.
- **Compilation.** The master compiles the Numba kernels once before the first pool starts, so
  the workers only load them from the cache.
- **Every step of `run_all.bat` uses the cores:** the theory verification with a process pool
  (`--workers 1` runs the original sequential code, with the same results), the faithfulness check
  with one process per instance, and the campaign with the pools above.

## Running a campaign on another machine

1. Copy this folder to a local folder **outside any synchronized folder** (OneDrive, Dropbox),
   for example `C:/runs/`. Synchronization can lock files while the runner writes them, and it
   fills the disk. For a fresh campaign, leave out any `results/` folder.
2. Do **not** copy the context cache (`~/.cri_framework_cache/ppmsp`). It holds the Certify
   artifacts with their measured times, which are charged to the methods, so they must be
   measured on the machine itself. Each results folder has its own cache,
   `~/.cri_framework_cache/ppmsp/<campaign id>/`, with the id in `results/05_logs/campaign_id.txt`;
   old cache folders can be deleted once their campaign is finished.
3. Install Python and the requirements, and the Gurobi licence. Each worker opens one Gurobi
   environment; with a WLS licence, check that the licence allows that many sessions at once and
   keep the machine online. Before each block, and before each retry, the runner checks that an
   environment starts. If not, it waits, trying every `licence_poll_s` (60 s) for up to
   `licence_wait_min` (30 min), then stops cleanly with a resume message. Runs that failed because
   the licence was lost mid-solve are retried once it is back.
4. Pause Windows Update for the duration of the campaign, so the machine does not restart in the
   middle. If it restarts anyway, run `run_all.bat` again.
5. Keep at least 10 GB free on the disk. The runner stops cleanly below `run.min_free_gb`.
6. Over Remote Desktop, **disconnect** (close the window): the campaign keeps running. Do not sign
   out, and do not close the console window: both stop it. Follow the progress in
   `results/05_logs/run_log.txt`. On Linux over SSH, use `nohup` or `tmux`.
7. Run `run_all.bat`: its first lines must say `Fresh run from A to Z`.

Each run records its machine (`info_host`), and the report has a table of runs per block and
machine. Every start of `run` also appends the machine's processor model, cores, RAM, OS and
software versions (Python, numpy, scipy, numba, pandas, Gurobi) to
`results/05_logs/environment.jsonl`; the report's overview names that campaign machine apart from
the machine that built the report. Compare times only within one machine: run every block entirely
on the same device.

## Outputs

Every folder name says what it holds; `results/README.md` and `03_report/README.md` (both generated)
explain them again.

```
results/
  README.md                         what each folder holds
  01_raw_runs/                      THE SOURCE OF TRUTH: one pickle per (block, instance) job
    <block>/<instance>__<job>.pkl     lists of RunResult (peak, charged time, trace, guarantee, bound, status)
    _references/<instance>.pkl        exact references       _contexts/  bound summaries per instance
  02_aggregated_data/               rebuilt from 01 by `report`: runs.pkl, runs_enriched.pkl/.csv (every gap
                                    measure, hits, time to target, primal integral), traces, bound_traces,
                                    solutions, references, contexts, instances, instance_features,
                                    instance_references (LB*, BKS, OPT), tuning_<block>, tuned_defaults.json
  03_report/
    report.html, report.md          the whole report (open first)
    index.md                        every table and figure with its caption
    README.md                       every folder below with its content
    00_campaign_overview/           status, machines, method library, defaults, headline results of every block
    01_instances/  02_exact_references_and_bounds/  03_validity_checks/
    04_faithfulness_of_earlier_methods/  05_theory_verification/
    10_E0_sanity/ 11_E1_certificates/ ... 26_E10b_state_of_the_art_60s/   one folder per block, config order
      budget_10s/ (or single_run/)
        tables/      summary (95% CIs), mean gap by family / size / n, charged time, all quality measures,
                     seed variability, per-instance peaks, time to target and primal integral
        statistics/  Friedman + Iman-Davenport test, average ranks (overall, by family), Nemenyi critical
                     difference diagram, pairwise Wilcoxon-Holm with sign test, Cliff's delta and A12,
                     Holm p-value heatmap, win matrix, control vs each method
        figures/     performance profile, gap and time box plots, quality vs time, gap vs n,
                     anytime curves (all, per family), Gurobi dual bound over time
      across_budgets/  (blocks with several budgets) mean gap and charged time by budget, gap vs budget
      certificates/ head_to_head/ scalability/ gurobi_status_and_bounds/ tuning/ managerial/
                     (when the block has that analysis), each with tables/ and figures/
    30_budget_ladders/<ladder>/     time-quality ladders across blocks (report.budget_ladders of the
                                    config: E4a -> E4b -> E3, 1 s to 600 s; E10a -> E10b) on common instances
  04_faithfulness_check/            original code vs re-implementation (step 4)
  05_logs/                          run_log.txt, config_used.json, audit.md, step markers of run_all.bat,
                                    campaign_id.txt (names this campaign's context cache),
                                    environment.jsonl (machine and software versions of every start of run),
                                    theory_verification.csv/.md (step 3)
```

Tables exist as `.csv`, `.md` and `.tex` (booktabs; unique labels `tab:<block>_b<budget>_<table>`),
figures as `.png` and `.pdf`.

**The raw pkl files are the only source of truth.** `python -m cri report` first rebuilds every
table of `02_aggregated_data/` from `01_raw_runs/` and the configuration (instances, tuning tables,
runs, traces, bound traces, solutions, references, context summaries), then writes the report. So a
report can be regenerated from `01_raw_runs/` alone. `tuned_defaults.json` is kept, because it
records what the later blocks used; the report says so if a recomputed tuning choice differs.

**Audit.** `python -m cri audit config.yaml` checks a results folder and writes `05_logs/audit.md`:
- every planned run stored, statuses, feasibility, guarantees, lower bounds, budget overruns;
- the folder layout;
- the files each analysis of each block must produce;
- every table readable and non-empty, every figure a valid PNG with its PDF;
- every link of the report, the index, unique LaTeX labels, path lengths;
- p-values, Holm adjustments, ranks and effect sizes in range.

`run_all.bat` runs it after the report.

The Certify artifacts of each instance (LP, lower bounds, certificates, with their measured costs)
are cached in `run.cache` (default `~/.cri_framework_cache/ppmsp`). Deleting the cache only costs
recomputation; keep it to reuse the same recorded costs across blocks.

## Statistics used in the report

- **Unit of analysis.** Each stochastic method is first averaged over its seeds on each instance,
  so blocks are instances.
- **Tests:** Friedman, with the tie correction and Iman-Davenport; Nemenyi critical difference
  (CD diagrams); pairwise Wilcoxon signed-rank (Pratt zeros), with Holm and Benjamini-Hochberg
  adjustment; exact sign test; comparison of a control against every other method, with Holm
  adjustment.
- **Effect sizes:** Cliff's delta with its magnitude, Vargha-Delaney A12, rank-biserial r.
- **Uncertainty:** bootstrap 95% confidence intervals of mean gaps.
- **Profiles and anytime measures:** Dolan-Moré performance profiles; time to target (gaps of 0,
  0.5, 1 and 5%); Berthold's primal integral.

## Tables and figures of the paper

`python tools/paper_artifacts.py <results folder> <paper folder> [--numbers FILE]` writes the 11
tables (`<paper>/tables/*.tex`) and 4 figures (`<paper>/figures/*.pdf`) of Section 6 of the paper
and of its Supplementary Material. It reads the campaign's data and the report's statistics
without changing them, and with `--numbers` it lists every number quoted in the text. For the
paper's own results, first rebuild their aggregated data (Results of the paper, below).

## Extending

- **A method from the literature.** Subclass `SotaMethod` in `cri/problems/ppmsp/sota.py`,
  decorate it with `@register_method("NAME")` and implement `solve(inst, budget, seed, deadline)`,
  which returns the 0/1 matrix, optionally with a trace. The wrapper times the run, checks
  feasibility and records the result. Add its id to the `methods:` library and to the E10 blocks;
  the report includes it from then on. A class with `implemented = False` is reported as not
  available, so a method can be listed before its code exists. Instance fields:
  - `inst.s`, `inst.e`: 0-based inclusive windows;
  - `inst.p`: durations;
  - `inst.w`: powers in kW, or `inst.wu` in units of 0.1 kW.
- **A new method.** Write a class with `run(inst, ctx, budget, seed, lower_bound)` returning
  the fields of `RunResult`, and decorate it with `@register_method("NAME")`. Add an id in the
  `methods:` library and use it in blocks.
- **Another problem.** Write a plug-in like `cri/problems/ppmsp/__init__.py`, with `build_set`,
  `make_env`, `make_context`, `evaluate`, `reference` and instance (de)serialization, plus its
  methods. Set `problem:` in the config. The runner, metrics, statistics and report are
  problem-agnostic, except the `certificates` analysis.
- **Real data.** Use the `csv` family (columns s, e, p, w; 1-based slots), for example with an
  instance built from ACN-Data: `{name: acn, family: csv, n: 0, T: 96, params: {path: data/acn.csv}}`.

## Faithfulness check

```
python -m faithfulness.check --instances 6 --runs 20
```

It writes `results/04_faithfulness_check/faithfulness.md`. For each instance and each
earlier method, it reports the mean peaks of the original code and of the re-implementation,
the Mann-Whitney p-value and the Holm-adjusted p-value. `RLP-sameLP` gives both implementations
the same LP solution, which isolates the rounding and repair logic from the choice of the
optimal vertex: CBC in the original code, Gurobi here.

## Results of the paper

The results of the paper's campaign are in `paper_results/` (its README describes them): the raw
runs, the full report (`paper_results/results/03_report/report.html`), the audit, the theory
verification, the faithfulness check and the logs. The aggregated pickles are not stored; the
report rebuilds them from the raw runs. To rebuild everything and the paper's tables and figures:

```
python -m cri report paper_results/report_config.yaml       # aggregated data and report
python -m cri audit  paper_results/report_config.yaml       # paper_results/results/05_logs/audit.md
python tools/paper_artifacts.py paper_results/results <paper folder> --numbers numbers.md
```

`run_all.bat` reproduces the campaign itself on any machine with a Gurobi licence; the measured
times, and therefore the results at short budgets, depend on the machine.

## Citation

If you use this code, please cite the paper (GitHub's "Cite this repository" button gives the same
reference from `CITATION.cff`):

```bibtex
@article{Khiar2026PeakPower,
  author  = {Khiar, Aimen and Brahmia, Mohamed El Amine and Oulamara, Ammar and Golabi, Mahmoud
             and Azerine, Abdennour and Idoumghar, Lhassane},
  title   = {Preemptive Peak Power Scheduling with A Priori Certificates: Bounds, Certified
             Algorithms and Large-Scale Experiments},
  journal = {Omega},
  year    = {2026},
  note    = {Submitted}
}
```

## License

The code is released under the MIT License (`LICENSE`). Gurobi is a commercial solver with its
own licence terms; it is not part of this repository.
