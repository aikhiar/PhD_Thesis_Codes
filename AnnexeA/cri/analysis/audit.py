"""
Audit of a results folder (python -m cri audit <config>): is the campaign complete and valid, and is its
output complete, organised and consistent? Every check returns (area, check, ok, detail); the command
prints them, writes 05_logs/audit.md and exits with 1 if any check fails.

  runs        every planned run stored; statuses; feasibility; guarantees; certified lower bounds; budgets
  layout      the results folders (01_raw_runs ... 05_logs, README) and the report folders (sections and blocks)
  outputs     the files each analysis of each block must produce (tables, statistics, figures)
  files       every table as .csv/.md/.tex, readable, not empty, not all-NaN; every figure as a valid .png/.pdf
  report      report.md/.html present, no failed analysis, every link resolves, index.md lists every file
              (and only existing ones), unique LaTeX labels, file names without spaces, path lengths
  statistics  p-values in [0, 1], Holm p >= raw p, average ranks consistent, effect sizes in range
"""

import glob
import os
import re

import numpy as np
import pandas as pd

from ..core.results import DATA, LOGS, RAW, REPORT, ResultStore, results_dir
from ..core.runner import Runner, resolve_path, run_key
from .report import budget_folder

from .metrics import OVER_BUDGET_OK


def overrun_check(r, documented=None, traces=None):
    """(ok, detail) of the budget check: runs whose charged time exceeds the budget by more than 10% + 0.05 s.
    Accepted: the single-shot methods of OVER_BUDGET_OK; runs whose result did not improve after the budget
    + tolerance (anytime trace: the extra time changed nothing; after metrics.evaluate_at_budget this holds for
    every run with a schedule by then); the methods whose overruns are documented in the
    configuration (report.documented_overruns: method -> reason). Without traces every overrun counts as
    improved. The detail lists the unexpected overruns per block and method with the worst ratio."""
    from .metrics import budget_overruns
    documented = documented or {}
    over = budget_overruns(r, traces)
    cls = over["info_class"].fillna("") if "info_class" in over else pd.Series("", index=over.index)
    single = cls.isin(OVER_BUDGET_OK)
    no_gain = over[~single & ~over.gained]
    doc = over[~single & over.gained & over.method.isin(list(documented))]
    unexpected = over[~single & over.gained & ~over.method.isin(list(documented))]
    parts = []
    if len(unexpected):
        g = unexpected.groupby(["experiment", "method"]).agg(runs=("ratio", "size"), worst=("ratio", "max"))
        parts.append("unexpected (improved after the budget): " + "; ".join(
            f"{e}/{m}: {int(x.runs)} runs, worst {x.worst:.2f}x budget"
            for (e, m), x in g.sort_values("runs", ascending=False).iterrows()))
    if len(no_gain):
        parts.append(f"no gain after the budget: {no_gain.method.value_counts().to_dict()}")
    if len(doc):
        parts.append(f"documented overruns: {doc.method.value_counts().to_dict()}")
    expected = over[single]
    if len(expected):
        parts.append(f"expected single-shot overruns: {expected.method.value_counts().to_dict()}")
    return unexpected.empty, " | ".join(parts)


class Audit:
    def __init__(self, cfg):
        self.cfg = cfg
        self.out = resolve_path(cfg, cfg["run"].get("output", "results"))
        self.store = ResultStore(self.out)
        self.rep = results_dir(self.out, REPORT)
        self.rows = []

    def check(self, area, name, ok, detail=""):
        self.rows.append({"area": area, "check": name, "ok": bool(ok), "detail": str(detail)[:400]})
        return ok

    # ---------------------------------------------------------------- runs
    def runs(self):
        runner = Runner(self.cfg, log=lambda m: None)
        from ..core.runner import build_instances
        runner.sets = build_instances(self.cfg, runner.problem, runner.store, lambda m: None)
        missing = {}
        for name, ex, tasks in runner.plan():
            done = runner.completed(name)
            left = [t for t in tasks if run_key(t[1].name, t[2]["id"], t[3], t[4]) not in done]
            if left:
                missing[name] = len(left)
        self.check("runs", "every planned run is stored", not missing, missing or "all stored")
        from .metrics import charge_wall_time, evaluate_at_budget
        r = charge_wall_time(self.store.load_table("runs"))
        traces = self.store.load_table("traces") or {}
        if r is not None and len(r):
            evaluate_at_budget(r, traces)           # as in the report: credited with what was found by the budget
        if r is None or not len(r):
            self.check("runs", "aggregated runs table exists", False, f"{DATA}/runs.pkl missing")
            return None
        self.check("runs", "aggregated runs table exists", True, f"{len(r)} runs")
        bad = r[~r.status.isin(["ok", "not_available"])]
        self.check("runs", "every run ended with status ok (or not_available)", bad.empty,
                   bad.status.value_counts().to_dict() if len(bad) else "")
        has = r.objective.notna()
        inf = r[has & ~r.feasible.astype(bool)]
        self.check("runs", "every schedule is feasible", inf.empty, f"{len(inf)} infeasible" if len(inf) else
                   f"{int(has.sum())} schedules")
        g = r[r.guarantee.notna() & has]
        viol = g[g.objective > g.guarantee + 1e-6]
        self.check("runs", "every a priori guarantee holds", viol.empty, f"{len(viol)} violated of {len(g)}")
        lb = r[r.lower_bound.notna() & has]
        best = r[has].groupby("instance").objective.min()
        lviol = lb[lb.lower_bound > lb.instance.map(best) + 1e-6]
        self.check("runs", "no lower bound above a feasible value", lviol.empty, f"{len(lviol)} violations of {len(lb)}")
        self.check("runs", "no unexpected budget overrun (>10% + 0.05 s)",
                   *overrun_check(r, self.cfg.get("report", {}).get("documented_overruns"), traces))
        return r

    # ---------------------------------------------------------------- layout
    def layout(self):
        for kind in (RAW, DATA, REPORT, LOGS):
            d = results_dir(self.out, kind)
            self.check("layout", f"results folder {os.path.basename(d)}/", os.path.isdir(d), os.path.relpath(d, self.out))
        self.check("layout", "results README.md", os.path.exists(os.path.join(self.out, "README.md")))
        self.check("layout", "run log", os.path.exists(os.path.join(self.out, LOGS, "run_log.txt")))
        for f in ("report.md", "report.html", "index.md", "README.md"):
            self.check("layout", f"report {f}", os.path.exists(os.path.join(self.rep, f)))
        for sec in ("00_campaign_overview", "01_instances", "02_exact_references_and_bounds", "03_validity_checks"):
            self.check("layout", f"report section {sec}/", os.path.isdir(os.path.join(self.rep, sec)))
        r = self.store.load_table("runs")
        for lad in self.cfg.get("report", {}).get("budget_ladders", []) or []:
            if r is not None and sum(b in set(r.experiment) for b in lad.get("blocks", [])) >= 2:
                f = os.path.join(self.rep, "30_budget_ladders", lad["name"], "tables", "mean_gap_by_budget.csv")
                self.check("layout", f"budget ladder {lad['name']}", os.path.exists(f), os.path.relpath(f, self.rep))

    # ---------------------------------------------------------------- outputs expected per analysis
    def outputs(self, r):
        if r is None:
            return
        exps = list(self.cfg["experiments"].items())
        for i, (name, ex) in enumerate(exps):
            rb_all = r[r.experiment == name]
            for inc in ex.get("include", []):
                o = r[(r.experiment == inc["experiment"]) & r.method.isin(inc.get("methods", []))
                      & r.instance.isin(rb_all.instance.unique()) & r.budget.isin(rb_all.budget.unique())]
                rb_all = pd.concat([rb_all, o], ignore_index=True)
            if rb_all.empty:
                continue
            block = os.path.join(self.rep, f"{10 + i:02d}_{name}")
            if not self.check("outputs", f"{name}: block folder", os.path.isdir(block), os.path.relpath(block, self.out)):
                continue
            an = ex.get("analyses", ["comparison"])
            need = []
            if any(a in an for a in ("comparison", "sota")):
                cmp_r = rb_all[(rb_all.status != "not_available") & (rb_all.get("info_class", "") != "CERTIFY")]
                for b, g in cmp_r.groupby("budget", dropna=False):
                    bf = os.path.join(block, budget_folder(b))
                    need += [os.path.join(bf, "tables", f) for f in ("summary.csv", "all_quality_measures.csv",
                                                                     "charged_time.csv", "per_instance_mean_peak.csv")]
                    need += [os.path.join(bf, "figures", f) for f in ("quality_vs_time.png", "charged_time_boxplot.png")]
                    per = g.groupby(["instance", "method"]).objective.mean().unstack()
                    k, n = per.shape[1], len(per.dropna())
                    if k >= 2:
                        need += [os.path.join(bf, "figures", "performance_profile.png"),
                                 os.path.join(bf, "figures", "gap_boxplot.png"),
                                 os.path.join(bf, "statistics", "pairwise_wilcoxon_holm_effect_sizes.csv")]
                        if n >= 1:
                            need.append(os.path.join(bf, "statistics", "win_matrix.csv"))
                    if k >= 3 and n >= 2:
                        need += [os.path.join(bf, "statistics", f) for f in (
                            "friedman_iman_davenport_test.csv", "friedman_average_ranks.csv",
                            "nemenyi_critical_difference_diagram.png", "holm_pvalue_heatmap.png")]
                    if ex.get("control") and ex["control"] in per and k >= 2:
                        need.append(os.path.join(bf, "statistics", f"{ex['control']}_vs_each_method.csv"))
            if any(a in an for a in ("comparison", "sota")) and rb_all.budget.dropna().nunique() >= 2:
                need += [os.path.join(block, "across_budgets", "tables", "mean_gap_by_budget.csv"),
                         os.path.join(block, "across_budgets", "figures", "gap_vs_budget.png")]
            if "anytime" in an:
                for b in rb_all.budget.dropna().unique():
                    if b:
                        bf = os.path.join(block, budget_folder(b))
                        need += [os.path.join(bf, "figures", "anytime_median_gap_all.png"),
                                 os.path.join(bf, "tables", "time_to_target_and_primal_integral.csv")]
                        if rb_all[rb_all.budget == b].method.str.startswith("GUROBI").any():
                            need.append(os.path.join(bf, "figures", "gurobi_dual_bound_over_time.png"))
            if "certificates" in an:
                cf = os.path.join(block, "certificates")
                need += [os.path.join(cf, "tables", "all_certificates_vs_optimum.csv"),
                         os.path.join(cf, "tables", "a_priori_optimal_instances.csv"),
                         os.path.join(cf, "figures", "upper_certificates_tightness.png"),
                         os.path.join(cf, "figures", "lower_certificates_tightness.png")]
            if "head_to_head" in an and ex.get("pairs"):
                need.append(os.path.join(block, "head_to_head", "tables", "pairs_wilcoxon_holm.csv"))
            if "scalability" in an:
                need += [os.path.join(block, "scalability", "tables", "median_time_by_n.csv"),
                         os.path.join(block, "scalability", "figures", "time_vs_n.png")]
            if "gurobi" in an and rb_all.method.str.startswith("GUROBI").any():
                gf = os.path.join(block, "gurobi_status_and_bounds", "tables")
                need += [os.path.join(gf, "termination_status.csv"),
                         os.path.join(gf, "per_instance_bounds_and_gaps.csv")]
            if "tuning" in an:
                need.append(os.path.join(block, "tuning", "tables"))
            if "managerial" in an:
                need.append(os.path.join(block, "managerial", "tables", "peak_reduction_vs_reference_policies.csv"))
                if ex.get("store_solutions"):
                    need.append(os.path.join(block, "managerial", "figures", "load_profile_example_0.png"))
            miss = [os.path.relpath(p, self.rep) for p in need if not os.path.exists(p)]
            self.check("outputs", f"{name}: {len(need)} expected outputs of {', '.join(an)}", not miss,
                       "missing: " + ", ".join(miss[:8]) if miss else "all present")

    # ---------------------------------------------------------------- files
    def files(self):
        csvs = glob.glob(os.path.join(self.rep, "**", "*.csv"), recursive=True)
        bad_triplet, unreadable, empty, all_nan = [], [], [], []
        for c in csvs:
            stem = c[:-4]
            if not (os.path.exists(stem + ".md") and os.path.exists(stem + ".tex")):
                bad_triplet.append(os.path.relpath(c, self.rep))
            try:
                df = pd.read_csv(c)
            except Exception as e:
                unreadable.append(f"{os.path.relpath(c, self.rep)} ({e!r})")
                continue
            if df.empty:
                empty.append(os.path.relpath(c, self.rep))
            else:
                num = df.select_dtypes("number")
                if num.shape[1] and num.isna().all().all():
                    all_nan.append(os.path.relpath(c, self.rep))
        self.check("files", f"{len(csvs)} tables exist as .csv, .md and .tex", not bad_triplet, bad_triplet[:5])
        self.check("files", "every table is readable", not unreadable, unreadable[:3])
        self.check("files", "no empty table", not empty, empty[:5])
        self.check("files", "no table whose numbers are all missing", not all_nan, all_nan[:5])
        pngs = glob.glob(os.path.join(self.rep, "**", "*.png"), recursive=True)
        bad_png, no_pdf = [], []
        for p_ in pngs:
            with open(p_, "rb") as f:
                head = f.read(8)
            if head != b"\x89PNG\r\n\x1a\n" or os.path.getsize(p_) < 1000:
                bad_png.append(os.path.relpath(p_, self.rep))
            if not os.path.exists(p_[:-4] + ".pdf"):
                no_pdf.append(os.path.relpath(p_, self.rep))
        self.check("files", f"{len(pngs)} figures are valid PNG files", not bad_png, bad_png[:5])
        self.check("files", "every figure also exists as PDF", not no_pdf, no_pdf[:5])
        return csvs, pngs

    # ---------------------------------------------------------------- report
    def report(self, csvs, pngs):
        md_p = os.path.join(self.rep, "report.md")
        if not os.path.exists(md_p):
            return
        md = open(md_p, encoding="utf-8").read()
        failed = re.findall(r"analysis `([^`]+)` failed", md)
        self.check("report", "no analysis failed", not failed, failed[:5])
        links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md) + re.findall(r"\(`([^`]+\.csv)`", md)
        dead = [l_ for l_ in links if not os.path.exists(os.path.join(self.rep, l_))]
        self.check("report", f"{len(links)} links of report.md resolve", not dead, dead[:5])
        idx = open(os.path.join(self.rep, "index.md"), encoding="utf-8").read() if os.path.exists(
            os.path.join(self.rep, "index.md")) else ""
        listed = set(re.findall(r"\| (\S+\.(?:csv|png)) \|", idx))
        on_disk = {os.path.relpath(p, self.rep).replace(os.sep, "/") for p in csvs + pngs}
        self.check("report", "index.md lists every table and figure", on_disk <= listed,
                   sorted(on_disk - listed)[:5])
        self.check("report", "index.md lists only existing files", listed <= on_disk, sorted(listed - on_disk)[:5])
        labels = {}
        for t in glob.glob(os.path.join(self.rep, "**", "*.tex"), recursive=True):
            for lab in re.findall(r"\\label\{([^}]+)\}", open(t, encoding="utf-8").read()):
                labels.setdefault(lab, []).append(t)
        dup = [k for k, v in labels.items() if len(v) > 1]
        self.check("report", f"{len(labels)} LaTeX labels are unique", not dup, dup[:5])
        allf = glob.glob(os.path.join(self.out, "**", "*"), recursive=True)
        spaced = [os.path.relpath(p, self.out) for p in allf if " " in os.path.basename(p)]
        self.check("report", "file names without spaces", not spaced, spaced[:5])
        rel_longest = max((len(os.path.relpath(p, self.out)) for p in allf), default=0)
        self.check("report", "paths inside the results folder at most 150 characters (fits the Windows limit of 260 "
                   "with a results folder path of up to 109 characters)", rel_longest <= 150, f"longest {rel_longest}")
        longest = max((len(os.path.abspath(p)) for p in allf), default=0)
        self.check("report", "absolute paths shorter than 260 characters where the results are", longest < 260,
                   f"longest {longest}; the results folder path has {len(os.path.abspath(self.out))} characters"
                   + ("; move the results folder higher" if longest >= 260 else ""))

    # ---------------------------------------------------------------- statistics
    def statistics(self):
        probs = []
        for c in glob.glob(os.path.join(self.rep, "**", "statistics", "*.csv"), recursive=True):
            df = pd.read_csv(c)
            rel = os.path.relpath(c, self.rep)
            for col in [k for k in df.columns if k == "p" or k.startswith("p_") or k.startswith("p (")
                        or k.endswith(" p")]:
                v = pd.to_numeric(df[col], errors="coerce").dropna()
                if ((v < -1e-12) | (v > 1 + 1e-12)).any():
                    probs.append(f"{rel}: {col} outside [0, 1]")
            if "p" in df and "p_holm" in df:
                d = pd.to_numeric(df.p_holm, errors="coerce") - pd.to_numeric(df.p, errors="coerce")
                if (d < -1e-9).any():
                    probs.append(f"{rel}: Holm p below the raw p")
            if "cliff_delta" in df:
                v = pd.to_numeric(df.cliff_delta, errors="coerce").dropna()
                if ((v < -1 - 1e-9) | (v > 1 + 1e-9)).any():
                    probs.append(f"{rel}: Cliff's delta outside [-1, 1]")
            if "A12" in df:
                v = pd.to_numeric(df.A12, errors="coerce").dropna()
                if ((v < -1e-9) | (v > 1 + 1e-9)).any():
                    probs.append(f"{rel}: A12 outside [0, 1]")
            if os.path.basename(c) == "friedman_average_ranks.csv" and "average rank" in df:
                k = len(df)
                v = df["average rank"].astype(float)
                if (v < 1 - 1e-9).any() or (v > k + 1e-9).any() or abs(v.mean() - (k + 1) / 2) > 1e-6:
                    probs.append(f"{rel}: average ranks not in [1, k] or not averaging (k+1)/2")
        self.check("statistics", "p-values, Holm adjustment, ranks and effect sizes are consistent", not probs,
                   probs[:5])

    def run(self):
        r = self.runs()
        self.layout()
        self.outputs(r)
        csvs, pngs = self.files()
        self.report(csvs, pngs)
        self.statistics()
        df = pd.DataFrame(self.rows)
        os.makedirs(os.path.join(self.out, LOGS), exist_ok=True)
        cid, p = "", os.path.join(self.out, LOGS, "campaign_id.txt")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                cid = f.read().strip()
        # the details name paths relative to the results folder, so the audit can be shared as it is
        lines = ["# Audit of the results folder" + (f" of campaign {cid}" if cid else ""), "",
                 f"{int(df.ok.sum())} of {len(df)} checks pass.", "",
                 "| area | check | result | detail |", "|---|---|---|---|"]
        for row in self.rows:
            lines.append(f"| {row['area']} | {row['check']} | {'PASS' if row['ok'] else 'FAIL'} | "
                         f"{row['detail'].replace('|', '/')} |")
        with open(os.path.join(self.out, LOGS, "audit.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return df


def audit(cfg, verbose=True):
    df = Audit(cfg).run()
    if verbose:
        for _, row in df.iterrows():
            print(f"[{'PASS' if row.ok else 'FAIL'}] {row.area:10s} {row.check}" + (f"  -- {row.detail}" if
                                                                                  row.detail and not row.ok else ""))
        print(f"\naudit: {int(df.ok.sum())} of {len(df)} checks pass")
    return df
