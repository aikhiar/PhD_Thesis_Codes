"""
Report builder: reads the aggregated tables of a campaign and writes, in 03_report/:
  report.md, report.html   the whole report (open first)
  index.md                 every table and figure, with its folder and caption
  README.md                what each folder holds
  00_campaign_overview/ ... 05_theory_verification/   the sections common to every campaign
  NN_<block>/              one folder per experiment block, in the order of the configuration; inside,
                           budget_<B>s/ (or single_run/) and one folder per block-level analysis
                           (certificates/, head_to_head/, scalability/, gurobi_status_and_bounds/,
                           tuning/, managerial/), each with tables/ (.csv .md .tex), statistics/ (tests,
                           ranks, effect sizes and their diagrams) and figures/ (.png .pdf).
File names say what they hold (e.g. pairwise_wilcoxon_holm_effect_sizes.csv); LaTeX labels stay unique
(tab:<block>_b<budget>_<table>).

Each experiment block of the configuration lists its analyses:
  comparison     per budget: summary, per family and per size, Friedman + Nemenyi (CD diagram),
                 pairwise Wilcoxon-Holm, comparison with the control, performance profile, boxplot
  head_to_head   the pairs of the block (e.g. RR vs MT): paired gaps, W/T/L, p-values, effect sizes
  anytime        median gap curves over time (log scale), Gurobi dual bound, time to targets, primal integral
  certificates   every lower and upper certificate against OPT / LB*: validity, tightness, cost, a priori optimality
  scalability    time and gap against n, per method
  gurobi         status, incumbent, dual bound, MIP gap and proven optima (Minor comment 1)
  tuning         the grid and the chosen parameters
  managerial     peak reduction against uncontrolled (ASAP) and non-preemptive (GNP) charging, load profiles
  sota           state-of-the-art baselines (lists the ones still to be integrated)
plus, always: overview, instances, exact references and validity checks.
"""

import json
import os
import re
import time
import warnings

import numpy as np
import pandas as pd

from ..core.resources import describe_environment, environment
from ..core.results import ENVIRONMENT, FAITHFULNESS, LOGS, REPORT, ResultStore, results_dir
from ..core.runner import method_specs, resolve_path
from . import figures as F
from . import metrics as MT
from . import stats as S
from .tables import TableWriter, multi_metric, pivot_mean, seed_variability, summary_by_method, to_markdown, wtl_matrix

# ── names of the output files: what they hold, and where (tables / statistics / figures) ─────
_TABLES = {"summary": "summary", "by_family": "mean_gap_by_family", "by_size": "mean_gap_by_size_class",
           "by_n": "mean_gap_by_n", "time": "charged_time", "all_metrics": "all_quality_measures",
           "seeds": "seed_variability", "per_instance": "per_instance_mean_peak",
           "ttt": "time_to_target_and_primal_integral", "certificates": "all_certificates_vs_optimum",
           "apriori": "a_priori_optimal_instances", "two_sided": "two_sided_certified_interval",
           "ratios_by_group": "certificate_ratios_by_family_and_n", "within": "certificates_within_tolerance",
           "costs": "certificate_step_costs", "scaling": "median_time_by_n", "scaling_gap": "median_gap_by_n",
           "certify_scaling": "certify_costs_by_n", "gurobi": "by_model_budget_family_size",
           "gurobi_status": "termination_status", "gurobi_per_instance": "per_instance_bounds_and_gaps",
           "pairs": "pairs_wilcoxon_holm", "reduction": "peak_reduction_vs_reference_policies",
           "peaks_by_family": "mean_peak_by_family", "certify_costs": "certify_costs_by_family_and_n",
           "references_summary": "exact_references_summary", "gap_by_budget": "mean_gap_by_budget",
           "time_by_budget": "mean_charged_time_by_budget"}
_FIGURES = {"quality_time": "quality_vs_time", "time_box": "charged_time_boxplot", "profile": "performance_profile",
            "box": "gap_boxplot", "gap_vs_n": "gap_vs_n", "gurobi_bound": "gurobi_dual_bound_over_time",
            "upper": "upper_certificates_tightness", "lower": "lower_certificates_tightness",
            "scaling_time": "time_vs_n", "scaling_gap": "gap_vs_n", "gap_vs_budget": "gap_vs_budget"}
_STATISTICS = {"friedman_test": "friedman_iman_davenport_test", "ranks": "friedman_average_ranks",
               "ranks_by_family": "friedman_ranks_by_family", "cd": "nemenyi_critical_difference_diagram",
               "pairwise": "pairwise_wilcoxon_holm_effect_sizes", "pheat": "holm_pvalue_heatmap",
               "wins": "win_matrix"}
SECTIONS = {
    "00_campaign_overview": "campaign status, machines, method library, default and tuned parameters, headline "
                            "results of every block",
    "01_instances": "instance sets and their features",
    "02_exact_references_and_bounds": "exact references (proven optima, dual bounds) and the cost of the Certify "
                                      "artifacts",
    "03_validity_checks": "feasibility of every schedule, a priori guarantees, certified lower bounds, budgets, "
                          "abnormal runs",
    "04_faithfulness_of_earlier_methods": "the original code of the earlier methods against their re-implementation",
    "05_theory_verification": "numerical verification of every statement of Sections 3-4 (theory/verify_theory.py)",
}
BLOCK_PARTS = {
    "budget": "comparison at one budget: tables/ (summary with 95% CIs, breakdowns by family, size and n, charged "
              "time, all quality measures, seed variability, per-instance peaks, time to target), statistics/ "
              "(Friedman and Iman-Davenport test, average ranks overall and by family, Nemenyi critical difference "
              "diagram, pairwise Wilcoxon signed-rank tests with Holm correction, sign test, Cliff's delta and "
              "Vargha-Delaney A12, p-value heatmap, win matrix, control against each method), figures/ (performance "
              "profile, gap and time box plots, quality against time, gap against n, anytime curves, Gurobi dual "
              "bound over time)",
    "certificates": "every lower and upper certificate against the optimum or best known value",
    "head_to_head": "the pairs of the block (earlier method against its certified counterpart)",
    "scalability": "charged time, gap and Certify costs against n",
    "gurobi_status_and_bounds": "Gurobi: termination status, proven optima, MIP gaps, dual bounds, per-instance "
                                "table, incumbent and bound over time",
    "tuning": "the tuning grid of each method and the marginal effect of each parameter",
    "managerial": "peak reduction against uncontrolled and non-preemptive charging, load profiles",
    "across_budgets": "every method at every budget of the block: mean gap and charged time by budget, gap against "
                      "the budget (does a longer run improve the solution?)",
}


def clear_name(stem, kind, section=""):
    """File name (saying what the file holds) and subfolder of a table ('table') or figure ('figure')."""
    if stem in _STATISTICS:
        return _STATISTICS[stem], "statistics"
    if stem.startswith("vs_"):
        return f"{stem[3:]}_vs_each_method", "statistics"
    if section == "tuning":
        return (f"marginal_effects_{stem[:-len('_marginal')]}" if stem.endswith("_marginal") else f"grid_{stem}"), \
            "tables"
    for pat, fmt in ((r"anytime_(.+)", "anytime_median_gap_{}"), (r"progress_(\d+)", "incumbent_and_bound_example_{}"),
                     (r"profile_(\d+)", "load_profile_example_{}")):
        m = re.fullmatch(pat, stem)
        if m and kind == "figure":
            return fmt.format(m.group(1)), "figures"
    if kind == "figure":
        return _FIGURES.get(stem, stem), "figures"
    return _TABLES.get(stem, stem), "tables"


def budget_folder(b):
    return f"budget_{b:g}s" if b == b and b is not None else "single_run"


CSS = """
:root{--bg:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--acc:#2a78d6;--code:#f5f5f3}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--acc:#3987e5;--code:#242423}}
body{background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:0 auto;padding:24px 16px}
h1,h2,h3,h4{line-height:1.25} h2{border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:2.2em}
table{border-collapse:collapse;font-size:13px;margin:10px 0;display:block;overflow-x:auto;max-width:100%}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left} th{color:var(--ink2);font-weight:600}
img{max-width:100%;height:auto;background:#fff;border-radius:4px} code{background:var(--code);padding:1px 4px;border-radius:3px}
.note{color:var(--ink2)} a{color:var(--acc)} nav li{margin:2px 0}
"""


class Report:
    def __init__(self, cfg):
        self.cfg = cfg
        self.out = resolve_path(cfg, cfg["run"].get("output", "results"))
        self.store = ResultStore(self.out)
        self.root = results_dir(self.out, REPORT)
        os.makedirs(self.root, exist_ok=True)
        self.where, self.prefix, self.block = (), "", ""
        self.folders = {}                               # folder -> what it holds (README of the report)
        self.headline = []                              # one row per block and budget (headline table)
        rc = cfg.get("report", {})
        self.alpha = float(rc.get("alpha", 0.05))
        self.metric = rc.get("metric", "gap_lb")
        self.targets = tuple(rc.get("targets", [0.0, 0.01, 0.05]))
        self.T = TableWriter(self.root)
        self.Fg = F.FigureWriter(self.root, tuple(rc.get("figure_formats", ["png", "pdf"])))
        order = [m.get("id", m["class"]) for m in cfg.get("methods", [])]
        self.styles = F.Styles(rc.get("color_order", order))
        self.highlight = tuple(rc.get("highlight", ["CRI"]))
        self.md = []
        self.toc = []

    # -- writing helpers --
    def h(self, level, text):
        try:
            from markdown.extensions.toc import slugify
            anchor = slugify(text, "-")
        except ImportError:
            anchor = "".join(ch if ch.isalnum() else "-" for ch in text.lower()).strip("-")
        if level == 2:
            self.toc.append((text, anchor))
        self.md.append(f"\n{'#' * level} {text}\n")

    def p(self, text):
        self.md.append(text + "\n")

    def at(self, *parts, prefix="", desc=None):
        """Folder of the next tables and figures (relative to the report root); `prefix` is removed from the
        names given by the analyses (they carry the block and budget, which the folders now say)."""
        self.where, self.prefix = tuple(parts), prefix
        if desc:
            self.folders.setdefault("/".join(parts), desc)

    def _place(self, name, kind):
        stem = name[len(self.prefix):] if self.prefix and name.startswith(self.prefix) else name
        stem, sub = clear_name(stem, kind, self.where[-1] if self.where else "")
        return os.path.join(self.root, *self.where, sub), stem

    def table(self, name, df, caption="", digits=2, index=True, bold_min=None, show=None):
        """Writes the table files; `show` (a DataFrame) replaces the table shown in the report."""
        if df is None or len(df) == 0:
            return
        folder, stem = self._place(name, "table")
        md = self.T.write(stem, df, caption, digits, index, bold_min, folder=folder, label=name)
        if caption:
            self.md.append(f"*{caption}*  (`{self.T.index[-1][0]}.csv`, `.tex`, `.md`)\n")
        if show is not None:
            md = to_markdown(show, digits, index, bold_min) if len(show) else "(none)"
        self.md.append(md + "\n")

    def figure(self, fig, name, caption=""):
        if fig is None:
            return
        folder, stem = self._place(name, "figure")
        path = self.Fg.save(fig, stem, folder=folder, caption=caption)
        self.md.append(f"![{caption or name}]({path.replace(os.sep, '/')})\n")
        if caption:
            self.md.append(f"*Figure: {caption}*\n")

    # -- data --
    def load(self):
        st = self.store
        self.runs = MT.charge_wall_time(st.load_table("runs"))
        if self.runs is None or len(self.runs) == 0:
            raise RuntimeError(f"no runs in {st.data_root}; run the campaign first")
        self.traces = st.load_table("traces") or {}
        # every run is credited with what it had found by its budget + tolerance (improvements later: not counted)
        self.n_at_budget = MT.evaluate_at_budget(self.runs, self.traces)
        self.bound_traces = st.load_table("bound_traces") or {}
        self.solutions = st.load_table("solutions") or {}
        self.refs = st.load_table("references")
        self.contexts = st.load_table("contexts")
        self.feats = st.load_table("instance_features")
        saved = st.load_table("instances") or {}
        self.instances = {d["name"]: d for v in saved.get("sets", {}).values() for d in v}
        self.ref = MT.references(self.runs, self.refs, self.contexts)
        r = MT.enrich(self.runs, self.traces, self.ref, self.targets, self.contexts)
        if self.feats is not None:
            r = r.merge(self.feats.drop_duplicates("instance"), on="instance", how="left")
        r["size"] = r.n.map(MT.size_class) if "n" in r else ""
        self.r = r
        st.save_table("runs_enriched", r)
        st.save_table("instance_references", self.ref)
        try:                                            # plain exports for spreadsheets / other tools
            r.to_csv(os.path.join(st.data_root, "runs_enriched.csv"), index=False)
            self.ref.to_csv(os.path.join(st.data_root, "instance_references.csv"))
        except OSError:
            pass

    def campaign_machines(self):
        """The machines that ran the campaign, from 05_logs/environment.jsonl (one line per start of `run`);
        for a campaign started before that file existed, the hosts of the runs and the machine lines of the
        run log."""
        logs = os.path.join(self.out, LOGS)
        seen = {}
        try:
            with open(os.path.join(logs, ENVIRONMENT), encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        e = json.loads(line)
                        seen.setdefault(describe_environment(e), e.get("time", ""))
        except (OSError, ValueError):
            pass
        if seen:
            return [f"- campaign run on {d} (first start {t})" for d, t in seen.items()]
        hosts = sorted(self.runs.info_host.dropna().unique()) if "info_host" in self.runs else []
        ml = []
        try:
            with open(os.path.join(logs, "run_log.txt"), encoding="utf-8", errors="replace") as f:
                ml = sorted({ln.split("] machine: ", 1)[1].strip() for ln in f if "] machine: " in ln})
        except OSError:
            pass
        return [f"- campaign run on {', '.join(hosts) or '(host not recorded)'}"
                + (f" ({'; '.join(ml)})" if ml else "")
                + f"; processor and software versions not recorded (no {LOGS}/{ENVIRONMENT})"]

    # -- sections --
    def overview(self):
        cfg = self.cfg
        self.md.insert(0, f"# {cfg.get('report', {}).get('title', 'Experimental report')}\n")
        self.h(2, "Overview")
        lines = [f"- generated {time.strftime('%Y-%m-%d %H:%M')}, configuration `{os.path.basename(cfg.get('_config_path', ''))}`"]
        lines += self.campaign_machines()
        lines.append(f"- report built on {describe_environment(environment())}")
        rc = cfg["run"]
        lines.append(f"- {rc.get('workers')} parallel workers; Gurobi threads per run as configured per method; "
                     "wall-clock budgets")
        lines.append("- **time accounting:** the time of a method includes the time of every Certify artifact it "
                     "uses (LP, lower bounds, certificates), computed once per instance and charged to every method "
                     "that uses it; methods built on such artifacts start their clock with that time already spent.")
        lines.append(f"- **metric:** `{self.metric}` = 100 (peak - LB*) / LB*, where LB* is the best valid lower "
                     "bound known for the instance (certified LB, Gurobi dual bounds, exact reference). Lower is better.")
        lines.append("- **statistics:** tests use per-instance means over seeds (blocks = instances); Friedman with "
                     "the Iman-Davenport correction, Nemenyi critical difference, Wilcoxon signed-rank with Holm "
                     f"correction (alpha = {self.alpha}), Cliff's delta, Vargha-Delaney A12, bootstrap 95% CIs.")
        self.p("\n".join(lines))
        r = self.runs
        st = r.groupby(["experiment", "status"]).size().unstack(fill_value=0)
        self.table("campaign_status", st, "Runs per experiment and status", digits=0)
        if "info_host" in r:
            hs = r.assign(host=r.info_host.fillna("(not recorded)")).groupby(["experiment", "host"]).size().unstack(fill_value=0)
            self.table("campaign_hosts", hs, "Runs per experiment and machine (time comparisons are meaningful only "
                       "between runs of the same machine)", digits=0)
        specs = method_specs(cfg, [m.get("id", m["class"]) for m in cfg.get("methods", [])])
        rows = [{"id": s["id"], "class": s["class"], "params": json.dumps(s.get("params", {}), sort_keys=True)}
                for s in specs]
        self.table("method_library", pd.DataFrame(rows).set_index("id"), "Method library (explicit parameters; "
                   "the class defaults of the configuration apply otherwise)")
        dflt = cfg.get("defaults", {})
        tuned_p = os.path.join(self.store.data_root, "tuned_defaults.json")
        if os.path.exists(tuned_p):
            with open(tuned_p, encoding="utf-8") as f:
                tuned = json.load(f)
            self.p("Tuned defaults applied to the later blocks: `" + json.dumps(tuned, sort_keys=True) + "`")
        self.table("class_defaults", pd.DataFrame([{"class": k, "defaults": json.dumps(v, sort_keys=True)}
                                                   for k, v in dflt.items()]).set_index("class"),
                   "Default parameters per method class")

    def instances_section(self):
        if self.feats is None or not len(self.feats):
            return
        self.h(2, "Instances")
        f = self.feats
        agg = f.groupby(["set", "family", "n", "T"]).agg(count=("instance", "size"), K=("K", "mean"),
                                                          mean_window=("mean_window", "mean"),
                                                          load_factor=("load_factor", "mean"),
                                                          n_vars=("n_vars", "mean")).reset_index()
        self.table("instance_sets", agg.set_index("set"), "Instance sets (means over the instances of each group)")

    def references_section(self):
        self.h(2, "Exact references and lower bounds")
        ref = self.ref
        if self.refs is not None and len(self.refs):
            rf = self.refs.copy()
            rf["gap_ref_pct"] = 100 * (rf.best - rf.bound) / rf.bound
            g = rf.groupby("set").agg(instances=("instance", "size"), proven=("proven", "sum"),
                                      mean_time=("time", "mean"), max_time=("time", "max"),
                                      mean_gap_pct=("gap_ref_pct", "mean"))
            self.table("references_summary", g, "Exact references (aggregated model, time limit "
                       f"{self.cfg.get('references', {}).get('time_limit', 600)} s)")
            both = rf[rf.proven.astype(bool)]
            if len(both):
                eq = (np.abs(both.lb_certified - both.best) < 1e-6).sum()
                self.p(f"On the {len(both)} instances with a proven optimum, the certified lower bound equals the "
                       f"optimum on **{eq}** ({100 * eq / len(both):.1f}%).")
        closed = ref.closed.sum()
        self.p(f"Instances closed (best known = best lower bound): **{int(closed)} of {len(ref)}**.")
        if self.contexts is not None and len(self.contexts):
            c = self.contexts.copy()
            cols = [k for k in c.columns if k.startswith("cost_")]
            if cols:
                c = c.merge(self.feats[["instance", "n", "family"]], on="instance", how="left") if self.feats is not None else c
                g = c.groupby(["family", "n"])[cols].median()
                self.table("certify_costs", g, "Median computing time (s) of the Certify artifacts", digits=3)

    def validity_section(self):
        self.h(2, "Validity checks")
        r = self.r
        lines = []
        has_sol = r.objective.notna()
        lines.append(f"- runs with a schedule: {int(has_sol.sum())}; infeasible schedules: "
                     f"**{int((has_sol & ~r.feasible.astype(bool)).sum())}**")
        if "guarantee_ok" in r:
            g = r[r.guarantee_ok.notna()]
            lines.append(f"- a priori guarantees checked: {len(g)}; violated: **{int((g.guarantee_ok == 0).sum())}**")
        lb_viol = 0
        if self.contexts is not None and "LB" in self.contexts and len(self.contexts):
            mm = self.contexts.groupby("instance").LB.max()
            bks = self.ref.BKS
            lb_viol = int((mm.reindex(bks.index) > bks + 1e-6).sum())
        lines.append(f"- certified lower bounds above a feasible value (impossible if correct): **{lb_viol}**")
        over = MT.budget_overruns(r, self.traces)
        doc = self.cfg.get("report", {}).get("documented_overruns", {}) or {}
        single = over.info_class.fillna("").isin(MT.OVER_BUDGET_OK) if "info_class" in over else over.method.eq("")
        no_gain = ~single & ~over.gained
        documented = ~single & over.gained & over.method.isin(list(doc))
        unexplained = over[~single & over.gained & ~over.method.isin(list(doc))]
        lines.append(f"- runs whose charged time exceeds the budget by more than 10% (+0.05 s): {len(over)} "
                     f"(single-shot methods: {int(single.sum())}; no improvement after the budget: "
                     f"{int(no_gain.sum())}; documented: {int(documented.sum())}; unexplained: **{len(unexplained)}**"
                     + (f" {unexplained.method.value_counts().to_dict()}" if len(unexplained) else "") + ")")
        at_b = r[r.evaluated_at_budget.astype(bool)] if "evaluated_at_budget" in r else r.iloc[:0]
        lines.append(f"- runs evaluated at their budget (improved only after the budget + 10% + 0.05 s; the incumbent "
                     f"they had then is used): {len(at_b)}"
                     + (f" ({at_b.method.value_counts().to_dict()})" if len(at_b) else ""))
        if len(over):
            ev = over.evaluated_at_budget.astype(bool) if "evaluated_at_budget" in over else over.method.eq("")
            ot = over.assign(no_schedule=np.isinf(over.at_budget), at_budget_eval=ev)
            if "n" in ot:
                ot["n range"] = ot.groupby(["experiment", "method", "budget"]).n.transform(
                    lambda x: f"{int(x.min())}-{int(x.max())}" if x.notna().any() else "")
            else:
                ot["n range"] = ""
            ot = ot.groupby(["experiment", "method", "budget"]).agg(
                runs=("ratio", "size"), worst=("ratio", "max"), mean=("ratio", "mean"), n=("n range", "first"),
                improved_after_budget=("gained", "sum"), no_schedule_by_budget=("no_schedule", "sum"),
                evaluated_at_budget=("at_budget_eval", "sum")).reset_index()
            cls = over.groupby(["experiment", "method", "budget"]).info_class.first() if "info_class" in over \
                else None

            def reason(e, m, b, gained, evaluated):
                if cls is not None and cls.get((e, m, b)) in MT.OVER_BUDGET_OK:
                    return "single-shot method: its certificate artifacts alone exceed the budget"
                if not gained and evaluated:
                    return (f"{int(evaluated)} run(s) improved only after the budget: evaluated at the budget, the "
                            "later improvements are not counted; the others gained nothing after the budget")
                if not gained:
                    return "no improvement after the budget (anytime trace): the extra time changed no result"
                return doc.get(m, "UNEXPLAINED")
            ot["reason"] = [reason(e, m, b, g, v) for e, m, b, g, v in
                            zip(ot.experiment, ot.method, ot.budget, ot.improved_after_budget, ot.evaluated_at_budget)]
            self.table("budget_overruns", ot.set_index(["experiment", "method", "budget"]),
                       "Runs whose charged time exceeds the budget by more than 10% (+0.05 s), per block, method and "
                       "budget: number of runs, worst and mean ratio charged time / budget, range of n, runs whose "
                       "credited result still comes after the budget, runs with no schedule by the budget, runs "
                       "evaluated at the budget, and the reason (budget = budget + 10% + 0.05 s)", digits=2)
        bad = r[~r.status.isin(["ok", "not_available"])]
        lines.append(f"- runs with another status: {len(bad)} " + (f"({bad.status.value_counts().to_dict()})" if len(bad) else ""))
        self.p("\n".join(lines))
        vt = pd.DataFrame([{"check": ln[2:].split(":")[0], "result": ln.split(":", 1)[1].strip().replace("*", "")}
                           for ln in lines]).set_index("check")
        self.table("validity_summary", vt, "Validity checks of the whole campaign")
        if len(bad):
            cols = [c for c in ("experiment", "instance", "method", "budget", "seed", "status", "info_error") if c in bad]
            self.table("abnormal_runs", bad[cols].head(50).set_index("experiment"), "Abnormal runs (first 50)")

    # -- analyses --
    def comparison(self, name, ex, r):
        metric = ex.get("metric", self.metric)
        control = ex.get("control")
        if "info_class" in r:
            r = r[r.info_class != "CERTIFY"]           # no schedule: not part of method comparisons
        for b, rb in r.groupby("budget", dropna=False):
            tag = f"{name}_b{b:g}" if b == b and b is not None else name
            bl = f"{b:g} s" if b == b and b is not None else "single run"
            self.at(self.block, budget_folder(b), prefix=tag + "_", desc=BLOCK_PARTS["budget"])
            self.h(3, f"Budget {b:g} s" if b == b and b is not None else "Single run")
            rb = rb[rb.status != "not_available"]
            if rb.empty:
                continue
            self.table(f"{tag}_summary", summary_by_method(rb, metric),
                       f"{name}, budget {bl}: {metric} (per-instance mean over seeds)", bold_min=[f"mean {metric}"])
            if "family" in rb and rb.family.nunique() > 1:
                self.table(f"{tag}_by_family", pivot_mean(rb, metric, "family"), f"Mean {metric} by family")
            if "size" in rb and rb["size"].nunique() > 1:
                self.table(f"{tag}_by_size", pivot_mean(rb, metric, "size"), f"Mean {metric} by size class")
            if "n" in rb and rb.n.nunique() > 1:
                self.table(f"{tag}_by_n", pivot_mean(rb, metric, "n"), f"Mean {metric} by n")
            tm = rb.groupby("method").time.agg(["mean", "median", "max"])
            self.table(f"{tag}_time", tm, "Charged time (s)")
            self.table(f"{tag}_all_metrics", multi_metric(rb), "All quality measures (means over instances): gap to "
                       "the best lower bound LB*, to the best known value, to the proven optimum, to our own certified "
                       "lower bound, to the LP bound, relative difference to the exact solver's incumbent; instances "
                       "proved optimal by our certificate alone and by the method's own bound", digits=3)
            sv = seed_variability(rb, metric)
            if len(sv):
                self.table(f"{tag}_seeds", sv, "Variability over seeds (per-instance standard deviation and range)",
                           digits=3)
            per = rb.pivot_table(index="instance", columns="method", values="objective", aggfunc="mean")
            self.table(f"{tag}_per_instance", per, "Per-instance mean peak (kW) of every method", digits=2,
                       show=pd.DataFrame())
            if "n" in rb and rb.n.nunique() >= 3:
                gn = rb.groupby(["method", "n"])[metric].mean().reset_index().rename(columns={metric: "y"})
                self.figure(F.scaling(gn, "n", "y", self.styles, f"{metric} against n, {name}, budget {bl}", "n (vehicles)",
                                      f"mean {metric} (%)", logy=False), f"{tag}_gap_vs_n",
                            f"Mean {metric} against the number of vehicles, per method")
            qt = rb.groupby("method").agg(time=("time", "mean"), gap=(metric, "mean")).reset_index()
            self.figure(F.quality_time(qt, self.styles, f"Quality against time, {name}, budget {bl}",
                                       ylabel=f"mean {metric} (%)"), f"{tag}_quality_time",
                        "Mean quality against mean charged time (one point per method)")
            self.figure(F.boxplot({m: g.time.values for m, g in rb.groupby("method")}, self.highlight,
                                  f"Charged time, {name}, budget {bl}", xlabel="charged time (s)"),
                        f"{tag}_time_box", "Distribution of the charged time per run")
            M = MT.per_instance(rb, metric)
            if M.shape[1] >= 2:
                self.stats_block(tag, M, control, f"{name}, budget {bl}")
                means = M.mean()
                row = {"block": name, "budget": bl, "instances": int(M.shape[0]),
                       "methods": int(M.shape[1]), f"best mean {metric}": float(means.min()),
                       "best method": str(means.idxmin())}
                if control and control in M:
                    row["control"] = control
                    row[f"control mean {metric}"] = float(means[control])
                Mc = M.dropna()
                if len(Mc) >= 2 and Mc.shape[1] >= 3:
                    fr = S.friedman(Mc)
                    row["Friedman p"] = fr["p"]
                    if control and control in M:
                        row["control average rank"] = float(fr["ranks"].get(control, np.nan))
                self.headline.append(row)
                obj = MT.per_instance(rb, "objective")
                self.figure(F.performance_profile(obj, self.styles, f"Performance profile, {name}, budget {bl}"),
                            f"{tag}_profile", "Dolan-More performance profile of the peak (per-instance mean)")
                self.figure(F.boxplot({m: M[m].values for m in M.columns}, self.highlight,
                                      f"{metric}, {name}, budget {bl}", xlabel=f"{metric} (%)"),
                            f"{tag}_box", f"Distribution of the per-instance {metric}")

    def stats_block(self, tag, M, control, title):
        fr = S.friedman(M)
        Mc = M.dropna()
        if fr["N"] >= 2 and fr["k"] >= 3:
            cd = S.nemenyi_cd(fr["k"], fr["N"], self.alpha)
            self.p(f"Friedman: N = {fr['N']} instances, k = {fr['k']} methods, chi2 = {fr['chi2']:.2f}, "
                   f"p = {fr['p']:.3g}; Iman-Davenport F = {fr['F_ID']:.2f}, p = {fr['p_ID']:.3g}; "
                   f"Nemenyi CD = {cd:.2f}.")
            self.table(f"{tag}_friedman_test", pd.DataFrame([{
                "instances (N)": fr["N"], "methods (k)": fr["k"], "chi2": fr["chi2"], "p": fr["p"],
                "Iman-Davenport F": fr["F_ID"], "p (F)": fr["p_ID"], f"Nemenyi CD (alpha {self.alpha:g})": cd}]).set_index(
                "instances (N)"), "Friedman test with the Iman-Davenport correction and the Nemenyi critical difference",
                digits=4)
            self.table(f"{tag}_ranks", fr["ranks"].rename("average rank").to_frame(), "Average ranks (1 = best)")
            fam = self._family_ranks(tag, M)
            if fam is not None:
                self.table(f"{tag}_ranks_by_family", fam, "Average ranks by family, with the Friedman p-value of "
                           "each family", digits=3)
            self.figure(F.cd_diagram(fr["ranks"], cd, f"Critical differences, {title.replace(', budget nan', '')}"), f"{tag}_cd",
                        "Average ranks; methods joined by a bar are not significantly different (Nemenyi)")
        pw = S.pairwise(M, self.alpha)
        if len(pw):
            cols = ["N", "mean_A", "mean_B", "win_A", "tie", "win_B", "p", "p_holm", "p_sign", "cliff_delta", "cliff_mag",
                    "A12", "significant"]
            full = pw.set_index(["A", "B"])[cols]
            sig = full[full.significant].sort_values("p_holm").head(40)
            self.table(f"{tag}_pairwise", full,
                       f"Pairwise Wilcoxon signed-rank tests (Holm), sign test, effect sizes (negative delta: A better); "
                       f"{len(full)} pairs in the files, the {len(sig)} most significant of the "
                       f"{int(full.significant.sum())} significant ones shown", digits=3, show=sig)
            if Mc.shape[1] >= 3:
                P = S.pvalue_matrix(pw, list(Mc.columns))
                self.figure(F.pvalue_heatmap(P, f"Holm-adjusted p-values, {title}", self.alpha), f"{tag}_pheat",
                            "Pairwise Holm-adjusted Wilcoxon p-values (* significant)")
        if Mc.shape[1] >= 2:
            self.table(f"{tag}_wins", wtl_matrix(M), "Win matrix: entry (row, column) = instances where the row "
                       "method is strictly better than the column method", digits=0)
        if control and control in M.columns:
            vc = S.versus_control(M, control, self.alpha)
            self.table(f"{tag}_vs_{control}", vc.set_index("method"), f"{control} against every other method "
                       "(wins = instances where the control is strictly better)", digits=3)

    def _family_ranks(self, tag, M):
        if self.feats is None or "family" not in self.feats:
            return None
        fam = self.feats.drop_duplicates("instance").set_index("instance").family
        rows = {}
        for f, idx in M.groupby(M.index.map(fam)).groups.items():
            sub = M.loc[idx]
            if len(sub.dropna()) < 2:
                continue
            fr = S.friedman(sub)
            row = fr["ranks"].to_dict()
            row["Friedman p"] = fr["p"]
            row["instances"] = fr["N"]
            rows[f] = row
        return pd.DataFrame(rows).T if rows else None

    def head_to_head(self, name, ex, r):
        metric = ex.get("metric", self.metric)
        pairs = ex.get("pairs", [])
        if not pairs:
            return
        self.at(self.block, "head_to_head", prefix=name + "_", desc=BLOCK_PARTS["head_to_head"])
        self.h(3, "Head-to-head comparisons")
        rows = []
        for b, rb in r.groupby("budget", dropna=False):
            M = MT.per_instance(rb, metric)
            Tm = MT.per_instance(rb, "time")
            for a, c in pairs:
                if a not in M or c not in M:
                    continue
                X = M[[a, c]].dropna()
                if not len(X):
                    continue
                p, _ = S.wilcoxon(X[a], X[c])
                w, t, l = S.win_tie_loss(X[a], X[c])
                cd = S.cliffs_delta(X[a], X[c])
                rows.append({"budget": b, "A": a, "B": c, "N": len(X), f"mean {metric} A": X[a].mean(),
                             f"mean {metric} B": X[c].mean(), "mean time A": Tm[a].mean(), "mean time B": Tm[c].mean(),
                             "A better": w, "ties": t, "B better": l, "p (Wilcoxon)": p, "Cliff delta": cd,
                             "magnitude": S.magnitude_cliff(cd) if cd == cd else ""})
        if rows:
            df = pd.DataFrame(rows)
            df["p (Holm)"] = S.holm(df["p (Wilcoxon)"].values)
            self.table(f"{name}_pairs", df.set_index(["budget", "A", "B"]),
                       "Head-to-head: earlier method (A) against its certified counterpart (B)", digits=3)
            for _, row in df.iterrows():
                better = row.A if row["A better"] > row["B better"] else row.B
                sig = "significant" if row["p (Holm)"] < self.alpha else "not significant"
                self.p(f"- budget {row.budget:g} s, {row.A} vs {row.B}: {better} better on more instances "
                       f"({row['A better']}/{row['ties']}/{row['B better']} W/T/L for {row.A}), {sig} "
                       f"(Holm p = {row['p (Holm)']:.3g}), mean {metric} {row[f'mean {metric} A']:.3f} vs "
                       f"{row[f'mean {metric} B']:.3f}.")

    def anytime(self, name, ex, r):
        self.h(3, "Anytime behaviour")
        for b, rb in r.groupby("budget", dropna=False):
            if not (b == b) or not b:
                continue
            self.at(self.block, budget_folder(b), prefix=f"{name}_b{b:g}_", desc=BLOCK_PARTS["budget"])
            rb = rb[rb.status != "not_available"]
            groups = [("all", rb)]
            if ex.get("anytime_by", "family") in rb and rb[ex.get("anytime_by", "family")].nunique() > 1:
                groups += list(rb.groupby(ex.get("anytime_by", "family")))
            for gname, g in groups:
                tmin = max(1e-3, min([t for k in zip(g.experiment, g.instance, g.method, g.budget, g.seed)
                                      for t, _ in self.traces.get(k, [])[:1]] or [1e-2]))
                grid = np.logspace(np.log10(tmin), np.log10(b), 60)
                curves = {}
                methods = ex.get("figure_methods") or list(g.method.unique())
                for m in methods:
                    gm = g[g.method == m]
                    if gm.empty:
                        continue
                    keys = list(zip(gm.experiment, gm.instance, gm.method, gm.budget, gm.seed))
                    C = MT.anytime_curve(self.traces, keys, lambda i: self.ref.LB_star.get(i, np.nan), grid)
                    # a run without a schedule yet counts as infinitely bad: the median exists once half of
                    # the runs have a schedule, and the curves are non-increasing
                    # (a finite sentinel, not inf: np.quantile interpolates to NaN next to an infinite value)
                    C = np.where(np.isnan(C), 1e12, C)
                    med, q1, q3 = (np.quantile(C, q, axis=0) for q in (0.5, 0.25, 0.75))
                    med, q1, q3 = (np.where(v < 1e11, v, np.nan) for v in (med, q1, q3))
                    curves[m] = (grid, med, q1, q3)
                # Gurobi dual bound (gap of the bound below LB*, shown as negative values would mislead: we
                # plot the bound's distance to LB* separately in the gurobi analysis)
                self.figure(F.anytime(curves, self.styles, f"{name}, {gname}, budget {b:g} s"),
                            f"{name}_b{b:g}_anytime_{gname}".replace(" ", ""),
                            f"Median (IQR band) gap to LB* over time, {gname}; the curve of a method starts "
                            "when its first schedule exists (artifact time included)")
            gb = rb[rb.method.str.startswith("GUROBI")]
            if len(gb):
                grid = np.logspace(-2, np.log10(b), 50)
                curves = {}
                for m, gm in gb.groupby("method"):
                    rows = []
                    for k in zip(gm.experiment, gm.instance, gm.method, gm.budget, gm.seed):
                        bt = sorted(self.bound_traces.get(k, []))
                        lbs = self.ref.LB_star.get(k[1], np.nan)
                        v, j, cur = np.full(len(grid), np.nan), 0, np.nan
                        for a, tt in enumerate(grid):
                            while j < len(bt) and bt[j][0] <= tt:
                                cur = bt[j][1] if cur != cur else max(cur, bt[j][1]); j += 1
                            if cur == cur:
                                v[a] = 100 * (lbs - cur) / lbs
                        rows.append(v)
                    C = np.where(np.isnan(np.array(rows)), np.inf, np.array(rows))
                    with np.errstate(all="ignore"):
                        med, q1, q3 = (np.quantile(C, q, axis=0) for q in (0.5, 0.25, 0.75))
                    curves[m + " bound"] = tuple([grid] + [np.where(np.isfinite(v), v, np.nan) for v in (med, q1, q3)])
                    self.styles.get(m + " bound")
                self.figure(F.anytime(curves, self.styles, f"Gurobi dual bound below LB* over time, {name}, budget {b:g} s",
                                      ylabel="(LB* - dual bound) / LB* (%)"), f"{name}_b{b:g}_gurobi_bound",
                            "Median distance of Gurobi's dual bound below the best known lower bound over time "
                            "(0 = as strong as LB*)")
            cols = [c for c in rb.columns if c.startswith("ttt_")]
            agg = {c: (c, lambda x: float(np.median(x))) for c in cols}
            tt = rb.groupby("method").agg(**{f"median {c}": v for c, v in agg.items()},
                                          **{f"reached {c}": (c, lambda x: float(np.isfinite(x).mean() * 100)) for c in cols},
                                          primal_integral=("pi", "mean"))
            self.table(f"{name}_b{b:g}_ttt", tt, "Time to targets (median s; inf = not reached), share of runs reaching "
                       "each target (%), mean primal integral (normalized by the budget)", digits=3,
                       bold_min=["primal_integral"])

    def certificates(self, name, ex, r):
        self.at(self.block, "certificates", prefix=name + "_", desc=BLOCK_PARTS["certificates"])
        self.h(3, "Certificates")
        c = r[r.method.str.startswith("CERTIFY")].copy()
        if c.empty:
            return
        cols = [k for k in c.columns if k.startswith("info_") and pd.api.types.is_numeric_dtype(c[k])]
        ref = c.instance.map(self.ref.OPT).fillna(c.instance.map(self.ref.BKS))
        from ..problems.ppmsp.context import LOWER_KEYS, _is_upper
        rows, ratios_up, ratios_lo = [], {}, {}
        meta = {"wall", "class", "restarts", "iterations", "final_theta", "tuning_point", "w_max", "a_priori_optimal",
                "conf_finished"}
        for k in cols:
            key = k[5:]
            v = c[k].astype(float)
            if key.startswith("cost_") or key.startswith("time_") or key in meta or v.notna().sum() == 0:
                continue
            if key == "U_best":                          # the best upper certificate (min over all of them)
                ratio = v / ref
                opt_known = c.instance.map(self.ref.OPT).notna()
                ratios_up[key] = ratio.values
                rows.append({"certificate": key, "kind": "upper (best)", "instances": int(v.notna().sum()),
                             "mean ratio to OPT/BKS": ratio.mean(), "max ratio": ratio.max(),
                             "#equal OPT/BKS": int((np.abs(v - ref) < 1e-6).sum()),
                             "violations": int(((v < ref - 1e-6) & opt_known).sum())})
                continue
            if key in LOWER_KEYS or key.startswith("published_d") or key.endswith("_n_components"):
                if key in LOWER_KEYS:
                    ratio = v / ref
                    viol = int((v > ref + 1e-6).sum())
                    ratios_lo[key] = ratio.values
                    rows.append({"certificate": key, "kind": "lower", "instances": int(v.notna().sum()),
                                 "mean ratio to OPT/BKS": ratio.mean(), "min ratio": ratio.min(),
                                 "#equal OPT/BKS": int((np.abs(v - ref) < 1e-6).sum()), "violations": viol})
                continue
            if _is_upper(key) and key not in ("a_priori_optimal",):
                ratio = v / ref
                opt_known = c.instance.map(self.ref.OPT).notna()
                viol = int(((v < ref - 1e-6) & opt_known).sum())
                ratios_up[key] = ratio.values
                rows.append({"certificate": key, "kind": "upper", "instances": int(v.notna().sum()),
                             "mean ratio to OPT/BKS": ratio.mean(), "max ratio": ratio.max(),
                             "#equal OPT/BKS": int((np.abs(v - ref) < 1e-6).sum()), "violations": viol})
        df = pd.DataFrame(rows).set_index("certificate")
        self.table(f"{name}_certificates", df, "Every certificate against the optimum (or best known value): "
                   "a violation is an upper certificate below a proven optimum or a lower one above a feasible value",
                   digits=4)
        if "info_a_priori_optimal" in c:
            c["_apo"] = c.info_a_priori_optimal.astype(float)
            by = ["family", "n"] if "family" in c else ["set"]
            ap = c.groupby(by)._apo.agg(["sum", "size"])
            ap.columns = ["a priori optimal", "instances"]
            ap["share %"] = 100 * ap["a priori optimal"] / ap["instances"]
            tot = c._apo.sum()
            self.p(f"Proved optimal by the certificates alone (U_best = LB, no search, no MILP): **{int(tot)} of "
                   f"{len(c)} instances** ({100 * tot / max(1, len(c)):.1f}%).")
            self.table(f"{name}_apriori", ap, "Instances proved optimal by the certificates alone (U_best = LB): no "
                       "search and no MILP", digits=0)
        if "info_U_best" in c and "info_LB" in c:
            g = c.assign(two_sided=100 * (c.info_U_best - c.info_LB) / c.info_LB,
                         ub_gap=100 * (c.info_U_best - ref) / ref, lb_gap=100 * (ref - c.info_LB) / ref)
            by = "family" if "family" in g else "set"
            t = g.groupby([by, "n"]).agg(two_sided_gap=("two_sided", "mean"), U_best_above_ref=("ub_gap", "mean"),
                                        LB_below_ref=("lb_gap", "mean"), instances=("instance", "size"))
            self.table(f"{name}_two_sided", t, "Two-sided certificate: mean (U_best - LB)/LB, U_best above and LB "
                       "below the optimum / best known value (%)", digits=3)
            pub = [k for k in cols if k.startswith("info_published_") and not k.startswith("info_published_d")]
            if pub:
                best_pub = c[pub].min(axis=1)
                imp = 100 * (best_pub - c.info_U_best) / best_pub
                rat = best_pub / c.info_U_best
                self.p(f"Best new upper certificate (U_best) against the best non-localized bound (same LP "
                       f"vertex): U_best is lower by {imp.mean():.1f}% on average (median {imp.median():.1f}%, min "
                       f"{imp.min():.1f}%, max {imp.max():.1f}%); the non-localized bound is {rat.mean():.2f} times U_best "
                       f"on average.")
        keys = [k for k in ("U_best", "lp_union", "lp_component_union", "lp_lll", "lp_exist_sys", "IR_guarantee",
                            "published_best", "LB", "LB_cut", "LB_conf", "P_LP") if "info_" + k in c]
        pub = [k for k in cols if k.startswith("info_published_") and not k.startswith("info_published_d")]
        cc_ = c.copy()
        if pub:
            cc_["info_published_best"] = cc_[pub].min(axis=1)
            if "published_best" not in keys:
                keys.append("published_best")
        by = [x for x in ("family", "n") if x in cc_]
        if keys and by:
            t = pd.DataFrame({k: cc_["info_" + k].astype(float) / ref for k in keys})
            t[by] = cc_[by].values
            self.table(f"{name}_ratios_by_group", t.groupby(by).mean(), "Mean certificate / OPT (or best known) by "
                       "family and size", digits=4)
            tol_rows = []
            for k in keys:
                rr_ = (cc_["info_" + k].astype(float) / ref - 1).abs()
                tol_rows.append({"certificate": k, "within 0.5%": float((rr_ <= 0.005).mean() * 100),
                                 "within 1%": float((rr_ <= 0.01).mean() * 100),
                                 "within 5%": float((rr_ <= 0.05).mean() * 100), "instances": int(rr_.notna().sum())})
            self.table(f"{name}_within", pd.DataFrame(tol_rows).set_index("certificate"),
                       "Share of instances (%) where the certificate is within 0.5 / 1 / 5 % of OPT (or best known)", digits=1)
        main_up = {k: v for k, v in ratios_up.items() if k.startswith("lp_") or k in ("U_best", "IR_guarantee")
                   or k.startswith("published_")}
        self.figure(F.strip_ratios(main_up, "Upper certificates / OPT (or best known)"), f"{name}_upper",
                    "Tightness of the upper certificates (1 = the optimum)")
        self.figure(F.strip_ratios(ratios_lo, "Lower certificates / OPT (or best known)"), f"{name}_lower",
                    "Tightness of the lower certificates (1 = the optimum)")
        cc = [k for k in cols if k.startswith("info_cost_")]
        if cc:
            by = ["family", "n"] if "family" in c else ["n"]
            self.table(f"{name}_costs", c.groupby(by)[cc].median(), "Median cost of each certificate step (s)", digits=3)

    def scalability(self, name, ex, r):
        self.at(self.block, "scalability", prefix=name + "_", desc=BLOCK_PARTS["scalability"])
        self.h(3, "Scalability")
        if "n" not in r:
            return
        rr = r[r.status != "not_available"]
        g = rr.groupby(["method", "n"]).agg(time=("time", "median"), gap=(self.metric, "median"),
                                            t_lo=("time", lambda x: np.quantile(x, 0.25)),
                                            t_hi=("time", lambda x: np.quantile(x, 0.75))).reset_index()
        self.table(f"{name}_scaling", g.pivot(index="n", columns="method", values="time"), "Median charged time (s) by n",
                   digits=3)
        self.table(f"{name}_scaling_gap", g.pivot(index="n", columns="method", values="gap"),
                   f"Median {self.metric} by n", digits=3)
        d = g.rename(columns={"time": "y"}).assign(y_lo=g.t_lo, y_hi=g.t_hi)
        self.figure(F.scaling(d, "n", "y", self.styles, f"Charged time against n, {name}", "n (vehicles)",
                              "median time (s)"), f"{name}_scaling_time", "Median time (IQR band) against n")
        d2 = g.rename(columns={"gap": "y"})[["method", "n", "y"]]
        d2 = d2[d2.y > 0] if (d2.y > 0).any() else d2
        self.figure(F.scaling(d2, "n", "y", self.styles, f"{self.metric} against n, {name}", "n (vehicles)",
                              f"median {self.metric} (%)", logy=bool((d2.y > 0).all())), f"{name}_scaling_gap",
                    f"Median {self.metric} against n")
        if self.contexts is not None and len(self.contexts):
            cc = self.contexts.merge(self.feats[["instance", "n"]], on="instance")
            cols = [k for k in cc.columns if k.startswith("cost_")]
            cc = cc[cc.instance.isin(rr.instance.unique())]
            if len(cc) and cols:
                self.table(f"{name}_certify_scaling", cc.groupby("n")[cols].median(), "Median Certify costs (s) by n",
                           digits=3)

    def gurobi(self, name, ex, r):
        self.at(self.block, "gurobi_status_and_bounds", prefix=name + "_", desc=BLOCK_PARTS["gurobi_status_and_bounds"])
        self.h(3, "Gurobi: status, bounds and MIP gaps")
        g = r[r.info_class.eq("GUROBI")] if "info_class" in r else r[r.method.str.startswith("GUROBI")]
        if g.empty:
            return
        g = g.copy()
        g["mip_gap_pct"] = 100 * g.get("info_mip_gap", np.nan)
        g["bound_vs_LBcert"] = 100 * (g.lower_bound - g.instance.map(self.ctx_lb())) / g.instance.map(self.ctx_lb())
        by = [c for c in ("method", "budget", "family", "n") if c in g]
        t = g.groupby(by).agg(runs=("instance", "size"), proven=("info_proven", "sum"),
                              mean_mip_gap_pct=("mip_gap_pct", "mean"), max_mip_gap_pct=("mip_gap_pct", "max"),
                              mean_gap_lb=(self.metric, "mean"),
                              bound_minus_certLB_pct=("bound_vs_LBcert", "mean"), time=("time", "mean"))
        self.table(f"{name}_gurobi", t, "Gurobi per model, budget, family and size: proven optima, MIP gap reported by "
                   "Gurobi (incumbent vs its own bound), gap to LB*, and its dual bound relative to our certified LB "
                   "(negative: our certificate is stronger)", digits=3)
        st = g.groupby(["method", "budget"]).info_grb_status.value_counts().unstack(fill_value=0)
        self.table(f"{name}_gurobi_status", st, "Gurobi termination status", digits=0)
        per = g[[c for c in ("instance", "method", "budget", "objective", "lower_bound", "mip_gap_pct", "info_grb_status",
                             "time") if c in g]].copy()
        per["LB_certified"] = per.instance.map(self.ctx_lb())
        if self.contexts is not None and "P_LP" in self.contexts:
            per["P_LP"] = per.instance.map(self.contexts.groupby("instance").P_LP.max())
        cri = r[r.method == ex.get("control", "CRI")].groupby(["instance", "budget"]).objective.mean()
        if len(cri):
            per["control"] = [cri.get((i, b_), np.nan) for i, b_ in zip(per.instance, per.budget)]
            per["control certified gap %"] = 100 * (per.control - per.LB_certified) / per.LB_certified
        self.table(f"{name}_gurobi_per_instance", per.set_index(["instance", "method", "budget"]),
                   "Per instance: Gurobi incumbent, dual bound, MIP gap, status (Minor comment 1)", digits=3)
        # progress of a few instances
        shown = 0
        for inst in g.instance.unique()[: ex.get("progress_examples", 3)]:
            series = {}
            for k, tr in self.traces.items():
                if k[0] == name and k[1] == inst and k[4] == r.seed.min():
                    if not tr:
                        continue
                    t_ = [x[0] for x in tr] + [k[3] or tr[-1][0]]
                    v_ = [x[1] for x in tr] + [tr[-1][1]]
                    series[f"{k[2]} incumbent ({k[3]:g} s)"] = (np.maximum(t_, 1e-3), v_, "inc", k[2])
                    bt = self.bound_traces.get(k)
                    if bt:
                        series[f"{k[2]} bound ({k[3]:g} s)"] = (np.maximum([x[0] for x in bt], 1e-3), [x[1] for x in bt],
                                                                "bnd", k[2])
            if series:
                fig = F.solver_progress(series, self.styles, f"Incumbent and bound over time, {inst}")
                lb = self.ctx_lb().get(inst)
                if lb == lb and lb is not None:
                    fig.axes[0].axhline(lb, color=F.INK2, lw=0.8)
                    fig.axes[0].text(fig.axes[0].get_xlim()[0], lb, " certified LB", va="bottom", fontsize=8,
                                     color=F.INK2)
                self.figure(fig, f"{name}_progress_{shown}", f"Incumbents (solid) and dual bounds (dashed), {inst}")
                shown += 1

    def across_budgets(self, name, ex, r):
        """Every method at every budget of the block (and, through ladder(), across blocks)."""
        self.at(self.block, "across_budgets", prefix=name + "_", desc=BLOCK_PARTS["across_budgets"])
        self.h(3, "Across budgets")
        self._ladder(name, r[r.budget.notna() & (r.status != "not_available")], f"{name}")

    def _ladder(self, tag, r, title):
        metric = self.metric
        if "info_class" in r:
            r = r[r.info_class != "CERTIFY"]
        per = r.groupby(["method", "budget", "instance"])[metric].mean().reset_index()
        g = per.groupby(["method", "budget"])[metric].mean().unstack("budget")
        if g.shape[1] < 2:
            return
        g.columns = [f"{c:g} s" for c in g.columns]
        self.table(f"{tag}_gap_by_budget", g, f"Mean {metric} (per-instance mean over seeds) of every method at every "
                   "budget", digits=3, bold_min=list(g.columns))
        t = r.groupby(["method", "budget"]).time.mean().unstack("budget")
        t.columns = [f"{c:g} s" for c in t.columns]
        self.table(f"{tag}_time_by_budget", t, "Mean charged time (s) at every budget", digits=3)
        d = per.groupby(["method", "budget"])[metric].median().reset_index().rename(columns={metric: "y"})
        d = d[d.groupby("method").budget.transform("nunique") >= 2]
        q = per.groupby(["method", "budget"])[metric]
        d["y_lo"] = q.quantile(0.25).reindex(list(zip(d.method, d.budget))).values
        d["y_hi"] = q.quantile(0.75).reindex(list(zip(d.method, d.budget))).values
        if len(d):
            self.figure(F.scaling(d, "budget", "y", self.styles, f"{metric} against the budget, {title}",
                                  "budget (s)", f"median {metric} (%)", logy=False), f"{tag}_gap_vs_budget",
                        f"Median {metric} (IQR band) against the time budget: the time-quality trade-off of each method")

    def budget_ladders(self):
        for lad in self.cfg.get("report", {}).get("budget_ladders", []) or []:
            blocks = [b for b in lad.get("blocks", []) if b in set(self.r.experiment)]
            if len(blocks) < 2:
                continue
            rr = self.r[self.r.experiment.isin(blocks) & self.r.budget.notna() & (self.r.status != "not_available")]
            common = set.intersection(*[set(rr[rr.experiment == b].instance) for b in blocks])
            rr = rr[rr.instance.isin(common)]
            keep = rr.groupby("method").experiment.nunique()
            rr = rr[rr.method.isin(keep[keep >= 2].index)]
            if rr.empty:
                continue
            self.at("30_budget_ladders", lad["name"], prefix=lad["name"] + "_",
                    desc=f"time-quality ladder over {', '.join(blocks)}: the methods present in at least two of these "
                         f"blocks, on the {len(common)} instances they share")
            self.h(3, f"Budget ladder {lad['name']}")
            self.p(f"Blocks {', '.join(blocks)}; {len(common)} common instances; methods present in at least two "
                   "blocks.")
            self._ladder(lad["name"], rr, lad["name"])

    def faithfulness_section(self):
        p = self.cfg.get("report", {}).get("faithfulness")
        p = resolve_path(self.cfg, p) if p else os.path.join(results_dir(self.out, FAITHFULNESS), "faithfulness.pkl")
        if not os.path.exists(p):
            return
        self.h(2, "Faithfulness of the re-implemented earlier methods")
        df = pd.read_pickle(p)
        self.p("Original code (verbatim copy, faithfulness/reference_old) against the Numba re-implementation: Mann-Whitney U "
               "on the peaks of R runs per instance and method, Holm-adjusted. RLP-sameLP feeds both with the same LP "
               "solution (isolates the rounding and repair logic from the choice of the optimal LP vertex).")
        summ = df.groupby("method").agg(pairs=("instance", "size"), mean_rel_diff_pct=("rel_diff_pct", "mean"),
                                        max_abs_rel_diff_pct=("rel_diff_pct", lambda x: x.abs().max()),
                                        min_p_holm=("p_holm", "min"), significant=("p_holm", lambda x: int((x < 0.01).sum())))
        self.table("faithfulness_summary", summ, "Faithfulness per method", digits=3)
        self.table("faithfulness_detail", df.set_index(["instance", "method"]), "Faithfulness per instance and method",
                   digits=3, show=pd.DataFrame())

    def theory_section(self):
        # written by step 3 of run_all.bat into 05_logs; report.theory_verification may point elsewhere
        rp = self.cfg.get("report", {}).get("theory_verification")
        p = resolve_path(self.cfg, rp) if rp else os.path.join(self.out, LOGS, "theory_verification.csv")
        if not os.path.exists(p):
            return
        self.h(2, "Numerical verification of the theory")
        df = pd.read_csv(p)
        tot, fails = int(df.tests.sum()), int(df.failures.sum())
        self.p(f"Every result of Sections 3-4 checked numerically (theory/verify_theory.py): **{tot:,} tests, "
               f"{fails} failures**.")
        self.table("theory_verification", df.set_index("result"), "Verification of each statement", digits=0)

    def ctx_lb(self):
        if self.contexts is None or not len(self.contexts) or "LB" not in self.contexts:
            return pd.Series(dtype=float)
        return self.contexts.groupby("instance").LB.max()

    def tuning(self, name, ex, r):
        self.at(self.block, "tuning", prefix=name + "_", desc=BLOCK_PARTS["tuning"])
        self.h(3, "Tuning")
        t = self.store.load_table(f"tuning_{name}")
        if t is None or not len(t):
            self.p("No tuning table.")
            return
        for m, g in t.groupby("method"):
            self.table(f"{name}_{m}", g.drop(columns="method").set_index("point"),
                       f"Tuning grid of {m} (mean relative gap to the best value per tuning instance)", digits=4,
                       bold_min=["mean_rel_gap"], show=g.drop(columns="method").set_index("point").sort_values(
                           "mean_rel_gap").head(15))
            pts = pd.DataFrame([json.loads(x) for x in g.point])
            rows = []
            for prm in pts.columns:
                for val, idx in pts.groupby(prm).groups.items():
                    rows.append({"parameter": prm, "value": f"{val:g}" if isinstance(val, float) else str(val),
                                 "mean_rel_gap": float(g.iloc[list(idx)].mean_rel_gap.mean()),
                                 "mean_time": float(g.iloc[list(idx)].mean_time.mean())})
            if rows:
                self.table(f"{name}_{m}_marginal", pd.DataFrame(rows).set_index(["parameter", "value"]),
                           f"Marginal effect of each parameter of {m} (mean over the other parameters)", digits=4)

    def managerial(self, name, ex, r):
        self.at(self.block, "managerial", prefix=name + "_", desc=BLOCK_PARTS["managerial"])
        self.h(3, "Managerial comparison")
        base = ex.get("baselines", ["ASAP", "GNP"])
        main = ex.get("control", "CRI")
        M = MT.per_instance(r, "objective")
        if main not in M:
            return
        rows = []
        for b in base:
            if b not in M:
                continue
            red = 100 * (M[b] - M[main]) / M[b]
            rows.append({"reference policy": b, "mean peak reduction (%)": red.mean(), "median": red.median(),
                         "min": red.min(), "max": red.max(), "instances": int(red.notna().sum())})
        self.table(f"{name}_reduction", pd.DataFrame(rows).set_index("reference policy"),
                   f"Peak reduction of {main} against the reference policies", digits=2)
        if "family" in r:
            fam = r.groupby(["family", "method"]).objective.mean().unstack()
            self.table(f"{name}_peaks_by_family", fam, "Mean peak (kW) by family", digits=2)
        # load profiles of a few instances
        for k, inst in enumerate(r.instance.unique()[: ex.get("profile_examples", 2)]):
            d = self.instances.get(inst)
            if d is None:
                continue
            w = np.asarray(d["w"], float)
            prof = {}
            for key, x in self.solutions.items():
                if key[0] == name and key[1] == inst and key[2] in [main] + list(base) and key[2] not in prof:
                    prof[key[2]] = (w[:, None] * x).sum(axis=0)
            if prof:
                self.figure(F.load_profiles(prof, self.styles, self.ref.LB_star.get(inst), f"Load profiles, {inst}",
                                            24.0 / d["T"]), f"{name}_profile_{k}", f"Charging load over the day, {inst}")

    def sota(self, name, ex, r):
        self.h(3, "State-of-the-art baselines")
        na = r[r.status == "not_available"].method.unique()
        if len(na):
            self.p("Still to be integrated (registered placeholders, no runs): " + ", ".join(f"`{m}`" for m in na) + ".")
        self.comparison(name, ex, r[r.status != "not_available"])

    # -- driver --
    def build(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        self.load()
        for folder, fn in (("00_campaign_overview", self.overview), ("01_instances", self.instances_section),
                           ("02_exact_references_and_bounds", self.references_section),
                           ("03_validity_checks", self.validity_section),
                           ("04_faithfulness_of_earlier_methods", self.faithfulness_section),
                           ("05_theory_verification", self.theory_section)):
            self.at(folder, desc=SECTIONS[folder])
            fn()
            if folder == "00_campaign_overview":
                self._headline_at = (len(self.md), len(self.toc))
        for i, (name, ex) in enumerate(self.cfg["experiments"].items()):
            self.block = f"{10 + i:02d}_{name}"
            r = self.r[self.r.experiment == name]
            for inc in ex.get("include", []):
                # runs of another block reused here (same instances and budgets, listed methods)
                o = self.r[(self.r.experiment == inc["experiment"]) & self.r.method.isin(inc.get("methods", []))
                           & self.r.instance.isin(r.instance.unique()) & self.r.budget.isin(r.budget.unique())]
                r = pd.concat([r, o], ignore_index=True)
            if r.empty:
                continue
            self.h(2, f"{name}")
            self.at(self.block, prefix=name + "_", desc=ex.get("description", name))
            self.p(f"Files: `{self.block}/`")
            if ex.get("description"):
                self.p(ex["description"])
            if ex.get("addresses"):
                self.p("*Addresses:* " + ex["addresses"])
            an = list(ex.get("analyses", ["comparison"]))
            if any(a in an for a in ("comparison", "sota")) and r.budget.dropna().nunique() >= 2:
                an.append("across_budgets")
            for a in an:
                try:
                    getattr(self, a)(name, ex, r)
                except Exception as e:                     # one failing analysis never stops the report
                    import traceback
                    self.p(f"> analysis `{a}` failed: `{e!r}`\n\n```\n{traceback.format_exc()[-1500:]}\n```")
        if self.cfg.get("report", {}).get("budget_ladders"):
            self.at("30_budget_ladders", desc="time-quality ladders across blocks: every method "
                    "at every budget from the shortest to the longest runs, on common instances")
            self.h(2, "Budget ladders across blocks")
            try:
                self.budget_ladders()
            except Exception as e:
                self.p(f"> analysis `budget_ladders` failed: `{e!r}`")
        if self.headline:                               # written last, shown right after the overview
            self.at("00_campaign_overview")
            md, toc = self.md, self.toc
            self.md, self.toc = [], []
            self.h(2, "Headline results of every block")
            self.table("headline_results", pd.DataFrame(self.headline).set_index(["block", "budget"]),
                       "Per block and budget: best method by the mean gap over instances, the control's mean gap and "
                       "average rank, and the Friedman p-value (details in each block's statistics/)", digits=3)
            pos, tpos = getattr(self, "_headline_at", (len(md), len(toc)))
            self.md, self.toc = md[:pos] + self.md + md[pos:], toc[:tpos] + self.toc + toc[tpos:]
        return self.write()

    def write(self):
        toc = "\n".join(f"- [{t}](#{a})" for t, a in self.toc)
        body = self.md[0] + "\n**Contents**\n\n" + toc + "\n" + "\n".join(self.md[1:])
        mdp = os.path.join(self.root, "report.md")
        with open(mdp, "w", encoding="utf-8") as f:
            f.write(body)
        try:
            import markdown
            html = markdown.markdown(body, extensions=["tables", "toc", "fenced_code"])
        except Exception:
            html = "<pre>" + body + "</pre>"
        title = self.cfg.get("report", {}).get("title", "Experimental report")
        page = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
                f"content='width=device-width,initial-scale=1'><title>{title}</title><style>{CSS}</style></head>"
                f"<body>{html}</body></html>")
        hp = os.path.join(self.root, "report.html")
        with open(hp, "w", encoding="utf-8") as f:
            f.write(page)
        rows = [{"kind": "table", "file": p_ + ".csv", "caption": c} for p_, c in self.T.index] + \
               [{"kind": "figure", "file": p_ + ".png", "caption": c} for p_, c in self.Fg.index]
        idx = pd.DataFrame(rows, columns=["kind", "file", "caption"]).sort_values(["file", "kind"])
        with open(os.path.join(self.root, "index.md"), "w", encoding="utf-8") as f:
            f.write(f"# Every table and figure of the report ({len(self.T.index)} tables, {len(self.Fg.index)} figures)\n\n"
                    "Tables exist as .csv, .md and .tex; figures as .png and .pdf.\n\n")
            f.write(to_markdown(idx, index=False) + "\n")
        lines = ["# Report folders", "", "Open `report.html` (or `report.md`) first; `index.md` lists every table and "
                 "figure with its caption. Inside each folder: `tables/` (.csv .md .tex), `statistics/` (tests, ranks, "
                 "effect sizes and their diagrams) and `figures/` (.png .pdf).", "", "| folder | content | subfolders |",
                 "|---|---|---|"]
        top = sorted(f for f in self.folders if "/" not in f and os.path.isdir(os.path.join(self.root, f)))
        for folder in top:
            subs = sorted(f.split("/", 1)[1] for f in self.folders if f.startswith(folder + "/")
                          and os.path.isdir(os.path.join(self.root, f)))
            lines.append(f"| `{folder}/` | {self.folders[folder]} | " + ", ".join(f"`{x}/`" for x in subs) + " |")
        lines += ["", "**Subfolders of a block**", "", "| subfolder | content |", "|---|---|",
                  "| `budget_<B>s/`, `single_run/` | " + BLOCK_PARTS["budget"] + " |"]
        lines += [f"| `{k}/` | {v} |" for k, v in BLOCK_PARTS.items() if k != "budget"]
        with open(os.path.join(self.root, "README.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return mdp, hp


def build_report(cfg):
    return Report(cfg).build()
