"""
Tables and figures of Section 6 (Computational Experiments) of the paper, computed from the results
folder of a campaign.

    python tools/paper_artifacts.py <results folder> <paper folder> [--numbers FILE]

Reads 02_aggregated_data (runs_enriched, instance_references, traces, solutions, instances) and the
statistics tables of 03_report (its Friedman ranks, Wilcoxon signed-rank tests with Holm's correction and
win counts), so that the paper and the report agree to the last digit. Writes <paper>/figures/*.pdf,
<paper>/tables/*.tex (complete table environments, \\input by sections/experiments.tex) and, with
--numbers, a markdown file with every number quoted in the text. Reads only; the results are not changed.
"""

import argparse
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.ERROR)        # the cmr10 font's header timestamps

# ---------------------------------------------------------------------------------------------- style
# categorical palette in fixed order (validated: adjacent-pair CVD and normal-vision separation pass);
# every series also has its own line style and marker, so identity never rests on color alone
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
DASHES = ["-", (0, (5, 2)), (0, (1, 1.2)), (0, (6, 2, 1, 2)), (0, (3, 1.5)), (0, (4, 1.5, 1, 1.5, 1, 1.5)),
          (0, (8, 2)), (0, (2, 2))]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# display names (Section 5) and their order in the tables
NAME = {
    "CRI": "CRI", "CRI+C": "CRI+C", "LLS": "LLS", "WLNS": "W-LNS", "HSC-b": "HSC", "SA-b": "SA", "RR-b": "RR",
    "RLP-b": "RLP", "G-b": "G", "BK": "BK", "GUROBI-std": "Gurobi-std", "GUROBI-agg": "Gurobi-agg",
    "GUROBI-std-4t": "Gurobi-std, 4 threads", "GUROBI-agg-4t": "Gurobi-agg, 4 threads",
    "NFDH": "NFDH", "ZAIDI_H": "Zaidi-H", "ZAIDI_ILS": "Zaidi-ILS", "ZAIDI_ILS-b": "Zaidi-ILS-r",
    "HUNSAKER": r"Hunsaker-$\alpha$", "AFSHAR_BB": "Afshar-BB", "AN_HYBRID": "AN-SER", "AN-PAR": "AN-PAR",
    "AN-dev": "AN-SER-dev", "ASAP": "ASAP", "GNP": "GNP", "G": "G", "RR": "RR", "RLP": "RLP", "SA": "SA",
    "HSG": "HSG", "HSR": "HSR", "HSL": "HSL", "IR": "IR",
    "MT-lp": r"MT$(\alpha^*)$", "MT-cut": r"MT$(\alpha^{\mathrm{cut}})$", "MT-conf": r"MT$(\alpha^{\mathrm{conf}})$",
    "MT-lp-b": r"MT$(\alpha^*)$", "MT-cut-b": r"MT$(\alpha^{\mathrm{cut}})$",
    "CE-lp-excess": r"CE$_E(\alpha^*)$", "CE-cut-excess": r"CE$_E(\alpha^{\mathrm{cut}})$",
    "CE-conf-comp": r"CE$_\Gamma(\alpha^{\mathrm{conf}})$", "CRI-R": "CRI-R",
    "CE-cut-union": r"CE$_\cup(\alpha^{\mathrm{cut}})$",
}
FIG_ORDER = ["CRI", "GUROBI-std", "LLS", "GUROBI-agg", "WLNS", "SA-b", "CRI+C", "RR-b"]   # slot order
SOTA_FIG = ["CRI", "GUROBI-std", "AN_HYBRID", "GUROBI-agg", "AN-PAR", "BK", "ZAIDI_H", "ZAIDI_ILS-b"]


def plain(m):
    """Display name without LaTeX markup (figure legends)."""
    return {"HUNSAKER": "Hunsaker-α"}.get(m, NAME.get(m, m))


def style(order, m):
    k = order.index(m)
    return dict(color=PALETTE[k], linestyle=DASHES[k], marker=MARKERS[k])


# ---------------------------------------------------------------------------------------------- formats
def fg(x):
    """A gap in %."""
    if x is None or not np.isfinite(x):
        return "n/a"
    if abs(x) < 0.005:
        return "0.00"
    return f"{x:.2f}" if x < 10 else (f"{x:.1f}" if x < 100 else f"{x:.0f}")


def fr(x, d=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def ft(x):
    """A time in seconds."""
    return "n/a" if x is None or not np.isfinite(x) else ("$<$0.01" if x < 0.01 else f"{x:.2f}")


def fp(p):
    """A p-value in LaTeX."""
    if not np.isfinite(p):
        return "n/a"
    if p >= 0.001:
        return f"{p:.3f}"
    e = int(np.floor(np.log10(p)))
    return f"$<10^{{{e + 1}}}$"


def wtl(row):
    """CRI's wins/ties/losses against a method; a dagger when Holm's test is not significant at 5%."""
    s = f"{int(row.wins)}/{int(row.ties)}/{int(row.losses)}"
    return s + (r"$^\dagger$" if row.p_holm >= 0.05 else "")


def bold(s):
    return r"\textbf{" + s + "}"


def fn(n):
    """An integer with a LaTeX thousands separator."""
    return f"{int(n):,}".replace(",", "{,}")


# ---------------------------------------------------------------------------------------------- data
class Campaign:
    def __init__(self, res):
        self.res = res
        D = os.path.join(res, "02_aggregated_data")
        self.rep = os.path.join(res, "03_report")
        self.e = pd.read_pickle(os.path.join(D, "runs_enriched.pkl"))
        self.ref = pd.read_pickle(os.path.join(D, "instance_references.pkl"))
        self.traces = pd.read_pickle(os.path.join(D, "traces.pkl"))
        self.solutions = pd.read_pickle(os.path.join(D, "solutions.pkl"))
        saved = pd.read_pickle(os.path.join(D, "instances.pkl"))
        self.instances = {d["name"]: d for v in saved.get("sets", {}).values() for d in v}
        self.numbers = []

    def csv(self, rel):
        return pd.read_csv(os.path.join(self.rep, rel))

    def stats(self, block, budget):
        """CRI_vs_each_method and the average ranks of a block at one budget (from the report)."""
        b = f"budget_{budget:g}s" if budget else "single_run"
        folder = next(f for f in os.listdir(self.rep) if f.endswith(block))
        vs = self.csv(f"{folder}/{b}/statistics/CRI_vs_each_method.csv").set_index("method")
        rk = self.csv(f"{folder}/{b}/statistics/friedman_average_ranks.csv").set_index("method")["average rank"]
        su = self.csv(f"{folder}/{b}/tables/summary.csv").set_index("method")
        fr_ = self.csv(f"{folder}/{b}/statistics/friedman_iman_davenport_test.csv").iloc[0]
        return vs, rk, su, fr_

    def note(self, key, value):
        self.numbers.append((key, value))


# ---------------------------------------------------------------------------------------------- tables
RAG = r">{\raggedright\arraybackslash}p"        # wrapped column without justified spacing (array package)


def table_env(label, caption, colspec, header, rows, notes=None, size=r"\footnotesize", star=False):
    out = [r"\begin{table}[htbp]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
           "{" + size, r"\begin{tabular}{" + colspec + "}", r"\toprule"]
    out += header
    out.append(r"\midrule")
    for r in rows:
        rule = r.startswith(("\\midrule", "\\addlinespace", "\\cmidrule")) or r.rstrip().endswith("\\\\")
        out.append(r if rule else r + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", "}"]
    if notes:
        out += [r"\par\smallskip", r"\begin{minipage}{0.97\textwidth}\scriptsize " + notes + r"\end{minipage}"]
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def t_instances(C, out):
    f = C.e.drop_duplicates("instance")[["instance", "set", "family", "n", "T"]]
    use = {"tune": "tuning only (E8)", "small": "proven optima (exact references)",
           "omega_published": "sizes of the earlier benchmark", "medium": "main comparisons",
           "large": "main comparisons, long runs", "xlarge": "scalability, long runs",
           "special": "special cases (Prop.~\\ref{prop:np-hard}, Cor.~\\ref{cor:single-class})"}
    fam = {"omega": "omega", "staggered": "staggered", "overnight": "overnight", "pcmax": "common window",
           "single": "one power level"}
    rows = []
    for s in ["tune", "small", "omega_published", "medium", "large", "xlarge", "special"]:
        g = f[f.set == s]
        if g.empty:
            continue
        ns = sorted(g.n.unique())
        Ts = sorted(g["T"].unique())
        per = g.groupby(["family", "n"]).size()
        rows.append(f"\\texttt{{{s.replace('_', chr(92) + '_')}}} & {', '.join(fam[x] for x in g.family.unique())} & "
                    f"{', '.join(f'{x:,}'.replace(',', '{,}') for x in ns)} & {', '.join(str(x) for x in Ts)} & "
                    f"{per.min()}{'' if per.min() == per.max() else ' to ' + str(per.max())} & {len(g)} & {use[s]}")
    tot = len(f)
    C.note("instances in total", tot)
    open(os.path.join(out, "instances.tex"), "w", encoding="utf-8").write(table_env(
        "tab:instances", "Instance sets", "l" + RAG + "{2.5cm}" + RAG + "{3.0cm}" + RAG + "{1.5cm}cr" + RAG + "{3.2cm}",
        [r"\textbf{Set} & \textbf{Families} & $n$ & $T$ & \textbf{Per pair} & \textbf{Total} & "
         r"\textbf{Use} \\"], rows,
        notes=f"{tot} instances in total. Per pair: instances per family and value of $n$. Every set is generated "
              "with its own seed; the tuning set is used only to choose parameters and never for evaluation."))


def t_blocks(C, out):
    st = C.csv("00_campaign_overview/tables/campaign_status.csv").set_index("experiment")
    runs = (st.sum(axis=1)).to_dict()
    B = [
        ("E1", "E1_certificates", "small, omega\\_published, medium, large, special", "certificates only", "none",
         "\\ref{subsec:exp-certificates}"),
        ("E2", "E2_head_to_head", "small, omega\\_published, medium", "earlier methods and their certified counterparts",
         "own rule", "\\ref{subsec:exp-roundings}"),
        ("E2b", "E2b_head_to_head_budgets", "small, medium", "the same pairs, anytime versions", "1, 10",
         "\\ref{subsec:exp-roundings}"),
        ("E4a", "E4a_equal_budgets", "omega\\_published, medium, large", "all methods, Gurobi (2 models)", "1, 10",
         "\\ref{subsec:exp-equal}"),
        ("E4b", "E4b_equal_budgets_60s", "medium, large", "anytime methods, Gurobi", "60",
         "\\ref{subsec:exp-equal}"),
        ("E3", "E3_long_runs", "medium, large (one per family and $n$)", "CRI, CRI+C, Gurobi", "600",
         "\\ref{subsec:exp-long}"),
        ("E3x", "E3x_long_runs_xlarge", "xlarge", "CRI, CRI+C, Gurobi", "600", "\\ref{subsec:exp-long}"),
        ("E6", "E6_scalability", "large, xlarge (one per family and $n$)", "fast methods, CRI, Gurobi", "120",
         "\\ref{subsec:exp-long}"),
        ("E7", "E7_gurobi_threads", "medium, large (one per family and $n$)", "CRI against Gurobi with 4 threads",
         "10, 60", "\\ref{subsec:exp-long}"),
        ("E10a", "E10a_state_of_the_art", "omega\\_published, medium, large", "methods from the literature",
         "10", "\\ref{subsec:exp-sota}"),
        ("E10b", "E10b_state_of_the_art_60s", "medium, large", "anytime methods from the literature", "60",
         "\\ref{subsec:exp-sota}"),
        ("E5", "E5_ablation", "medium", "configurations of CRI, SA starts", "10", "\\ref{subsec:exp-ablation}"),
        ("E5b", "E5b_ablation_60s", "medium", "Improve stage", "60", "\\ref{subsec:exp-ablation}"),
        ("E9", "E9_managerial", "medium, large", "CRI against reference policies", "10",
         "\\ref{subsec:exp-managerial}"),
        ("E8", "E8a_tuning_faithful", "tune", "tuning of SA (original and anytime), LLS and W-LNS", "own rule, 10",
         "\\ref{subsec:exp-setup}"),
    ]
    rows = []
    for short, key, sets, methods, bud, sec in B:
        n = runs.get(key, 0) + (runs.get("E8b_tuning_budget", 0) if short == "E8" else 0)
        rows.append(f"{short} & {sets} & {methods} & {bud} & {fn(n)} & {sec}")
    total = int(sum(runs.values()))
    C.note("runs in total", total)
    e0 = int(runs.get("E0_sanity", 0))
    with_e0 = f", including {fn(e0)} in a sanity check that runs every method once on a few small instances" if e0 else ""
    open(os.path.join(out, "blocks.tex"), "w", encoding="utf-8").write(table_env(
        "tab:blocks", "Experimental blocks", "l" + RAG + "{4.1cm}" + RAG + "{4.3cm}lll",
        [r"\textbf{Block} & \textbf{Instances} & \textbf{Methods} & \textbf{Budget (s)} & \textbf{Runs} & "
         r"\textbf{Section} \\"], rows,
        notes=f"{fn(total)} runs in total, all on the same machine{with_e0}. ``Own rule'': each method stops by its own rule."))


def t_certificates(C, out):
    c = C.e[C.e.experiment == "E1_certificates"].copy()
    ref = c.instance.map(C.ref.OPT).fillna(c.instance.map(C.ref.BKS))
    opt_known = c.instance.map(C.ref.OPT).notna()

    def best(cols):
        cols = [k for k in cols if k in c]
        return c[cols].min(axis=1, skipna=True) if cols else pd.Series(np.nan, index=c.index)

    pub = ["info_published_hoeffding_union", "info_published_bernstein_union", "info_published_hoeffding_lll",
           "info_published_bernstein_lll"]
    closed = [k for k in c if k.startswith(("info_lp_closed_", "info_cut_closed_", "info_conf_closed_"))
              and not k.endswith(("_d_random", "_largest_component", "_n_components"))]
    exact = [k for k in c if k.startswith(("info_lp_", "info_cut_", "info_conf_")) and k.split("_", 2)[-1] in
             ("union", "component_union", "lll")]
    averaging = [k for k in c if k.endswith(("_averaging",)) and not k.split("_")[2] == "dr"
                 and k.startswith(("info_lp_", "info_cut_", "info_conf_"))]
    dr = [k for k in c if "_dr_" in k]
    dp = [k for k in c if k.endswith(("_exist_full", "_exist_sys"))]
    rows_lo = [("$\\mathbf{P}^{\\mathrm{LP}}$", "Prop.~\\ref{prop:lp-lower}", c.info_P_LP),
               ("$\\lceil\\mathbf{P}^{\\mathrm{LP}}\\rceil_{\\mathcal{A}}$", "Sec.~\\ref{subsec:lower}",
                c.info_LB_LP_rounded),
               ("$\\mathrm{LB}_{\\mathrm{cut}}$", "Thm.~\\ref{thm:cuts}", c.info_LB_cut),
               ("$\\mathrm{LB}$ (eq.~\\eqref{eq:LB})", "Thm.~\\ref{thm:conf}", c.info_LB)]
    rows_up = [("Non-localized bounds, best of four", "Cor.~\\ref{cor:published}", best(pub)),
               ("$\\mathbf{P}^{\\mathrm{LP}}+w_{\\max}$", "Thm.~\\ref{thm:rounding}", c.info_LP_plus_wmax),
               ("Closed forms, best", "Thm.~\\ref{thm:closed}", best(closed)),
               ("Exact slot laws (union, components, LLL), best", "Thm.~\\ref{thm:exact-cert}", best(exact)),
               ("Averaging, best", "Thm.~\\ref{thm:averaging}", best(averaging)),
               ("Dependent rounding, best", "Thm.~\\ref{thm:dr}", best(dr)),
               ("Optimal rounding of the forest DP, best", "Thm.~\\ref{thm:exact-dp}", best(dp)),
               ("$U$ (eq.~\\eqref{eq:sandwich-U})", "Thm.~\\ref{thm:sandwich}", c.info_U_best)]
    rows = [r"\multicolumn{7}{l}{\emph{Lower certificates}} \\"]
    has_ref = ref.notna()                       # the special-case instances run the certificates only
    for name, src, v in rows_lo:
        ok = v.notna() & has_ref
        r = (v / ref)[ok]
        viol = int((v[ok] > ref[ok] + 1e-6).sum())
        eq = int((np.abs(v[ok] - ref[ok]) < 1e-6).sum())
        rows.append(f"{name} & {src} & {int(ok.sum())} & {fr(r.mean())} & {fr(r.min())} & {eq} & {viol}")
        C.note(f"E1 lower {name}", f"mean {r.mean():.4f}, min {r.min():.4f}, equal {eq}/{int(ok.sum())}, violations {viol}")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{7}{l}{\emph{Upper certificates}} \\")
    for name, src, v in rows_up:
        ok = v.notna() & has_ref
        r = (v / ref)[ok]
        viol = int(((v[ok] < ref[ok] - 1e-6) & opt_known[ok]).sum())
        eq = int((np.abs(v[ok] - ref[ok]) < 1e-6).sum())
        rows.append(f"{name} & {src} & {int(ok.sum())} & {fr(r.mean())} & {fr(r.max())} & {eq} & {viol}")
        C.note(f"E1 upper {name}", f"mean {r.mean():.4f}, max {r.max():.4f}, equal {eq}/{int(ok.sum())}, violations {viol}")
    apo = int(c.info_a_priori_optimal.astype(float).fillna(0).sum())
    C.note("E1 instances", len(c))
    C.note("E1 a priori optimal", f"{apo} of {len(c)} ({100 * apo / len(c):.1f}%)")
    C.note("E1 references proven optimal", int(opt_known.sum()))
    C.note("E1 instances with a reference", int(has_ref.sum()))
    sp = c.set.eq("special")
    C.note("E1 a priori optimal among special", int(c.info_a_priori_optimal[sp].astype(float).fillna(0).sum()))
    # same LP vertex: earlier bounds against U(alpha*)
    pub_lp = best(pub)
    lp_best = best([k for k in c if k.startswith("info_lp_")
                    and not k.endswith(("_d_random", "_largest_component", "_n_components", "_emax_exact"))])
    C.note("E1 earlier bounds / U(alpha*) on the same vertex", f"{(pub_lp / lp_best).mean():.3f}")
    gap2 = 100 * (c.info_U_best - c.info_LB) / c.info_LB
    for lo, hi in ((10, 30), (40, 200), (500, 2000)):
        m = c.n.between(lo, hi) & c.family.isin(["omega", "staggered", "overnight"])
        C.note(f"E1 two-sided gap (U-LB)/LB, n {lo}-{hi}", f"mean {gap2[m].mean():.2f}%, median {gap2[m].median():.2f}%")
    open(os.path.join(out, "certificates.tex"), "w", encoding="utf-8").write(table_env(
        "tab:certificates",
        "A priori certificates against the optimal or best known peak (block E1; ratio = certificate / reference)",
        "llrrrrr",
        [r"\textbf{Certificate} & \textbf{Result} & \textbf{Inst.} & \textbf{Mean ratio} & \textbf{Worst ratio} & "
         r"\textbf{\# equal} & \textbf{Violations} \\"], rows,
        notes=f"Reference: the proven optimum ({int(opt_known.sum())} instances) or the best known peak, on the "
              f"{int(has_ref.sum())} instances on which schedules were computed (the {int((~has_ref).sum())} special-case "
              "instances ran the certificates only). Worst "
              "ratio: the smallest for lower certificates, the largest for upper certificates. A violation would be a "
              "lower certificate above the reference, or an upper certificate below a proven optimum. The "
              "configuration bound is computed for $n\\le500$ within 60 s; above, $\\mathrm{LB}=\\mathrm{LB}_{\\mathrm{cut}}$. "
              f"The certificates alone prove {apo} of the {len(c)} instances optimal ($\\mathrm{{LB}}=U$)."))
    return c, ref


def t_h2h(C, out):
    a = C.csv("14_E2_head_to_head/head_to_head/tables/pairs_wilcoxon_holm.csv")
    b = C.csv("15_E2b_head_to_head_budgets/head_to_head/tables/pairs_wilcoxon_holm.csv")
    b10 = b[b.budget == 10.0]
    pairs = [("RR", "MT-lp", "RR-b", "MT-lp-b"), ("RR", "MT-cut", "RR-b", "MT-cut-b"),
             ("RLP", "CE-lp-excess", "RLP-b", "CE-lp-excess"), ("RLP", "CE-cut-excess", "RLP-b", "CE-cut-excess"),
             ("RLP", "IR", "RLP-b", "IR"), ("G", "CE-lp-excess", "G-b", "CE-cut-excess"), (None, None, "G-b", "CRI-R"),
             ("SA", "HSG", None, None), ("SA", "HSR", None, None), ("SA", "HSL", None, None)]
    rows = []
    for A, Bm, A2, B2 in pairs:
        cells = []
        for df, x, y in ((a, A, Bm), (b10, A2, B2)):
            if x is None:                      # pair not run in this block: empty half-row (keeps the rule)
                cells.append("\\multicolumn{4}{c|}{}" if df is a else "\\multicolumn{4}{c}{}")
                continue
            r = df[(df.A == x) & (df.B == y)].iloc[0]
            cells.append(f"{NAME.get(x, x)} vs {NAME.get(y, y)} & {fg(r['mean gap_lb A'])} / {fg(r['mean gap_lb B'])} & "
                         f"{int(r['B better'])}/{int(r['ties'])}/{int(r['A better'])} & {fp(r['p (Holm)'])}")
            C.note(f"h2h {df is a and 'own rule' or '10 s'} {x} vs {y}",
                   f"{r['mean gap_lb A']:.3f} vs {r['mean gap_lb B']:.3f}, W/T/L(B) {int(r['B better'])}/{int(r['ties'])}/"
                   f"{int(r['A better'])}, p_holm {r['p (Holm)']:.3g}, N {int(r['N'])}")
        rows.append(" & ".join(cells))
    open(os.path.join(out, "head_to_head.tex"), "w", encoding="utf-8").write(table_env(
        "tab:head-to-head", "Earlier methods against their certified counterparts (blocks E2 and E2b)",
        "lccc|lccc",
        [r"\multicolumn{4}{c|}{\textbf{Own stopping rules (E2, 130 instances, 10 seeds)}} & "
         r"\multicolumn{4}{c}{\textbf{Equal budget of 10 s (E2b, 90 instances, 3 seeds)}} \\",
         r"\textbf{A vs B} & \textbf{Gap A / B (\%)} & \textbf{W/T/L} & $p_{\mathrm{Holm}}$ & "
         r"\textbf{A vs B} & \textbf{Gap A / B (\%)} & \textbf{W/T/L} & $p_{\mathrm{Holm}}$ \\"], rows,
        notes="Gap: mean gap to $\\mathrm{LB}^*$. W/T/L: instances on which B is better than, equal to, worse than A "
              "(mean over seeds). $p_{\\mathrm{Holm}}$: Wilcoxon signed-rank test, Holm-adjusted within the block. In E2b "
              "the earlier methods run in their anytime versions and MT is repeated with new seeds; CE and IR are "
              "single-shot and stop before the budget.", size=r"\scriptsize"))


def t_equal(C, out):
    blocks = [("E4a_equal_budgets", 1), ("E4a_equal_budgets", 10), ("E4b_equal_budgets_60s", 60)]
    S = [C.stats(bl, b) for bl, b in blocks]
    methods = ["CRI", "CRI+C", "LLS", "WLNS", "HSC-b", "SA-b", "RR-b", "RLP-b", "G-b", "BK", "GUROBI-std", "GUROBI-agg"]
    best_gap = [min(s[2].loc[m, "mean gap_lb"] for m in methods if m in s[2].index) for s in S]
    rows = []
    for m in methods:
        cells = [NAME[m]]
        for (vs, rk, su, _), bg in zip(S, best_gap):
            if m not in su.index:
                cells += ["n/a", "n/a", "n/a"]
                continue
            g = su.loc[m, "mean gap_lb"]
            gs = fg(g)
            cells += [bold(gs) if abs(g - bg) < 1e-12 else gs, "n/a" if m == "CRI" else wtl(vs.loc[m]), f"{rk[m]:.2f}"]
        rows.append(" & ".join(cells))
        if m == "LLS":
            rows.append(r"\addlinespace")
        if m == "BK":
            rows.append(r"\addlinespace")
    for (bl, b), (vs, rk, su, f) in zip(blocks, S):
        C.note(f"{bl} {b} s Friedman", f"N {int(f['instances (N)'])}, k {int(f['methods (k)'])}, chi2 {f['chi2']:.1f}, "
                                       f"p {f['p']:.3g}, Iman-Davenport F {f['Iman-Davenport F']:.1f}")
        for m in vs.index:
            r = vs.loc[m]
            C.note(f"{bl} {b} s CRI vs {m}", f"{r.mean_control:.3f} vs {r.mean_method:.3f}, W/T/L {int(r.wins)}/"
                                             f"{int(r.ties)}/{int(r.losses)}, p_holm {r.p_holm:.3g}, cliff {r.cliff_delta:.3f}")
    open(os.path.join(out, "equal_budgets.tex"), "w", encoding="utf-8").write(table_env(
        "tab:equal-budgets", "Comparison at equal wall-clock budgets (blocks E4a and E4b)", "l|ccc|ccc|ccc",
        [r" & \multicolumn{3}{c|}{\textbf{1 s (112 instances)}} & \multicolumn{3}{c|}{\textbf{10 s (112 instances)}} & "
         r"\multicolumn{3}{c}{\textbf{60 s (72 instances)}} \\",
         r"\textbf{Method} & \textbf{Gap (\%)} & \textbf{W/T/L} & \textbf{Rank} & \textbf{Gap (\%)} & \textbf{W/T/L} & "
         r"\textbf{Rank} & \textbf{Gap (\%)} & \textbf{W/T/L} & \textbf{Rank} \\"], rows,
        notes="Gap: mean gap to $\\mathrm{LB}^*$ over instances and seeds (3 seeds at 1 and 10 s, 2 at 60 s; one run "
              "for the deterministic BK and for Gurobi). W/T/L: instances on which CRI is better than, equal to, "
              "worse than the method (mean over seeds); $^\\dagger$: the difference is not significant (Wilcoxon "
              "signed-rank test, Holm-adjusted, 5\\%). Rank: Friedman average rank. The earlier methods (SA, RR, "
              "RLP, G) and HSC run in their anytime versions. Best gap of each budget in bold."))


def t_gurobi(C, out):
    """Gurobi's status, bounds and MIP gaps against CRI's own certificate (Minor comment 1)."""
    e = C.e
    rows = []
    sets = [("E4a_equal_budgets", 1.0, "1"), ("E4a_equal_budgets", 10.0, "10"), ("E4b_equal_budgets_60s", 60.0, "60"),
            ("E3_long_runs", 600.0, "600"), ("E3x_long_runs_xlarge", 600.0, "600 (xlarge)")]
    for bl, b, lab in sets:
        x = e[(e.experiment == bl) & (e.budget == b)]
        ni = x.instance.nunique()
        cells = [lab, str(ni)]
        for m in ("GUROBI-std", "GUROBI-agg"):
            g = x[x.method == m]
            mg = 100 * g.info_mip_gap.astype(float)
            mg = mg[np.isfinite(mg)]
            prov = int((g.info_grb_status == "OPTIMAL").sum())
            cells += [str(prov), fg(mg.median()), fg(g.gap_lb.mean())]
            C.note(f"Gurobi {bl} {b} {m}", f"optimal {prov}/{len(g)}, median MIP gap {mg.median():.3f}%, mean MIP gap "
                                          f"{mg.mean():.3f}% (finite {len(mg)}/{len(g)}), mean gap_lb {g.gap_lb.mean():.3f}")
        cr = x[x.method == "CRI"]
        cg = 100 * cr.info_gap_certified.astype(float)
        hit = int((cr.info_gap_certified <= 1e-12).sum())
        cells += [f"{hit}/{len(cr)}", fg(cg.mean()), fg(cr.gap_lb.mean())]
        C.note(f"CRI certified {bl} {b}", f"certified optimal {hit}/{len(cr)}, mean certified gap {cg.mean():.3f}%, "
                                         f"median {cg.median():.3f}%, mean gap_lb {cr.gap_lb.mean():.3f}")
        rows.append(" & ".join(cells))
    open(os.path.join(out, "gurobi_bounds.tex"), "w", encoding="utf-8").write(table_env(
        "tab:gurobi", "Gurobi's termination, reported MIP gap and gap to $\\mathrm{LB}^*$, against CRI's own certificate",
        "lr|ccc|ccc|ccc",
        [r" & & \multicolumn{3}{c|}{\textbf{Gurobi-std}} & \multicolumn{3}{c|}{\textbf{Gurobi-agg}} & "
         r"\multicolumn{3}{c}{\textbf{CRI}} \\",
         r"\textbf{Budget (s)} & \textbf{Inst.} & \textbf{Opt.} & \textbf{MIP gap} & \textbf{Gap} & \textbf{Opt.} & "
         r"\textbf{MIP gap} & \textbf{Gap} & \textbf{Certified} & \textbf{Cert. gap} & \textbf{Gap} \\"], rows,
        notes="Opt.: runs that Gurobi proves optimal. MIP gap: median of the gap Gurobi reports between its incumbent "
              "and its own dual bound (\\%). Gap: mean gap to $\\mathrm{LB}^*$ (\\%). Certified: CRI runs whose peak "
              "equals CRI's own lower certificate, which proves them optimal without a solver. Cert. gap: mean of "
              "CRI's certified gap $(\\mathbf{P}(X)-\\mathrm{LB})/\\mathrm{LB}$, computed from its own certificate "
              "(\\%). Rows 1 to 60 s: blocks E4a and E4b; 600 s: E3 (18 instances) and E3x (5 instances)."))


def t_long(C, out):
    e = C.e
    rows = []
    groups = [("E3_long_runs", "medium", "medium ($n=50$ to 200)"), ("E3_long_runs", "large", "large ($n=500$ to 2{,}000)"),
              ("E3x_long_runs_xlarge", "xlarge", "xlarge ($n=5{,}000$ to 10{,}000)")]
    for bl, s, lab in groups:
        x = e[(e.experiment == bl) & (e.set == s)]
        best = x.groupby("instance").objective.min()
        first = True
        for m in ("CRI", "CRI+C", "GUROBI-std", "GUROBI-agg"):
            g = x[x.method == m]
            nb = int((g.objective.values <= g.instance.map(best).values + 1e-9).sum())
            if m.startswith("GUROBI"):
                prov = int((g.info_grb_status == "OPTIMAL").sum())
                own = 100 * g.info_mip_gap.astype(float)
            else:
                prov = int((g.info_gap_certified <= 1e-12).sum())
                own = 100 * g.info_gap_certified.astype(float)
            cells = [(f"\\multirow{{4}}{{*}}{{{lab}}}" if first else ""), NAME[m], str(len(g)), fg(g.gap_lb.mean()),
                     fg(g.gap_lb.max()), str(nb), str(prov), fg(own.mean())]
            first = False
            rows.append(" & ".join(cells))
            C.note(f"long {s} {m}", f"mean gap {g.gap_lb.mean():.4f}, max {g.gap_lb.max():.3f}, best {nb}, proven {prov}, "
                                    f"own gap {own.mean():.3f}")
        rows.append(r"\midrule" if s != "xlarge" else r"\addlinespace[0pt]")
    rows = rows[:-1]
    open(os.path.join(out, "long_runs.tex"), "w", encoding="utf-8").write(table_env(
        "tab:long-runs", "Long runs of 600 s (blocks E3 and E3x)", "llrrrrrr",
        [r"\textbf{Instances} & \textbf{Method} & \textbf{Inst.} & \textbf{Mean gap} & \textbf{Max gap} & "
         r"\textbf{\# best} & \textbf{\# proved} & \textbf{Own gap} \\"], rows,
        notes="Gaps to $\\mathrm{LB}^*$ in \\%. \\# best: instances on which the method has the lowest peak of the four. "
              "\\# proved: optimality proved by the method itself (Gurobi: status optimal; CRI: peak equal to its own "
              "certificate). Own gap: mean of Gurobi's reported MIP gap, or of CRI's certified gap, at the end of the run."))


def t_sota(C, out):
    v10, r10, s10, f10 = C.stats("E10a_state_of_the_art", 10)
    v60, r60, s60, f60 = C.stats("E10b_state_of_the_art_60s", 60)
    pi = C.csv("25_E10a_state_of_the_art/budget_10s/tables/per_instance_mean_peak.csv").set_index("instance")
    methods = ["CRI", "GUROBI-std", "GUROBI-agg", "AN_HYBRID", "AN-PAR", "AN-dev", "ZAIDI_ILS", "ZAIDI_ILS-b", "ZAIDI_H",
               "BK", "NFDH", "HUNSAKER", "AFSHAR_BB"]
    rows = []
    for m in methods:
        cells = [NAME[m], fg(s10.loc[m, "mean gap_lb"]), fg(s10.loc[m, "median"]),
                 "n/a" if m == "CRI" else wtl(v10.loc[m]), f"{r10[m]:.2f}", ft(s10.loc[m, "mean time (s)"])]
        if m in s60.index:
            cells += [fg(s60.loc[m, "mean gap_lb"]), "n/a" if m == "CRI" else wtl(v60.loc[m]), f"{r60[m]:.2f}"]
        else:
            cells += ["n/a", "n/a", "n/a"]
        if m == "CRI":
            cells = [cells[0]] + [bold(c) if k in (0, 1, 3, 5) else c for k, c in enumerate(cells[1:])]
        rows.append(" & ".join(cells))
        if m == "GUROBI-agg":
            rows.append(r"\addlinespace")
    for name, (f, k) in (("E10a", (f10, 10)), ("E10b", (f60, 60))):
        C.note(f"{name} Friedman", f"N {int(f['instances (N)'])}, k {int(f['methods (k)'])}, chi2 {f['chi2']:.1f}, p {f['p']:.3g}")
    rel = pi.div(pi.min(axis=1), axis=0)
    for m in methods:
        C.note(f"E10a profile {m}", f"share within 1% of best {100 * (rel[m] <= 1.01).mean():.1f}%, within 5% "
                                    f"{100 * (rel[m] <= 1.05).mean():.1f}%, max ratio {rel[m].max():.3f}")
    for m in v10.index:
        r = v10.loc[m]
        C.note(f"E10a CRI vs {m}", f"{r.mean_control:.3f} vs {r.mean_method:.3f}, W/T/L {int(r.wins)}/{int(r.ties)}/"
                                   f"{int(r.losses)}, p_holm {r.p_holm:.3g}")
    for m in v60.index:
        r = v60.loc[m]
        C.note(f"E10b CRI vs {m}", f"{r.mean_control:.3f} vs {r.mean_method:.3f}, W/T/L {int(r.wins)}/{int(r.ties)}/"
                                   f"{int(r.losses)}, p_holm {r.p_holm:.3g}")
    open(os.path.join(out, "sota.tex"), "w", encoding="utf-8").write(table_env(
        "tab:sota", "Comparison with methods from the literature (blocks E10a and E10b)", "l|ccccc|ccc",
        [r" & \multicolumn{5}{c|}{\textbf{10 s (112 instances)}} & \multicolumn{3}{c}{\textbf{60 s (72 instances)}} \\",
         r"\textbf{Method} & \textbf{Mean gap} & \textbf{Median gap} & \textbf{W/T/L} & \textbf{Rank} & \textbf{Time (s)} & "
         r"\textbf{Mean gap} & \textbf{W/T/L} & \textbf{Rank} \\"], rows,
        notes="Gaps to $\\mathrm{LB}^*$ in \\%. W/T/L, $^\\dagger$ and Rank as in Table~\\ref{tab:equal-budgets}. Time: "
              "mean charged time; the constructive methods and Hunsaker's online policy stop long before the budget, "
              "Afshar-BB when its search ends. CRI and Gurobi are the runs of E4a and E4b. Afshar-BB is exact when its "
              "search completes within the budget; it does so only on the smallest instances."))


def t_ablation(C, out):
    v10, r10, s10, _ = C.stats("E5_ablation", 10)
    v60, r60, s60, _ = C.stats("E5b_ablation_60s", 60)
    groups = [
        ("Improve stage", [("CRI-R", "Round only (no Improve stage)"), ("CRI-R+LLS", "LLS only"),
                           ("CRI-R+WLNS", "W-LNS only"), ("CRI-R+SA", "SA only"), ("CRI", "LLS and W-LNS (default)")]),
        ("Probability matrices", [("CRI-lp", "$\\alpha^*$ only"), ("CRI-cut", "$\\alpha^{\\mathrm{cut}}$ only"),
                                  ("CRI-noconf", "no configuration LP")]),
        ("Roundings", [("CRI-CE", "CE only"), ("CRI-MT", "MT only"), ("CRI-IR", "IR only"), ("CRI-G", "G only")]),
        ("Start of SA (hybrids)", [("SA-b", "random (SA)"), ("HSG-b", "G (HSG)"), ("HSR-b", "RR (HSR)"),
                                   ("HSL-b", "RLP (HSL)"), ("HSC-b", "CRI's Round stage (HSC)")]),
    ]
    rows = []
    for gname, items in groups:
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{gname}}}}} \\\\")
        for m, lab in items:
            cells = [lab, fg(s10.loc[m, "mean gap_lb"]), "n/a" if m == "CRI" else wtl(v10.loc[m])]
            if m in s60.index:
                cells += [fg(s60.loc[m, "mean gap_lb"]), "n/a" if m == "CRI" else wtl(v60.loc[m])]
            else:
                cells += ["", ""]
            rows.append(" & ".join(cells))
            C.note(f"ablation {m}", f"10 s {s10.loc[m, 'mean gap_lb']:.3f}" +
                   (f", 60 s {s60.loc[m, 'mean gap_lb']:.3f}" if m in s60.index else ""))
        rows.append(r"\addlinespace")
    rows = rows[:-1]
    open(os.path.join(out, "ablation.tex"), "w", encoding="utf-8").write(table_env(
        "tab:ablation", "Ablation of CRI (blocks E5 and E5b, 45 medium instances)", "lcccc",
        [r" & \multicolumn{2}{c}{\textbf{10 s (3 seeds)}} & \multicolumn{2}{c}{\textbf{60 s (1 seed)}} \\",
         r"\textbf{Configuration} & \textbf{Gap (\%)} & \textbf{W/T/L} & \textbf{Gap (\%)} & \textbf{W/T/L} \\"], rows,
        notes="Each row changes one stage of the default configuration. W/T/L: instances on which the default CRI is "
              "better than, equal to, worse than the configuration; $^\\dagger$: not significant (Wilcoxon, Holm, "
              "5\\%). The SA rows are SA in its anytime version, started from the schedule given in the row."))


def t_scalability(C, out):
    g = C.csv("22_E6_scalability/scalability/tables/median_gap_by_n.csv").set_index("n")
    t = C.csv("22_E6_scalability/scalability/tables/median_time_by_n.csv").set_index("n")
    cc = C.csv("22_E6_scalability/scalability/tables/certify_costs_by_n.csv").set_index("n")
    x = C.e[C.e.experiment == "E6_scalability"]
    nv = x.drop_duplicates("instance").groupby("n").n_vars.mean()
    ni = x.drop_duplicates("instance").groupby("n").size()
    rows = []
    for n in g.index:
        cells = [f"{n:,}".replace(",", "{,}"), str(int(ni[n])), f"{nv[n]:,.0f}".replace(",", "{,}"),
                 fr(cc.loc[n, "cost_lp"], 1), fr(cc.loc[n, "cost_lb_cut"], 1),
                 fg(g.loc[n, "CRI"]), fg(g.loc[n, "LLS"]), f"{fg(g.loc[n, 'CE-cut-excess'])} ({fr(t.loc[n, 'CE-cut-excess'], 1)})",
                 fg(g.loc[n, "G"]), fg(g.loc[n, "BK"]), fg(g.loc[n, "GUROBI-std"]), fg(g.loc[n, "GUROBI-agg"])]
        rows.append(" & ".join(cells))
    su = C.csv("22_E6_scalability/budget_120s/tables/summary.csv").set_index("method")
    for m in su.index:
        C.note(f"E6 {m}", f"mean gap {su.loc[m, 'mean gap_lb']:.3f}, median {su.loc[m, 'median']:.3f}, time {su.loc[m, 'mean time (s)']:.2f}")
    open(os.path.join(out, "scalability.tex"), "w", encoding="utf-8").write(table_env(
        "tab:scalability", "Scalability at a budget of 120 s (block E6): Certify costs and median gaps to $\\mathrm{LB}^*$",
        "rrr|rr|rrrrrrr",
        [r"& & & \multicolumn{2}{c|}{\textbf{Certify (s)}} & \multicolumn{7}{c}{\textbf{Median gap (\%)}} \\",
         r"$n$ & \textbf{Inst.} & \textbf{Binaries} & \textbf{LP} & $\mathrm{LB}_{\mathrm{cut}}$ & \textbf{CRI} & "
         r"\textbf{LLS} & \textbf{CE$_E(\alpha^{\mathrm{cut}})$ (s)} & \textbf{G} & \textbf{BK} & \textbf{Gurobi-std} & "
         r"\textbf{Gurobi-agg} \\"], rows,
        notes="One instance per family and $n$ ($n=10{,}000$: omega and staggered, $T=96$ and 288). Binaries: mean number of binary "
              "variables of model~\\eqref{eq:obj} to~\\eqref{eq:nonneg}. Certify: median time of the relaxation and of the cover-cut bound. "
              "The time of CE, a single-shot rounding, is given in parentheses; G and BK take less than 0.1 s. CRI, LLS "
              "and Gurobi use the whole budget."))


def t_managerial(C, out):
    x = C.e[C.e.experiment == "E9_managerial"]
    pi = x.pivot_table(index=["instance", "family"], columns="method", values="objective").reset_index()
    rows = []
    for fam in ["omega", "staggered", "overnight"]:
        g = pi[pi.family == fam]
        ra = 100 * (g.ASAP - g.CRI) / g.ASAP
        rg = 100 * (g.GNP - g.CRI) / g.GNP
        rows.append(f"{fam} & {len(g)} & {g.ASAP.mean():.1f} & {g.GNP.mean():.1f} & {g.CRI.mean():.1f} & {ra.mean():.1f} "
                    f"& {rg.mean():.1f}")
        C.note(f"E9 {fam}", f"reduction vs ASAP {ra.mean():.1f}% (min {ra.min():.1f}, max {ra.max():.1f}); vs GNP {rg.mean():.1f}%")
    ra = 100 * (pi.ASAP - pi.CRI) / pi.ASAP
    rg = 100 * (pi.GNP - pi.CRI) / pi.GNP
    rows.append(r"\midrule")
    rows.append(f"all & {len(pi)} & {pi.ASAP.mean():.1f} & {pi.GNP.mean():.1f} & {pi.CRI.mean():.1f} & {ra.mean():.1f} & "
                f"{rg.mean():.1f}")
    C.note("E9 all", f"vs ASAP {ra.mean():.1f}% (median {ra.median():.1f}), vs GNP {rg.mean():.1f}% (median {rg.median():.1f})")
    open(os.path.join(out, "managerial.tex"), "w", encoding="utf-8").write(table_env(
        "tab:managerial", "Peak power of CRI against reference charging policies (block E9, 10 s)", "lrrrrrr",
        [r" & & \multicolumn{3}{c}{\textbf{Mean peak (kW)}} & \multicolumn{2}{c}{\textbf{Reduction by CRI (\%)}} \\",
         r"\textbf{Family} & \textbf{Inst.} & \textbf{ASAP} & \textbf{GNP} & \textbf{CRI} & \textbf{vs ASAP} & "
         r"\textbf{vs GNP} \\"], rows,
        notes="ASAP: uncontrolled charging, each vehicle in the first $p_i$ slots of its window. GNP: non-preemptive "
              "charging; the vehicles, by non-increasing power, are each placed on the block of $p_i$ consecutive slots "
              "of their window with the lowest maximum load. Reductions: mean over instances of the relative peak "
              "reduction."))


# ---------------------------------------------------------------------------------------------- figures
def anytime_curves(C, block, budget, methods):
    x = C.e[(C.e.experiment == block) & (C.e.budget == budget) & C.e.method.isin(methods)]
    lb = C.ref.LB_star
    grid = np.logspace(-3, np.log10(budget), 90)
    out = {}
    for m in methods:
        g = x[x.method == m]
        if g.empty:
            continue
        M = np.full((len(g), len(grid)), np.inf)
        for a, k in enumerate(zip(g.experiment, g.instance, g.method, g.budget, g.seed)):
            tr = sorted(C.traces.get(k, []))
            ref = lb.get(k[1], np.nan)
            j, best = 0, np.inf
            for b, t in enumerate(grid):
                while j < len(tr) and tr[j][0] <= t:
                    best = min(best, tr[j][1]); j += 1
                if np.isfinite(best):
                    M[a, b] = 100 * (best - ref) / ref
        # a run without a schedule yet counts as infinitely bad; a finite sentinel instead of inf, since the
        # interpolation of np.quantile gives NaN next to an infinite value even with weight 0
        med = np.quantile(np.where(np.isinf(M), 1e12, M), 0.5, axis=0)
        out[m] = (grid, np.where(med < 1e11, med, np.nan))
    return out


def fig_anytime(C, out):
    FLOOR = 1e-3
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35), gridspec_kw=dict(wspace=0.4))
    panels = [("E4b_equal_budgets_60s", 60, FIG_ORDER, "(a) 60 s, medium and large"),
              ("E3x_long_runs_xlarge", 600, ["CRI", "GUROBI-std", "GUROBI-agg", "CRI+C"], "(b) 600 s, xlarge")]
    for ax, (bl, b, ms, title) in zip(axes[:2], panels):
        cur = anytime_curves(C, bl, b, ms)
        for m in FIG_ORDER:
            if m not in cur:
                continue
            t, y = cur[m]
            ok = np.isfinite(y)
            if not ok.any():
                continue
            # CRI on top: CRI+C coincides with it until its Complete stage starts
            ax.plot(t[ok], np.maximum(y[ok], FLOOR), lw=1.1, markevery=max(1, ok.sum() // 6), ms=3.2,
                    label=plain(m), zorder=5 if m == "CRI" else 3, **style(FIG_ORDER, m))
        ax.set_xscale("log"); ax.set_yscale("log")
        first = [tr[0][0] for k, tr in C.traces.items() if k[0] == bl and k[3] == b and tr]
        ax.set_xlim(max(1e-3, 10 ** np.floor(np.log10(max(min(first), 1e-3)))), b); ax.set_ylim(FLOOR * 0.8, 200)
        ax.set_xlabel("time (s)"); ax.set_title(title, fontsize=7.5, loc="left", color=INK)
    axes[0].set_ylabel(r"median gap to LB$^*$ (%)")
    lad = C.csv("30_budget_ladders/equal_budgets_1s_to_600s/tables/mean_gap_by_budget.csv").set_index("method")
    B = [1, 10, 60, 600]
    ax = axes[2]
    for m in FIG_ORDER:
        if m not in lad.index:
            continue
        y = np.array([lad.loc[m].get(f"{b} s", np.nan) for b in B], float)
        ok = np.isfinite(y)
        ax.plot(np.array(B)[ok], np.maximum(y[ok], FLOOR), lw=1.1, ms=3.6, zorder=5 if m == "CRI" else 3,
                **style(FIG_ORDER, m))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(B); ax.set_xticklabels([str(b) for b in B])
    ax.set_ylim(FLOOR * 0.8, 200)
    ax.set_xlabel("budget (s)"); ax.set_ylabel(r"mean gap to LB$^*$ (%)")
    ax.set_title("(c) gap against the budget", fontsize=7.5, loc="left", color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.13), handlelength=2.8, columnspacing=1.4)
    fig.savefig(os.path.join(out, "anytime.pdf"))
    plt.close(fig)


def fig_profile(C, out):
    pi = C.csv("25_E10a_state_of_the_art/budget_10s/tables/per_instance_mean_peak.csv").set_index("instance")
    rel = pi.div(pi.min(axis=1), axis=0)
    # performance profile on the excess over the best peak, log scale (the methods differ below 5%)
    eps = np.logspace(-2, np.log10(30), 400)                 # excess in %
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    for m in SOTA_FIG:
        r = np.sort(100 * (rel[m].values - 1))
        rho = np.searchsorted(r, eps + 1e-9, side="right") / len(r)
        ax.plot(eps, rho, lw=1.1, markevery=57, ms=3.2, label=plain(m), zorder=5 if m == "CRI" else 3,
                **style(SOTA_FIG, m))
    ax.set_xscale("log"); ax.set_xlim(eps[0], eps[-1]); ax.set_ylim(0, 1.02)
    ax.set_xticks([0.01, 0.1, 1, 10]); ax.set_xticklabels(["0.01", "0.1", "1", "10"])
    ax.set_xlabel("excess over the best peak on the instance (%)")
    ax.set_ylabel("share of instances within the excess")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=2.8)
    fig.savefig(os.path.join(out, "profile_sota.pdf"))
    plt.close(fig)
    for m in SOTA_FIG:
        C.note(f"profile {m}", f"best on {100 * (rel[m] <= 1 + 1e-9).mean():.1f}% of instances, within 1% "
                               f"{100 * (rel[m] <= 1.01).mean():.1f}%")
    for m in ("NFDH", "HUNSAKER", "AFSHAR_BB"):
        C.note(f"profile at tau 1.3, {m}", f"{100 * (rel[m] <= 1.3).mean():.1f}%")


def fig_certificates(C, out, c, ref):
    fams = ["omega", "staggered", "overnight"]
    m = c.family.isin(fams)
    d = c[m].assign(ref=ref[m])
    pub = d[["info_published_hoeffding_union", "info_published_bernstein_union", "info_published_hoeffding_lll",
             "info_published_bernstein_lll"]].min(axis=1)
    series = [(r"$U$", d.info_U_best / d.ref), ("non-localized bounds", pub / d.ref),
              (r"$\mathbf{P}^{\mathrm{LP}}+w_{\max}$", d.info_LP_plus_wmax / d.ref), (r"$\mathrm{LB}$", d.info_LB / d.ref)]
    order = [s[0] for s in series]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), gridspec_kw=dict(wspace=0.3))
    ax = axes[0]
    for lab, v in series:
        y = v.groupby(d.n).mean()
        ax.plot(y.index, y.values, lw=1.1, ms=3.4, label=lab, **style(order, lab))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_yticks([1, 1.5, 2, 3, 4]); ax.set_yticklabels(["1", "1.5", "2", "3", "4"])
    ax.set_xlabel("$n$"); ax.set_ylabel("mean ratio to optimum or best known")
    ax.set_title("(a) certificates against the reference", fontsize=7.5, loc="left", color=INK)
    ax.legend(loc="upper right", handlelength=2.6)
    ax = axes[1]
    gap = 100 * (d.info_U_best - d.info_LB) / d.info_LB
    fam_order = ["omega", "staggered", "overnight"]
    for k, f in enumerate(fam_order):
        y = gap[d.family == f].groupby(d.n[d.family == f]).mean()
        ax.plot(y.index, y.values, lw=1.1, ms=3.4, label=f, color=PALETTE[4 + k], linestyle=DASHES[4 + k],
                marker=MARKERS[4 + k])
    ax.set_xscale("log"); ax.set_ylim(bottom=0)
    ax.set_xlabel("$n$"); ax.set_ylabel(r"mean $(U-\mathrm{LB})/\mathrm{LB}$ (%)")
    ax.set_title("(b) width of the certified interval", fontsize=7.5, loc="left", color=INK)
    ax.legend(loc="upper right", handlelength=2.6)
    fig.savefig(os.path.join(out, "certificates.pdf"))
    plt.close(fig)


def fig_profiles(C, out):
    picks = ["medium_omega_n200_T96_000", "medium_overnight_n200_T96_000"]
    order = ["CRI", "ASAP", "GNP"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2), gridspec_kw=dict(wspace=0.22))
    labels = {"CRI": "CRI", "ASAP": "ASAP (uncontrolled)", "GNP": "GNP (non-preemptive)"}
    for ax, inst in zip(axes, picks):
        d = C.instances[inst]
        w = np.asarray(d["w"], float)
        T = int(d["T"])
        peaks = {}
        for m in order:
            x = C.solutions.get(("E9_managerial", inst, m, 10, 0))
            if x is None:
                x = C.solutions.get(("E9_managerial", inst, m, 10.0, 0))
            L = (np.asarray(x, float) * w[:, None]).sum(axis=0)
            k = order.index(m)
            ax.step(np.arange(1, T + 1), L, where="mid", lw=1.0 if m == "CRI" else 0.9, color=PALETTE[k],
                    linestyle=DASHES[k], label=labels[m], zorder=5 if m == "CRI" else 3)
            peaks[m] = L.max()
            C.note(f"profile {inst} {m}", f"peak {L.max():.1f} kW")
        ax.set_xlim(1, T); ax.set_ylim(0, 1.15 * max(peaks.values()))
        omega = "omega" in inst
        ax.set_xlabel("slot $t$ (15 min)"); ax.set_title("(a) omega, $n=200$" if omega else "(b) overnight, $n=200$",
                                                         fontsize=7.5, loc="left", color=INK)
        ax.text(0.02 if omega else 0.98, 0.97, "peaks (kW): " + ", ".join(f"{m} {peaks[m]:.0f}" for m in order),
                transform=ax.transAxes, ha="left" if omega else "right", va="top", fontsize=6.5, color=INK2)
    axes[0].set_ylabel("load $L_t$ (kW)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08), handlelength=2.8, columnspacing=1.6)
    fig.savefig(os.path.join(out, "load_profiles.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("paper")
    ap.add_argument("--numbers", default=None, help="markdown file for the numbers quoted in the text")
    a = ap.parse_args(argv)
    C = Campaign(a.results)
    tdir, fdir = os.path.join(a.paper, "tables"), os.path.join(a.paper, "figures")
    os.makedirs(tdir, exist_ok=True); os.makedirs(fdir, exist_ok=True)
    t_instances(C, tdir)
    t_blocks(C, tdir)
    c, ref = t_certificates(C, tdir)
    t_h2h(C, tdir)
    t_equal(C, tdir)
    t_gurobi(C, tdir)
    t_long(C, tdir)
    t_sota(C, tdir)
    t_ablation(C, tdir)
    t_scalability(C, tdir)
    t_managerial(C, tdir)
    fig_certificates(C, fdir, c, ref)
    fig_anytime(C, fdir)
    fig_profile(C, fdir)
    fig_profiles(C, fdir)
    if a.numbers:
        with open(a.numbers, "w", encoding="utf-8") as f:
            f.write("# Numbers of Section 6\n\n" + "\n".join(f"- {k}: {v}" for k, v in C.numbers) + "\n")
    print(f"tables in {tdir}, figures in {fdir}" + (f", numbers in {a.numbers}" if a.numbers else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
