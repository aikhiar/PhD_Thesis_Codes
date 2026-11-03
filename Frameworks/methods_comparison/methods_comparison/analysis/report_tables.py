"""
report_tables.py  —  the tables of the report.

Per comparison block (objective or hypervolume, gap, execution time):

  * an INSTANCE table: one row per instance (grouped by size class when the
    study declares classes), one column per method, cells "mean (std)" over
    the runs; the best mean of each row in bold. Where the chapter reports them
    (config statistics.mann_whitney), every method other than the control
    carries a marker from a per-instance two-sided Mann-Whitney U test against
    the control: "+" significantly better, "-" significantly worse, "=" no
    significant difference. Closing rows: mean over instances, average
    Friedman rank, number of instances where the method is best, and the
    "+/=/-" counts.
  * a SUMMARY table (config report.summary, Appendix D): one row per method
    with the mean over instances and its 95% bootstrap confidence interval,
    robustness figures (median, worst instance, 90% quantile, mean coefficient
    of variation over runs), average rank, #best, and the tests against the
    control that the study runs (+/=/- counts, Friedman post-hoc and Wilcoxon
    Holm p, A12, Bayesian probabilities).

Each table is written as CSV and as a booktabs LaTeX table (.tex, needs
\\usepackage{booktabs}).
"""

import os
import re
import numpy as np

from ..utils.io import write_csv, slug
from ..utils.stats import marker_vs_control, bootstrap_mean_ci


# ── formatting helpers ───────────────────────────────────────────────────────
def fmt(v, d=2):
    """Number for a table cell: fixed decimals, scientific for very large or
    very small magnitudes; 'n/a' when missing."""
    if v is None or not np.isfinite(v):
        return "n/a"
    a = abs(v)
    if a != 0 and (a >= 1e6 or a < 10 ** (-d)):
        return f"{v:.{max(d, 1)}e}"
    return f"{v:.{d}f}"


def tex_escape(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    # in the default OT1 encoding these print as other glyphs in text mode
    # ("|" as an em dash, "<" / ">" as inverted punctuation)
    for a, b in (("|", r"$|$"), ("<", r"$<$"), (">", r"$>$")):
        s = s.replace(a, b)
    return s


_SCI = re.compile(r"(-?\d+(?:\.\d+)?)e([+-]\d+)")


def _tex_num(txt):
    """'1.23e+05' -> '$1.23\\times10^{5}$' inside LaTeX cells."""
    return _SCI.sub(lambda m: f"${m.group(1)}\\times10^{{{int(m.group(2))}}}$", txt)


def write_latex_table(header, rows, path, caption="", label="", note="",
                      align=None, bold=None, groups=None):
    """Generic booktabs table. `rows`: list of lists of cell strings (already
    LaTeX-safe). `bold`: optional set of (row, col) to embolden. `groups`:
    optional {row_index: group title} inserted as a full-width header row
    before that row. Returns the LaTeX source."""
    ncol = len(header)
    align = align or ("l" + "r" * (ncol - 1))
    L = [r"\begin{table}[htbp]", r"\centering"]
    if caption:
        L.append(rf"\caption{{{caption}}}")
    if label:
        L.append(rf"\label{{{label}}}")
    L += [r"\small", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
          " & ".join(header) + r" \\", r"\midrule"]
    for ri, row in enumerate(rows):
        if groups and ri in groups:
            if ri > 0:
                L.append(r"\midrule")
            L.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{{groups[ri]}}}}} \\")
        cells = []
        for ci, c in enumerate(row):
            c = _tex_num(c) if isinstance(c, str) else c
            cells.append(rf"\textbf{{{c}}}" if bold and (ri, ci) in bold else str(c))
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    if note:
        L += [r"\par\smallskip", rf"{{\footnotesize {note}}}"]
    L.append(r"\end{table}")
    src = "\n".join(L) + "\n"
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
    return src


# ── per-block instance table ─────────────────────────────────────────────────
def _clean(v):
    return [x for x in v if x is not None and np.isfinite(x)]


def _ordered_ids(problem, ids):
    """Instance order + {row_index: class title} when size classes exist."""
    classes = getattr(problem, "size_classes", {}) or {}
    if not classes:
        return list(ids), None
    order, groups, seen = [], {}, set()
    for cls, members in classes.items():
        mem = [i for i in members if i in ids and i not in seen]
        if mem:
            groups[len(order)] = str(cls)
            order += mem; seen.update(mem)
    rest = [i for i in ids if i not in seen]
    if rest:
        groups[len(order)] = "Other instances"
        order += rest
    return order, groups


def block_tables(scores, problem, cfg, out_dir, block_name, label, hib,
                 control, stats_res=None, markers=True):
    """Write the instance table of one block (CSV and .tex). `markers`: False
    leaves the Mann-Whitney markers out whatever the configuration (gap table).
    Returns {"markers": {method: (n_plus, n_eq, n_minus)}, "n_best": {...},
    "avg_rank": {...}, "inst_means": {...}} for the summary table."""
    rcfg = cfg.get("report", {})
    d = int(rcfg.get("decimals", 2))
    alpha = float(rcfg.get("marker_alpha", 0.05))
    names = list(scores.keys())
    lab = {m: problem.label(m) for m in names}
    ids = sorted({i for m in scores for i in scores[m]})
    order, groups = _ordered_ids(problem, ids)
    has_ctl = (markers and cfg.get("statistics", {}).get("mann_whitney", False)
               and control in names and len(names) >= 2)

    rows_csv, rows_tex, bold = [], [], set()
    counts = {m: [0, 0, 0] for m in names}
    n_best = {m: 0 for m in names}
    inst_means = {m: [] for m in names}
    for ri, i in enumerate(order):
        cells = {}
        means = {}
        for m in names:
            c = _clean(scores[m].get(i, []))
            means[m] = float(np.mean(c)) if c else np.nan
            sd = float(np.std(c, ddof=1)) if len(c) >= 2 else 0.0      # sample std (n - 1)
            cells[m] = (f"{fmt(means[m], d)} ({fmt(sd, d)})" if c else "n/a")
            inst_means[m].append(means[m])
        fin = [v for v in means.values() if np.isfinite(v)]
        best_txt = None
        if fin:
            best = max(fin) if hib else min(fin)
            best_txt = fmt(best, d)
        marks = {}
        for m in names:
            if has_ctl and m != control:
                mk = marker_vs_control(scores[m].get(i, []), scores[control].get(i, []),
                                       alpha, hib)
                marks[m] = mk
                counts[m]["+=-".index(mk)] += 1
        row_c, row_t = [i], [tex_escape(i)]
        for ci, m in enumerate(names, start=1):
            is_best = best_txt is not None and fmt(means[m], d) == best_txt
            if is_best:
                n_best[m] += 1
                bold.add((ri, ci))
            mk = marks.get(m, "")
            row_c.append(cells[m] + (f" {mk}" if mk else ""))
            tex_mk = {"+": r"$^{+}$", "-": r"$^{-}$", "=": r"$^{\approx}$"}.get(mk, "")
            row_t.append(cells[m] + tex_mk)
        rows_csv.append(row_c); rows_tex.append(row_t)

    # closing rows
    ranks = (stats_res or {}).get("friedman", {}).get("average_ranks", {}) or {}
    avg_rank = {m: ranks.get(lab[m], np.nan) for m in names}
    if not ranks and len(names) >= 2:
        # Friedman not applicable (e.g. 2 methods): average per-instance rank
        # over the instances where every method has a value
        from scipy.stats import rankdata
        M = np.array([inst_means[m] for m in names], dtype=float)
        full = np.all(np.isfinite(M), axis=0)
        if full.any():
            Rk = np.vstack([rankdata(-M[:, j] if hib else M[:, j])
                            for j in np.where(full)[0]])
            avg_rank = {m: float(Rk[:, k].mean()) for k, m in enumerate(names)}
    mean_row = ["Mean"] + [fmt(float(np.nanmean(inst_means[m])) if np.any(np.isfinite(inst_means[m]))
                               else np.nan, d) for m in names]
    rank_row = ["Avg. rank"] + [fmt(avg_rank[m], 2) for m in names]
    best_row = ["#Best"] + [str(n_best[m]) for m in names]
    summary = [mean_row, rank_row, best_row]
    if has_ctl:
        summary.append(["+/=/-"] + ["control" if m == control else
                                    "{}/{}/{}".format(*counts[m]) for m in names])
    n_body = len(rows_csv)
    header = ["Instance"] + [lab[m] + (" (control)" if has_ctl and m == control else "")
                             for m in names]
    for r in summary:
        rows_csv.append(r)
        rows_tex.append([tex_escape(c) for c in r])

    stem = f"{problem.code}_{block_name} table" if problem.code else f"{block_name} table"
    if out_dir:
        write_csv(rows_csv, os.path.join(out_dir, stem + ".csv"), header=header)
        note = ("Mean (sample standard deviation) over the runs; best mean of each instance in bold."
                + (rf" $^{{+}}$, $^{{-}}$, $^{{\approx}}$: significantly better, worse, or not "
                   rf"significantly different from the control ({tex_escape(lab[control])}) on "
                   rf"the instance (two-sided Mann-Whitney U test, $\alpha = {alpha:g}$)."
                   if has_ctl else ""))
        write_latex_table([tex_escape(h) for h in header], rows_tex,
                          os.path.join(out_dir, stem + ".tex"),
                          caption=tex_escape(f"{label}: {problem.name}"),
                          label=f"tab:{slug(problem.code or '')}_{slug(block_name)}",
                          note=note, bold=bold,
                          groups={**dict(groups or {}), n_body: "Summary"})
    return {"markers": {m: tuple(counts[m]) for m in names} if has_ctl else {},
            "n_best": n_best, "avg_rank": avg_rank, "inst_means": inst_means}


# ── per-block summary (one row per method) ───────────────────────────────────
def block_summary(scores, problem, cfg, out_dir, block_name, label, hib, control,
                  table_info, stats_res=None):
    """Summary table of one block (CSV and .tex), with the columns of the tests
    the study runs; returns (header, rows)."""
    rcfg = cfg.get("report", {})
    st = cfg.get("statistics", {})
    d = int(rcfg.get("decimals", 2))
    B = int(rcfg.get("bootstrap_samples", 2000))
    names = list(scores.keys())
    lab = {m: problem.label(m) for m in names}
    ids = sorted({i for m in scores for i in scores[m]})
    vs = (stats_res or {}).get("vs_control", {}) or {}
    fh = {r["method"]: r for r in vs.get("friedman_holm", {}).get("comparisons", [])}
    wh = {r["method"]: r for r in vs.get("wilcoxon_holm", {}).get("comparisons", [])}
    by = {r["method"]: r for r in vs.get("bayesian", {}).get("comparisons", [])}

    header = ["Method", "Mean", "95% CI", "Median", "Worst", "Q90" if not hib else "Q10",
              "Mean CV", "Avg. rank", "#Best", "+/=/-", "p Holm (Friedman)",
              "p Holm (Wilcoxon)", "A12 (control)", "P(control better)",
              "P(equivalent)", "P(method better)"]
    # the columns of the tests the study does not run are left out
    with_mw = bool(table_info.get("markers"))
    cols = list(range(9)) + ([9] if with_mw else []) \
        + ([10, 11, 12] if st.get("holm") else []) + ([13, 14, 15] if st.get("bayesian") else [])
    rows = []
    for m in names:
        im = np.asarray(table_info["inst_means"][m], dtype=float)
        mean, lo, hi = bootstrap_mean_ci(im, B, seed=0)
        fin = im[np.isfinite(im)]
        worst = (np.min(fin) if hib else np.max(fin)) if len(fin) else np.nan
        q = (np.quantile(fin, 0.10) if hib else np.quantile(fin, 0.90)) if len(fin) else np.nan
        cvs = []
        for i in ids:
            c = _clean(scores[m].get(i, []))
            if len(c) >= 2 and abs(np.mean(c)) > 1e-12:
                cvs.append(np.std(c, ddof=1) / abs(np.mean(c)))
        L = lab[m]
        is_c = m == control
        mk = table_info["markers"].get(m, (0, 0, 0))
        rows.append([
            L + (" (control)" if is_c else ""), fmt(mean, d),
            f"[{fmt(lo, d)}, {fmt(hi, d)}]", fmt(float(np.median(fin)) if len(fin) else np.nan, d),
            fmt(worst, d), fmt(q, d), fmt(float(np.mean(cvs)) if cvs else np.nan, 3),
            fmt(table_info["avg_rank"].get(m, np.nan), 2), str(table_info["n_best"].get(m, 0)),
            "control" if is_c else ("{}/{}/{}".format(*mk) if control in names else "n/a"),
            "" if is_c else fmt(fh.get(L, {}).get("p_holm", np.nan), 4),
            "" if is_c else fmt(wh.get(L, {}).get("p_holm", np.nan), 4),
            "" if is_c else fmt(wh.get(L, {}).get("a12_control", np.nan), 3),
            "" if is_c else fmt(by.get(L, {}).get("p_control_better", np.nan), 3),
            "" if is_c else fmt(by.get(L, {}).get("p_equivalent", np.nan), 3),
            "" if is_c else fmt(by.get(L, {}).get("p_method_better", np.nan), 3),
        ])
    stem = f"{problem.code}_{block_name} summary" if problem.code else f"{block_name} summary"
    header = [header[k] for k in cols]
    rows = [[r[k] for k in cols] for r in rows]
    if out_dir:
        write_csv(rows, os.path.join(out_dir, stem + ".csv"), header=header)
        # compact paper version
        keep = [k for k, h in enumerate(header)
                if h in ("Method", "Mean", "95% CI", "Worst", "Avg. rank", "#Best", "+/=/-",
                         "p Holm (Friedman)", "A12 (control)", "P(control better)")]
        note = ("Mean over instances of the per-instance mean, with its 95\\% percentile "
                "bootstrap interval; Worst: worst per-instance mean; Avg. rank: Friedman "
                "average rank"
                + ("; +/=/-: instances where the method is significantly better / "
                   "not different / worse than the control" if with_mw else "")
                + ("; p Holm: Friedman post-hoc against the control with Holm adjustment; "
                   "A12: probability that the control beats the method on an instance"
                   if st.get("holm") else "")
                + ("; P(control better): Bayesian signed-rank test" if st.get("bayesian") else "")
                + ".")
        write_latex_table([tex_escape(header[k]) for k in keep],
                          [[tex_escape(r[k]) for k in keep] for r in rows],
                          os.path.join(out_dir, stem + ".tex"),
                          caption=tex_escape(f"Summary, {label}: {problem.name}"),
                          label=f"tab:{slug(problem.code or '')}_{slug(block_name)}_summary",
                          note=note)
    return header, rows
