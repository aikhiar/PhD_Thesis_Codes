"""Table writers (Markdown, LaTeX booktabs, CSV) without extra dependencies, and the standard
summary tables of a comparison."""

import os

import numpy as np
import pandas as pd

from . import stats as S


def _fmt(v, digits=2):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}"
    if isinstance(v, (float, np.floating)):
        if v != v:
            return "n/a"
        if np.isinf(v):
            return "∞" if v > 0 else "−∞"
        if v != 0 and (abs(v) < 10 ** -digits or abs(v) >= 1e6):
            return f"{v:.{digits}e}"
        return f"{v:.{digits}f}"
    return str(v)


def to_markdown(df, digits=2, index=True, bold_min=None):
    d = (df.reset_index() if index else df).astype(object)      # object: iterrows keeps ints as ints
    cols = [str(c) for c in d.columns]
    best = {}
    for c in bold_min or []:
        if c in d:
            best[c] = pd.to_numeric(d[c], errors="coerce").min()
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        cells = []
        for c in d.columns:
            s = _fmt(r[c], digits)
            if c in best and isinstance(r[c], (float, int, np.floating, np.integer)) and r[c] == best[c]:
                s = f"**{s}**"
            cells.append(s.replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _tex(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("∞", r"$\infty$"), ("–", "-"), ("−", "-"),
                 ("≤", r"$\le$"), ("≥", r"$\ge$")):
        s = s.replace(a, b)
    return s


def to_latex(df, caption="", label="", digits=2, index=True, bold_min=None):
    d0 = df.reset_index() if index else df
    d = d0.astype(object)
    best = {c: pd.to_numeric(d[c], errors="coerce").min() for c in (bold_min or []) if c in d}
    align = "l" * (1 if index else 0) + "".join("r" if pd.api.types.is_numeric_dtype(d0[c]) else "l"
                                                for c in d0.columns[(1 if index else 0):])
    out = [r"\begin{table}[htbp]", r"\centering", r"\small"]
    if caption:
        out.append(r"\caption{" + _tex(caption) + "}")
    if label:
        out.append(r"\label{" + label + "}")
    out += [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(_tex(c) for c in d.columns) + r" \\", r"\midrule"]
    for _, r in d.iterrows():
        cells = []
        for c in d.columns:
            s = _tex(_fmt(r[c], digits))
            if c in best and isinstance(r[c], (float, int, np.floating, np.integer)) and r[c] == best[c]:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


class TableWriter:
    """Writes every table as .md, .tex and .csv in a folder of the report and returns the Markdown.
    index: (path relative to the report root, without extension; caption)."""

    def __init__(self, root):
        self.root = root
        self.index = []

    def write(self, name, df, caption="", digits=2, index=True, bold_min=None, folder=None, label=None):
        folder = folder or os.path.join(self.root, "tables")
        os.makedirs(folder, exist_ok=True)
        md = to_markdown(df, digits, index, bold_min)
        with open(os.path.join(folder, name + ".md"), "w", encoding="utf-8") as f:
            f.write(md + "\n")
        with open(os.path.join(folder, name + ".tex"), "w", encoding="utf-8") as f:
            f.write(to_latex(df, caption, "tab:" + (label or name), digits, index, bold_min) + "\n")
        df.to_csv(os.path.join(folder, name + ".csv"), index=index)
        self.index.append((os.path.relpath(os.path.join(folder, name), self.root).replace(os.sep, "/"), caption))
        return md


# ── standard summaries ──────────────────────────────────────────────────────────

def summary_by_method(r, metric="gap_lb", extra=True):
    """One row per method: statistics of the per-instance mean of the metric, success counts, times."""
    rows = []
    for m, g in r.groupby("method", sort=False):
        per = g.groupby("instance")[metric].mean()
        d = S.describe(per.values)
        row = {"method": m, "instances": int(per.notna().sum()), "runs": len(g),
               f"mean {metric}": d["mean"], "sd": d["sd"], "CI95 lo": d["ci_lo"], "CI95 hi": d["ci_hi"],
               "median": d["median"], "max": d["max"]}
        if extra:
            row["#hit LB*"] = int(g.groupby("instance").hit_lb.max().sum())
            row["#best"] = int(g.groupby("instance").hit_bks.max().sum())
            row["mean time (s)"] = float(g.time.mean())
            row["infeasible/failed"] = int((~g.feasible.astype(bool)).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def multi_metric(r):
    """One row per method: means (over instances) of every gap measure, success counts, time, primal integral."""
    rows = []
    for m, g in r.groupby("method", sort=False):
        per = g.groupby("instance")
        row = {"method": m, "instances": int(per.ngroups)}
        if "gap_opt" in g:
            row["#with OPT"] = int(per.gap_opt.mean().notna().sum())
        for c, lab in (("gap_lb", "gap LB* %"), ("gap_bks", "gap BKS %"), ("gap_opt", "gap OPT % (proven-optimum inst.)"),
                       ("gap_cert", "gap own cert. LB %"), ("gap_lp", "gap P_LP %"), ("gap_ref", "rel. diff. exact ref. %")):
            if c in g:
                row[lab] = float(per[c].mean().mean())
        for c, lab in (("hit_bks", "#best"), ("hit_lb", "#= LB*"), ("hit_cert", "#proved by certificate"),
                       ("hit_own", "#proved by own bound")):
            if c in g:
                row[lab] = int(per[c].max().sum())
        row["mean time (s)"] = float(g.time.mean())
        if "pi" in g:
            row["primal integral"] = float(g.pi.mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def seed_variability(r, metric="gap_lb"):
    """Spread over seeds: mean per-instance standard deviation and range of the metric and of the peak."""
    rows = []
    for m, g in r.groupby("method", sort=False):
        if g.seed.nunique() < 2:
            continue
        per = g.groupby("instance")
        rows.append({"method": m, "seeds": int(g.seed.nunique()), f"mean sd {metric}": float(per[metric].std().mean()),
                     f"mean range {metric}": float((per[metric].max() - per[metric].min()).mean()),
                     "mean sd peak (kW)": float(per.objective.std().mean())})
    return pd.DataFrame(rows).set_index("method") if rows else pd.DataFrame()


def wtl_matrix(M, tol=S.TIE):
    """Matrix of wins: entry (A, B) = number of instances where A is strictly better than B."""
    M = M.dropna()
    W = pd.DataFrame(0, index=M.columns, columns=M.columns)
    for a in M.columns:
        for b in M.columns:
            if a != b:
                W.loc[a, b] = int((M[a] < M[b] - tol).sum())
    return W


def pivot_mean(r, metric, rows, cols="method", agg="mean"):
    per = r.groupby([rows, cols, "instance"])[metric].mean().reset_index()
    out = per.pivot_table(index=rows, columns=cols, values=metric, aggfunc=agg)
    if rows in ("n", "T"):
        out.index = out.index.astype(int)
    return out
