"""
analysis.py -- statistical tests and figures of the experiment "exact method against Gurobi"
(called by experiment.py; everything is computed from the rows of the comparison).

The reference of a comparison is Gurobi applied to a time-indexed formulation: (P), the
formulation of the paper, or (C), its covering form. The exact method is (UF).

Tests (paired over the instances, on the computing times; an unsolved run counts for the
time it used, i.e. the time limit: this favours the reference)
    Wilcoxon signed-rank test, one-sided (UF faster) and two-sided, Holm-adjusted over the
        references; wins / ties / losses; matched-pairs rank-biserial correlation r
    speed-up t_ref / t_UF: median and geometric mean with percentile bootstrap intervals
    instances solved to proven optimality: exact McNemar test on the discordant pairs
    Friedman test on the times of the three formulations, with their average ranks
Figures (PNG and SVG)
    solved_vs_time        share of the instances solved to optimality within t
    performance_profile   Dolan-More profile of the computing times
    time_per_instance     time of every formulation on every instance (log scale)
    speedup               speed-up of (UF) per class of instances (log scale)
    scaling               time against the number of bookings, per family (log scale)
    binaries              binary variables of each formulation per class (log scale)
"""
import json
import math

import numpy as np

COLOURS = {"UF": "#0072B2", "UF0": "#56B4E9", "C": "#009E73", "P": "#D55E00"}     # Wong palette
MARKERS = {"UF": "o", "UF0": "^", "C": "s", "P": "D"}
NAMES = {"UF": "(UF)", "UF0": "(UF), plain rounding", "C": "Gurobi on (C)", "P": "Gurobi on (P)"}
BOOTSTRAP = 5000
SEED = 20261012


def _holm(pvals):
    """Holm step-down adjusted p-values."""
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, k in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[k])
        adj[k] = min(1.0, running)
    return adj.tolist()


def _bootstrap(values, stat, rng):
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))
    sims = np.array([stat(values[i]) for i in idx])
    return float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))


def _geo(x):
    return float(np.exp(np.mean(np.log(np.maximum(np.asarray(x, dtype=float), 1e-12)))))


def _mcnemar(b, c):
    """Exact two-sided McNemar test: b, c = discordant pairs."""
    from scipy.stats import binomtest
    return 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)


def paired_tests(rows, methods, alpha=0.05, key="time"):
    """Tests of (UF) against every other formulation, over all the instances and per family.
    key: "time" (model, search and schedule) or "solver" (time of the Gurobi search alone)."""
    from scipy.stats import wilcoxon, friedmanchisquare, rankdata, binomtest
    rng = np.random.default_rng(SEED)
    refs = [m for m in methods if m != "UF"]
    out = {"alpha": alpha, "measure": key, "comparisons": [], "friedman": None}
    if "UF" not in methods or not refs:
        return out
    scopes = [("All", rows)] + [(src, [r for r in rows if r["class"].startswith(src)])
                                for src in sorted({r["class"].split(" ")[0] for r in rows})]
    scopes = [(s, sel) for s, sel in scopes if sel] if len(scopes) > 2 else scopes[:1]
    for scope, sel in scopes:
        block = []
        for ref in refs:
            d = [r for r in sel if r.get(f"UF_{key}") is not None and r.get(f"{ref}_{key}") is not None]
            if len(d) < 2:
                continue
            a = np.array([r[f"UF_{key}"] for r in d])
            b = np.array([r[f"{ref}_{key}"] for r in d])
            diff = b - a
            rec = {"scope": scope, "reference": ref, "n": len(d),
                   "wins": int(np.sum(diff > 1e-9)), "ties": int(np.sum(np.abs(diff) <= 1e-9)),
                   "losses": int(np.sum(diff < -1e-9))}
            decided = rec["wins"] + rec["losses"]
            rec["p_sign"] = float(binomtest(rec["wins"], decided, 0.5, alternative="greater").pvalue) if decided else 1.0
            if np.all(np.abs(diff) <= 1e-9):
                rec.update(p_one_sided=1.0, p_two_sided=1.0, rank_biserial=0.0)
            else:
                rec["p_one_sided"] = float(wilcoxon(b, a, alternative="greater", zero_method="wilcox").pvalue)
                rec["p_two_sided"] = float(wilcoxon(b, a, alternative="two-sided", zero_method="wilcox").pvalue)
                nz = diff[np.abs(diff) > 1e-9]
                ranks = rankdata(np.abs(nz))
                rec["rank_biserial"] = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
            su = b / np.maximum(a, 1e-9)
            rec["speedup_median"] = float(np.median(su))
            rec["speedup_median_ci"] = _bootstrap(su, np.median, rng)
            rec["speedup_geomean"] = _geo(su)
            rec["speedup_geomean_ci"] = _bootstrap(su, _geo, rng)
            rec["speedup_min"], rec["speedup_max"] = float(su.min()), float(su.max())
            s_uf = np.array([r["UF_status"] == "optimal" for r in d])
            s_rf = np.array([r[f"{ref}_status"] == "optimal" for r in d])
            rec["solved_UF"], rec["solved_ref"] = int(s_uf.sum()), int(s_rf.sum())
            rec["only_UF"], rec["only_ref"] = int(np.sum(s_uf & ~s_rf)), int(np.sum(~s_uf & s_rf))
            rec["p_mcnemar"] = _mcnemar(rec["only_UF"], rec["only_ref"])
            block.append(rec)
        for rec, p in zip(block, _holm([r["p_one_sided"] for r in block]) if block else []):
            rec["p_holm"] = p
            rec["significant"] = bool(p < alpha)
        out["comparisons"] += block
    full = [r for r in rows if all(r.get(f"{m}_{key}") is not None for m in methods)]
    if len(methods) >= 3 and len(full) >= 3:
        T = np.array([[r[f"{m}_{key}"] for m in methods] for r in full])
        try:
            stat, p = friedmanchisquare(*[T[:, k] for k in range(len(methods))])
        except ValueError:
            stat, p = float("nan"), 1.0
        ranks = np.array([rankdata(row) for row in T]).mean(axis=0)
        out["friedman"] = {"n": len(full), "statistic": float(stat), "p": float(p),
                           "kendall_w": float(stat / (len(full) * (len(methods) - 1))) if stat == stat else None,
                           "average_ranks": {m: float(x) for m, x in zip(methods, ranks)}}
    return out


def _p(x):
    return "--" if x is None else ("$<10^{-4}$" if x < 1e-4 else f"{x:.4f}")


def tests_table(tests, limit, label):
    """LaTeX table of the paired tests."""
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Statistical tests on the computing times: the unit-flow formulation against Gurobi on the "
         r"time-indexed formulations.}", rf"\label{{{label}}}", r"\small",
         r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
         r"Reference & Instances & $N$ & W/T/L & \multicolumn{2}{c}{Speed-up} & $p$ (Holm) & $r$ & Solved & "
         r"$p$ (McNemar) \\",
         r" & & & & median [95\% CI] & geometric mean [95\% CI] & & & (UF)/ref. & \\", r"\midrule"]
    last = None
    for c in sorted(tests["comparisons"], key=lambda c: list(NAMES).index(c["reference"])):
        if last is not None and c["reference"] != last:
            L.append(r"\addlinespace")
        last = c["reference"]
        L.append(f"{NAMES[c['reference']]} & {c['scope']} & {c['n']} & {c['wins']}/{c['ties']}/{c['losses']} & "
                 f"{c['speedup_median']:.1f} [{c['speedup_median_ci'][0]:.1f}, {c['speedup_median_ci'][1]:.1f}] & "
                 f"{c['speedup_geomean']:.1f} [{c['speedup_geomean_ci'][0]:.1f}, {c['speedup_geomean_ci'][1]:.1f}] & "
                 f"{_p(c['p_holm'])} & {c['rank_biserial']:.2f} & {c['solved_UF']}/{c['solved_ref']} & "
                 f"{_p(c['p_mcnemar'])} \\\\")
    note = (rf"{{\footnotesize Paired tests over the instances; time limit {limit:.0f}~s, an unsolved run counting "
            r"for the time it used. W/T/L: instances where (UF) is faster, equal, slower; speed-up: time of the "
            r"reference over time of (UF), a lower bound when the reference reaches the limit, with percentile "
            r"bootstrap intervals; $p$ (Holm): one-sided Wilcoxon "
            r"signed-rank test, Holm-adjusted over the references; $r$: matched-pairs rank-biserial correlation; "
            r"Solved: instances solved to proven optimality; McNemar: exact test on the discordant pairs.")
    f = tests.get("friedman")
    if f:
        ranks = ", ".join(f"{NAMES[m]} {x:.2f}" for m, x in f["average_ranks"].items())
        note += (rf" Friedman test on the times of the {len(f['average_ranks'])} formulations ($N={f['n']}$): "
                 rf"$\chi^2={f['statistic']:.1f}$, $p$ {'$=$ ' + _p(f['p']) if f['p'] >= 1e-4 else _p(f['p'])}; "
                 rf"average ranks: {ranks}.")
    L += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip", note + "}", r"\end{table}"]
    return "\n".join(L) + "\n"


def tests_text(tests):
    L = []
    for c in tests["comparisons"]:
        L.append("(UF) against %-14s %-4s N=%2d  W/T/L %d/%d/%d  speed-up median %.1f [%.1f, %.1f], geo. mean %.1f "
                 "[%.1f, %.1f], range %.1f to %.0f  Wilcoxon one-sided p=%.2e (Holm %.2e), r=%.2f, sign test "
                 "p=%.2e  solved %d vs %d (McNemar p=%.3g)"
                 % (NAMES[c["reference"]], c["scope"], c["n"], c["wins"], c["ties"], c["losses"], c["speedup_median"],
                    *c["speedup_median_ci"], c["speedup_geomean"], *c["speedup_geomean_ci"], c["speedup_min"],
                    c["speedup_max"], c["p_one_sided"], c["p_holm"], c["rank_biserial"], c["p_sign"], c["solved_UF"],
                    c["solved_ref"], c["p_mcnemar"]))
    f = tests.get("friedman")
    if f:
        L.append("Friedman test on the times (N=%d): chi2=%.2f, p=%.2e, Kendall W=%s; average ranks %s"
                 % (f["n"], f["statistic"], f["p"], "%.2f" % f["kendall_w"] if f["kendall_w"] is not None else "-",
                    {m: round(x, 2) for m, x in f["average_ranks"].items()}))
    return L


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def figures(rows, methods, limit, out_dir, code="appendix_d", dpi=300, larger=()):
    """rows: the instances of the comparison; larger: rows of larger instances, added to the
    scaling figure only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (out_dir / "PNG").mkdir(parents=True, exist_ok=True)
    (out_dir / "SVG").mkdir(parents=True, exist_ok=True)
    present = [m for m in methods if any(f"{m}_time" in r for r in rows)]
    classes = []
    for r in rows:
        if r["class"] not in classes:
            classes.append(r["class"])

    def save(fig, name):
        fig.savefig(out_dir / "PNG" / f"{code}_exact_{name}.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(out_dir / "SVG" / f"{code}_exact_{name}.svg", bbox_inches="tight")
        plt.close(fig)

    def solved_times(m):
        return sorted(r[f"{m}_time"] for r in rows if r.get(f"{m}_status") == "optimal")

    # 1. share of the instances solved to optimality within t
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in present:
        n = sum(f"{m}_time" in r for r in rows)
        ts = solved_times(m)
        xs = [min([limit] + ts) * 0.8 if ts else limit] + ts + [limit]
        ys = [0.0] + [100.0 * (k + 1) / n for k in range(len(ts))] + [100.0 * len(ts) / n]
        ax.step(xs, ys, where="post", color=COLOURS[m], label=NAMES[m], lw=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Instances solved to proven optimality (%)")
    ax.set_ylim(0, 103)
    ax.axvline(limit, color="0.6", ls=":", lw=0.8)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "solved_vs_time")

    # 2. performance profile of the times (an unsolved instance has an infinite ratio)
    full = [r for r in rows if all(f"{m}_time" in r for m in present)]
    if len(present) >= 2 and full:
        T = np.array([[r[f"{m}_time"] if r[f"{m}_status"] == "optimal" else np.inf for m in present] for r in full])
        best = np.min(T, axis=1)
        ok = np.isfinite(best)
        ratios = T[ok] / best[ok][:, None]
        finite = ratios[np.isfinite(ratios)]
        tmax = max(2.0, float(finite.max()) * 1.3) if finite.size else 2.0
        grid = np.geomspace(1.0, tmax, 400)
        fig, ax = plt.subplots(figsize=(6, 4))
        for k, m in enumerate(present):
            ax.step(grid, [100.0 * np.mean(ratios[:, k] <= t) * ok.mean() for t in grid], where="post",
                    color=COLOURS[m], label=NAMES[m], lw=1.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"Time ratio to the fastest formulation, $\theta$")
        ax.set_ylabel(r"Instances solved within $\theta$ $\times$ best time (%)", fontsize=9)
        ax.set_ylim(0, 103)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower right")
        save(fig, "performance_profile")

    # 3. time on every instance
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(rows))
    for m in present:
        ys = [r.get(f"{m}_time", np.nan) for r in rows]
        closed = [r.get(f"{m}_status") == "optimal" for r in rows]
        ax.scatter(x[np.array(closed)], np.array(ys)[np.array(closed)], color=COLOURS[m], marker=MARKERS[m], s=22,
                   label=NAMES[m], zorder=3)
        opened = ~np.array(closed) & np.isfinite(np.array(ys, dtype=float))
        ax.scatter(x[opened], np.array(ys)[opened], facecolors="none", edgecolors=COLOURS[m], marker=MARKERS[m],
                   s=22, zorder=3)
    for k in range(1, len(rows)):
        if rows[k]["class"] != rows[k - 1]["class"]:
            ax.axvline(k - 0.5, color="0.85", lw=0.8, zorder=0)
    ax.axhline(limit, color="0.5", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r["id"]) if r["id"] else r["name"] for r in rows], fontsize=6, rotation=90)
    ax.set_xlabel("Instance (hollow marker: not solved within the limit)")
    ax.set_ylabel("Time (s)")
    ax.legend(fontsize=8, ncol=len(present), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    save(fig, "time_per_instance")

    # 4. speed-up of (UF) per class
    refs = [m for m in present if m != "UF"]
    if "UF" in present and refs:
        fig, ax = plt.subplots(figsize=(6, 4))
        width = 0.8 / len(refs)
        for k, ref in enumerate(refs):
            data = [[r[f"{ref}_time"] / max(r["UF_time"], 1e-9) for r in rows
                     if r["class"] == cl and f"{ref}_time" in r and "UF_time" in r] for cl in classes]
            pos = np.arange(len(classes)) - 0.4 + width * (k + 0.5)
            keep = [i for i, d in enumerate(data) if d]
            if not keep:
                continue
            bp = ax.boxplot([data[i] for i in keep], positions=pos[keep], widths=width * 0.85, patch_artist=True,
                            medianprops={"color": "black"}, flierprops={"markersize": 3})
            for box in bp["boxes"]:
                box.set_facecolor(COLOURS[ref])
                box.set_alpha(0.75)
            ax.plot([], [], color=COLOURS[ref], lw=6, alpha=0.75, label=f"against {NAMES[ref]}")
        ax.axhline(1.0, color="0.4", lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("Speed-up of (UF) (time ratio)")
        ax.grid(alpha=0.3, axis="y", which="both")
        ax.legend(fontsize=8)
        save(fig, "speedup")

    # 5. time against the number of bookings, per family
    every = list(rows) + list(larger)
    families = sorted({r["class"].split(" ")[0] for r in every})
    fig, axes = plt.subplots(1, len(families), figsize=(4.6 * len(families), 3.6), squeeze=False, sharey=True)
    for ax, fam in zip(axes[0], families):
        sel = [r for r in every if r["class"].startswith(fam)]
        sizes = sorted({r["n"] for r in sel})
        for m in present:
            med = [np.median([r[f"{m}_time"] for r in sel if r["n"] == n and f"{m}_time" in r] or [np.nan])
                   for n in sizes]
            lo = [np.min([r[f"{m}_time"] for r in sel if r["n"] == n and f"{m}_time" in r] or [np.nan]) for n in sizes]
            hi = [np.max([r[f"{m}_time"] for r in sel if r["n"] == n and f"{m}_time" in r] or [np.nan]) for n in sizes]
            ax.plot(sizes, med, color=COLOURS[m], marker=MARKERS[m], ms=5, label=NAMES[m])
            ax.fill_between(sizes, lo, hi, color=COLOURS[m], alpha=0.15)
        ax.axhline(limit, color="0.5", ls=":", lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xlabel(f"Number of bookings ({fam})")
        ax.grid(alpha=0.3, which="both")
    axes[0][0].set_ylabel("Time (s): median, minimum to maximum")
    axes[0][0].legend(fontsize=8)
    save(fig, "scaling")

    # 6. binary variables per class
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.8 / len(present)
    for k, m in enumerate(present):
        vals = [np.mean([r[f"{m}_bin"] for r in rows if r["class"] == cl and f"{m}_bin" in r] or [np.nan])
                for cl in classes]
        ax.bar(np.arange(len(classes)) - 0.4 + width * (k + 0.5), vals, width * 0.9, color=COLOURS[m], label=NAMES[m])
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Binary variables (mean)")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=8)
    save(fig, "binaries")


def write_tests(tests, path):
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clean(tests), fh, indent=2)
