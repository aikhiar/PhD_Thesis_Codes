"""Command line:  python -m cri <command> [config.yaml] [options]

  all      run the campaign, then build the report (default)
  run      run the campaign only (resumes from the stored runs)
  report   rebuild 02_aggregated_data from the raw pkl files, then build the report
  audit    check a results folder: runs, folder layout, expected outputs, files, links, statistics
  plan     print the tasks and the budget-seconds per experiment, run nothing
  list     list the registered problems, instance families and methods
  forget   remove stored runs (--only blocks, --methods ids) so that the next run recomputes them
"""

import argparse
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cri", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="all", choices=["all", "run", "report", "plan", "list", "forget", "audit"])
    ap.add_argument("config", nargs="?", default="config.yaml")
    ap.add_argument("--only", default="", help="comma-separated experiment names (default: all enabled)")
    ap.add_argument("--workers", type=int, default=None, help="override run.workers")
    ap.add_argument("--output", default=None, help="override run.output")
    ap.add_argument("--cache", default=None, help="override run.cache (folder of the context cache)")
    ap.add_argument("--no-references", action="store_true", help="skip the exact reference solves")
    ap.add_argument("--methods", default="", help="forget: comma-separated method ids (default: all)")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if a.command == "list":
        import cri.problems.ppmsp  # noqa: F401
        from cri.core.registry import METHODS, PROBLEMS
        from cri.problems.ppmsp.generators import FAMILIES
        print("problems:", ", ".join(sorted(PROBLEMS)))
        print("instance families (ppmsp):", ", ".join(sorted(FAMILIES)))
        for k, c in sorted(METHODS.items()):
            doc = (c.__doc__ or "").strip().splitlines()
            tag = "" if getattr(c, "implemented", True) else "  [placeholder, to be integrated]"
            print(f"  {k:10s} {doc[0] if doc else ''}{tag}")
        return 0

    from cri.core.config import load_config
    over = {"run": {}}
    if a.workers:
        over["run"]["workers"] = a.workers
    if a.output:
        over["run"]["output"] = a.output
    if a.cache:
        over["run"]["cache"] = a.cache
    cfg = load_config(a.config, over)
    only = [s.strip() for s in a.only.split(",") if s.strip()] or None
    t0 = time.time()
    if a.command == "forget":
        from cri.core.runner import forget_runs
        if not only:
            print("forget: give the blocks with --only"); return 2
        methods = [m.strip() for m in a.methods.split(",") if m.strip()] or None
        n = forget_runs(cfg, only, methods)
        print(f"forgot {n} runs; they will be recomputed by the next run")
        return 0
    if a.command in ("all", "run", "plan"):
        from cri.core.runner import LowDiskSpace, SolverUnavailable, run_campaign
        try:
            run_campaign(cfg, only=only, dry=a.command == "plan", skip_references=a.no_references)
        except (LowDiskSpace, SolverUnavailable) as ex:
            print(f"stopped: {ex}. Everything finished so far is stored; run the same command again to resume.")
            return 2
    if a.command == "report":
        from cri.core.runner import Runner
        Runner(cfg).rebuild_data()                 # the raw job files are the only source of truth
    if a.command == "audit":
        from cri.analysis.audit import audit
        df = audit(cfg)
        return 0 if df.ok.all() else 1
    if a.command in ("all", "report"):
        from cri.analysis.report import build_report
        md, html = build_report(cfg)
        print(f"report: {html}")
    print(f"total {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
