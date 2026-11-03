"""
verify.py -- end-to-end check of the repository (run by verify_all.bat at its root).

It checks the environment and the files, then runs every experiment of the thesis in a toy
configuration: the toy sizes are set at run time by the scripts of this folder, the code of the
repository is never changed. Every toy output goes to a "toy" folder of its framework (real results
are never touched), and is removed by the next verification.

A step passes only if its process ends with exit code 0, its log contains the expected success
message and no error: the exit code alone is not enough. The toy scripts also check that each
study writes the outputs its chapter reports, and no other.

    python verification/verify.py                              every step
    python verification/verify.py --only "appendix d,chapter 3"   the steps whose name contains one of these
"""

import argparse
import datetime
import importlib
import importlib.metadata as md
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOGS = HERE / "logs"
REPORT = HERE / "VERIFICATION_REPORT.txt"
TOY_MC = HERE / "toy_methods_comparison.py"
TOY_SURROGATE = HERE / "toy_surrogate.py"
PILOT = HERE / "pilot_det_vs_stoch.py"

MC = ROOT / "Frameworks" / "methods_comparison"
DS = ROOT / "Frameworks" / "det_vs_stoch"
ST = ROOT / "Frameworks" / "surrogate_trainer"
AA = ROOT / "AnnexeA"
AD = ROOT / "AnnexeD"
SURROGATE_STUDIES = ("chapter5", "chapter6")
SURROGATES = {s: MC / "problems" / s / "surrogate_models" for s in SURROGATE_STUDIES}
SURROGATE_FILES = ("*.joblib", "model_card.json")
BACKUPS = {s: HERE / f"{s}_surrogates_backup" for s in SURROGATE_STUDIES}
STUDY_D = MC / "experiments" / "toy" / "appendix_d"

MINUTE, HOUR = 60, 3600
MIN_FREE_GB = 3
LINE = "=" * 78
TRACEBACK = "Traceback (most recent call last)"
OUTPUT_FOLDERS = {"experiments", "runs", "results", "results_quick", "__pycache__", "logs"}

IMPORT_NAMES = {"scikit-learn": "sklearn", "pyyaml": "yaml"}

# the files without which the repository is incomplete
REQUIRED_FILES = [
    "README.md", "requirements.txt", "verify_all.bat",
    "Frameworks/methods_comparison/run_all.bat", "Frameworks/methods_comparison/comparison_config.py",
    "Frameworks/methods_comparison/methods_comparison/run_all.py",
    "Frameworks/det_vs_stoch/main.py", "Frameworks/surrogate_trainer/main.py",
    "Frameworks/methods_comparison/problems/chapter5/surrogate_models/README.md",
    "Frameworks/methods_comparison/problems/chapter6/surrogate_models/README.md",
    "AnnexeA/run_all.bat", "AnnexeA/config.yaml", "AnnexeD/experiment.py", "AnnexeD/run.bat",
    "AnnexeD/complements.py",
] + [f"Frameworks/methods_comparison/problems/problem_{s}.py" for s in
     ("chapter2", "chapter3", "chapter4", "chapter5", "chapter6", "appendix_b", "appendix_c", "appendix_d")] \
  + [f"Frameworks/det_vs_stoch/problems/problem_{s}.py" for s in ("chapter2", "chapter3", "chapter4", "chapter6")] \
  + [f"Frameworks/surrogate_trainer/problems/problem_{s}.py" for s in ("chapter5", "chapter6")]

# scripts that the verification compiles but does not run; printed in every report
NOT_RUN = [
    "AnnexeA/tools/paper_artifacts.py: needs the results of the full campaign",
    "AnnexeA/tools/hunsaker_alpha_table.py: would rewrite a data file of the repository",
]

ENV = dict(os.environ, NO_AUTO_INSTALL="1", MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
ENV["PATH"] = str(Path(sys.executable).parent) + os.pathsep + ENV.get("PATH", "")

report = None


def say(text: str = "") -> None:
    print(text, flush=True)
    report.write(text + "\n")
    report.flush()


@dataclass
class Row:
    group: str
    name: str
    reason: Optional[str] = None          # None: passed


@dataclass
class Step:
    group: str
    name: str
    cwd: Optional[Path] = None
    cmd: Any = None                       # the command, or a function that returns it
    expect: tuple = ()                    # every text that the log must contain
    forbid: tuple = (TRACEBACK,)          # the texts that the log must not contain
    after: Any = None                     # function: None, or what is wrong
    env: Optional[dict] = None
    timeout: int = 30 * MINUTE
    action: Any = None                    # function run in place of a command: None, or what is wrong


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

def check_python() -> bool:
    bits = struct.calcsize("P") * 8
    say(f"  Python {platform.python_version()} ({bits} bits): {sys.executable}")
    if sys.version_info < (3, 10) or bits != 64:
        say("  FAIL: Python 3.10 or later, 64 bits, is needed.")
        return False
    return True


def required_packages() -> list:
    out = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"([A-Za-z0-9_.-]+)\s*(==|>=)\s*([0-9][^\s#]*)", line)
        if match:
            out.append(match.groups())
    return out


def check_packages() -> bool:
    packages = required_packages()
    missing, different = [], []
    for name, operator, version in packages:
        try:
            found = md.version(name)
            importlib.import_module(IMPORT_NAMES.get(name, name))
        except Exception as exc:
            missing.append(f"{name} ({type(exc).__name__}: {exc})")
            continue
        if operator == "==" and found != version:
            different.append(f"{name} {found} (tested with {version})")
    say(f"  Packages: {len(packages) - len(missing)} of {len(packages)} installed and importable")
    for item in different:
        say(f"    other version, harmless: {item}")
    if missing:
        say("  FAIL: missing or unusable packages:")
        for item in missing:
            say(f"    {item}")
        say(f"  To do: python -m pip install -r \"{ROOT / 'requirements.txt'}\"")
        return False
    return True


def check_machine() -> bool:
    import psutil
    physical = psutil.cpu_count(logical=False) or 0
    logical = psutil.cpu_count(logical=True) or os.cpu_count() or 0
    ram = psutil.virtual_memory().total / 2 ** 30
    free = shutil.disk_usage(ROOT).free / 2 ** 30
    say(f"  System: {platform.platform()}")
    say(f"  Cores: {physical} physical, {logical} logical")
    say(f"    methods_comparison will use {physical} processes; det_vs_stoch and surrogate_trainer {logical}")
    say(f"  Memory: {ram:.1f} GB; free disk: {free:.1f} GB")
    say(f"  Folder: {ROOT}")
    if "onedrive" in str(ROOT).lower():
        say("  Note: the folder is synchronized by OneDrive, which can lock files during a run.")
    if free < MIN_FREE_GB:
        say(f"  FAIL: less than {MIN_FREE_GB} GB free on the disk.")
        return False
    return True


def check_gurobi() -> bool:
    """A model above the 2000 variables of the size-limited licence: AnnexeA and AnnexeD need a full one."""
    try:
        import gurobipy as gp
        version = ".".join(map(str, gp.gurobi.version()))
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:
                x = model.addVars(2500, vtype=gp.GRB.BINARY)
                model.addConstr(x.sum() <= 10)
                model.setObjective(x.sum(), gp.GRB.MAXIMIZE)
                model.optimize()
                if model.Status != gp.GRB.OPTIMAL or round(model.ObjVal) != 10:
                    raise RuntimeError(f"status {model.Status}")
    except Exception as exc:
        say(f"  FAIL: Gurobi unusable ({type(exc).__name__}: {exc})")
        say("    AnnexeA, AnnexeD and the exact method of Appendix D need Gurobi with a full licence.")
        return False
    say(f"  Gurobi {version}: full licence valid")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# FILES
# ══════════════════════════════════════════════════════════════════════════════

def check_files() -> bool:
    """The key files are there, and every Python file compiles."""
    missing = [rel for rel in REQUIRED_FILES if not (ROOT / rel).is_file()]
    python_files = [p for p in ROOT.rglob("*.py") if not OUTPUT_FOLDERS & set(p.relative_to(ROOT).parts)]
    broken = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in python_files:
            try:
                compile(p.read_bytes(), str(p), "exec")
            except (SyntaxError, ValueError) as exc:
                broken.append(f"{p.relative_to(ROOT)} ({exc})")
    say(f"  Key files: {len(REQUIRED_FILES) - len(missing)} of {len(REQUIRED_FILES)} present")
    say(f"  Python: {len(python_files) - len(broken)} of {len(python_files)} files without syntax error")
    for title, items in (("missing", missing), ("syntax error", broken)):
        for item in items[:30]:
            say(f"    FAIL, {title}: {item}")
    return not (missing or broken)


# ══════════════════════════════════════════════════════════════════════════════
# THE EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════

def python(*args) -> list:
    return [sys.executable, *map(str, args)]


def files_under(base: Path, pattern: str, minimum: int = 1):
    """Step.after: `pattern` matches at least `minimum` files under `base`."""
    def check():
        found = len(list(base.glob(pattern)))
        return None if found >= minimum else f"expected at least {minimum} file(s) {base / pattern}, found {found}"
    return check


def annexe_a_results() -> Path:
    """The audit of AnnexeA needs a results folder path of at most 109 characters."""
    inside = AA / "results_quick"
    return inside if len(str(inside)) <= 105 else Path(tempfile.gettempdir()) / "cri_toy"


def surrogate_files(folder: Path) -> list:
    return [f for pattern in SURROGATE_FILES for f in folder.glob(pattern)]


def deploy_toy_surrogates(study: str) -> Optional[str]:
    """The toy models of `study` (chapter5 or chapter6), for this run only (the chain surrogate_trainer ->
    methods_comparison). Models already deployed there are set aside, and restored at the end by
    restore_surrogates()."""
    chapter = study.replace("chapter", "Chapter ")
    runs = sorted(ST.glob(f"runs/toy/surrogate_{study}_*/surrogate_export/best_models"),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        return f"no toy model: the toy training of {chapter} did not succeed"
    target, backup = SURROGATES[study], BACKUPS[study]
    present = surrogate_files(target)
    backup.mkdir(parents=True, exist_ok=True)   # also marks that toy models are deployed, even if none was set aside
    if present:
        for f in present:
            shutil.move(str(f), str(backup / f.name))
        say(f"        (deployed models of {chapter} set aside in {backup}, restored at the end)")
    target.mkdir(parents=True, exist_ok=True)
    for f in runs[-1].iterdir():
        if f.is_file():
            shutil.copy2(f, target / f.name)
            DEPLOYED.append(target / f.name)
    return files_under(target, "*.joblib", 2)()


DEPLOYED = []          # the toy models copied by this verification


def restore_surrogates(after_interruption: bool = False) -> None:
    """Removes the toy models of Chapters 5 and 6 and puts back the models set aside, if any. With
    `after_interruption` (a backup left by a verification that was stopped), every model file
    present in a folder with a backup is a toy one."""
    for study in SURROGATE_STUDIES:
        target, backup = SURROGATES[study], BACKUPS[study]
        if after_interruption:
            toy = surrogate_files(target) if backup.is_dir() else []
        else:
            toy = [f for f in DEPLOYED if f.exists() and f.parent == target]
        for f in toy:
            f.unlink()
        if backup.is_dir():
            for f in backup.iterdir():
                shutil.move(str(f), str(target / f.name))
            backup.rmdir()
    DEPLOYED.clear()


def study_d() -> str:
    found = sorted(STUDY_D.rglob("appendix_d_instances.pkl"))
    if not found:
        raise FileNotFoundError(f"toy study of Appendix D not found in {STUDY_D}")
    return str(found[0].parent)


def mc_steps(label: str, study: str) -> list:
    return [
        Step("methods_comparison", f"{label}: toy study", MC, python(TOY_MC, study),
             expect=("Toy run passed !",), timeout=HOUR),
        Step("methods_comparison", f"{label}: duration estimate", MC, python("-m", f"problems.problem_{study}"),
             expect=("core-h",)),
    ]


def build_steps() -> list:
    steps = []

    group = "methods_comparison"
    for label, study in (("chapter 2", "chapter2"), ("chapter 3", "chapter3"), ("chapter 4", "chapter4"),
                         ("appendix B", "appendix_b"), ("appendix C", "appendix_c"), ("appendix D", "appendix_d")):
        steps += mc_steps(label, study)
    steps.append(Step(group, "launcher run_all.bat", MC,
                      ["cmd", "/c", ".\\run_all.bat", "--estimate", "--only", "chapter2"], expect=("core-h",)))

    group = "det_vs_stoch"
    for label, code in (("chapter 2", "chapter2"), ("chapter 3", "chapter3"), ("chapter 4", "chapter4"),
                        ("chapter 6", "chapter6")):
        steps.append(Step(group, f"{label}: toy comparison", DS, python(PILOT, code),
                          expect=(f"=== {code}: done", "Pilot run passed !"), timeout=HOUR,
                          after=files_under(DS / "runs" / "toy", f"det_stoch_{code}_*/manifest.json")))
    steps.append(Step(group, "launcher main.py", DS, python("main.py", "--estimate", "chapter2"), expect=("core-h",)))

    group = "surrogate_trainer"
    for label, code in (("chapter 5", "chapter5"), ("chapter 6", "chapter6")):
        steps.append(Step(group, f"{label}: toy training", ST, python(TOY_SURROGATE, code),
                          expect=(f"Toy training done: {code}",), timeout=HOUR,
                          after=files_under(ST / "runs" / "toy",
                                            f"surrogate_{code}_*/surrogate_export/best_models/*.joblib", 2)))
    steps.append(Step(group, "launcher main.py", ST, python("main.py", "--estimate", "chapter5"),
                      expect=("=> data generation",)))

    group = "methods_comparison"
    for label, study in (("chapter 5", "chapter5"), ("chapter 6", "chapter6")):
        steps.append(Step(group, f"{label}: toy surrogates deployed",
                          action=lambda study=study: deploy_toy_surrogates(study)))
        steps += mc_steps(label, study)

    group = "AnnexeD"
    steps += [
        Step(group, "toy experiment UF, C, P", AD,
             python("experiment.py", "--name", "toy", "--limit", 60, "--ids", "1,21", "--methods", "UF,C,P",
                    "--extra", 300, "--extra-methods", "UF"),
             expect=("results ->",), timeout=HOUR, after=files_under(AD / "results" / "toy", "report.txt")),
        Step(group, "launcher run.bat", AD,
             ["cmd", "/c", ".\\run.bat", "--name", "toy_bat", "--limit", "20", "--ids", "1", "--methods", "UF"],
             expect=("results ->",)),
        Step(group, "complements of the toy study of Appendix D", AD,
             lambda: python("complements.py", study_d(), "--out", AD / "results" / "toy_complements"),
             expect=("tables ->",), after=files_under(AD / "results" / "toy_complements" / "Tables", "*.tex", 3)),
    ]

    group = "AnnexeA"
    results = annexe_a_results()
    steps += [
        Step(group, "list of the methods", AA, python("-m", "cri", "list"), expect=("WLNS",)),
        # the real run_all.bat on the quick settings: theory, faithfulness, campaign, report and audit
        Step(group, "whole chain, run_all.bat", AA, ["cmd", "/c", ".\\run_all.bat"],
             expect=("FINISHED",), forbid=("STOPPED at the step above",), timeout=5 * HOUR,
             env={"OUT": str(results), "CFG": "..\\verification\\config_quick.yaml", "THEORY_ARGS": "--quick",
                  "FAITH_ARGS": "--instances 2 --runs 3"},
             after=lambda: files_under(results, "03_report/report.html")() or files_under(results, "05_logs/audit.md")()),
    ]
    return steps


def clean_outputs() -> None:
    """The toy results of a previous verification, so that a step can never pass on old files. Only
    the toy folders: the results of real runs are never touched."""
    for folder in (MC / "experiments" / "toy", DS / "runs" / "toy", ST / "runs" / "toy",
                   AD / "results" / "toy", AD / "results" / "toy_bat", AD / "results" / "toy_complements",
                   AA / "results_quick", annexe_a_results(), LOGS):
        shutil.rmtree(folder, ignore_errors=True)
    LOGS.mkdir(parents=True, exist_ok=True)


def execute(step: Step, log_path: Path) -> Optional[str]:
    """Runs one step. Returns None when it passes, otherwise what went wrong."""
    if step.action:
        return step.action()
    cmd = step.cmd() if callable(step.cmd) else step.cmd
    env = dict(ENV, **(step.env or {}))
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write("> " + " ".join(cmd) + "\n\n")
        log.flush()
        process = subprocess.Popen(cmd, cwd=step.cwd, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=step.timeout)
        except subprocess.TimeoutExpired:
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(process.pid)], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
            return f"exceeds the limit of {step.timeout / MINUTE:.0f} min"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if code != 0:
        return f"exit code {code}"
    for mark in step.forbid:
        if mark in text:
            return f"the log contains: {mark}"
    for mark in step.expect:
        if mark not in text:
            return f"success message missing from the log: {mark}"
    return step.after() if step.after else None


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def run_step(step: Step, number: int, total: int) -> Row:
    prefix = f"  [{number:2d}/{total}] {step.group} - {step.name}"
    print(f"{prefix} ... ", end="", flush=True)
    log_path = LOGS / f"{number:02d}_{slug(step.group)}_{slug(step.name)}.log"
    t0 = time.time()
    try:
        reason = execute(step, log_path)
    except Exception as exc:                  # a step that cannot even start
        reason = f"{type(exc).__name__}: {exc}"
    verdict = f"{'PASS' if reason is None else 'FAIL'} ({time.time() - t0:.0f} s)"
    print(verdict, flush=True)
    report.write(f"{prefix} ... {verdict}\n")
    report.flush()
    if reason is not None:
        say(f"        Problem: {reason}")
        if log_path.is_file():
            say(f"        Log: {log_path}")
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]:
                say(f"        | {line[:150]}")
    return Row(step.group, step.name, reason)


def conclude(rows: list, partial: bool) -> int:
    failed = [r for r in rows if r.reason is not None]
    say()
    say(LINE)
    say("SUMMARY")
    for group in dict.fromkeys(r.group for r in rows):
        members = [r for r in rows if r.group == group]
        good = sum(r.reason is None for r in members)
        say(f"  {group:30s} {'PASS' if good == len(members) else 'FAIL'} ({good}/{len(members)})")
    if failed:
        say()
        say("To fix:")
        for r in failed:
            say(f"  {r.group} - {r.name}: {r.reason}")
    say()
    say("Compiled but not run:")
    for item in NOT_RUN:
        say(f"  {item}")
    say()
    say(f"Duration: {datetime.datetime.now() - STARTED}".split(".")[0])
    say(f"Report: {REPORT}")
    say(f"Logs: {LOGS}")
    say(LINE)
    if failed:
        say(f"RESULT: FAIL, {len(failed)} of {len(rows)} steps")
    elif partial:
        say(f"RESULT: PASS for the {len(rows)} chosen steps (partial verification, --only)")
    else:
        say(f"RESULT: ALL PASS, {len(rows)} of {len(rows)} steps. The codes run without error.")
    say(LINE)
    return 1 if failed else 0


STARTED = datetime.datetime.now()


def main() -> int:
    global report
    parser = argparse.ArgumentParser(description="End-to-end check of the repository.")
    parser.add_argument("--only", default="", help="comma-separated texts: the steps whose name contains one")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = open(REPORT, "w", encoding="utf-8")
    say(LINE)
    say(f"VERIFICATION OF THE REPOSITORY, {STARTED:%Y-%m-%d %H:%M}")
    say(LINE)

    rows = []
    say()
    say("[1/3] ENVIRONMENT")
    python_ok = check_python()
    rows.append(Row("Environment", "Python", None if python_ok else "Python version"))
    packages_ok = python_ok and check_packages()
    rows.append(Row("Environment", "packages", None if packages_ok else "missing packages"))
    if not packages_ok:
        return conclude(rows, False)
    rows.append(Row("Environment", "machine", None if check_machine() else "not enough disk space"))
    rows.append(Row("Environment", "Gurobi", None if check_gurobi() else "Gurobi or its licence"))

    say()
    say("[2/3] FILES")
    stopped = [s for s in SURROGATE_STUDIES if BACKUPS[s].is_dir()]
    if stopped:                                  # a previous verification was stopped mid-way
        restore_surrogates(after_interruption=True)
        say("  Deployed models restored from " + ", ".join(str(BACKUPS[s]) for s in stopped))
    clean_outputs()
    rows.append(Row("Files", "repository complete", None if check_files() else "missing files or syntax errors"))

    say()
    say("[3/3] EXPERIMENTS IN TOY CONFIGURATION")
    steps = build_steps()
    wanted = [w.strip().lower() for w in args.only.split(",") if w.strip()]
    if wanted:
        steps = [s for s in steps if any(w in f"{s.group} {s.name}".lower() for w in wanted)]
    try:
        for number, step in enumerate(steps, 1):
            rows.append(run_step(step, number, len(steps)))
    finally:
        restore_surrogates()
    return conclude(rows, bool(wanted))


if __name__ == "__main__":
    sys.exit(main())
