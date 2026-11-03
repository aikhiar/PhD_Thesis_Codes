@echo off
setlocal
REM ============================================================================
REM  run_all.bat -- everything, end to end, on this machine (Windows)
REM
REM    run_all.bat              the full run: verification, campaign, report -> results\
REM    run_all.bat --workers 8  extra arguments go to the campaign
REM
REM    Every step uses the cores: theory and faithfulness checks (process pools),
REM    campaign (one worker per physical core but one, sized by memory per block).
REM    Verification (each step once per machine; the campaign does not start
REM    if one of them fails):
REM      [1/4] numerical verification of the theory     (theory\verify_theory.py, 15 min to 1 h)
REM      [2/4] faithfulness check against the original code (~10 min)
REM    Experiments:
REM      [3/4] the campaign: every block of config.yaml (resumes from the stored runs)
REM      [4/4] the report, rebuilt from the raw pkl files -> results\03_report\report.html,
REM            then its audit (runs, folders, expected outputs, files, links, statistics)
REM            -> results\05_logs\audit.md
REM
REM  If anything stops it (licence, disk space, restart of the machine), run this
REM  file again: finished steps and stored runs are skipped.
REM  A fresh results folder (none of its step markers, no stored run) is a clean
REM  run from A to Z: every step runs and the campaign gets its own, empty context
REM  cache (05_logs\campaign_id.txt).
REM  Settings (environment variables; the full run when not set): CFG, OUT, PY,
REM  THEORY_ARGS, FAITH_ARGS.
REM  Keep this window open while it runs (closing it stops the campaign).
REM ============================================================================
cd /d "%~dp0"
set "ARGS=%*"
if not defined CFG set "CFG=config.yaml"
if not defined OUT set "OUT=results"
if not defined FAITH_ARGS set "FAITH_ARGS=--instances 6 --runs 20"
if not defined PY (
  set "PY=python"
  where python >nul 2>&1 || set "PY=py -3"
)
if not exist "%OUT%" mkdir "%OUT%"
for %%I in ("%OUT%") do set "OUTABS=%%~fI"
set "MK=%OUT%\05_logs"
if not exist "%MK%" mkdir "%MK%"
echo Campaign - %date% %time% - config %CFG%, results in %OUT%, step markers in %MK%
set "FRESH=1"
if exist "%MK%\step_*.txt" set "FRESH="
if exist "%OUT%\01_raw_runs" set "FRESH="
if defined FRESH (
  echo Fresh run from A to Z: nothing is stored in %OUT% yet, every step runs.
) else (
  echo Resuming %OUT%: finished steps and stored runs are skipped.
)

echo.
echo ===== [1/4] Numerical verification of the theory (theory\verify_theory.py) =====
if exist "%MK%\step_1_theory_verification_passed.txt" (
  echo already passed on this machine - skipped
) else (
  pushd "theory"
  %PY% verify_theory.py %THEORY_ARGS% --out "%OUTABS%\05_logs"
  if errorlevel 1 (
    popd
    goto failed
  )
  popd
  %PY% -c "import sys, pandas as pd; f = int(pd.read_csv(sys.argv[1]).failures.sum()); print('theory verification failures:', f); sys.exit(1 if f else 0)" "%OUTABS%\05_logs\theory_verification.csv"
  if errorlevel 1 goto failed
  echo passed %date% %time%> "%MK%\step_1_theory_verification_passed.txt"
)

echo.
echo ===== [2/4] Faithfulness check against the original code =====
if exist "%MK%\step_2_faithfulness_check_done.txt" (
  echo already done on this machine - skipped
) else (
  %PY% -m faithfulness.check %FAITH_ARGS% --out "%OUT%\04_faithfulness_check" || goto failed
  echo done %date% %time%> "%MK%\step_2_faithfulness_check_done.txt"
)

echo.
echo ===== [3/4] Campaign (resumes from the stored runs) =====
%PY% -m cri plan %CFG% --output "%OUT%" %ARGS%
%PY% -m cri run %CFG% --output "%OUT%" %ARGS% || goto failed

echo.
echo ===== [4/4] Report (rebuilt from the raw pkl files), then its audit =====
%PY% -m cri report %CFG% --output "%OUT%" || goto failed
%PY% -m cri audit %CFG% --output "%OUT%"
if errorlevel 1 (
  echo The audit found problems: see %MK%\audit.md ^(the report is written^)
  goto failed
)

echo.
echo FINISHED %date% %time%. Report: %OUT%\03_report\report.html
goto end

:failed
echo.
echo STOPPED at the step above (see its messages, and %MK%\run_log.txt for the campaign).
echo Fix the cause if needed, then run run_all.bat again: finished steps and stored runs are skipped.

:end
pause
endlocal
