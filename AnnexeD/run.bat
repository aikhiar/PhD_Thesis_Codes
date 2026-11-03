@echo off
REM ---------------------------------------------------------------------------
REM Appendix D, experiment 1: the exact method against Gurobi.
REM The unit-flow formulation (UF) is compared, through the computing time, with
REM Gurobi applied to the formulation (P) of the paper and to its covering form
REM (C): same solver, time limit, threads and gap tolerance.
REM
REM   run.bat             the experiment of the paper -> results\appendix_d_exact_methods
REM                         the 40 instances, (UF), (C) and (P), 1800 s per run;
REM                         larger instances (300, 400, 500 bookings), (UF) only;
REM                         then tables, statistical tests and figures.
REM                       Resumable: finished runs are cached (runs\*.json).
REM   run.bat --report    tables, tests and figures only, from the cached runs
REM   run.bat --name X --limit 300 --ids 1-10,21-30 --methods UF,C,P
REM                       any other experiment (options of experiment.py) -> results\X
REM
REM Worst case: (P) and (C) reach the limit on many instances, 2 x 40 x 30 min = 40 h.
REM (UF) alone takes a few minutes: run.bat --name appendix_d_exact_methods --methods UF
REM --extra 300,400,500, then run.bat adds (C) and (P): the finished runs are kept.
REM Do not run anything else on the machine: times are measured.
REM
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"
if "%~1"=="" goto :full
if "%~1"=="--report" goto :report
python experiment.py %*
goto :end

:full
python experiment.py --name appendix_d_exact_methods --limit 1800 --ids 1-40 --methods UF,C,P --extra 300,400,500 --extra-methods UF
goto :end

:report
python experiment.py --name appendix_d_exact_methods --limit 1800 --ids 1-40 --methods UF,C,P --extra 300,400,500 --extra-methods UF --report

:end
set CODE=%ERRORLEVEL%
endlocal & exit /b %CODE%
