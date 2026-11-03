@echo off
REM ---------------------------------------------------------------------------
REM methods_comparison: the metaheuristics comparisons of the thesis.
REM Studies and protocol: comparison_config.py (this folder). Results: experiments\
REM
REM   run_all.bat                        every study (Chapters 2 to 6, Appendices B, C, D)
REM   run_all.bat --only chapter6        one study (several names allowed)
REM   run_all.bat --estimate             expected duration of each study
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"
python -m methods_comparison.run_all %*
set CODE=%ERRORLEVEL%
endlocal & exit /b %CODE%
