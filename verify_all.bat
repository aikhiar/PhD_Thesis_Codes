@echo off
REM ---------------------------------------------------------------------------
REM End-to-end verification of the repository, in toy configuration.
REM
REM   verify_all.bat                           every step
REM   verify_all.bat --only "appendix d"       the steps whose name contains this text
REM
REM Result on screen and in verification\VERIFICATION_REPORT.txt, logs in
REM verification\logs. See verification\README.md.
REM ---------------------------------------------------------------------------
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PY=python"
where python >nul 2>&1
if errorlevel 1 set "PY=py -3"
%PY% --version >nul 2>&1
if errorlevel 1 (
    echo FAIL: Python was not found on this machine. Install Python 3.10 or later.
    pause
    exit /b 2
)
%PY% -B "%~dp0verification\verify.py" %*
set "CODE=%ERRORLEVEL%"
pause
endlocal & exit /b %CODE%
