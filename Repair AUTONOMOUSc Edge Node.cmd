@echo off
setlocal
cd /d "%~dp0"
C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\repair.ps1"
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo.
  echo Repair could not finish. Review the message above, then try again.
  pause
)
exit /b %EXITCODE%
