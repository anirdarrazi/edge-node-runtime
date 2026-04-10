@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo.
  echo The local node app could not be opened. Press any key to close this window.
  pause >nul
)
exit /b %EXITCODE%
