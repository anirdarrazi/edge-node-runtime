@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0app.ps1" %*
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo.
  echo The AUTONOMOUSc Edge Node app could not be opened. Press any key to close this window.
  pause >nul
)
exit /b %EXITCODE%
