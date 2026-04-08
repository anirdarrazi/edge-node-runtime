Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$venvPath = Join-Path $PSScriptRoot ".installer-venv"
if (-not (Test-Path $venvPath)) {
  throw "The local node service virtual environment does not exist yet."
}

$venvPython = Join-Path $venvPath "Scripts\\python.exe"
Write-Host "Stopping the local node runtime service..."
& $venvPython -m node_agent.service stop
