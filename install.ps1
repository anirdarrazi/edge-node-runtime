Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$pythonLauncher = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
  $pythonLauncher = @{
    Command = "py"
    Args = @("-3")
  }
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
  $pythonLauncher = @{
    Command = "python"
    Args = @()
  }
}

if (-not $pythonLauncher) {
  throw "Python 3.11 or newer is required to launch the guided installer."
}

$venvPath = Join-Path $PSScriptRoot ".installer-venv"
if (-not (Test-Path $venvPath)) {
  Write-Host "Creating installer virtual environment..."
  & $pythonLauncher.Command @($pythonLauncher.Args + @("-m", "venv", $venvPath))
}

$venvPython = Join-Path $venvPath "Scripts\\python.exe"

Write-Host "Installing local node service dependencies..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e .

Write-Host "Starting the local node runtime service..."
& $venvPython -m node_agent.service start --open
