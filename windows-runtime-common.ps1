Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-PythonLauncher {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Command,

    [Parameter()]
    [string[]]$Args = @()
  )

  $probe = @($Args + @("-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"))
  & $Command @probe *> $null
  return $LASTEXITCODE -eq 0
}

function Resolve-PythonLauncher {
  $candidates = @()

  if (Get-Command py -ErrorAction SilentlyContinue) {
    $candidates += @(
      @{ Command = "py"; Args = @("-3.11") },
      @{ Command = "py"; Args = @("-3") }
    )
  }

  if (Get-Command python -ErrorAction SilentlyContinue) {
    $candidates += @(
      @{ Command = "python"; Args = @() }
    )
  }

  foreach ($candidate in $candidates) {
    try {
      if (Test-PythonLauncher -Command $candidate.Command -Args $candidate.Args) {
        return $candidate
      }
    } catch {
      continue
    }
  }

  throw "Python 3.11 or newer is required. Install Python 3.11+ and then run this launcher again."
}

function Get-NodeServiceVenvPython {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RootPath,

    [switch]$Create,
    [switch]$RefreshDependencies
  )

  $venvPath = Join-Path $RootPath ".installer-venv"

  if (-not (Test-Path $venvPath)) {
    if (-not $Create) {
      throw "The local node service environment has not been created yet. Run the install launcher first."
    }

    $pythonLauncher = Resolve-PythonLauncher
    Write-Host "Creating the local AUTONOMOUSc Edge Node environment..."
    & $pythonLauncher.Command @($pythonLauncher.Args + @("-m", "venv", $venvPath))
  }

  $venvPython = Join-Path $venvPath "Scripts\python.exe"
  if (-not (Test-Path $venvPython)) {
    throw "The local node service Python environment is incomplete. Delete .installer-venv and run the install launcher again."
  }

  $needsBootstrap = $RefreshDependencies
  if (-not $needsBootstrap) {
    & $venvPython -c "import node_agent" *> $null
    $needsBootstrap = $LASTEXITCODE -ne 0
  }

  if ($needsBootstrap) {
    Write-Host "Preparing the local AUTONOMOUSc Edge Node app..."
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -e .
  }

  return $venvPython
}

function Start-NodeApp {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RootPath,

    [switch]$RefreshDependencies
  )

  Set-Location $RootPath
  $venvPython = Get-NodeServiceVenvPython -RootPath $RootPath -Create -RefreshDependencies:$RefreshDependencies
  Write-Host "Opening the AUTONOMOUSc Edge Node app..."
  & $venvPython -m node_agent.service start --open
}

function Stop-NodeApp {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RootPath
  )

  Set-Location $RootPath
  $venvPython = Get-NodeServiceVenvPython -RootPath $RootPath
  Write-Host "Stopping the AUTONOMOUSc Edge Node app..."
  & $venvPython -m node_agent.service stop
}

function Repair-NodeApp {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RootPath,

    [switch]$RefreshDependencies
  )

  Set-Location $RootPath
  $venvPython = Get-NodeServiceVenvPython -RootPath $RootPath -Create -RefreshDependencies:$RefreshDependencies
  Write-Host "Repairing the AUTONOMOUSc Edge Node app..."
  & $venvPython -m node_agent.service repair --open
}
