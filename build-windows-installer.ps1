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
  throw "Python 3.11 or newer is required to build the Windows installer."
}

$venvPath = Join-Path $PSScriptRoot ".build-venv"
if (-not (Test-Path $venvPath)) {
  Write-Host "Creating build virtual environment..."
  & $pythonLauncher.Command @($pythonLauncher.Args + @("-m", "venv", $venvPath))
}

$venvPython = Join-Path $venvPath "Scripts\\python.exe"
$distPath = Join-Path $PSScriptRoot "dist\\windows"

Write-Host "Installing build dependencies..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e . pyinstaller

if (Test-Path $distPath) {
  Remove-Item -Recurse -Force $distPath
}

Write-Host "Building AUTONOMOUSc Edge Node Setup.exe ..."
& $venvPython -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --name "AUTONOMOUSc Edge Node Setup" `
  --collect-data node_agent `
  --paths src `
  --distpath $distPath `
  src/node_agent/launcher.py

Write-Host "Windows installer output:"
Write-Host (Join-Path $distPath "AUTONOMOUSc Edge Node Setup.exe")
