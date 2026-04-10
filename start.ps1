Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
. (Join-Path $PSScriptRoot "windows-runtime-common.ps1")

Start-NodeApp -RootPath $PSScriptRoot -RefreshDependencies
