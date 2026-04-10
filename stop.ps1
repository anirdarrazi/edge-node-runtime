Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
. (Join-Path $PSScriptRoot "windows-runtime-common.ps1")

Stop-NodeApp -RootPath $PSScriptRoot
