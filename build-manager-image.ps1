Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$imageName = if ($args.Length -ge 1 -and $args[0]) { $args[0] } else { "anirdarrazi/autonomousc-ai-edge-runtime:latest" }

Write-Host "Building Linux runtime manager image $imageName ..."
docker build -f Dockerfile.service -t $imageName .
Write-Host "Built $imageName"
