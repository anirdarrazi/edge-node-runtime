Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$imageName = if ($args.Length -ge 1 -and $args[0]) { $args[0] } else { "anirdarrazi/autonomousc-ai-edge-runtime:latest" }
$buildArgs = @()

if (Test-Path env:PRELOAD_HF_MODELS) {
    $buildArgs += "--build-arg"
    $buildArgs += "PRELOAD_HF_MODELS=$($env:PRELOAD_HF_MODELS)"
}

python .\scripts\generate_model_artifacts_manifest.py --check

Write-Host "Building unified Linux runtime image $imageName ..."
docker build @buildArgs -f Dockerfile.service -t $imageName .
Write-Host "Built $imageName"
