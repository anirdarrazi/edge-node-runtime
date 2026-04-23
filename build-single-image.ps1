Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$imageName = if ($args.Length -ge 1 -and $args[0]) { $args[0] } else { "anirdarrazi/autonomousc-ai-edge-runtime:single-cuda-latest" }
$buildArgs = @()

if (Test-Path env:PRELOAD_HF_MODELS) {
    $buildArgs += "--build-arg"
    $buildArgs += "PRELOAD_HF_MODELS=$($env:PRELOAD_HF_MODELS)"
}

python .\scripts\generate_model_artifacts_manifest.py --check

Write-Host "Building Linux NVIDIA single-container image $imageName ..."
docker build --platform linux/amd64 @buildArgs -f Dockerfile.single -t $imageName .
Write-Host "Built $imageName"
