Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$imageName = if ($args.Length -ge 1 -and $args[0]) { $args[0] } else { "anirdarrazi/autonomousc-ai-edge-runtime:latest" }

python .\scripts\generate_model_artifacts_manifest.py --check

Write-Host "Building unified Linux runtime image $imageName ..."
docker build -f Dockerfile.service -t $imageName .
Write-Host "Built $imageName"

Write-Host "Pushing $imageName ..."
docker push $imageName
docker buildx imagetools inspect $imageName
