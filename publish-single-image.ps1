Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$imageName = if ($args.Length -ge 1 -and $args[0]) { $args[0] } else { "anirdarrazi/autonomousc-ai-edge-runtime:single-cuda-latest" }

.\build-single-image.ps1 $imageName

Write-Host "Pushing $imageName ..."
docker push $imageName
Write-Host "Pushed $imageName"
docker buildx imagetools inspect $imageName
