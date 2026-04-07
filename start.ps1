Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
docker compose up -d node-agent vllm vector
Write-Host "Runtime started. Follow logs with: docker compose logs -f node-agent"
