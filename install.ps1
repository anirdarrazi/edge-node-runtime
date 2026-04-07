Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  throw "Docker is required to install the edge node runtime."
}

docker compose version | Out-Null

if (-not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
  Write-Host "Created .env from .env.example"
}

Write-Host "Starting the local model runtime..."
docker compose up -d vllm

Write-Host "Launching the interactive node claim flow..."
docker compose run --rm node-agent-bootstrap

Write-Host "Starting the long-running runtime services..."
docker compose up -d node-agent vector

Write-Host "Installation complete. Follow runtime logs with: docker compose logs -f node-agent"
