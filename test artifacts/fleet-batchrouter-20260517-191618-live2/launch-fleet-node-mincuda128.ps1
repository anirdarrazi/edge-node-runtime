param(
  [Parameter(Mandatory=$true)][int]$Index,
  [Parameter(Mandatory=$true)][string]$ArtifactRoot
)
$ErrorActionPreference = 'Stop'
function Sha256Hex([string]$Value) {
  $sha = [System.Security.Cryptography.SHA256]::Create()
  try { return -join ($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($Value)) | ForEach-Object { $_.ToString('x2') }) }
  finally { $sha.Dispose() }
}
function SqlQuote([string]$Value) { return "'" + $Value.Replace("'", "''") + "'" }
$repoRoot = 'C:\Users\Anir\Documents\AI eco future\AUTONOMOUSc Edge Node Network'
$runtimeRoot = Join-Path $repoRoot 'edge-node-runtime'
$controlRoot = Join-Path $repoRoot 'edge-control'
$nodeId = 'node_fleet_' + ([guid]::NewGuid().ToString('N'))
$nodeKey = ([guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N'))
$nodeKeyHash = Sha256Hex $nodeKey
$now = (Get-Date).ToUniversalTime().ToString('o')
$nodeDir = Join-Path $ArtifactRoot ("node-$Index")
New-Item -ItemType Directory -Force -Path $nodeDir | Out-Null
Set-Content -Path (Join-Path $nodeDir 'node-id.txt') -Value $nodeId -Encoding UTF8
$capabilities = [ordered]@{
  supported_models = @('google/gemma-4-E4B-it')
  operations = @('responses')
  gpu_name = 'RTX 5060 Ti'
  gpu_memory_gb = 16
  max_context_tokens = 32768
  target_batch_items = 100
  max_batch_items = 250
  target_batch_tokens = 12000
  max_batch_tokens = 32768
  max_concurrent_chunks = 4
  max_concurrent_batches = 4
  recommended_batch_items = 100
  max_concurrent_assignments = 8
  max_local_queue_assignments = 64
  max_pull_bundle_assignments = 64
  available_queue_items = 5000
  available_queue_tokens = 262144
  max_queued_items = 5000
  capacity_status = 'active'
  batchrouter_capacity_tier = 'edge'
  heartbeat_ttl_seconds = 120
  thermal_headroom = 0.95
  heat_demand = 'none'
  target_gpu_utilization_pct = 100
  min_gpu_memory_headroom_pct = 5
} | ConvertTo-Json -Compress
$runtime = [ordered]@{
  agent_version = '0.1.0'
  runtime_profile = 'rtx_5060_ti_16gb_gemma4_e4b_it'
  runtime_profile_label = 'RTX 5060 Ti 16GB Gemma 4 E4B IT'
  inference_engine = 'vllm'
  deployment_target = 'vast_ai'
  model_format = 'safetensors'
  runtime_image = 'anirdarrazi/autonomousc-ai-edge-runtime:single-cuda-latest'
  docker_image = 'anirdarrazi/autonomousc-ai-edge-runtime:single-cuda-latest'
  readiness_path = '/v1/models'
  supported_apis = @('responses')
  capacity_class = 'elastic_burst'
  routing_lane = 'elastic_exact_vast'
  routing_lane_label = 'Elastic exact Vast.ai'
  max_privacy_tier = 'standard'
  exact_model_guarantee = $true
  quantized_output_disclosure_required = $false
  quality_class = 'exact_audited'
  exactness_class = 'exact_audited'
  temporary_node = $true
  burst_provider = 'vast_ai'
  current_model = 'google/gemma-4-E4B-it'
  effective_context_tokens = 32768
} | ConvertTo-Json -Compress
$reputation = @{ success_rate = 1; score = 0.8 } | ConvertTo-Json -Compress
$sql = "INSERT INTO nodes (id, org_id, label, region, node_key_hash, status, approval_status, trust_tier, node_trust_class, node_trust_source, restricted_capable, attestation_status, attestation_provider, capabilities_json, runtime_json, reputation_json, queue_depth, active_assignments, last_heartbeat_at, last_bootstrap_at, last_auth_failure_at, last_auth_failure_reason, last_canary_verification_status, last_canary_verified_at, created_at, updated_at) VALUES (" +
  (SqlQuote $nodeId) + ", " +
  (SqlQuote 'org_2bae74f86c454cfaafafc1ea2f52a1fd') + ", " +
  (SqlQuote ("BatchRouter fleet RTX 5060 Ti #$Index")) + ", " +
  (SqlQuote 'eu-se-1') + ", " +
  (SqlQuote $nodeKeyHash) + ", " +
  (SqlQuote 'pending_attestation') + ", " +
  (SqlQuote 'approved') + ", " +
  (SqlQuote 'standard') + ", " +
  (SqlQuote 'untrusted') + ", " +
  (SqlQuote 'community') + ", 0, " +
  (SqlQuote 'missing') + ", " +
  (SqlQuote 'unknown') + ", " +
  (SqlQuote $capabilities) + ", " +
  (SqlQuote $runtime) + ", " +
  (SqlQuote $reputation) + ", 0, 0, NULL, " +
  (SqlQuote $now) + ", NULL, NULL, NULL, NULL, " +
  (SqlQuote $now) + ", " + (SqlQuote $now) + "); SELECT id, status, approval_status, attestation_status, created_at FROM nodes WHERE id = " + (SqlQuote $nodeId) + ";"
$sqlPath = Join-Path $nodeDir 'create-node.sql'
Set-Content -Path $sqlPath -Value $sql -Encoding UTF8
$env:PATH = "$repoRoot\node_modules\.bin;C:\Users\Anir\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin;$env:APPDATA\npm;$env:PATH"
Push-Location $controlRoot
try {
  $oldPreference = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  & '..\node_modules\.bin\wrangler.CMD' d1 execute autonomousc-edge-network-db --remote --file $sqlPath > (Join-Path $nodeDir 'create-node-output.txt') 2>&1
  $wranglerExit = $LASTEXITCODE
  $ErrorActionPreference = $oldPreference
  if ($wranglerExit -ne 0) { throw "wrangler D1 insert failed with exit code $wranglerExit" }
} finally {
  Pop-Location
}
$redacted = [ordered]@{ node_id = $nodeId; node_key_hash = $nodeKeyHash; created_at = $now; capacity = @{ target_batch_items = 100; max_batch_items = 250; max_concurrent_chunks = 4; available_queue_items = 5000; available_queue_tokens = 262144 } }
$redacted | ConvertTo-Json -Depth 10 | Set-Content -Path (Join-Path $nodeDir 'node-create-metadata.redacted.json') -Encoding UTF8
Push-Location $runtimeRoot
try {
  $env:PYTHONPATH = 'src'
  $oldPreference = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  python -m node_agent.vast_smoke `
    --durable-node `
    --edge-control-url 'https://edge.autonomousc.com' `
    --node-id $nodeId `
    --node-key $nodeKey `
    --node-region 'eu-se-1' `
    --runtime-profile 'rtx_5060_ti_16gb_gemma4_e4b_it' `
    --model 'google/gemma-4-E4B-it' `
    --api responses `
    --image 'anirdarrazi/autonomousc-ai-edge-runtime:single-cuda-latest' `
    --max-context-tokens 32768 `
    --max-batch-tokens 32768 `
    --target-batch-items 100 `
    --max-batch-items 250 `
    --target-batch-tokens 12000 `
    --max-concurrent-chunks 4 `
    --available-queue-items 5000 `
    --available-queue-tokens 262144 `
    --max-queued-items 5000 `
    --max-concurrent-assignments 8 `
    --max-local-queue-assignments 64 `
    --pull-bundle-size 64 `
    --max-price 0.25 `
    --min-cuda-max-good 12.8 `
    --min-reliability 0.90 `
    --min-inet-down-mbps 100 `
    --offer-limit 120 `
    --launch-timeout-seconds 900 `
    --readiness-timeout-seconds 2400 `
    --poll-interval-seconds 10 `
    --benchmark-requests 1 `
    --benchmark-concurrency 1 `
    --json-indent 2 > (Join-Path $nodeDir 'vast-launch-output.json') 2> (Join-Path $nodeDir 'vast-launch-error.txt')
  $exitCode = $LASTEXITCODE
  $ErrorActionPreference = $oldPreference
} catch {
  $_ | Out-String | Set-Content -Path (Join-Path $nodeDir 'launch-exception.txt') -Encoding UTF8
  $exitCode = 1
} finally {
  Pop-Location
}
Set-Content -Path (Join-Path $nodeDir 'exit-code.txt') -Value ([string]$exitCode) -Encoding UTF8
Set-Content -Path (Join-Path $nodeDir 'completed.txt') -Value ((Get-Date).ToUniversalTime().ToString('o')) -Encoding UTF8
exit $exitCode

