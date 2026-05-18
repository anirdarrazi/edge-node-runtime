param(
  [Parameter(Mandatory=$true)][int]$Index,
  [Parameter(Mandatory=$true)][string]$ArtifactRoot,
  [Parameter(Mandatory=$true)][int]$PreferredOfferId
)

$ErrorActionPreference = 'Stop'

function Write-JsonFile([string]$Path, [object]$Value) {
  $Value | ConvertTo-Json -Depth 30 | Set-Content -Path $Path -Encoding UTF8
}

$repoRoot = 'C:\Users\Anir\Documents\AI eco future\AUTONOMOUSc Edge Node Network'
$runtimeRoot = Join-Path $repoRoot 'edge-node-runtime'
$edgeControlUrl = $env:EDGE_CONTROL_URL
if (-not $edgeControlUrl) { $edgeControlUrl = 'https://edge.autonomousc.com' }
$edgeControlUrl = $edgeControlUrl.TrimEnd('/')
$operatorToken = $env:SMOKE_OPERATOR_TOKEN
if (-not $operatorToken) { throw 'SMOKE_OPERATOR_TOKEN is required for live fleet node enrollment.' }

$nodeDir = Join-Path $ArtifactRoot ("node-$Index")
New-Item -ItemType Directory -Force -Path $nodeDir | Out-Null

$capabilities = [ordered]@{
  supported_models = @('google/gemma-4-E4B-it')
  operations = @('responses')
  gpu_name = 'RTX 5060 Ti'
  gpu_memory_gb = 16
  max_context_tokens = 32768
  max_batch_tokens = 32768
  max_concurrent_assignments = 8
  max_concurrent_assignments_embeddings = 1
  max_pull_bundle_assignments = 64
  thermal_headroom = 0.95
  heat_demand = 'none'
  power_watts = 165
  estimated_heat_output_watts = 165
}
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
}
$enrollBody = [ordered]@{
  label = "BatchRouter fleet RTX 5060 Ti #$Index"
  region = 'eu-se-1'
  trust_tier = 'standard'
  restricted_capable = $false
  capabilities = $capabilities
  runtime = $runtime
  operator_token = $operatorToken
}

Write-JsonFile (Join-Path $nodeDir 'enroll-request.redacted.json') (@{
  label = $enrollBody.label
  region = $enrollBody.region
  trust_tier = $enrollBody.trust_tier
  restricted_capable = $enrollBody.restricted_capable
  capabilities = $capabilities
  runtime = $runtime
  operator_token_present = $true
  preferred_offer_id = $PreferredOfferId
})

$enrollResponse = Invoke-RestMethod `
  -Method Post `
  -Uri "$edgeControlUrl/nodes/enroll" `
  -ContentType 'application/json' `
  -Body ($enrollBody | ConvertTo-Json -Depth 30)

Write-JsonFile (Join-Path $nodeDir 'enroll-response.redacted.json') (@{
  node_id = $enrollResponse.node_id
  status = $enrollResponse.status
  approved = $enrollResponse.approved
  node_key_present = [bool]$enrollResponse.node_key
})
Set-Content -Path (Join-Path $nodeDir 'node-id.txt') -Value ([string]$enrollResponse.node_id) -Encoding UTF8

if (-not $enrollResponse.approved) {
  $headers = @{ Authorization = "Bearer $operatorToken" }
  $approveResponse = Invoke-RestMethod `
    -Method Post `
    -Uri "$edgeControlUrl/admin/nodes/$($enrollResponse.node_id)/approve" `
    -Headers $headers `
    -ContentType 'application/json' `
    -Body '{}'
  Write-JsonFile (Join-Path $nodeDir 'approve-response.redacted.json') (@{
    node_id = $approveResponse.node.id
    status = $approveResponse.node.status
    approval_status = $approveResponse.node.approval_status
  })
}

Push-Location $runtimeRoot
try {
  $env:PYTHONPATH = 'src'
  python -m node_agent.vast_smoke `
    --durable-node `
    --edge-control-url $edgeControlUrl `
    --node-id $enrollResponse.node_id `
    --node-key $enrollResponse.node_key `
    --node-region 'eu-se-1' `
    --runtime-profile 'rtx_5060_ti_16gb_gemma4_e4b_it' `
    --model 'google/gemma-4-E4B-it' `
    --api responses `
    --image 'anirdarrazi/autonomousc-ai-edge-runtime:single-cuda-latest' `
    --preferred-offer-id $PreferredOfferId `
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
} finally {
  Pop-Location
}
Set-Content -Path (Join-Path $nodeDir 'exit-code.txt') -Value ([string]$exitCode) -Encoding UTF8
Set-Content -Path (Join-Path $nodeDir 'completed.txt') -Value ((Get-Date).ToUniversalTime().ToString('o')) -Encoding UTF8
exit $exitCode
