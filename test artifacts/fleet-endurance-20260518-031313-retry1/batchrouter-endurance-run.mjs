import fs from "node:fs/promises";
import { performance } from "node:perf_hooks";

const baseUrl = process.env.BATCHROUTER_BASE_URL ?? "https://batchrouter.com";
const apiKey = process.env.BATCHROUTER_API_KEY ?? process.env.BATCHROUTER_SMOKE_API_KEY;
if (!apiKey) throw new Error("BATCHROUTER_API_KEY or BATCHROUTER_SMOKE_API_KEY missing");

const artifact = process.env.FLEET_ARTIFACT_ROOT;
if (!artifact) throw new Error("FLEET_ARTIFACT_ROOT missing");

const size = Number.parseInt(process.env.ENDURANCE_BATCH_SIZE ?? "1000", 10);
const runId = process.env.ENDURANCE_RUN_ID ?? `endurance_${Date.now()}`;
const pollSeconds = Number.parseInt(process.env.ENDURANCE_POLL_SECONDS ?? "15", 10);
const timeoutMs = Number.parseInt(process.env.ENDURANCE_TIMEOUT_MS ?? "7200000", 10);
const maxOutputTokens = Number.parseInt(process.env.ENDURANCE_MAX_OUTPUT_TOKENS ?? "8", 10);
const promptWords = Number.parseInt(process.env.ENDURANCE_PROMPT_WORDS ?? "3", 10);
const quoteOnly = process.env.ENDURANCE_QUOTE_ONLY === "1";
const checkResultsEndpoint = process.env.ENDURANCE_CHECK_RESULTS === "1";
const operation = process.env.ENDURANCE_OPERATION ?? "responses";
const model = process.env.ENDURANCE_MODEL ?? "gemma-4-e4b-it";
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function headers(extra = {}) {
  return { authorization: `Bearer ${apiKey}`, "content-type": "application/json", ...extra };
}

async function requestText(path, init = {}) {
  const started = performance.now();
  let response;
  let text;
  try {
    response = await fetch(`${baseUrl}${path}`, init);
    text = await response.text();
  } catch (error) {
    const failedMs = Math.round(performance.now() - started);
    return {
      status: 0,
      ok: false,
      elapsed_ms: failedMs,
      bytes: 0,
      payload: {
        transport_error: true,
        name: error instanceof Error ? error.name : "Error",
        message: error instanceof Error ? error.message : String(error),
        cause: error instanceof Error && error.cause instanceof Error ? error.cause.message : null
      }
    };
  }
  let payload;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = { parse_error: true, text: text.slice(0, 4000) };
  }
  const elapsedMs = Math.round(performance.now() - started);
  return { status: response.status, ok: response.ok, elapsed_ms: elapsedMs, bytes: Buffer.byteLength(text), payload };
}

function promptFor(index) {
  const base = `Return exactly endurance-ok-${index}.`;
  if (promptWords <= 3) return base;
  const filler = " Keep the answer short and deterministic.";
  const repeat = Math.max(0, Math.ceil((promptWords - 3) / 6));
  return `${base}${filler.repeat(repeat)}`;
}

function manifest() {
  return {
    sla_tier: "standard",
    routing_mode: "hybrid",
    privacy_tier: "standard",
    allowed_regions: ["global"],
    provider_preferences: { only: ["autonomousc"], allow_fallbacks: false },
    metadata: {
      fleet_test: true,
      endurance_test: true,
      fleet_run_id: runId,
      requested_item_count: size,
      provider_under_test: "autonomousc",
      prompt_words: promptWords,
      max_output_tokens: maxOutputTokens
    },
    items: Array.from({ length: size }, (_, i) => ({
      customer_item_id: `${runId}_${size}_${String(i).padStart(7, "0")}`,
      operation,
      model,
      input: { input: promptFor(i), max_output_tokens: maxOutputTokens, temperature: 0 }
    }))
  };
}

function batchState(payload) {
  const batch = payload?.payload?.batch ?? payload?.batch ?? payload?.payload ?? payload;
  return {
    batch,
    id: batch?.id,
    state: batch?.state,
    counts: batch?.counts ?? {},
    execution_summary: batch?.execution_summary ?? null
  };
}

const runDir = `${artifact}/batchrouter-${size}-${runId}`;
await fs.mkdir(runDir, { recursive: true });

const body = manifest();
const jsonStarted = performance.now();
const bodyJson = JSON.stringify(body);
const bodyBytes = Buffer.byteLength(bodyJson);
const jsonMs = Math.round(performance.now() - jsonStarted);
const approxInputTokens = size * Math.max(3, promptWords);
const approxMaxOutputTokens = size * maxOutputTokens;
const metadata = {
  run_id: runId,
  size,
  model,
  operation,
  prompt_words: promptWords,
  max_output_tokens: maxOutputTokens,
  approx_input_tokens: approxInputTokens,
  approx_max_output_tokens: approxMaxOutputTokens,
  approx_total_token_budget: approxInputTokens + approxMaxOutputTokens,
  request_body_bytes: bodyBytes,
  stringify_ms: jsonMs,
  quote_only: quoteOnly,
  check_results_endpoint: checkResultsEndpoint
};
await fs.writeFile(`${runDir}/metadata.json`, JSON.stringify(metadata, null, 2));
console.log(JSON.stringify({ event: "prepared", ...metadata }));

const quote = await requestText("/v1/batches/quote", { method: "POST", headers: headers(), body: bodyJson });
await fs.writeFile(`${runDir}/quote.json`, JSON.stringify(quote, null, 2));
if (!quote.ok) {
  console.log(JSON.stringify({ event: "quote_failed", status: quote.status, elapsed_ms: quote.elapsed_ms, bytes: quote.bytes }));
  process.exitCode = 1;
  process.exit();
}
const quotePayload = quote.payload?.payload ?? quote.payload;
const workUnits = Array.isArray(quotePayload?.work_units) ? quotePayload.work_units : [];
await fs.writeFile(`${runDir}/quote-summary.json`, JSON.stringify({
  quote_id: quotePayload?.quote_id,
  work_unit_count: workUnits.length,
  work_unit_item_counts: workUnits.map((unit) => unit.item_count),
  selected_providers: [...new Set(workUnits.map((unit) => unit.selected_provider).filter(Boolean))]
}, null, 2));
console.log(JSON.stringify({
  event: "quoted",
  status: quote.status,
  elapsed_ms: quote.elapsed_ms,
  quote_id: quotePayload?.quote_id,
  work_unit_count: workUnits.length,
  first_work_unit_items: workUnits[0]?.item_count ?? null,
  last_work_unit_items: workUnits.at(-1)?.item_count ?? null
}));

if (quoteOnly) {
  process.exit();
}

const create = await requestText("/v1/batches", {
  method: "POST",
  headers: headers({ "idempotency-key": `${runId}:${size}:create` }),
  body: JSON.stringify({ ...body, quote_id: quotePayload.quote_id })
});
await fs.writeFile(`${runDir}/created.json`, JSON.stringify(create, null, 2));
if (!create.ok) {
  console.log(JSON.stringify({ event: "create_failed", status: create.status, elapsed_ms: create.elapsed_ms, bytes: create.bytes }));
  process.exitCode = 1;
  process.exit();
}
const created = create.payload?.payload ?? create.payload;
const batchId = created?.batch?.id ?? created?.id;
if (!batchId) throw new Error("missing batch id");
console.log(JSON.stringify({ event: "created", batch_id: batchId, elapsed_ms: create.elapsed_ms, bytes: create.bytes }));

let latest = null;
let polls = 0;
const started = Date.now();
while (Date.now() - started <= timeoutMs) {
  await sleep(pollSeconds * 1000);
  polls += 1;
  const status = await requestText(`/v1/batches/${batchId}`, { headers: headers() });
  await fs.writeFile(`${runDir}/latest-status.json`, JSON.stringify(status, null, 2));
  if (!status.ok) {
    console.log(JSON.stringify({ event: "poll_error", status: status.status, elapsed_ms: status.elapsed_ms, polls }));
    continue;
  }
  latest = batchState(status.payload);
  console.log(JSON.stringify({
    event: "poll",
    size,
    batch_id: batchId,
    state: latest.state,
    counts: latest.counts,
    polls,
    elapsed_seconds: Math.round((Date.now() - started) / 1000),
    response_ms: status.elapsed_ms,
    response_bytes: status.bytes
  }));
  if (["completed", "failed", "canceled", "expired"].includes(latest.state)) break;
}

if (!latest) throw new Error("no successful status response");
await fs.writeFile(`${runDir}/final-status.json`, JSON.stringify(latest, null, 2));

let resultsProbe = null;
if (checkResultsEndpoint) {
  resultsProbe = await requestText(`/v1/batches/${batchId}/results`, { headers: headers() });
  await fs.writeFile(`${runDir}/results-probe.json`, JSON.stringify(resultsProbe, null, 2));
}

const receipt = await requestText(`/v1/batches/${batchId}/billing-receipt`, { headers: headers() });
await fs.writeFile(`${runDir}/billing-receipt.json`, JSON.stringify(receipt, null, 2));

const summary = {
  ...metadata,
  batch_id: batchId,
  quote_id: quotePayload.quote_id,
  state: latest.state,
  counts: latest.counts,
  elapsed_seconds: Math.round((Date.now() - started) / 1000),
  polls,
  work_unit_count: workUnits.length,
  work_unit_item_counts: workUnits.map((unit) => unit.item_count),
  selected_providers: [...new Set(workUnits.map((unit) => unit.selected_provider).filter(Boolean))],
  execution_summary: latest.execution_summary,
  billing_receipt_ok: receipt.ok,
  results_probe_status: resultsProbe?.status ?? null,
  results_probe_elapsed_ms: resultsProbe?.elapsed_ms ?? null,
  results_probe_bytes: resultsProbe?.bytes ?? null
};
await fs.writeFile(`${runDir}/summary.json`, JSON.stringify(summary, null, 2));
console.log(JSON.stringify({ event: "summary", ...summary }));
if (latest.state !== "completed" || Number(latest.counts?.failed ?? 0) !== 0) process.exitCode = 1;
