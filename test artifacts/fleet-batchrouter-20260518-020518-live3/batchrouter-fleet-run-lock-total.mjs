import fs from "node:fs/promises";

const apiKey = process.env.BATCHROUTER_API_KEY;
if (!apiKey) throw new Error("BATCHROUTER_API_KEY missing");

const artifact = process.env.FLEET_ARTIFACT_ROOT;
if (!artifact) throw new Error("FLEET_ARTIFACT_ROOT missing");

const size = Number.parseInt(process.env.FLEET_BATCH_SIZE ?? "1200", 10);
const runId = process.env.FLEET_RUN_ID ?? `fleet_${Date.now()}`;
const pollSeconds = Number.parseInt(process.env.FLEET_POLL_SECONDS ?? "10", 10);
const timeoutMs = Number.parseInt(process.env.FLEET_TIMEOUT_MS ?? "2400000", 10);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function headers(extra = {}) {
  return { authorization: `Bearer ${apiKey}`, "content-type": "application/json", ...extra };
}

async function requestJson(path, init = {}) {
  const response = await fetch(`https://batchrouter.com${path}`, init);
  const text = await response.text();
  let payload;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = { parse_error: true, text: text.slice(0, 2000) };
  }
  return { status: response.status, ok: response.ok, payload };
}

function manifest() {
  return {
    sla_tier: "standard",
    routing_mode: "hybrid",
    privacy_tier: "standard",
    allowed_regions: ["global"],
    price_protection_policy: "lock_total",
    provider_preferences: { only: ["autonomousc"], allow_fallbacks: false },
    metadata: {
      fleet_test: true,
      fleet_run_id: runId,
      requested_item_count: size,
      provider_under_test: "autonomousc",
      quote_lock_policy: "lock_total"
    },
    items: Array.from({ length: size }, (_, i) => ({
      customer_item_id: `${runId}_${size}_${String(i).padStart(5, "0")}`,
      operation: "responses",
      model: "gemma-4-e4b-it",
      input: {
        input: `Reply with exactly: fleet-ok-${i}`,
        max_output_tokens: 8,
        temperature: 0
      }
    }))
  };
}

const body = manifest();
const quote = await requestJson("/v1/batches/quote", {
  method: "POST",
  headers: headers(),
  body: JSON.stringify(body)
});
await fs.writeFile(`${artifact}/batchrouter-${size}-quote.json`, JSON.stringify(quote, null, 2));
if (!quote.ok) throw new Error(`quote failed HTTP ${quote.status}`);

const quotePayload = quote.payload?.payload ?? quote.payload;
const create = await requestJson("/v1/batches", {
  method: "POST",
  headers: headers({ "idempotency-key": `${runId}:${size}:${Date.now()}` }),
  body: JSON.stringify({ ...body, quote_id: quotePayload.quote_id })
});
await fs.writeFile(`${artifact}/batchrouter-${size}-created.json`, JSON.stringify(create, null, 2));
if (!create.ok) throw new Error(`create failed HTTP ${create.status}`);

const createdPayload = create.payload?.payload ?? create.payload;
const batchId = createdPayload?.batch?.id ?? createdPayload?.id;
if (!batchId) throw new Error("missing batch id");

let latest = null;
let polls = 0;
const started = Date.now();
while (Date.now() - started <= timeoutMs) {
  await sleep(pollSeconds * 1000);
  polls += 1;
  const results = await requestJson(`/v1/batches/${batchId}/results`, { headers: headers() });
  await fs.writeFile(`${artifact}/batchrouter-${size}-latest-results.json`, JSON.stringify(results, null, 2));
  if (!results.ok) continue;
  latest = results.payload?.payload ?? results.payload;
  const counts = latest?.counts ?? {};
  console.log(
    JSON.stringify({
      event: "poll",
      size,
      batch_id: batchId,
      state: latest?.state,
      counts,
      polls,
      elapsed_seconds: Math.round((Date.now() - started) / 1000)
    })
  );
  if (["completed", "failed", "canceled", "expired"].includes(latest?.state)) break;
}

if (!latest) throw new Error("no successful results response");
await fs.writeFile(`${artifact}/batchrouter-${size}-final-results.json`, JSON.stringify(latest, null, 2));

const receipt = await requestJson(`/v1/batches/${batchId}/billing-receipt`, { headers: headers() });
await fs.writeFile(`${artifact}/batchrouter-${size}-billing-receipt.json`, JSON.stringify(receipt, null, 2));

const counts = latest?.counts ?? {};
const workUnits = Array.isArray(quotePayload?.work_units) ? quotePayload.work_units : [];
const summary = {
  run_id: runId,
  size,
  batch_id: batchId,
  state: latest?.state,
  counts,
  elapsed_seconds: Math.round((Date.now() - started) / 1000),
  polls,
  quote_id: quotePayload.quote_id,
  quote_work_unit_item_counts: workUnits.map((u) => u.item_count),
  selected_providers: [...new Set(workUnits.map((u) => u.selected_provider).filter(Boolean))],
  provider_breakdown: latest?.execution_summary?.final_provider_breakdown ?? null,
  output_preview_rows: Array.isArray(latest?.output_preview_rows) ? latest.output_preview_rows.length : null,
  receipt_ok: receipt.ok
};
await fs.writeFile(`${artifact}/batchrouter-${size}-summary.json`, JSON.stringify(summary, null, 2));
console.log(JSON.stringify({ event: "summary", ...summary }, null, 2));

if (latest?.state !== "completed" || Number(counts.failed ?? 0) !== 0) {
  process.exitCode = 1;
}
