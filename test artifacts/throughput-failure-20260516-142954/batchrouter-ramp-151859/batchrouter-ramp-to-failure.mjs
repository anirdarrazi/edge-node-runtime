#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";

const baseUrl = (process.env.BATCHROUTER_BASE_URL ?? "https://batchrouter.com").replace(/\/+$/, "");
const apiKey = process.env.BATCHROUTER_SMOKE_API_KEY;
const artifactRoot = process.env.ENDURANCE_ARTIFACT_ROOT;
const runId = process.env.ENDURANCE_RUN_ID ?? `endurance_${Date.now()}`;
const pollSeconds = Number.parseInt(process.env.ENDURANCE_POLL_SECONDS ?? "10", 10);
const timeoutMs = Number.parseInt(process.env.ENDURANCE_TIMEOUT_MS ?? "1500000", 10);
const maxSpendUsd = Number.parseFloat(process.env.ENDURANCE_MAX_SPEND_USD ?? "0.20");

if (!apiKey) throw new Error("BATCHROUTER_SMOKE_API_KEY is required.");
if (!artifactRoot) throw new Error("ENDURANCE_ARTIFACT_ROOT is required.");

const quoteSizes = [250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 5250];
const submitSizes = [500, 1000, 1500, 2000, 3000, 4000, 5000];
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function authHeaders(extra = {}) {
  return { authorization: `Bearer ${apiKey}`, "content-type": "application/json", ...extra };
}

async function writeJson(name, value) {
  await fs.writeFile(path.join(artifactRoot, name), JSON.stringify(value, null, 2));
}

async function appendEvent(event) {
  await fs.appendFile(path.join(artifactRoot, "batchrouter-endurance-output.jsonl"), `${JSON.stringify({ ...event, at: new Date().toISOString() })}\n`);
}

async function requestJson(pathname, init = {}) {
  const response = await fetch(`${baseUrl}${pathname}`, init);
  const text = await response.text();
  let payload = null;
  try { payload = text ? JSON.parse(text) : null; } catch { payload = { parse_error: true, text: text.slice(0, 1000) }; }
  return { status: response.status, ok: response.ok, payload, text };
}

function manifest(itemCount) {
  return {
    sla_tier: "standard",
    routing_mode: "hybrid",
    privacy_tier: "standard",
    allowed_regions: ["global"],
    provider_preferences: { only: ["autonomousc"], allow_fallbacks: false },
    metadata: {
      endurance_test: true,
      endurance_run_id: runId,
      requested_item_count: itemCount,
      provider_under_test: "autonomousc"
    },
    items: Array.from({ length: itemCount }, (_, index) => ({
      customer_item_id: `${runId}_${itemCount}_${String(index).padStart(4, "0")}`,
      operation: "responses",
      model: "gemma-4-e4b-it",
      input: {
        input: `Reply with exactly: endurance-ok-${index}`,
        max_output_tokens: 8,
        temperature: 0
      }
    }))
  };
}

function unwrap(response) {
  return response.payload?.payload ?? response.payload;
}

function quotedTotal(quotePayload) {
  const estimate = quotePayload?.pricing_estimate;
  const direct = estimate?.total_usd ?? estimate?.total ?? estimate?.customer_charge_usd;
  if (direct !== undefined && direct !== null && Number.isFinite(Number(direct))) return Number(direct);
  const workUnits = Array.isArray(quotePayload?.work_units) ? quotePayload.work_units : [];
  const summed = workUnits.reduce((sum, unit) => {
    const total = unit?.execution_lane?.estimated_cost?.total ?? unit?.estimated_cost?.total;
    return sum + (Number.isFinite(Number(total)) ? Number(total) : 0);
  }, 0);
  return Number(summed.toFixed(6));
}

function quoteSummary(size, response) {
  const payload = unwrap(response);
  const error = payload?.error;
  const workUnits = Array.isArray(payload?.work_units) ? payload.work_units : [];
  return {
    size,
    http_status: response.status,
    ok: response.ok,
    quote_id: payload?.quote_id ?? null,
    total_usd: response.ok ? quotedTotal(payload) : null,
    work_unit_item_counts: workUnits.map((unit) => unit.item_count),
    selected_providers: [...new Set(workUnits.map((unit) => unit.selected_provider).filter(Boolean))],
    route_candidate_providers: [...new Set(workUnits.flatMap((unit) => (Array.isArray(unit.route_candidates) ? unit.route_candidates.map((candidate) => candidate.provider) : [])).filter(Boolean))],
    error_code: error?.code ?? null,
    error_message: error?.message ?? null,
    error_details: error?.details ?? null
  };
}

async function quote(size) {
  const body = manifest(size);
  const response = await requestJson("/v1/batches/quote", {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify(body)
  });
  await writeJson(`batchrouter-${size}-quote.json`, response);
  const summary = quoteSummary(size, response);
  await appendEvent({ event: "quote", ...summary });
  return { body, response, summary };
}

async function createBatch(size, body, quoteId) {
  const created = await requestJson("/v1/batches", {
    method: "POST",
    headers: authHeaders({ "idempotency-key": `${runId}:${size}:${Date.now()}` }),
    body: JSON.stringify({ ...body, ...(quoteId ? { quote_id: quoteId } : {}) })
  });
  await writeJson(`batchrouter-${size}-created.json`, created);
  if (!created.ok) {
    await appendEvent({ event: "create_failed", size, http_status: created.status, payload: unwrap(created) });
    throw new Error(`Create failed for ${size}: HTTP ${created.status}`);
  }
  const payload = unwrap(created);
  const batchId = payload?.batch?.id ?? payload?.id;
  if (!batchId) throw new Error(`Create response for ${size} did not include a batch id.`);
  await appendEvent({ event: "created", size, batch_id: batchId, state: payload?.batch?.state ?? null, work_order_id: payload?.work_order?.id ?? null });
  return { batchId, created: payload };
}

async function pollResults(size, batchId) {
  const started = Date.now();
  let latest = null;
  let polls = 0;
  while (Date.now() - started <= timeoutMs) {
    await sleep(pollSeconds * 1000);
    polls += 1;
    const response = await requestJson(`/v1/batches/${batchId}/results`, { headers: authHeaders() });
    await writeJson(`batchrouter-${size}-latest-results.json`, response);
    if (!response.ok) {
      await appendEvent({ event: "poll_error", size, batch_id: batchId, http_status: response.status, payload: unwrap(response) });
      continue;
    }
    latest = unwrap(response);
    const counts = latest?.counts ?? {};
    await appendEvent({ event: "poll", size, batch_id: batchId, state: latest?.state, counts, polls });
    if (["completed", "failed", "canceled", "expired"].includes(latest?.state)) break;
  }
  if (!latest) throw new Error(`No results payload received for ${batchId}.`);
  await writeJson(`batchrouter-${size}-final-results.json`, latest);
  return { latest, elapsed_seconds: Math.round((Date.now() - started) / 1000), polls };
}

async function fetchReceipt(size, batchId) {
  const response = await requestJson(`/v1/batches/${batchId}/billing-receipt`, { headers: authHeaders() });
  await writeJson(`batchrouter-${size}-billing-receipt.json`, response);
  return response.ok ? unwrap(response) : null;
}

function providerBreakdown(latest) {
  return latest?.execution_summary?.final_provider_breakdown ?? {};
}

async function main() {
  await appendEvent({ event: "start", run_id: runId, base_url: baseUrl, quote_sizes: quoteSizes, submit_sizes: submitSizes });
  const quotes = new Map();
  for (const size of quoteSizes) {
    const result = await quote(size);
    quotes.set(size, result);
  }

  let spentQuoted = 0;
  const batches = [];
  for (const size of submitSizes) {
    const quoted = quotes.get(size);
    if (!quoted?.response.ok) {
      batches.push({ size, skipped: true, reason: "quote_failed", quote_error_code: quoted?.summary?.error_code ?? null });
      continue;
    }
    const quoteSpend = quoted.summary.total_usd ?? 0;
    if (spentQuoted + quoteSpend > maxSpendUsd) {
      batches.push({ size, skipped: true, reason: "spend_guard", quote_total_usd: quoteSpend });
      continue;
    }
    spentQuoted += quoteSpend;
    const quotePayload = unwrap(quoted.response);
    const { batchId } = await createBatch(size, quoted.body, quotePayload.quote_id);
    const { latest, elapsed_seconds, polls } = await pollResults(size, batchId);
    const receipt = await fetchReceipt(size, batchId);
    const counts = latest?.counts ?? {};
    const completed = Number(counts.completed ?? 0);
    const failed = Number(counts.failed ?? 0);
    const total = Number(counts.total ?? size);
    batches.push({
      size,
      batch_id: batchId,
      state: latest?.state ?? null,
      elapsed_seconds,
      polls,
      counts,
      provider_breakdown: providerBreakdown(latest),
      preview_rows: Array.isArray(latest?.output_preview_rows) ? latest.output_preview_rows.length : 0,
      output_file_present: Boolean(latest?.output_file),
      quote_total_usd: quoteSpend,
      billing_receipt_present: Boolean(receipt),
      success_rate: total > 0 ? Number((completed / total).toFixed(4)) : null,
      failed_items: failed
    });
    await appendEvent({ event: "batch_done", size, batch_id: batchId, state: latest?.state, counts, elapsed_seconds });
  }

  const summary = {
    run_id: runId,
    base_url: baseUrl,
    quoted_at: new Date().toISOString(),
    quote_summaries: [...quotes.values()].map((entry) => entry.summary),
    batches,
    spent_quoted_usd: Number(spentQuoted.toFixed(6)),
    passed: batches.some((entry) => entry.state === "completed") && batches.filter((entry) => entry.state).every((entry) => entry.state === "completed" && entry.failed_items === 0)
  };
  await writeJson("batchrouter-endurance-summary.json", summary);
  await appendEvent({ event: "complete", passed: summary.passed, spent_quoted_usd: summary.spent_quoted_usd });
  console.log(JSON.stringify(summary, null, 2));
  if (!summary.passed) process.exitCode = 1;
}

main().catch(async (error) => {
  await appendEvent({ event: "fatal", message: error instanceof Error ? error.message : String(error) }).catch(() => undefined);
  console.error(error instanceof Error ? error.stack : error);
  process.exit(1);
});

