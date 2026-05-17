import fs from "node:fs/promises";
const apiKey = process.env.BATCHROUTER_API_KEY;
if (!apiKey) throw new Error("BATCHROUTER_API_KEY missing");
const artifact = process.env.FLEET_ARTIFACT_ROOT;
const size = Number.parseInt(process.env.FLEET_QUOTE_SIZE ?? "1200", 10);
const items = Array.from({ length: size }, (_, i) => ({
  customer_item_id: `fleet_quote_${size}_${i}`,
  operation: "responses",
  model: "gemma-4-e4b-it",
  input: { input: `Reply with exactly: fleet-ok-${i}`, max_output_tokens: 8, temperature: 0 }
}));
const manifest = {
  sla_tier: "standard",
  routing_mode: "hybrid",
  privacy_tier: "standard",
  allowed_regions: ["global"],
  provider_preferences: { only: ["autonomousc"], allow_fallbacks: false },
  metadata: { fleet_test: true, requested_item_count: size },
  items
};
const response = await fetch("https://batchrouter.com/v1/batches/quote", {
  method: "POST",
  headers: { authorization: `Bearer ${apiKey}`, "content-type": "application/json" },
  body: JSON.stringify(manifest)
});
const text = await response.text();
let payload;
try { payload = text ? JSON.parse(text) : null; } catch { payload = { parse_error: true, text }; }
await fs.writeFile(`${artifact}/batchrouter-${size}-quote-probe.json`, JSON.stringify({ status: response.status, ok: response.ok, payload }, null, 2));
const p = payload?.payload ?? payload;
const workUnits = Array.isArray(p?.work_units) ? p.work_units : [];
const summary = {
  status: response.status,
  ok: response.ok,
  quote_id: p?.quote_id ?? null,
  work_units: workUnits.length,
  item_counts: workUnits.map((u) => u.item_count),
  selected_providers: [...new Set(workUnits.map((u) => u.selected_provider).filter(Boolean))],
  total_usd: p?.pricing_estimate?.total_usd ?? null,
  error: p?.error ?? null
};
console.log(JSON.stringify(summary, null, 2));
if (!response.ok) process.exitCode = 1;
