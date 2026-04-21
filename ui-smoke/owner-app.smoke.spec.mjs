import http from "node:http";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test } from "@playwright/test";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ownerAppHtml = readFileSync(
  path.resolve(__dirname, "../src/node_agent/service_ui.html"),
  "utf8",
);

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function buildSetupChecks() {
  return [
    {
      key: "docker",
      label: "Docker Desktop",
      status: "ok",
      summary: "Docker Desktop is running.",
      detail: "Compose and the Docker daemon are reachable from the local service.",
      fix: "",
    },
    {
      key: "gpu",
      label: "NVIDIA runtime",
      status: "ok",
      summary: "The GPU is visible to runtime containers.",
      detail: "CUDA and the NVIDIA container runtime are ready for local inference.",
      fix: "",
    },
    {
      key: "disk",
      label: "Disk budget",
      status: "ok",
      summary: "There is enough space for the bootstrap cache.",
      detail: "The node cache budget and free-space reserve are both healthy.",
      fix: "",
    },
    {
      key: "network",
      label: "Control plane reachability",
      status: "ok",
      summary: "Cloudflare control endpoints are reachable.",
      detail: "DNS, HTTPS reachability, and signed artifact fetches all passed.",
      fix: "",
    },
  ];
}

function buildDashboardCards() {
  return {
    health: {
      tone: "success",
      badge: "Healthy",
      value: "Healthy",
      detail: "The node is online and recovered from a brief DNS flap earlier today.",
    },
    earnings: {
      tone: "success",
      badge: "Live",
      value: "$1.82/hr",
      detail: "Current gross earnings estimate from active work.",
    },
    today_earnings: {
      tone: "success",
      badge: "Today",
      value: "$14.20",
      detail: "Gross earnings since local midnight.",
    },
    heartbeat: {
      tone: "success",
      badge: "Online",
      value: "12s ago",
      detail: "Control-plane heartbeat is fresh.",
    },
    model: {
      tone: "success",
      badge: "Warm",
      value: "Llama 3.1 8B Instruct",
      detail: "The bootstrap model is warm and serving local traffic.",
    },
    uptime: {
      tone: "success",
      badge: "Stable",
      value: "18h 22m",
      detail: "No owner intervention has been needed today.",
    },
    electricity: {
      tone: "warning",
      badge: "Estimate",
      value: "$4.06",
      detail: "Using live GPU watts and the current owner power cap.",
    },
    heat_offset: {
      tone: "success",
      badge: "Offset",
      value: "$5.10",
      detail: "Estimated value of useful heat delivered into the room today.",
    },
    net_value: {
      tone: "success",
      badge: "Net",
      value: "+$15.24",
      detail: "Earnings plus heat offset minus electricity.",
    },
    idle: {
      tone: "warning",
      badge: "Watching",
      value: "Ready",
      detail: "The node will pause cleanly when the room target is reached.",
    },
    changes: {
      tone: "warning",
      badge: "Recent",
      value: "Power cap set to 260 W",
      detail: "Quiet hours start at 22:00 local time.",
    },
    autopilot: {
      tone: "success",
      badge: "Tuned",
      value: "Concurrency 3",
      detail: "Ramping against utilization and VRAM headroom.",
    },
    model_cache: {
      tone: "success",
      badge: "Warm",
      value: "2 pinned models",
      detail: "Bootstrap plus likely next model kept warm.",
    },
  };
}

function buildHeatGovernor(mode = "100", overrides = {}) {
  const target = mode === "auto" ? 50 : Number(mode);
  const quietStart = overrides.quiet_hours_start_local ?? "22:00";
  const quietEnd = overrides.quiet_hours_end_local ?? "06:00";
  const powerCap = overrides.max_power_cap_watts ?? 260;
  const ownerObjective = overrides.owner_objective ?? "balanced";
  const ownerObjectiveLabel =
    ownerObjective === "earnings_only"
      ? "Earnings only"
      : ownerObjective === "heat_first"
        ? "Heat first"
        : "Balanced";
  const paused = mode === "0";
  const quietHoursActive = Boolean(overrides.quiet_hours_active);
  const reason = paused
    ? "The owner requested 0%, so new work is paused and active jobs can finish safely."
    : quietHoursActive
      ? `Quiet hours are active, so the node is holding a quieter ${target}% trickle.`
      : mode === "auto"
        ? "Auto heat mode is following the room target and current weather hint."
        : `Running at ${target}% because the owner heat target allows it.`;
  const decisionReasons = paused
    ? ["Owner heat target is 0%.", "Active work can finish safely before the node idles."]
    : [
        `Owner target is ${mode === "auto" ? "auto heat mode" : `${target}%`}.`,
        overrides.outside_temp_c != null
          ? `Outside temperature hint is ${overrides.outside_temp_c} C.`
          : "The room target has not been reached yet.",
      ];

  return {
    mode,
    owner_objective: ownerObjective,
    room_temp_c: overrides.room_temp_c ?? 21,
    target_temp_c: overrides.target_temp_c ?? 22,
    outside_temp_c: overrides.outside_temp_c ?? 5,
    quiet_hours_start_local: quietStart,
    quiet_hours_end_local: quietEnd,
    gpu_temp_limit_c: overrides.gpu_temp_limit_c ?? 82,
    gpu_power_limit_enabled: overrides.gpu_power_limit_enabled ?? true,
    max_power_cap_watts: powerCap,
    energy_price_kwh: overrides.energy_price_kwh ?? 0.22,
    policy_summary: `${ownerObjectiveLabel} policy. Quiet hours from ${quietStart} to ${quietEnd}. Max GPU power cap ${powerCap} W.`,
    plan: {
      mode,
      requested_target_pct: mode === "auto" ? 100 : target,
      effective_target_pct: target,
      owner_objective: ownerObjective,
      owner_objective_label: ownerObjectiveLabel,
      reason,
      decision_reasons: decisionReasons,
      quiet_hours_start_local: quietStart,
      quiet_hours_end_local: quietEnd,
      quiet_hours_active: quietHoursActive,
      paused,
      max_power_cap_watts: powerCap,
      gpu_temp_limit_c: overrides.gpu_temp_limit_c ?? 82,
      room_temp_c: overrides.room_temp_c ?? 21,
      target_temp_c: overrides.target_temp_c ?? 22,
      outside_temp_c: overrides.outside_temp_c ?? 5,
    },
  };
}

function buildStatus() {
  return {
    service: {
      logs: [
        "Local owner service ready.",
        "Watching Docker, GPU health, cache pressure, and support actions.",
      ],
    },
    runtime: {
      stage: "ready",
      message: "Quick Start can bring this machine online.",
      credentials_present: false,
    },
    installer: {
      state: {
        busy: false,
        error: "",
        logs: ["Waiting for Quick Start."],
        claim: null,
      },
      preflight: {
        docker_cli: true,
        docker_compose: true,
        docker_daemon: true,
        docker_error: "",
        running_services: ["node-agent", "vllm"],
        gpu: {
          detected: true,
          name: "NVIDIA RTX 4090",
          memory_gb: 24,
        },
        nvidia_container_runtime: {
          visible: true,
        },
        disk: {
          ok: true,
          free_gb: 612,
          total_gb: 1024,
        },
        setup_checks: buildSetupChecks(),
        setup_audit: {
          failed: 0,
          warnings: 0,
          total: 4,
          summary: "Everything needed for claim and startup is ready on this machine.",
          blocking_checks: [],
        },
        ready_for_claim: true,
        claim_gate_blockers: [],
        automatic_fixes: {
          attempted: false,
          summary: "The local service can repair startup, launcher, and firewall settings automatically.",
        },
      },
      config: {
        node_label: "Living Room Node",
        setup_profile: "balanced",
        recommended_setup_profile: "balanced",
        edge_control_url: "https://edge.autonomousc.com",
        offline_install_bundle_dir: "",
        node_region: "eu-se-1",
        trust_tier: "standard",
        restricted_capable: false,
        vllm_model: "meta-llama/Llama-3.1-8B-Instruct",
        supported_models: "meta-llama/Llama-3.1-8B-Instruct\nmistralai/Mistral-7B-Instruct-v0.3",
        max_concurrent_assignments: 4,
        target_gpu_utilization_pct: 100,
        min_gpu_memory_headroom_pct: 20,
        profile_summary: "Balanced profile with steady daytime heat and overnight quiet hours.",
        hugging_face_repository: "meta-llama/Llama-3.1-8B-Instruct",
        hugging_face_token_required: false,
      },
    },
    owner_setup: {
      headline: "Quick Start is ready.",
      detail: "Name this machine and continue.",
      machine_plan_summary: "NVIDIA RTX 4090 detected with 24 GB VRAM.",
      profile_summary: "Balanced profile with steady daytime heat and overnight quiet hours.",
      current_step: "detect",
      progress_percent: 18,
      eta_label: "About 4 minutes",
      recommendations: [
        {
          label: "Heat profile",
          value: "Balanced",
          detail: "Good default for home comfort and steady earnings.",
        },
        {
          label: "Bootstrap model",
          value: "Llama 3.1 8B Instruct",
          detail: "Fast to warm on this GPU and useful for first real traffic.",
        },
      ],
      first_run_wizard: {
        headline: "First-run appliance plan",
        detail: "GPU detection, heat profile, earnings estimate, install path, and bootstrap checks are ready.",
        steps: [
          {
            label: "Detect GPU",
            status: "complete",
            value: "RTX 4090 found",
            detail: "Docker and the NVIDIA runtime are healthy.",
          },
          {
            label: "Choose heat profile",
            status: "active",
            value: "Balanced",
            detail: "This keeps the node useful without making the room stuffy too early.",
          },
          {
            label: "Bootstrap model",
            status: "pending",
            value: "Llama 3.1 8B Instruct",
            detail: "The starter model will warm after claim.",
          },
        ],
      },
      steps: [
        {
          key: "detect",
          label: "Detect hardware",
          status: "complete",
          detail: "GPU, Docker, and disk checks passed.",
        },
        {
          key: "approve",
          label: "Approve node",
          status: "active",
          detail: "Quick Start will open approval once you continue.",
        },
        {
          key: "warm",
          label: "Warm bootstrap model",
          status: "pending",
          detail: "The node will warm the starter model after approval.",
        },
      ],
    },
    autostart: {
      label: "Enabled",
      enabled: true,
      supported: true,
      detail: "This node launches automatically on sign-in.",
    },
    desktop_launcher: {
      label: "Installed",
      enabled: true,
      supported: true,
      detail: "A desktop launcher is available for normal owners.",
    },
    updates: {
      auto_update_enabled: true,
      interval_hours: 24,
      last_result: "Checked 20 minutes ago. No restart is pending.",
      pending_restart: false,
    },
    diagnostics: {
      last_result: "Send a redacted support bundle with one click.",
      last_error: "",
      last_case_id: "",
      last_bundle_name: "",
      last_bundle_created_at: "",
      last_bundle_sent_at: "",
    },
    local_doctor: {
      status: "standing_by",
      headline: "Local Doctor is standing by",
      detail: "Run Local Doctor to re-check Docker, GPU access, network reachability, model cache, warm readiness, and one tiny local inference.",
      last_checked_at: "",
      last_check_mode: "manual",
      last_trigger: "",
      last_background_check_at: "",
      last_background_status: "",
      last_transition_alert: {},
      last_fix_attempt: {},
      recommended_fix: {},
      checks: [
        {
          key: "docker",
          label: "Docker",
          status: "pass",
          summary: "Docker is ready for the local runtime.",
          detail: "Docker Desktop and its engine look healthy for this machine.",
          tone: "success",
        },
        {
          key: "gpu",
          label: "GPU",
          status: "pass",
          summary: "The GPU stack is ready for local inference.",
          detail: "The NVIDIA driver, CUDA support, and container GPU access all look healthy.",
          tone: "success",
        },
      ],
      warm_readiness: {
        status: "warn",
        summary: "Local Doctor has not checked warm readiness yet.",
        detail: "Run the doctor once to verify the active model and warm-up state.",
        tone: "warning",
      },
      inference_probe: {
        status: "warn",
        summary: "The tiny local inference probe has not run yet.",
        detail: "Run the doctor once to send a tiny local inference request.",
        tone: "warning",
      },
      attached_bundle_name: "",
      attached_bundle_created_at: "",
    },
    self_healing: {
      status: "healthy",
      detail: "Self-healing is monitoring runtime, warm-up health, and cache pressure.",
      prerequisite_action: {},
    },
    dashboard: {
      headline: "Owner dashboard",
      detail: "Health, earnings, and owner-facing explanations stay in one place.",
      cards: buildDashboardCards(),
      setup_verification: {
        notification: {
          show: false,
        },
      },
    },
    heat_governor: buildHeatGovernor(),
  };
}

class OwnerAppStubServer {
  constructor() {
    this.server = null;
    this.url = "";
    this.state = buildStatus();
    this.requestLog = [];
  }

  async start() {
    this.server = http.createServer(async (request, response) => {
      try {
        await this.handle(request, response);
      } catch (error) {
        response.writeHead(500, { "content-type": "application/json" });
        response.end(
          JSON.stringify({
            error: {
              message: error instanceof Error ? error.message : "Stub server error",
            },
          }),
        );
      }
    });

    await new Promise((resolve) => this.server.listen(0, "127.0.0.1", resolve));
    const address = this.server.address();
    this.url = `http://127.0.0.1:${address.port}`;
  }

  async stop() {
    if (!this.server) {
      return;
    }
    await new Promise((resolve, reject) => {
      this.server.close((error) => {
        if (error) {
          reject(error);
          return;
        }
        resolve();
      });
    });
    this.server = null;
  }

  reset() {
    this.state = buildStatus();
    this.requestLog = [];
  }

  async handle(request, response) {
    const url = new URL(request.url, this.url);
    if (request.method === "GET" && (url.pathname === "/" || url.pathname === "/index.html")) {
      response.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      response.end(ownerAppHtml);
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/status") {
      this.logRequest(request.method, url.pathname);
      this.sendJson(response, this.state);
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/setup-preflight") {
      const body = await this.readJson(request);
      this.logRequest(request.method, url.pathname, body);
      this.state.installer.config.node_label = body.node_label || this.state.installer.config.node_label;
      this.state.installer.state.logs = [
        "Quick Start checked Docker, GPU access, disk, and claim readiness.",
      ];
      this.state.owner_setup.headline = "Quick Start verified the machine.";
      this.state.owner_setup.detail = "Docker, the GPU runtime, disk budget, and claim readiness are all clear.";
      this.state.owner_setup.current_step = "approve";
      this.state.owner_setup.progress_percent = 42;
      this.state.owner_setup.eta_label = "About 3 minutes";
      this.state.owner_setup.steps = [
        {
          key: "detect",
          label: "Detect hardware",
          status: "complete",
          detail: "GPU, Docker, and disk checks passed.",
        },
        {
          key: "approve",
          label: "Approve node",
          status: "active",
          detail: "Quick Start is ready to open the approval page.",
        },
        {
          key: "warm",
          label: "Warm bootstrap model",
          status: "pending",
          detail: "The node will warm the starter model after approval.",
        },
      ];
      this.sendJson(response, {
        ...this.state,
        setup_preview: {
          preflight: clone(this.state.installer.preflight),
        },
      });
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/install") {
      const body = await this.readJson(request);
      this.logRequest(request.method, url.pathname, body);
      this.state.installer.config.node_label = body.node_label || this.state.installer.config.node_label;
      this.state.installer.state.busy = true;
      this.state.installer.state.logs = [
        `Quick Start is claiming ${this.state.installer.config.node_label}.`,
        "The local service opened the approval page and is waiting for sign-in.",
      ];
      this.state.installer.state.claim = {
        status: "waiting_for_approval",
        claim_id: "claim-2048",
        claim_code: "AUTC-2048",
        approval_url: "https://example.invalid/claim/AUTC-2048",
        expires_at: "2026-04-21 12:30",
        renewal_count: 0,
        approval_qr_svg_data_url:
          "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='156' height='156'%3E%3Crect width='156' height='156' fill='white'/%3E%3Crect x='18' y='18' width='36' height='36' fill='black'/%3E%3Crect x='102' y='18' width='36' height='36' fill='black'/%3E%3Crect x='18' y='102' width='36' height='36' fill='black'/%3E%3Crect x='72' y='72' width='12' height='12' fill='black'/%3E%3C/svg%3E",
      };
      this.state.runtime.stage = "registration_required";
      this.state.runtime.message = "Waiting for owner approval.";
      this.state.owner_setup.headline = "Quick Start is waiting for approval.";
      this.state.owner_setup.detail = "Approve the sign-in page and the node will finish warming automatically.";
      this.state.owner_setup.current_step = "approve";
      this.state.owner_setup.progress_percent = 64;
      this.state.owner_setup.eta_label = "Waiting for approval";
      this.state.owner_setup.steps = [
        {
          key: "detect",
          label: "Detect hardware",
          status: "complete",
          detail: "GPU, Docker, and disk checks passed.",
        },
        {
          key: "approve",
          label: "Approve node",
          status: "active",
          detail: "Approve the sign-in page and the node will finish warming automatically.",
        },
        {
          key: "warm",
          label: "Warm bootstrap model",
          status: "pending",
          detail: "The node will warm the starter model after approval.",
        },
      ];
      this.state.dashboard.cards.changes = {
        tone: "warning",
        badge: "Recent",
        value: "Approval opened",
        detail: `Quick Start is claiming ${this.state.installer.config.node_label}.`,
      };
      this.sendJson(response, this.state);
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/heat-governor") {
      const body = await this.readJson(request);
      this.logRequest(request.method, url.pathname, body);
      const nextMode = body.mode || this.state.heat_governor.mode || "100";
      const nextQuietStart = body.quiet_hours_start_local || this.state.heat_governor.quiet_hours_start_local || "22:00";
      const nextQuietEnd = body.quiet_hours_end_local || this.state.heat_governor.quiet_hours_end_local || "06:00";
      const nextObjective = body.owner_objective || this.state.heat_governor.owner_objective || "balanced";
      const nextPowerCap =
        body.max_power_cap_watts == null
          ? (this.state.heat_governor.max_power_cap_watts ?? 260)
          : body.max_power_cap_watts;

      this.state.heat_governor = buildHeatGovernor(nextMode, {
        room_temp_c: body.room_temp_c ?? this.state.heat_governor.room_temp_c,
        target_temp_c: body.target_temp_c ?? this.state.heat_governor.target_temp_c,
        outside_temp_c: body.outside_temp_c ?? this.state.heat_governor.outside_temp_c,
        quiet_hours_start_local: nextQuietStart,
        quiet_hours_end_local: nextQuietEnd,
        gpu_temp_limit_c: body.gpu_temp_limit_c ?? this.state.heat_governor.gpu_temp_limit_c,
        gpu_power_limit_enabled:
          body.gpu_power_limit_enabled ?? this.state.heat_governor.gpu_power_limit_enabled,
        max_power_cap_watts: nextPowerCap,
        energy_price_kwh: body.energy_price_kwh ?? this.state.heat_governor.energy_price_kwh,
        owner_objective: nextObjective,
      });
      this.state.dashboard.cards.changes = {
        tone: "warning",
        badge: "Recent",
        value: `Heat mode ${this.state.heat_governor.mode}`,
        detail: `Quiet hours ${nextQuietStart}-${nextQuietEnd}. Power cap ${nextPowerCap} W.`,
      };
      this.sendJson(response, this.state);
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/diagnostics") {
      const body = await this.readJson(request);
      this.logRequest(request.method, url.pathname, body);
      this.state.diagnostics.last_bundle_name = "edge-node-support-20260421T121500Z.zip";
      this.state.diagnostics.last_bundle_created_at = "2026-04-21 12:15";
      this.state.diagnostics.last_result = "Created a redacted diagnostics bundle for local review.";
      this.state.diagnostics.last_error = "";
      this.sendJson(response, this.state);
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/local-doctor") {
      const body = await this.readJson(request);
      this.logRequest(request.method, url.pathname, body);
      this.state.local_doctor = {
        status: "attention",
        headline: "Local Doctor found one thing to fix next",
        detail: "The local inference probe failed after readiness. Attached diagnostics bundle: edge-node-support-20260421T122000Z.zip.",
        last_checked_at: "2026-04-21 12:20",
        last_check_mode: "manual",
        last_trigger: "manual",
        last_background_check_at: "",
        last_background_status: "",
        last_transition_alert: {},
        last_fix_attempt: {},
        recommended_fix: {
          code: "repair_runtime",
          label: "Run prerequisite-healing",
          detail: "The runtime answered readiness but did not pass a tiny local inference. Restart the local runtime once, then run Local Doctor again.",
          automated: true,
        },
        checks: [
          {
            key: "docker",
            label: "Docker",
            status: "pass",
            summary: "Docker is ready for the local runtime.",
            detail: "Docker Desktop and its engine look healthy for this machine.",
            tone: "success",
          },
          {
            key: "gpu",
            label: "GPU",
            status: "pass",
            summary: "The GPU stack is ready for local inference.",
            detail: "The NVIDIA driver, CUDA support, and container GPU access all look healthy.",
            tone: "success",
          },
          {
            key: "network",
            label: "Network",
            status: "pass",
            summary: "DNS and control-plane reachability look healthy.",
            detail: "The machine resolved setup hostnames, reached the claim service, and can likely fetch signed artifacts.",
            tone: "success",
          },
          {
            key: "cache",
            label: "Cache",
            status: "pass",
            summary: "The local cache budget looks healthy for warm-ups.",
            detail: "Disk space and the local model cache budget look good for this machine.",
            tone: "success",
          },
        ],
        warm_readiness: {
          status: "pass",
          summary: "Llama 3.1 8B Instruct is warm and ready for local inference.",
          detail: "The local inference runtime answered readiness checks successfully on this machine.",
          tone: "success",
        },
        inference_probe: {
          status: "fail",
          summary: "The tiny local inference probe failed.",
          detail: "The runtime passed readiness, but the test request did not complete cleanly.",
          tone: "danger",
        },
        attached_bundle_name: "edge-node-support-20260421T122000Z.zip",
        attached_bundle_created_at: "2026-04-21 12:20",
      };
      this.state.diagnostics.last_bundle_name = "edge-node-support-20260421T122000Z.zip";
      this.state.diagnostics.last_bundle_created_at = "2026-04-21 12:20";
      this.state.diagnostics.last_result = "Local Doctor attached edge-node-support-20260421T122000Z.zip so support evidence stays ready on this machine.";
      this.state.diagnostics.last_error = "";
      this.sendJson(response, this.state);
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/local-doctor/fix") {
      const body = await this.readJson(request);
      this.logRequest(request.method, url.pathname, body);
      this.state.local_doctor = {
        ...this.state.local_doctor,
        status: "healthy",
        headline: "Local Doctor passed",
        detail: "Docker, GPU access, network reachability, warm readiness, and a tiny local inference all look healthy.",
        last_checked_at: "2026-04-21 12:22",
        last_check_mode: "manual",
        last_trigger: "auto_fix_verify",
        recommended_fix: {
          code: "none",
          label: "No fix needed",
          detail: "The machine passed Docker, GPU, network, warm readiness, and a tiny local inference.",
          automated: false,
        },
        inference_probe: {
          status: "pass",
          summary: "A tiny local responses probe completed successfully.",
          detail: "The local responses path answered on the startup model.",
          tone: "success",
        },
        attached_bundle_name: "",
        attached_bundle_created_at: "",
        last_fix_attempt: {
          applied_fix_code: "repair_runtime",
          applied_fix_label: "Run prerequisite-healing",
          automated: true,
          recovered: true,
          changed: true,
          summary: "Applied Run prerequisite-healing and Local Doctor passed on the automatic re-check.",
          before_after: "Before: Local Doctor found one thing to fix next. After: Local Doctor passed.",
        },
      };
      this.sendJson(response, this.state);
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/support/send") {
      const body = await this.readJson(request);
      this.logRequest(request.method, url.pathname, body);
      if (!this.state.diagnostics.last_bundle_name) {
        this.state.diagnostics.last_bundle_name = "edge-node-support-20260421T121500Z.zip";
        this.state.diagnostics.last_bundle_created_at = "2026-04-21 12:15";
      }
      this.state.diagnostics.last_case_id = "SUP-2048";
      this.state.diagnostics.last_bundle_sent_at = "2026-04-21 12:17";
      this.state.diagnostics.last_result = "Sent the latest redacted support bundle.";
      this.sendJson(response, this.state);
      return;
    }

    if (request.method === "GET" && url.pathname.startsWith("/downloads/")) {
      this.logRequest(request.method, url.pathname);
      response.writeHead(200, {
        "content-type": "application/zip",
        "content-disposition": `attachment; filename="${path.basename(url.pathname)}"`,
      });
      response.end("stub bundle");
      return;
    }

    if (request.method === "POST" && url.pathname.startsWith("/api/")) {
      const body = await this.readJson(request);
      this.logRequest(request.method, url.pathname, body);
      this.sendJson(response, this.state);
      return;
    }

    response.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    response.end("Not found");
  }

  sendJson(response, payload) {
    response.writeHead(200, { "content-type": "application/json; charset=utf-8" });
    response.end(JSON.stringify(payload));
  }

  async readJson(request) {
    const chunks = [];
    for await (const chunk of request) {
      chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : chunk);
    }
    if (!chunks.length) {
      return {};
    }
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  }

  logRequest(method, pathname, body = null) {
    this.requestLog.push({
      method,
      pathname,
      body: body == null ? null : clone(body),
    });
  }

  requestsFor(pathname, method = null) {
    return this.requestLog.filter(
      (entry) => entry.pathname === pathname && (method == null || entry.method === method),
    );
  }
}

const stubServer = new OwnerAppStubServer();

async function openOwnerApp(page) {
  await page.addInitScript(() => {
    window.open = () => {
      let isClosed = false;
      return {
        get closed() {
          return isClosed;
        },
        document: {
          write() {},
          close() {},
        },
        location: {
          replace() {},
        },
        close() {
          isClosed = true;
        },
      };
    };
  });
  await page.goto(stubServer.url, { waitUntil: "networkidle" });
  await expect(page.locator("#dashboard-headline")).toHaveText("Owner dashboard");
}

test.beforeAll(async () => {
  await stubServer.start();
});

test.afterAll(async () => {
  await stubServer.stop();
});

test.beforeEach(() => {
  stubServer.reset();
});

test("first-run Quick Start submits preflight then install", async ({ page }) => {
  await openOwnerApp(page);

  await page.locator("#node_label").fill("Kitchen Radiator Node");
  await page.locator("#setup-button").click();

  await expect(page.locator("#setup-button")).toHaveText("Quick Start is running...");
  await expect(page.locator("#progress-step")).toHaveText("Approve node");
  await expect(page.locator("#progress-detail")).toContainText("Approve the sign-in page");

  await expect
    .poll(() => stubServer.requestsFor("/api/setup-preflight", "POST").length)
    .toBe(1);
  await expect
    .poll(() => stubServer.requestsFor("/api/install", "POST").length)
    .toBe(1);

  const [preflightRequest] = stubServer.requestsFor("/api/setup-preflight", "POST");
  const [installRequest] = stubServer.requestsFor("/api/install", "POST");
  expect(preflightRequest.body).toMatchObject({
    setup_mode: "quickstart",
    node_label: "Kitchen Radiator Node",
  });
  expect(installRequest.body).toMatchObject({
    setup_mode: "quickstart",
    node_label: "Kitchen Radiator Node",
  });
});

test("heat mode changes explain the new target", async ({ page }) => {
  await openOwnerApp(page);

  await page.locator('button[data-heat-mode="20"]').click();

  await expect(page.locator("#heat-governor-copy")).toContainText("Heat target is 20%");
  await expect(page.locator("#heat-governor-detail")).toContainText("Running at 20%");
  await expect(page.locator("#heat-governor-why")).toContainText("Owner target is 20%");

  await expect
    .poll(() => stubServer.requestsFor("/api/heat-governor", "POST").length)
    .toBe(1);
  const [request] = stubServer.requestsFor("/api/heat-governor", "POST");
  expect(request.body).toMatchObject({
    mode: "20",
    owner_objective: "balanced",
  });
});

test("quiet hours save cleanly and stay visible in policy copy", async ({ page }) => {
  await openOwnerApp(page);

  await page.locator("#heat-quiet-start").fill("22:30");
  await page.locator("#heat-quiet-end").fill("06:15");
  await page.locator("#heat-max-power-cap").fill("235");
  await page.locator("#save-heat-governor").click();

  await expect(page.locator("#heat-governor-policy")).toContainText("Quiet hours from 22:30 to 06:15");
  await expect(page.locator("#heat-governor-policy")).toContainText("235 W");

  await expect
    .poll(() => stubServer.requestsFor("/api/heat-governor", "POST").length)
    .toBe(1);
  const [request] = stubServer.requestsFor("/api/heat-governor", "POST");
  expect(request.body).toMatchObject({
    quiet_hours_start_local: "22:30",
    quiet_hours_end_local: "06:15",
    max_power_cap_watts: 235,
  });
});

test("economics dashboard renders owner-facing values", async ({ page }) => {
  await openOwnerApp(page);

  await expect(page.locator("#dashboard-today_earnings-value")).toHaveText("$14.20");
  await expect(page.locator("#dashboard-electricity-value")).toHaveText("$4.06");
  await expect(page.locator("#dashboard-heat_offset-value")).toHaveText("$5.10");
  await expect(page.locator("#dashboard-net_value-value")).toHaveText("+$15.24");
});

test("support bundle creation and send flow stay one click away", async ({ page }) => {
  await openOwnerApp(page);

  await page.locator("#create-diagnostics").click();

  await expect(page.locator("#download-link")).toBeVisible();
  await expect(page.locator("#download-link")).toHaveText(
    "Download edge-node-support-20260421T121500Z.zip",
  );
  await expect(page.locator("#download-link")).toHaveAttribute(
    "href",
    /\/downloads\/edge-node-support-20260421T121500Z\.zip$/,
  );
  await expect(page.locator("#support-copy")).toContainText(
    "Created a redacted diagnostics bundle for local review.",
  );

  await page.locator("#send-support-bundle").click();

  await expect(page.locator("#support-copy")).toContainText(
    "Sent the latest redacted support bundle.",
  );
  await expect(page.locator("#support-case")).toContainText("Latest case ID: SUP-2048");

  await expect
    .poll(() => stubServer.requestsFor("/api/diagnostics", "POST").length)
    .toBe(1);
  await expect
    .poll(() => stubServer.requestsFor("/api/support/send", "POST").length)
    .toBe(1);
});

test("Local Doctor can fix and verify the node in one loop", async ({ page }) => {
  await openOwnerApp(page);

  await page.locator("#run-local-doctor").click();

  await expect(page.locator("#local-doctor-copy")).toContainText(
    "The local inference probe failed after readiness.",
  );
  await expect(page.locator("#local-doctor-meta")).toContainText(
    "Local Doctor found one thing to fix next",
  );
  await expect(page.locator("#local-doctor-fix")).toContainText(
    "Run prerequisite-healing",
  );
  await expect(page.locator("#local-doctor-download-link")).toBeVisible();
  await expect(page.locator("#local-doctor-download-link")).toHaveAttribute(
    "href",
    /\/downloads\/edge-node-support-20260421T122000Z\.zip$/,
  );
  await expect(page.locator("#apply-local-doctor-fix")).toBeEnabled();

  await expect
    .poll(() => stubServer.requestsFor("/api/local-doctor", "POST").length)
    .toBe(1);

  await page.locator("#apply-local-doctor-fix").click();

  await expect(page.locator("#local-doctor-copy")).toContainText(
    "Docker, GPU access, network reachability, warm readiness, and a tiny local inference all look healthy.",
  );
  await expect(page.locator("#local-doctor-verify")).toContainText(
    "Applied Run prerequisite-healing and Local Doctor passed on the automatic re-check.",
  );
  await expect(page.locator("#local-doctor-verify-detail")).toContainText(
    "Before: Local Doctor found one thing to fix next. After: Local Doctor passed.",
  );
  await expect(page.locator("#apply-local-doctor-fix")).toBeDisabled();

  await expect
    .poll(() => stubServer.requestsFor("/api/local-doctor/fix", "POST").length)
    .toBe(1);
});
