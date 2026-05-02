import json
import threading
import time
from contextlib import redirect_stdout
from io import BytesIO
from io import StringIO
from pathlib import Path

import node_agent.vast_smoke as vast_smoke


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict[str, object]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = None
            response = type(
                "ErrorResponse",
                (),
                {
                    "status_code": self.status_code,
                    "text": self.text,
                    "json": lambda _self: self._payload,
                },
            )()
            raise vast_smoke.httpx.HTTPStatusError("request failed", request=request, response=response)


class FakeVastAPI:
    def __init__(
        self,
        *,
        offers: list[dict[str, object]],
        instances: list[dict[str, object] | None],
        create_errors: dict[int, Exception] | None = None,
    ) -> None:
        self.offers = offers
        self.instances = list(instances)
        self.create_errors = dict(create_errors or {})
        self.created: list[dict[str, object]] = []
        self.destroyed: list[int] = []

    def search_offers(self, config: vast_smoke.VastSmokeConfig) -> list[dict[str, object]]:
        return list(self.offers)

    def create_instance(
        self,
        offer_id: int,
        *,
        image: str,
        env: dict[str, str],
        disk_gb: int,
        label: str,
        runtype: str,
    ) -> int:
        self.created.append(
            {
                "offer_id": offer_id,
                "image": image,
                "env": dict(env),
                "disk_gb": disk_gb,
                "label": label,
                "runtype": runtype,
            }
        )
        error = self.create_errors.get(offer_id)
        if error is not None:
            raise error
        return 424242

    def get_instance(self, instance_id: int) -> dict[str, object] | None:
        if self.instances:
            return self.instances.pop(0)
        return None

    def destroy_instance(self, instance_id: int) -> None:
        self.destroyed.append(instance_id)


class FakeRuntimeProbeClient:
    def __init__(self, *, get_responses: list[FakeResponse | Exception], post_responses: list[FakeResponse | Exception]) -> None:
        self.get_responses = list(get_responses)
        self.post_responses = list(post_responses)
        self.get_calls: list[str] = []
        self.post_calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str):
        self.get_calls.append(url)
        result = self.get_responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def post(self, url: str, *, json_body: dict[str, object]):
        self.post_calls.append((url, dict(json_body)))
        result = self.post_responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeDeleteClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.delete_calls: list[str] = []

    def delete(self, path: str) -> FakeResponse:
        self.delete_calls.append(path)
        return self.responses.pop(0)


class FakeGetClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.get_calls: list[str] = []

    def get(self, path: str) -> FakeResponse:
        self.get_calls.append(path)
        return self.responses.pop(0)


class TimedRuntimeProbeClient(FakeRuntimeProbeClient):
    def __init__(
        self,
        *,
        clock: FakeClock,
        get_responses: list[FakeResponse | Exception],
        post_responses: list[FakeResponse | Exception],
        seconds_per_post: float,
    ) -> None:
        super().__init__(get_responses=get_responses, post_responses=post_responses)
        self.clock = clock
        self.seconds_per_post = seconds_per_post

    def post(self, url: str, *, json_body: dict[str, object]):
        self.clock.now += self.seconds_per_post
        return super().post(url, json_body=json_body)


class ConcurrentTimedRuntimeProbeClient(FakeRuntimeProbeClient):
    def __init__(
        self,
        *,
        get_responses: list[FakeResponse | Exception],
        post_responses: list[FakeResponse | Exception],
        seconds_per_post: float,
    ) -> None:
        super().__init__(get_responses=get_responses, post_responses=post_responses)
        self.seconds_per_post = seconds_per_post
        self._lock = threading.Lock()
        self.active_posts = 0
        self.max_active_posts = 0

    def post(self, url: str, *, json_body: dict[str, object]):
        with self._lock:
            self.post_calls.append((url, dict(json_body)))
            self.active_posts += 1
            self.max_active_posts = max(self.max_active_posts, self.active_posts)
            result = self.post_responses.pop(0)
        try:
            time.sleep(self.seconds_per_post)
        finally:
            with self._lock:
                self.active_posts -= 1
        if isinstance(result, Exception):
            raise result
        return result


def test_choose_cheapest_offer_prefers_lower_price_once_hosts_are_model_suitable() -> None:
    selected = vast_smoke.choose_cheapest_offer(
        [
            {"id": 2, "dph_total": 0.19, "reliability": 0.99, "inet_down": 500},
            {"id": 1, "dph_total": 0.11, "reliability": 0.98, "inet_down": 300},
            {"id": 3, "dph_total": 0.11, "reliability": 0.97, "inet_down": 900},
        ],
        max_price=0.20,
        model="BAAI/bge-large-en-v1.5",
    )

    assert selected["id"] == 1


def test_choose_cheapest_offer_prefers_model_sized_vram_for_gemma() -> None:
    selected = vast_smoke.choose_cheapest_offer(
        [
            {
                "id": 1,
                "gpu_name": "RTX 5060 Ti",
                "dph_total": 0.18,
                "reliability": 0.998,
                "inet_down": 900,
                "gpu_ram": 16384,
                "disk_space": 120,
                "direct_port_count": 2,
            },
            {
                "id": 2,
                "gpu_name": "RTX 5090",
                "dph_total": 0.23,
                "reliability": 0.996,
                "inet_down": 700,
                "gpu_ram": 32768,
                "disk_space": 120,
                "direct_port_count": 2,
            },
        ],
        max_price=0.25,
        model="google/gemma-4-E4B-it",
    )

    assert selected["id"] == 1


def test_gemma_offer_policy_requires_5060_ti_runtime_fit() -> None:
    selected = vast_smoke.choose_cheapest_offer(
        [
            {
                "id": 1,
                "gpu_name": "RTX 4090",
                "gpu_ram": 24576,
                "dph_total": 0.06,
                "reliability": 0.999,
                "inet_down": 900,
                "disk_space": 120,
                "direct_port_count": 4,
            },
            {
                "id": 2,
                "gpu_name": "RTX 5060 Ti",
                "gpu_ram": 16384,
                "dph_total": 0.08,
                "reliability": 0.996,
                "inet_down": 900,
                "disk_space": 80,
                "direct_port_count": 2,
            },
            {
                "id": 3,
                "gpu_name": "RTX 5060 Ti",
                "gpu_ram": 16384,
                "dph_total": 0.07,
                "reliability": 0.996,
                "inet_down": 900,
                "disk_space": 80,
                "direct_port_count": 1,
            },
        ],
        max_price=0.25,
        model="google/gemma-4-E4B-it",
    )

    assert selected["id"] == 2


def test_offer_fit_tier_prefers_hosts_that_meet_both_vram_and_network_targets() -> None:
    assert vast_smoke.offer_fit_tier(
        {"gpu_ram": 24576, "inet_down": 650},
        model="Qwen/Qwen2.5-1.5B-Instruct",
    ) == 0
    assert vast_smoke.offer_fit_tier(
        {"gpu_ram": 24576, "inet_down": 400},
        model="Qwen/Qwen2.5-1.5B-Instruct",
    ) == 1
    assert vast_smoke.offer_fit_tier(
        {"gpu_ram": 16384, "inet_down": 650},
        model="Qwen/Qwen2.5-1.5B-Instruct",
    ) == 2
    assert vast_smoke.offer_fit_tier(
        {"gpu_ram": 16384, "inet_down": 400},
        model="Qwen/Qwen2.5-1.5B-Instruct",
    ) == 3


def test_choose_cheapest_offer_uses_price_when_readiness_is_tied() -> None:
    selected = vast_smoke.choose_cheapest_offer(
        [
            {"id": 2, "dph_total": 0.19, "reliability": 0.99, "inet_down": 500},
            {"id": 1, "dph_total": 0.11, "reliability": 0.99, "inet_down": 500},
        ],
        max_price=0.20,
    )

    assert selected["id"] == 1


def test_runner_launches_serves_and_destroys() -> None:
    clock = FakeClock()
    api = FakeVastAPI(
        offers=[
            {"id": 33560251, "gpu_name": "RTX 4000Ada", "gpu_ram": 20475, "dph_total": 0.108148, "reliability": 0.9987, "inet_down": 319.5, "disk_space": 445.0, "cuda_max_good": 13.0, "geolocation": "Japan, JP", "verification": "verified"},
        ],
        instances=[
            {"actual_status": "loading", "cur_state": "running", "status_msg": "pulling image"},
            {
                "actual_status": "running",
                "cur_state": "running",
                "public_ipaddr": "114.179.27.171",
                "gpu_name": "RTX 4000Ada",
                "gpu_ram": 20475,
                "dph_total": 0.121481,
                "geolocation": "Japan, JP",
                "ports": {"8000/tcp": [{"HostPort": "40188"}], "8011/tcp": [{"HostPort": "40189"}]},
            },
        ],
    )
    runtime = FakeRuntimeProbeClient(
        get_responses=[
            FakeResponse(
                200,
                {
                    "status": "warming",
                    "current_model": "BAAI/bge-large-en-v1.5",
                    "status_url": "http://127.0.0.1:8011/startup-status",
                },
            ),
            vast_smoke.httpx.ConnectError("connection refused"),
            FakeResponse(
                200,
                {
                    "status": "ready",
                    "current_model": "BAAI/bge-large-en-v1.5",
                    "status_url": "http://127.0.0.1:8011/startup-status",
                },
            ),
            FakeResponse(
                200,
                {
                    "object": "list",
                    "data": [{"id": "BAAI/bge-large-en-v1.5", "object": "model"}],
                },
            ),
        ],
        post_responses=[
            FakeResponse(
                200,
                {
                    "object": "list",
                    "model": "BAAI/bge-large-en-v1.5",
                    "data": [{"embedding": [0.0, 1.0, 2.0]}],
                    "usage": {"prompt_tokens": 8, "total_tokens": 8},
                },
            )
        ],
    )
    config = vast_smoke.VastSmokeConfig(api_key="secret", max_price=0.20)

    report = vast_smoke.VastSmokeRunner(
        api,
        runtime,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ).run(config)

    assert report["status"] == "ok"
    assert report["selected_offer"]["id"] == 33560251
    assert report["instance"]["id"] == 424242
    assert report["instance"]["runtype"] == "args"
    assert report["runtime"]["served_model"] == "BAAI/bge-large-en-v1.5"
    assert report["probe"]["api"] == "embeddings"
    assert report["probe"]["embedding_length"] == 3
    assert api.created[0]["env"]["RUN_MODE"] == "serve_only"
    assert api.created[0]["env"]["START_NODE_AGENT"] == "false"
    assert api.created[0]["env"]["STARTUP_STATUS_HOST"] == "0.0.0.0"
    assert api.created[0]["env"]["-p 8011:8011"] == "1"
    assert api.created[0]["env"]["STARTUP_STATUS_PORT"] == "8011"
    assert report["requested"]["expected_api_path"] == "/v1/embeddings"
    assert report["requested"]["readiness_path"] == "/v1/models"
    assert report["requested"]["startup_status_path"] == "/startup-status"
    assert report["runtime"]["startup_status"]["status"] == "ready"
    assert report["runtime"]["startup_status_code"] == 200
    assert report["instance"]["startup_status_host_port"] == 40189
    assert runtime.get_calls[0].endswith("/startup-status")
    assert runtime.get_calls[1].endswith("/v1/models")
    assert api.created[0]["runtype"] == "args"
    assert api.destroyed == [424242]


def test_runner_durable_gemma_node_uses_full_mode_and_stays_live() -> None:
    clock = FakeClock()
    api = FakeVastAPI(
        offers=[
            {
                "id": 506016,
                "gpu_name": "RTX 5060 Ti",
                "gpu_ram": 16384,
                "dph_total": 0.081,
                "reliability": 0.996,
                "inet_down": 900,
                "disk_space": 120,
                "cuda_max_good": 13.0,
                "direct_port_count": 2,
                "geolocation": "Sweden, SE",
                "verification": "verified",
            },
        ],
        instances=[
            {
                "actual_status": "running",
                "cur_state": "running",
                "public_ipaddr": "203.0.113.10",
                "gpu_name": "RTX 5060 Ti",
                "gpu_ram": 16384,
                "dph_total": 0.081,
                "geolocation": "Sweden, SE",
                "ports": {"8000/tcp": [{"HostPort": "41000"}], "8011/tcp": [{"HostPort": "41011"}]},
            },
        ],
    )
    runtime = FakeRuntimeProbeClient(
        get_responses=[
            FakeResponse(
                200,
                {
                    "status": "ready",
                    "current_model": "google/gemma-4-E4B-it",
                    "status_url": "http://127.0.0.1:8011/startup-status",
                },
            ),
            FakeResponse(
                200,
                {
                    "object": "list",
                    "data": [{"id": "google/gemma-4-E4B-it", "object": "model"}],
                },
            ),
        ],
        post_responses=[
            FakeResponse(
                200,
                {
                    "output": [{"content": [{"text": "ready"}]}],
                    "usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
                },
            )
        ],
    )
    config = vast_smoke.VastSmokeConfig(
        api_key="secret",
        model="google/gemma-4-E4B-it",
        max_price=0.20,
        api_kind="responses",
        durable_node=True,
        edge_control_url="https://edge.autonomousc.com",
        node_id="node_test",
        node_key="node_key_test",
        node_region="eu-se-1",
        runtime_profile="rtx_5060_ti_16gb_gemma4_e4b",
        max_context_tokens=32768,
    )

    report = vast_smoke.VastSmokeRunner(
        api,
        runtime,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ).run(config)

    env = api.created[0]["env"]
    assert report["status"] == "ok"
    assert report["requested"]["run_mode"] == "full"
    assert report["cleanup"] == {"destroyed": False, "kept_alive": True, "instance_id": 424242}
    assert api.destroyed == []
    assert env["RUN_MODE"] == "full"
    assert env["START_NODE_AGENT"] == "true"
    assert env["TEMPORARY_NODE"] == "false"
    assert env["DISABLE_PUBLIC_BOOTSTRAP_FALLBACK"] == "true"
    assert env["RUNTIME_PROFILE"] == "rtx_5060_ti_16gb_gemma4_e4b"
    assert env["VLLM_MODEL"] == "google/gemma-4-E4B-it"
    assert env["SUPPORTED_MODELS"] == "google/gemma-4-E4B-it"
    assert env["NODE_REGION"] == "eu-se-1"
    assert env["MAX_CONTEXT_TOKENS"] == "32768"
    assert env["MAX_BATCH_TOKENS"] == "32768"
    assert env["MAX_CONCURRENT_ASSIGNMENTS"] == "12"
    assert env["MAX_CONCURRENT_ASSIGNMENTS_CAP"] == "12"
    assert env["MAX_LOCAL_QUEUE_ASSIGNMENTS"] == "24"
    assert env["PULL_BUNDLE_SIZE"] == "40"
    assert env["VLLM_STARTUP_TIMEOUT_SECONDS"] == "1800"
    assert env["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] == "1"
    assert env["HEAT_GOVERNOR_MODE"] == "100"
    assert env["TARGET_GPU_UTILIZATION_PCT"] == "100"
    assert env["THERMAL_HEADROOM"] == "0.95"
    assert env["OWNER_OBJECTIVE"] == "earnings_only"
    assert env["GPU_POWER_LIMIT_ENABLED"] == "false"
    assert env["MIN_GPU_MEMORY_HEADROOM_PCT"] == "5"
    assert env["ALLOW_HIGH_GPU_MEMORY_PRESSURE"] == "true"
    assert "--quantization fp8" in env["VLLM_EXTRA_ARGS"]
    assert "--max-num-seqs 12" in env["VLLM_EXTRA_ARGS"]


def test_runner_retries_stale_offer_and_keeps_real_error_body() -> None:
    clock = FakeClock()
    api = FakeVastAPI(
        offers=[
            {"id": 1001, "gpu_name": "RTX 3090", "gpu_ram": 24576, "dph_total": 0.11, "reliability": 0.997, "inet_down": 800, "cuda_max_good": 13.0},
            {"id": 1002, "gpu_name": "RTX 3090", "gpu_ram": 24576, "dph_total": 0.12, "reliability": 0.995, "inet_down": 600, "cuda_max_good": 13.0},
        ],
        instances=[
            {
                "actual_status": "running",
                "cur_state": "running",
                "public_ipaddr": "114.179.27.171",
                "gpu_name": "RTX 3090",
                "gpu_ram": 24576,
                "dph_total": 0.12,
                "geolocation": "Japan, JP",
                "ports": {"8000/tcp": [{"HostPort": "40188"}], "8011/tcp": [{"HostPort": "40189"}]},
            }
        ],
        create_errors={
            1001: vast_smoke.VastInstanceLaunchError(
                "Vast rejected ask 1001 with HTTP 400: error 404/3603: no_such_ask Instance type by id 1001 is not available. (ask_id=1001)",
                retryable=True,
                offer_id=1001,
            )
        },
    )
    runtime = FakeRuntimeProbeClient(
        get_responses=[
            FakeResponse(
                200,
                {
                    "status": "ready",
                    "current_model": "BAAI/bge-large-en-v1.5",
                },
            ),
            FakeResponse(
                200,
                {
                    "object": "list",
                    "data": [{"id": "BAAI/bge-large-en-v1.5", "object": "model"}],
                },
            )
        ],
        post_responses=[
            FakeResponse(
                200,
                {
                    "object": "list",
                    "model": "BAAI/bge-large-en-v1.5",
                    "data": [{"embedding": [0.0, 1.0, 2.0]}],
                    "usage": {"prompt_tokens": 8, "total_tokens": 8},
                },
            )
        ],
    )
    config = vast_smoke.VastSmokeConfig(api_key="secret", max_price=0.20)

    report = vast_smoke.VastSmokeRunner(
        api,
        runtime,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ).run(config)

    assert report["status"] == "ok"
    assert [attempt["offer_id"] for attempt in api.created] == [1001, 1002]
    assert report["selected_offer"]["id"] == 1002
    assert "no_such_ask" in report["notes"][0]
    assert "Trying the next cheapest suitable offer." in report["notes"][0]
    assert api.destroyed == [424242]


def test_runner_retries_next_host_after_transient_candidate_probe_failure() -> None:
    clock = FakeClock()
    api = FakeVastAPI(
        offers=[
            {"id": 1001, "gpu_name": "RTX 3090", "gpu_ram": 24576, "dph_total": 0.11, "reliability": 0.997, "inet_down": 800, "cuda_max_good": 13.0},
            {"id": 1002, "gpu_name": "RTX 3090", "gpu_ram": 24576, "dph_total": 0.12, "reliability": 0.995, "inet_down": 600, "cuda_max_good": 13.0},
        ],
        instances=[
            {
                "actual_status": "running",
                "cur_state": "running",
                "public_ipaddr": "114.179.27.171",
                "gpu_name": "RTX 3090",
                "gpu_ram": 24576,
                "dph_total": 0.11,
                "geolocation": "Japan, JP",
                "ports": {"8000/tcp": [{"HostPort": "40188"}], "8011/tcp": [{"HostPort": "40189"}]},
            },
            {
                "actual_status": "running",
                "cur_state": "running",
                "public_ipaddr": "114.179.27.172",
                "gpu_name": "RTX 3090",
                "gpu_ram": 24576,
                "dph_total": 0.12,
                "geolocation": "Japan, JP",
                "ports": {"8000/tcp": [{"HostPort": "40190"}], "8011/tcp": [{"HostPort": "40191"}]},
            },
        ],
    )
    runtime = FakeRuntimeProbeClient(
        get_responses=[
            FakeResponse(200, {"status": "ready", "current_model": "BAAI/bge-large-en-v1.5"}),
            FakeResponse(200, {"object": "list", "data": [{"id": "BAAI/bge-large-en-v1.5", "object": "model"}]}),
            FakeResponse(200, {"status": "ready", "current_model": "BAAI/bge-large-en-v1.5"}),
            FakeResponse(200, {"object": "list", "data": [{"id": "BAAI/bge-large-en-v1.5", "object": "model"}]}),
        ],
        post_responses=[
            vast_smoke.httpx.ConnectError("connection refused"),
            vast_smoke.httpx.ConnectError("connection refused"),
            vast_smoke.httpx.ConnectError("connection refused"),
            FakeResponse(
                200,
                {
                    "object": "list",
                    "model": "BAAI/bge-large-en-v1.5",
                    "data": [{"embedding": [0.0, 1.0, 2.0]}],
                    "usage": {"prompt_tokens": 8, "total_tokens": 8},
                },
            ),
        ],
    )
    config = vast_smoke.VastSmokeConfig(api_key="secret", max_price=0.20)

    report = vast_smoke.VastSmokeRunner(
        api,
        runtime,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ).run(config)

    assert report["status"] == "ok"
    assert report["selected_offer"]["id"] == 1002
    assert len(report["candidate_failures"]) == 1
    assert "connection refused" in report["candidate_failures"][0]["error"]
    assert any("Trying the next suitable Vast host." in note for note in report["notes"])
    assert api.destroyed == [424242, 424242]


def test_runner_refreshes_startup_status_after_models_are_ready() -> None:
    clock = FakeClock()
    api = FakeVastAPI(
        offers=[
            {"id": 101, "gpu_name": "RTX 4000Ada", "gpu_ram": 20475, "dph_total": 0.12, "reliability": 0.99, "inet_down": 500, "cuda_max_good": 13.0},
        ],
        instances=[
            {
                "actual_status": "running",
                "cur_state": "running",
                "public_ipaddr": "114.179.27.171",
                "gpu_name": "RTX 4000Ada",
                "gpu_ram": 20475,
                "dph_total": 0.12,
                "geolocation": "Japan, JP",
                "ports": {"8000/tcp": [{"HostPort": "40188"}], "8011/tcp": [{"HostPort": "40189"}]},
            }
        ],
    )
    runtime = FakeRuntimeProbeClient(
        get_responses=[
            FakeResponse(
                200,
                {
                    "status": "warming",
                    "current_model": "Qwen/Qwen2.5-1.5B-Instruct",
                },
            ),
            FakeResponse(
                200,
                {
                    "object": "list",
                    "data": [{"id": "Qwen/Qwen2.5-1.5B-Instruct", "object": "model"}],
                },
            ),
            FakeResponse(
                200,
                {
                    "status": "ready",
                    "current_model": "Qwen/Qwen2.5-1.5B-Instruct",
                    "warm_source": "relay_cache",
                },
            ),
        ],
        post_responses=[
            FakeResponse(
                200,
                {
                    "id": "resp_123",
                    "object": "response",
                    "output": [{"content": [{"text": "Ready"}]}],
                    "usage": {"input_tokens": 12, "output_tokens": 2, "total_tokens": 14},
                },
            )
        ],
    )
    config = vast_smoke.VastSmokeConfig(
        api_key="secret",
        max_price=0.20,
        model="Qwen/Qwen2.5-1.5B-Instruct",
    )

    report = vast_smoke.VastSmokeRunner(
        api,
        runtime,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ).run(config)

    assert report["status"] == "ok"
    assert report["runtime"]["startup_status"]["status"] == "ready"
    assert report["runtime"]["startup_status"]["warm_source"] == "relay_cache"
    assert runtime.get_calls == [
        "http://114.179.27.171:40189/startup-status",
        "http://114.179.27.171:40188/v1/models",
        "http://114.179.27.171:40189/startup-status",
    ]


def test_runner_waits_briefly_for_ready_startup_status_after_probe() -> None:
    clock = FakeClock()
    api = FakeVastAPI(
        offers=[
            {"id": 101, "gpu_name": "RTX 4000Ada", "gpu_ram": 20475, "dph_total": 0.12, "reliability": 0.99, "inet_down": 500, "cuda_max_good": 13.0},
        ],
        instances=[
            {
                "actual_status": "running",
                "cur_state": "running",
                "public_ipaddr": "114.179.27.171",
                "gpu_name": "RTX 4000Ada",
                "gpu_ram": 20475,
                "dph_total": 0.12,
                "geolocation": "Japan, JP",
                "ports": {"8000/tcp": [{"HostPort": "40188"}], "8011/tcp": [{"HostPort": "40189"}]},
            }
        ],
    )
    runtime = FakeRuntimeProbeClient(
        get_responses=[
            FakeResponse(
                200,
                {
                    "status": "warming",
                    "current_model": "Qwen/Qwen2.5-1.5B-Instruct",
                },
            ),
            FakeResponse(
                200,
                {
                    "object": "list",
                    "data": [{"id": "Qwen/Qwen2.5-1.5B-Instruct", "object": "model"}],
                },
            ),
            FakeResponse(
                200,
                {
                    "status": "warming",
                    "current_model": "Qwen/Qwen2.5-1.5B-Instruct",
                },
            ),
            FakeResponse(
                200,
                {
                    "status": "ready",
                    "current_model": "Qwen/Qwen2.5-1.5B-Instruct",
                    "warm_source": "relay_cache",
                },
            ),
        ],
        post_responses=[
            FakeResponse(
                200,
                {
                    "id": "resp_123",
                    "object": "response",
                    "output": [{"content": [{"text": "Ready"}]}],
                    "usage": {"input_tokens": 12, "output_tokens": 2, "total_tokens": 14},
                },
            )
        ],
    )
    config = vast_smoke.VastSmokeConfig(
        api_key="secret",
        max_price=0.20,
        model="Qwen/Qwen2.5-1.5B-Instruct",
    )

    report = vast_smoke.VastSmokeRunner(
        api,
        runtime,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ).run(config)

    assert report["status"] == "ok"
    assert report["runtime"]["startup_status"]["status"] == "ready"
    assert report["runtime"]["startup_status"]["warm_source"] == "relay_cache"
    assert runtime.get_calls == [
        "http://114.179.27.171:40189/startup-status",
        "http://114.179.27.171:40188/v1/models",
        "http://114.179.27.171:40189/startup-status",
        "http://114.179.27.171:40189/startup-status",
    ]


def test_runner_destroys_instance_after_probe_failure() -> None:
    clock = FakeClock()
    api = FakeVastAPI(
        offers=[{"id": 101, "gpu_name": "RTX 4000Ada", "gpu_ram": 20475, "dph_total": 0.12, "reliability": 0.99, "inet_down": 500, "cuda_max_good": 13.0}],
        instances=[
            {
                "actual_status": "running",
                "cur_state": "running",
                "public_ipaddr": "114.179.27.171",
                "gpu_name": "RTX 4000Ada",
                "gpu_ram": 20475,
                "dph_total": 0.12,
                "geolocation": "Japan, JP",
                "ports": {"8000/tcp": [{"HostPort": "40188"}], "8011/tcp": [{"HostPort": "40189"}]},
            }
        ],
    )
    runtime = FakeRuntimeProbeClient(
        get_responses=[
            FakeResponse(
                200,
                {
                    "status": "ready",
                    "current_model": "BAAI/bge-large-en-v1.5",
                },
            ),
            FakeResponse(
                200,
                {
                    "object": "list",
                    "data": [{"id": "BAAI/bge-large-en-v1.5", "object": "model"}],
                },
            )
        ],
        post_responses=[
            vast_smoke.httpx.ConnectError("runtime probe failed"),
            vast_smoke.httpx.ConnectError("runtime probe failed"),
            vast_smoke.httpx.ConnectError("runtime probe failed"),
        ],
    )
    config = vast_smoke.VastSmokeConfig(api_key="secret", max_price=0.20)

    report = vast_smoke.VastSmokeRunner(
        api,
        runtime,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ).run(config)

    assert report["status"] == "error"
    assert "runtime probe failed" in report["error"]
    assert report["cleanup"]["destroyed"] is True
    assert api.destroyed == [424242]


def test_runner_retries_transient_probe_errors_before_succeeding() -> None:
    clock = FakeClock()
    api = FakeVastAPI(
        offers=[{"id": 101, "gpu_name": "RTX 4000Ada", "gpu_ram": 20475, "dph_total": 0.12, "reliability": 0.99, "inet_down": 500, "cuda_max_good": 13.0}],
        instances=[
            {
                "actual_status": "running",
                "cur_state": "running",
                "public_ipaddr": "114.179.27.171",
                "gpu_name": "RTX 4000Ada",
                "gpu_ram": 20475,
                "dph_total": 0.12,
                "geolocation": "Japan, JP",
                "ports": {"8000/tcp": [{"HostPort": "40188"}], "8011/tcp": [{"HostPort": "40189"}]},
            }
        ],
    )
    runtime = FakeRuntimeProbeClient(
        get_responses=[
            FakeResponse(
                200,
                {
                    "status": "ready",
                    "current_model": "Qwen/Qwen2.5-1.5B-Instruct",
                },
            ),
            FakeResponse(
                200,
                {
                    "object": "list",
                    "data": [{"id": "Qwen/Qwen2.5-1.5B-Instruct", "object": "model"}],
                },
            ),
        ],
        post_responses=[
            vast_smoke.httpx.ConnectError("connection refused"),
            FakeResponse(
                200,
                {
                    "id": "resp_123",
                    "object": "response",
                    "output": [{"content": [{"text": "Ready"}]}],
                    "usage": {"input_tokens": 12, "output_tokens": 2, "total_tokens": 14},
                },
            ),
        ],
    )
    config = vast_smoke.VastSmokeConfig(
        api_key="secret",
        max_price=0.20,
        model="Qwen/Qwen2.5-1.5B-Instruct",
    )

    report = vast_smoke.VastSmokeRunner(
        api,
        runtime,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ).run(config)

    assert report["status"] == "ok"
    assert report["probe"]["api"] == "chat_completions_fallback"
    assert report["notes"] == []


def test_auto_probe_falls_back_to_chat_completions_on_transport_error() -> None:
    runtime = FakeRuntimeProbeClient(
        get_responses=[],
        post_responses=[
            vast_smoke.httpx.ConnectError("connection refused"),
            FakeResponse(
                200,
                {
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "choices": [{"message": {"content": "Ready"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
                },
            ),
        ],
    )
    runner = vast_smoke.VastSmokeRunner(
        api=FakeVastAPI(offers=[], instances=[]),
        runtime=runtime,
    )

    probe = runner.run_probe(
        "http://127.0.0.1:8000",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        api_kind="auto",
        smoke_test_api_path="/v1/responses",
    )

    assert probe["api"] == "chat_completions_fallback"
    assert runtime.post_calls[0][0].endswith("/v1/responses")
    assert runtime.post_calls[1][0].endswith("/v1/chat/completions")


def test_responses_probe_uses_deterministic_short_output_settings() -> None:
    runtime = FakeRuntimeProbeClient(
        get_responses=[],
        post_responses=[
            FakeResponse(
                200,
                {
                    "id": "resp_123",
                    "object": "response",
                    "output": [{"content": [{"text": "Ready"}]}],
                    "usage": {"input_tokens": 12, "output_tokens": 2, "total_tokens": 14},
                },
            ),
        ],
    )
    runner = vast_smoke.VastSmokeRunner(
        api=FakeVastAPI(offers=[], instances=[]),
        runtime=runtime,
    )

    probe = runner.run_probe(
        "http://127.0.0.1:8000",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        api_kind="responses",
    )

    assert probe["api"] == "responses"
    assert runtime.post_calls[0][1]["temperature"] == 0
    assert runtime.post_calls[0][1]["max_output_tokens"] == 8


def test_runner_fails_fast_when_startup_status_reports_failure_reason() -> None:
    clock = FakeClock()
    api = FakeVastAPI(
        offers=[{"id": 101, "gpu_name": "RTX 4000Ada", "gpu_ram": 20475, "dph_total": 0.12, "reliability": 0.99, "inet_down": 500, "cuda_max_good": 13.0}],
        instances=[
            {
                "actual_status": "running",
                "cur_state": "running",
                "public_ipaddr": "114.179.27.171",
                "gpu_name": "RTX 4000Ada",
                "gpu_ram": 20475,
                "dph_total": 0.12,
                "geolocation": "Japan, JP",
                "ports": {"8000/tcp": [{"HostPort": "40188"}], "8011/tcp": [{"HostPort": "40189"}]},
            }
        ],
    )
    runtime = FakeRuntimeProbeClient(
        get_responses=[
            FakeResponse(
                200,
                {
                    "status": "failed",
                    "current_model": "Qwen/Qwen2.5-1.5B-Instruct",
                    "failure_reason": "model warm failed repeatedly",
                    "warm_source": "relay_cache",
                },
            )
        ],
        post_responses=[],
    )
    config = vast_smoke.VastSmokeConfig(api_key="secret", max_price=0.20)

    report = vast_smoke.VastSmokeRunner(
        api,
        runtime,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    ).run(config)

    assert report["status"] == "error"
    assert "model warm failed repeatedly" in report["error"]
    assert report["runtime"]["startup_status"]["status"] == "failed"
    assert report["runtime"]["startup_status"]["failure_reason"] == "model warm failed repeatedly"
    assert runtime.get_calls == ["http://114.179.27.171:40189/startup-status"]
    assert api.destroyed == [424242]


def test_destroy_instance_retries_transient_http_errors(monkeypatch) -> None:
    client = FakeDeleteClient(
        responses=[
            FakeResponse(429, {"success": False, "msg": "too many requests"}),
            FakeResponse(503, {"success": False, "msg": "temporarily unavailable"}),
            FakeResponse(200, {"success": True}),
        ]
    )
    api = vast_smoke.VastAPI("secret", client=client)
    monkeypatch.setattr(vast_smoke.time, "sleep", lambda _seconds: None)

    api.destroy_instance(424242)

    assert client.delete_calls == [
        "/instances/424242/",
        "/instances/424242/",
        "/instances/424242/",
    ]


def test_get_instance_retries_transient_http_errors(monkeypatch) -> None:
    client = FakeGetClient(
        responses=[
            FakeResponse(429, {"success": False, "msg": "too many requests"}),
            FakeResponse(200, {"instances": {"id": 424242, "actual_status": "running"}}),
        ]
    )
    api = vast_smoke.VastAPI("secret", client=client)
    monkeypatch.setattr(vast_smoke.time, "sleep", lambda _seconds: None)

    instance = api.get_instance(424242)

    assert instance == {"id": 424242, "actual_status": "running"}
    assert client.get_calls == [
        "/instances/424242/",
        "/instances/424242/",
    ]


def test_wait_for_instance_allows_extra_time_while_image_layers_are_still_downloading() -> None:
    clock = FakeClock()
    api = FakeVastAPI(
        offers=[],
        instances=[
            {"actual_status": "loading", "cur_state": "running", "status_msg": "Downloading [===========>     ]"},
            {"actual_status": "loading", "cur_state": "running", "status_msg": "Download complete"},
            {
                "actual_status": "running",
                "cur_state": "running",
                "public_ipaddr": "114.179.27.171",
                "ports": {"8000/tcp": [{"HostPort": "40188"}], "8011/tcp": [{"HostPort": "40189"}]},
            },
        ],
    )
    runner = vast_smoke.VastSmokeRunner(
        api,
        FakeRuntimeProbeClient(get_responses=[], post_responses=[]),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    instance = runner.wait_for_instance(
        424242,
        timeout_seconds=1.0,
        poll_interval_seconds=1.0,
    )

    assert instance["actual_status"] == "running"
    assert clock.now > 1001.0


def test_should_allow_launch_grace_covers_common_vast_cold_start_statuses() -> None:
    assert vast_smoke.should_allow_launch_grace("Pull complete")
    assert vast_smoke.should_allow_launch_grace("Verifying Checksum")
    assert vast_smoke.should_allow_launch_grace("Successfully loaded anirdarrazi/autonomousc-ai-edge-runtime:single-cuda-latest")
    assert vast_smoke.should_allow_launch_grace("success, running anirdarrazi/autonomousc-ai-edge-runtime:single-cuda-latest")


def test_main_prints_json_report(monkeypatch) -> None:
    fake_report = {
        "status": "ok",
        "selected_offer": {"id": 123},
        "cleanup": {"destroyed": True},
    }

    class FakeAPI:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def close(self) -> None:
            pass

    class FakeRuntime:
        def close(self) -> None:
            pass

    class FakeRunner:
        def __init__(self, api, runtime) -> None:
            self.api = api
            self.runtime = runtime

        def run(self, config: vast_smoke.VastSmokeConfig) -> dict[str, object]:
            return dict(fake_report)

    monkeypatch.setattr(vast_smoke, "VastAPI", FakeAPI)
    monkeypatch.setattr(vast_smoke, "RuntimeProbeClient", FakeRuntime)
    monkeypatch.setattr(vast_smoke, "VastSmokeRunner", FakeRunner)

    buffer = StringIO()
    with redirect_stdout(buffer):
        exit_code = vast_smoke.main(["--api-key", "secret"])

    assert exit_code == 0
    assert json.loads(buffer.getvalue()) == fake_report


def test_emit_json_report_falls_back_to_utf8_stdout_buffer(monkeypatch) -> None:
    class FakeStdout:
        def __init__(self) -> None:
            self.buffer = BytesIO()

        def write(self, text: str) -> int:
            text.encode("cp1252")
            return len(text)

        def flush(self) -> None:
            return None

    fake_stdout = FakeStdout()
    monkeypatch.setattr(vast_smoke.sys, "stdout", fake_stdout)

    vast_smoke.emit_json_report({"status": "error", "error": "█ blocked"}, indent=2)

    payload = json.loads(fake_stdout.buffer.getvalue().decode("utf-8"))
    assert payload["status"] == "error"
    assert payload["error"] == "█ blocked"


def test_build_config_accepts_api_key_from_env(monkeypatch) -> None:
    monkeypatch.delenv("NODE_AGENT_VAST_SMOKE_CONFIG", raising=False)
    monkeypatch.setenv("VAST_API_KEY", "env-secret")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    config = vast_smoke.build_config_from_args(
        vast_smoke.parse_args([])
    )

    assert config.api_key == "env-secret"
    assert config.model == vast_smoke.DEFAULT_VAST_SMOKE_MODEL


def test_build_config_accepts_api_key_from_local_config(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "vast-smoke.json"
    config_path.write_text(
        json.dumps(
            {
                "api_key": "config-secret",
                "hf_token": "hf-config-secret",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NODE_AGENT_VAST_SMOKE_CONFIG", str(config_path))
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    config = vast_smoke.build_config_from_args(
        vast_smoke.parse_args([])
    )

    assert config.api_key == "config-secret"
    assert config.hf_token == "hf-config-secret"


def test_build_config_prefers_cli_secret_over_local_config(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "vast-smoke.json"
    config_path.write_text(json.dumps({"api_key": "config-secret"}), encoding="utf-8")
    monkeypatch.setenv("NODE_AGENT_VAST_SMOKE_CONFIG", str(config_path))
    monkeypatch.delenv("VAST_API_KEY", raising=False)

    config = vast_smoke.build_config_from_args(
        vast_smoke.parse_args(["--api-key", "cli-secret"])
    )

    assert config.api_key == "cli-secret"


def test_build_config_defaults_follow_vast_launch_profile() -> None:
    config = vast_smoke.build_config_from_args(
        vast_smoke.parse_args(["--api-key", "secret"])
    )

    assert vast_smoke.DEFAULT_VAST_SMOKE_MODEL == "BAAI/bge-large-en-v1.5"
    assert config.model == vast_smoke.DEFAULT_VAST_LAUNCH_PROFILE.preferred_smoke_test_model
    assert config.model == vast_smoke.DEFAULT_VAST_SMOKE_MODEL
    assert config.max_price == vast_smoke.DEFAULT_VAST_LAUNCH_PROFILE.safe_price_ceiling_usd
    assert config.disk_gb == vast_smoke.DEFAULT_VAST_LAUNCH_PROFILE.min_disk_gb
    assert config.runtype == vast_smoke.DEFAULT_VAST_LAUNCH_PROFILE.runtype
    assert config.smoke_test_api_path == vast_smoke.DEFAULT_VAST_LAUNCH_PROFILE.smoke_test_api_path
    assert config.min_inet_down_mbps == 250.0
    assert config.min_cuda_max_good == 12.9


def test_build_config_uses_response_probe_defaults_for_non_embedding_models() -> None:
    config = vast_smoke.build_config_from_args(
        vast_smoke.parse_args(["--api-key", "secret", "--model", "Qwen/Qwen2.5-1.5B-Instruct"])
    )

    assert config.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.api_kind == "auto"
    assert config.smoke_test_api_path == "/v1/responses"


def test_direct_config_derives_response_probe_defaults_for_non_embedding_models() -> None:
    config = vast_smoke.VastSmokeConfig(
        api_key="secret",
        model="Qwen/Qwen2.5-1.5B-Instruct",
    )

    assert config.api_kind == "auto"
    assert config.smoke_test_api_path == "/v1/responses"


def test_direct_config_applies_safer_vast_context_defaults_for_known_models() -> None:
    qwen = vast_smoke.VastSmokeConfig(api_key="secret", model="Qwen/Qwen2.5-1.5B-Instruct")
    llama = vast_smoke.VastSmokeConfig(api_key="secret", model="meta-llama/Llama-3.1-8B-Instruct")
    gemma = vast_smoke.VastSmokeConfig(api_key="secret", model="google/gemma-4-E4B-it")
    gemma_26b = vast_smoke.VastSmokeConfig(api_key="secret", model="google/gemma-4-26b-a4b-it")
    gemma_26b_fp8 = vast_smoke.VastSmokeConfig(api_key="secret", model="aeyeops/gemma-4-26b-a4b-it-fp8")

    assert qwen.max_context_tokens == 16384
    assert llama.max_context_tokens == 8192
    assert gemma.max_context_tokens == 32768
    assert gemma_26b.max_context_tokens == 8192
    assert gemma_26b_fp8.max_context_tokens == 8192
    assert qwen.min_inet_down_mbps == 500.0
    assert llama.min_inet_down_mbps == 600.0
    assert gemma.min_inet_down_mbps == 600.0
    assert gemma_26b.min_inet_down_mbps == 700.0
    assert gemma_26b_fp8.min_inet_down_mbps == 700.0
    assert llama.min_vram_gb == 24.0
    assert gemma.min_vram_gb == 15.0
    assert gemma_26b.min_vram_gb == 70.0
    assert gemma_26b_fp8.min_vram_gb == 70.0


def test_benchmark_embeddings_input_respects_smaller_embedding_contexts() -> None:
    payload = vast_smoke.benchmark_embeddings_input_for_model("BAAI/bge-large-en-v1.5")
    words = payload[0].split()

    assert len(payload) == 1
    assert 48 <= len(words) <= 96


def test_benchmark_response_profiles_change_input_and_output_shape() -> None:
    balanced = vast_smoke.benchmark_responses_input("balanced", model="google/gemma-4-26b-a4b-it")
    input_heavy = vast_smoke.benchmark_responses_input("input_heavy", model="google/gemma-4-26b-a4b-it")
    output_heavy = vast_smoke.benchmark_responses_input("output_heavy", model="google/gemma-4-26b-a4b-it")

    assert "single word ready" in input_heavy.lower()
    assert len(input_heavy.split()) > len(balanced.split())
    assert "exactly 320 plain-text words" in output_heavy
    assert vast_smoke.benchmark_max_output_tokens("input_heavy") == 16
    assert vast_smoke.benchmark_max_output_tokens("output_heavy") == 384


def test_build_config_accepts_benchmark_requests() -> None:
    config = vast_smoke.build_config_from_args(
        vast_smoke.parse_args(
            [
                "--api-key",
                "secret",
                "--model",
                "Qwen/Qwen2.5-1.5B-Instruct",
                "--benchmark-requests",
                "6",
                "--benchmark-concurrency",
                "3",
                "--benchmark-profile",
                "input_heavy",
                "--vllm-extra-args",
                "--gpu-memory-utilization 0.95",
            ]
        )
    )

    assert config.benchmark_requests == 6
    assert config.benchmark_concurrency == 3
    assert config.benchmark_profile == "input_heavy"
    assert config.vllm_extra_args == "--gpu-memory-utilization 0.95"


def test_api_kind_for_probe_result_maps_chat_completions_fallback() -> None:
    assert vast_smoke.api_kind_for_probe_result({"api": "chat_completions_fallback"}) == "chat_completions"
    assert vast_smoke.api_kind_for_probe_result({"api": "responses"}) == "responses"
    assert vast_smoke.api_kind_for_probe_result({"api": "embeddings"}) == "embeddings"


def test_should_retry_candidate_after_startup_failure_for_memory_like_runtime_errors() -> None:
    runtime_report = {
        "startup_status": {
            "status": "failed",
            "failure_reason": "Engine core initialization failed. Failed core proc(s): {}",
        }
    }

    assert vast_smoke.should_retry_candidate_after_error(RuntimeError("startup failed"), runtime_report) is True


def test_normalized_usage_from_probe_tracks_cached_input_tokens() -> None:
    usage = vast_smoke.normalized_usage_from_probe(
        {
            "usage": {
                "input_tokens": 120,
                "output_tokens": 30,
                "total_tokens": 150,
                "input_tokens_details": {"cached_tokens": 40},
            }
        }
    )

    assert usage["input_tokens"] == 120
    assert usage["cached_input_tokens"] == 40
    assert usage["uncached_input_tokens"] == 80
    assert usage["output_tokens"] == 30
    assert usage["total_tokens"] == 150


def test_run_benchmark_reports_cost_per_million_tokens() -> None:
    clock = FakeClock()
    runtime = TimedRuntimeProbeClient(
        clock=clock,
        get_responses=[],
        post_responses=[
            FakeResponse(
                200,
                {
                    "id": "resp_1",
                    "object": "response",
                    "output": [{"content": [{"text": "ready"}]}],
                    "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                },
            ),
            FakeResponse(
                200,
                {
                    "id": "resp_2",
                    "object": "response",
                    "output": [{"content": [{"text": "ready"}]}],
                    "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                },
            ),
        ],
        seconds_per_post=0.5,
    )
    runner = vast_smoke.VastSmokeRunner(
        api=FakeVastAPI(offers=[], instances=[]),
        runtime=runtime,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    benchmark, notes = runner.run_benchmark(
        "http://127.0.0.1:8000",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        api_kind="responses",
        smoke_test_api_path="/v1/responses",
        request_count=2,
        concurrency=1,
        benchmark_profile="balanced",
        hourly_cost_usd=0.18,
        warmup_seconds=300.0,
    )

    assert notes == []
    assert benchmark["request_count"] == 2
    assert benchmark["concurrency"] == 1
    assert benchmark["benchmark_profile"] == "balanced"
    assert benchmark["api_counts"] == {"responses": 2}
    assert benchmark["requests_per_second"] == 2.0
    assert benchmark["usage"]["input_tokens"] == 200
    assert benchmark["usage"]["cached_input_tokens"] == 0
    assert benchmark["usage"]["uncached_input_tokens"] == 200
    assert benchmark["usage"]["output_tokens"] == 100
    assert benchmark["usage"]["total_tokens"] == 300
    assert benchmark["economics"]["hourly_cost_usd"] == 0.18
    assert benchmark["economics"]["input_tokens_per_second"] == 200.0
    assert benchmark["economics"]["cached_input_tokens_per_second"] == 0.0
    assert benchmark["economics"]["uncached_input_tokens_per_second"] == 200.0
    assert benchmark["economics"]["output_tokens_per_second"] == 100.0
    assert benchmark["economics"]["total_tokens_per_second"] == 300.0
    assert benchmark["economics"]["usd_per_million_input_tokens"] == 0.25
    assert benchmark["economics"]["usd_per_million_cached_input_tokens"] is None
    assert benchmark["economics"]["usd_per_million_uncached_input_tokens"] == 0.25
    assert benchmark["economics"]["usd_per_million_output_tokens"] == 0.5
    assert benchmark["economics"]["usd_per_million_total_tokens"] == 0.166667
    assert benchmark["pricing"]["warmup_seconds"] == 300.0
    assert benchmark["pricing"]["warmup_overhead_pct"] == 0.083333
    assert benchmark["pricing"]["effective_floor_usd_per_million_total_tokens"] == 0.180556
    assert benchmark["pricing"]["recommended_price_usd_per_million_total_tokens"] == 0.45139


def test_run_benchmark_supports_parallel_burst_concurrency() -> None:
    runtime = ConcurrentTimedRuntimeProbeClient(
        get_responses=[],
        post_responses=[
            FakeResponse(
                200,
                {
                    "id": f"resp_{index}",
                    "object": "response",
                    "output": [{"content": [{"text": "ready"}]}],
                    "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                },
            )
            for index in range(4)
        ],
        seconds_per_post=0.05,
    )
    runner = vast_smoke.VastSmokeRunner(
        api=FakeVastAPI(offers=[], instances=[]),
        runtime=runtime,
    )

    benchmark, notes = runner.run_benchmark(
        "http://127.0.0.1:8000",
        model="google/gemma-4-26b-a4b-it",
        api_kind="responses",
        smoke_test_api_path="/v1/responses",
        request_count=4,
        concurrency=2,
        benchmark_profile="output_heavy",
        hourly_cost_usd=0.35,
        warmup_seconds=120.0,
    )

    assert notes == []
    assert benchmark["request_count"] == 4
    assert benchmark["concurrency"] == 2
    assert benchmark["benchmark_profile"] == "output_heavy"
    assert benchmark["api_counts"] == {"responses": 4}
    assert benchmark["usage"]["total_tokens"] == 480
    assert runtime.max_active_posts >= 2
    assert benchmark["requests_per_second"] > 10.0


def test_affordable_offers_require_minimum_cuda_max_good() -> None:
    offers = vast_smoke.affordable_offers(
        [
            {"id": 1, "dph_total": 0.11, "reliability": 0.99, "inet_down": 500, "cuda_max_good": 12.4},
            {"id": 2, "dph_total": 0.12, "reliability": 0.995, "inet_down": 600, "cuda_max_good": 13.0},
        ],
        max_price=0.20,
        min_cuda_max_good=12.9,
    )

    assert [offer["id"] for offer in offers] == [2]
