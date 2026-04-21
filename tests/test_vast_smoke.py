import json
from contextlib import redirect_stdout
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

    def json(self) -> dict[str, object]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = None
            response = type("ErrorResponse", (), {"status_code": self.status_code})()
            raise vast_smoke.httpx.HTTPStatusError("request failed", request=request, response=response)


class FakeVastAPI:
    def __init__(self, *, offers: list[dict[str, object]], instances: list[dict[str, object] | None]) -> None:
        self.offers = offers
        self.instances = list(instances)
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


def test_choose_cheapest_offer_prefers_lowest_price() -> None:
    selected = vast_smoke.choose_cheapest_offer(
        [
            {"id": 2, "dph_total": 0.19, "reliability": 0.99, "inet_down": 500},
            {"id": 1, "dph_total": 0.11, "reliability": 0.98, "inet_down": 300},
            {"id": 3, "dph_total": 0.11, "reliability": 0.97, "inet_down": 900},
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
                "ports": {"8000/tcp": [{"HostPort": "40188"}]},
            },
        ],
    )
    runtime = FakeRuntimeProbeClient(
        get_responses=[
            vast_smoke.httpx.ConnectError("connection refused"),
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
    assert report["requested"]["expected_api_path"] == "/v1/embeddings"
    assert report["requested"]["readiness_path"] == "/v1/models"
    assert api.created[0]["runtype"] == "args"
    assert api.destroyed == [424242]


def test_runner_destroys_instance_after_probe_failure() -> None:
    clock = FakeClock()
    api = FakeVastAPI(
        offers=[{"id": 101, "gpu_name": "RTX 4000Ada", "gpu_ram": 20475, "dph_total": 0.12, "reliability": 0.99, "inet_down": 500}],
        instances=[
            {
                "actual_status": "running",
                "cur_state": "running",
                "public_ipaddr": "114.179.27.171",
                "gpu_name": "RTX 4000Ada",
                "gpu_ram": 20475,
                "dph_total": 0.12,
                "geolocation": "Japan, JP",
                "ports": {"8000/tcp": [{"HostPort": "40188"}]},
            }
        ],
    )
    runtime = FakeRuntimeProbeClient(
        get_responses=[
            FakeResponse(
                200,
                {
                    "object": "list",
                    "data": [{"id": "BAAI/bge-large-en-v1.5", "object": "model"}],
                },
            )
        ],
        post_responses=[vast_smoke.httpx.ConnectError("runtime probe failed")],
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
