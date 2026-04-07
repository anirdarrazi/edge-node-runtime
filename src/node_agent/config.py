from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class NodeAgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    edge_control_url: str = Field(default="http://edge-control:8787")
    operator_token: str | None = None
    node_label: str = "AUTONOMOUSc Edge Node"
    node_region: str = "eu-se-1"
    trust_tier: str = "restricted"
    restricted_capable: bool = True
    node_id: str | None = None
    node_key: str | None = None
    credentials_path: str = "/var/lib/autonomousc/credentials/node-credentials.json"
    recovery_note_path: str = "/var/lib/autonomousc/credentials/recovery-note.txt"
    vllm_base_url: str = "http://vllm:8000"
    gpu_name: str = "Generic GPU"
    gpu_memory_gb: float = 24.0
    max_context_tokens: int = 32768
    max_batch_tokens: int = 50000
    max_concurrent_assignments: int = 2
    thermal_headroom: float = 0.8
    supported_models: str = "meta-llama/Llama-3.1-8B-Instruct,BAAI/bge-large-en-v1.5"
    poll_interval_seconds: int = 10
    agent_version: str = "0.1.0"
    docker_image: str = "autonomousc/node-agent:local"
    attestation_provider: Literal["simulated", "hardware"] = "simulated"


class AssignmentEnvelope(BaseModel):
    assignment_id: str
    execution_id: str
    operation: str
    model: str
    privacy_tier: str
    allowed_regions: list[str]
    token_budget: dict
    item_count: int
    input_artifact_url: str
    input_artifact_encryption: dict


class NodeClaimSession(BaseModel):
    claim_id: str
    claim_code: str
    approval_url: str
    poll_token: str
    expires_at: str
    poll_interval_seconds: int


class NodeClaimPollResult(BaseModel):
    status: Literal["pending", "approved", "consumed", "expired"]
    expires_at: str
    node_id: str | None = None
    node_key: str | None = None
