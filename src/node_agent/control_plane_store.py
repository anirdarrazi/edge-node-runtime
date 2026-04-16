from __future__ import annotations

import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import NodeAgentSettings


class NodeCredentialStore:
    """Owns local credential, attestation, and recovery-note persistence."""

    def __init__(self, settings: NodeAgentSettings):
        self.settings = settings

    @staticmethod
    def tighten_directory_permissions(path: Path) -> None:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

    @classmethod
    def tighten_credentials_permissions(cls, path: Path) -> None:
        if path.parent.exists():
            cls.tighten_directory_permissions(path.parent)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)

    @classmethod
    def write_private_json(cls, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cls.tighten_directory_permissions(path.parent)
        serialized = json.dumps(payload, indent=2)
        temp_handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        )
        try:
            with temp_handle as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            temp_path = Path(temp_handle.name)
            cls.tighten_credentials_permissions(temp_path)
            temp_path.replace(path)
            cls.tighten_credentials_permissions(path)
        finally:
            temp_path = Path(temp_handle.name)
            if temp_path.exists():
                temp_path.unlink()

    def load_credentials(self) -> tuple[str, str] | None:
        credentials_path = Path(self.settings.credentials_path)
        if not credentials_path.exists():
            return None
        self.tighten_credentials_permissions(credentials_path)

        payload = json.loads(credentials_path.read_text(encoding="utf-8"))
        node_id = payload.get("node_id")
        node_key = payload.get("node_key")
        if not isinstance(node_id, str) or not isinstance(node_key, str):
            raise RuntimeError(f"invalid credentials file at {credentials_path}")

        self.settings.node_id = node_id
        self.settings.node_key = node_key
        return node_id, node_key

    def persist_credentials_file(self, node_id: str, node_key: str) -> None:
        self.write_private_json(Path(self.settings.credentials_path), {"node_id": node_id, "node_key": node_key})
        self.clear_recovery_note()

    def persist_credentials(self, node_id: str, node_key: str) -> tuple[str, str]:
        self.settings.node_id = node_id
        self.settings.node_key = node_key
        self.persist_credentials_file(node_id, node_key)
        self.clear_attestation_state()
        return node_id, node_key

    def clear_credentials(self) -> None:
        credentials_path = Path(self.settings.credentials_path)
        self.settings.node_id = None
        self.settings.node_key = None
        if credentials_path.exists():
            credentials_path.unlink()
        self.clear_attestation_state()

    def load_attestation_state(self) -> dict[str, Any] | None:
        state_path = Path(self.settings.attestation_state_path)
        if not state_path.exists():
            return None
        self.tighten_credentials_permissions(state_path)
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def persist_attestation_state(self, *, attestation_provider: str) -> None:
        if not self.settings.node_id:
            raise RuntimeError("node id is required before persisting attestation state")
        self.write_private_json(
            Path(self.settings.attestation_state_path),
            {
                "node_id": self.settings.node_id,
                "attestation_provider": attestation_provider,
                "status": "verified",
                "attested_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def clear_attestation_state(self) -> None:
        state_path = Path(self.settings.attestation_state_path)
        if state_path.exists():
            state_path.unlink()

    def write_recovery_note(self, message: str) -> None:
        note_path = Path(self.settings.recovery_note_path)
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(message + "\n", encoding="utf-8")

    def clear_recovery_note(self) -> None:
        note_path = Path(self.settings.recovery_note_path)
        if note_path.exists():
            note_path.unlink()
