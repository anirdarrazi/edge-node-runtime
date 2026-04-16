from __future__ import annotations

import time
import webbrowser
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .config import NodeAgentSettings, NodeClaimPollResult, NodeClaimSession
from .control_plane_store import NodeCredentialStore


class NodeBootstrapOrchestrator:
    """Owns the interactive and legacy node bootstrap flows."""

    def __init__(
        self,
        settings: NodeAgentSettings,
        credential_store: NodeCredentialStore,
        *,
        legacy_enroll: Callable[[], tuple[str, str]],
        create_claim_session: Callable[[], NodeClaimSession],
        poll_claim_session: Callable[[str, str], NodeClaimPollResult],
        persist_from_response: Callable[[dict[str, Any]], tuple[str, str]],
        terminal_available: Callable[[], bool],
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.settings = settings
        self.credential_store = credential_store
        self.legacy_enroll = legacy_enroll
        self.create_claim_session = create_claim_session
        self.poll_claim_session = poll_claim_session
        self.persist_from_response = persist_from_response
        self.terminal_available = terminal_available
        self.sleep = sleep or time.sleep

    @staticmethod
    def setup_ui_claim_message() -> str:
        return (
            "No stored node credentials were found. Open the setup UI and run Quick Start to claim this node. "
            "Use `node-agent bootstrap` only for direct terminal debugging."
        )

    @staticmethod
    def format_remaining_time(expires_at: str) -> str:
        remaining_seconds = max(
            0,
            int(datetime.fromisoformat(expires_at.replace("Z", "+00:00")).timestamp() - datetime.now(timezone.utc).timestamp()),
        )
        minutes, seconds = divmod(remaining_seconds, 60)
        if minutes >= 10:
            return f"{minutes} min remaining"
        if minutes > 0:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"

    @staticmethod
    def print_claim_instructions(claim: NodeClaimSession) -> None:
        print()
        print("AUTONOMOUSc Edge Node Claim")
        print("---------------------------")
        print("Open the approval URL in your browser, sign in as an existing operator, and claim this node.")
        print(f"Claim code: {claim.claim_code}")
        print(f"Approval URL: {claim.approval_url}")
        print(f"Claim expires at: {claim.expires_at}")
        try:
            opened = webbrowser.open(claim.approval_url, new=2)
        except Exception:
            opened = False
        if opened:
            print("A browser tab was opened for you. If it did not appear, copy the approval URL above.")
        else:
            print("If your browser does not open automatically, copy the approval URL above into a browser on this device.")
        print("Waiting for browser approval...")
        print()

    def has_credentials(self) -> bool:
        if self.settings.node_id and self.settings.node_key:
            return True
        return self.credential_store.load_credentials() is not None

    def require_credentials(self) -> tuple[str, str]:
        if self.settings.node_id and self.settings.node_key:
            return self.settings.node_id, self.settings.node_key
        persisted = self.credential_store.load_credentials()
        if persisted:
            return persisted
        raise RuntimeError(self.setup_ui_claim_message())

    def bootstrap(self, interactive: bool = True) -> tuple[str, str]:
        if self.settings.node_id and self.settings.node_key:
            return self.settings.node_id, self.settings.node_key
        persisted = self.credential_store.load_credentials()
        if persisted:
            return persisted
        if self.settings.operator_token:
            return self.legacy_enroll()
        if not interactive or not self.terminal_available():
            raise RuntimeError(self.setup_ui_claim_message())

        claim = self.create_claim_session()
        self.print_claim_instructions(claim)
        last_status: str | None = None
        last_remaining: str | None = None

        while True:
            result = self.poll_claim_session(claim.claim_id, claim.poll_token)
            if result.node_id and result.node_key:
                print("Claim approved. Storing node credentials and finishing bootstrap...")
                return self.persist_from_response(result.model_dump())
            if result.status == "consumed":
                raise RuntimeError(
                    "Node claim was consumed but did not return credentials. Start Quick Start again from the setup UI, "
                    "or rerun `node-agent bootstrap` if you are using the direct terminal flow."
                )
            if result.status == "expired":
                raise RuntimeError(
                    "Node claim expired before approval completed. Start Quick Start again from the setup UI, "
                    "or rerun `node-agent bootstrap` if you are using the direct terminal flow."
                )
            remaining = self.format_remaining_time(result.expires_at)
            if result.status != last_status or remaining != last_remaining:
                print(f"Status: waiting for operator login and claim approval. Time remaining: {remaining}.")
                last_status = result.status
                last_remaining = remaining
            self.sleep(claim.poll_interval_seconds)
