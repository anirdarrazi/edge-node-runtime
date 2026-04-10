from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .local_api_security import tighten_private_path
from .runtime_layout import RUNTIME_DIR_ENV


CommandRunner = Callable[[list[str], Path], subprocess.CompletedProcess[str]]


def run_command(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"{args[0]} is not installed or is not available on PATH.") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"Command failed: {' '.join(args)}"
        raise RuntimeError(detail)
    return completed


def powershell_literal(value: str) -> str:
    return value.replace("'", "''")


class AutoStartManager:
    def __init__(
        self,
        runtime_dir: Path,
        *,
        command_runner: CommandRunner = run_command,
        platform_name: str | None = None,
        launcher_path: Path | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.runtime_dir = runtime_dir.resolve()
        self.command_runner = command_runner
        self.platform_name = os.name if platform_name is None else platform_name
        self.launcher_path = (launcher_path or Path(__file__).with_name("launcher.py")).resolve()
        self.python_executable = python_executable or self._default_python_executable()
        self.service_dir = self.runtime_dir / "data" / "service"
        self.script_path = self.service_dir / "autostart-launch.ps1"

    def _default_python_executable(self) -> str:
        candidate = Path(sys.executable)
        if self.platform_name == "nt" and candidate.name.lower() == "python.exe":
            pythonw = candidate.with_name("pythonw.exe")
            if pythonw.exists():
                return str(pythonw)
        return str(candidate)

    def task_name(self) -> str:
        suffix = hashlib.sha1(str(self.runtime_dir).encode("utf-8")).hexdigest()[:8]
        return f"AUTONOMOUSc-Node-Service-{suffix}"

    def _powershell_command(self) -> str:
        return shutil.which("powershell.exe") or shutil.which("powershell") or "powershell.exe"

    def _windows_supported(self) -> tuple[bool, str]:
        if self.platform_name != "nt":
            return False, "Automatic start is currently available on Windows only."
        if shutil.which("schtasks") is None:
            return False, "Windows Task Scheduler is not available on this machine."
        return True, "Automatic start is available and can launch the local node service when you sign in."

    def status(self) -> dict[str, Any]:
        supported, detail = self._windows_supported()
        if not supported:
            return {
                "supported": False,
                "enabled": False,
                "label": "Unavailable",
                "detail": detail,
                "mechanism": None,
                "task_name": self.task_name(),
            }
        try:
            completed = self.command_runner(
                ["schtasks", "/Query", "/TN", self.task_name(), "/FO", "LIST", "/V"],
                self.runtime_dir,
            )
            status_line = next(
                (line.split(":", 1)[1].strip() for line in completed.stdout.splitlines() if line.lower().startswith("status:")),
                "Ready",
            )
            return {
                "supported": True,
                "enabled": True,
                "label": "Enabled",
                "detail": f"Windows Task Scheduler is set to start the local node service when you sign in. Current task status: {status_line}.",
                "mechanism": "windows_task_scheduler",
                "task_name": self.task_name(),
            }
        except RuntimeError:
            return {
                "supported": True,
                "enabled": False,
                "label": "Disabled",
                "detail": "Automatic start is available for this node but is not enabled yet.",
                "mechanism": "windows_task_scheduler",
                "task_name": self.task_name(),
            }

    def _write_windows_script(self) -> None:
        self.service_dir.mkdir(parents=True, exist_ok=True)
        content = "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                f"$env:{RUNTIME_DIR_ENV} = '{powershell_literal(str(self.runtime_dir))}'",
                f"Set-Location -LiteralPath '{powershell_literal(str(self.runtime_dir))}'",
                f"& '{powershell_literal(self.python_executable)}' '{powershell_literal(str(self.launcher_path))}' start",
                "",
            ]
        )
        self.script_path.write_text(content, encoding="utf-8")
        tighten_private_path(self.script_path)

    def enable(self) -> dict[str, Any]:
        supported, detail = self._windows_supported()
        if not supported:
            raise RuntimeError(detail)
        self._write_windows_script()
        action = (
            f'{self._powershell_command()} -NoProfile -WindowStyle Hidden '
            f'-ExecutionPolicy Bypass -File "{self.script_path}"'
        )
        self.command_runner(
            ["schtasks", "/Create", "/F", "/SC", "ONLOGON", "/TN", self.task_name(), "/TR", action],
            self.runtime_dir,
        )
        return self.status()

    def ensure_enabled(self) -> dict[str, Any]:
        current = self.status()
        if not current.get("supported") or current.get("enabled"):
            return current
        return self.enable()

    def disable(self) -> dict[str, Any]:
        supported, detail = self._windows_supported()
        if not supported:
            raise RuntimeError(detail)
        try:
            self.command_runner(["schtasks", "/Delete", "/F", "/TN", self.task_name()], self.runtime_dir)
        except RuntimeError:
            pass
        if self.script_path.exists():
            self.script_path.unlink()
        return self.status()
