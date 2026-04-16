from __future__ import annotations

import hashlib
import os
import plistlib
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
        self.system_name = sys.platform if platform_name is None else platform_name
        self.launcher_path = (launcher_path or Path(__file__).with_name("launcher.py")).resolve()
        self.python_executable = python_executable or self._default_python_executable()
        self.service_dir = self.runtime_dir / "data" / "service"
        self.script_path = self.service_dir / "autostart-launch.ps1"
        self.launch_agent_path = (
            Path.home()
            / "Library"
            / "LaunchAgents"
            / f"{self.launchd_label()}.plist"
        )
        self.systemd_dir = Path.home() / ".config" / "systemd" / "user"
        self.systemd_unit_path = self.systemd_dir / self.systemd_unit_name()

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

    def launchd_label(self) -> str:
        suffix = hashlib.sha1(str(self.runtime_dir).encode("utf-8")).hexdigest()[:8]
        return f"com.autonomousc.node.{suffix}"

    def systemd_unit_name(self) -> str:
        suffix = hashlib.sha1(str(self.runtime_dir).encode("utf-8")).hexdigest()[:8]
        return f"autonomousc-node-{suffix}.service"

    def _platform_key(self) -> str:
        if self.platform_name == "nt":
            return "windows"
        platform_name = self.system_name or self.platform_name
        if platform_name == "darwin":
            return "macos"
        if platform_name == "linux" or (platform_name == "posix" and sys.platform.startswith("linux")):
            return "linux"
        return "unsupported"

    def _powershell_command(self) -> str:
        return shutil.which("powershell.exe") or shutil.which("powershell") or "powershell.exe"

    def _windows_supported(self) -> tuple[bool, str]:
        if self._platform_key() != "windows":
            return False, "Windows Task Scheduler is only available on Windows."
        if shutil.which("schtasks") is None:
            return False, "Windows Task Scheduler is not available on this machine."
        return True, "Automatic start is available and can launch the local node service when you sign in."

    def _macos_supported(self) -> tuple[bool, str]:
        if self._platform_key() != "macos":
            return False, "macOS LaunchAgents are only available on macOS."
        if shutil.which("launchctl") is None:
            return False, "macOS launchctl is not available on this machine."
        return True, "Automatic start can use a macOS LaunchAgent to reopen the local node service at sign-in."

    def _linux_supported(self) -> tuple[bool, str]:
        if self._platform_key() != "linux":
            return False, "Linux user services are only available on Linux."
        if shutil.which("systemctl") is None:
            return False, "Linux user systemd is not available on this machine."
        return True, "Automatic start can use a Linux user systemd service to reopen the local node service at sign-in."

    def _unsupported_status(self) -> dict[str, Any]:
        return {
            "supported": False,
            "enabled": False,
            "label": "Unavailable",
            "detail": "Automatic start is not available on this operating system yet.",
            "mechanism": None,
            "task_name": self.task_name(),
        }

    def status(self) -> dict[str, Any]:
        platform_key = self._platform_key()
        if platform_key == "macos":
            return self._macos_status()
        if platform_key == "linux":
            return self._linux_status()
        if platform_key != "windows":
            return self._unsupported_status()

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

    def _macos_status(self) -> dict[str, Any]:
        supported, detail = self._macos_supported()
        if not supported:
            return {
                "supported": False,
                "enabled": False,
                "label": "Unavailable",
                "detail": detail,
                "mechanism": "macos_launch_agent",
                "task_name": self.launchd_label(),
            }
        enabled = self.launch_agent_path.exists()
        return {
            "supported": True,
            "enabled": enabled,
            "label": "Enabled" if enabled else "Disabled",
            "detail": (
                "macOS launch-on-sign-in is enabled for the local node service."
                if enabled
                else "macOS launch-on-sign-in is available for this node but is not enabled yet."
            ),
            "mechanism": "macos_launch_agent",
            "task_name": self.launchd_label(),
        }

    def _linux_status(self) -> dict[str, Any]:
        supported, detail = self._linux_supported()
        if not supported:
            return {
                "supported": False,
                "enabled": False,
                "label": "Unavailable",
                "detail": detail,
                "mechanism": "systemd_user_service",
                "task_name": self.systemd_unit_name(),
            }
        try:
            self.command_runner(
                ["systemctl", "--user", "is-enabled", self.systemd_unit_name()],
                self.runtime_dir,
            )
            enabled = True
        except RuntimeError:
            enabled = False
        return {
            "supported": True,
            "enabled": enabled,
            "label": "Enabled" if enabled else "Disabled",
            "detail": (
                "Linux user systemd is set to start the local node service when you sign in."
                if enabled
                else "Linux user systemd is available for this node but is not enabled yet."
            ),
            "mechanism": "systemd_user_service",
            "task_name": self.systemd_unit_name(),
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

    def _write_macos_launch_agent(self) -> None:
        self.launch_agent_path.parent.mkdir(parents=True, exist_ok=True)
        plist = {
            "Label": self.launchd_label(),
            "ProgramArguments": [self.python_executable, str(self.launcher_path), "start"],
            "WorkingDirectory": str(self.runtime_dir),
            "EnvironmentVariables": {RUNTIME_DIR_ENV: str(self.runtime_dir)},
            "RunAtLoad": True,
            "KeepAlive": False,
            "StandardOutPath": str(self.service_dir / "autostart.out.log"),
            "StandardErrorPath": str(self.service_dir / "autostart.err.log"),
        }
        self.service_dir.mkdir(parents=True, exist_ok=True)
        with self.launch_agent_path.open("wb") as handle:
            plistlib.dump(plist, handle)
        tighten_private_path(self.launch_agent_path)

    def _systemd_quote(self, value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def _write_linux_unit(self) -> None:
        self.systemd_dir.mkdir(parents=True, exist_ok=True)
        content = "\n".join(
            [
                "[Unit]",
                "Description=AUTONOMOUSc Edge Node Service",
                "After=network-online.target",
                "",
                "[Service]",
                "Type=simple",
                f"WorkingDirectory={self._systemd_quote(str(self.runtime_dir))}",
                f"Environment={self._systemd_quote(f'{RUNTIME_DIR_ENV}={self.runtime_dir}')}",
                f"ExecStart={self._systemd_quote(self.python_executable)} {self._systemd_quote(str(self.launcher_path))} start",
                "Restart=on-failure",
                "RestartSec=10",
                "",
                "[Install]",
                "WantedBy=default.target",
                "",
            ]
        )
        self.systemd_unit_path.write_text(content, encoding="utf-8")
        tighten_private_path(self.systemd_unit_path)

    def enable(self) -> dict[str, Any]:
        platform_key = self._platform_key()
        if platform_key == "macos":
            supported, detail = self._macos_supported()
            if not supported:
                raise RuntimeError(detail)
            self._write_macos_launch_agent()
            try:
                self.command_runner(["launchctl", "unload", "-w", str(self.launch_agent_path)], self.runtime_dir)
            except RuntimeError:
                pass
            self.command_runner(["launchctl", "load", "-w", str(self.launch_agent_path)], self.runtime_dir)
            return self.status()
        if platform_key == "linux":
            supported, detail = self._linux_supported()
            if not supported:
                raise RuntimeError(detail)
            self._write_linux_unit()
            self.command_runner(["systemctl", "--user", "daemon-reload"], self.runtime_dir)
            self.command_runner(["systemctl", "--user", "enable", "--now", self.systemd_unit_name()], self.runtime_dir)
            return self.status()

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
        platform_key = self._platform_key()
        if platform_key == "macos":
            supported, detail = self._macos_supported()
            if not supported:
                raise RuntimeError(detail)
            try:
                self.command_runner(["launchctl", "unload", "-w", str(self.launch_agent_path)], self.runtime_dir)
            except RuntimeError:
                pass
            if self.launch_agent_path.exists():
                self.launch_agent_path.unlink()
            return self.status()
        if platform_key == "linux":
            supported, detail = self._linux_supported()
            if not supported:
                raise RuntimeError(detail)
            try:
                self.command_runner(["systemctl", "--user", "disable", "--now", self.systemd_unit_name()], self.runtime_dir)
            except RuntimeError:
                pass
            if self.systemd_unit_path.exists():
                self.systemd_unit_path.unlink()
            self.command_runner(["systemctl", "--user", "daemon-reload"], self.runtime_dir)
            return self.status()

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
