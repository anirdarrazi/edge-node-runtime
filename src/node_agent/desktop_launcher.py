from __future__ import annotations

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


class DesktopLauncherManager:
    def __init__(
        self,
        runtime_dir: Path,
        *,
        command_runner: CommandRunner = run_command,
        platform_name: str | None = None,
        launcher_path: Path | None = None,
        python_executable: str | None = None,
        desktop_dir: Path | None = None,
        shortcut_name: str = "AUTONOMOUSc Edge Node App.lnk",
    ) -> None:
        self.runtime_dir = runtime_dir.resolve()
        self.command_runner = command_runner
        self.platform_name = os.name if platform_name is None else platform_name
        self.launcher_path = (launcher_path or Path(__file__).with_name("launcher.py")).resolve()
        self.python_executable = python_executable or self._default_python_executable()
        self.desktop_dir = (
            desktop_dir.resolve()
            if desktop_dir is not None
            else Path.home().joinpath("Desktop").resolve()
        )
        self.shortcut_name = shortcut_name
        self.service_dir = self.runtime_dir / "data" / "service"
        self.script_path = self.service_dir / "desktop-launch.ps1"
        self.shortcut_path = self.desktop_dir / shortcut_name

    def _default_python_executable(self) -> str:
        candidate = Path(sys.executable)
        if self.platform_name == "nt" and candidate.name.lower() == "python.exe":
            pythonw = candidate.with_name("pythonw.exe")
            if pythonw.exists():
                return str(pythonw)
        return str(candidate)

    def _powershell_command(self) -> str:
        return shutil.which("powershell.exe") or shutil.which("powershell") or "powershell.exe"

    def _windows_supported(self) -> tuple[bool, str]:
        if self.platform_name != "nt":
            return False, "Desktop launcher installation is currently available on Windows only."
        if not self.desktop_dir.exists():
            return False, "The Desktop folder could not be found for this account."
        if shutil.which("powershell.exe") is None and shutil.which("powershell") is None:
            return False, "PowerShell is required to create a desktop launcher on Windows."
        return True, "A desktop launcher can reopen the local node app with one click."

    def status(self) -> dict[str, Any]:
        supported, detail = self._windows_supported()
        if not supported:
            return {
                "supported": False,
                "enabled": False,
                "label": "Unavailable",
                "detail": detail,
                "path": str(self.shortcut_path),
            }
        enabled = self.shortcut_path.exists()
        return {
            "supported": True,
            "enabled": enabled,
            "label": "Installed" if enabled else "Not installed",
            "detail": (
                f"The desktop launcher is available at {self.shortcut_path}."
                if enabled
                else "Install a desktop launcher so this node app can be reopened with one click."
            ),
            "path": str(self.shortcut_path),
        }

    def _write_windows_script(self) -> None:
        self.service_dir.mkdir(parents=True, exist_ok=True)
        content = "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                f"$env:{RUNTIME_DIR_ENV} = '{powershell_literal(str(self.runtime_dir))}'",
                f"Set-Location -LiteralPath '{powershell_literal(str(self.runtime_dir))}'",
                f"& '{powershell_literal(self.python_executable)}' '{powershell_literal(str(self.launcher_path))}' start --open",
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
        self.desktop_dir.mkdir(parents=True, exist_ok=True)
        arguments = (
            f'-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{self.script_path}"'
        )
        command = "\n".join(
            [
                "$WshShell = New-Object -ComObject WScript.Shell",
                f"$Shortcut = $WshShell.CreateShortcut('{powershell_literal(str(self.shortcut_path))}')",
                f"$Shortcut.TargetPath = '{powershell_literal(self._powershell_command())}'",
                f"$Shortcut.Arguments = '{powershell_literal(arguments)}'",
                f"$Shortcut.WorkingDirectory = '{powershell_literal(str(self.runtime_dir))}'",
                "$Shortcut.IconLocation = 'shell32.dll,220'",
                "$Shortcut.Description = 'Open AUTONOMOUSc Edge Node App'",
                "$Shortcut.Save()",
            ]
        )
        self.command_runner(
            [self._powershell_command(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            self.runtime_dir,
        )
        if not self.shortcut_path.exists():
            raise RuntimeError("PowerShell did not create the desktop launcher.")
        return self.status()

    def ensure_enabled(self) -> dict[str, Any]:
        current = self.status()
        if not current.get("supported") or current.get("enabled"):
            return current
        return self.enable()

    def disable(self) -> dict[str, Any]:
        if self.shortcut_path.exists():
            self.shortcut_path.unlink()
        if self.script_path.exists():
            self.script_path.unlink()
        return self.status()
