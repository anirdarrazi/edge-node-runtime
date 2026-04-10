import subprocess
from pathlib import Path

import pytest

from node_agent.desktop_launcher import DesktopLauncherManager


def completed(args: list[str], stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")


def test_windows_desktop_launcher_enable_and_disable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launcher_path = tmp_path / "launcher.py"
    launcher_path.write_text("print('launcher')\n", encoding="utf-8")
    desktop_dir = tmp_path / "Desktop"
    desktop_dir.mkdir()

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        if args[0].lower().startswith("powershell"):
            shortcut = desktop_dir / "AUTONOMOUSc Edge Node.lnk"
            shortcut.write_text("shortcut", encoding="utf-8")
            return completed(args)
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr("node_agent.desktop_launcher.shutil.which", lambda name: name)
    manager = DesktopLauncherManager(
        tmp_path,
        command_runner=runner,
        platform_name="nt",
        launcher_path=launcher_path,
        python_executable="C:\\Python311\\python.exe",
        desktop_dir=desktop_dir,
    )

    enabled = manager.enable()

    assert enabled["enabled"] is True
    assert enabled["label"] == "Installed"
    assert manager.script_path.exists()
    assert manager.shortcut_path.exists()
    script = manager.script_path.read_text(encoding="utf-8")
    assert "AUTONOMOUSC_RUNTIME_DIR" in script
    assert str(launcher_path) in script

    disabled = manager.disable()

    assert disabled["supported"] is True
    assert disabled["enabled"] is False
    assert disabled["label"] == "Not installed"


def test_desktop_launcher_status_reports_unsupported_platform(tmp_path: Path) -> None:
    manager = DesktopLauncherManager(tmp_path, platform_name="posix")

    status = manager.status()

    assert status["supported"] is False
    assert status["enabled"] is False
    assert status["label"] == "Unavailable"
