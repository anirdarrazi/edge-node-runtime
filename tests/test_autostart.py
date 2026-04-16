import plistlib
import subprocess
from pathlib import Path

import pytest

from node_agent.autostart import AutoStartManager


def completed(args: list[str], stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")


def test_windows_autostart_enable_and_disable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_created = False

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        nonlocal task_created
        if args[:2] == ["schtasks", "/Create"]:
            task_created = True
            return completed(args)
        if args[:2] == ["schtasks", "/Delete"]:
            task_created = False
            return completed(args)
        if args[:2] == ["schtasks", "/Query"]:
            if not task_created:
                raise RuntimeError("ERROR: The system cannot find the file specified.")
            return completed(args, stdout="Status: Ready\n")
        raise AssertionError(f"Unexpected command: {args}")

    launcher_path = tmp_path / "launcher.py"
    launcher_path.write_text("print('launcher')\n", encoding="utf-8")

    monkeypatch.setattr("node_agent.autostart.shutil.which", lambda name: name)
    manager = AutoStartManager(
        tmp_path,
        command_runner=runner,
        platform_name="nt",
        launcher_path=launcher_path,
        python_executable="C:\\Python311\\python.exe",
    )

    enabled = manager.enable()

    assert enabled["enabled"] is True
    assert enabled["label"] == "Enabled"
    assert manager.script_path.exists()
    script = manager.script_path.read_text(encoding="utf-8")
    assert "AUTONOMOUSC_RUNTIME_DIR" in script
    assert str(launcher_path) in script

    disabled = manager.disable()

    assert disabled["supported"] is True
    assert disabled["enabled"] is False
    assert disabled["label"] == "Disabled"


def test_macos_autostart_enable_and_disable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home_dir = tmp_path / "home"
    loaded = False

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        nonlocal loaded
        if args[:2] == ["launchctl", "unload"]:
            loaded = False
            return completed(args)
        if args[:2] == ["launchctl", "load"]:
            loaded = True
            return completed(args)
        raise AssertionError(f"Unexpected command: {args}")

    launcher_path = tmp_path / "launcher.py"
    launcher_path.write_text("print('launcher')\n", encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: home_dir)
    monkeypatch.setattr("node_agent.autostart.shutil.which", lambda name: f"/usr/bin/{name}" if name == "launchctl" else None)
    manager = AutoStartManager(
        tmp_path,
        command_runner=runner,
        platform_name="darwin",
        launcher_path=launcher_path,
        python_executable="/usr/bin/python3",
    )

    enabled = manager.enable()

    assert loaded is True
    assert enabled["enabled"] is True
    assert enabled["mechanism"] == "macos_launch_agent"
    assert manager.launch_agent_path.exists()
    with manager.launch_agent_path.open("rb") as handle:
        plist = plistlib.load(handle)
    assert plist["Label"] == manager.launchd_label()
    assert plist["ProgramArguments"] == ["/usr/bin/python3", str(launcher_path), "start"]
    assert plist["EnvironmentVariables"]["AUTONOMOUSC_RUNTIME_DIR"] == str(tmp_path.resolve())

    disabled = manager.disable()

    assert disabled["supported"] is True
    assert disabled["enabled"] is False
    assert not manager.launch_agent_path.exists()


def test_linux_autostart_enable_and_disable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home_dir = tmp_path / "home"
    enabled_state = False

    def runner(args: list[str], _cwd: Path) -> subprocess.CompletedProcess[str]:
        nonlocal enabled_state
        if args == ["systemctl", "--user", "daemon-reload"]:
            return completed(args)
        if args[:3] == ["systemctl", "--user", "enable"]:
            enabled_state = True
            return completed(args)
        if args[:3] == ["systemctl", "--user", "disable"]:
            enabled_state = False
            return completed(args)
        if args[:3] == ["systemctl", "--user", "is-enabled"]:
            if not enabled_state:
                raise RuntimeError("disabled")
            return completed(args, stdout="enabled\n")
        raise AssertionError(f"Unexpected command: {args}")

    launcher_path = tmp_path / "launcher.py"
    launcher_path.write_text("print('launcher')\n", encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: home_dir)
    monkeypatch.setattr("node_agent.autostart.shutil.which", lambda name: f"/usr/bin/{name}" if name == "systemctl" else None)
    manager = AutoStartManager(
        tmp_path,
        command_runner=runner,
        platform_name="linux",
        launcher_path=launcher_path,
        python_executable="/usr/bin/python3",
    )

    enabled = manager.enable()

    assert enabled["enabled"] is True
    assert enabled["mechanism"] == "systemd_user_service"
    assert manager.systemd_unit_path.exists()
    unit = manager.systemd_unit_path.read_text(encoding="utf-8")
    assert "AUTONOMOUSc Edge Node Service" in unit
    assert "AUTONOMOUSC_RUNTIME_DIR=" in unit
    assert "launcher.py" in unit

    disabled = manager.disable()

    assert disabled["supported"] is True
    assert disabled["enabled"] is False
    assert not manager.systemd_unit_path.exists()


def test_autostart_status_reports_unsupported_platform(tmp_path: Path) -> None:
    manager = AutoStartManager(tmp_path, platform_name="freebsd")

    status = manager.status()

    assert status["supported"] is False
    assert status["enabled"] is False
    assert status["label"] == "Unavailable"
