"""Running the daemon as a background service.

A sidecar you have to remember to start is not a sidecar. This writes the
platform's own service definition — a launchd agent on macOS, a systemd user
unit on Linux — so the daemon comes up at login and restarts if it dies.

Deliberately not a wrapper around a process manager we invent: `launchctl` and
`systemctl` already do this correctly, including restart policy and logging,
and a user who knows those tools should be able to inspect and override what
we wrote with them.

The API key is never written into the service definition. A launchd plist is
world-readable, and an environment variable there also breaks the moment
someone edits the file. `llm-sidecar config key --save` puts it in one known
place instead.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .types import SidecarError

LABEL = "com.llm-sidecar.daemon"

# macOS gates these behind TCC. A launchd agent gets no access, so a venv
# living in one cannot be launched as a service — Python dies before it starts
# on `PermissionError: pyvenv.cfg`, and KeepAlive then restarts it forever.
# Found the hard way: a repo cloned into ~/Documents crash-looped on install.
PROTECTED_DIRS = ("Documents", "Desktop", "Downloads", "Library/Mobile Documents")

PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
UNIT = Path.home() / ".config" / "systemd" / "user" / "llm-sidecar.service"
LOG = Path.home() / "Library" / "Logs" / "llm-sidecar.log"


def _executable() -> str:
    """The llm-sidecar entry point to run.

    Prefers the console script next to the running interpreter, which is the
    one the user actually installed; falls back to whatever is on PATH."""
    candidate = Path(sys.executable).parent / "llm-sidecar"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("llm-sidecar")
    if found:
        return found
    raise SidecarError(
        "Could not find the llm-sidecar executable. Reinstall with "
        "`pipx install llm-sidecar`, or run the daemon manually."
    )


def _protected_location(exe: str) -> str | None:
    """The protected folder this executable sits in, if any."""
    if platform.system() != "Darwin":
        return None
    try:
        rel = Path(exe).resolve().relative_to(Path.home())
    except ValueError:
        return None                     # outside the home directory: fine
    parts = rel.parts
    for guarded in PROTECTED_DIRS:
        g = tuple(guarded.split("/"))
        if parts[:len(g)] == g:
            return guarded
    return None


def _plist(exe: str, port: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" \
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{exe}</string>
    <string>serve</string>
    <string>--port</string>
    <string>{port}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <!-- Without this, a process that fails at startup is restarted as fast as
       launchd can manage. Ten seconds keeps a genuine crash visible in the
       log rather than drowning it. -->
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>{LOG}</string>
  <key>StandardErrorPath</key><string>{LOG}</string>
</dict>
</plist>
"""


def _unit(exe: str, port: int) -> str:
    return f"""[Unit]
Description=llm-sidecar
After=network.target

[Service]
ExecStart={exe} serve --port {port}
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""


def install(port: int = 4001) -> dict:
    """Write and start the service. Returns what it did."""
    system = platform.system()
    exe = _executable()

    guarded = _protected_location(exe)
    if guarded:
        raise SidecarError(
            f"This copy of llm-sidecar lives in ~/{guarded}, which macOS keeps "
            f"behind a privacy prompt that background services never get.\n"
            f"  {exe}\n\n"
            "A launchd agent pointed at it dies on startup and is restarted "
            "forever. Install it somewhere unprotected instead:\n\n"
            "  pipx install llm-sidecar\n"
            "  llm-sidecar service install\n\n"
            "pipx keeps its environments in ~/Library/Application Support, "
            "which services can read."
        )

    if system == "Darwin":
        PLIST.parent.mkdir(parents=True, exist_ok=True)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        PLIST.write_text(_plist(exe, port), encoding="utf-8")
        # Unload first so a re-install picks up the new file rather than
        # silently keeping the old definition running.
        subprocess.run(["launchctl", "unload", str(PLIST)],
                       capture_output=True, timeout=20)
        r = subprocess.run(["launchctl", "load", str(PLIST)],
                           capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            raise SidecarError(f"launchctl load failed: {(r.stderr or '').strip()}")
        return {"path": str(PLIST), "manager": "launchd", "log": str(LOG), "port": port}

    if system == "Linux":
        UNIT.parent.mkdir(parents=True, exist_ok=True)
        UNIT.write_text(_unit(exe, port), encoding="utf-8")
        subprocess.run(["systemctl", "--user", "daemon-reload"],
                       capture_output=True, timeout=20)
        r = subprocess.run(["systemctl", "--user", "enable", "--now", "llm-sidecar"],
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise SidecarError(f"systemctl failed: {(r.stderr or '').strip()}")
        return {"path": str(UNIT), "manager": "systemd", "port": port,
                "note": "Run `loginctl enable-linger $USER` to keep it running after logout."}

    raise SidecarError(
        f"No service integration for {system}. Run `llm-sidecar serve` yourself, "
        "or use your platform's startup mechanism."
    )


def uninstall() -> dict:
    system = platform.system()
    if system == "Darwin":
        subprocess.run(["launchctl", "unload", str(PLIST)], capture_output=True, timeout=20)
        existed = PLIST.exists()
        PLIST.unlink(missing_ok=True)
        return {"removed": existed, "path": str(PLIST)}
    if system == "Linux":
        subprocess.run(["systemctl", "--user", "disable", "--now", "llm-sidecar"],
                       capture_output=True, timeout=30)
        existed = UNIT.exists()
        UNIT.unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, timeout=20)
        return {"removed": existed, "path": str(UNIT)}
    raise SidecarError(f"No service integration for {system}.")


def status() -> dict:
    """Installed? Running? Answering? Those are three different questions."""
    system = platform.system()
    path = PLIST if system == "Darwin" else UNIT
    info = {"platform": system, "path": str(path), "installed": path.exists(),
            "running": False, "log": str(LOG) if system == "Darwin" else "journalctl --user -u llm-sidecar"}

    try:
        if system == "Darwin":
            r = subprocess.run(["launchctl", "list"], capture_output=True, text=True, timeout=20)
            info["running"] = LABEL in (r.stdout or "")
        elif system == "Linux":
            r = subprocess.run(["systemctl", "--user", "is-active", "llm-sidecar"],
                               capture_output=True, text=True, timeout=20)
            info["running"] = (r.stdout or "").strip() == "active"
    except (OSError, subprocess.SubprocessError):
        pass
    return info
