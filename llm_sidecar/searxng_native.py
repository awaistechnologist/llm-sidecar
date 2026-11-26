"""SearXNG run natively, as a login service the sidecar owns.

Until 0.11.0 this was a Docker container. On a Mac a container needs a Linux
VM, and Docker Desktop's VM took ~5 GiB of memory, plus ~1.8 GiB for its own
apps, to host a search engine that uses 172 MiB. On a machine whose image
models are sized to the last gigabyte that is not a rounding error. It also
meant search depended on another app being started: every reboot without
Docker at login quietly turned search off.

SearXNG is an ordinary Python web app. Natively it is a git checkout, a venv
and the same `granian` server the official container runs, under launchd
(or a systemd user unit), using ~150 MiB and starting at login.

Layout, all under the sidecar's data directory:

    searxng/src/          git checkout of github.com/searxng/searxng
    searxng/venv/         its own venv (never the sidecar's: their pins differ)
    searxng/sidecar.json  what was installed, from which commit, with which Python

The settings file lives in the config directory, ~/.config/llm-sidecar/
searxng/settings.yml, and is yours to edit once written. Without it the API
returns 403: SearXNG ships with JSON output off, which reads as
"unavailable" and silently falls back to DuckDuckGo.

The cost of running it natively is updates: SearXNG scrapes other sites, and
when they change, engines break until SearXNG ships a fix. A container hid
that behind `docker pull`. Here it is `llm-sidecar searxng update`, which rolls
back if the new version does not answer, and `status` says when the checkout
is old or engines are failing.
"""

from __future__ import annotations

import json
import logging
import platform
import secrets
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from .config import CONFIG_DIR, Config
from .types import SidecarError

logger = logging.getLogger("llm_sidecar.searxng")

REPO = "https://github.com/searxng/searxng"
LABEL = "com.llm-sidecar.searxng"
UNIT_NAME = "llm-sidecar-searxng"
DEFAULT_PORT = 8888
STALE_DAYS = 30
STARTUP_TIMEOUT = 60

# Newest first. SearXNG needs >= 3.10; the newest CPython is tried last
# because its dependencies (lxml, curl_cffi) can lag on wheels for it.
PYTHONS = ("python3.13", "python3.12", "python3.11", "python3.10")


# ── paths ─────────────────────────────────────────────────────────────────


def home() -> Path:
    from .ledger import DATA_DIR
    return DATA_DIR / "searxng"


def src_dir() -> Path:
    return home() / "src"


def venv_dir() -> Path:
    return home() / "venv"


def marker() -> Path:
    return home() / "sidecar.json"


SETTINGS_TEMPLATE = Path(__file__).parent / "deploy" / "searxng" / "settings.yml"


def settings_path() -> Path:
    return CONFIG_DIR / "searxng" / "settings.yml"


def write_settings(force: bool = False) -> Path:
    """settings.yml with JSON on and a secret generated for this machine.
    Never overwritten once it exists, unless `force`."""
    dst = settings_path()
    if dst.exists() and not force:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = SETTINGS_TEMPLATE.read_text(encoding="utf-8")
    # A shared secret key across every install would be worse than no key.
    dst.write_text(text.replace("GENERATED_ON_FIRST_START", secrets.token_hex(32)),
                   encoding="utf-8")
    return dst


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / f"{UNIT_NAME}.service"


def log_path() -> Path:
    return Path.home() / "Library" / "Logs" / "llm-sidecar-searxng.log"


# ── small helpers (patched in tests) ──────────────────────────────────────


def _sh(cmd: list[str], *, cwd: Path | None = None, timeout: int = 600,
        check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip()[-1500:]
        raise SidecarError(f"`{' '.join(cmd[:3])} …` failed:\n{tail}")
    return r


def _port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _answering(url: str, timeout: int = 0) -> bool:
    """Answers a JSON query — not merely reachable, which SearXNG is long
    before (and even when it never will be) serving the JSON we need."""
    if not timeout:
        return _probe(url)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _probe(url):
            return True
        time.sleep(2)
    return False


def _probe(url: str) -> bool:
    import httpx
    try:
        r = httpx.get(f"{url.rstrip('/')}/search",
                      params={"q": "ping", "format": "json"}, timeout=8.0)
        return r.status_code == 200 and isinstance(r.json().get("results"), list)
    except Exception:
        return False


def find_python() -> str:
    """An interpreter SearXNG supports, preferring ones with wheels for
    everything it needs."""
    for name in PYTHONS:
        exe = shutil.which(name)
        if exe:
            return exe
    if sys.version_info >= (3, 10):
        return sys.executable
    raise SidecarError("SearXNG needs Python 3.10 or newer, and none was found.")


def info() -> dict | None:
    try:
        return json.loads(marker().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def installed() -> bool:
    return info() is not None


def _write_marker(**fields) -> None:
    data = info() or {}
    data.update(fields)
    marker().write_text(json.dumps(data, indent=2), encoding="utf-8")


def _commit() -> tuple[str, str]:
    r = _sh(["git", "-C", str(src_dir()), "log", "-1", "--format=%h %cI"], check=False)
    parts = (r.stdout or "").strip().split(" ", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else ("", "")


# ── the service definition ────────────────────────────────────────────────


def _env(port: int) -> dict[str, str]:
    """The same granian settings the official container uses, bound to
    loopback: the limiter is off, so on the network this would be a free
    search proxy for anyone."""
    return {
        "SEARXNG_SETTINGS_PATH": str(settings_path()),
        "PYTHONPATH": str(src_dir()),
        "GRANIAN_INTERFACE": "wsgi",
        "GRANIAN_HOST": "127.0.0.1",
        "GRANIAN_PORT": str(port),
        "GRANIAN_WEBSOCKETS": "false",
        "GRANIAN_BLOCKING_THREADS": "4",
        "GRANIAN_PROCESS_NAME": "searxng",
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    }


def _plist(port: int) -> str:
    from xml.sax.saxutils import escape
    env = "\n".join(f"    <key>{k}</key><string>{escape(v)}</string>"
                    for k, v in _env(port).items())
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" \
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{escape(str(venv_dir() / "bin" / "granian"))}</string>
    <string>searx.webapp:app</string>
  </array>
  <key>WorkingDirectory</key><string>{escape(str(src_dir()))}</string>
  <key>EnvironmentVariables</key>
  <dict>
{env}
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>{escape(str(log_path()))}</string>
  <key>StandardErrorPath</key><string>{escape(str(log_path()))}</string>
</dict>
</plist>
"""


def _unit(port: int) -> str:
    env = "\n".join(f'Environment="{k}={v}"' for k, v in _env(port).items())
    return f"""[Unit]
Description=SearXNG for llm-sidecar
After=network.target

[Service]
WorkingDirectory={src_dir()}
{env}
ExecStart={venv_dir() / "bin" / "granian"} searx.webapp:app
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""


def _start_service(port: int) -> None:
    system = platform.system()
    if system == "Darwin":
        p = plist_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        log_path().parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_plist(port), encoding="utf-8")
        _sh(["launchctl", "unload", str(p)], check=False, timeout=20)
        _sh(["launchctl", "load", str(p)], timeout=20)
        return
    if system == "Linux":
        u = unit_path()
        u.parent.mkdir(parents=True, exist_ok=True)
        u.write_text(_unit(port), encoding="utf-8")
        _sh(["systemctl", "--user", "daemon-reload"], check=False, timeout=20)
        _sh(["systemctl", "--user", "enable", "--now", UNIT_NAME], timeout=30)
        return
    raise SidecarError(f"No service integration for {system}. Run SearXNG yourself and "
                       f"set SEARXNG_URL.")


def restart() -> None:
    system = platform.system()
    if system == "Darwin":
        import os
        _sh(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{LABEL}"], timeout=20)
    elif system == "Linux":
        _sh(["systemctl", "--user", "restart", UNIT_NAME], timeout=30)


def service_running() -> bool:
    system = platform.system()
    try:
        if system == "Darwin":
            r = _sh(["launchctl", "list"], check=False, timeout=20)
            return LABEL in (r.stdout or "")
        if system == "Linux":
            r = _sh(["systemctl", "--user", "is-active", UNIT_NAME], check=False, timeout=20)
            return (r.stdout or "").strip() == "active"
    except (OSError, subprocess.SubprocessError):
        pass
    return False


# ── install / update / uninstall ──────────────────────────────────────────


def _pip_install() -> None:
    pip = str(venv_dir() / "bin" / "pip")
    _sh([pip, "install", "-q", "-U", "pip", "setuptools", "wheel"])
    _sh([pip, "install", "-q", "-r", str(src_dir() / "requirements.txt"),
         "-r", str(src_dir() / "requirements-server.txt")])


def install(config: Config, port: int = DEFAULT_PORT,
            out: Callable[[str], None] = print) -> dict:
    """Set up SearXNG natively and start it at login. Safe to re-run."""
    if not shutil.which("git"):
        raise SidecarError("Installing SearXNG natively needs git. On a Mac: "
                           "`xcode-select --install`.")

    url = f"http://localhost:{port}"
    if _port_busy(port) and not service_running():
        raise SidecarError(f"Port {port} is in use by something else (an old SearXNG "
                           f"container?). Free it, or pass --port and set SEARXNG_URL "
                           f"to match.")

    home().mkdir(parents=True, exist_ok=True)
    if not (src_dir() / ".git").exists():
        out("Downloading SearXNG…")
        shutil.rmtree(src_dir(), ignore_errors=True)
        _sh(["git", "clone", "--depth", "1", REPO, str(src_dir())])

    python = (info() or {}).get("python") or find_python()
    if not (venv_dir() / "bin" / "python").exists():
        out(f"Creating its environment ({Path(python).name})…")
        _sh([python, "-m", "venv", str(venv_dir())])
    out("Installing its dependencies (about a minute the first time)…")
    _pip_install()

    write_settings()
    commit, date = _commit()
    _write_marker(port=port, python=python, commit=commit, commit_date=date,
                  installed_at=time.strftime("%Y-%m-%dT%H:%M:%S"))

    out("Starting it as a login service…")
    _start_service(port)
    if not _answering(url, timeout=STARTUP_TIMEOUT):
        raise SidecarError(f"SearXNG was installed but {url} did not answer within "
                           f"{STARTUP_TIMEOUT}s. Its log: {log_path()}")

    hint = None
    if config.searxng_url.rstrip("/") != url:
        hint = f"Set SEARXNG_URL={url} so search finds it."
    return {"url": url, "commit": commit, "commit_date": date, "home": str(home()),
            "hint": hint}


def update(out: Callable[[str], None] = print) -> dict:
    """Pull the latest SearXNG, and roll back if it does not answer."""
    data = info()
    if data is None:
        raise SidecarError("SearXNG is not installed natively. Run `llm-sidecar searxng install`.")
    port = data.get("port", DEFAULT_PORT)
    url = f"http://localhost:{port}"
    before, _ = _commit()

    out("Fetching the latest SearXNG…")
    _sh(["git", "-C", str(src_dir()), "fetch", "--depth", "1", "origin", "HEAD"])
    _sh(["git", "-C", str(src_dir()), "reset", "--hard", "FETCH_HEAD"])
    after, date = _commit()
    if after == before:
        return {"updated": False, "commit": after, "commit_date": date}

    out(f"{before} → {after}; installing dependencies…")
    try:
        _pip_install()
        restart()
        ok = _answering(url, timeout=STARTUP_TIMEOUT)
    except SidecarError:
        ok = False
    if not ok:
        out(f"The new version did not answer; going back to {before}.")
        _sh(["git", "-C", str(src_dir()), "fetch", "--depth", "1", "origin", before], check=False)
        _sh(["git", "-C", str(src_dir()), "reset", "--hard", before])
        _pip_install()
        restart()
        _answering(url, timeout=STARTUP_TIMEOUT)
        raise SidecarError(f"Update to {after} failed to start; rolled back to {before}. "
                           f"Log: {log_path()}")
    _write_marker(commit=after, commit_date=date,
                  updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return {"updated": True, "from": before, "commit": after, "commit_date": date}


def uninstall(purge: bool = False) -> dict:
    system = platform.system()
    if system == "Darwin":
        _sh(["launchctl", "unload", str(plist_path())], check=False, timeout=20)
        plist_path().unlink(missing_ok=True)
    elif system == "Linux":
        _sh(["systemctl", "--user", "disable", "--now", UNIT_NAME], check=False, timeout=30)
        unit_path().unlink(missing_ok=True)
        _sh(["systemctl", "--user", "daemon-reload"], check=False, timeout=20)
    if purge:
        shutil.rmtree(home(), ignore_errors=True)
    return {"removed_service": True, "purged": purge}


# ── status ────────────────────────────────────────────────────────────────


def age_days(commit_date: str) -> int | None:
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(commit_date)
    except (TypeError, ValueError):
        return None
    return (datetime.now(timezone.utc) - dt).days


def engine_report(url: str) -> dict:
    """Run one real query and report which upstream engines failed.

    SearXNG answers even when half its engines are broken, so "it responds"
    is not the same as "it works". unresponsive_engines is SearXNG's own
    account of what failed and why (CAPTCHA, timeout, parse error)."""
    import httpx
    try:
        r = httpx.get(f"{url.rstrip('/')}/search",
                      params={"q": "weather", "format": "json"}, timeout=15.0)
        d = r.json()
        return {"results": len(d.get("results") or []),
                "failing": [{"engine": e[0], "reason": e[1]}
                            for e in d.get("unresponsive_engines") or []]}
    except Exception as e:
        return {"results": 0, "failing": [], "error": str(e)[:200]}


def status(config: Config, with_engines: bool = False) -> dict:
    data = info()
    out = {"installed": data is not None, "service_running": False,
           "answering": False, "home": str(home())}
    if data is None:
        return out
    url = f"http://localhost:{data.get('port', DEFAULT_PORT)}"
    days = age_days(data.get("commit_date", ""))
    out.update({
        "url": url, "commit": data.get("commit"), "commit_date": data.get("commit_date"),
        "age_days": days, "stale": days is not None and days > STALE_DAYS,
        "service_running": service_running(), "answering": _probe(url),
        "log": str(log_path()), "python": data.get("python"),
    })
    if with_engines and out["answering"]:
        out["engines"] = engine_report(url)
    return out


# ── self-heal ─────────────────────────────────────────────────────────────


HEAL_INTERVAL_S = 60
HEAL_COOLDOWN_S = 300
_last_heal = 0.0


def heal_once(config: Config) -> str | None:
    """Restart the native instance if it is installed but not answering.

    launchd already restarts a crashed process; this covers the rest — a
    process that is alive and wedged, or a service someone unloaded.
    Rate-limited so a genuinely broken install is restarted every five
    minutes, not every minute. Returns what it did, for the log."""
    global _last_heal
    data = info()
    if data is None:
        return None
    url = f"http://localhost:{data.get('port', DEFAULT_PORT)}"
    if _probe(url):
        return None
    if time.time() - _last_heal < HEAL_COOLDOWN_S:
        return None
    _last_heal = time.time()
    try:
        if service_running():
            restart()
            action = "restarted"
        else:
            _start_service(data.get("port", DEFAULT_PORT))
            action = "started"
    except SidecarError as e:
        logger.warning("SearXNG is down and could not be restarted: %s", e)
        return "failed"
    logger.warning("SearXNG was not answering; %s its service", action)
    from .search import searxng as provider
    provider._probe_cache.pop(config.searxng_url, None)
    return action


def start_healer(config: Config) -> None:
    import threading

    def loop():
        while True:
            time.sleep(HEAL_INTERVAL_S)
            try:
                heal_once(config)
            except Exception:
                logger.debug("searxng heal check failed", exc_info=True)

    threading.Thread(target=loop, name="searxng-healer", daemon=True).start()
