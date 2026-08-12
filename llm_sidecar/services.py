"""Managing the optional SearXNG container.

The search provider itself has existed since 0.1.0 and is auto-detected. What
did not exist was a way to *get an instance running* — the instructions were
"run this docker command, then hand-edit settings.yml to enable JSON output",
which is two steps too many and one of them is easy to get wrong. Without the
JSON format the API returns 403 and everything silently falls back to
DuckDuckGo, so the failure looks like nothing happening.

So: ship the compose file and the settings, generate the secret, start it,
and wait until it actually answers a JSON query before claiming success.

Config is copied to ~/.config/llm-sidecar/searxng/ on first start and never
overwritten after that — it's yours to edit once it exists.
"""

from __future__ import annotations

import logging
import secrets
import shutil
import subprocess
import time
from pathlib import Path

import httpx

from .config import CONFIG_DIR, Config
from .types import SidecarError

logger = logging.getLogger("llm_sidecar.services")

ASSETS = Path(__file__).parent / "deploy" / "searxng"
INSTANCE_DIR = CONFIG_DIR / "searxng"
CONTAINER = "llm-sidecar-searxng"
STARTUP_TIMEOUT = 90


# Runtimes we know how to drive, in preference order. Podman is included
# because rootless containers are a reasonable thing to prefer, and its
# compose subcommand is close enough to Docker's for our four verbs.
RUNTIMES = (
    (["docker", "compose"], ["docker", "info"]),
    (["podman", "compose"], ["podman", "info"]),
    (["podman-compose"], ["podman", "info"]),
)


def _runtime() -> list[str]:
    """The compose invocation to use, or an error explaining what's missing."""
    installed = [(cmd, probe) for cmd, probe in RUNTIMES if shutil.which(cmd[0])]
    if not installed:
        raise SidecarError(
            "No container runtime found. Install Docker Desktop "
            "(https://docker.com/products/docker-desktop) or Podman — or run SearXNG "
            "however you like and point SEARXNG_URL at it."
        )

    not_running = []
    for cmd, probe in installed:
        try:
            if subprocess.run(probe, capture_output=True, timeout=20).returncode != 0:
                not_running.append(cmd[0])
                continue
            # The binary exists and its daemon answers; does compose work?
            if subprocess.run(cmd + ["version"], capture_output=True, timeout=20).returncode == 0:
                return cmd
        except (OSError, subprocess.SubprocessError):
            continue

    if not_running:
        name = "Docker Desktop" if "docker" in not_running else not_running[0]
        raise SidecarError(f"{name} is installed but not running. Start it and try again.")
    raise SidecarError(
        f"A container runtime is installed but its compose plugin is not. Upgrade it, or run "
        f"the compose file at {INSTANCE_DIR} yourself."
    )


# Kept as an alias: the name reads better in the errors above than in callers.
_docker = _runtime


def install(port: int = 8888, force: bool = False) -> Path:
    """Write the instance directory. Returns its path.

    Existing files are left alone unless `force` — the whole point of copying
    them out of the package is that the user can then change them."""
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)

    compose_dst = INSTANCE_DIR / "docker-compose.yml"
    if force or not compose_dst.exists():
        shutil.copy(ASSETS / "docker-compose.yml", compose_dst)

    settings_dst = INSTANCE_DIR / "settings.yml"
    if force or not settings_dst.exists():
        text = (ASSETS / "settings.yml").read_text()
        # A shared secret key across every install would be worse than no key.
        text = text.replace("GENERATED_ON_FIRST_START", secrets.token_hex(32))
        settings_dst.write_text(text)

    env_dst = INSTANCE_DIR / ".env"
    if force or not env_dst.exists():
        env_dst.write_text(f"SEARXNG_PORT={port}\n")

    return INSTANCE_DIR


def _compose(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        _runtime() + args, cwd=INSTANCE_DIR,
        capture_output=True, text=True, timeout=timeout,
    )


def wait_until_ready(url: str, timeout: int = STARTUP_TIMEOUT) -> bool:
    """Poll until the instance answers a JSON query.

    Deliberately checks `format=json` rather than mere reachability: the
    container serves HTML long before — and even when it will never — serve
    the JSON we need. A "started" that can't answer is not started."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url.rstrip('/')}/search",
                          params={"q": "ping", "format": "json"}, timeout=5.0)
            if r.status_code == 200 and isinstance(r.json().get("results"), list):
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def up(config: Config, port: int | None = None) -> dict:
    """Install if needed, start the container, wait until it really answers."""
    port = port or _port_from(config)
    install(port)

    result = _compose(["up", "-d"])
    if result.returncode != 0:
        err = (result.stderr or result.stdout).strip()
        if "port is already allocated" in err or "address already in use" in err:
            raise SidecarError(
                f"Port {port} is already in use. Either something else is on it, or an old "
                f"instance is still running — try `llm-sidecar searxng down`, or pick another "
                f"port with `--port`."
            )
        raise SidecarError(f"docker compose up failed:\n{err}")

    url = f"http://localhost:{port}"
    if not wait_until_ready(url):
        raise SidecarError(
            f"The container started but {url} never answered a JSON query within "
            f"{STARTUP_TIMEOUT}s. Check `llm-sidecar searxng logs`. The usual cause is a "
            f"settings.yml that lost the `formats: [html, json]` block."
        )

    return {"ok": True, "url": url, "dir": str(INSTANCE_DIR), "port": port}


def down(remove: bool = False) -> dict:
    """Stop the container. `remove` also deletes its volumes."""
    args = ["down"]
    if remove:
        args.append("--volumes")
    result = _compose(args)
    if result.returncode != 0:
        raise SidecarError(f"docker compose down failed:\n{(result.stderr or '').strip()}")
    return {"ok": True}


def logs(lines: int = 50) -> str:
    result = _compose(["logs", "--tail", str(lines)], timeout=30)
    return (result.stdout or result.stderr or "").strip()


def _port_from(config: Config) -> int:
    """Port implied by the configured searxng_url, defaulting to 8888."""
    tail = config.searxng_url.rstrip("/").rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() else 8888


def status(config: Config) -> dict:
    """Where SearXNG stands, without requiring Docker to be installed.

    Reachability is checked first and independently: an instance the user runs
    some other way is just as good as one we started, and reporting "not
    installed" at something that is plainly answering would be wrong."""
    url = config.searxng_url
    from .search import searxng as provider

    # Bypass the module-level probe cache so this always reflects reality.
    provider._probe_cache.pop(url, None)
    answering = provider.available(config)

    info = {
        "url": url,
        "answering_json": answering,
        "installed": (INSTANCE_DIR / "docker-compose.yml").exists(),
        "instance_dir": str(INSTANCE_DIR),
        "container": None,
        "docker": None,
    }

    runtime = next((c[0][0] for c in RUNTIMES if shutil.which(c[0][0])), None)
    if runtime is None:
        info["docker"] = "not installed"
        return info
    info["runtime"] = runtime

    try:
        r = subprocess.run(
            [runtime, "ps", "-a", "--filter", f"name={CONTAINER}",
             "--format", "{{.State}}\t{{.Status}}"],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode != 0:
            info["docker"] = "daemon not running"
        else:
            info["docker"] = "ok"
            line = r.stdout.strip()
            if line:
                state, _, human = line.partition("\t")
                info["container"] = {"state": state, "status": human}
    except (OSError, subprocess.SubprocessError):
        info["docker"] = "unavailable"

    return info
