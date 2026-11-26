"""End-to-end: real daemons, the real CLI and the MCP server over stdio, in
a throwaway home. Nothing here touches your config, data or sign-ins.

Off by default, because it starts processes and takes tens of seconds:

    LLM_SIDECAR_E2E=1 pytest -m e2e              # daemon, CLI, MCP, security
    LLM_SIDECAR_E2E=1 LLM_SIDECAR_E2E_INSTALL=1 pytest -m e2e   # + a clean install

Local model calls use the smallest installed Ollama model and are skipped
when Ollama has none. Nothing here calls a paid provider.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not os.getenv("LLM_SIDECAR_E2E"), reason="set LLM_SIDECAR_E2E=1"),
]

ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _env(home: Path, **extra) -> dict:
    """A clean environment pointing every path the sidecar uses at `home`."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ANTHROPIC_", "OPENROUTER_", "LLM_SIDECAR_"))}
    env.update(
        LLM_SIDECAR_HOME=str(home / "config"),
        LLM_SIDECAR_DATA=str(home / "data"),
        LLM_SIDECAR_CACHE=str(home / "cache"),
        ANTHROPIC_CONFIG_DIR=str(home / "anthropic"),
        PYTHONPATH=str(ROOT),
    )
    env.update({k: str(v) for k, v in extra.items()})
    return env


class Daemon:
    def __init__(self, home: Path, **env):
        self.home, self.extra = home, env
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc = None

    def start(self):
        log = (self.home / "daemon.log").open("ab")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "llm_sidecar", "serve", "--port", str(self.port)],
            env=_env(self.home, **self.extra), stdout=log, stderr=log, cwd=self.home)
        for _ in range(150):
            try:
                if httpx.get(self.url + "/health", timeout=1).status_code < 500:
                    return self
            except httpx.HTTPError:
                pass
            if self.proc.poll() is not None:
                break
            time.sleep(0.1)
        raise RuntimeError("daemon did not start:\n"
                           + (self.home / "daemon.log").read_text(encoding="utf-8", errors="replace")[-2000:])

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def get(self, path, **kw):
        return httpx.get(self.url + path, timeout=30, **kw)

    def post(self, path, body=None, **kw):
        return httpx.post(self.url + path, json=body or {}, timeout=120, **kw)


@pytest.fixture
def home(tmp_path):
    return tmp_path


@pytest.fixture
def daemon(home):
    d = Daemon(home).start()
    yield d
    d.stop()


def _cli(home, *args, timeout=120):
    return subprocess.run([sys.executable, "-m", "llm_sidecar", *args],
                          env=_env(home), capture_output=True, text=True, timeout=timeout)


def _small_ollama_model() -> str | None:
    try:
        tags = httpx.get("http://localhost:11434/api/tags", timeout=2).json().get("models") or []
    except (httpx.HTTPError, ValueError):
        return None
    chat = [m for m in tags if "embed" not in m["name"]]
    return f"ollama/{min(chat, key=lambda m: m.get('size', 1e18))['name']}" if chat else None


# ── the web app and dashboard ────────────────────────────────────────────


def test_pages_are_served_compressed_and_cheap_to_recheck(daemon):
    r = daemon.get("/app/", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200 and r.headers.get("content-encoding") == "gzip"
    assert int(r.headers["content-length"]) < 40_000, "the phone page should compress well"
    tag = r.headers["etag"]
    again = daemon.get("/app/", headers={"If-None-Match": tag})
    assert again.status_code == 304
    for path in ("/", "/app/sw.js", "/app/manifest.webmanifest", "/app/icon-192.png"):
        assert daemon.get(path).status_code == 200, path


def test_streaming_is_not_held_back_by_compression(daemon):
    """GZip must not buffer server-sent events: tokens arrive as they are
    produced, not all at the end."""
    model = _small_ollama_model()
    if not model:
        pytest.skip("no local Ollama model")
    stamps = []
    with httpx.stream("POST", daemon.url + "/v1/chat/completions", timeout=120,
                      headers={"Accept-Encoding": "gzip"},
                      json={"model": model, "stream": True, "max_tokens": 60,
                            "messages": [{"role": "user", "content": "Count from 1 to 30."}]}) as r:
        assert r.headers.get("content-encoding") != "gzip"
        for line in r.iter_lines():
            if line.startswith("data:") and '"content"' in line:
                stamps.append(time.time())
    assert len(stamps) > 3 and stamps[-1] - stamps[0] > 0.05, "tokens arrived in one burst"


def test_local_chat_works_and_is_recorded(daemon):
    model = _small_ollama_model()
    if not model:
        pytest.skip("no local Ollama model")
    r = daemon.post("/v1/chat/completions", {
        "model": model, "max_tokens": 20,
        "messages": [{"role": "user", "content": "Say hi."}]}).json()
    assert r["choices"][0]["message"]["content"].strip()
    assert daemon.get("/usage").json()["calls"] >= 1


def test_models_and_allowances_list_what_exists(daemon):
    ids = [m["id"] for m in daemon.get("/v1/models").json()["data"]]
    assert ids[:1] == ["auto"]
    if _small_ollama_model():
        assert any(i.startswith("ollama/") for i in ids)
    assert not any(i.startswith(("claude-api/",)) for i in ids), "off without credentials"
    providers = [p["provider"] for p in daemon.get("/allowances").json()["providers"]]
    assert providers[0] == "ollama" and "claude-api" not in providers
    eng = daemon.get("/app/api/engines").json()
    assert set(eng) >= {"version", "local", "claude", "claude_api", "cloud", "images", "search"}


# ── settings ─────────────────────────────────────────────────────────────


def test_settings_survive_a_restart_and_the_key_only_with_consent(home):
    d = Daemon(home).start()
    try:
        assert d.post("/config/budget", {"budget": "cheap"}).json()["persisted"]
        assert d.post("/config/tier", {"tier": "fast", "model": "ollama/x"}).json()["persisted"]
        d.post("/config/claude-api", {"monthly_credit_usd": 40, "reset_day": 18})
        # A key set without "remember" stays in memory, even when other
        # settings are saved after it.
        d.post("/config/api-key", {"key": "sk-or-e2e-test", "persist": False})
        d.post("/config/budget", {"budget": "cheap"})
        cfg = home / "config" / "config.json"
        assert "sk-or-e2e-test" not in cfg.read_text(encoding="utf-8")
        assert d.get("/status").json()["key_source"] == "session"
        d.post("/config/api-key", {"key": "sk-or-e2e-test", "persist": True})
        assert "sk-or-e2e-test" in cfg.read_text(encoding="utf-8")
        assert stat.S_IMODE(cfg.stat().st_mode) == 0o600, "a file holding a key must be owner-only"
        assert "sk-or-e2e-test" not in d.get("/status").text, "the key must never be echoed"
        d.post("/config/api-key", {"key": "", "persist": True})
    finally:
        d.stop()

    d = Daemon(home).start()
    try:
        s = d.get("/status").json()
        assert s["default_budget"] == "cheap"
        assert s["pinned_tiers"] == {"fast": "ollama/x"}
        assert s["cloud_configured"] is False
        a = json.loads(cfg.read_text(encoding="utf-8"))
        assert a["claude_api_monthly_credit_usd"] == 40 and a["claude_api_reset_day"] == 18
    finally:
        d.stop()


# ── security ─────────────────────────────────────────────────────────────


def test_only_loopback_and_tailnet_hosts_are_answered(daemon):
    """DNS rebinding: a page on evil.example resolving to 127.0.0.1 must
    not be able to drive the daemon."""
    assert daemon.get("/status", headers={"Host": "evil.example"}).status_code == 403
    assert daemon.get("/status", headers={"Host": f"localhost:{daemon.port}"}).status_code == 200
    assert daemon.get("/status", headers={"Host": "my-mac.tail0000.ts.net"}).status_code == 200


def test_no_cross_origin_access_by_default(daemon):
    r = daemon.get("/status", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_static_files_cannot_be_escaped(daemon):
    for p in ("/app/..%2f..%2fconfig.py", "/app/%2e%2e%2fdaemon.py", "/app/../../etc/passwd",
              "/app/images/..%2f..%2fconfig.json"):
        r = httpx.get(daemon.url + p, timeout=5)
        assert r.status_code in (400, 404), p


def test_the_daemon_listens_on_loopback_only(daemon):
    out = subprocess.run(["lsof", "-nP", "-a", "-p", str(daemon.proc.pid), "-iTCP", "-sTCP:LISTEN"],
                         capture_output=True, text=True).stdout
    assert f"127.0.0.1:{daemon.port}" in out and "*:" not in out


def test_a_token_locks_the_api_but_not_the_page_shell(home):
    d = Daemon(home, LLM_SIDECAR_TOKEN="e2e-secret").start()
    try:
        assert d.get("/status").status_code == 401
        assert d.get("/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert d.get("/status", headers={"Authorization": "Bearer e2e-secret"}).status_code == 200
        assert d.post("/config/budget", {"budget": "best"}).status_code == 401
        assert d.get("/app/").status_code == 200        # asks for the token itself
        assert d.get("/app/api/engines").status_code == 401
    finally:
        d.stop()


# ── memory ───────────────────────────────────────────────────────────────


def test_an_idle_daemon_stays_small(daemon):
    for path in ("/status", "/v1/models", "/allowances", "/app/", "/app/api/engines"):
        daemon.get(path)
    rss_kib = int(subprocess.run(["ps", "-o", "rss=", "-p", str(daemon.proc.pid)],
                                 capture_output=True, text=True).stdout.strip())
    assert rss_kib < 400 * 1024, f"idle daemon uses {rss_kib // 1024} MiB"


# ── CLI ──────────────────────────────────────────────────────────────────


def test_cli_journeys_in_a_fresh_home(home):
    for args in (["--help"], ["status"], ["config", "show"], ["usage"], ["allowances"],
                 ["models"], ["cache"], ["searxng", "status"], ["service", "status"]):
        r = _cli(home, *args)
        assert r.returncode in (0, 1), (args, r.stderr[-500:])
        assert "Traceback" not in r.stderr, (args, r.stderr[-1500:])
    assert _cli(home, "config", "tier", "fast", "ollama/llama3.2:1b").returncode == 0
    assert "ollama/llama3.2:1b" in _cli(home, "config", "show").stdout
    assert _cli(home, "config", "budget", "best").returncode == 0
    assert json.loads(_cli(home, "allowances", "--json").stdout)["providers"]
    r = _cli(home, "ask", "--model", "claude-api/haiku", "hi")
    assert r.returncode == 1 and "off" in r.stderr, "API credits must be off until switched on"


def test_cli_ask_with_a_local_model(home):
    model = _small_ollama_model()
    if not model:
        pytest.skip("no local Ollama model")
    r = _cli(home, "ask", "--model", model, "--max-tokens", "20", "Say hi.")
    assert r.returncode == 0 and r.stdout.strip(), r.stderr


# ── MCP ──────────────────────────────────────────────────────────────────


def test_mcp_over_stdio(home):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-m", "llm_sidecar", "mcp"],
                                       env=_env(home))
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                tools = {t.name for t in (await s.list_tools()).tools}
                status = await s.call_tool("sidecar_status", {})
                allow = await s.call_tool("allowances", {})
                return tools, status, allow

    tools, status, allow = asyncio.run(asyncio.wait_for(run(), 60))
    assert {"delegate", "allowances", "answer_question", "generate_image"} <= tools
    assert not status.is_error and not allow.is_error
    assert "ollama" in allow.content[0].text


# ── install ──────────────────────────────────────────────────────────────


@pytest.mark.skipif(not os.getenv("LLM_SIDECAR_E2E_INSTALL"), reason="set LLM_SIDECAR_E2E_INSTALL=1")
def test_a_clean_install_works(tmp_path):
    """Build the wheel, install it into an empty venv, and run it: the
    package must carry its UI, changelog and entry points."""
    dist = tmp_path / "dist"
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(dist), str(ROOT)],
                   check=True, capture_output=True)
    wheel = next(dist.glob("*.whl"))
    venv = tmp_path / "venv"
    subprocess.run(["uv", "venv", "-q", "--python", sys.executable, str(venv)], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "--python", str(venv / "bin" / "python"),
                    f"{wheel}[claude-api]"], check=True, capture_output=True)
    env = _env(tmp_path)
    env.pop("PYTHONPATH")
    for cmd in (["llm-sidecar", "--help"], ["llm-sidecar", "status"],
                ["llm-sidecar", "config", "claude-api"]):
        r = subprocess.run([str(venv / "bin" / cmd[0]), *cmd[1:]], env=env, cwd=tmp_path,
                           capture_output=True, text=True, timeout=120)
        assert "Traceback" not in r.stderr, (cmd, r.stderr[-1500:])
    py = str(venv / "bin" / "python")
    r = subprocess.run([py, "-c", "import llm_sidecar, pathlib; p = pathlib.Path(llm_sidecar.__file__).parent;"
                        "print(all((p / f).exists() for f in ('ui/app.html', 'ui/index.html', 'ui/sw.js', 'CHANGELOG.md')))"],
                       env=env, capture_output=True, text=True, cwd=tmp_path)
    assert r.stdout.strip() == "True", r.stderr
