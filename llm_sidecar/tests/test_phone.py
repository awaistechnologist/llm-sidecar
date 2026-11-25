"""The sidecar from a phone: the Host check, the /app/ routes, and
`llm-sidecar phone`.

The Host check gets the most attention because it is the one piece here that
can lock a working client out. Every name a real caller uses must still pass,
and the rebinding case it exists for must not.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from llm_sidecar import phone
from llm_sidecar.config import Config
from llm_sidecar.types import ModelInfo


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    from llm_sidecar import cache, catalogue, ledger

    monkeypatch.setattr(cache, "_COMPLETIONS", tmp_path / "completions")
    monkeypatch.setattr(cache, "_SEARCHES", tmp_path / "searches")
    monkeypatch.setattr(ledger, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(ledger, "LEDGER_FILE", tmp_path / "data" / "usage.jsonl")
    monkeypatch.setattr(catalogue, "openrouter_models", lambda config, force_refresh=False: [
        ModelInfo("acme/model-a:free", "a", context_length=8000)])
    monkeypatch.setattr(catalogue, "ollama_models", lambda config: [ModelInfo("ollama/local-1", "local-1")])
    return Config(openrouter_api_key="test-key", models={}, default_budget="free")


def client(cfg, host="localhost"):
    from fastapi.testclient import TestClient

    from llm_sidecar import daemon
    return TestClient(daemon.create_app(cfg), base_url=f"http://{host}")


# ── which names the daemon answers to ─────────────────────────────────────


@pytest.mark.parametrize("host", [
    "localhost", "localhost:4001", "127.0.0.1:4001", "[::1]:4001", "::1",
    # Docker Desktop and Podman: a container calling the host-side sidecar
    # is the commonest non-local use, and it worked before this check.
    "host.docker.internal:4001", "host.containers.internal:4001",
    # Tailscale: what the phone sends, with and without the HTTPS port.
    "my-mac.tail1234.ts.net", "my-mac.tail1234.ts.net:8443", "My-Mac.Tail1234.TS.NET.",
    "",  # HTTP/1.0 without Host: no browser does this, so it is not rebinding
])
def test_names_a_real_caller_uses_are_answered(host):
    assert phone.allowed_host(host, Config()) is True


@pytest.mark.parametrize("host", [
    # The DNS rebinding case: a page on a name its owner controls, re-pointed
    # at 127.0.0.1. Its scripts would be same-origin with the daemon.
    "evil.example", "evil.example:4001", "attacker.test",
    # Names that merely contain an allowed one.
    "ts.net.evil.example", "localhost.evil.example", "nots.net",
])
def test_other_names_are_refused(host):
    assert phone.allowed_host(host, Config()) is False


def test_named_extra_hosts_are_answered():
    c = Config(allowed_hosts=["mac.lan", "Studio.Local"])
    assert phone.allowed_host("mac.lan:4001", c)
    assert phone.allowed_host("studio.local", c)
    assert not phone.allowed_host("other.lan", c)


def test_hosts_come_from_the_environment(monkeypatch, tmp_path):
    from llm_sidecar import config as config_mod
    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "none.json")
    monkeypatch.setenv("LLM_SIDECAR_HOSTS", "mac.lan, Studio.Local ,")
    assert config_mod.load().allowed_hosts == ["mac.lan", "studio.local"]


@pytest.mark.parametrize("bind", ["0.0.0.0", "::"])
def test_a_wildcard_bind_turns_the_check_off(bind):
    """Someone who bound every interface reaches it by whatever name their
    network uses. A list would only break the clients they set it up for."""
    assert phone.allowed_host("anything.example", Config(daemon_host=bind))


def test_the_bind_address_itself_is_answered():
    assert phone.allowed_host("192.168.1.20:4001", Config(daemon_host="192.168.1.20"))


def test_a_refused_request_never_reaches_a_route(cfg):
    r = client(cfg, "evil.example").get("/health")
    assert r.status_code == 403
    assert "LLM_SIDECAR_HOSTS" in r.json()["detail"]
    # And an expensive route is refused just the same.
    r = client(cfg, "evil.example").post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 403


def test_a_tailnet_request_is_served(cfg):
    assert client(cfg, "my-mac.tail1234.ts.net:8443").get("/health").status_code == 200


# ── /app/ ─────────────────────────────────────────────────────────────────


def test_app_redirects_to_its_canonical_address(cfg):
    r = client(cfg).get("/app", follow_redirects=False)
    assert r.status_code == 308 and r.headers["location"] == "/app/"


def test_app_page_is_served_and_not_cached(cfg):
    r = client(cfg).get("/app/")
    assert r.status_code == 200
    assert "<title>Sidecar</title>" in r.text
    assert r.headers["cache-control"] == "no-cache"


def test_the_service_worker_may_control_app(cfg):
    r = client(cfg).get("/app/sw.js")
    assert r.status_code == 200
    assert r.headers["service-worker-allowed"] == "/app/"
    assert "javascript" in r.headers["content-type"]


@pytest.mark.parametrize("name", ["index.html", "app.html", "..%2Fphone.py", "secret.json", "phone.py"])
def test_only_named_files_are_served_from_ui(cfg, name):
    """A path parameter into a directory is a traversal bug waiting for
    someone to add a file, so only an explicit list is served."""
    assert client(cfg).get(f"/app/{name}").status_code == 404


def test_every_listed_static_file_exists():
    for name in phone.STATIC:
        assert (phone.UI / name).is_file(), name


def test_the_manifest_is_installable():
    m = json.loads((phone.UI / "manifest.webmanifest").read_text(encoding="utf-8"))
    assert m["start_url"] == m["scope"] == "/app/"
    assert m["display"] == "standalone"
    sizes = {i["sizes"] for i in m["icons"]}
    assert {"192x192", "512x512"} <= sizes, "Chrome needs both to offer Install"
    assert any(i.get("purpose") == "maskable" for i in m["icons"])
    # Share → Sidecar from any app.
    assert m["share_target"]["action"] == "/app/"
    assert set(m["share_target"]["params"]) == {"title", "text", "url"}


def test_the_phone_page_loads_nothing_from_anywhere_else():
    """A page that loaded a CDN would tell it each time you opened the app,
    and break whenever the CDN was unreachable and your Mac was not."""
    html = (phone.UI / "app.html").read_text(encoding="utf-8")
    assert not re.search(r'<(script|link|img|iframe)[^>]+(src|href)="https?:', html)
    assert not re.search(r"url\(\s*['\"]?https?:", html)
    assert not re.search(r"fetch\(\s*['\"`]https?:", html)


def test_the_ui_can_be_turned_off(cfg):
    cfg.ui_enabled = False
    c = client(cfg)
    assert c.get("/app/").status_code == 404
    assert c.get("/app/manifest.webmanifest").status_code == 404


# ── engines ───────────────────────────────────────────────────────────────


def test_engines_lists_what_the_picker_needs_without_running_claude(cfg, monkeypatch):
    """/claude-cli/status costs a real model call to check sign-in. Opening
    a model picker must not spend anything."""
    from llm_sidecar import claude_cli

    def boom(*a, **k):
        raise AssertionError("the picker ran the Claude CLI")

    monkeypatch.setattr(claude_cli, "check", boom)
    monkeypatch.setattr(claude_cli, "binary", lambda config: "/usr/local/bin/claude")
    cfg.claude_cli_enabled = True

    r = client(cfg).get("/app/api/engines")
    assert r.status_code == 200
    e = r.json()
    assert e["local"] == ["ollama/local-1"]
    assert "claude/sonnet" in e["claude"]
    assert e["cloud"] == {"configured": True, "count": 1}
    assert "images" in e and "version" in e


def test_engines_omits_claude_when_it_is_off(cfg):
    cfg.claude_cli_enabled = False
    assert client(cfg).get("/app/api/engines").json()["claude"] == []


# ── images from the phone ─────────────────────────────────────────────────


def test_a_phone_image_goes_where_the_sidecar_says_not_where_the_phone_says(cfg, monkeypatch, tmp_path):
    from llm_sidecar import images as im

    seen = {}

    def fake(req, config=None, *, allow_daemon=True):
        seen["req"], seen["allow_daemon"] = req, allow_daemon
        out = Path(req.out_dir) / f"{req.stem}_7.png"
        out.write_bytes(b"\x89PNG")
        return im.ImageResult(path=str(out), manifest_path=str(out) + ".json", profile=req.profile,
                              model="flux2-klein-4b", seed=7, digest="d", seconds=2.0, warm=True,
                              width=1024, height=1024)

    monkeypatch.setattr(im, "generate", fake)
    r = client(cfg).post("/app/api/image", json={"prompt": "A Red Apple!", "out_dir": "/etc"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert Path(seen["req"].out_dir) == phone.image_dir()
    assert seen["allow_daemon"] is False, "inside the daemon, delegating would call itself"
    assert re.match(r"^/app/images/\d{8}-\d{6}-a-red-apple_7\.png$", body["url"])
    # And it can be fetched back.
    assert client(cfg).get(body["url"]).content == b"\x89PNG"


def test_phone_image_refusals_keep_their_status(cfg, monkeypatch):
    from llm_sidecar import images as im

    def refuse(req, config=None, *, allow_daemon=True):
        raise im.LicenceError("non-commercial")

    monkeypatch.setattr(im, "generate", refuse)
    r = client(cfg).post("/app/api/image", json={"prompt": "x", "profile": "cinematic"})
    assert r.status_code == 400 and "non-commercial" in r.json()["detail"]


@pytest.mark.parametrize("name", ["..%2F..%2Fusage.jsonl", "x.jpg", "a%2Fb.png", ".png", "x.png.json"])
def test_image_files_are_png_basenames_only(cfg, name):
    (phone.image_dir() / "x.png.json").write_text("{}")
    assert client(cfg).get(f"/app/images/{name}").status_code == 404


def test_gallery_lists_newest_first_with_prompts(cfg):
    import os
    import time as _t

    d = phone.image_dir()
    for i, prompt in enumerate(["older", "newer"]):
        f = d / f"img{i}.png"
        f.write_bytes(b"\x89PNG")
        Path(str(f) + ".json").write_text(json.dumps({"request": {"prompt": prompt}}))
        os.utime(f, (_t.time() + i, _t.time() + i))
    imgs = client(cfg).get("/app/api/gallery").json()["images"]
    assert [x["prompt"] for x in imgs] == ["newer", "older"]


def test_with_a_token_an_image_can_be_fetched_by_query(cfg):
    """An <img> tag cannot send an Authorization header."""
    cfg.daemon_token = "s3cret"
    (phone.image_dir() / "a.png").write_bytes(b"\x89PNG")
    c = client(cfg)
    assert c.get("/app/images/a.png").status_code == 401
    assert c.get("/app/images/a.png?t=wrong").status_code == 401
    assert c.get("/app/images/a.png?t=s3cret").status_code == 200
    assert c.get("/app/images/a.png", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    # Only image files take the token that way.
    assert c.get("/app/api/gallery?t=s3cret").status_code == 401


# ── `llm-sidecar phone` ───────────────────────────────────────────────────


class FakeTailscale:
    """Records the commands and answers like tailscale would."""

    def __init__(self, name="my-mac.tail1234.ts.net", funnel=False, serve_rc=0):
        self.calls: list[list[str]] = []
        self.name, self.funnel, self.serve_rc = name, funnel, serve_rc

    def __call__(self, ts, *args, timeout=30):
        self.calls.append(list(args))
        out, rc = "", 0
        if args[:2] == ("status", "--json"):
            out = json.dumps({"Self": {"DNSName": self.name + "."}})
        elif args[:3] == ("serve", "status", "--json"):
            out = json.dumps({"AllowFunnel": {f"{self.name}:8443": True}} if self.funnel else {})
        elif args and args[0] == "serve" and "--bg" in args:
            rc = self.serve_rc
        return subprocess.CompletedProcess(args, rc, out, "boom" if rc else "")


@pytest.fixture
def ts(monkeypatch):
    fake = FakeTailscale()
    monkeypatch.setattr(phone, "tailscale_binary", lambda: "/usr/bin/tailscale")
    monkeypatch.setattr(phone, "_ts", fake)
    monkeypatch.setattr(phone, "_qr", lambda text: None)
    return fake


@pytest.fixture
def daemon_up(monkeypatch):
    import httpx

    class R:
        status_code = 200

    monkeypatch.setattr(httpx, "get", lambda *a, **k: R())


def run(action="on", config=None):
    lines: list[str] = []
    rc = phone.run(action, config or Config(), out=lines.append)
    return rc, "\n".join(lines)


def test_phone_serves_the_daemon_privately_on_its_own_port(ts, daemon_up):
    rc, out = run()
    assert rc == 0
    serve = next(c for c in ts.calls if "--bg" in c)
    assert serve == ["serve", "--bg", "--https=8443", "http://127.0.0.1:4001"]
    assert not any("funnel" in c for c in ts.calls), "never Funnel: that is public"
    assert "https://my-mac.tail1234.ts.net:8443/app/" in out


def test_phone_refuses_and_undoes_a_funnelled_port(ts, daemon_up):
    """`serve` keeps a Funnel someone already enabled on the port, and an
    unauthenticated daemon holding an API key must never be public."""
    ts.funnel = True
    rc, out = run()
    assert rc == 1
    assert "public" in out.lower()
    assert ["serve", "--https=8443", "off"] in ts.calls


def test_phone_wants_the_daemon_running_first(ts, monkeypatch):
    import httpx

    def down(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", down)
    rc, out = run()
    assert rc == 1 and "service install" in out
    assert not any("--bg" in c for c in ts.calls)


def test_phone_without_tailscale_says_how_to_get_it(monkeypatch):
    monkeypatch.setattr(phone, "tailscale_binary", lambda: None)
    rc, out = run()
    assert rc == 1 and "brew install --cask tailscale" in out


def test_phone_off(ts):
    rc, _ = run("off")
    assert rc == 0 and ts.calls == [["serve", "--https=8443", "off"]]


def test_phone_warns_about_shared_tailnets_only_without_a_token(ts, daemon_up):
    assert "LLM_SIDECAR_TOKEN" in run()[1]
    assert "LLM_SIDECAR_TOKEN" not in run(config=Config(daemon_token="x"))[1]
