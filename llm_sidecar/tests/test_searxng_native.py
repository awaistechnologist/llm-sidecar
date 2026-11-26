"""Native SearXNG: install, migrate from Docker, update with rollback,
self-heal, and search health. No real git, pip, launchctl or network."""

from __future__ import annotations

import json
import time

import pytest

from llm_sidecar.config import Config
from llm_sidecar.types import SidecarError


@pytest.fixture
def sx(monkeypatch, tmp_path):
    from llm_sidecar import ledger, searxng_native as n

    monkeypatch.setattr(ledger, "DATA_DIR", tmp_path / "data")
    settings = tmp_path / "config" / "searxng" / "settings.yml"
    monkeypatch.setattr(n, "settings_path", lambda: settings)
    monkeypatch.setattr(n, "plist_path", lambda: tmp_path / "LaunchAgents" / "sx.plist")
    monkeypatch.setattr(n, "log_path", lambda: tmp_path / "sx.log")
    monkeypatch.setattr(n.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(n.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(n.time, "sleep", lambda s: None)

    state = {"calls": [], "commit": "aaa1111", "port_busy": False,
             "answering": True, "running": False, "commits_answer": None}

    def sh(cmd, *, cwd=None, timeout=600, check=True):
        state["calls"].append(cmd)
        out = ""
        if cmd[:2] == ["git", "clone"]:
            (n.src_dir() / ".git").mkdir(parents=True)
        elif cmd[0].endswith("python3.13") and cmd[1:3] == ["-m", "venv"]:
            (n.venv_dir() / "bin").mkdir(parents=True)
            (n.venv_dir() / "bin" / "python").write_text("")
        elif cmd[:2] == ["git", "-C"] and cmd[3] == "log":
            out = f"{state['commit']} 2026-11-23T08:15:23+01:00"
        elif cmd[:2] == ["git", "-C"] and cmd[3] == "reset":
            target = cmd[-1]
            if target == "FETCH_HEAD":
                state["commit"] = state.get("remote", state["commit"])
            else:
                state["commit"] = target
        elif cmd[:2] == ["launchctl", "list"]:
            out = n.LABEL if state["running"] else ""
        elif cmd[:2] == ["launchctl", "load"]:
            state["running"] = True
        return type("R", (), {"returncode": 0, "stdout": out, "stderr": ""})()

    def answering(url, timeout=0):
        if state["commits_answer"] is not None:
            return state["commit"] in state["commits_answer"]
        return state["answering"]

    monkeypatch.setattr(n, "_sh", sh)
    monkeypatch.setattr(n, "_port_busy", lambda port: state["port_busy"])
    monkeypatch.setattr(n, "_answering", answering)
    monkeypatch.setattr(n, "_probe", lambda url: answering(url))
    monkeypatch.setattr(n, "find_python", lambda: "/opt/homebrew/bin/python3.13")
    return n, state


def test_a_fresh_install_clones_builds_and_starts_a_login_service(sx):
    n, state = sx
    r = n.install(Config(), out=lambda s: None)

    assert r["url"] == "http://localhost:8888"
    assert r["commit"] == "aaa1111"
    cmds = [" ".join(c[:2]) for c in state["calls"]]
    assert "git clone" in cmds and "launchctl load" in cmds
    assert n.info()["python"].endswith("python3.13")
    settings = n.settings_path().read_text(encoding="utf-8")
    assert "json" in settings and "GENERATED_ON_FIRST_START" not in settings


def test_the_service_listens_on_loopback_only(sx):
    """The limiter is off: on the network this is a free search proxy."""
    n, _ = sx
    n.install(Config(), out=lambda s: None)
    plist = n.plist_path().read_text(encoding="utf-8")
    assert "<key>GRANIAN_HOST</key><string>127.0.0.1</string>" in plist
    assert "0.0.0.0" not in plist and "<string>::</string>" not in plist
    assert "<key>KeepAlive</key><true/>" in plist
    assert "searx.webapp:app" in plist


def test_a_port_taken_by_something_else_is_refused(sx):
    n, state = sx
    state.update(port_busy=True)
    with pytest.raises(SidecarError, match="in use by something else"):
        n.install(Config(), out=lambda s: None)
    assert not any(c[:2] == ["git", "clone"] for c in state["calls"])


def test_reinstalling_reuses_the_checkout(sx):
    n, state = sx
    n.install(Config(), out=lambda s: None)
    state["calls"].clear()
    state["port_busy"] = True          # our own instance is on the port
    n.install(Config(), out=lambda s: None)
    assert not any(c[:2] == ["git", "clone"] for c in state["calls"])


def test_an_install_that_never_answers_says_where_the_log_is(sx):
    n, state = sx
    state["answering"] = False
    with pytest.raises(SidecarError, match="did not answer"):
        n.install(Config(), out=lambda s: None)


def test_update_with_nothing_new_is_a_no_op(sx):
    n, state = sx
    n.install(Config(), out=lambda s: None)
    assert n.update(out=lambda s: None)["updated"] is False


def test_update_moves_to_the_new_commit(sx):
    n, state = sx
    n.install(Config(), out=lambda s: None)
    state["remote"] = "bbb2222"
    r = n.update(out=lambda s: None)
    assert r == {"updated": True, "from": "aaa1111", "commit": "bbb2222",
                 "commit_date": "2026-11-23T08:15:23+01:00"}
    assert n.info()["commit"] == "bbb2222"


def test_an_update_that_breaks_search_is_rolled_back(sx):
    """SearXNG ships often, and engines break; a bad update must not leave
    search down until someone notices."""
    n, state = sx
    n.install(Config(), out=lambda s: None)
    state["remote"] = "bad3333"
    state["commits_answer"] = {"aaa1111"}
    with pytest.raises(SidecarError, match="rolled back to aaa1111"):
        n.update(out=lambda s: None)
    assert state["commit"] == "aaa1111"
    assert n.info()["commit"] == "aaa1111"


def test_update_needs_an_install(sx):
    n, _ = sx
    with pytest.raises(SidecarError, match="not installed"):
        n.update(out=lambda s: None)


def test_uninstall_keeps_files_unless_purged(sx):
    n, _ = sx
    n.install(Config(), out=lambda s: None)
    n.uninstall()
    assert not n.plist_path().exists() and n.home().exists()
    n.uninstall(purge=True)
    assert not n.home().exists()


def test_status_flags_an_old_checkout(sx, monkeypatch):
    n, _ = sx
    n.install(Config(), out=lambda s: None)
    data = n.info()
    data["commit_date"] = "2026-01-01T00:00:00+00:00"
    n.marker().write_text(json.dumps(data))
    s = n.status(Config())
    assert s["installed"] and s["answering"] and s["stale"] is True


def test_the_healer_restarts_an_installed_instance_that_stopped_answering(sx, monkeypatch):
    n, state = sx
    n.install(Config(), out=lambda s: None)
    monkeypatch.setattr(n, "_last_heal", 0.0)
    state["answering"] = False
    assert n.heal_once(Config()) == "restarted"
    assert any(c[:3] == ["launchctl", "kickstart", "-k"] for c in state["calls"])
    assert n.heal_once(Config()) is None, "restarted again inside the cooldown"


def test_the_healer_leaves_a_healthy_or_absent_instance_alone(sx, monkeypatch):
    n, state = sx
    monkeypatch.setattr(n, "_last_heal", 0.0)
    assert n.heal_once(Config()) is None          # not installed
    n.install(Config(), out=lambda s: None)
    assert n.heal_once(Config()) is None          # answering


# ── search health ──────────────────────────────────────────────────────────


@pytest.fixture
def health_env(monkeypatch):
    from llm_sidecar import search, searxng_native
    from llm_sidecar.search import searxng

    st = {"up": True, "native": False}
    monkeypatch.setattr(searxng, "available", lambda config: st["up"])
    monkeypatch.setattr(searxng_native, "installed", lambda: st["native"])
    monkeypatch.setitem(search._HEALTH, "last_ok", None)
    monkeypatch.setitem(search._HEALTH, "last_empty", None)
    return search, st


def test_health_is_ok_when_searxng_answers(health_env):
    search, _ = health_env
    h = search.health(Config())
    assert h["state"] == "ok" and h["provider"] == "searxng"


def test_health_says_how_to_fix_a_missing_searxng(health_env):
    search, st = health_env
    st["up"] = False
    h = search.health(Config())
    assert h["state"] == "degraded" and "searxng install" in h["message"]


def test_health_says_the_native_one_is_being_restarted(health_env):
    search, st = health_env
    st.update(up=False, native=True)
    assert "restarting" in search.health(Config())["message"]


def test_empty_fallback_searches_mean_search_is_down(health_env):
    """The case that went unnoticed: SearXNG off, DuckDuckGo throttled,
    every search returning nothing, and no one told."""
    search, st = health_env
    st["up"] = False
    search._record("ddg", False, "DuckDuckGo: 202 Ratelimit")
    h = search.health(Config())
    assert h["state"] == "down" and h["last_error"].startswith("DuckDuckGo")
    search._record("ddg", True)
    assert search.health(Config())["state"] == "degraded"


def test_search_records_what_happened(health_env, monkeypatch):
    from llm_sidecar import cache, search
    from llm_sidecar.search import ddg, searxng

    monkeypatch.setattr(cache, "get_search", lambda *a, **k: None)
    monkeypatch.setattr(cache, "put_search", lambda *a, **k: None)
    monkeypatch.setattr(searxng, "search", lambda *a, **k: [])
    monkeypatch.setattr(ddg, "search", lambda *a, **k: [])
    assert search.search("x", Config()) == []
    assert search._HEALTH["provider"] == "ddg"
    assert search._HEALTH["last_empty"] and search._HEALTH["last_empty"] <= time.time()
