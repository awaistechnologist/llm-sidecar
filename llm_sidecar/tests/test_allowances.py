"""The claude-api/ provider and the allowances report. No real SDK calls,
no network: the SDK client and httpx are replaced at the seam."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from types import SimpleNamespace as NS

import pytest

from llm_sidecar.config import Config


# ── claude-api/ ──────────────────────────────────────────────────────────


def _msg(text="hi", model="claude-haiku-5-5", inp=1000, out=200, stop="end_turn"):
    return NS(stop_reason=stop, model=model, content=[NS(type="text", text=text)],
              usage=NS(input_tokens=inp, output_tokens=out,
                       cache_creation_input_tokens=0, cache_read_input_tokens=0))


@pytest.fixture
def api(monkeypatch):
    from llm_sidecar import claude_api as m

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    m._resolved.clear()
    state = {"calls": [], "reply": _msg(), "error": None,
             "models": [NS(id="claude-haiku-4-5", created_at=1),
                        NS(id="claude-haiku-5-5", created_at=2),
                        NS(id="claude-sonnet-5-5", created_at=3)]}

    def create(**params):
        state["calls"].append(params)
        if state["error"]:
            raise state["error"]
        return state["reply"]

    fake = NS(beta=NS(messages=NS(create=create)),
              models=NS(list=lambda: state["models"]))
    monkeypatch.setattr(m, "_client", lambda config, asynchronous=False: fake)
    return m, state


def _cfg(**kw):
    return Config(**{"claude_api_enabled": True, **kw})


def test_off_by_default_like_the_subscription_provider():
    assert Config().claude_api_enabled is False


def test_a_call_answers_and_counts_its_cost(api):
    m, state = api
    done = m.complete([{"role": "user", "content": "hello"}], "claude-api/haiku", _cfg())
    assert done.text == "hi" and done.model == "claude-api/haiku"
    # 1000 in at $0.10/M plus 200 out at $0.50/M.
    assert done.usage.cost_usd == pytest.approx(0.0002)
    assert m.spend(_cfg())["spent_usd"] == pytest.approx(0.0002)
    assert m.spend(_cfg())["calls"] == 1


def test_the_request_suits_the_current_api(api):
    """System lifted out, no sampling parameters (the current API version
    rejects them), low effort, thinking off where the model allows it, and
    refusal fallback on."""
    m, state = api
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    m.complete([{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}],
               "claude-api/haiku", _cfg(), max_tokens=500,
               response_format={"type": "json_schema", "json_schema": {"schema": schema}})
    p = state["calls"][0]
    assert p["model"] == "claude-haiku-5-5"
    assert p["system"] == "S" and p["messages"] == [{"role": "user", "content": "hi"}]
    assert "temperature" not in p
    assert p["output_config"] == {"effort": "low",
                                  "format": {"type": "json_schema", "schema": schema}}
    assert p["thinking"] == {"type": "disabled"} and p["max_tokens"] == 500
    assert p["fallbacks"] == "default" and m.FALLBACK_BETA in p["betas"]


def test_a_model_that_must_think_gets_room_for_it(api):
    m, state = api
    m.complete([{"role": "user", "content": "hi"}], "claude-api/claude-opus-5-5",
               _cfg(), max_tokens=500)
    p = state["calls"][0]
    assert "thinking" not in p and p["max_tokens"] == 500 + m.THINKING_ROOM


def test_aliases_resolve_to_the_newest_model_of_the_family(api):
    m, state = api
    m.complete([{"role": "user", "content": "x"}], "claude-api/haiku", _cfg())
    assert state["calls"][0]["model"] == "claude-haiku-5-5"


def test_aliases_fall_back_when_the_model_list_fails(api, monkeypatch):
    m, state = api

    def boom():
        raise RuntimeError("offline")

    state["models"] = None
    monkeypatch.setattr(m, "_client", lambda c, asynchronous=False: NS(
        beta=NS(messages=NS(create=lambda **p: state["calls"].append(p) or _msg())),
        models=NS(list=boom)))
    m.complete([{"role": "user", "content": "x"}], "claude-api/opus", _cfg())
    assert state["calls"][0]["model"] == m.FALLBACK_IDS["opus"]


def test_a_call_that_could_pass_the_allowance_is_never_sent(api):
    """The point of the whole module: nothing is billed past the free credit."""
    m, state = api
    cfg = _cfg(claude_api_monthly_credit_usd=1.0)
    m._add_spend(cfg, 0.95)
    with pytest.raises(m.ClaudeApiBudgetError, match="renews"):
        m.complete([{"role": "user", "content": "x"}], "claude-api/haiku", cfg)
    assert state["calls"] == []


def test_no_allowance_means_no_calls(api):
    m, state = api
    with pytest.raises(m.ClaudeApiBudgetError, match="no monthly allowance"):
        m.complete([{"role": "user", "content": "x"}], "claude-api/haiku",
                   _cfg(claude_api_monthly_credit_usd=0))
    assert state["calls"] == []


def test_a_low_balance_stops_the_provider_for_the_period(api):
    import anthropic

    m, state = api
    import httpx2
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    state["error"] = anthropic.BadRequestError(
        "Your credit balance is too low to access the Anthropic API.",
        response=httpx2.Response(400, request=req), body=None)
    with pytest.raises(m.ClaudeApiBudgetError, match="too low"):
        m.complete([{"role": "user", "content": "x"}], "claude-api/haiku", _cfg())
    state["error"] = None
    with pytest.raises(m.ClaudeApiBudgetError, match="credit balance too low"):
        m.complete([{"role": "user", "content": "x"}], "claude-api/haiku", _cfg())
    assert len(state["calls"]) == 1, "tried again after the API said the credit was gone"


def test_a_refusal_is_an_error_not_an_empty_answer(api):
    m, state = api
    state["reply"] = _msg(text="", stop="refusal")
    with pytest.raises(m.ClaudeApiError, match="declined"):
        m.complete([{"role": "user", "content": "x"}], "claude-api/haiku", _cfg())


def test_off_or_unsigned_says_what_to_do(api, monkeypatch, tmp_path):
    m, _ = api
    with pytest.raises(m.ClaudeApiUnavailable, match="config claude-api on"):
        m.complete([{"role": "user", "content": "x"}], "claude-api/haiku", Config())
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "none"))
    with pytest.raises(m.ClaudeApiAuthError, match="ant auth login"):
        m.complete([{"role": "user", "content": "x"}], "claude-api/haiku", _cfg())


def test_an_ant_sign_in_counts_as_credentials(monkeypatch, tmp_path):
    from llm_sidecar import claude_api as m

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    (tmp_path / "credentials").mkdir()
    (tmp_path / "credentials" / "default.json").write_text("{}")
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
    assert m.credential_source() == "ant auth login"


def test_the_period_follows_the_billing_day():
    from llm_sidecar import claude_api as m

    cfg = Config(claude_api_reset_day=28)
    assert m.period_start(cfg, dt.date(2026, 11, 26)) == dt.date(2026, 10, 28)
    assert m.next_reset(cfg, dt.date(2026, 11, 26)) == dt.date(2026, 11, 28)
    feb = Config(claude_api_reset_day=31)
    assert m.period_start(feb, dt.date(2026, 3, 5)) == dt.date(2026, 2, 28)


def test_a_new_period_starts_from_zero(api, monkeypatch):
    m, _ = api
    cfg = _cfg()
    m._add_spend(cfg, 5.0)
    monkeypatch.setattr(m, "period_start", lambda c, today=None: dt.date(2099, 1, 1))
    assert m.spend(cfg)["spent_usd"] == 0.0


def test_an_unknown_model_is_priced_high_so_the_guard_errs_safe():
    from llm_sidecar import claude_api as m

    assert m.price("claude-future-9") == m.UNKNOWN_PRICE
    assert m.price("claude-haiku-5-5", 200_000) == (0.50, 2.50)


def test_the_prefix_does_not_steal_openrouters_anthropic_models(monkeypatch):
    """OpenRouter names its Claude models anthropic/...; those must still go
    to OpenRouter."""
    from llm_sidecar import client

    url, wire, _ = client._route("anthropic/claude-sonnet-4.5", Config(openrouter_api_key="k"))
    assert url == client.OPENROUTER_CHAT_URL and wire == "anthropic/claude-sonnet-4.5"


def test_the_client_dispatches_claude_api_ids(monkeypatch):
    from llm_sidecar import claude_api, client

    seen = {}
    monkeypatch.setattr(claude_api, "complete",
                        lambda msgs, model, cfg, **kw: seen.setdefault("model", model))
    client.complete([{"role": "user", "content": "x"}], "claude-api/haiku", Config())
    assert seen["model"] == "claude-api/haiku"


def test_streaming_yields_tokens_then_usage(api, monkeypatch):
    m, _ = api

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        @property
        def text_stream(self):
            async def gen():
                for t in ("he", "llo"):
                    yield t
            return gen()

        async def get_final_message(self):
            return _msg(text="hello")

    monkeypatch.setattr(m, "_client", lambda c, asynchronous=False: NS(
        beta=NS(messages=NS(stream=lambda **p: Stream())),
        models=NS(list=lambda: [])))

    async def run():
        return [e async for e in m.stream([{"role": "user", "content": "x"}],
                                          "claude-api/claude-haiku-5-5", _cfg())]

    events = asyncio.run(run())
    assert [e["text"] for e in events if e["type"] == "token"] == ["he", "llo"]
    assert events[-1]["type"] == "usage" and events[-1]["usage"].cost_usd > 0


def test_neither_claude_provider_is_cached_as_deterministic():
    from llm_sidecar import _is_claude

    assert _is_claude("claude-api/haiku") and _is_claude("claude/haiku")
    assert not _is_claude("anthropic/claude-sonnet-4.5")


# ── the subscription's usage windows ─────────────────────────────────────

# Shape as Claude Code 2.1 reports it in stream-json.
WINDOW_EVENT = {"type": "rate_limit_event", "rate_limit_info": {
    "status": "allowed_warning", "resetsAt": None, "rateLimitType": "seven_day",
    "utilization": 0.83, "overageStatus": "rejected", "overageEnabled": False,
    "unifiedWindows": {
        "five_hour": {"utilization": 0.72, "resetsAt": None, "status": "allowed"},
        "seven_day": {"utilization": 0.83, "resetsAt": None, "status": "allowed_warning"},
    }}}


def _window_event(status_7d="allowed_warning", in_s=3600):
    ev = json.loads(json.dumps(WINDOW_EVENT))
    info = ev["rate_limit_info"]
    soon = time.time() + in_s
    info["resetsAt"] = soon
    info["status"] = status_7d
    for w in info["unifiedWindows"].values():
        w["resetsAt"] = soon
    info["unifiedWindows"]["seven_day"]["status"] = status_7d
    return ev


@pytest.fixture
def cli(monkeypatch, tmp_path):
    from llm_sidecar import claude_cli

    exe = tmp_path / "claude"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: str(exe))
    monkeypatch.setattr(claude_cli, "_workdir", lambda: str(tmp_path))
    out = {"lines": [], "runs": 0}

    def fake_run(argv, prompt, config, timeout):
        out["runs"] += 1
        return "\n".join(json.dumps(x) for x in out["lines"])

    monkeypatch.setattr(claude_cli, "_run", fake_run)
    return claude_cli, out


def _result(**kw):
    return {"type": "result", "is_error": False, "result": "ok", "total_cost_usd": 0.003,
            "usage": {"input_tokens": 2, "output_tokens": 1}, **kw}


def test_a_claude_call_records_the_usage_windows(cli):
    from llm_sidecar import accounts

    m, out = cli
    out["lines"] = [{"type": "system", "subtype": "init", "apiKeySource": "none"},
                    _window_event(), _result()]
    done = m.complete([{"role": "user", "content": "x"}], "claude/haiku",
                      Config(claude_cli_enabled=True))
    assert done.text == "ok" and done.usage.cost_usd == 0.0
    e = accounts._claude_subscription(Config(claude_cli_enabled=True), refresh=False)
    assert e["state"] == "warn"
    assert "weekly 83% used" in e["summary"] and "5-hour 72% used" in e["summary"]


def test_a_call_billed_to_a_key_is_counted_as_spend(cli):
    """The env is stripped of keys, but an `ant` sign-in is a file, not an
    env var. The CLI says which credential it used; believe it."""
    m, out = cli
    out["lines"] = [{"type": "system", "subtype": "init", "apiKeySource": "ant auth login"},
                    _result()]
    done = m.complete([{"role": "user", "content": "x"}], "claude/haiku",
                      Config(claude_cli_enabled=True))
    assert done.usage.cost_usd == pytest.approx(0.003)


def test_an_exhausted_subscription_is_refused_without_starting_claude(cli):
    from llm_sidecar import accounts

    m, out = cli
    accounts.note_claude_window(_window_event("rejected")["rate_limit_info"])
    with pytest.raises(m.ClaudeCliLimitError, match="weekly limit reached"):
        m.complete([{"role": "user", "content": "x"}], "claude/haiku",
                   Config(claude_cli_enabled=True))
    assert out["runs"] == 0


def test_a_window_that_has_reset_no_longer_blocks(cli):
    from llm_sidecar import accounts

    m, out = cli
    accounts.note_claude_window(_window_event("rejected", in_s=-60)["rate_limit_info"])
    out["lines"] = [_result()]
    assert m.complete([{"role": "user", "content": "x"}], "claude/haiku",
                      Config(claude_cli_enabled=True)).text == "ok"
    e = accounts._claude_subscription(Config(claude_cli_enabled=True), refresh=False)
    assert e["state"] == "ok" and "reset" in e["summary"]


def test_old_single_envelope_output_still_parses(cli):
    m, out = cli
    out["lines"] = [{"is_error": False, "result": "plain", "usage": {}}]
    assert m.complete([{"role": "user", "content": "x"}], "claude/haiku",
                      Config(claude_cli_enabled=True)).text == "plain"


# ── openrouter ───────────────────────────────────────────────────────────


@pytest.fixture
def openrouter(monkeypatch):
    from llm_sidecar import accounts

    accounts._openrouter.update(at=0.0, key=None, data=None, error=None)
    reply = {"status": 200, "data": {
        "usage_monthly": 1.234, "limit": 10, "limit_remaining": 4.5, "limit_reset": "monthly",
        "is_free_tier": False,
        "free_model_daily_requests": {"used": 1000, "limit": 1000, "remaining": 0}}}
    calls = []

    def get(url, headers=None, timeout=None):
        calls.append(url)
        if "openrouter" not in url:
            return NS(status_code=200, json=lambda: {"models": [{"name": "a"}]})
        return NS(status_code=reply["status"], json=lambda: {"data": reply["data"]})

    monkeypatch.setattr(accounts.httpx, "get", get)
    return accounts, reply, calls


def test_openrouter_reports_spend_limit_and_free_requests(openrouter):
    accounts, _, _ = openrouter
    e = accounts._openrouter_entry(Config(openrouter_api_key="k"), refresh=False)
    assert "$1.23 spent this month" in e["summary"]
    assert "$4.50 of $10.00 key limit left" in e["summary"]
    assert "0 of 1000 free-model requests left today" in e["summary"]
    assert e["state"] == "warn" and e["source"] == "live"


def test_openrouter_is_read_once_per_few_minutes(openrouter):
    accounts, _, calls = openrouter
    cfg = Config(openrouter_api_key="k")
    accounts._openrouter_entry(cfg, refresh=False)
    accounts._openrouter_entry(cfg, refresh=False)
    assert len(calls) == 1
    accounts._openrouter_entry(cfg, refresh=True)
    assert len(calls) == 2


def test_free_requests_used_up_sends_the_picker_straight_to_local(openrouter, monkeypatch):
    from llm_sidecar import catalogue, picker
    from llm_sidecar.types import ModelInfo

    accounts, _, _ = openrouter
    cfg = Config(openrouter_api_key="k")
    monkeypatch.setattr(catalogue, "openrouter_models",
                        lambda c, **k: [ModelInfo(id="meta/llama:free", name="x")])
    monkeypatch.setattr(catalogue, "ollama_models",
                        lambda c: [ModelInfo(id="ollama/llama3.2:3b", name="y", size_bytes=1)])
    assert "meta/llama:free" in picker.candidates(cfg, "free")   # no reading yet
    accounts._openrouter_entry(cfg, refresh=False)               # reading: 0 left
    assert picker.candidates(cfg, "free") == ["ollama/llama3.2:3b"]


def test_a_rejected_key_is_said_plainly(openrouter):
    accounts, reply, _ = openrouter
    reply["status"] = 401
    e = accounts._openrouter_entry(Config(openrouter_api_key="k"), refresh=True)
    assert e["state"] == "unknown" and "rejected the key" in e["summary"]


# ── the whole report ─────────────────────────────────────────────────────


def test_the_report_covers_every_configured_provider(openrouter, api):
    accounts, _, _ = openrouter
    m, _ = api
    cfg = Config(openrouter_api_key="k", claude_api_enabled=True)
    m._add_spend(cfg, 12.5)
    r = accounts.report(cfg)
    by = {e["provider"]: e for e in r["providers"]}
    assert set(by) == {"ollama", "openrouter", "claude-api"}
    assert by["ollama"]["summary"].startswith("1 models")
    assert by["claude-api"]["source"] == "estimated"
    assert by["claude-api"]["spent_usd"] == 12.5 and by["claude-api"]["state"] == "ok"
    assert "$12.50 of $100.00 used" in by["claude-api"]["summary"]


def test_one_failing_provider_does_not_sink_the_report(openrouter, monkeypatch):
    accounts, _, _ = openrouter

    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(accounts, "_openrouter_entry", boom)
    r = accounts.report(Config(openrouter_api_key="k"))
    assert [e["provider"] for e in r["providers"]] == ["ollama"]


def test_ask_takes_a_model(monkeypatch, capsys):
    """The README has always said `ask --model ...`; the flag was missing."""
    from llm_sidecar import Sidecar, cli
    from llm_sidecar.types import Completion

    seen = {}

    def complete(self, prompt, **kw):
        seen.update(kw)
        return Completion(text="ok", model=kw["model"])

    monkeypatch.setattr(Sidecar, "complete", complete)
    assert cli.main(["ask", "--model", "claude-api/haiku", "-q", "hi"]) == 0
    assert seen["model"] == "claude-api/haiku" and capsys.readouterr().out.strip() == "ok"


# ── choosing models and defaults ─────────────────────────────────────────


def test_cli_sets_and_clears_a_tier_default():
    from llm_sidecar import cli, config as config_mod

    assert cli.main(["config", "tier", "powerful", "claude/opus"]) == 0
    assert config_mod.load().models == {"powerful": "claude/opus"}
    assert cli.main(["config", "tier", "powerful", "auto"]) == 0
    assert config_mod.load().models == {}
    assert cli.main(["config", "tier", "turbo", "x"]) == 1


def test_key_source_says_where_the_key_lives(monkeypatch):
    from llm_sidecar import config as config_mod

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert config_mod.key_source(Config()) is None
    assert config_mod.key_source(Config(openrouter_api_key="k1")) == "session"
    config_mod.save(Config(openrouter_api_key="k1"), include_api_key=True)
    assert config_mod.key_source(Config(openrouter_api_key="k1")) == "file"
    monkeypatch.setenv("OPENROUTER_API_KEY", "k1")
    assert config_mod.key_source(Config(openrouter_api_key="k1")) == "env"


def test_claude_api_settings_from_the_dashboard_are_saved():
    from fastapi.testclient import TestClient
    from llm_sidecar import config as config_mod, daemon

    app = daemon.create_app(Config())
    r = TestClient(app, base_url="http://127.0.0.1").post("/config/claude-api",
                             json={"enabled": True, "monthly_credit_usd": 50, "reset_day": 18}).json()
    assert r["allowance_usd"] == 50 and r["resets"].endswith("-18")
    saved = config_mod.load()
    assert saved.claude_api_enabled and saved.claude_api_monthly_credit_usd == 50
    assert saved.claude_api_reset_day == 18


def test_delegate_over_mcp_can_name_a_model(monkeypatch):
    from llm_sidecar import mcp_server
    from llm_sidecar.types import Completion

    seen = {}

    class Fake:
        def complete(self, prompt, **kw):
            seen.update(kw)
            return Completion(text="ok", model=kw.get("model") or "auto")

    monkeypatch.setattr(mcp_server, "sidecar", lambda: Fake())
    assert mcp_server.delegate("x", model="claude-api/haiku")["model"] == "claude-api/haiku"
    mcp_server.delegate("x")
    assert seen["model"] is None


def test_every_model_list_on_the_dashboard_includes_every_provider():
    """The chat picker used to list Ollama and OpenRouter only, so neither
    Claude provider could be chosen there even when switched on."""
    from pathlib import Path
    import llm_sidecar

    html = (Path(llm_sidecar.__file__).parent / "ui" / "index.html").read_text(encoding="utf-8")
    for owner in ('"ollama"', '"claude-cli"', '"anthropic"', '"openrouter"'):
        assert owner in html.split("const PROVIDERS")[1].split("];")[0]
    assert 'id="cfg-tier-powerful"' in html and 'id="cfg-api"' in html
