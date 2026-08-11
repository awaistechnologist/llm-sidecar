"""Tests for llm_sidecar.

Everything here runs offline. Network-touching paths (the catalogue fetch, the
pretest, real completions) are stubbed — they're verified manually against
live providers, but they don't belong in a suite that has to pass on a plane.

    python -m pytest llm_sidecar/tests -q
"""

from __future__ import annotations

import json

import pytest

from llm_sidecar import Sidecar
from llm_sidecar.config import Config
from llm_sidecar.types import ModelInfo, NoWorkingModel, SidecarError


# ── fixtures ──────────────────────────────────────────────────────────────────

CATALOGUE = [
    ModelInfo("free/a:free", "Free A", context_length=8000),
    ModelInfo("free/b:free", "Free B", context_length=32000),
    ModelInfo("cheap/x", "Cheap X", prompt_price_per_million=0.5, completion_price_per_million=0.5),
    ModelInfo("cheap/y", "Cheap Y", prompt_price_per_million=0.2, completion_price_per_million=0.2),
    ModelInfo("best/z", "Best Z", prompt_price_per_million=15.0, completion_price_per_million=30.0),
    ModelInfo("acme/ocr-reader", "OCR", context_length=4000),  # specialised, must be filtered
]


@pytest.fixture
def cfg():
    return Config(openrouter_api_key="test-key", models={}, default_budget="free")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Nothing in this file may make a real HTTP call."""
    from llm_sidecar import catalogue

    monkeypatch.setattr(catalogue, "openrouter_models", lambda config, force_refresh=False: list(CATALOGUE))
    monkeypatch.setattr(catalogue, "ollama_models", lambda config: [ModelInfo("ollama/local-1", "local-1 (local)")])


# ── config ────────────────────────────────────────────────────────────────────

def test_env_overrides_file(monkeypatch, tmp_path):
    from llm_sidecar import config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.json")
    (tmp_path / "config.json").write_text(json.dumps({"default_budget": "cheap", "daemon_port": 9999}))
    monkeypatch.setenv("LLM_SIDECAR_BUDGET", "best")

    c = config_mod.load()
    assert c.default_budget == "best"   # env wins over file
    assert c.daemon_port == 9999        # file still applies where env is silent


def test_kwargs_beat_everything(monkeypatch, tmp_path):
    from llm_sidecar import config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "nope.json")
    monkeypatch.setenv("LLM_SIDECAR_BUDGET", "best")
    assert config_mod.load(default_budget="free").default_budget == "free"


def test_save_omits_api_key_by_default(monkeypatch, tmp_path):
    from llm_sidecar import config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.json")

    config_mod.save(Config(openrouter_api_key="sk-secret"))
    written = json.loads((tmp_path / "config.json").read_text())
    assert "openrouter_api_key" not in written

    config_mod.save(Config(openrouter_api_key="sk-secret"), include_api_key=True)
    assert json.loads((tmp_path / "config.json").read_text())["openrouter_api_key"] == "sk-secret"


def test_broken_config_file_does_not_crash(monkeypatch, tmp_path):
    from llm_sidecar import config as config_mod

    bad = tmp_path / "config.json"
    bad.write_text("{not json")
    monkeypatch.setattr(config_mod, "CONFIG_FILE", bad)
    assert config_mod.load().default_budget == "free"


# ── picker ────────────────────────────────────────────────────────────────────

def test_budget_buckets_are_disjoint(cfg):
    from llm_sidecar import picker

    free = picker.candidates(cfg, "free")
    cheap = picker.candidates(cfg, "cheap")
    best = picker.candidates(cfg, "best")
    assert "free/a:free" in free and "free/a:free" not in cheap
    assert "cheap/x" in cheap and "cheap/x" not in best
    assert "best/z" in best and "best/z" not in cheap


def test_specialised_models_excluded(cfg):
    from llm_sidecar import picker

    assert not any("ocr" in c for c in picker.candidates(cfg, "free"))


def test_ollama_first_without_key_cloud_first_with(cfg):
    from llm_sidecar import picker

    with_key = picker.candidates(cfg, "free")
    without = picker.candidates(Config(openrouter_api_key=None), "free")
    assert not with_key[0].startswith("ollama/")
    assert without[0].startswith("ollama/")


def test_cheap_sorted_by_price(cfg):
    from llm_sidecar import picker

    c = picker.candidates(cfg, "cheap")
    assert c.index("cheap/y") < c.index("cheap/x")


def test_pick_skips_failures_and_records_attempts(cfg, monkeypatch):
    from llm_sidecar import picker

    # free/b sorts first (largest context), so failing it proves we walk past
    # a throttled candidate rather than giving up on it.
    def fake_pretest(model_id, config):
        return (model_id == "free/a:free", None if model_id == "free/a:free" else "HTTP 429")

    monkeypatch.setattr(picker, "pretest", fake_pretest)
    p = picker.pick(cfg, "free")
    assert p.model_id == "free/a:free"
    assert p.attempts[0] == {"id": "free/b:free", "ok": False, "reason": "HTTP 429"}


def test_pick_raises_when_all_fail(cfg, monkeypatch):
    from llm_sidecar import picker

    monkeypatch.setattr(picker, "pretest", lambda m, c: (False, "HTTP 429"))
    with pytest.raises(NoWorkingModel) as e:
        picker.pick(cfg, "free")
    assert e.value.attempts


def test_pick_excludes(cfg, monkeypatch):
    from llm_sidecar import picker

    monkeypatch.setattr(picker, "pretest", lambda m, c: (True, None))
    first = picker.pick(cfg, "free").model_id
    assert picker.pick(cfg, "free", exclude={first}).model_id != first


def test_pool_diversifies_on_cloud(cfg, monkeypatch):
    from llm_sidecar import picker

    monkeypatch.setattr(picker, "pretest", lambda m, c: (True, None))
    pool = picker.pick_pool(cfg, "free")
    assert set(pool) == {"fast", "balanced", "powerful"}
    assert len({p.model_id for p in pool.values()}) > 1


def test_pool_collapses_when_local_only(monkeypatch):
    from llm_sidecar import picker

    monkeypatch.setattr(picker, "pretest", lambda m, c: (True, None))
    pool = picker.pick_pool(Config(openrouter_api_key=None), "free")
    assert len({p.model_id for p in pool.values()}) == 1


def test_unknown_budget_rejected(cfg):
    from llm_sidecar import picker

    with pytest.raises(NoWorkingModel):
        picker.pick(cfg, "lavish")


# ── facade ────────────────────────────────────────────────────────────────────

def test_pinned_tier_skips_picking(monkeypatch):
    from llm_sidecar import picker

    def boom(*a, **k):
        raise AssertionError("should not pick when a tier is pinned")

    monkeypatch.setattr(picker, "pick", boom)
    sc = Sidecar(Config(models={"fast": "ollama/pinned"}))
    assert sc.model_for("fast") == "ollama/pinned"


def test_budget_is_part_of_the_resolution_cache(cfg, monkeypatch):
    """Regression: a `free` request used to poison the cache so a later
    `best` request silently got the cheap model."""
    from llm_sidecar import picker

    calls = []

    def fake_pick(config, budget=None, **k):
        calls.append(budget)
        from llm_sidecar.types import Pick
        return Pick(model_id=f"model-for-{budget}", model_name=budget or "?")

    monkeypatch.setattr(picker, "pick", fake_pick)
    sc = Sidecar(cfg)
    assert sc.model_for(budget="free") == "model-for-free"
    assert sc.model_for(budget="best") == "model-for-best"
    assert calls == ["free", "best"]

    sc.model_for(budget="free")  # cached, no third call
    assert len(calls) == 2


def test_rotation_drops_failed_model(cfg, monkeypatch):
    from llm_sidecar import client, picker
    from llm_sidecar.types import Completion, Pick

    handed = iter(["bad/model", "good/model"])
    monkeypatch.setattr(picker, "pick", lambda *a, **k: Pick(model_id=next(handed), model_name="m"))

    def fake_complete(prompt, model, config, **k):
        if model == "bad/model":
            raise RuntimeError("429 forever")
        return Completion(text="ok", model=model)

    monkeypatch.setattr(client, "complete", fake_complete)
    sc = Sidecar(cfg)
    assert sc.complete("hi").model == "good/model"
    assert "bad/model" in sc._failed


def test_pinned_model_failure_is_not_rotated(cfg, monkeypatch):
    """An explicit model pin is the caller's decision — silently substituting
    a different model would hide the failure."""
    from llm_sidecar import client

    def always_fails(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(client, "complete", always_fails)
    with pytest.raises(RuntimeError):
        Sidecar(cfg).complete("hi", model="ollama/pinned")


# ── verify ────────────────────────────────────────────────────────────────────

def test_judge_json_survives_fences_and_prose():
    from llm_sidecar.verify import _parse_json

    assert _parse_json('```json\n{"results": [1]}\n```')["results"] == [1]
    assert _parse_json('Sure!\n{"results": [2]}\nHope that helps')["results"] == [2]
    with pytest.raises(SidecarError):
        _parse_json("no json at all")


def test_unknown_verdict_becomes_unverified(cfg, monkeypatch):
    from llm_sidecar import verify as verify_mod

    monkeypatch.setattr(verify_mod, "gather_evidence", lambda c, cfg, **k: [{"title": "t", "snippet": "s", "url": "http://e"}])
    monkeypatch.setattr(verify_mod, "_judge_batch", lambda b, m, c: [{"claim": 1, "verdict": "definitely-true", "note": "n"}])

    out = verify_mod.verify_claims(["a claim"], cfg, model="x")
    assert out[0].verdict == "unverified"
    assert out[0].sources == ["http://e"]


def test_judge_failure_degrades_not_crashes(cfg, monkeypatch):
    from llm_sidecar import verify as verify_mod

    monkeypatch.setattr(verify_mod, "gather_evidence", lambda c, cfg, **k: [])

    def boom(*a, **k):
        raise RuntimeError("judge exploded")

    monkeypatch.setattr(verify_mod, "_judge_batch", boom)
    out = verify_mod.verify_claims(["a", "b"], cfg, model="x")
    assert [v.verdict for v in out] == ["unverified", "unverified"]


def test_claim_limit_enforced(cfg):
    from llm_sidecar import verify as verify_mod

    with pytest.raises(SidecarError):
        verify_mod.verify_claims([f"claim {i}" for i in range(50)], cfg, model="x")


def test_empty_claims_short_circuits(cfg):
    from llm_sidecar import verify as verify_mod

    assert verify_mod.verify_claims(["", "   "], cfg, model="x") == []


# ── search ────────────────────────────────────────────────────────────────────

def test_auto_prefers_searxng_when_up(cfg, monkeypatch):
    from llm_sidecar import search as search_mod
    from llm_sidecar.search import searxng

    monkeypatch.setattr(searxng, "available", lambda config: True)
    assert search_mod.resolve_provider(cfg) is searxng

    monkeypatch.setattr(searxng, "available", lambda config: False)
    assert search_mod.resolve_provider(cfg).__name__.endswith("ddg")


def test_unknown_provider_rejected():
    from llm_sidecar import search as search_mod

    with pytest.raises(SidecarError):
        search_mod.resolve_provider(Config(search_provider="altavista"))


def test_searxng_empty_falls_back_to_ddg(cfg, monkeypatch):
    from llm_sidecar import search as search_mod
    from llm_sidecar.search import ddg, searxng
    from llm_sidecar.types import SearchResult

    monkeypatch.setattr(cfg, "search_provider", "searxng")
    monkeypatch.setattr(searxng, "search", lambda *a, **k: [])
    monkeypatch.setattr(ddg, "search", lambda *a, **k: [SearchResult("t", "u", "s")])
    assert len(search_mod.search("q", cfg)) == 1


def test_read_url_strips_markup(cfg, monkeypatch):
    import httpx

    from llm_sidecar import search as search_mod

    html_doc = "<html><head><style>b{}</style></head><body><p>Hello</p><script>x()</script><p>World &amp; co</p></body></html>"
    monkeypatch.setattr(
        httpx, "get",
        lambda url, *a, **k: httpx.Response(
            200, text=html_doc, headers={"content-type": "text/html"},
            request=httpx.Request("GET", url),
        ),
    )
    text = search_mod.read_url("http://example.com", cfg)
    assert "Hello" in text and "World & co" in text
    assert "<p>" not in text and "x()" not in text


def test_read_url_rejects_binary(cfg, monkeypatch):
    import httpx

    from llm_sidecar import search as search_mod

    monkeypatch.setattr(
        httpx, "get",
        lambda url, *a, **k: httpx.Response(
            200, content=b"\x00", headers={"content-type": "image/png"},
            request=httpx.Request("GET", url),
        ),
    )
    with pytest.raises(SidecarError):
        search_mod.read_url("http://example.com/x.png", cfg)


# ── daemon ────────────────────────────────────────────────────────────────────

@pytest.fixture
def api(cfg, monkeypatch):
    from fastapi.testclient import TestClient

    from llm_sidecar import client as client_mod
    from llm_sidecar import daemon, picker
    from llm_sidecar.types import Completion, Pick, Usage

    monkeypatch.setattr(picker, "pick", lambda config, budget=None, **k: Pick(f"auto/{budget}", "auto"))
    monkeypatch.setattr(
        client_mod, "complete",
        lambda prompt, model, config, system=None, **k: Completion(
            text=f"[{system or 'no-system'}] {prompt}",
            model=model,
            usage=Usage(1, 2, 3, 0.001),
        ),
    )
    return TestClient(daemon.create_app(cfg))


def test_unknown_model_name_is_ignored(api):
    """The whole point: a tool that hardcodes gpt-4o gets routed anyway."""
    r = api.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "auto/free"
    assert body["x_sidecar"]["requested_model"] == "gpt-4o"


def test_aliases_and_explicit_ids_route(api):
    assert api.post("/v1/chat/completions", json={"model": "best", "messages": [{"role": "user", "content": "x"}]}).json()["model"] == "auto/best"
    assert api.post("/v1/chat/completions", json={"model": "ollama/thing", "messages": [{"role": "user", "content": "x"}]}).json()["model"] == "ollama/thing"


def test_response_has_the_standard_shape(api):
    body = api.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]}).json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert set(body["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"}
    assert body["x_sidecar"]["cost_usd"] == 0.001


def test_system_messages_are_split_out(api):
    body = api.post("/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "system", "content": "BE TERSE"}, {"role": "user", "content": "hi"}],
    }).json()
    assert body["choices"][0]["message"]["content"] == "[BE TERSE] hi"


def test_empty_messages_rejected(api):
    assert api.post("/v1/chat/completions", json={"model": "auto", "messages": []}).status_code == 400


def test_no_working_model_is_503(api, monkeypatch):
    from llm_sidecar import picker

    def boom(*a, **k):
        raise NoWorkingModel("nothing works")

    monkeypatch.setattr(picker, "pick", boom)
    r = api.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503


def test_models_endpoint_lists_aliases_first(api):
    data = api.get("/v1/models").json()["data"]
    assert [m["id"] for m in data[:4]] == ["auto", "fast", "balanced", "powerful"]


def test_token_gate(cfg, monkeypatch):
    from fastapi.testclient import TestClient

    from llm_sidecar import daemon

    cfg.daemon_token = "s3cret"
    c = TestClient(daemon.create_app(cfg))
    assert c.get("/v1/models").status_code == 401
    assert c.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/v1/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_health_needs_no_token(cfg):
    from fastapi.testclient import TestClient

    from llm_sidecar import daemon

    cfg.daemon_token = "s3cret"
    assert TestClient(daemon.create_app(cfg)).get("/health").status_code == 200
