"""Tests for llm_sidecar.

Everything here runs offline. Network-touching paths (the catalogue fetch, the
pretest, real completions) are stubbed — they're verified manually against
live providers, but they don't belong in a suite that has to pass on a plane.

    python -m pytest llm_sidecar/tests -q
"""

from __future__ import annotations

import json
import re
import time

import pytest

from llm_sidecar import Sidecar, catalogue as _catalogue
from llm_sidecar.config import Config

# Captured before the autouse no-network fixture replaces it, so the test that
# measures memoisation can exercise the real implementation.
_REAL_CATALOGUE = _catalogue.openrouter_models
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
def cfg(tmp_path, monkeypatch):
    """A config with cache and ledger pointed at a temp dir, so tests never
    touch the developer's real cache or spend history."""
    from llm_sidecar import cache, ledger

    monkeypatch.setattr(cache, "_COMPLETIONS", tmp_path / "completions")
    monkeypatch.setattr(cache, "_SEARCHES", tmp_path / "searches")
    monkeypatch.setattr(ledger, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(ledger, "LEDGER_FILE", tmp_path / "data" / "usage.jsonl")
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
    written = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert "openrouter_api_key" not in written

    config_mod.save(Config(openrouter_api_key="sk-secret"), include_api_key=True)
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))["openrouter_api_key"] == "sk-secret"


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

    def fake_complete(messages, model, config, **k):
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

def test_verifier_json_survives_fences_and_prose():
    from llm_sidecar.verify import _parse_json

    assert _parse_json('```json\n{"results": [1]}\n```')["results"] == [1]
    assert _parse_json('Sure!\n{"results": [2]}\nHope that helps')["results"] == [2]
    with pytest.raises(SidecarError):
        _parse_json("no json at all")


def test_unknown_verdict_becomes_unverified(cfg, monkeypatch):
    from llm_sidecar import verify as verify_mod

    monkeypatch.setattr(verify_mod, "gather_evidence", lambda c, cfg, **k: [{"title": "t", "snippet": "s", "url": "http://e"}])
    monkeypatch.setattr(verify_mod, "_verify_batch", lambda b, m, c: [{"claim": 1, "verdict": "definitely-true", "note": "n"}])

    out = verify_mod.verify_claims(["a claim"], cfg, model="x")
    assert out[0].verdict == "unverified"
    assert out[0].sources == ["http://e"]


def test_verifier_failure_degrades_not_crashes(cfg, monkeypatch):
    from llm_sidecar import verify as verify_mod

    monkeypatch.setattr(verify_mod, "gather_evidence", lambda c, cfg, **k: [])

    def boom(*a, **k):
        raise RuntimeError("verifier exploded")

    monkeypatch.setattr(verify_mod, "_verify_batch", boom)
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


def test_extract_text_strips_markup():
    from llm_sidecar.search import extract_text

    doc = ("<html><head><style>b{}</style></head><body>"
           "<p>Hello</p><script>x()</script><p>World &amp; co</p></body></html>")
    text = extract_text(doc)
    assert "Hello" in text and "World & co" in text
    assert "<p>" not in text and "x()" not in text


def test_extract_text_drops_page_furniture():
    """Regression: reading a Wikipedia article returned its navigation menu
    and 170-language switcher before the first sentence of content."""
    from llm_sidecar.search import extract_text

    doc = ("<body><nav><a>Main page</a><a>Random article</a></nav>"
           "<header>Site header</header>"
           "<main><p>The actual content of the article goes here.</p></main>"
           "<footer>Privacy policy</footer></body>")
    text = extract_text(doc)
    assert "actual content" in text
    for chrome in ("Main page", "Random article", "Site header", "Privacy policy"):
        assert chrome not in text


def test_extract_text_prefers_the_content_region():
    from llm_sidecar.search import extract_text

    body = "Real article body. " * 40
    doc = f"<body><div>sidebar junk</div><article><p>{body}</p></article></body>"
    assert "sidebar junk" not in extract_text(doc)


def test_tiny_content_region_is_not_trusted():
    """A stray <main> must not throw the page away — keeping everything is a
    far better failure than returning an empty string."""
    from llm_sidecar.search import extract_text

    doc = "<body><main>x</main><p>" + ("The real content. " * 40) + "</p></body>"
    assert "The real content" in extract_text(doc)


def test_extract_text_truncates():
    from llm_sidecar.search import extract_text

    assert extract_text("<p>" + "a" * 500 + "</p>", max_chars=100).endswith("[truncated]")


def test_read_url_rejects_binary(cfg, monkeypatch):
    import httpx

    from llm_sidecar import search as search_mod

    class FakeStream:
        def __init__(self, *a, **k): pass
        def __enter__(self):
            return httpx.Response(200, content=b"\x00", headers={"content-type": "image/png"},
                                  request=httpx.Request("GET", "http://x"))
        def __exit__(self, *a): return False

    monkeypatch.setattr(httpx, "stream", FakeStream)
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
        lambda messages, model, config, **k: Completion(
            text=json.dumps(messages), model=model, usage=Usage(1, 2, 3, 0.001),
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


def test_messages_pass_through_intact(api):
    """Regression: the daemon used to flatten a conversation into one prompt,
    losing turn boundaries. Multi-turn must reach the model as-is."""
    convo = [
        {"role": "system", "content": "BE TERSE"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "and now?"},
    ]
    body = api.post("/v1/chat/completions", json={"model": "auto", "messages": convo}).json()
    assert json.loads(body["choices"][0]["message"]["content"]) == convo


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


# ── messages ──────────────────────────────────────────────────────────────────

def test_build_messages_shapes():
    from llm_sidecar.client import build_messages

    assert build_messages("hi") == [{"role": "user", "content": "hi"}]
    assert build_messages("hi", system="S") == [
        {"role": "system", "content": "S"}, {"role": "user", "content": "hi"}
    ]
    convo = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert build_messages(messages=convo) == convo
    # Both given: the prompt continues the conversation.
    assert build_messages("c", messages=convo)[-1] == {"role": "user", "content": "c"}
    with pytest.raises(ValueError):
        build_messages()


# ── cache ─────────────────────────────────────────────────────────────────────

def test_only_deterministic_requests_are_cached():
    from llm_sidecar import cache

    assert cache.cacheable(0.0)
    assert not cache.cacheable(0.7)


def test_cache_round_trip_and_temperature_split(cfg):
    from llm_sidecar import cache

    msgs = [{"role": "user", "content": "hi"}]
    assert cache.get_completion(cfg, msgs, "m", 100, 0.0) is None
    cache.put_completion(cfg, msgs, "m", 100, 0.0, {"text": "cached!"})
    assert cache.get_completion(cfg, msgs, "m", 100, 0.0)["text"] == "cached!"
    # A creative request must not be served the deterministic answer.
    assert cache.get_completion(cfg, msgs, "m", 100, 0.9) is None


def test_cache_respects_ttl(cfg):
    from llm_sidecar import cache

    msgs = [{"role": "user", "content": "hi"}]
    cache.put_completion(cfg, msgs, "m", 100, 0.0, {"text": "x"})
    cfg.cache_ttl_seconds = -1  # everything is instantly stale
    assert cache.get_completion(cfg, msgs, "m", 100, 0.0) is None


def test_cache_can_be_disabled(cfg):
    from llm_sidecar import cache

    cfg.cache_enabled = False
    msgs = [{"role": "user", "content": "hi"}]
    cache.put_completion(cfg, msgs, "m", 100, 0.0, {"text": "x"})
    assert cache.get_completion(cfg, msgs, "m", 100, 0.0) is None


def test_sidecar_serves_second_identical_call_from_cache(cfg, monkeypatch):
    from llm_sidecar import client, picker
    from llm_sidecar.types import Completion, Pick, Usage

    calls = []
    monkeypatch.setattr(picker, "pick", lambda *a, **k: Pick("m/1", "m"))

    def counted(messages, model, config, **k):
        calls.append(1)
        return Completion(text="answer", model=model, usage=Usage(1, 1, 2, 0.5))

    monkeypatch.setattr(client, "complete", counted)
    sc = Sidecar(cfg)
    first = sc.complete("q", temperature=0.0)
    second = sc.complete("q", temperature=0.0)

    assert len(calls) == 1
    assert first.text == second.text == "answer"
    assert not first.cached and second.cached


# ── ledger ────────────────────────────────────────────────────────────────────

def test_ledger_records_and_summarises(cfg):
    from llm_sidecar import ledger

    ledger.record("m/a", 10, 20, 0.5, operation="complete")
    ledger.record("m/a", 5, 5, 0.25, operation="verify")
    ledger.record("m/b", 1, 1, 0.0, cached=True)

    s = ledger.summary()
    assert s["calls"] == 3
    assert s["cost_usd"] == 0.75
    assert s["total_tokens"] == 42
    assert s["cached"] == 1
    assert s["by_model"][0]["model"] == "m/a"      # sorted by spend
    assert s["by_model"][0]["calls"] == 2


def test_ledger_skips_corrupt_lines(cfg):
    from llm_sidecar import ledger

    ledger.record("m/a", 1, 1, 0.1)
    with ledger.LEDGER_FILE.open("a", encoding="utf-8") as f:
        f.write("{not json\n\n")
    ledger.record("m/b", 1, 1, 0.2)
    assert ledger.summary()["calls"] == 2


def test_ledger_never_raises_on_bad_path(monkeypatch, tmp_path):
    from llm_sidecar import ledger

    # A file where the directory should be — writes must fail silently.
    blocker = tmp_path / "blocked"
    blocker.write_text("x")
    monkeypatch.setattr(ledger, "DATA_DIR", blocker)
    monkeypatch.setattr(ledger, "LEDGER_FILE", blocker / "usage.jsonl")
    ledger.record("m", 1, 1, 0.0)   # must not raise
    assert ledger.summary()["calls"] == 0


# ── hardware ──────────────────────────────────────────────────────────────────

def test_assess_verdicts():
    from llm_sidecar import hardware

    hw = {"total_ram_gib": 16.0}          # 13 GiB usable after headroom
    assert hardware.assess(2 * 1024**3, hw)["verdict"] == "fits"
    assert hardware.assess(11 * 1024**3, hw)["verdict"] == "tight"
    # Exactly at the edge counts as over: 12 GiB weights + 1 GiB KV > 13.
    assert hardware.assess(12 * 1024**3, hw)["verdict"] == "too_big"
    assert hardware.assess(40 * 1024**3, hw)["verdict"] == "too_big"
    assert hardware.assess(1024**3, {})["verdict"] == "unknown"


def test_bigger_context_costs_memory():
    from llm_sidecar import hardware

    small = hardware.requirement_gib(4 * 1024**3, context_tokens=2000)
    large = hardware.requirement_gib(4 * 1024**3, context_tokens=64000)
    assert large > small


def test_vram_wins_over_system_ram():
    from llm_sidecar import hardware

    assert hardware.usable_gib({"total_ram_gib": 64.0, "vram_gib": 8.0}) == 5.0


# ── ops ───────────────────────────────────────────────────────────────────────

class _Stub:
    """Minimal stand-in for Sidecar that returns a canned completion."""

    def __init__(self, text):
        self.text = text
        self.seen = {}

    def complete(self, prompt=None, **k):
        from llm_sidecar.types import Completion
        self.seen = {"prompt": prompt, **k}
        return Completion(text=self.text, model="stub")


def test_classify_rejects_invented_labels():
    from llm_sidecar import ops

    stub = _Stub('{"results": [{"item": 1, "label": "bug"}, {"item": 2, "label": "wombat"}]}')
    out = ops.classify(stub, ["crash on save", "make it blue"], ["bug", "feature"])
    assert out[0]["label"] == "bug"
    assert out[1]["label"] == "unknown"      # not silently accepted


def test_classify_handles_missing_entries():
    from llm_sidecar import ops

    stub = _Stub('{"results": [{"item": 1, "label": "a"}]}')
    out = ops.classify(stub, ["one", "two"], ["a", "b"])
    assert [o["label"] for o in out] == ["a", "unknown"]


def test_classify_needs_two_labels():
    from llm_sidecar import ops

    with pytest.raises(SidecarError):
        ops.classify(_Stub("{}"), ["x"], ["only"])


def test_extract_pins_the_schema():
    from llm_sidecar import ops

    stub = _Stub('{"total": "42", "surprise": "extra", "due": null}')
    out = ops.extract(stub, "some invoice", {"total": "the total", "due": "the due date"})
    assert out == {"total": "42", "due": None}   # extra key dropped, gap explicit


def test_ops_run_deterministically():
    from llm_sidecar import ops

    stub = _Stub("a summary")
    ops.summarise(stub, "long text")
    assert stub.seen["temperature"] == 0.0


def test_summarise_rejects_bad_style_and_empty():
    from llm_sidecar import ops

    with pytest.raises(SidecarError):
        ops.summarise(_Stub("x"), "text", style="interpretive-dance")
    with pytest.raises(SidecarError):
        ops.summarise(_Stub("x"), "   ")


def test_parse_json_handles_arrays_and_fences():
    from llm_sidecar.ops import parse_json_response

    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_response("[1, 2]") == {"result": [1, 2]}


# ── resolution TTL ────────────────────────────────────────────────────────────

def test_resolution_expires(cfg, monkeypatch):
    from llm_sidecar import picker
    from llm_sidecar.types import Pick

    picks = iter(["m/1", "m/2"])
    monkeypatch.setattr(picker, "pick", lambda *a, **k: Pick(next(picks), "m"))

    cfg.resolution_ttl_seconds = 900
    sc = Sidecar(cfg)
    assert sc.model_for() == "m/1"
    assert sc.model_for() == "m/1"          # cached

    cfg.resolution_ttl_seconds = -1          # everything instantly stale
    assert sc.model_for() == "m/2"           # re-verified


# ── parallel evidence ─────────────────────────────────────────────────────────

def test_gather_all_returns_every_claim(cfg, monkeypatch):
    from llm_sidecar import verify as verify_mod

    monkeypatch.setattr(verify_mod, "gather_evidence",
                        lambda c, cfg, **k: [{"title": c, "url": f"http://{c}", "snippet": "s"}])
    out = verify_mod.gather_all(["a", "b", "c"], cfg)
    assert set(out) == {"a", "b", "c"}


def test_one_failed_search_does_not_sink_the_batch(cfg, monkeypatch):
    from llm_sidecar import verify as verify_mod

    def flaky(claim, config, **k):
        if claim == "boom":
            raise RuntimeError("provider down")
        return [{"title": claim, "url": "http://x", "snippet": "s"}]

    monkeypatch.setattr(verify_mod, "gather_evidence", flaky)
    out = verify_mod.gather_all(["ok", "boom"], cfg)
    assert out["boom"] == []
    assert len(out["ok"]) == 1


# ── daemon: new endpoints ─────────────────────────────────────────────────────

def test_usage_and_cache_endpoints(api):
    assert api.get("/usage").status_code == 200
    assert "removed" in api.post("/cache/clear").json()


def test_verify_endpoint(api, monkeypatch):
    from llm_sidecar import verify as verify_mod
    from llm_sidecar.types import ClaimVerdict

    monkeypatch.setattr(
        verify_mod, "verify_claims",
        lambda claims, config, model=None, sidecar=None: [
            ClaimVerdict(claim=c, verdict="supported", note="n", sources=["u"]) for c in claims
        ],
    )
    body = api.post("/v1/verify", json={"claims": ["x", "y"]}).json()
    assert len(body["results"]) == 2
    assert body["results"][0]["verdict"] == "supported"


def test_receipt_reports_cache_hits(api):
    a = api.post("/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": "q"}], "temperature": 0.0}).json()
    b = api.post("/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": "q"}], "temperature": 0.0}).json()
    assert a["x_sidecar"]["cached"] is False
    assert b["x_sidecar"]["cached"] is True


def test_verify_prompt_covers_namesakes():
    """Regression: a search for "The Eiffel Tower is in Berlin" surfaced a
    replica, and the verifier graded the claim supported. The prompt now has to
    tell it that a namesake is not the subject."""
    from llm_sidecar.verify import VERIFY_SYSTEM

    lowered = VERIFY_SYSTEM.lower()
    assert "replica" in lowered
    assert "namesake" in lowered


# ── searxng service management ────────────────────────────────────────────────

@pytest.fixture
def instance(tmp_path, monkeypatch):
    from llm_sidecar import services

    monkeypatch.setattr(services, "INSTANCE_DIR", tmp_path / "searxng")
    return tmp_path / "searxng"


def test_install_writes_a_runnable_instance(instance):
    from llm_sidecar import services

    services.install(port=9999)
    assert (instance / "docker-compose.yml").exists()
    assert (instance / "settings.yml").exists()
    assert (instance / ".env").read_text(encoding="utf-8").strip() == "SEARXNG_PORT=9999"


def test_install_enables_json_output(instance):
    """Without this the API 403s and search silently falls back to DDG —
    the single reason shipping our own settings.yml is worth it."""
    from llm_sidecar import services

    services.install()
    settings = (instance / "settings.yml").read_text(encoding="utf-8")
    assert "formats:" in settings
    assert "json" in settings
    assert "limiter: false" in settings


def test_secret_key_is_generated_and_unique(instance, tmp_path, monkeypatch):
    from llm_sidecar import services

    services.install()
    first = (instance / "settings.yml").read_text(encoding="utf-8")
    assert "GENERATED_ON_FIRST_START" not in first

    other = tmp_path / "second"
    monkeypatch.setattr(services, "INSTANCE_DIR", other)
    services.install()
    assert (other / "settings.yml").read_text(encoding="utf-8") != first


def test_install_does_not_clobber_user_edits(instance):
    from llm_sidecar import services

    services.install()
    (instance / "settings.yml").write_text("# mine\n")
    services.install()
    assert (instance / "settings.yml").read_text(encoding="utf-8") == "# mine\n"
    services.install(force=True)
    assert (instance / "settings.yml").read_text(encoding="utf-8") != "# mine\n"


def test_compose_binds_loopback_only():
    """An instance with the bot limiter off must not be reachable off-box."""
    from llm_sidecar import services

    compose = (services.ASSETS / "docker-compose.yml").read_text(encoding="utf-8")
    assert "127.0.0.1:" in compose
    assert '"${SEARXNG_PORT:-8888}:8080"' not in compose


def test_port_is_read_from_configured_url():
    from llm_sidecar import services

    assert services._port_from(Config(searxng_url="http://localhost:7777")) == 7777
    assert services._port_from(Config(searxng_url="http://localhost:7777/")) == 7777
    assert services._port_from(Config(searxng_url="http://searx.example.com")) == 8888


def test_status_reports_a_reachable_instance_we_did_not_start(cfg, instance, monkeypatch):
    """Someone running SearXNG their own way is just as good as our container;
    reporting 'not installed' at something plainly answering would be wrong."""
    from llm_sidecar import services
    from llm_sidecar.search import searxng

    monkeypatch.setattr(searxng, "available", lambda config: True)
    monkeypatch.setattr(services.shutil, "which", lambda n: None)
    s = services.status(cfg)
    assert s["answering_json"] is True
    assert s["installed"] is False
    assert s["docker"] == "not installed"


def test_status_ignores_the_probe_cache(cfg, monkeypatch):
    """A stale 'unavailable' from earlier in the process must not make a
    running instance look dead."""
    from llm_sidecar import services
    from llm_sidecar.search import searxng

    searxng._probe_cache[cfg.searxng_url] = False
    monkeypatch.setattr(services.shutil, "which", lambda n: None)
    monkeypatch.setattr(
        services.httpx, "get",
        lambda *a, **k: httpx_ok(),
    )
    assert services.status(cfg)["answering_json"] is True


def httpx_ok():
    import httpx
    return httpx.Response(200, json={"results": [{"title": "t"}]},
                          request=httpx.Request("GET", "http://x"))


def test_missing_docker_gives_an_actionable_error(monkeypatch):
    from llm_sidecar import services

    monkeypatch.setattr(services.shutil, "which", lambda n: None)
    with pytest.raises(SidecarError) as e:
        services._docker()
    assert "SEARXNG_URL" in str(e.value)   # tells you the no-Docker way out


# ── efficiency: memoisation, parallel probing, concurrency ────────────────────

def test_catalogue_is_memoised(cfg, monkeypatch):
    """pick_pool() read and re-parsed a 400-model JSON file six times per call
    for data that cannot change mid-process."""
    from llm_sidecar import catalogue

    catalogue.forget()
    monkeypatch.setattr(catalogue, "openrouter_models", _REAL_CATALOGUE)
    reads = []
    monkeypatch.setattr(catalogue, "_read_cache",
                        lambda: (reads.append(1), ([ModelInfo("a/b", "b")], time.time()))[1])
    for _ in range(10):
        catalogue.openrouter_models(cfg)
    assert len(reads) == 1
    catalogue.forget()


def test_probing_is_parallel_but_keeps_priority_order(cfg, monkeypatch):
    """Concurrency must speed up the search, not change which model wins."""
    from llm_sidecar import picker

    order = picker.candidates(cfg, "free")
    winner = order[2]

    def slow(model_id, config):
        time.sleep(0.4)
        return (model_id == winner, None if model_id == winner else "HTTP 429")

    monkeypatch.setattr(picker, "pretest", slow)
    started = time.time()
    p = picker.pick(cfg, "free")
    elapsed = time.time() - started

    assert p.model_id == winner            # highest-priority working model
    assert elapsed < 1.0                   # one wave, not three sequential probes
    assert len(p.attempts) == 3            # every probe still recorded


def test_concurrent_resolution_picks_once(cfg, monkeypatch):
    """The daemon shares one Sidecar across a threadpool; twelve simultaneous
    requests must not each trigger their own pretest round."""
    import threading

    from llm_sidecar import picker
    from llm_sidecar.types import Pick

    calls = []

    def slow_pick(config, budget=None, **k):
        calls.append(1)
        time.sleep(0.2)
        return Pick("model/one", "m")

    monkeypatch.setattr(picker, "pick", slow_pick)
    sc = Sidecar(cfg)
    seen = []
    threads = [threading.Thread(target=lambda: seen.append(sc.model_for())) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert set(seen) == {"model/one"}      # everyone agrees
    assert len(sc.resolved) == 1           # and the map didn't get corrupted


def test_complete_many_preserves_order_and_survives_failures(cfg, monkeypatch):
    from llm_sidecar import client, picker
    from llm_sidecar.types import Completion, Pick

    monkeypatch.setattr(picker, "pick", lambda *a, **k: Pick("m/1", "m"))

    def flaky(messages, model, config, **k):
        text = messages[-1]["content"]
        if text == "boom":
            raise RuntimeError("nope")
        return Completion(text=text.upper(), model=model)

    monkeypatch.setattr(client, "complete", flaky)
    out = Sidecar(cfg).complete_many(["a", "boom", "c"], temperature=0.5)
    assert [c.text for c in out] == ["A", "", "C"]


def test_complete_many_empty():
    assert Sidecar(Config()).complete_many([]) == []


# ── retry-after ───────────────────────────────────────────────────────────────

def test_retry_after_is_honoured_and_clamped():
    import httpx

    from llm_sidecar.client import MAX_RETRY_AFTER, _retry_delay

    def resp(headers):
        return httpx.Response(429, headers=headers, request=httpx.Request("GET", "http://x"))

    assert _retry_delay(resp({}), 5.0) == 5.0
    assert _retry_delay(resp({"retry-after": "30"}), 5.0) == 30.0
    # A provider must not be able to shorten a backoff we chose.
    assert _retry_delay(resp({"retry-after": "1"}), 5.0) == 5.0
    # Nor stall us indefinitely.
    assert _retry_delay(resp({"retry-after": "9999"}), 5.0) == MAX_RETRY_AFTER
    # HTTP-date form falls back rather than crashing.
    assert _retry_delay(resp({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}), 5.0) == 5.0
    assert _retry_delay(None, 5.0) == 5.0


# ── cache eviction and page cache ─────────────────────────────────────────────

def test_evict_trims_oldest_first(cfg, tmp_path, monkeypatch):
    from llm_sidecar import cache

    d = tmp_path / "completions"
    d.mkdir(parents=True)
    monkeypatch.setattr(cache, "_COMPLETIONS", d)
    monkeypatch.setattr(cache, "_SEARCHES", tmp_path / "nope")
    monkeypatch.setattr(cache, "_PAGES", tmp_path / "nope2")

    for i in range(5):
        f = d / f"{i}.json"
        f.write_text("x" * 1000)
        import os
        os.utime(f, (1000 + i, 1000 + i))     # oldest first

    freed = cache.evict(max_bytes=2500)
    assert freed > 0
    survivors = sorted(f.name for f in d.glob("*.json"))
    assert "0.json" not in survivors          # oldest went
    assert "4.json" in survivors              # newest stayed


def test_evict_noop_under_budget(cfg, tmp_path, monkeypatch):
    from llm_sidecar import cache

    monkeypatch.setattr(cache, "_COMPLETIONS", tmp_path / "a")
    monkeypatch.setattr(cache, "_SEARCHES", tmp_path / "b")
    monkeypatch.setattr(cache, "_PAGES", tmp_path / "c")
    assert cache.evict(max_bytes=10_000_000) == 0


def test_page_cache_round_trip(cfg, tmp_path, monkeypatch):
    from llm_sidecar import cache

    monkeypatch.setattr(cache, "_PAGES", tmp_path / "pages")
    assert cache.get_page(cfg, "http://x", 100) is None
    cache.put_page(cfg, "http://x", 100, "hello")
    assert cache.get_page(cfg, "http://x", 100) == "hello"
    # Different truncation is a different entry — it yields different text.
    assert cache.get_page(cfg, "http://x", 200) is None


# ── ledger rotation ───────────────────────────────────────────────────────────

def test_ledger_rotates_when_large(cfg, monkeypatch):
    from llm_sidecar import ledger

    monkeypatch.setattr(ledger, "MAX_LEDGER_BYTES", 200)
    for _ in range(30):
        ledger.record("m/a", 1, 1, 0.0)
    assert ledger.LEDGER_FILE.with_suffix(".jsonl.1").exists()
    assert ledger.LEDGER_FILE.stat().st_size <= 400


# ── container runtime selection ───────────────────────────────────────────────

def test_podman_is_accepted_when_docker_is_absent(monkeypatch):
    from llm_sidecar import services

    monkeypatch.setattr(services.shutil, "which", lambda n: "/usr/bin/podman" if n == "podman" else None)
    monkeypatch.setattr(services.subprocess, "run",
                        lambda *a, **k: type("R", (), {"returncode": 0})())
    assert services._runtime() == ["podman", "compose"]


def test_runtime_installed_but_stopped_says_so(monkeypatch):
    from llm_sidecar import services

    monkeypatch.setattr(services.shutil, "which", lambda n: "/usr/bin/docker" if n == "docker" else None)
    monkeypatch.setattr(services.subprocess, "run",
                        lambda *a, **k: type("R", (), {"returncode": 1})())
    with pytest.raises(SidecarError) as e:
        services._runtime()
    assert "not running" in str(e.value)


# ── dashboard ─────────────────────────────────────────────────────────────────

def test_dashboard_is_served_at_root(api):
    r = api.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<title>llm-sidecar</title>" in r.text


def test_dashboard_needs_no_token(cfg):
    """The page must load so it can prompt for the token; every endpoint it
    then calls is still gated."""
    from fastapi.testclient import TestClient

    from llm_sidecar import daemon

    cfg.daemon_token = "s3cret"
    c = TestClient(daemon.create_app(cfg))
    assert c.get("/").status_code == 200
    assert c.get("/local-models").status_code == 401
    assert c.post("/config/tier", json={"tier": "fast", "model": "x"}).status_code == 401


def test_dashboard_ships_in_the_package():
    from pathlib import Path

    import llm_sidecar

    assert (Path(llm_sidecar.__file__).parent / "ui" / "index.html").exists()


def test_dashboard_has_no_external_requests():
    """No CDN, no font host, no analytics. A loopback dashboard that phones
    out is both a privacy problem and broken offline."""
    from pathlib import Path

    import llm_sidecar

    html = (Path(llm_sidecar.__file__).parent / "ui" / "index.html").read_text(encoding="utf-8")
    for marker in ("src=\"http", "href=\"http://cdn", "cdn.", "googleapis", "unpkg", "jsdelivr"):
        assert marker not in html, f"dashboard reaches out to {marker}"


def test_tier_pin_and_clear(api):
    r = api.post("/config/tier", json={"tier": "fast", "model": "ollama/x"}).json()
    assert r["models"]["fast"] == "ollama/x"
    assert r["persisted"] is False        # runtime only, never rewrites config
    assert api.post("/config/tier", json={"tier": "fast", "model": ""}).json()["models"] == {}


def test_tier_pin_rejects_unknown_tier(api):
    assert api.post("/config/tier", json={"tier": "turbo", "model": "x"}).status_code == 400


def test_local_models_endpoint(api, monkeypatch):
    from llm_sidecar import hardware

    monkeypatch.setattr(hardware, "advise", lambda config, ctx=8000: [
        {"id": "ollama/m", "name": "m", "size_gib": 4.0, "needs_gib": 5.0,
         "verdict": "fits", "parameter_size": "7B", "quantization": "Q4"}
    ])
    body = api.get("/local-models").json()
    assert body["models"][0]["verdict"] == "fits"


def test_usage_daily_fills_gaps(cfg):
    """A chart that omits quiet days compresses the timeline and turns a
    single busy day into an apparent trend."""
    from llm_sidecar import ledger

    ledger.record("m/a", 1, 1, 0.0)
    days = ledger.daily(7)
    assert len(days) == 7
    assert [d["date"] for d in days] == sorted(d["date"] for d in days)
    assert days[-1]["calls"] == 1          # today
    assert days[0]["calls"] == 0           # a week ago, filled not dropped


def test_status_reports_pins_separately_from_resolution(cfg):
    """The dashboard needs to distinguish "you pinned this" from "auto-resolve
    landed here", so it can show the right pin button as active."""
    cfg.models = {"fast": "ollama/pinned"}
    s = Sidecar(cfg).status()
    assert s["pinned_tiers"] == {"fast": "ollama/pinned"}
    assert s["resolved_tiers"] == {}


# ── searxng probe caching ─────────────────────────────────────────────────────

def test_slow_probe_does_not_permanently_disable_searxng(cfg, monkeypatch):
    """Regression: a healthy SearXNG that took longer than the 2s probe was
    classified as dead, and the false negative was cached for the life of the
    process — so one slow moment silently downgraded every later search."""
    import httpx

    from llm_sidecar.search import searxng

    searxng._probe_cache.clear()

    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("too slow")
        return httpx.Response(200, json={"results": [{"title": "t"}]},
                              request=httpx.Request("GET", "http://x"))

    monkeypatch.setattr(httpx, "get", flaky)
    assert searxng.available(cfg) is False       # timed out

    # A negative is short-lived, so the next check re-probes rather than
    # inheriting the earlier failure.
    searxng._probe_cache[cfg.searxng_url] = (False, time.time() - searxng._NEGATIVE_TTL - 1)
    assert searxng.available(cfg) is True
    searxng._probe_cache.clear()


def test_positive_probe_is_cached(cfg, monkeypatch):
    import httpx

    from llm_sidecar.search import searxng

    searxng._probe_cache.clear()
    calls = []
    monkeypatch.setattr(httpx, "get", lambda *a, **k: (
        calls.append(1),
        httpx.Response(200, json={"results": []}, request=httpx.Request("GET", "http://x")),
    )[1])
    searxng.available(cfg)
    searxng.available(cfg)
    assert len(calls) == 1
    searxng._probe_cache.clear()


def test_probe_timeout_allows_for_engine_fanout():
    """SearXNG waits on a dozen upstream engines; the timeout has to reflect
    that or a healthy instance reads as dead."""
    from llm_sidecar.search import searxng

    assert searxng.PROBE_TIMEOUT >= 5.0


# ── ui opt-out and capability endpoints ───────────────────────────────────────

def test_ui_can_be_disabled_without_touching_the_api(cfg):
    from fastapi.testclient import TestClient

    from llm_sidecar import daemon

    cfg.ui_enabled = False
    c = TestClient(daemon.create_app(cfg))
    r = c.get("/")
    assert r.status_code == 404
    assert "disabled" in r.json()["detail"]
    assert c.get("/v1/models").status_code == 200      # API unaffected


def test_every_capability_has_an_endpoint(api):
    """The dashboard is meant to be one place to try everything, so each
    library capability needs a route behind it."""
    from llm_sidecar import daemon
    from llm_sidecar.config import Config

    paths = {r.path for r in daemon.create_app(Config()).routes if hasattr(r, "path")}
    for expected in ("/v1/chat/completions", "/v1/verify", "/ops/summarise", "/ops/classify",
                     "/ops/extract", "/ops/extract-claims", "/ops/fact-check",
                     "/ops/search", "/ops/read-url"):
        assert expected in paths, expected


def test_ops_endpoints_translate_errors(api, monkeypatch):
    from llm_sidecar import Sidecar

    monkeypatch.setattr(Sidecar, "classify",
                        lambda self, *a, **k: (_ for _ in ()).throw(SidecarError("only one label")))
    r = api.post("/ops/classify", json={"items": ["a"], "labels": ["x"]})
    assert r.status_code == 400
    assert "only one label" in r.json()["detail"]


def test_ops_search_and_read(api, monkeypatch):
    from llm_sidecar import Sidecar
    from llm_sidecar.types import SearchResult

    monkeypatch.setattr(Sidecar, "search",
                        lambda self, q, **k: [SearchResult("t", "http://u", "s")])
    monkeypatch.setattr(Sidecar, "read_url", lambda self, u, **k: "page text")
    assert api.post("/ops/search", json={"query": "x"}).json()["results"][0]["url"] == "http://u"
    assert api.post("/ops/read-url", json={"url": "http://u"}).json()["text"] == "page text"


def test_dashboard_exposes_every_tool():
    """Each capability endpoint should have a matching panel, or the 'one shop
    to try it' claim quietly stops being true as capabilities are added."""
    from pathlib import Path

    import llm_sidecar

    html = (Path(llm_sidecar.__file__).parent / "ui" / "index.html").read_text(encoding="utf-8")
    for endpoint in ("/v1/chat/completions", "/v1/verify", "/ops/summarise", "/ops/classify",
                     "/ops/extract", "/ops/extract-claims", "/ops/fact-check",
                     "/ops/search", "/ops/read-url"):
        assert endpoint in html, f"dashboard has no panel calling {endpoint}"


# ── streaming receipts ────────────────────────────────────────────────────────

def test_streaming_is_recorded_in_the_ledger(cfg, monkeypatch):
    """Regression: Sidecar.stream returned the client generator directly and
    never recorded, so every streamed call was invisible to usage and spend —
    the totals were quietly wrong rather than merely incomplete."""
    import asyncio

    from llm_sidecar import client, ledger, picker
    from llm_sidecar.types import Pick, Usage as U

    monkeypatch.setattr(picker, "pick", lambda *a, **k: Pick("m/1", "m"))

    async def fake_stream(messages, model, config, **k):
        yield {"type": "token", "text": "hi"}
        yield {"type": "usage", "usage": U(5, 7, 12, 0.25)}

    monkeypatch.setattr(client, "stream", fake_stream)
    sc = Sidecar(cfg)

    async def drain():
        return [e async for e in sc.stream("q")]

    events = asyncio.run(drain())
    assert [e["type"] for e in events] == ["token", "usage"]

    entry = ledger.read()[-1]
    assert entry["operation"] == "stream"
    assert entry["completion_tokens"] == 7
    assert entry["cost_usd"] == 0.25


def test_abandoned_stream_still_records(cfg, monkeypatch):
    """Tokens burned before a consumer walks away were still spent."""
    import asyncio

    from llm_sidecar import client, ledger, picker
    from llm_sidecar.types import Pick

    monkeypatch.setattr(picker, "pick", lambda *a, **k: Pick("m/1", "m"))

    async def fake_stream(messages, model, config, **k):
        yield {"type": "token", "text": "a"}
        yield {"type": "token", "text": "b"}

    monkeypatch.setattr(client, "stream", fake_stream)
    sc = Sidecar(cfg)

    async def take_one():
        async for _ in sc.stream("q"):
            break            # walk away after the first token

    asyncio.run(take_one())
    assert ledger.read()[-1]["operation"] == "stream"


def test_stream_emits_a_usage_frame(api):
    """The chat panel's whole point is the receipt, and the streaming format
    has no slot for cost — so a final frame carries it."""
    import json as _json

    from llm_sidecar import client
    from llm_sidecar.types import Usage as U

    async def fake_stream(messages, model, config, **k):
        yield {"type": "token", "text": "hello"}
        yield {"type": "usage", "usage": U(3, 4, 7, 0.002)}

    import llm_sidecar.client as client_mod
    client_mod.stream = fake_stream

    r = api.post("/v1/chat/completions", json={
        "model": "auto", "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    })
    frames = [_json.loads(line[6:]) for line in r.text.splitlines()
              if line.startswith("data: ") and line[6:].strip() != "[DONE]"]

    final = [f for f in frames if f.get("usage")]
    assert final, "no usage frame emitted"
    assert final[0]["usage"]["total_tokens"] == 7
    assert final[0]["x_sidecar"]["cost_usd"] == 0.002
    assert final[0]["x_sidecar"]["streamed"] is True
    # Empty choices is what makes it safe for clients that don't expect it.
    assert final[0]["choices"] == []
    assert r.text.rstrip().endswith("[DONE]")


def test_dashboard_renders_a_receipt_for_streams():
    """The word "streamed" alone used to stand in for the receipt."""
    from pathlib import Path

    import llm_sidecar

    html = (Path(llm_sidecar.__file__).parent / "ui" / "index.html").read_text(encoding="utf-8")
    assert "chunk.usage" in html, "UI never reads the usage frame"
    assert "receiptFor(Object.assign({ model: used }, tally))" in html


# ── escalation to full page text ──────────────────────────────────────────────

def test_unverified_claims_escalate_to_full_text(cfg, monkeypatch):
    """Regression: "there's a london in US" came back unverified because the
    search snippets were about London UK, even though the pages behind them
    listed London, Ohio."""
    from llm_sidecar import verify as verify_mod

    monkeypatch.setattr(verify_mod, "gather_all", lambda claims, config: {
        c: [{"title": "t", "url": "http://page", "snippet": "unhelpful snippet"}] for c in claims
    })
    monkeypatch.setattr(verify_mod, "_fetch_pages",
                        lambda claim, ev, config: [{"title": "t", "url": "http://page",
                                                    "snippet": "London, Ohio is a city in the US."}])

    passes = []

    def grader(batch, sidecar, model):
        passes.append(batch[0][1][0]["snippet"])
        verdict = "supported" if "Ohio" in batch[0][1][0]["snippet"] else "unverified"
        return [{"claim": 1, "verdict": verdict, "note": "n"}]

    monkeypatch.setattr(verify_mod, "_verify_batch", grader)
    out = verify_mod.verify_claims(["there's a london in US"], cfg, sidecar=object())

    assert len(passes) == 2                    # snippets, then full text
    assert out[0].verdict == "supported"


def test_escalation_only_improves(cfg, monkeypatch):
    """A second "unverified" must not overwrite the first with the same answer
    and a different note."""
    from llm_sidecar import verify as verify_mod

    monkeypatch.setattr(verify_mod, "gather_all", lambda claims, config: {
        c: [{"title": "t", "url": "http://p", "snippet": "s"}] for c in claims
    })
    monkeypatch.setattr(verify_mod, "_fetch_pages", lambda c, ev, config: ev)

    calls = []

    def grader(batch, sidecar, model):
        calls.append(1)
        note = "first note" if len(calls) == 1 else "second note"
        return [{"claim": 1, "verdict": "unverified", "note": note}]

    monkeypatch.setattr(verify_mod, "_verify_batch", grader)

    out = verify_mod.verify_claims(["x"], cfg, sidecar=object())
    assert len(calls) == 2                     # it did escalate
    assert out[0].verdict == "unverified"
    assert out[0].note == "first note"         # but the original stands


def test_settled_claims_are_not_escalated(cfg, monkeypatch):
    """Escalation costs a page fetch per claim, so it must only touch the
    claims that snippets failed to settle."""
    from llm_sidecar import verify as verify_mod

    monkeypatch.setattr(verify_mod, "gather_all", lambda claims, config: {
        c: [{"title": "t", "url": "http://p", "snippet": "s"}] for c in claims
    })
    fetched = []
    monkeypatch.setattr(verify_mod, "_fetch_pages",
                        lambda c, ev, config: (fetched.append(c), ev)[1])
    monkeypatch.setattr(verify_mod, "_verify_batch",
                        lambda b, s, m: [{"claim": i, "verdict": "supported", "note": ""}
                                         for i in range(1, len(b) + 1)])

    verify_mod.verify_claims(["a", "b"], cfg, sidecar=object())
    assert fetched == []


def test_escalation_can_be_turned_off(cfg, monkeypatch):
    from llm_sidecar import verify as verify_mod

    monkeypatch.setattr(verify_mod, "gather_all", lambda claims, config: {
        c: [{"title": "t", "url": "http://p", "snippet": "s"}] for c in claims
    })
    fetched = []
    monkeypatch.setattr(verify_mod, "_fetch_pages",
                        lambda c, ev, config: (fetched.append(c), ev)[1])
    monkeypatch.setattr(verify_mod, "_verify_batch",
                        lambda b, s, m: [{"claim": 1, "verdict": "unverified", "note": ""}])

    cfg.verify_escalate = False
    verify_mod.verify_claims(["x"], cfg, sidecar=object())
    assert fetched == []


def test_failed_page_fetch_keeps_the_snippets(cfg, monkeypatch):
    """Escalation may add evidence; it must never remove any."""
    from llm_sidecar import verify as verify_mod

    original = [{"title": "t", "url": "http://p", "snippet": "original"}]
    monkeypatch.setattr(verify_mod, "search" if False else "logger", verify_mod.logger)

    from llm_sidecar import search as search_mod
    monkeypatch.setattr(search_mod, "read_url",
                        lambda *a, **k: (_ for _ in ()).throw(SidecarError("404")))
    assert verify_mod._fetch_pages("c", original, cfg) == original


# ── grounded answering ────────────────────────────────────────────────────────

def _stub_answer(monkeypatch, payload, docs=None):
    from llm_sidecar import answer as answer_mod
    from llm_sidecar.types import Completion

    monkeypatch.setattr(answer_mod, "gather", lambda *a, **k: docs if docs is not None else [
        {"title": "t1", "url": "http://one", "text": "page one"},
        {"title": "t2", "url": "http://two", "text": "page two"},
    ])

    from llm_sidecar.ops import parse_json_response

    class Stub:
        config = None

        def complete(self, *a, **k):
            return Completion(text=payload, model="m/1")

        def complete_json(self, *a, **k):
            done = Completion(text=payload, model="m/1")
            return parse_json_response(done.text), done   # raises like the real one

    return Stub()


def test_answer_cites_only_the_sources_it_used(cfg, monkeypatch):
    from llm_sidecar import answer as answer_mod

    stub = _stub_answer(monkeypatch, '{"answered": true, "answer": "92 million.", "sources_used": [2]}')
    a = answer_mod.answer_question("population?", cfg, sidecar=stub)
    assert a.grounded is True
    assert a.sources == ["http://two"]        # not everything fetched


def test_answer_reports_when_sources_dont_settle_it(cfg, monkeypatch):
    """The point of the capability. Regression: the model read "answered" as
    "I wrote something using the sources" and returned a confident answer
    about microphone checks when asked what someone ate in 2019."""
    from llm_sidecar import answer as answer_mod

    stub = _stub_answer(monkeypatch,
                        '{"answered": false, "answer": "The sources do not cover this.", "sources_used": []}')
    a = answer_mod.answer_question("what did I eat?", cfg, sidecar=stub)
    assert a.grounded is False
    assert a.sources                          # still shows what was consulted


def test_answer_prompt_forbids_topical_matches():
    from llm_sidecar.answer import ANSWER_SYSTEM

    lowered = ANSWER_SYSTEM.lower()
    assert "specific question" in lowered
    assert "share a topic" in lowered


def test_answer_fails_honestly_when_no_model_can_do_the_schema(cfg, monkeypatch):
    """Regression: a reasoning model narrated its scratchpad instead of
    emitting JSON, and the whole 2500-token deliberation was printed as the
    "answer". Rotating is handled by complete_json; when nothing works, say so
    rather than surfacing someone's reasoning as a result."""
    from llm_sidecar import answer as answer_mod

    stub = _stub_answer(monkeypatch, "Here's a thinking process: 1. Analyze the user input…")
    a = answer_mod.answer_question("q", cfg, sidecar=stub)
    assert a.grounded is False
    assert "thinking process" not in a.text          # scratchpad not surfaced
    assert "could not produce an answer" in a.text.lower()
    assert a.sources                                  # still shows what was read


def test_answer_with_no_results_is_honest(cfg, monkeypatch):
    from llm_sidecar import answer as answer_mod

    stub = _stub_answer(monkeypatch, "{}", docs=[])
    a = answer_mod.answer_question("q", cfg, sidecar=stub)
    assert a.grounded is False
    assert "no search results" in a.text.lower()


def test_answer_rejects_empty_and_oversized(cfg):
    from llm_sidecar import answer as answer_mod

    with pytest.raises(SidecarError):
        answer_mod.answer_question("   ", cfg, sidecar=object())
    with pytest.raises(SidecarError):
        answer_mod.answer_question("x" * 5000, cfg, sidecar=object())


def test_answer_endpoint(api, monkeypatch):
    from llm_sidecar import Sidecar
    from llm_sidecar.types import Answer

    monkeypatch.setattr(Sidecar, "answer", lambda self, q, **k: Answer(
        question=q, text="42", grounded=True, sources=["http://u"]))
    body = api.post("/v1/answer", json={"question": "meaning of life?"}).json()
    assert body["grounded"] is True and body["sources"] == ["http://u"]


# ── settings from the UI ──────────────────────────────────────────────────────

def test_api_key_can_be_set_at_runtime(api):
    assert api.get("/config").json()["cloud_configured"] is True   # fixture has one

    r = api.post("/config/api-key", json={"key": "sk-or-abcd1234"}).json()
    assert r["cloud_configured"] is True
    assert r["key_preview"] == "…1234"
    assert r["persisted"] is False


def test_api_key_is_never_echoed_back(api):
    """A loopback port is not a vault: setting the key is fine, reading it
    back would let anything on localhost lift it."""
    api.post("/config/api-key", json={"key": "sk-or-supersecret"})
    for body in (api.get("/config").json(), api.get("/status").json()):
        assert "supersecret" not in json.dumps(body)


def test_clearing_the_key_returns_to_local_only(api):
    api.post("/config/api-key", json={"key": "sk-or-abcd1234"})
    r = api.post("/config/api-key", json={"key": ""}).json()
    assert r["cloud_configured"] is False
    assert r["key_preview"] == ""


def test_setting_a_key_drops_stale_resolution(cfg, monkeypatch):
    """Tiers resolved without a key were picked from an Ollama-only pool, so
    they must not survive a key being added."""
    from fastapi.testclient import TestClient

    from llm_sidecar import daemon

    cfg.openrouter_api_key = None
    app = daemon.create_app(cfg)
    c = TestClient(app)
    c.post("/config/api-key", json={"key": "sk-or-abcd1234"})
    assert c.get("/status").json()["resolved_tiers"] == {}


def test_persist_writes_the_key_only_when_asked(cfg, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from llm_sidecar import config as config_mod, daemon

    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.json")
    c = TestClient(daemon.create_app(cfg))

    c.post("/config/api-key", json={"key": "sk-or-abcd1234", "persist": False})
    assert not (tmp_path / "config.json").exists()

    r = c.post("/config/api-key", json={"key": "sk-or-abcd1234", "persist": True}).json()
    assert r["persisted"] is True
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))["openrouter_api_key"] == "sk-or-abcd1234"


def test_budget_can_be_set_and_is_validated(api):
    assert api.post("/config/budget", json={"budget": "best"}).json()["default_budget"] == "best"
    assert api.post("/config/budget", json={"budget": "lavish"}).status_code == 400


def test_dashboard_has_a_settings_panel():
    from pathlib import Path

    import llm_sidecar

    html = (Path(llm_sidecar.__file__).parent / "ui" / "index.html").read_text(encoding="utf-8")
    assert "/config/api-key" in html
    assert 'id="cfg-key"' in html and 'type="password"' in html
    assert "/v1/answer" in html


# ── OpenRouter web retrieval ──────────────────────────────────────────────────

def test_web_plugin_is_only_sent_to_cloud_models(cfg, monkeypatch):
    """Ollama has no such facility; sending the plugin would be a 400."""
    import httpx

    from llm_sidecar import client

    sent = {}

    def capture(url, json=None, **k):
        sent.update(json or {})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", capture)
    client.complete([{"role": "user", "content": "q"}], "openai/gpt-4o", cfg, web=True)
    assert sent["plugins"] == [{"id": "web", "max_results": 5}]

    sent.clear()
    client.complete([{"role": "user", "content": "q"}], "ollama/x", cfg, web=True)
    assert "plugins" not in sent


def test_web_result_count_reaches_the_plugin(cfg, monkeypatch):
    """Each result is billed, so an ignored count costs real money."""
    import httpx

    from llm_sidecar import client

    sent = {}
    monkeypatch.setattr(httpx, "post", lambda url, json=None, **k: (
        sent.update(json or {}),
        httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]},
                       request=httpx.Request("POST", url)),
    )[1])
    client.complete([{"role": "user", "content": "q"}], "openai/gpt-4o", cfg,
                    web=True, web_results=2)
    assert sent["plugins"][0]["max_results"] == 2


def test_citations_are_extracted(cfg, monkeypatch):
    import httpx

    from llm_sidecar import client

    body = {"choices": [{"message": {
        "content": "answer",
        "annotations": [
            {"type": "url_citation", "url_citation": {"url": "http://a", "title": "A", "content": "x"}},
            {"type": "something_else", "other": {}},
            {"type": "url_citation", "url_citation": {"title": "no url"}},
        ],
    }}]}
    monkeypatch.setattr(httpx, "post", lambda url, **k: httpx.Response(
        200, json=body, request=httpx.Request("POST", url)))
    c = client.complete([{"role": "user", "content": "q"}], "openai/gpt-4o", cfg, web=True)
    assert [x["url"] for x in c.citations] == ["http://a"]


def test_provider_error_inside_a_200_raises(cfg, monkeypatch):
    """Regression: OpenRouter returns HTTP 200 with an error object and no
    choices when an upstream provider fails. Checking only the status code
    turned that into an empty completion recorded as a success, and no
    rotation happened because nothing raised."""
    import httpx

    from llm_sidecar import client

    monkeypatch.setattr(httpx, "post", lambda url, **k: httpx.Response(
        200, json={"error": {"message": "rate-limited upstream", "code": 429}},
        request=httpx.Request("POST", url)))
    cfg.retry_delays = ()
    with pytest.raises(client.UpstreamError) as e:
        client.complete([{"role": "user", "content": "q"}], "m/1", cfg)
    assert e.value.status == 429


def test_response_with_no_choices_raises(cfg, monkeypatch):
    import httpx

    from llm_sidecar import client

    monkeypatch.setattr(httpx, "post", lambda url, **k: httpx.Response(
        200, json={"choices": []}, request=httpx.Request("POST", url)))
    cfg.retry_delays = ()
    with pytest.raises(client.UpstreamError):
        client.complete([{"role": "user", "content": "q"}], "m/1", cfg)


def test_upstream_429_rotates_to_another_model(cfg, monkeypatch):
    """The point of catching it: a throttled model must be swapped, not
    returned as an empty answer."""
    from llm_sidecar import client, picker
    from llm_sidecar.types import Completion, Pick

    handed = iter(["bad/model", "good/model"])
    monkeypatch.setattr(picker, "pick", lambda *a, **k: Pick(next(handed), "m"))

    def flaky(messages, model, config, **k):
        if model == "bad/model":
            raise client.UpstreamError("throttled", status=429)
        return Completion(text="ok", model=model)

    monkeypatch.setattr(client, "complete", flaky)
    assert Sidecar(cfg).complete("hi").model == "good/model"


def test_openrouter_answer_needs_a_key(cfg):
    from llm_sidecar import answer as answer_mod

    cfg.openrouter_api_key = None
    with pytest.raises(SidecarError) as e:
        answer_mod.answer_question("q", cfg, sidecar=Sidecar(cfg), via="openrouter")
    assert "via='local'" in str(e.value)      # names the free way out


def test_unknown_retrieval_mode_rejected(cfg):
    from llm_sidecar import answer as answer_mod

    with pytest.raises(SidecarError):
        answer_mod.answer_question("q", cfg, sidecar=Sidecar(cfg), via="telepathy")


def test_openrouter_answer_needs_citations_to_be_grounded(cfg, monkeypatch):
    """A model claiming "answered": true having retrieved nothing is not
    grounded, whatever it says about itself."""
    from llm_sidecar import answer as answer_mod
    from llm_sidecar.types import Completion

    class Stub:
        config = cfg
        def complete(self, *a, **k):
            return Completion(text='{"answered": true, "answer": "yes"}', model="m", citations=[])

    a = answer_mod._answer_via_openrouter("q", Stub(), None, "fast", 3)
    assert a.grounded is False


def test_web_calls_are_not_cached(cfg, monkeypatch):
    """Paying for retrieval and then serving a stored answer defeats the
    point of paying for retrieval."""
    from llm_sidecar import cache, client, picker
    from llm_sidecar.types import Completion, Pick

    monkeypatch.setattr(picker, "pick", lambda *a, **k: Pick("m/1", "m"))
    calls = []
    monkeypatch.setattr(client, "complete", lambda *a, **k: (
        calls.append(1), Completion(text="answer", model="m/1"))[1])

    sc = Sidecar(cfg)
    sc.complete("q", temperature=0.0, web=True)
    sc.complete("q", temperature=0.0, web=True)
    assert len(calls) == 2


# ── tier and budget as separate axes ──────────────────────────────────────────

def test_budget_is_its_own_field(api):
    """`model` can only express one axis, so "powerful" and "best" together
    was unsayable. A separate field beats inventing compound aliases."""
    body = api.post("/v1/chat/completions", json={
        "model": "powerful", "budget": "best",
        "messages": [{"role": "user", "content": "hi"}],
    }).json()
    assert body["model"] == "auto/best"       # fixture names the model after the budget


def test_budget_field_is_validated(api):
    r = api.post("/v1/chat/completions", json={
        "model": "fast", "budget": "lavish",
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400


def test_exact_model_ignores_budget(api):
    """An explicit id is the caller's decision; a budget alongside it would be
    ambiguous, so the id wins outright."""
    body = api.post("/v1/chat/completions", json={
        "model": "ollama/pinned", "budget": "best",
        "messages": [{"role": "user", "content": "hi"}]}).json()
    assert body["model"] == "ollama/pinned"


def test_auto_means_defaults_and_nothing_else(api):
    """Documenting the surprise: "auto" is not a mode. It is an unrecognised
    string, and every unrecognised string means "use both defaults"."""
    from llm_sidecar import daemon

    app_target = daemon.create_app
    a = api.post("/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": "hi"}]}).json()
    b = api.post("/v1/chat/completions", json={
        "model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}).json()
    assert a["model"] == b["model"] == "auto/free"


def test_chat_panel_separates_the_axes():
    """Regression on the UI, not the API: one dropdown mixing auto, tiers,
    budgets and model ids made them look like peers when they are not."""
    from pathlib import Path

    import llm_sidecar

    html = (Path(llm_sidecar.__file__).parent / "ui" / "index.html").read_text(encoding="utf-8")
    assert 'id="chat-tier"' in html
    assert 'id="chat-budget"' in html
    assert 'id="chat-resolved"' in html        # shows what it landed on


# ── making the decision visible ───────────────────────────────────────────────

def test_preview_explains_a_keyless_free_budget(cfg, monkeypatch):
    """The chat readout showed only a cached resolution, so "free" looked like
    it meant Ollama when it actually meant "Ollama, because there's no key"."""
    from fastapi.testclient import TestClient

    from llm_sidecar import daemon

    cfg.openrouter_api_key = None
    c = TestClient(daemon.create_app(cfg))
    r = c.get("/resolve-preview?tier=fast&budget=free").json()

    assert r["state"] == "unresolved"
    assert r["cloud_configured"] is False
    assert "no api key" in r["note"].lower()
    assert all(m.startswith("ollama/") for m in r["would_try"])


def test_preview_shows_cloud_first_with_a_key(api):
    r = api.get("/resolve-preview?tier=fast&budget=free").json()
    assert r["cloud_configured"] is True
    assert not r["would_try"][0].startswith("ollama/")


def test_preview_reports_a_pin_and_says_budget_is_moot(cfg):
    """A pin beats the budget entirely, which is exactly the case that made
    "fast + free says Ollama" look like a contradiction."""
    from fastapi.testclient import TestClient

    from llm_sidecar import daemon

    cfg.models = {"fast": "ollama/pinned"}
    c = TestClient(daemon.create_app(cfg))
    r = c.get("/resolve-preview?tier=fast&budget=best").json()
    assert r["state"] == "pinned"
    assert r["model"] == "ollama/pinned"
    assert "budget is ignored" in r["note"].lower()   # and says so in plain words


def test_preview_flags_a_paid_budget_with_no_key(cfg):
    from fastapi.testclient import TestClient

    from llm_sidecar import daemon

    cfg.openrouter_api_key = None
    c = TestClient(daemon.create_app(cfg))
    r = c.get("/resolve-preview?tier=fast&budget=best").json()
    assert r["would_try"] == []
    assert "api key" in r["note"].lower()


def test_preview_does_not_probe(cfg, monkeypatch):
    """It has to be cheap enough to run on every dropdown change."""
    from llm_sidecar import picker

    monkeypatch.setattr(picker, "pretest",
                        lambda *a, **k: pytest.fail("preview must not probe"))
    from fastapi.testclient import TestClient

    from llm_sidecar import daemon

    TestClient(daemon.create_app(cfg)).get("/resolve-preview?tier=fast&budget=free")


def test_dashboard_shows_key_state_as_a_badge():
    """Regression: the only signal that a key was set was the placeholder text
    of an empty password field — which reads as a prompt, not a status, so
    "is my key set?" was unanswerable from the UI."""
    from pathlib import Path

    import llm_sidecar

    html = (Path(llm_sidecar.__file__).parent / "ui" / "index.html").read_text(encoding="utf-8")
    assert 'id="cfg-key-state"' in html
    assert "no API key — local only" in html
    assert "API key set" in html


def test_complete_json_rotates_past_a_model_that_cannot_do_schema(cfg, monkeypatch):
    """Structured output is a capability. A model that narrates instead of
    emitting JSON is as unusable as a throttled one, and gets the same
    treatment — marked dead, next candidate tried."""
    from llm_sidecar import client, picker
    from llm_sidecar.types import Completion, Pick

    handed = iter(["rambler/1", "obedient/1"])
    monkeypatch.setattr(picker, "pick", lambda *a, **k: Pick(next(handed), "m"))
    monkeypatch.setattr(client, "complete", lambda msgs, model, config, **k: Completion(
        text="Here's a thinking process: 1." if model == "rambler/1" else '{"ok": true}',
        model=model))

    sc = Sidecar(cfg)
    parsed, done = sc.complete_json("q")
    assert parsed == {"ok": True}
    assert done.model == "obedient/1"
    assert "rambler/1" in sc._failed


def test_complete_json_does_not_rotate_a_pinned_model(cfg, monkeypatch):
    from llm_sidecar import client, picker
    from llm_sidecar.types import Completion, Pick

    monkeypatch.setattr(picker, "pick", lambda *a, **k: Pick("other/1", "m"))
    monkeypatch.setattr(client, "complete",
                        lambda msgs, model, config, **k: Completion(text="not json", model=model))

    sc = Sidecar(cfg)
    with pytest.raises(SidecarError):
        sc.complete_json("q", model="mine/1")
    assert sc._failed == set()          # the caller's choice is left alone


def test_throttled_auto_picked_model_is_not_waited_out(cfg, monkeypatch):
    """25 seconds of backoff to reuse a rate-limited endpoint, when another
    verified model is one probe away, is the wrong trade."""
    import httpx

    from llm_sidecar import client

    slept = []
    monkeypatch.setattr(client.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(httpx, "post", lambda url, **k: httpx.Response(
        429, json={"error": "slow down"}, request=httpx.Request("POST", url)))

    with pytest.raises(Exception):
        client.complete([{"role": "user", "content": "q"}], "m/1", cfg,
                        retry_on_throttle=False)
    assert slept == []                  # rotate instead

    slept.clear()
    with pytest.raises(Exception):
        client.complete([{"role": "user", "content": "q"}], "m/1", cfg,
                        retry_on_throttle=True)
    assert slept                        # pinned model: waiting is all we can do


def test_every_text_read_declares_utf8():
    """Regression, caught by CI on windows-latest before any user hit it.

    Python's text mode defaults to the *locale* encoding, which is cp1252 on
    Windows. The dashboard HTML is full of em-dashes and box-drawing
    characters, so the daemon served a mangled page — or crashed — on Windows
    while being perfectly fine on macOS and Linux. Same hazard for the SearXNG
    settings, the JSON caches and the ledger."""
    import re
    from pathlib import Path

    import llm_sidecar

    root = Path(llm_sidecar.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\.read_text\(\)|\.open\(\)|\.open\(['\"][wa]['\"]\)", line):
                offenders.append(f"{path.relative_to(root)}:{n}: {line.strip()}")
    assert not offenders, "text IO without an explicit encoding:\n" + "\n".join(offenders)


# ── model families rather than exact ids ──────────────────────────────────────

def test_family_rank_survives_version_bumps():
    """Regression on rot, not on a crash. The previous curated list held exact
    ids, so claude-haiku-4-5 -> 4-6 silently dropped an entry; of nine free
    picks, one still existed a few months later."""
    from llm_sidecar import picker

    for old, new in [
        ("anthropic/claude-haiku-4-5", "anthropic/claude-haiku-9.9"),
        ("google/gemma-3-27b-it:free", "google/gemma-7-99b-it:free"),
        ("anthropic/claude-opus-4-7", "anthropic/claude-opus-12"),
    ]:
        budget = "free" if ":free" in old else ("cheap" if "haiku" in old else "best")
        assert picker.family_rank(old, budget) == picker.family_rank(new, budget)


def test_unknown_models_are_ranked_not_excluded():
    """A good model nobody has heard of should sort last among known families,
    not vanish."""
    from llm_sidecar import picker

    rank = picker.family_rank("newvendor/brand-new-thing", "free")
    assert rank == len(picker.PREFERRED_FAMILIES["free"])
    assert rank > picker.family_rank("deepseek/anything", "free")


def test_batch_and_online_variants_are_excluded(cfg, monkeypatch):
    """Neither can be caught by probing: both answer "OK" happily. :batch has
    asynchronous semantics, and :online silently attaches billed retrieval to
    every call when web search here is meant to be an explicit choice."""
    from llm_sidecar import catalogue, picker

    monkeypatch.setattr(catalogue, "openrouter_models", lambda config, force_refresh=False: [
        ModelInfo("vendor/good:free", "good", context_length=8000),
        ModelInfo("vendor/good:batch", "batch", context_length=8000),
        ModelInfo("vendor/good:online", "online", context_length=8000),
    ])
    monkeypatch.setattr(catalogue, "ollama_models", lambda config: [])

    got = picker.candidates(cfg, "free")
    assert "vendor/good:free" in got
    assert not any(":batch" in c or ":online" in c for c in got)


def test_every_budget_still_matches_something_real():
    """The families must actually fire against ids of the shape providers use.
    Offline: a handful of representative ids, not a live catalogue."""
    from llm_sidecar import picker

    samples = {
        "free": ["deepseek/deepseek-v9:free", "qwen/qwen4-80b:free", "google/gemma-5-9b:free"],
        "cheap": ["anthropic/claude-haiku-9", "google/gemini-9-flash", "openai/gpt-9-mini"],
        "best": ["anthropic/claude-opus-9", "openai/gpt-5.9", "google/gemini-9-pro"],
    }
    for budget, ids in samples.items():
        for mid in ids:
            assert picker.family_rank(mid, budget) < len(picker.PREFERRED_FAMILIES[budget]), \
                f"{mid} matches no family for {budget}"


def test_dashboard_does_not_still_say_pin():
    """The rename from "pinned" to "locked" missed a footnote, which a
    screenshot caught: the heading said locks and the note beneath it said
    pins. Mixed vocabulary for one concept is worse than either word."""
    from pathlib import Path

    import llm_sidecar

    html = (Path(llm_sidecar.__file__).parent / "ui" / "index.html").read_text(encoding="utf-8")
    visible = html[html.index("<body>"):]
    assert "Pins apply" not in visible
    assert "Pin to tier" not in visible


# ── the Windows setup scripts ─────────────────────────────────────────────────

def test_install_bat_avoids_delayed_expansion_hazards():
    """CI caught this: with `enabledelayedexpansion`, cmd treats !...! as a
    variable reference, so "[!] No Ollama ... https://ollama.com" printed as
    "[//ollama.com" — everything between the two ! was swallowed."""
    from pathlib import Path

    bat = (Path(__file__).parent.parent.parent / "install.bat")
    if not bat.exists():          # not shipped inside the wheel
        pytest.skip("install.bat is only present in a source checkout")

    for n, line in enumerate(bat.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip().lower().startswith("echo") and "!" in line:
            # !VAR! references are fine; a bare ! in prose is not.
            assert re.fullmatch(r"[^!]*(![A-Za-z_][A-Za-z0-9_]*![^!]*)*", line), \
                f"install.bat:{n} has an unpaired ! in echoed text: {line.strip()}"


def test_install_bat_upgrades_pip_through_python():
    """pip.exe cannot replace its own running executable on Windows."""
    from pathlib import Path

    bat = (Path(__file__).parent.parent.parent / "install.bat")
    if not bat.exists():
        pytest.skip("install.bat is only present in a source checkout")
    text = bat.read_text(encoding="utf-8")
    assert "-m pip install --quiet --upgrade pip" in text
    assert "pip.exe install --quiet --upgrade" not in text


def test_install_bat_exits_zero_on_success():
    """A failing `where ollama` used to leak its errorlevel, so a perfectly
    good install ended in exit code 1."""
    from pathlib import Path

    bat = (Path(__file__).parent.parent.parent / "install.bat")
    if not bat.exists():
        pytest.skip("install.bat is only present in a source checkout")
    assert bat.read_text(encoding="utf-8").rstrip().endswith("exit /b 0")
