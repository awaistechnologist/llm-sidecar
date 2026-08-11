"""Model picker — hand back a model that is verifiably working *right now*.

This is the piece that makes free tiers usable. A catalogue entry tells you a
model exists; it doesn't tell you the provider hasn't throttled you, retired
the checkpoint, or started returning empty completions. So before returning a
model we spend one tiny call ("Reply with OK") proving it answers.

`pick_pool` extends that to a set of distinct working models mapped onto
fast/balanced/powerful, so a caller running several requests in parallel
spreads them across providers instead of stampeding one free endpoint.

Ported from Agora's backend/services/model_picker.py, with the SQLAlchemy
session dependency removed.
"""

from __future__ import annotations

import logging

import httpx

from . import catalogue
from .config import Config
from .types import ModelInfo, NoWorkingModel, Pick

logger = logging.getLogger("llm_sidecar.picker")

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_PREFIX = "ollama/"
BUDGETS = ("free", "cheap", "best")

# Curated first choices per budget. Order is preference. Anything not in the
# live catalogue is skipped silently — the model landscape churns constantly,
# so this list is a hint, not a contract.
PREFERRED = {
    "free": [
        "z-ai/glm-4.5-air:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "nousresearch/hermes-3-llama-3.1-405b:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "google/gemma-4-31b-it:free",
        "google/gemma-3-27b-it:free",
        "deepseek/deepseek-chat-v3:free",
        "deepseek/deepseek-r1:free",
        "meta-llama/llama-3.1-70b-instruct:free",
    ],
    "cheap": [
        "anthropic/claude-haiku-4-5",
        "google/gemini-2.5-flash",
        "openai/gpt-5-nano",
        "openai/gpt-5-mini",
        "anthropic/claude-3-5-haiku",
        "google/gemini-flash-1.5",
        "mistralai/mistral-small-3.2",
    ],
    "best": [
        "anthropic/claude-opus-4-7",
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-5",
        "openai/gpt-5.2",
        "google/gemini-3-pro",
        "anthropic/claude-3-5-sonnet",
        "openai/gpt-4o",
    ],
}


def _matches_budget(m: ModelInfo, budget: str) -> bool:
    p = m.prompt_price_per_million
    if budget == "free":
        return m.is_free
    if budget == "cheap":
        return (not m.is_free) and 0 < p <= 1.0
    if budget == "best":
        return p >= 5.0
    return False


def candidates(config: Config, budget: str) -> list[str]:
    """Ordered candidate model ids for a budget: curated picks first, then the
    rest of the tier sorted by the property that matters for that tier.

    For `free`, local Ollama models join the list. Their position depends on
    whether a cloud key exists: with a key, cloud free models go first (they
    are faster and stronger than most laptop models) and Ollama is the safety
    net; without one, Ollama is the only thing that can actually work."""
    cloud = catalogue.openrouter_models(config)
    by_id = {m.id: m for m in cloud}
    in_budget = [m for m in cloud if _matches_budget(m, budget) and not catalogue.is_specialised(m.id)]

    preferred_ids = [pid for pid in PREFERRED.get(budget, []) if pid in by_id]
    fallback_ids = [m.id for m in in_budget if m.id not in preferred_ids]

    if budget == "free":
        # Nothing to price-sort on; prefer the roomiest context.
        fallback_ids.sort(key=lambda i: -by_id[i].context_length)
    elif budget == "cheap":
        fallback_ids.sort(key=lambda i: by_id[i].prompt_price_per_million)
    elif budget == "best":
        fallback_ids.sort(key=lambda i: by_id[i].prompt_price_per_million, reverse=True)

    cloud_candidates = preferred_ids + fallback_ids
    if budget != "free":
        return cloud_candidates

    local = [m.id for m in catalogue.ollama_models(config)]
    return cloud_candidates + local if config.has_cloud else local + cloud_candidates


def pretest(model_id: str, config: Config) -> tuple[bool, str | None]:
    """One tiny call proving the model answers. Returns (ok, reason_if_not)."""
    is_ollama = model_id.startswith(OLLAMA_PREFIX)
    if is_ollama:
        url = f"{config.ollama_host}/v1/chat/completions"
        wire_model = model_id[len(OLLAMA_PREFIX):]
        headers = {"Content-Type": "application/json"}
        # A cold local model has to be loaded into memory on first call.
        timeout = 30.0
    else:
        if not config.has_cloud:
            return False, "no api key"
        url = OPENROUTER_CHAT_URL
        wire_model = model_id
        headers = {
            "Authorization": f"Bearer {config.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": config.referer,
            "X-Title": config.app_title,
        }
        timeout = config.pretest_timeout

    payload = {
        "model": wire_model,
        "messages": [{"role": "user", "content": "Reply with 'OK' and nothing else."}],
        "max_tokens": 10,
        "temperature": 0.0,
    }
    try:
        r = httpx.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        choices = r.json().get("choices") or []
        if not choices:
            return False, "no choices"
        content = (choices[0].get("message") or {}).get("content") or ""
        if not content.strip():
            return False, "empty content"
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"


def pick(
    config: Config,
    budget: str | None = None,
    max_attempts: int = 6,
    exclude: set[str] | None = None,
) -> Pick:
    """Return the first candidate that passes a live pretest.

    `exclude` skips ids already known bad in this run — a model can pass
    pretest and then 429 on the real call, and we shouldn't hand it back again.

    Raises NoWorkingModel if every candidate fails."""
    budget = budget or config.default_budget
    if budget not in BUDGETS:
        raise NoWorkingModel(f"Unknown budget: {budget!r}. Expected one of {BUDGETS}.")

    cand = [c for c in candidates(config, budget) if not exclude or c not in exclude]

    # Without a key, cloud candidates can only fail — drop them so we don't
    # burn the attempt allowance discovering that one at a time.
    if not config.has_cloud:
        cand = [c for c in cand if c.startswith(OLLAMA_PREFIX)]

    if not cand:
        raise NoWorkingModel(
            f"No candidates for budget {budget!r}. "
            + ("Install Ollama or set OPENROUTER_API_KEY." if not config.has_cloud
               else "Try a different budget or refresh the catalogue.")
        )

    names = {m.id: m.name for m in catalogue.openrouter_models(config)}
    attempts: list[dict] = []
    for model_id in cand[:max_attempts]:
        ok, reason = pretest(model_id, config)
        attempts.append({"id": model_id, "ok": ok, "reason": reason})
        if ok:
            if model_id.startswith(OLLAMA_PREFIX):
                friendly = f"{model_id[len(OLLAMA_PREFIX):]} (local)"
            else:
                friendly = names.get(model_id, model_id)
            return Pick(model_id=model_id, model_name=friendly, attempts=attempts)

    raise NoWorkingModel(
        f"Tried {len(attempts)} candidate model(s) for budget {budget!r}; none responded.",
        attempts=attempts,
    )


def pick_pool(
    config: Config,
    budget: str | None = None,
    max_attempts: int = 8,
    exclude: set[str] | None = None,
) -> dict[str, Pick]:
    """Up to three *distinct* working models, mapped to fast/balanced/powerful.

    Spreading parallel work across providers is the difference between a free
    tier that works and one that 429s halfway through. Local runs skip this —
    a single Ollama model has no per-provider quota to spread across, and
    loading three models into memory would be actively worse.

    Falls back to repeating the primary pick for tiers we couldn't fill, so
    the returned dict always has all three keys."""
    first = pick(config, budget, max_attempts=max_attempts, exclude=exclude)

    if first.model_id.startswith(OLLAMA_PREFIX) or not config.has_cloud:
        return {tier: first for tier in ("fast", "balanced", "powerful")}

    excluded = set(exclude or ()) | {first.model_id}
    extras: list[Pick] = []
    for _ in range(2):
        try:
            nxt = pick(config, budget, max_attempts=max_attempts, exclude=excluded)
        except NoWorkingModel:
            break
        extras.append(nxt)
        excluded.add(nxt.model_id)

    # Balanced gets the strongest verified pick, since it's the default tier
    # and takes the most traffic.
    picks = [first] + extras
    return {
        tier: picks[min(i, len(picks) - 1)]
        for i, tier in enumerate(("balanced", "fast", "powerful"))
    }
