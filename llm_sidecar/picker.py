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
from concurrent.futures import ThreadPoolExecutor

import httpx

from . import catalogue
from .config import Config
from .types import ModelInfo, NoWorkingModel, Pick

logger = logging.getLogger("llm_sidecar.picker")

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_PREFIX = "ollama/"
BUDGETS = ("free", "cheap", "best")

# Candidates are probed in waves rather than one at a time. Sequential walking
# meant a cold start paid the full timeout for every dead model in front of
# the live one — with a 15s timeout and four stale entries, over a minute
# before the first token. A wave costs a few extra tiny calls (free models,
# ten tokens each) in exchange for bounding that to one timeout.
PROBE_WAVE = 3

# Preference is expressed as model *families*, not exact ids.
#
# An earlier version listed exact ids and rotted fast: of nine curated free
# picks, one still existed a few months later — every version bump
# (claude-haiku-4-5 -> 4-6, gemma-3 -> gemma-4) silently dropped an entry, and
# the "curated" list quietly stopped curating anything.
#
# A family survives those bumps. Order is preference, best first; anything
# unmatched still competes on the tie-breaker below, so a good model we have
# never heard of is ranked, not excluded.
PREFERRED_FAMILIES = {
    # Free: the open-weight families that reason well enough for real work.
    "free": (
        "deepseek", "qwen", "llama", "gemma", "mistral", "nemotron",
        "glm", "hermes", "command", "phi", "olmo",
    ),
    # Cheap: every vendor's small/fast line, whatever they call it this year.
    "cheap": (
        "haiku", "flash", "mini", "nano", "small", "lite", "turbo",
        "gemma", "llama", "qwen",
    ),
    # Best: the frontier lines.
    "best": (
        "opus", "sonnet", "gpt-5", "gpt-4", "pro", "ultra", "grok",
        "deepseek-r", "large",
    ),
}


def family_rank(model_id: str, budget: str) -> int:
    """Index of the first matching family, or a value past the end.

    Lower is better. Unmatched models are not excluded — they sort after the
    known families and then compete on price or context like everyone else."""
    lower = model_id.lower()
    families = PREFERRED_FAMILIES.get(budget, ())
    for i, family in enumerate(families):
        if family in lower:
            return i
    return len(families)


def _matches_budget(m: ModelInfo, budget: str) -> bool:
    p = m.prompt_price_per_million
    if budget == "free":
        return m.is_free
    if budget == "cheap":
        return (not m.is_free) and 0 < p <= 1.0
    if budget == "best":
        return p >= 5.0
    return False


def _local_for_tier(models: list, tier: str | None) -> list[str]:
    """Order local models to match what the tier word actually promises.

    Without this, every tier took the largest installed model — so "fast" on a
    laptop meant a 19 GB 32B, the slowest thing available. The label said the
    opposite of what it did.

    balanced and powerful both take the biggest that fits, deliberately: RAM,
    not preference, is the ceiling locally, and pretending otherwise would
    just mean two names for one model."""
    by_size = sorted(models, key=lambda m: m.size_bytes)
    if tier == "fast":
        # Smallest first — the point of the tier is latency. Anything tiny
        # enough to be useless is still better here than a 30-second load.
        return [m.id for m in by_size]
    return [m.id for m in reversed(by_size)]


def candidates(config: Config, budget: str, tier: str | None = None) -> list[str]:
    """Ordered candidate model ids for a budget.

    Sorted by model family first (see PREFERRED_FAMILIES), then by whatever
    matters for that budget — context length when everything is free, price
    when it isn't.

    For `free`, local Ollama models join the list. Their position depends on
    whether a cloud key exists: with a key, cloud free models go first (they
    are usually stronger than what fits on a laptop) and Ollama is the safety
    net; without one, Ollama is the only thing that can work."""
    cloud = catalogue.openrouter_models(config)
    in_budget = [
        m for m in cloud
        if _matches_budget(m, budget) and not catalogue.is_specialised(m.id)
    ]

    if budget == "free":
        # Nothing to price-sort on; roomier context breaks the tie.
        in_budget.sort(key=lambda m: (family_rank(m.id, budget), -m.context_length))
    elif budget == "cheap":
        in_budget.sort(key=lambda m: (family_rank(m.id, budget), m.prompt_price_per_million))
    else:
        in_budget.sort(key=lambda m: (family_rank(m.id, budget), -m.prompt_price_per_million))

    cloud_candidates = [m.id for m in in_budget]
    if budget != "free":
        return cloud_candidates

    local = _local_for_tier(catalogue.ollama_models(config), tier)
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
    tier: str | None = None,
) -> Pick:
    """Return the first candidate that passes a live pretest.

    `exclude` skips ids already known bad in this run — a model can pass
    pretest and then 429 on the real call, and we shouldn't hand it back again.

    Raises NoWorkingModel if every candidate fails."""
    budget = budget or config.default_budget
    if budget not in BUDGETS:
        raise NoWorkingModel(f"Unknown budget: {budget!r}. Expected one of {BUDGETS}.")

    cand = [c for c in candidates(config, budget, tier) if not exclude or c not in exclude]

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

    cand = cand[:max_attempts]
    names = {m.id: m.name for m in catalogue.openrouter_models(config)}
    attempts: list[dict] = []

    def friendly(model_id: str) -> str:
        if model_id.startswith(OLLAMA_PREFIX):
            return f"{model_id[len(OLLAMA_PREFIX):]} (local)"
        return names.get(model_id, model_id)

    for i in range(0, len(cand), PROBE_WAVE):
        wave = cand[i:i + PROBE_WAVE]
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            results = list(pool.map(lambda m: (m, *pretest(m, config)), wave))

        for model_id, ok, reason in results:
            attempts.append({"id": model_id, "ok": ok, "reason": reason})

        # Priority order within the wave still wins — probing concurrently
        # must not change *which* model gets chosen, only how fast we find it.
        for model_id, ok, _ in results:
            if ok:
                return Pick(model_id=model_id, model_name=friendly(model_id), attempts=attempts)

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
