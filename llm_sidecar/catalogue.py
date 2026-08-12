"""Model catalogue — the list of models we could route to.

Two sources: the OpenRouter model list (fetched over HTTP, cached to disk with
a TTL) and whatever Ollama has pulled locally (queried live; it's a localhost
call and changes whenever the user pulls a model, so caching it would only
create staleness bugs).
"""

from __future__ import annotations

import json
import logging
import time

import httpx

from .config import CACHE_DIR, CATALOGUE_TTL_SECONDS, Config
from .types import ModelInfo

logger = logging.getLogger("llm_sidecar.catalogue")

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OLLAMA_PREFIX = "ollama/"

_CACHE_FILE = CACHE_DIR / "openrouter-models.json"

# Process-local memo on top of the disk cache. Re-reading and re-parsing a
# 400-model JSON file cost ~3ms, and pick_pool() was doing it six times per
# call for data that cannot change mid-process.
_memo: tuple[list[ModelInfo], float] | None = None
_MEMO_TTL = 300.0

# Substrings marking a model as specialised (OCR, vision-only, embeddings,
# audio, image-gen). These are excluded from general-purpose routing — they
# make poor text reasoners and waste pretest budget.
SPECIALISED_KEYWORDS = (
    "ocr", "vision", "embed", "embedding", "image-gen", "imagegen",
    "audio", "tts", "stt", "whisper", "moderation", "rerank",
    "lyria", "clip", "music", "video", "imagen", "dall-e", "midjourney",
    "stable-diffusion", "sora", "veo",
)


def is_specialised(model_id: str) -> bool:
    lower = model_id.lower()
    return any(k in lower for k in SPECIALISED_KEYWORDS)


def _price_per_million(pricing: dict, key: str) -> float:
    """OpenRouter quotes price per token as a string. Normalise to $/1M."""
    try:
        return float(pricing.get(key, 0) or 0) * 1_000_000
    except (TypeError, ValueError):
        return 0.0


def _parse(raw: list[dict]) -> list[ModelInfo]:
    out = []
    for m in raw:
        mid = m.get("id")
        if not mid:
            continue
        pricing = m.get("pricing") or {}
        out.append(ModelInfo(
            id=mid,
            name=m.get("name") or mid,
            context_length=int(m.get("context_length") or 0),
            prompt_price_per_million=_price_per_million(pricing, "prompt"),
            completion_price_per_million=_price_per_million(pricing, "completion"),
        ))
    return out


def fetch_openrouter(api_key: str | None, timeout: float = 20.0) -> list[ModelInfo]:
    """Fetch the live OpenRouter catalogue. The endpoint is public, so this
    works without a key — the key only matters when actually calling a model."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = httpx.get(OPENROUTER_MODELS_URL, headers=headers, timeout=timeout)
    r.raise_for_status()
    return _parse(r.json().get("data") or [])


def _read_cache() -> tuple[list[ModelInfo], float] | None:
    if not _CACHE_FILE.exists():
        return None
    try:
        with _CACHE_FILE.open(encoding="utf-8") as f:
            blob = json.load(f)
        return _parse(blob.get("models") or []), float(blob.get("fetched_at") or 0)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None


def _write_cache(raw: list[dict]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_FILE.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump({"fetched_at": time.time(), "models": raw}, f)
        tmp.replace(_CACHE_FILE)
    except OSError as e:
        logger.warning(f"Could not write catalogue cache: {e}")


def openrouter_models(config: Config, force_refresh: bool = False) -> list[ModelInfo]:
    """Catalogue from cache when fresh, otherwise refetched.

    On a network failure we return whatever stale cache exists rather than
    raising — a day-old catalogue still routes fine, and failing here would
    take down inference for a reason that has nothing to do with inference."""
    global _memo

    if not force_refresh and _memo and (time.time() - _memo[1]) < _MEMO_TTL:
        return _memo[0]

    cached = _read_cache()
    if cached and not force_refresh:
        models, fetched_at = cached
        if time.time() - fetched_at < CATALOGUE_TTL_SECONDS:
            _memo = (models, time.time())
            return models

    try:
        headers = {"Authorization": f"Bearer {config.openrouter_api_key}"} if config.has_cloud else {}
        r = httpx.get(OPENROUTER_MODELS_URL, headers=headers, timeout=20.0)
        r.raise_for_status()
        raw = r.json().get("data") or []
        _write_cache(raw)
        models = _parse(raw)
        _memo = (models, time.time())
        return models
    except Exception as e:
        if cached:
            logger.warning(f"Catalogue refresh failed ({e}); using stale cache.")
            _memo = (cached[0], time.time())
            return cached[0]
        logger.warning(f"Catalogue fetch failed and no cache available: {e}")
        return []


def ollama_models(config: Config) -> list[ModelInfo]:
    """Locally-installed Ollama models, largest first (bigger local models
    generally reason better, and being installed means they already fit).
    Returns [] when Ollama isn't running — that's a normal state, not an error."""
    try:
        r = httpx.get(f"{config.ollama_host}/api/tags", timeout=1.0)
        if r.status_code != 200:
            return []
        raw = r.json().get("models") or []
    except Exception:
        return []

    keep = [m for m in raw if m.get("name") and not is_specialised(m["name"])]
    keep.sort(key=lambda m: m.get("size", 0), reverse=True)
    return [
        ModelInfo(id=f"{OLLAMA_PREFIX}{m['name']}", name=f"{m['name']} (local)")
        for m in keep
    ]


def forget() -> None:
    """Drop the in-process memo. Tests and long-lived daemons want this."""
    global _memo
    _memo = None
