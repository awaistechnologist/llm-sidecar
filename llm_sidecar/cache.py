"""Response cache — skip work you've already paid for.

Two things get cached, for different reasons:

  completions — deterministic requests (temperature 0) repeated across runs.
                Verification is the motivating case: re-checking a document
                after an edit re-judges every unchanged claim identically.
  searches    — the same claim searched twice in a session, and the reason
                a cached verify run does not hammer DuckDuckGo.

Non-deterministic completions (temperature > 0) are deliberately *not*
cached. Returning a byte-identical "creative" answer to a repeated prompt is
a surprising behaviour to inflict on a caller who asked for randomness.

Storage is one JSON file per entry under the cache dir. That's crude, but it
needs no daemon, no lock, and no schema migration — and an entry write that
gets interrupted leaves a junk file that the reader discards rather than a
corrupted database.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from .config import CACHE_DIR, Config

logger = logging.getLogger("llm_sidecar.cache")

_COMPLETIONS = CACHE_DIR / "completions"
_SEARCHES = CACHE_DIR / "searches"


def _key(*parts) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _read(path: Path, ttl: float) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            entry = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    # Always applied. A ttl of 0 means "expire immediately", not "never" —
    # the latter reading makes a disabled cache silently permanent. Use
    # float("inf") if you genuinely want entries to live forever.
    if time.time() - entry.get("stored_at", 0) > ttl:
        return None
    return entry.get("value")


def _write(path: Path, value) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump({"stored_at": time.time(), "value": value}, f)
        tmp.replace(path)
    except (OSError, TypeError) as e:
        # A cache that can't write is a slow cache, not a broken program.
        logger.debug(f"cache write failed: {e}")


# ── completions ───────────────────────────────────────────────────────────────

def cacheable(temperature: float) -> bool:
    """Only deterministic requests are worth (and safe) caching."""
    return temperature <= 0.0


def get_completion(config: Config, messages: list[dict], model: str,
                   max_tokens: int, temperature: float) -> dict | None:
    if not config.cache_enabled or not cacheable(temperature):
        return None
    k = _key("completion", messages, model, max_tokens, temperature)
    return _read(_COMPLETIONS / f"{k}.json", config.cache_ttl_seconds)


def put_completion(config: Config, messages: list[dict], model: str,
                   max_tokens: int, temperature: float, value: dict) -> None:
    if not config.cache_enabled or not cacheable(temperature):
        return
    k = _key("completion", messages, model, max_tokens, temperature)
    _write(_COMPLETIONS / f"{k}.json", value)


# ── searches ──────────────────────────────────────────────────────────────────

def get_search(config: Config, query: str, provider: str, max_results: int, news: bool):
    if not config.cache_enabled:
        return None
    k = _key("search", query, provider, max_results, news)
    return _read(_SEARCHES / f"{k}.json", config.search_cache_ttl_seconds)


def put_search(config: Config, query: str, provider: str, max_results: int, news: bool, value) -> None:
    if not config.cache_enabled:
        return
    k = _key("search", query, provider, max_results, news)
    _write(_SEARCHES / f"{k}.json", value)


# ── maintenance ───────────────────────────────────────────────────────────────

def stats() -> dict:
    def measure(d: Path) -> tuple[int, int]:
        if not d.exists():
            return 0, 0
        files = list(d.glob("*.json"))
        return len(files), sum(f.stat().st_size for f in files)

    c_n, c_b = measure(_COMPLETIONS)
    s_n, s_b = measure(_SEARCHES)
    return {
        "completions": c_n,
        "searches": s_n,
        "bytes": c_b + s_b,
        "path": str(CACHE_DIR),
    }


def clear() -> int:
    """Delete every cached entry. Returns how many files were removed."""
    removed = 0
    for d in (_COMPLETIONS, _SEARCHES):
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return removed
