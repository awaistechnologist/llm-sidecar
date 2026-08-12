"""SearXNG search — the opt-in upgrade.

SearXNG aggregates many upstream engines and, being self-hosted, imposes no
rate limit you didn't set yourself. That matters when several parallel jobs
each fire their own searches.

We talk to its JSON API directly rather than going through an MCP server:
this package is a plain Python process, and spawning a Node subprocess to
perform an HTTP GET would be all cost and no benefit. The MCP surface is for
agents calling *us*, not for us calling other people's tools.

Requires a running instance — see README for the docker one-liner.
"""

from __future__ import annotations

import logging
import time

import httpx

from ..types import SearchResult

logger = logging.getLogger("llm_sidecar.search.searxng")

# Cached probe results per base URL, with expiry. Checking liveness on every
# search would double the request count, but caching forever is worse: a
# single slow probe would disable SearXNG for the life of the process.
#
# Negatives expire quickly and positives slowly, deliberately. Being wrong
# about "it's down" costs you the better search engine silently; being wrong
# about "it's up" costs one failed request that already falls back.
_probe_cache: dict[str, tuple[bool, float]] = {}
_POSITIVE_TTL = 300.0
_NEGATIVE_TTL = 30.0

# SearXNG fans a query out to a dozen upstream engines and waits on them, so a
# healthy instance can take several seconds. Two seconds classified a working
# instance as dead and fell back to DuckDuckGo without saying anything.
PROBE_TIMEOUT = 8.0


def available(config) -> bool:
    """Is a SearXNG instance reachable, and does it allow JSON output?

    The JSON format is disabled in SearXNG's default settings.yml, so a live
    instance can still be unusable to us. Probing with `format=json` tests
    the thing we actually need rather than mere reachability."""
    url = config.searxng_url
    cached = _probe_cache.get(url)
    if cached is not None:
        ok, checked_at = cached
        ttl = _POSITIVE_TTL if ok else _NEGATIVE_TTL
        if time.time() - checked_at < ttl:
            return ok

    ok = False
    try:
        r = httpx.get(
            f"{url.rstrip('/')}/search",
            params={"q": "test", "format": "json"},
            timeout=PROBE_TIMEOUT,
        )
        ok = r.status_code == 200 and isinstance(r.json().get("results"), list)
        if r.status_code == 403:
            logger.warning(
                f"SearXNG at {url} rejected format=json — add 'json' to "
                "search.formats in settings.yml to enable it."
            )
    except httpx.TimeoutException:
        logger.warning(
            f"SearXNG at {url} did not answer within {PROBE_TIMEOUT}s — treating as "
            "unavailable for now and falling back to DuckDuckGo."
        )
    except Exception:
        ok = False

    _probe_cache[url] = (ok, time.time())
    return ok


def search(query: str, config, max_results: int = 5, news: bool = False) -> list[SearchResult]:
    params = {
        "q": query,
        "format": "json",
        "safesearch": 0,
    }
    if news:
        params["categories"] = "news"

    try:
        r = httpx.get(
            f"{config.searxng_url.rstrip('/')}/search",
            params=params,
            timeout=15.0,
        )
        r.raise_for_status()
        results = r.json().get("results") or []
    except Exception as e:
        logger.warning(f"SearXNG search failed: {e}")
        return []

    out = []
    for item in results[:max_results]:
        out.append(SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("content", ""),
            published=item.get("publishedDate") or "",
            is_news=news,
        ))
    return out
