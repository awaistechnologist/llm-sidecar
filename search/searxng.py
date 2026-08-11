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

import httpx

from ..types import SearchResult

logger = logging.getLogger("llm_sidecar.search.searxng")

# Cached probe results per base URL. Checking liveness on every search would
# double the request count for a fact that changes ~never within a process.
_probe_cache: dict[str, bool] = {}


def available(config) -> bool:
    """Is a SearXNG instance reachable, and does it allow JSON output?

    The JSON format is disabled in SearXNG's default settings.yml, so a live
    instance can still be unusable to us. Probing with `format=json` tests
    the thing we actually need rather than mere reachability."""
    url = config.searxng_url
    if url in _probe_cache:
        return _probe_cache[url]

    ok = False
    try:
        r = httpx.get(
            f"{url.rstrip('/')}/search",
            params={"q": "test", "format": "json"},
            timeout=2.0,
        )
        ok = r.status_code == 200 and isinstance(r.json().get("results"), list)
        if r.status_code == 403:
            logger.warning(
                f"SearXNG at {url} rejected format=json — add 'json' to "
                "search.formats in settings.yml to enable it."
            )
    except Exception:
        ok = False

    _probe_cache[url] = ok
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
