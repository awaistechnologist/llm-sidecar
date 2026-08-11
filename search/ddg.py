"""DuckDuckGo search — the zero-config default.

Keyless and works on a fresh clone, which is why it's the fallback. It also
rate-limits aggressively and returns short snippets, which is why SearXNG
exists as an upgrade path.
"""

from __future__ import annotations

import logging

from ..types import SearchResult

logger = logging.getLogger("llm_sidecar.search.ddg")


def available(config) -> bool:
    """DDG needs no server, only the `ddgs` package."""
    try:
        import ddgs  # noqa: F401
        return True
    except ImportError:
        return False


def search(query: str, config, max_results: int = 5, news: bool = False) -> list[SearchResult]:
    from ddgs import DDGS

    out: list[SearchResult] = []

    if news:
        try:
            for r in DDGS().news(query, max_results=max_results):
                out.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url") or r.get("href", ""),
                    snippet=r.get("body", ""),
                    published=r.get("date", ""),
                    is_news=True,
                ))
        except Exception as e:
            logger.warning(f"DDG news search failed: {e}")

    try:
        for r in DDGS().text(query, max_results=max_results):
            out.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("href") or r.get("url", ""),
                snippet=r.get("body", ""),
            ))
    except Exception as e:
        logger.warning(f"DDG text search failed: {e}")

    return out
