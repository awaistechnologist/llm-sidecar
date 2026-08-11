"""Search — provider dispatch plus full-text URL reading.

`read_url` matters as much as `search` does: search gives you a ~200-character
snippet, which is a hard ceiling on how well anything downstream can reason
about a page. Fetching the page is the difference between grading a claim on a
headline and grading it on the article.
"""

from __future__ import annotations

import html
import logging
import re

import httpx

from ..config import Config
from ..types import SearchResult, SidecarError
from . import ddg, searxng

logger = logging.getLogger("llm_sidecar.search")

PROVIDERS = {"ddg": ddg, "searxng": searxng}


def resolve_provider(config: Config):
    """Pick the provider module to use.

    "auto" prefers SearXNG when an instance is actually answering and falls
    back to DDG, so a fresh install works with no setup but silently upgrades
    the moment the user runs one."""
    name = config.search_provider
    if name == "auto":
        if searxng.available(config):
            return searxng
        return ddg
    mod = PROVIDERS.get(name)
    if mod is None:
        raise SidecarError(f"Unknown search provider {name!r}. Expected one of: auto, ddg, searxng.")
    return mod


def search(
    query: str,
    config: Config | None = None,
    max_results: int = 5,
    news: bool = False,
) -> list[SearchResult]:
    """Search the web. Returns [] rather than raising when a provider is down —
    callers treat "no evidence" and "search broken" the same way, and an
    exception here would take down whatever it was feeding."""
    from .. import cache, config as config_mod

    config = config or config_mod.load()
    provider = resolve_provider(config)
    name = provider.__name__.rsplit(".", 1)[-1]

    hit = cache.get_search(config, query, name, max_results, news)
    if hit is not None:
        return [SearchResult(**r) for r in hit]

    results = provider.search(query, config, max_results=max_results, news=news)

    if not results and provider is searxng:
        logger.info("SearXNG returned nothing; falling back to DuckDuckGo.")
        results = ddg.search(query, config, max_results=max_results, news=news)

    # Only cache a non-empty result. An empty list is usually a throttle or a
    # transient failure, and caching that would turn a blip into an hour of
    # confidently returning nothing.
    if results:
        cache.put_search(config, query, name, max_results, news, [r.__dict__ for r in results])
    return results


# ── URL reading ───────────────────────────────────────────────────────────────

_SCRIPT_STYLE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.S | re.I)
_TAGS = re.compile(r"<[^>]+>")
_BLANKS = re.compile(r"\n{3,}")
# Block-level tags become newlines so paragraph structure survives stripping.
_BLOCKS = re.compile(r"</?(p|div|br|h[1-6]|li|tr|section|article)[^>]*>", re.I)


def read_url(url: str, config: Config | None = None, max_chars: int = 20000) -> str:
    """Fetch a page and return readable plain text.

    Intentionally a dependency-light extractor rather than a full readability
    port — good enough to feed a model, and it doesn't drag in a parser tree.
    Raises on fetch failure, since a caller asking for one specific URL wants
    to know it failed (unlike search, where empty is a normal answer)."""
    from .. import config as config_mod
    config = config or config_mod.load()

    try:
        r = httpx.get(
            url,
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": f"{config.app_title}/0.1 (+{config.referer})"},
        )
        r.raise_for_status()
    except Exception as e:
        raise SidecarError(f"Could not fetch {url}: {e}") from e

    ctype = r.headers.get("content-type", "")
    if "html" not in ctype and "text" not in ctype:
        raise SidecarError(f"Unsupported content-type for {url}: {ctype or 'unknown'}")

    text = _SCRIPT_STYLE.sub(" ", r.text)
    text = _BLOCKS.sub("\n", text)
    text = _TAGS.sub(" ", text)
    text = html.unescape(text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _BLANKS.sub("\n\n", text).strip()

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[truncated]"
    return text
