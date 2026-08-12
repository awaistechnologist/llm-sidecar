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
# HTML5 marks its own furniture. Dropping these is more general than trying to
# recognise individual menu items, and it catches things a line filter can't —
# Wikipedia's 170-language switcher lives inside <main>, but inside a <nav>.
_FURNITURE = re.compile(
    r"<(nav|header|footer|aside)\b[^>]*>.*?</\1>", re.S | re.I
)
_TAGS = re.compile(r"<[^>]+>")
# Block-level tags become newlines so paragraph structure survives stripping.
_BLOCKS = re.compile(r"</?(p|div|br|h[1-6]|li|tr|section|article)[^>]*>", re.I)
# Wrappers that mark the actual content. Most pages worth reading have one,
# and pulling it out removes the navigation wholesale instead of trying to
# recognise every menu item by name.
_MAIN_REGION = [
    re.compile(r'<div\b[^>]*\bid=["\']mw-content-text["\'][^>]*>(.*)</div>', re.S | re.I),
    re.compile(r"<article\b[^>]*>(.*?)</article>", re.S | re.I),
    re.compile(r"<main\b[^>]*>(.*?)</main>", re.S | re.I),
    re.compile(r'<div\b[^>]*\brole=["\']main["\'][^>]*>(.*)</div>', re.S | re.I),
    re.compile(r'<div\b[^>]*\bid=["\'](?:content|main|mw-content-text)["\'][^>]*>(.*)</div>', re.S | re.I),
]
# Chrome that survives tag-stripping and adds nothing a model can use.
_BOILERPLATE = re.compile(
    r"^(jump to content|main menu|move to sidebar|hide|show|toggle .*|navigation|"
    r"skip to (main )?content|search|menu|close|cookie[s]? (policy|settings)|"
    r"accept( all)?( cookies)?|sign in|log in|subscribe|share|print|donate)$",
    re.I,
)
# Never pull more than this off the wire. Without it, one link to a large file
# is an out-of-memory error — max_chars truncates *after* the whole body has
# already been read into the process.
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024


def read_url(url: str, config: Config | None = None, max_chars: int = 20000) -> str:
    """Fetch a page and return readable plain text.

    Intentionally a dependency-light extractor rather than a full readability
    port — good enough to feed a model, and it doesn't drag in a parser tree.
    Raises on fetch failure, since a caller asking for one specific URL wants
    to know it failed (unlike search, where empty is a normal answer)."""
    from .. import cache, config as config_mod

    config = config or config_mod.load()

    hit = cache.get_page(config, url, max_chars)
    if hit is not None:
        return hit

    try:
        with httpx.stream(
            "GET", url, timeout=20.0, follow_redirects=True,
            headers={"User-Agent": f"{config.app_title}/{__import__('llm_sidecar').__version__} "
                                   f"(+{config.referer})"},
        ) as r:
            r.raise_for_status()

            ctype = r.headers.get("content-type", "")
            if "html" not in ctype and "text" not in ctype:
                raise SidecarError(f"Unsupported content-type for {url}: {ctype or 'unknown'}")

            # Streamed and capped: we stop pulling once we have enough, rather
            # than buffering a whole file to throw most of it away.
            chunks, size = [], 0
            for chunk in r.iter_bytes():
                chunks.append(chunk)
                size += len(chunk)
                if size >= MAX_DOWNLOAD_BYTES:
                    logger.info(f"{url}: stopped at {MAX_DOWNLOAD_BYTES} bytes")
                    break
            raw = b"".join(chunks).decode(r.encoding or "utf-8", errors="replace")
    except SidecarError:
        raise
    except Exception as e:
        raise SidecarError(f"Could not fetch {url}: {e}") from e

    text = extract_text(raw, max_chars=max_chars)
    cache.put_page(config, url, max_chars, text)
    return text


def extract_text(raw_html: str, max_chars: int = 20000) -> str:
    """HTML to readable plain text. Separate from fetching so it's testable
    without a network call."""
    text = _SCRIPT_STYLE.sub(" ", raw_html)
    text = _FURNITURE.sub(" ", text)

    # Narrow to the content region when the page marks one. Guarded on length
    # because a mis-matched region (unbalanced tags, a stray <main> in a
    # sidebar) would silently throw the article away — keeping the whole page
    # is a much better failure than returning a nav menu.
    for pattern in _MAIN_REGION:
        m = pattern.search(text)
        if m and len(m.group(1)) > 500:
            text = m.group(1)
            break

    text = _BLOCKS.sub("\n", text)
    text = _TAGS.sub(" ", text)
    text = html.unescape(text)

    kept, seen_blank = [], False
    for line in text.splitlines():
        line = " ".join(line.split())        # collapse runs of whitespace
        if not line:
            # One blank line between blocks, never a run of them.
            if kept and not seen_blank:
                kept.append("")
                seen_blank = True
            continue
        if _BOILERPLATE.match(line):
            continue
        seen_blank = False
        kept.append(line)

    text = "\n".join(kept).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[truncated]"
    return text
