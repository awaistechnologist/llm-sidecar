"""MCP server — the capability door, for agents.

The daemon and this file serve the same core to two audiences that can't use
each other's interface. A coding tool can be told to POST somewhere else; an
agent can't, but it can call tools. So:

    daemon.py     -> raw inference, for programs      (HTTP /v1)
    mcp_server.py -> grounded capabilities, for agents (MCP stdio)

What's exposed here is deliberately *not* "call an LLM" — an agent already is
one. It's the things an agent genuinely can't do for itself: fetch live
evidence, and grade claims against it.

Client config:
    {"mcpServers": {"llm-sidecar": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "llm_sidecar.mcp_server"]}}}
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from . import Sidecar
from .types import SidecarError

# stdio is the transport — anything written to stdout that isn't protocol
# traffic corrupts the session.
logging.basicConfig(level=logging.WARNING)

mcp = FastMCP(
    name="llm-sidecar",
    instructions=(
        "Local capability sidecar. Use search_web and read_url to gather live "
        "evidence, verify_claims to fact-check statements against that evidence "
        "with citations, and delegate to hand bulk work to a cheap or local "
        "model instead of doing it yourself."
    ),
)

_sidecar: Sidecar | None = None


def sidecar() -> Sidecar:
    """Built lazily so an import never triggers a network call."""
    global _sidecar
    if _sidecar is None:
        _sidecar = Sidecar()
    return _sidecar


@mcp.tool()
def search_web(query: str, max_results: int = 5, news: bool = False) -> dict:
    """
    Search the web. Uses a local SearXNG instance when one is running,
    otherwise DuckDuckGo. No API key required either way.

    Args:
        query:       What to search for.
        max_results: How many results to return (default 5).
        news:        Bias toward recent news rather than general pages.

    Returns {"results": [{title, url, snippet, published}]}.
    Snippets are short — use read_url on a result to get the full text.
    """
    try:
        results = sidecar().search(query, max_results=max_results, news=news)
    except Exception as e:
        return {"error": f"Search failed: {e}"}
    return {
        "query": query,
        "results": [
            {"title": r.title, "url": r.url, "snippet": r.snippet, "published": r.published}
            for r in results
        ],
    }


@mcp.tool()
def read_url(url: str, max_chars: int = 20000) -> dict:
    """
    Fetch a web page and return its readable text.

    Use this after search_web when a snippet isn't enough to answer properly.
    Extraction is plain-text and keeps some navigation boilerplate.

    Args:
        url:       The page to fetch.
        max_chars: Truncation limit (default 20000).
    """
    try:
        return {"url": url, "text": sidecar().read_url(url, max_chars=max_chars)}
    except SidecarError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Could not read {url}: {e}"}


@mcp.tool()
def verify_claims(claims: list[str], model: str = "") -> dict:
    """
    Fact-check claims against fresh web evidence, with citations.

    Each claim gets its own targeted search; a judge model then grades every
    claim strictly against the evidence found — it is not allowed to mark a
    claim "supported" from its own knowledge. Claims are graded exactly as
    stated, so a line asserting "X is a myth" is supported when evidence
    debunks X.

    Args:
        claims: Self-contained factual statements (max 40). Resolve pronouns
                first — "it was founded in 1998" cannot be checked alone.
        model:  Optional judge model override, e.g. "ollama/qwen2.5:32b" to
                keep the whole check on-device.

    Returns {"results": [{claim, verdict, note, sources}]} where verdict is
    "supported" | "contradicted" | "unverified" | "not_a_claim".
    """
    try:
        verdicts = sidecar().verify(claims, model=model or None)
    except SidecarError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Verification failed: {e}"}
    return {
        "results": [
            {"claim": v.claim, "verdict": v.verdict, "note": v.note, "sources": v.sources}
            for v in verdicts
        ]
    }


@mcp.tool()
def extract_claims(text: str) -> dict:
    """
    Pull the checkable factual claims out of a block of prose, as
    self-contained sentences with pronouns resolved.

    Pair with verify_claims to fact-check an article or a draft.
    """
    try:
        return {"claims": sidecar().extract_claims(text)}
    except Exception as e:
        return {"error": f"Extraction failed: {e}"}


@mcp.tool()
def delegate(prompt: str, tier: str = "fast", budget: str = "") -> dict:
    """
    Hand a bounded task to a cheaper or local model instead of doing it
    yourself. Good for bulk summarising, log triage, reformatting, and
    classification over long text.

    Args:
        prompt: The full task, self-contained.
        tier:   "fast" | "balanced" | "powerful" (default "fast").
        budget: "free" | "cheap" | "best" — overrides the configured default.

    Returns {"text", "model", "cost_usd", "local"}. The model is chosen and
    verified at call time; you don't need to name one.
    """
    try:
        c = sidecar().complete(prompt, tier=tier, budget=budget or None, max_tokens=4000)
    except Exception as e:
        return {"error": f"Delegation failed: {e}"}
    return {
        "text": c.text,
        "model": c.model,
        "cost_usd": c.usage.cost_usd,
        "local": c.is_local,
        "total_tokens": c.usage.total_tokens,
    }


@mcp.tool()
def sidecar_status() -> dict:
    """
    What this sidecar can currently do: whether a cloud key is present, which
    local models are installed, the active search provider, and which models
    the tiers have resolved to so far.
    """
    sc = sidecar()
    from .search import resolve_provider

    local = [m.id for m in sc.models() if m.is_local]
    try:
        provider = resolve_provider(sc.config).__name__.rsplit(".", 1)[-1]
    except Exception:
        provider = "unknown"
    return {
        "cloud_configured": sc.config.has_cloud,
        "default_budget": sc.config.default_budget,
        "search_provider": provider,
        "local_models": local,
        "resolved_tiers": sc.resolved,
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
