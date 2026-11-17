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

# SDK 2.0 renamed FastMCP to MCPServer and moved it up a level. The decorator
# and run() APIs are unchanged, so supporting both is just the import.
try:
    from mcp.server import MCPServer as _Server
except ImportError:  # mcp < 2.0
    from mcp.server.fastmcp import FastMCP as _Server

from . import Sidecar
from .types import SidecarError

# stdio is the transport — anything written to stdout that isn't protocol
# traffic corrupts the session.
logging.basicConfig(level=logging.WARNING)

mcp = _Server(
    name="llm-sidecar",
    instructions=(
        "Local capability sidecar. Use answer_question for anything whose answer "
        "could have changed since training — it searches, reads the pages and "
        "answers with citations. Use search_web and read_url to gather evidence "
        "yourself, verify_claims to fact-check statements against it, and "
        "delegate to hand bulk work to a cheap or local model."
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
def answer_question(question: str, query: str = "", max_sources: int = 4,
                    via: str = "local") -> dict:
    """
    Answer a question from live web sources, with citations.

    Searches, reads the top pages in full, and answers from those pages only —
    so the answer reflects the web now rather than training data. Use this
    instead of answering from memory whenever the answer could have changed:
    prices, populations, versions, who currently holds a position, what
    happened recently.

    Args:
        question:    The question, in full.
        query:       Optional search query, when the question phrases badly.
        max_sources: How many results to consider (default 4).
        via:         "local" searches with DuckDuckGo/SearXNG and reads the
                     pages — free, but degrades when those are rate-limited.
                     "openrouter" has OpenRouter retrieve and answer in one
                     call: better sources and no CAPTCHAs, but it is BILLED
                     per result and needs an API key. Only choose it when
                     local retrieval has already returned something useless.

    Returns {"answered", "answer", "sources", "caveat"}. Check "answered":
    false means the sources did not settle it and "answer" says what was
    missing — do not present that as a fact.
    """
    try:
        a = sidecar().answer(question, query=query or None,
                             max_sources=max_sources, via=via)
    except SidecarError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Answering failed: {e}"}
    return {"answered": a.grounded, "answer": a.text,
            "sources": a.sources, "caveat": a.caveat}


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

    Each claim gets its own targeted search; a verification model then grades every
    claim strictly against the evidence found — it is not allowed to mark a
    claim "supported" from its own knowledge. Claims are graded exactly as
    stated, so a line asserting "X is a myth" is supported when evidence
    debunks X.

    Args:
        claims: Self-contained factual statements (max 40). Resolve pronouns
                first — "it was founded in 1998" cannot be checked alone.
        model:  Optional verification model override, e.g. "ollama/qwen2.5:32b" to
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
def summarise(text: str, style: str = "brief", focus: str = "") -> dict:
    """
    Condense text using a cheap or local model, so long documents don't have
    to pass through your own context.

    Args:
        text:  The text to summarise.
        style: "brief" (2-3 sentences), "bullets", "detailed", or "tldr".
        focus: Optional — narrow the summary to one aspect, e.g. "security
               implications" or "what changed".
    """
    try:
        return {"summary": sidecar().summarise(text, style=style, focus=focus)}
    except SidecarError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Summarise failed: {e}"}


@mcp.tool()
def classify(items: list[str], labels: list[str], multi: bool = False) -> dict:
    """
    Sort items into labels you provide. Useful for triaging logs, tickets,
    or search results in bulk.

    Labels are enforced: anything the model invents is discarded and the item
    comes back as "unknown" rather than silently mislabelled.

    Args:
        items:  The things to classify.
        labels: The allowed labels (at least two).
        multi:  Allow more than one label per item.
    """
    try:
        return {"results": sidecar().classify(items, labels, multi=multi)}
    except SidecarError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Classify failed: {e}"}


@mcp.tool()
def extract_fields(text: str, fields: dict) -> dict:
    """
    Pull named fields out of unstructured text into a JSON object.

    Args:
        text:   The source text.
        fields: Map of field name -> description of what it is, e.g.
                {"invoice_total": "the total amount including tax",
                 "due_date": "the payment due date in ISO format"}.

    Fields not present in the text come back as null rather than a guess.
    """
    try:
        return {"fields": sidecar().extract(text, fields)}
    except SidecarError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Extract failed: {e}"}


@mcp.tool()
def fact_check_document(text: str) -> dict:
    """
    Extract every factual claim from a document and verify each against live
    web evidence. Combines extract_claims and verify_claims in one pass.

    Use on drafts, articles, or AI-generated text before publishing.
    """
    try:
        verdicts = sidecar().fact_check(text)
    except SidecarError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Fact check failed: {e}"}
    return {
        "checked": len(verdicts),
        "contradicted": sum(1 for v in verdicts if v.verdict == "contradicted"),
        "results": [
            {"claim": v.claim, "verdict": v.verdict, "note": v.note, "sources": v.sources}
            for v in verdicts
        ],
    }


@mcp.tool()
def usage_report(days: int = 30) -> dict:
    """
    What this sidecar has spent and on which models, over the last N days.
    Counts cached responses separately since those cost nothing.
    """
    try:
        return sidecar().usage(days)
    except Exception as e:
        return {"error": f"Usage lookup failed: {e}"}


@mcp.tool()
def sidecar_status() -> dict:
    """
    What this sidecar can currently do: whether a cloud key is present, the
    active search provider, this machine's memory and which local models
    actually fit in it, resolved tiers, cache size, and 30-day spend.
    """
    try:
        return sidecar().status()
    except Exception as e:
        return {"error": f"Status failed: {e}"}


# ── images ────────────────────────────────────────────────────────────────
#
# These let Claude Code make images from *any* repo on this machine without
# that repo carrying mflux, a model choice, or generation code of its own.


@mcp.tool()
def list_image_profiles() -> dict:
    """
    The image profiles available, with licence, memory peak and reference
    limit for each.

    Call this before generate_image if you are unsure which to use. Profiles
    are named for intent (draft, fast, cinematic, characters-consistent)
    rather than for a model, so you never need to know a model id.
    """
    try:
        from .images import profiles as p
        from .images import engine
        return {"available": engine.available(), "mflux": engine.mflux_version(),
                "profiles": p.describe()}
    except Exception as e:
        return {"error": f"Could not list profiles: {e}"}


@mcp.tool()
def generate_image(prompt: str, out_dir: str, profile: str = "draft",
                   stem: str = "image", negative: str = "",
                   width: int = 0, height: int = 0, seed: int = 0,
                   project: str = "", commercial: bool = True) -> dict:
    """
    Generate an image locally and write it to out_dir, with a manifest
    recording exactly how it was made.

    Takes 20s to several minutes depending on profile. Generation is
    serialised across this whole machine, so a second call waits rather than
    competing for memory.

    `commercial` defaults to true, which REFUSES non-commercial profiles
    (cinematic, kontext-edit). Pass false only if the output genuinely is not
    commercial. Leave width/height/seed at 0 for the profile's defaults and
    a random seed.
    """
    try:
        from . import images as im
        r = im.generate(im.ImageRequest(
            prompt=prompt, profile=profile, out_dir=out_dir, stem=stem,
            negative=negative or None,
            width=width or None, height=height or None, seed=seed or None,
            project=project, commercial=commercial,
        ))
        return {"path": r.path, "manifest": r.manifest_path, "seed": r.seed,
                "profile": r.profile, "model": r.model, "seconds": r.seconds,
                "warm_model": r.warm, "skipped": r.skipped}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def edit_with_references(prompt: str, refs: list, out_dir: str,
                         stem: str = "image", profile: str = "characters-consistent",
                         seed: int = 0, project: str = "") -> dict:
    """
    Generate an image that keeps the same character or object as the
    reference images — the same face, clothes and colours rather than a fresh
    interpretation of a description.

    `refs` is up to 3 image paths. This is the only way to get consistency
    across frames; describing a character in words produces a different one
    every time.
    """
    try:
        from . import images as im
        r = im.generate(im.ImageRequest(
            prompt=prompt, profile=profile, refs=list(refs), out_dir=out_dir,
            stem=stem, seed=seed or None, project=project,
        ))
        return {"path": r.path, "manifest": r.manifest_path, "seed": r.seed,
                "refs": len(refs), "seconds": r.seconds, "warm_model": r.warm}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def upscale_image(image_path: str, out_dir: str, resolution: int = 2048,
                  stem: str = "upscaled", project: str = "") -> dict:
    """
    Upscale an existing image with SeedVR2. `resolution` is the target long
    edge in pixels. Roughly 130s to 2400px on an M3 Max.
    """
    try:
        from . import images as im
        r = im.generate(im.ImageRequest(
            prompt="", profile="upscale", refs=[image_path], out_dir=out_dir,
            stem=stem, resolution=resolution, project=project,
        ))
        return {"path": r.path, "manifest": r.manifest_path,
                "seconds": r.seconds, "warm_model": r.warm}
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
