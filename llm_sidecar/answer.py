"""Grounded question answering — search, read, answer, cite.

Everything else in this package assumes you arrive with a *statement*:
`verify` grades a claim, `fact_check` extracts claims from prose. Nothing took
a *question*, which is backwards for a tool whose point is grounding — you had
to already know the answer before you could check it.

The pipeline: search the question, read the top pages in full, then answer
from those pages only. Three properties matter more than fluency:

  - the model is told it may use nothing but the retrieved text, so a stale
    training-data answer can't leak through dressed as a current one
  - it must say so when the sources don't contain the answer, rather than
    producing a confident guess — `grounded` is False and the caller can act
  - citations are the sources it actually used, not everything we fetched

Reads full pages rather than snippets from the outset. A 400-character search
snippet rarely contains a specific figure, and this is exactly the case where
the number is the whole answer.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from .config import Config
from .types import Answer, SearchResult, SidecarError

logger = logging.getLogger("llm_sidecar.answer")

MAX_QUESTION_CHARS = 2000

ANSWER_SYSTEM = """You answer questions using ONLY the numbered sources provided.

Rules:
- Use nothing but the sources. Your own knowledge is not evidence, and may be
  out of date. If the sources disagree with what you remember, the sources win.
- "answered" means the sources answer THE SPECIFIC QUESTION ASKED. It does not
  mean you managed to write something using the sources. Sources that merely
  share a topic, or share words, with the question are NOT an answer: set
  "answered" to false and say what is missing.
  Example: asked "what did I have for breakfast on 3 March 2019", sources
  about breakfast in general answer nothing — that is "answered": false.
- Questions about a specific private individual, or about a person's own past,
  cannot be answered from web sources. Set "answered" to false.
- A wrong confident answer is far worse than admitting the sources fall short.
  When you are unsure whether the sources really settle it, they do not.
- Prefer the most recent and most authoritative source when they conflict, and
  say that they conflicted.
- Include specific figures, dates and units exactly as the sources give them.
- List the numbers of the sources you actually used, not every source shown.

Respond ONLY with valid JSON, no markdown fences, no commentary."""

ANSWER_TEMPLATE = """SOURCES:

{sources}

QUESTION: {question}

Respond with exactly this shape:
{{"answered": true|false,
  "answer": "a direct answer in one to three sentences, or what is missing",
  "sources_used": [1, 2],
  "caveat": "optional: a conflict between sources, or how dated the figure is"}}"""


def _fetch(result: SearchResult, config: Config, max_chars: int) -> dict:
    """Full text for one result, falling back to its snippet."""
    from . import search as search_mod

    try:
        text = search_mod.read_url(result.url, config, max_chars=max_chars)
    except Exception as e:
        logger.debug(f"answer: could not read {result.url}: {e}")
        text = result.snippet
    return {"title": result.title, "url": result.url, "text": text}


def gather(question: str, config: Config, query: str | None = None,
           max_sources: int = 4, read_pages: int = 3,
           page_chars: int = 5000) -> list[dict]:
    """Search, then read the top pages in parallel."""
    from . import search as search_mod

    results = search_mod.search(query or question, config, max_results=max_sources)
    if not results:
        return []

    to_read = results[:read_pages]
    workers = max(1, min(config.max_search_workers, len(to_read)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        docs = list(pool.map(lambda r: _fetch(r, config, page_chars), to_read))

    # Anything past read_pages contributes its snippet — cheap extra coverage
    # in case the pages we fetched turn out to be the wrong ones.
    for r in results[read_pages:]:
        docs.append({"title": r.title, "url": r.url, "text": r.snippet})
    return docs


def answer_question(
    question: str,
    config: Config,
    sidecar=None,
    *,
    query: str | None = None,
    model: str | None = None,
    tier: str = "balanced",
    max_sources: int = 4,
    read_pages: int = 3,
) -> Answer:
    """Answer a question from live sources. See module docstring."""
    question = (question or "").strip()
    if not question:
        raise SidecarError("Ask something.")
    if len(question) > MAX_QUESTION_CHARS:
        raise SidecarError(f"Question is {len(question)} chars; the limit is {MAX_QUESTION_CHARS}.")

    if sidecar is None:
        from . import Sidecar
        sidecar = Sidecar(config)

    docs = gather(question, config, query=query,
                  max_sources=max_sources, read_pages=read_pages)
    if not docs:
        return Answer(
            question=question,
            text="No search results came back, so there is nothing to answer from.",
            grounded=False,
        )

    blocks = "\n\n".join(
        f"[{i}] {d['title']} — {d['url']}\n{d['text']}"
        for i, d in enumerate(docs, 1)
    )
    completion = sidecar.complete(
        ANSWER_TEMPLATE.format(sources=blocks, question=question),
        system=ANSWER_SYSTEM,
        model=model,
        tier=tier,
        temperature=0.0,
        max_tokens=900,
        operation="answer",
    )

    from .ops import parse_json_response

    try:
        parsed = parse_json_response(completion.text)
    except SidecarError:
        # A model that ignored the schema still produced prose worth showing,
        # but we can't claim it was grounded or say which sources it used.
        return Answer(
            question=question,
            text=completion.text.strip(),
            grounded=False,
            sources=[d["url"] for d in docs],
            model=completion.model,
            caveat="The model did not return the expected format; treat this as unchecked.",
        )

    used = []
    for n in parsed.get("sources_used") or []:
        try:
            i = int(n)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= len(docs):
            used.append(docs[i - 1]["url"])

    return Answer(
        question=question,
        text=str(parsed.get("answer") or "").strip(),
        grounded=bool(parsed.get("answered")),
        # Fall back to everything consulted when the model cited nothing, so a
        # grounded answer is never presented without any provenance at all.
        sources=used or [d["url"] for d in docs],
        caveat=str(parsed.get("caveat") or "").strip(),
        model=completion.model,
    )
