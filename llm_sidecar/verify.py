"""Claim verification — extract, search, judge, cite.

The pipeline: each claim gets its own targeted search, then a judge model
grades a batch of claims strictly against the evidence gathered for them.
Verdicts come back with the source URLs they were based on.

Two things make this worth having over a bare "ask a model if this is true":
  - the model is only allowed to say "supported" on the strength of retrieved
    evidence, not its own memory, which is what stops confident hallucination
  - claims are graded *as stated*, including negations — "X is a myth" asserts
    NOT-X, so evidence debunking X makes that line supported, not contradicted

Ported from Agora's engine/verify.py, with search behind the provider layer so
it gains SearXNG and full-text fetching automatically.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from .config import Config
from .types import ClaimVerdict, SidecarError

logger = logging.getLogger("llm_sidecar.verify")

VERDICTS = ("supported", "contradicted", "unverified", "not_a_claim")
BATCH_SIZE = 5
MAX_EVIDENCE_PER_CLAIM = 4
MAX_CLAIMS = 40

JUDGE_SYSTEM = """You are a careful fact-checking judge. For each numbered claim you are
given web search evidence. Grade each claim STRICTLY on the evidence provided:
- "supported": the evidence clearly backs the claim AS STATED.
- "contradicted": the evidence clearly conflicts with the claim AS STATED.
- "unverified": the evidence is insufficient, off-topic, or mixed. When in doubt, use this.
- "not_a_claim": the line is not a checkable factual statement (an opinion, greeting,
  question, or call to action like "follow for more").
IMPORTANT: grade the claim exactly as stated, including its negations. A line that says
"X is a myth" or "X? Not true" is asserting NOT-X — if the evidence debunks X, that line
is "supported", not "contradicted".
IMPORTANT: match the claim's actual subject, not merely its words. Evidence about a
replica, a namesake, a model, a different entity sharing the name, or a differently
located branch does NOT support a claim about the principal subject. "The Eiffel Tower
is in Berlin" is contradicted by evidence placing the Eiffel Tower in Paris, even if
some source mentions an Eiffel Tower replica in Berlin.
Never use outside knowledge to mark a claim "supported" — evidence only. You may use
well-established knowledge to mark an obviously false claim "contradicted".
Respond ONLY with valid JSON, no markdown fences, no commentary."""

JUDGE_TEMPLATE = """Grade these claims against their evidence:

{blocks}

Respond with EXACTLY this JSON shape, one entry per claim, in order:
{{"results": [{{"claim": 1, "verdict": "supported|contradicted|unverified|not_a_claim",
"note": "one short sentence explaining the verdict"}}]}}"""

EXTRACT_SYSTEM = """Extract every checkable factual claim from the text as a list of
self-contained sentences. Resolve pronouns and references so each claim stands alone
without the surrounding text. Skip opinions, questions, and calls to action.
Respond ONLY with valid JSON: {"claims": ["...", "..."]}"""


def _parse_json(raw: str) -> dict:
    """Shared with ops — models wrap JSON in prose or fences regardless."""
    from .ops import parse_json_response

    return parse_json_response(raw)


def gather_evidence(claim: str, config: Config, max_results: int = MAX_EVIDENCE_PER_CLAIM) -> list[dict]:
    """Targeted search for one claim."""
    from . import search as search_mod

    results = search_mod.search(claim, config, max_results=max_results)
    return [
        {"title": r.title, "url": r.url, "snippet": r.snippet[:400]}
        for r in results
    ]


def gather_all(claims: list[str], config: Config) -> dict[str, list[dict]]:
    """Evidence for every claim, searched in parallel.

    Sequential search was the dominant cost of a multi-claim check — twenty
    claims meant twenty round trips end to end. The worker count is capped in
    config because providers notice; DuckDuckGo especially."""
    workers = max(1, min(config.max_search_workers, len(claims)))
    if workers == 1:
        return {c: gather_evidence(c, config) for c in claims}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {c: pool.submit(gather_evidence, c, config) for c in claims}
        out = {}
        for claim, fut in futures.items():
            try:
                out[claim] = fut.result()
            except Exception as e:
                # One failed search shouldn't sink the batch — the claim just
                # gets graded with no evidence, which means "unverified".
                logger.warning(f"Evidence search failed for {claim[:50]!r}: {e}")
                out[claim] = []
        return out


def _judge_batch(
    batch: list[tuple[str, list[dict]]],
    sidecar,
    model: str | None,
) -> list[dict]:
    blocks = []
    for i, (claim, evidence) in enumerate(batch, 1):
        lines = [f"CLAIM {i}: {claim}", "EVIDENCE:"]
        if evidence:
            for e in evidence:
                lines.append(f"- {e['title']}: {e['snippet']}")
        else:
            lines.append("- (no evidence found)")
        blocks.append("\n".join(lines))

    completion = sidecar.complete(
        JUDGE_TEMPLATE.format(blocks="\n\n".join(blocks)),
        system=JUDGE_SYSTEM,
        model=model,
        tier="fast",
        max_tokens=1200,
        temperature=0.0,
        operation="verify",
    )
    return _parse_json(completion.text).get("results") or []


def extract_claims(text: str, sidecar, model: str | None = None) -> list[str]:
    """Pull self-contained factual claims out of a block of prose."""
    completion = sidecar.complete(
        f"Text:\n\n{text}",
        system=EXTRACT_SYSTEM,
        model=model,
        tier="fast",
        max_tokens=1500,
        temperature=0.0,
        operation="extract_claims",
    )
    claims = _parse_json(completion.text).get("claims") or []
    return [c for c in claims if isinstance(c, str) and c.strip()]


def verify_claims(
    claims: list[str],
    config: Config,
    model: str | None = None,
    sidecar=None,
) -> list[ClaimVerdict]:
    """Verify a list of claims. Returns one verdict per claim, in order.

    Routing goes through a Sidecar so the judge calls are cached and recorded
    like any other completion — re-checking a document after an edit should
    not re-pay for the claims that didn't change."""
    claims = [c.strip() for c in claims if c and c.strip()]
    if not claims:
        return []
    if len(claims) > MAX_CLAIMS:
        raise SidecarError(f"Too many claims ({len(claims)}); the limit is {MAX_CLAIMS}.")

    if sidecar is None:
        from . import Sidecar
        sidecar = Sidecar(config)

    evidence_by_claim = gather_all(claims, config)

    verdicts: list[ClaimVerdict] = []
    for start in range(0, len(claims), BATCH_SIZE):
        chunk = claims[start:start + BATCH_SIZE]
        batch = [(c, evidence_by_claim[c]) for c in chunk]
        try:
            graded = _judge_batch(batch, sidecar, model)
        except Exception as e:
            logger.warning(f"Judge call failed for batch at {start}: {e}")
            graded = []

        by_index = {}
        for g in graded:
            try:
                by_index[int(g.get("claim", 0))] = g
            except (TypeError, ValueError):
                continue

        for i, claim in enumerate(chunk, 1):
            g = by_index.get(i, {})
            verdict = g.get("verdict")
            if verdict not in VERDICTS:
                verdict = "unverified"
            verdicts.append(ClaimVerdict(
                claim=claim,
                verdict=verdict,
                note=g.get("note", "") if isinstance(g.get("note"), str) else "",
                sources=[e["url"] for e in evidence_by_claim[claim] if e.get("url")],
            ))

    return verdicts
