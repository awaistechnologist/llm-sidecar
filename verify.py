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

import json
import logging
import re

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
    """Models wrap JSON in prose or fences no matter how firmly you ask them not to."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Last resort: the outermost brace pair.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise SidecarError(f"Judge returned unparseable JSON: {raw[:200]}")


def gather_evidence(claim: str, config: Config, max_results: int = MAX_EVIDENCE_PER_CLAIM) -> list[dict]:
    """Targeted search for one claim."""
    from . import search as search_mod

    results = search_mod.search(claim, config, max_results=max_results)
    return [
        {"title": r.title, "url": r.url, "snippet": r.snippet[:400]}
        for r in results
    ]


def _judge_batch(
    batch: list[tuple[str, list[dict]]],
    model: str,
    config: Config,
) -> list[dict]:
    from . import client

    blocks = []
    for i, (claim, evidence) in enumerate(batch, 1):
        lines = [f"CLAIM {i}: {claim}", "EVIDENCE:"]
        if evidence:
            for e in evidence:
                lines.append(f"- {e['title']}: {e['snippet']}")
        else:
            lines.append("- (no evidence found)")
        blocks.append("\n".join(lines))

    prompt = JUDGE_TEMPLATE.format(blocks="\n\n".join(blocks))
    completion = client.complete(
        prompt,
        model=model,
        config=config,
        system=JUDGE_SYSTEM,
        max_tokens=1200,
        temperature=0.0,
    )
    parsed = _parse_json(completion.text)
    return parsed.get("results") or []


def extract_claims(text: str, config: Config, model: str) -> list[str]:
    """Pull self-contained factual claims out of a block of prose."""
    from . import client

    completion = client.complete(
        f"Text:\n\n{text}",
        model=model,
        config=config,
        system=EXTRACT_SYSTEM,
        max_tokens=1500,
        temperature=0.0,
    )
    claims = _parse_json(completion.text).get("claims") or []
    return [c for c in claims if isinstance(c, str) and c.strip()]


def verify_claims(
    claims: list[str],
    config: Config,
    model: str | None = None,
) -> list[ClaimVerdict]:
    """Verify a list of claims. Returns one verdict per claim, in order.

    `model` defaults to the configured fast tier, or an auto-picked one."""
    from . import picker

    claims = [c.strip() for c in claims if c and c.strip()]
    if not claims:
        return []
    if len(claims) > MAX_CLAIMS:
        raise SidecarError(f"Too many claims ({len(claims)}); the limit is {MAX_CLAIMS}.")

    if not model:
        model = config.tier_model("fast") or picker.pick(config).model_id

    evidence_by_claim = {c: gather_evidence(c, config) for c in claims}

    verdicts: list[ClaimVerdict] = []
    for start in range(0, len(claims), BATCH_SIZE):
        chunk = claims[start:start + BATCH_SIZE]
        batch = [(c, evidence_by_claim[c]) for c in chunk]
        try:
            graded = _judge_batch(batch, model, config)
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
