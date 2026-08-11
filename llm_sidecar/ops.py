"""Structured operations — bounded tasks with a shape you can rely on.

These are the jobs people actually delegate to a cheap model: condense this,
sort these into buckets, pull these fields out. What makes them annoying to
do by hand is not the prompt, it's that the output has to be *parseable* and
models drift — a stray fence, an invented category, a missing field.

Each function here pins the contract: a fixed schema, a validated result, and
a documented failure mode. Every one runs at temperature 0, which also makes
them cacheable.
"""

from __future__ import annotations

import json
import logging
import re

from .types import SidecarError

logger = logging.getLogger("llm_sidecar.ops")

MAX_CHARS = 100_000


def parse_json_response(raw: str) -> dict:
    """Recover JSON from a model that was asked for JSON.

    Fences, preamble and trailing chatter are all common enough that treating
    them as errors would mean failing on output that's perfectly usable."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
        # A top-level array is valid JSON but breaks the dict contract every
        # caller here relies on, so it gets wrapped rather than returned raw.
        return parsed if isinstance(parsed, dict) else {"result": parsed}
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
                return parsed if isinstance(parsed, dict) else {"result": parsed}
            except json.JSONDecodeError:
                continue
    raise SidecarError(f"Expected JSON, got: {raw[:200]}")


def _guard(text: str) -> str:
    if not text or not text.strip():
        raise SidecarError("Nothing to work with: input text is empty.")
    if len(text) > MAX_CHARS:
        raise SidecarError(f"Input is {len(text)} chars; the limit is {MAX_CHARS}. Chunk it first.")
    return text


# ── summarise ─────────────────────────────────────────────────────────────────

STYLES = {
    "brief": "Two or three sentences. No preamble.",
    "bullets": "Five to eight bullet points, each one line, starting with '- '.",
    "detailed": "A thorough prose summary, several paragraphs, preserving specifics and numbers.",
    "tldr": "One sentence, under 30 words.",
}


def summarise(sidecar, text: str, style: str = "brief", focus: str = "") -> str:
    """Condense text. `focus` narrows what's worth keeping."""
    _guard(text)
    if style not in STYLES:
        raise SidecarError(f"Unknown style {style!r}. Expected one of: {', '.join(STYLES)}.")

    system = (
        "You summarise text accurately and without editorialising. "
        f"{STYLES[style]} "
        "Never invent detail that is not in the source. Output only the summary."
    )
    if focus:
        system += f" Focus specifically on: {focus}."

    return sidecar.complete(
        f"Summarise the following:\n\n{text}",
        system=system,
        tier="fast",
        temperature=0.0,
        max_tokens=2000,
        operation="summarise",
    ).text.strip()


# ── classify ──────────────────────────────────────────────────────────────────

def classify(sidecar, items: list[str], labels: list[str], multi: bool = False) -> list[dict]:
    """Sort items into your labels.

    The invented-category problem is the whole difficulty here, so anything
    the model returns that isn't one of `labels` is dropped, and an item left
    with nothing is reported as "unknown" rather than silently mislabelled."""
    items = [i for i in items if i and i.strip()]
    if not items:
        return []
    if len(labels) < 2:
        raise SidecarError("Provide at least two labels.")

    listing = "\n".join(f"{i}. {t}" for i, t in enumerate(items, 1))
    shape = (
        '{"results": [{"item": 1, "labels": ["..."]}]}' if multi
        else '{"results": [{"item": 1, "label": "..."}]}'
    )
    system = (
        "You are a precise classifier. Use ONLY the labels provided — never "
        "invent a new one. " + ("Assign every applicable label." if multi else "Assign exactly one label.")
        + " Respond with valid JSON only."
    )
    prompt = (
        f"Labels: {', '.join(labels)}\n\nItems:\n{listing}\n\n"
        f"Respond with exactly this shape, one entry per item, in order:\n{shape}"
    )

    raw = sidecar.complete(
        prompt, system=system, tier="fast", temperature=0.0,
        max_tokens=200 + 60 * len(items), operation="classify",
    ).text

    valid = set(labels)
    by_index: dict[int, dict] = {}
    for entry in parse_json_response(raw).get("results") or []:
        try:
            by_index[int(entry.get("item", 0))] = entry
        except (TypeError, ValueError):
            continue

    out = []
    for i, item in enumerate(items, 1):
        entry = by_index.get(i, {})
        if multi:
            got = [x for x in (entry.get("labels") or []) if x in valid]
            out.append({"item": item, "labels": got or ["unknown"]})
        else:
            label = entry.get("label")
            out.append({"item": item, "label": label if label in valid else "unknown"})
    return out


# ── extract ───────────────────────────────────────────────────────────────────

def extract(sidecar, text: str, fields: dict[str, str]) -> dict:
    """Pull named fields out of unstructured text.

    `fields` maps a field name to a description of what it is. Anything not
    found comes back as None — an explicit gap the caller can branch on,
    rather than a plausible-looking guess."""
    _guard(text)
    if not fields:
        raise SidecarError("Provide at least one field to extract.")

    spec = "\n".join(f'  "{k}": {v}' for k, v in fields.items())
    system = (
        "You extract structured data from text. Use null for anything not "
        "stated in the text — never guess or infer a value that is not there. "
        "Respond with valid JSON only."
    )
    prompt = (
        f"Extract these fields:\n{{\n{spec}\n}}\n\n"
        f"From this text:\n\n{text}\n\n"
        "Respond with a JSON object using exactly those field names."
    )

    raw = sidecar.complete(
        prompt, system=system, tier="fast", temperature=0.0,
        max_tokens=1500, operation="extract",
    ).text
    parsed = parse_json_response(raw)
    # Pin the shape to what was asked for, so callers can index without
    # checking — extra keys the model invented are dropped.
    return {k: parsed.get(k) for k in fields}
