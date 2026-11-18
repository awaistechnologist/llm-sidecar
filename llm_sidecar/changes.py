"""What changed, in a form a machine can act on.

A consumer integrating against this service pins assumptions: profile
names, default values, where files land, what a parameter means. Those
assumptions go stale when the service moves, and the usual remedy is the
maintainer telling each consumer by hand — which does not scale and, worse,
is silently skipped.

So the changelog is served rather than published. `serverInfo.version` tells
a client the service moved; `since()` tells it what moved and whether that
demands anything. The two together let a consumer answer "do I need to
change anything?" without a human in the loop.

The source of truth is CHANGELOG.md, parsed rather than duplicated. A second
hand-maintained list would drift from the first, which is the exact failure
this module exists to prevent.

An **Action required** section in a release is the load-bearing part: it is
the machine-readable promise that a consumer must do something, as opposed
to the ordinary case where they need only upgrade.
"""

from __future__ import annotations

import re
from pathlib import Path

# Shipped inside the package (see force-include in pyproject), with the repo
# root as the fallback so a source checkout works without a build.
_CANDIDATES = (
    Path(__file__).parent / "CHANGELOG.md",
    Path(__file__).parent.parent / "CHANGELOG.md",
)

_VERSION_RE = re.compile(r"^##\s*\[(?P<version>[0-9][^\]]*)\]\s*[—-]\s*(?P<date>.+?)\s*$")
_SECTION_RE = re.compile(r"^###\s+(?P<name>.+?)\s*$")
_BULLET_RE = re.compile(r"^[-*]\s+(?P<text>.+?)\s*$")

# A section whose contents a consumer must act on rather than merely read.
ACTION_SECTIONS = ("action required", "breaking", "removed")


def _source() -> str:
    for path in _CANDIDATES:
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                continue
    return ""


def parse_version(v: str) -> tuple:
    """(1, 2, 3) from "1.2.3". Unparseable parts sort last, not crash."""
    out = []
    for part in str(v).strip().lstrip("v").split("."):
        m = re.match(r"(\d+)", part)
        out.append(int(m.group(1)) if m else 0)
    while len(out) < 3:
        out.append(0)
    return tuple(out[:3])


def releases() -> list[dict]:
    """Every release, newest first, as {version, date, sections, action_required}."""
    text = _source()
    if not text:
        return []

    out: list[dict] = []
    current: dict | None = None
    section: str | None = None

    for line in text.splitlines():
        m = _VERSION_RE.match(line)
        if m:
            current = {
                "version": m.group("version"),
                "date": m.group("date"),
                "sections": {},
            }
            out.append(current)
            section = None
            continue
        if current is None:
            continue
        m = _SECTION_RE.match(line)
        if m:
            section = m.group("name")
            current["sections"].setdefault(section, [])
            continue
        m = _BULLET_RE.match(line)
        if m and section:
            # Markdown emphasis is noise to a consumer reading this
            # programmatically.
            text_ = re.sub(r"\*\*(.+?)\*\*", r"\1", m.group("text"))
            text_ = re.sub(r"`(.+?)`", r"\1", text_)
            current["sections"][section].append(text_)

    for rel in out:
        rel["action_required"] = [
            item
            for name, items in rel["sections"].items()
            if any(k in name.lower() for k in ACTION_SECTIONS)
            for item in items
        ]
    return out


def since(version: str | None = None, limit: int = 10) -> dict:
    """Everything released after `version`.

    `version` is what the caller last integrated against — not what is
    running. Omit it and you get the most recent release only, which is the
    sensible default for "what just changed".
    """
    from . import __version__

    rels = releases()
    if not rels:
        return {"current": __version__, "releases": [], "action_required": [],
                "note": "No changelog shipped with this build."}

    if version:
        floor = parse_version(version)
        newer = [r for r in rels if parse_version(r["version"]) > floor]
    else:
        newer = rels[:1]

    newer = newer[:limit]
    actions = [
        {"version": r["version"], "item": item}
        for r in newer for item in r["action_required"]
    ]
    return {
        "current": __version__,
        "asked_since": version or None,
        "behind_by": len(newer) if version else None,
        "releases": newer,
        "action_required": actions,
        "note": (
            f"{len(actions)} thing(s) need you to change something."
            if actions else
            "Nothing here requires a change on your side; upgrading is enough."
        ),
    }
