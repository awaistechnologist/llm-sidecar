"""Usage ledger — an append-only record of what was spent and on what.

Every response already carries its own cost. What that doesn't answer is
"what did this week cost me", "which model am I actually using", or "did
switching the default budget change anything" — questions that only exist
across calls.

JSONL, appended, never rewritten. A crash mid-write costs one line, which the
reader skips. There is no schema to migrate and the file is greppable, which
matters more here than query speed: this is a few thousand lines a month.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("llm_sidecar.ledger")

DATA_DIR = Path(os.getenv("LLM_SIDECAR_DATA", Path.home() / ".local" / "share" / "llm-sidecar"))
LEDGER_FILE = DATA_DIR / "usage.jsonl"
# Rotated once past this, keeping one previous generation. Append-only files
# are fine until they aren't; a busy daemon writes a line per call forever.
MAX_LEDGER_BYTES = 8 * 1024 * 1024


def _rotate_if_large() -> None:
    try:
        if LEDGER_FILE.exists() and LEDGER_FILE.stat().st_size > MAX_LEDGER_BYTES:
            LEDGER_FILE.replace(LEDGER_FILE.with_suffix(".jsonl.1"))
    except OSError:
        pass


def record(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    *,
    operation: str = "complete",
    cached: bool = False,
    latency_s: float = 0.0,
) -> None:
    """Append one entry. Never raises — a failed write must not fail a call
    that already succeeded."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.time(),
            "operation": operation,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": round(cost_usd, 8),
            "cached": cached,
            "latency_s": round(latency_s, 3),
        }
        _rotate_if_large()
        with LEDGER_FILE.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        logger.debug(f"ledger write failed: {e}")


def read(since: float | None = None) -> list[dict]:
    """Entries newest-last. Malformed lines are skipped rather than fatal."""
    if not LEDGER_FILE.exists():
        return []
    out = []
    try:
        with LEDGER_FILE.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if since and e.get("ts", 0) < since:
                    continue
                out.append(e)
    except OSError:
        return []
    return out


def summary(days: int | None = None) -> dict:
    """Totals, plus a per-model breakdown sorted by spend."""
    since = time.time() - days * 86400 if days else None
    entries = read(since)
    if not entries:
        return {"calls": 0, "cost_usd": 0.0, "total_tokens": 0, "cached": 0, "by_model": [], "days": days}

    by_model: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "cost_usd": 0.0, "total_tokens": 0, "cached": 0, "latency_s": 0.0}
    )
    total_cost, total_tokens, cached = 0.0, 0, 0

    for e in entries:
        tok = e.get("prompt_tokens", 0) + e.get("completion_tokens", 0)
        m = by_model[e.get("model", "unknown")]
        m["calls"] += 1
        m["cost_usd"] += e.get("cost_usd", 0.0)
        m["total_tokens"] += tok
        m["latency_s"] += e.get("latency_s", 0.0)
        if e.get("cached"):
            m["cached"] += 1
            cached += 1
        total_cost += e.get("cost_usd", 0.0)
        total_tokens += tok

    models = []
    for name, m in by_model.items():
        models.append({
            "model": name,
            "calls": m["calls"],
            "cost_usd": round(m["cost_usd"], 6),
            "total_tokens": m["total_tokens"],
            "cached": m["cached"],
            "avg_latency_s": round(m["latency_s"] / m["calls"], 2) if m["calls"] else 0.0,
        })
    models.sort(key=lambda x: (-x["cost_usd"], -x["calls"]))

    return {
        "calls": len(entries),
        "cost_usd": round(total_cost, 6),
        "total_tokens": total_tokens,
        "cached": cached,
        "since": entries[0]["ts"],
        "days": days,
        "by_model": models,
    }


def daily(days: int = 30) -> list[dict]:
    """Per-day totals, oldest first, including days with no activity.

    Gaps are filled deliberately: a chart that silently omits quiet days
    compresses the timeline and makes a spike look like a trend."""
    import datetime as _dt

    today = _dt.date.today()
    buckets = {
        (today - _dt.timedelta(days=n)).isoformat(): {"calls": 0, "cost_usd": 0.0, "tokens": 0}
        for n in range(days)
    }
    for e in read(time.time() - days * 86400):
        day = _dt.date.fromtimestamp(e.get("ts", 0)).isoformat()
        b = buckets.get(day)
        if b is None:
            continue
        b["calls"] += 1
        b["cost_usd"] += e.get("cost_usd", 0.0)
        b["tokens"] += e.get("prompt_tokens", 0) + e.get("completion_tokens", 0)

    return [
        {"date": d, **{k: (round(v, 6) if k == "cost_usd" else v) for k, v in vals.items()}}
        for d, vals in sorted(buckets.items())
    ]


def clear() -> bool:
    try:
        LEDGER_FILE.unlink(missing_ok=True)
        return True
    except OSError:
        return False
