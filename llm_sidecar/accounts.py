"""Allowances: what each provider has left, in one place.

Every provider limits you differently, and none of them in the same unit:

  ollama        runs on this machine; no allowance at all
  openrouter    prepaid credit, an optional cap per key, and a daily count
                of free-model requests. Its /key endpoint reports all three
                for an ordinary key.
  claude/       a subscription's rolling usage windows (five hours and
                seven days). Claude Code reports them on every call, so they
                are recorded as calls happen and shown "as of" the last one.
  claude-api/   API credits. No ordinary key can read the balance, so this
                is the sidecar's own count against the allowance you set.

Each entry says where its numbers came from (`source`: live, last-seen or
estimated), because a figure that looks live but is a day old is worse than
one labelled as old.

The readings also steer routing: a provider that is known to be out is
skipped or refused up front instead of being tried and failing (see
openrouter_free_exhausted and claude_window_blocked).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import threading
import time
from pathlib import Path

import httpx

from . import ledger
from .config import Config

logger = logging.getLogger("llm_sidecar.accounts")

OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
OPENROUTER_TTL = 5 * 60
WINDOWS_FILE_NAME = "claude-windows.json"

_lock = threading.Lock()
_openrouter: dict = {"at": 0.0, "key": None, "data": None, "error": None}


# ── claude/ subscription windows ─────────────────────────────────────────


def _windows_file() -> Path:
    return ledger.DATA_DIR / WINDOWS_FILE_NAME


def note_claude_window(info: dict) -> None:
    """Record the usage windows a `claude -p` call reported. Never raises."""
    try:
        path = _windows_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"seen_at": time.time(), "info": info}), encoding="utf-8")
        tmp.replace(path)
    except (OSError, TypeError, ValueError) as e:
        logger.debug(f"could not save claude windows: {e}")


def claude_windows() -> dict | None:
    try:
        data = json.loads(_windows_file().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and isinstance(data.get("info"), dict) else None
    except (OSError, ValueError):
        return None


def _windows_from(info: dict) -> list[dict]:
    """[{name, used, resets_at, status}] from either shape the CLI sends."""
    out = []
    unified = info.get("unifiedWindows")
    if isinstance(unified, dict) and unified:
        for name, w in unified.items():
            if isinstance(w, dict):
                out.append({"name": name, "used": w.get("utilization"),
                            "resets_at": w.get("resetsAt"), "status": w.get("status")})
    elif info.get("rateLimitType"):
        out.append({"name": info["rateLimitType"], "used": info.get("utilization"),
                    "resets_at": info.get("resetsAt"), "status": info.get("status")})
    return out


def claude_window_blocked(now: float | None = None) -> str | None:
    """A message if the last reading said the subscription is out and the
    window has not reset since, else None."""
    data = claude_windows()
    if not data:
        return None
    now = now or time.time()
    info = data["info"]
    for w in _windows_from(info) or [{"status": info.get("status"),
                                      "resets_at": info.get("resetsAt"),
                                      "name": "usage"}]:
        if w.get("status") == "rejected" and (w.get("resets_at") or 0) > now:
            return (f"Claude subscription {_window_name(w['name'])} limit reached; "
                    f"resets {_when(w['resets_at'])}.")
    return None


_WINDOW_NAMES = {"five_hour": "5-hour", "seven_day": "weekly",
                 "seven_day_opus": "weekly Opus", "seven_day_sonnet": "weekly Sonnet"}


def _window_name(name: str) -> str:
    return _WINDOW_NAMES.get(name, name.replace("_", " "))


def _when(ts) -> str:
    """'15:00 today', 'Thu 09:00', or a date, in local time."""
    if not ts:
        return "?"
    t = _dt.datetime.fromtimestamp(float(ts))
    today = _dt.date.today()
    if t.date() == today:
        return f"{t:%H:%M} today"
    if 0 < (t.date() - today).days < 7:
        return f"{t:%a %H:%M}"
    return f"{t:%Y-%m-%d}"


def _ago(ts: float) -> str:
    s = max(0, time.time() - ts)
    if s < 90:
        return "just now"
    if s < 5400:
        return f"{int(s // 60)} min ago"
    if s < 172800:
        return f"{int(s // 3600)} h ago"
    return f"{int(s // 86400)} days ago"


def _claude_subscription(config: Config, refresh: bool) -> dict | None:
    from . import claude_cli

    if refresh and claude_cli.available(config):
        try:
            claude_cli.check(config)   # one minimal call; records the windows
        except Exception as e:
            logger.debug(f"claude refresh failed: {e}")

    data = claude_windows()
    if data is None and not claude_cli.available(config):
        return None
    entry = {"provider": "claude", "label": "Claude subscription (claude/)",
             "kind": "subscription", "state": "unknown", "source": "last-seen",
             "as_of": None, "summary": "", "windows": []}
    if data is None:
        entry["summary"] = "No reading yet: it appears after the next claude/ call."
        return entry

    info = data["info"]
    windows = _windows_from(info)
    entry["as_of"] = data["seen_at"]
    entry["windows"] = [{**w, "name": _window_name(w["name"]),
                         "resets": _when(w["resets_at"])} for w in windows]
    statuses = {w.get("status") for w in windows} | {info.get("status")}
    worst = max((w.get("used") or 0) for w in windows) if windows else None
    stale = all((w.get("resets_at") or 0) < time.time() for w in windows) if windows else True

    if stale:
        entry["state"] = "ok"
        entry["summary"] = ("Every window has reset since the last reading "
                            f"({_ago(data['seen_at'])}).")
        return entry
    if "rejected" in statuses:
        entry["state"] = "exhausted"
    elif "allowed_warning" in statuses or (worst is not None and worst >= 0.9):
        entry["state"] = "warn"
    else:
        entry["state"] = "ok"
    parts = [f"{_window_name(w['name'])} {round((w.get('used') or 0) * 100)}% used "
             f"(resets {_when(w['resets_at'])})" for w in windows]
    entry["summary"] = " · ".join(parts) + f" · as of {_ago(data['seen_at'])}"
    if info.get("overageStatus") == "rejected" or info.get("overageEnabled") is False:
        entry["extra_usage"] = False
    return entry


# ── openrouter ───────────────────────────────────────────────────────────


def _fetch_openrouter(config: Config, refresh: bool) -> tuple[dict | None, str | None, float]:
    key = config.openrouter_api_key
    with _lock:
        fresh = (_openrouter["key"] == key
                 and time.time() - _openrouter["at"] < OPENROUTER_TTL)
        if fresh and not refresh:
            return _openrouter["data"], _openrouter["error"], _openrouter["at"]
    data, error = None, None
    try:
        r = httpx.get(OPENROUTER_KEY_URL, headers={"Authorization": f"Bearer {key}"},
                      timeout=8.0)
        if r.status_code == 200:
            data = (r.json() or {}).get("data") or {}
        elif r.status_code == 401:
            error = "OpenRouter rejected the key."
        else:
            error = f"OpenRouter answered HTTP {r.status_code}."
    except (httpx.HTTPError, ValueError) as e:
        error = f"Could not reach OpenRouter ({type(e).__name__})."
    with _lock:
        _openrouter.update(at=time.time(), key=key, data=data, error=error)
    return data, error, _openrouter["at"]


def openrouter_free_exhausted(config: Config) -> bool:
    """True when the last reading (no fetch) says today's free-model
    requests are used up. Used by the picker to go straight to local models
    instead of burning a round of 429s."""
    with _lock:
        if _openrouter["key"] != config.openrouter_api_key or not _openrouter["data"]:
            return False
        if time.time() - _openrouter["at"] > OPENROUTER_TTL:
            return False
        free = (_openrouter["data"].get("free_model_daily_requests") or {})
    rem = free.get("remaining")
    return isinstance(rem, (int, float)) and rem <= 0


def _usd(v) -> str:
    """Two decimals, except a real but tiny spend, which "$0.00" would hide."""
    v = float(v or 0)
    return "<$0.01" if 0 < v < 0.005 else f"${v:,.2f}"


def _openrouter_entry(config: Config, refresh: bool) -> dict | None:
    if not config.openrouter_api_key:
        return None
    data, error, at = _fetch_openrouter(config, refresh)
    entry = {"provider": "openrouter", "label": "OpenRouter", "kind": "credits",
             "state": "unknown", "source": "live", "as_of": at, "summary": ""}
    if error:
        entry["summary"] = error
        return entry
    parts, state = [], "ok"
    if data.get("usage_monthly") is not None:
        parts.append(f"{_usd(data['usage_monthly'])} spent this month")
    if data.get("limit") is not None:
        left = data.get("limit_remaining")
        reset = f", resets {data['limit_reset']}" if data.get("limit_reset") else ""
        parts.append(f"{_usd(left or 0)} of {_usd(data['limit'])} key limit left{reset}")
        if left is not None and float(left) <= 0:
            state = "exhausted"
        elif left is not None and float(left) < 0.1 * float(data["limit"] or 1):
            state = "warn"
    free = data.get("free_model_daily_requests") or {}
    if free.get("limit") is not None:
        parts.append(f"{free.get('remaining', '?')} of {free['limit']} free-model "
                     "requests left today")
        if (free.get("remaining") or 0) <= 0 and state == "ok":
            state = "warn"
    if data.get("is_free_tier"):
        parts.append("no credit bought yet")
    entry["state"] = state
    entry["summary"] = " · ".join(parts) or "Key accepted; no usage reported."
    entry["details"] = {k: data.get(k) for k in (
        "usage", "usage_daily", "usage_weekly", "usage_monthly", "limit",
        "limit_remaining", "limit_reset", "is_free_tier", "free_model_daily_requests")}
    # The account's credit balance needs a management key, which this
    # sidecar deliberately never asks for.
    entry["note"] = "Account balance needs a management key; this shows the key's view."
    return entry


# ── claude-api/ credits ──────────────────────────────────────────────────


def _claude_api_entry(config: Config) -> dict | None:
    from . import claude_api

    if not config.claude_api_enabled and claude_api.credential_source() is None:
        return None
    s = claude_api.spend(config)
    entry = {"provider": "claude-api", "label": "Claude API credits (claude-api/)",
             "kind": "credits", "source": "estimated", "as_of": time.time(),
             "spent_usd": round(s["spent_usd"], 4), "allowance_usd": s["allowance_usd"],
             "remaining_usd": s["remaining_usd"], "resets": s["resets"],
             "calls": s.get("calls", 0)}
    if not config.claude_api_enabled:
        entry["state"] = "off"
        entry["summary"] = ("Credentials found, provider off "
                            "(`llm-sidecar config claude-api on`).")
        return entry
    allowance = s["allowance_usd"] or 0
    used = s["spent_usd"] / allowance if allowance else 1
    if s.get("exhausted") or used >= claude_api.STOP_AT:
        entry["state"] = "exhausted"
    elif used >= 0.75:
        entry["state"] = "warn"
    else:
        entry["state"] = "ok"
    entry["summary"] = (f"{_usd(s['spent_usd'])} of {_usd(allowance)} used this period "
                        f"({s.get('calls', 0)} calls, sidecar's own count) · renews {s['resets']}")
    if s.get("exhausted"):
        entry["summary"] += " · the API reported the balance too low"
    return entry


# ── ollama ───────────────────────────────────────────────────────────────


def _ollama_entry(config: Config) -> dict:
    entry = {"provider": "ollama", "label": "Ollama (local)", "kind": "local",
             "source": "live", "as_of": time.time()}
    try:
        r = httpx.get(f"{config.ollama_host}/api/tags", timeout=2.0)
        n = len((r.json() or {}).get("models") or []) if r.status_code == 200 else 0
        entry["state"] = "ok" if n else "unknown"
        entry["summary"] = (f"{n} models on this machine · no allowance"
                            if n else "Running, but no models installed.")
    except (httpx.HTTPError, ValueError):
        entry["state"] = "off"
        entry["summary"] = "Not running."
    return entry


# ── the report ───────────────────────────────────────────────────────────


def report(config: Config, refresh: bool = False) -> dict:
    """Every provider's allowance. `refresh` re-reads OpenRouter now and
    spends one minimal claude/ call to update the subscription windows."""
    entries = [_ollama_entry(config)]
    for build in (lambda: _openrouter_entry(config, refresh),
                  lambda: _claude_subscription(config, refresh),
                  lambda: _claude_api_entry(config)):
        try:
            e = build()
        except Exception as ex:
            logger.warning(f"allowance check failed: {ex}")
            continue
        if e:
            entries.append(e)
    return {"checked_at": time.time(), "providers": entries}
