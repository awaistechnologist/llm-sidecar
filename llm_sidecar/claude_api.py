"""The `claude-api/` provider: Claude over the Anthropic API, paid from API
credits.

This is a separate thing from `claude/`. That provider runs `claude -p` and
draws on a Claude *subscription*'s usage windows. This one calls the API
directly with the official SDK and is billed per token against *API
credits*. Max, Ultra and Team plans include a monthly API credit grant
(it arrives each month and does not roll over), so for many people this is
free up to an allowance and paid after it.

That "and paid after it" is what most of this module is for. No ordinary API
key can read the account's credit balance, so the sidecar keeps its own
count: every call's cost is worked out from its token counts and added to a
running total for the billing period. A call is refused *before* it is sent
if its worst case could take the total past the allowance. The count only
covers calls made through this sidecar, so the guard stops short of the
allowance rather than at it (STOP_AT).

Credentials come from the SDK's own resolution, so nothing is stored here:
ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or a sign-in made with
`ant auth login` (saved under ~/.config/anthropic/). The last is the one
that works for the background service, which does not see your shell's
environment.

The prefix is `claude-api/`, not `anthropic/`, because OpenRouter already
names its Claude models `anthropic/...` and those ids must keep routing to
OpenRouter.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import AsyncIterator

from . import ledger
from .config import Config
from .types import Completion, ModelInfo, SidecarError, Usage

logger = logging.getLogger("llm_sidecar.claude_api")

PREFIX = "claude-api/"

# Stop when the period's tracked spend plus the next call's worst case would
# pass this share of the allowance. The margin is for spend the sidecar
# cannot see: other tools, the Console's workbench, estimates that run low.
STOP_AT = 0.95

# Short names, resolved to the newest model of that family the account can
# use (see resolve()). Exact ids work too: claude-api/claude-opus-5-5.
FAMILIES: tuple[tuple[str, str, str], ...] = (
    # (alias, tier it suits, label)
    ("haiku", "fast", "Claude Haiku (API credits)"),
    ("sonnet", "balanced", "Claude Sonnet (API credits)"),
    ("opus", "powerful", "Claude Opus (API credits)"),
)
# Used when the Models API cannot be reached. Exact ids rot, which is why
# they are the fallback and not the source.
FALLBACK_IDS = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
}

# USD per million tokens: (input, output). For the cost estimate only, so a
# stale row costs accuracy, not money: an unknown model is priced at the top
# row, which can only make the guard stop sooner.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-2": (8.0, 40.0),
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.0, 5.0),
}
UNKNOWN_PRICE = (10.0, 50.0)
# Haiku 5.5 costs five times as much past 100K prompt tokens.
LONG_PROMPT = {"claude-haiku-5-5": (100_000, (0.50, 2.50))}
CACHE_WRITE, CACHE_READ = 1.25, 0.10

# How each model turns thinking off, where it can. A sidecar call is a
# summary or a classification, not a proof, and thinking tokens are billed
# as output. Models not listed keep adaptive thinking at low effort, with
# THINKING_ROOM added to max_tokens so the reasoning cannot eat the answer.
THINKING_OFF = {
    "claude-haiku-5-5": {"type": "disabled"},
    "claude-sonnet-5-5": {"type": "between_tools"},
}
THINKING_ROOM = 2048

# Refusal fallback: on a model that declines for policy reasons, the API
# re-runs the request on a server-chosen substitute inside the same call.
# On a model without fallbacks it is a no-op.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SPEND_FILE_NAME = "claude-api-spend.json"
_lock = threading.Lock()


class ClaudeApiError(SidecarError):
    """The API call failed."""


class ClaudeApiUnavailable(ClaudeApiError):
    """Switched off, SDK missing, or no credentials."""


class ClaudeApiAuthError(ClaudeApiError):
    """Credentials missing or rejected. Retrying will not help."""


class ClaudeApiBudgetError(ClaudeApiError):
    """This period's allowance is used up, by the sidecar's own count or by
    the API saying the credit balance is too low. Use another provider until
    the allowance renews."""


# ── the billing period and its running total ─────────────────────────────


def _spend_file() -> Path:
    return ledger.DATA_DIR / SPEND_FILE_NAME


def period_start(config: Config, today: _dt.date | None = None) -> _dt.date:
    """First day of the current billing period.

    Credits are granted on the plan's billing date, not the 1st, so the day
    is configurable. A day past the end of a short month means its last
    day."""
    today = today or _dt.date.today()
    day = max(1, min(31, int(config.claude_api_reset_day or 1)))

    def clamp(y: int, m: int) -> _dt.date:
        import calendar
        return _dt.date(y, m, min(day, calendar.monthrange(y, m)[1]))

    this = clamp(today.year, today.month)
    if today >= this:
        return this
    y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    return clamp(y, m)


def next_reset(config: Config, today: _dt.date | None = None) -> _dt.date:
    start = period_start(config, today)
    y, m = (start.year, start.month + 1) if start.month < 12 else (start.year + 1, 1)
    import calendar
    day = max(1, int(config.claude_api_reset_day or 1))
    return _dt.date(y, m, min(day, calendar.monthrange(y, m)[1]))


def _read_spend(config: Config) -> dict:
    """{period, spent_usd, calls, exhausted}. A new period starts at zero."""
    period = period_start(config).isoformat()
    try:
        data = json.loads(_spend_file().read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("period") == period:
            return data
    except (OSError, ValueError):
        pass
    return {"period": period, "spent_usd": 0.0, "calls": 0, "exhausted": None}


def _write_spend(data: dict) -> None:
    path = _spend_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        logger.warning(f"could not save claude-api spend: {e}")


def spend(config: Config) -> dict:
    """This period's tracked spend, for the allowances report."""
    with _lock:
        data = _read_spend(config)
    allowance = float(config.claude_api_monthly_credit_usd or 0)
    return {
        **data,
        "allowance_usd": allowance,
        "remaining_usd": round(max(0.0, allowance - data["spent_usd"]), 4),
        "resets": next_reset(config).isoformat(),
    }


def _add_spend(config: Config, cost: float) -> None:
    with _lock:
        data = _read_spend(config)
        data["spent_usd"] = round(data["spent_usd"] + cost, 6)
        data["calls"] = data.get("calls", 0) + 1
        _write_spend(data)


def _mark_exhausted(config: Config, why: str) -> None:
    with _lock:
        data = _read_spend(config)
        data["exhausted"] = {"at": time.time(), "why": why[:200]}
        _write_spend(data)


# ── cost ─────────────────────────────────────────────────────────────────


def price(model: str, prompt_tokens: int = 0) -> tuple[float, float]:
    base = PRICES.get(model, UNKNOWN_PRICE)
    long = LONG_PROMPT.get(model)
    if long and prompt_tokens > long[0]:
        return long[1]
    return base


def cost_of(model: str, usage) -> float:
    """USD for one response, from its usage block."""
    def get(name):
        v = getattr(usage, name, None) if not isinstance(usage, dict) else usage.get(name)
        return int(v or 0)

    inp, out = get("input_tokens"), get("output_tokens")
    write, read = get("cache_creation_input_tokens"), get("cache_read_input_tokens")
    p_in, p_out = price(model, inp + write + read)
    return (inp * p_in + write * p_in * CACHE_WRITE + read * p_in * CACHE_READ
            + out * p_out) / 1_000_000


def worst_case(model: str, messages: list[dict], max_tokens: int) -> float:
    """A generous bound for a call before it is sent: a token for every two
    characters (English runs nearer four), and the whole output budget
    used, thinking room included."""
    tokens = sum(len(_text(m.get("content"))) for m in messages) // 2 + 1
    p_in, p_out = price(model, tokens)
    room = 0 if model in THINKING_OFF else THINKING_ROOM
    return (tokens * p_in + (max_tokens + room) * p_out) / 1_000_000


def _guard(config: Config, model: str, messages: list[dict], max_tokens: int) -> None:
    allowance = float(config.claude_api_monthly_credit_usd or 0)
    s = _read_spend(config)
    resets = next_reset(config).isoformat()
    if s.get("exhausted"):
        raise ClaudeApiBudgetError(
            f"claude-api/: the API reported the credit balance too low this period "
            f"(resets {resets}). Using it again before then would need paid credit."
        )
    if allowance <= 0:
        raise ClaudeApiBudgetError(
            "claude-api/: no monthly allowance set. Set claude_api_monthly_credit_usd "
            "to what your plan grants (for example 100)."
        )
    need = worst_case(model, messages, max_tokens)
    if s["spent_usd"] + need > allowance * STOP_AT:
        raise ClaudeApiBudgetError(
            f"claude-api/: ${s['spent_usd']:.2f} of the ${allowance:.0f} allowance is used "
            f"and this call could cost up to ${need:.2f}. Stopping at "
            f"{int(STOP_AT * 100)}% so nothing is billed past the free credit; "
            f"it renews {resets}."
        )


# ── credentials and the client ───────────────────────────────────────────


def sdk_installed() -> bool:
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def credential_source() -> str | None:
    """Where the SDK will find credentials, without making a call.

    Mirrors the SDK's order: key, then token, then a saved `ant` sign-in."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY"
    if os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return "ANTHROPIC_AUTH_TOKEN"
    base = Path(os.getenv("ANTHROPIC_CONFIG_DIR") or Path.home() / ".config" / "anthropic")
    creds = base / "credentials"
    try:
        if creds.is_dir() and any(creds.glob("*.json")):
            return "ant auth login"
    except OSError:
        pass
    return None


def available(config: Config) -> bool:
    return bool(config.claude_api_enabled) and sdk_installed() and credential_source() is not None


def _require(config: Config) -> None:
    if not config.claude_api_enabled:
        raise ClaudeApiUnavailable(
            "The claude-api/ provider is off. Turn it on with "
            "`llm-sidecar config claude-api on`."
        )
    if not sdk_installed():
        raise ClaudeApiUnavailable(
            "The Anthropic SDK is not installed. pip install 'llm-sidecar[claude-api]'."
        )
    if credential_source() is None:
        raise ClaudeApiAuthError(
            "No Anthropic API credentials. Run `ant auth login`, or set ANTHROPIC_API_KEY."
        )


_clients: dict[str, object] = {}


def _client(config: Config, asynchronous: bool = False):
    import anthropic
    key = "async" if asynchronous else "sync"
    if key not in _clients:
        cls = anthropic.AsyncAnthropic if asynchronous else anthropic.Anthropic
        _clients[key] = cls(timeout=config.request_timeout, max_retries=2)
    return _clients[key]


# ── model names ──────────────────────────────────────────────────────────

_resolved: dict[str, tuple[str, float]] = {}
RESOLVE_TTL = 24 * 60 * 60


def resolve(name: str, config: Config) -> str:
    """`haiku` -> the newest Haiku the account can use. Anything else is
    taken as an exact id and passed through."""
    if name not in FALLBACK_IDS:
        return name
    hit = _resolved.get(name)
    if hit and time.time() - hit[1] < RESOLVE_TTL:
        return hit[0]
    try:
        newest = None
        for m in _client(config).models.list():
            if f"-{name}-" in f"{m.id}-" and (newest is None or m.created_at > newest.created_at):
                newest = m
        if newest is not None:
            _resolved[name] = (newest.id, time.time())
            return newest.id
    except Exception as e:
        logger.debug(f"model list failed, using the built-in id for {name}: {e}")
    return FALLBACK_IDS[name]


def models(config: Config) -> list[ModelInfo]:
    """Catalogue entries, or [] when the provider cannot be used. No call is
    made: the aliases resolve when used."""
    if not available(config):
        return []
    return [ModelInfo(id=f"{PREFIX}{alias}", name=label, context_length=1_000_000)
            for alias, _tier, label in FAMILIES]


def tier_default(tier: str) -> str | None:
    for alias, alias_tier, _ in FAMILIES:
        if alias_tier == tier:
            return f"{PREFIX}{alias}"
    return None


# ── requests ─────────────────────────────────────────────────────────────


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def to_wire(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """OpenAI-shaped messages to (system, messages) for the Messages API.

    System turns are lifted out and joined, since the API takes the system
    prompt as its own field. Anything that is not an assistant turn is sent
    as a user turn. A conversation that starts with the assistant gets an
    empty-looking user turn in front, which the API requires."""
    system = "\n\n".join(_text(m.get("content")) for m in messages
                         if m.get("role") == "system" and _text(m.get("content")))
    out = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        out.append({"role": "assistant" if role == "assistant" else "user",
                    "content": _text(m.get("content"))})
    if not out or out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": "(continue)"})
    return system or None, out


def _params(config: Config, wire_model: str, messages: list[dict], max_tokens: int,
            response_format: dict | None) -> dict:
    system, msgs = to_wire(messages)
    params: dict = {
        "model": wire_model,
        "messages": msgs,
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
    }
    if system:
        params["system"] = system
    output_config: dict = {"effort": config.claude_api_effort or "low"}
    schema = ((response_format or {}).get("json_schema") or {}).get("schema")
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    params["output_config"] = output_config
    off = THINKING_OFF.get(wire_model)
    if off:
        params["thinking"] = off
        params["max_tokens"] = max_tokens
    else:
        params["max_tokens"] = max_tokens + THINKING_ROOM
    return params


def _translate(e: Exception, config: Config, model: str) -> Exception:
    """SDK errors to this package's, marking the period exhausted when the
    API says the credit is gone."""
    import anthropic

    text = str(e)
    low = text.lower()
    if "credit balance" in low or isinstance(e, getattr(anthropic, "PaymentRequiredError", ())):
        _mark_exhausted(config, text)
        return ClaudeApiBudgetError(
            f"{model}: the credit balance is too low. Stopped using claude-api/ until "
            f"{next_reset(config).isoformat()}; buy credit in the Console if you want it sooner."
        )
    if isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                      getattr(anthropic, "CredentialsError", ()))):
        return ClaudeApiAuthError(
            f"{model}: credentials rejected ({text[:160]}). Run `ant auth login` again."
        )
    return ClaudeApiError(f"{model}: {text[:300]}")


def _finish(config: Config, model: str, msg, started: float) -> Completion:
    if msg.stop_reason == "refusal":
        raise ClaudeApiError(f"{model}: the model declined this request.")
    served = getattr(msg, "model", None) or model
    cost = cost_of(served, msg.usage)
    _add_spend(config, cost)
    u = msg.usage
    prompt = int(u.input_tokens or 0) + int(getattr(u, "cache_read_input_tokens", 0) or 0) \
        + int(getattr(u, "cache_creation_input_tokens", 0) or 0)
    completion = int(u.output_tokens or 0)
    return Completion(
        text="".join(b.text for b in msg.content if getattr(b, "type", "") == "text"),
        model=model,
        usage=Usage(prompt_tokens=prompt, completion_tokens=completion,
                    total_tokens=prompt + completion, cost_usd=cost),
        latency_s=round(time.time() - started, 3),
    )


def complete(messages: list[dict], model: str, config: Config, max_tokens: int = 1000,
             temperature: float | None = None, response_format: dict | None = None) -> Completion:
    """One completion. `temperature` is accepted and ignored: the current API
    version rejects sampling parameters on every model."""
    _require(config)
    alias = model[len(PREFIX):] if model.startswith(PREFIX) else model
    if not alias:
        raise ClaudeApiError("No model named: expected claude-api/<model>, e.g. claude-api/haiku.")
    wire = resolve(alias, config)
    _guard(config, wire, messages, max_tokens)
    import anthropic

    started = time.time()
    try:
        msg = _client(config).beta.messages.create(
            **_params(config, wire, messages, max_tokens, response_format))
    except anthropic.AnthropicError as e:
        raise _translate(e, config, f"{PREFIX}{alias}") from e
    return _finish(config, f"{PREFIX}{alias}", msg, started)


async def stream(messages: list[dict], model: str, config: Config, max_tokens: int = 1000,
                 temperature: float | None = None) -> AsyncIterator[dict]:
    """Yield {"type": "token"} events, then one {"type": "usage"}."""
    _require(config)
    alias = model[len(PREFIX):] if model.startswith(PREFIX) else model
    wire = resolve(alias, config)
    _guard(config, wire, messages, max_tokens)
    import anthropic

    started = time.time()
    try:
        async with _client(config, asynchronous=True).beta.messages.stream(
                **_params(config, wire, messages, max_tokens, None)) as s:
            async for text in s.text_stream:
                if text:
                    yield {"type": "token", "text": text}
            msg = await s.get_final_message()
    except anthropic.AnthropicError as e:
        raise _translate(e, config, f"{PREFIX}{alias}") from e
    done = _finish(config, f"{PREFIX}{alias}", msg, started)
    yield {"type": "usage", "usage": done.usage}


def check(config: Config) -> dict:
    """Usable or not, and why. Makes no model call."""
    src = credential_source()
    out = {
        "enabled": bool(config.claude_api_enabled),
        "sdk": sdk_installed(),
        "credentials": src,
        "detail": "",
    }
    if not out["sdk"]:
        out["detail"] = "Anthropic SDK not installed: pip install 'llm-sidecar[claude-api]'."
    elif src is None:
        out["detail"] = "No credentials: run `ant auth login`, or set ANTHROPIC_API_KEY."
    elif not config.claude_api_enabled:
        out["detail"] = "Ready, but switched off."
    else:
        out["detail"] = "Ready."
    return out
