"""The `claude/` provider — inference through the Claude Code CLI.

Ollama and OpenRouter both speak OpenAI-compatible HTTP, which is what makes
`client._route()` a three-line function. This provider does not: it spawns
`claude -p` and reads a JSON envelope off stdout. That asymmetry is the whole
cost of the feature, and it buys one thing nothing else here offers — a Claude
subscription the user already pays for, with no per-token bill.

Model ids are `claude/<model>`, where <model> is anything `--model` accepts:
an alias that always points at the current release (`sonnet`, `opus`,
`haiku`), or a pinned name (`claude-opus-5`). Aliases are preferred for the
same reason the picker ranks families rather than exact ids — exact ids rot.

Three things make the subprocess safe to treat as a text generator:

  --tools ""              no Bash, no file access, no web. A completion
                          backend that can read ~/.ssh is not a completion
                          backend.
  --safe-mode             ignores CLAUDE.md, skills, plugins, hooks, MCP
                          servers. Without it, output depends on whichever
                          repo the daemon happens to have been started from.
  --no-session-persistence nothing written to ~/.claude. These are stateless
                          calls, not a conversation the user wants to resume.

Deliberately NOT `--bare`, tempting as it looks: it forces ANTHROPIC_API_KEY
auth and never reads OAuth, which would bill the API for every call and
defeat the entire point.

Two parameters cannot be honoured. The CLI exposes no flag for `max_tokens`
or `temperature`, so both are ignored rather than quietly approximated — see
IGNORED_PARAMS. Everything else maps cleanly, and `--json-schema` maps
*better* than Ollama's shim: it is enforced at decode time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from typing import AsyncIterator

from .config import CACHE_DIR, Config
from .types import Completion, ModelInfo, SidecarError, Usage

logger = logging.getLogger("llm_sidecar.claude_cli")

PREFIX = "claude/"

# Parameters the CLI has no flag for. Named so the docs, the dashboard and
# the warning log all say the same thing.
IGNORED_PARAMS = ("max_tokens", "temperature")

# Sent whenever the caller supplies no system message, and the single
# highest-value line in this module.
#
# --system-prompt *replaces* Claude Code's own prompt. Omit it and every call
# carries the full coding-agent framing: measured at 2480 prompt tokens for
# "Say OK." against 2 tokens with this default in place. That is roughly a
# thousandfold overhead on short calls, charged to the user's allowance, and
# it also told a general-purpose completion it was a coding agent.
DEFAULT_SYSTEM = "You are a helpful assistant. Answer directly and concisely."

# Aliases that track the current release, so this list cannot rot the way an
# exact-id list does. Anything else the user names is passed through
# untouched — a model released after this version still works.
ALIASES: tuple[tuple[str, str, str], ...] = (
    # (alias, tier it suits, human label)
    ("haiku", "fast", "Claude Haiku (subscription)"),
    ("sonnet", "balanced", "Claude Sonnet (subscription)"),
    ("opus", "powerful", "Claude Opus (subscription)"),
)

# Env vars that would send the call somewhere other than the user's
# subscription. Stripped from the child unless the user opts out — the
# failure they would otherwise hit is an API bill for calls they believed
# their subscription covered, which is the kind of mistake you discover at
# the end of the month.
REDIRECT_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
)


class ClaudeCliError(SidecarError):
    """The CLI ran and failed."""


class ClaudeCliUnavailable(ClaudeCliError):
    """No `claude` binary, or the provider is switched off."""


class ClaudeCliAuthError(ClaudeCliError):
    """Not signed in. Needs `claude auth` — retrying will not help."""


class ClaudeCliLimitError(ClaudeCliError):
    """Subscription allowance exhausted. Rotate to another provider.

    Distinct from an HTTP 429 in one way that matters: there is no
    Retry-After, and the window is hours or days rather than seconds. Backing
    off is the wrong move; using a different model is the right one."""


# Where `claude` actually lives when PATH does not say.
#
# This is not belt-and-braces: launchd hands a service a minimal PATH
# (/usr/bin:/bin:/usr/sbin:/sbin), so a CLI installed by npm or the native
# installer is invisible to the daemon even though it works perfectly in the
# user's shell. Found by installing the service and watching claude/ requests
# fail on a machine where `claude --version` was fine.
WELL_KNOWN = (
    "~/.claude/local/claude",      # native installer
    "~/.npm-global/bin/claude",    # npm with a user prefix
    "~/.local/bin/claude",
    "/opt/homebrew/bin/claude",
    "/usr/local/bin/claude",
)


def _usable(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def binary(config: Config) -> str | None:
    """Path to the CLI, or None.

    An explicit config path wins, so a machine with several installs can say
    which one. Otherwise PATH, then the well-known locations — see
    WELL_KNOWN for why that last step is load-bearing rather than tidy."""
    if config.claude_cli_path:
        p = os.path.expanduser(config.claude_cli_path)
        return p if _usable(p) else None

    found = shutil.which("claude")
    if found:
        return found

    for candidate in WELL_KNOWN:
        p = os.path.expanduser(candidate)
        if _usable(p):
            logger.debug("claude not on PATH; using %s", p)
            return p
    return None


def available(config: Config) -> bool:
    """Whether `claude/` ids can be routed at all.

    Enablement is checked first and separately: the provider is off by
    default even on a machine that has the CLI, because it spends an
    allowance the user cannot see from here."""
    return bool(config.claude_cli_enabled) and binary(config) is not None


def _workdir() -> str:
    """An empty directory to run in.

    `--safe-mode` already suppresses CLAUDE.md discovery, but the working
    directory still leaks into trust handling and anything that inspects git
    state. An empty, stable, owned-by-us directory keeps a completion from
    depending on where the daemon was started."""
    d = CACHE_DIR / "claude-cli-cwd"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _child_env(config: Config) -> dict:
    env = dict(os.environ)
    if config.claude_cli_use_subscription:
        for name in REDIRECT_ENV:
            env.pop(name, None)
    return env


def _schema_from(response_format: dict | None) -> str | None:
    """Pull the bare JSON schema out of the OpenAI response_format envelope.

    The rest of this package speaks response_format because that is what two
    HTTP providers accept; the CLI wants the schema on its own."""
    if not response_format:
        return None
    if response_format.get("type") == "json_object":
        # No schema, just "must be JSON". The CLI has no equivalent flag, and
        # ops.parse_json_response already salvages fenced or prefixed output.
        return None
    schema = (response_format.get("json_schema") or {}).get("schema")
    return json.dumps(schema) if schema else None


def split_messages(messages: list[dict]) -> tuple[str | None, str]:
    """Turn a message list into (system_prompt, stdin_prompt).

    The CLI takes one prompt, not a conversation, so a multi-turn exchange is
    rendered as a labelled transcript. That is lossy — the model sees prior
    turns as quoted text rather than as its own output — and it is the
    honest version of lossy: the alternative was pretending turn boundaries
    survived when they had not.

    The single-user-message case, which is almost all traffic, is exact."""
    system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    system = "\n\n".join(p for p in system_parts if p) or None

    if len(rest) == 1:
        return system, rest[0].get("content", "")

    lines = []
    for m in rest:
        label = "Assistant" if m.get("role") == "assistant" else "User"
        lines.append(f"{label}: {m.get('content', '')}")
    lines.append("Assistant:")
    return system, "\n\n".join(lines)


def _argv(
    config: Config,
    model: str,
    system: str | None,
    schema: str | None,
    stream: bool = False,
) -> list[str]:
    exe = binary(config)
    if exe is None:
        raise ClaudeCliUnavailable(
            "No `claude` binary found. Install Claude Code, or set "
            "claude_cli_path to its location."
        )
    argv = [
        exe, "-p",
        "--model", model,
        # Order matters to nobody but a reader: capability removal first.
        "--tools", "",
        "--safe-mode",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--permission-prompts", "none",
        # stream-json even for a blocking call: it is the only format that
        # carries the subscription's usage windows (rate_limit_event) and
        # which credential the call used (apiKeySource), and the final
        # `result` line is the same envelope `json` prints.
        "--output-format", "stream-json", "--verbose",
    ]
    if stream:
        argv += ["--include-partial-messages"]
    if system:
        # Via argv, because there is no --system-prompt-file. argv is visible
        # in `ps`, so the *prompt* goes on stdin and only the system text is
        # exposed; callers putting secrets in a system prompt should know.
        argv += ["--system-prompt", system]
    if schema:
        argv += ["--json-schema", schema]
    return argv


def _classify(message: str) -> type[ClaudeCliError]:
    low = message.lower()
    if any(w in low for w in ("authenticate", "oauth", "not logged in", "log in", "/login")):
        return ClaudeCliAuthError
    if any(w in low for w in ("usage limit", "rate limit", "quota", "limit reached",
                              "out of credit", "too many requests")):
        return ClaudeCliLimitError
    return ClaudeCliError


def _raise_envelope(data: dict, model: str) -> None:
    """Fail on the CLI's own error reporting.

    `claude -p --output-format json` reports failure *inside* a successful
    exit: a well-formed envelope with is_error true and the reason in
    `result`. Checking only the exit code turned an expired login into an
    empty completion recorded as a success — the same trap UpstreamError
    exists for on the HTTP side."""
    if not data.get("is_error"):
        return
    detail = str(data.get("result") or data.get("terminal_reason") or "unknown error")
    raise _classify(detail)(f"claude/{model}: {detail}")


def _observe(event: dict, meta: dict) -> None:
    """Note what a stream event says about the account, not the answer.

    The usage windows go to accounts, so the allowances report can show
    them without spending a call of its own."""
    kind = event.get("type")
    if kind == "rate_limit_event" and isinstance(event.get("rate_limit_info"), dict):
        from . import accounts
        accounts.note_claude_window(event["rate_limit_info"])
    elif kind == "system" and event.get("subtype") == "init":
        meta["api_key_source"] = event.get("apiKeySource")


def _read_output(raw: str, model: str) -> tuple[dict, dict]:
    """(final envelope, meta) from the CLI's stdout, one JSON object a line."""
    final, meta = None, {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "result" or ("type" not in event and
                                              ("result" in event or "is_error" in event)):
            final = event
        else:
            _observe(event, meta)
    if final is None:
        raise ClaudeCliError(f"claude/{model}: output was not JSON ({raw[:200]!r})")
    return final, meta


def _usage_from(data: dict, config: Config, api_key_source: str | None = None) -> Usage:
    """Token counts, and the one genuinely subtle field in this module.

    The CLI reports `total_cost_usd` as the metered-API equivalent even on a
    subscription — a live haiku call came back at $0.00057 with OAuth auth.
    Passing that through would make `llm-sidecar usage` overstate the user's
    real bill by whatever their subscription covered, which is the opposite
    of what a spend report is for.

    We can tell the two apart with certainty, because we control the child's
    environment: with claude_cli_use_subscription on, ANTHROPIC_API_KEY and
    friends are stripped, so the call *cannot* have been billed to an API
    key. Off, the user opted into metered auth and the figure is a real
    charge."""
    u = data.get("usage") or {}
    prompt = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)
    completion = u.get("output_tokens") or 0
    reported = float(data.get("total_cost_usd") or 0.0)
    # The CLI says which credential it used. Anything but "none" (the
    # subscription sign-in) is metered, whatever the setting intended: an
    # `ant auth login` profile, say, which env-stripping does not remove.
    metered = (not config.claude_cli_use_subscription) or (
        api_key_source not in (None, "none"))
    billed = reported if metered else 0.0
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cost_usd=billed,
        notional_cost_usd=reported,
    )


def _refuse_if_out() -> None:
    """Fail at once when the last reading says the subscription is out and
    has not reset, rather than starting a process to be told so. A sidecar
    picking a model rotates on this straight away."""
    from . import accounts
    blocked = accounts.claude_window_blocked()
    if blocked:
        raise ClaudeCliLimitError(blocked)


def _warn_ignored(max_tokens: int | None, temperature: float | None) -> None:
    ignored = []
    if max_tokens is not None:
        ignored.append(f"max_tokens={max_tokens}")
    if temperature is not None:
        ignored.append(f"temperature={temperature}")
    if ignored:
        logger.debug("claude/ has no flag for %s; ignored", ", ".join(ignored))


def complete(
    messages: list[dict],
    model: str,
    config: Config,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    response_format: dict | None = None,
    timeout: float | None = None,
) -> Completion:
    """One completion via `claude -p`. Blocking, one process per call."""
    if not config.claude_cli_enabled:
        raise ClaudeCliUnavailable(
            "The claude/ provider is off. Enable it with "
            "`llm-sidecar config --claude-cli on`."
        )
    wire = model[len(PREFIX):] if model.startswith(PREFIX) else model
    if not wire:
        raise ClaudeCliError("No model named: expected claude/<model>, e.g. claude/sonnet.")

    _warn_ignored(max_tokens, temperature)
    _refuse_if_out()
    system, prompt = split_messages(messages)
    argv = _argv(config, wire, system or DEFAULT_SYSTEM,
                 _schema_from(response_format))

    started = time.time()
    raw = _run(argv, prompt, config, timeout or config.claude_cli_timeout)
    data, meta = _read_output(raw, wire)

    _raise_envelope(data, wire)
    return Completion(
        text=data.get("result") or "",
        model=f"{PREFIX}{wire}",
        usage=_usage_from(data, config, meta.get("api_key_source")),
        latency_s=round(time.time() - started, 3),
    )


def _run(argv: list[str], prompt: str, config: Config, timeout: float) -> str:
    """Run the CLI with the prompt on stdin, killing the whole group on timeout.

    `subprocess.run(timeout=...)` kills the direct child only. The CLI is a
    node process that spawns its own children, so a plain kill can leave
    those running and holding the model. start_new_session plus killpg takes
    the group down together."""
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=_workdir(),
            env=_child_env(config),
            start_new_session=True,
        )
    except OSError as e:
        raise ClaudeCliUnavailable(f"Could not run {argv[0]}: {e}") from e

    try:
        out, err = proc.communicate(input=prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        proc.communicate()
        raise ClaudeCliError(
            f"claude -p exceeded {timeout:g}s. Raise claude_cli_timeout, or use "
            "a smaller model."
        ) from None

    if proc.returncode != 0 and not out.strip():
        detail = (err or "").strip()[:300] or f"exit {proc.returncode}"
        raise _classify(detail)(f"claude -p failed: {detail}")
    return out


def _kill_group(proc: subprocess.Popen) -> None:
    import signal
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        proc.kill()


# ── streaming ─────────────────────────────────────────────────────────────
#
# The stream-json envelope is read tolerantly on purpose. Partial deltas
# arrive as content_block_delta events, but the exact nesting has moved
# between CLI versions, and a parser that recognises only one shape yields
# silence on the next one. So: take text from whatever shape carries it, and
# if no deltas were seen at all, fall back to emitting the final result in
# one chunk. Degrading to "unstreamed" is a worse experience; degrading to
# "nothing" is a bug.


def _delta_text(event: dict) -> str:
    """Text from an incremental delta, or ""."""
    if event.get("type") != "stream_event":
        return ""
    inner = event.get("event") or {}
    if inner.get("type") != "content_block_delta":
        return ""
    delta = inner.get("delta") or {}
    if delta.get("type") in ("text_delta", None):
        return delta.get("text") or ""
    return ""


def _message_text(event: dict) -> str:
    """Text from a whole-message event, or "".

    Kept strictly separate from _delta_text, because the CLI emits *both*:
    the deltas as they arrive, and then the finished assistant message
    carrying the same text again. Reading text from whichever shape appeared
    duplicated every answer — "12345" came back as "1234512345". Deltas win,
    and this is the fallback for a version that sends no deltas at all.

    Thinking blocks are skipped: a scratchpad is not the answer."""
    if event.get("type") != "assistant":
        return ""
    blocks = ((event.get("message") or {}).get("content")) or []
    return "".join(b.get("text") or "" for b in blocks
                   if isinstance(b, dict) and b.get("type") == "text")


async def stream(
    messages: list[dict],
    model: str,
    config: Config,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    timeout: float | None = None,
) -> AsyncIterator[dict]:
    """Yield {"type": "token"} events, then one {"type": "usage"}."""
    if not config.claude_cli_enabled:
        raise ClaudeCliUnavailable(
            "The claude/ provider is off. Enable it with "
            "`llm-sidecar config --claude-cli on`."
        )
    wire = model[len(PREFIX):] if model.startswith(PREFIX) else model
    _warn_ignored(max_tokens, temperature)
    _refuse_if_out()
    system, prompt = split_messages(messages)
    argv = _argv(config, wire, system or DEFAULT_SYSTEM, None, stream=True)

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=_workdir(),
        env=_child_env(config),
        start_new_session=True,
    )

    emitted = False
    whole: list[str] = []
    final: dict | None = None
    meta: dict = {}
    try:
        proc.stdin.write(prompt.encode())
        await proc.stdin.drain()
        proc.stdin.close()

        while True:
            line = await asyncio.wait_for(
                proc.stdout.readline(),
                timeout=timeout or config.claude_cli_timeout,
            )
            if not line:
                break
            try:
                event = json.loads(line.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue

            if event.get("type") == "result":
                final = event
                continue
            _observe(event, meta)

            text = _delta_text(event)
            if text:
                emitted = True
                yield {"type": "token", "text": text}
                continue
            # Buffered, not yielded: this is the same text the deltas already
            # carried. It is only used if no deltas ever arrived.
            whole.append(_message_text(event))

        if final is None:
            raise ClaudeCliError(f"claude/{wire}: stream ended with no result event")
        _raise_envelope(final, wire)

        # No deltas recognised — hand over the whole answer in one chunk
        # rather than nothing. See the note above.
        if not emitted:
            fallback = "".join(whole) or (final.get("result") or "")
            if fallback:
                yield {"type": "token", "text": fallback}

        yield {"type": "usage", "usage": _usage_from(final, config, meta.get("api_key_source"))}
    finally:
        if proc.returncode is None:
            _kill_group_async(proc)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                pass


def _kill_group_async(proc) -> None:
    import signal
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


# ── introspection ─────────────────────────────────────────────────────────


def models(config: Config) -> list[ModelInfo]:
    """Catalogue entries for the `claude/` aliases, or [] when unavailable.

    Priced at zero, which is true under a subscription and the reason these
    are kept out of the budget-based candidate pool: `free` means "costs no
    money", and these cost an allowance instead. See picker.candidates."""
    if not available(config):
        return []
    return [
        ModelInfo(id=f"{PREFIX}{alias}", name=label, context_length=200_000)
        for alias, _tier, label in ALIASES
    ]


def tier_default(tier: str) -> str | None:
    """The alias that suits a tier, for the dashboard's one-click pinning."""
    for alias, alias_tier, _ in ALIASES:
        if alias_tier == tier:
            return f"{PREFIX}{alias}"
    return None


def check(config: Config) -> dict:
    """Is this provider actually usable? Cheap, and makes no model call.

    Reports version and auth separately because they fail for different
    reasons and need different fixes. Auth is checked with a real but minimal
    call, since nothing short of one can tell a live login from an expired
    one — the CLI reports an expired OAuth session only when it tries to use
    it."""
    exe = binary(config)
    out: dict = {
        "enabled": bool(config.claude_cli_enabled),
        "binary": exe,
        "version": None,
        "authenticated": None,
        "detail": "",
    }
    if exe is None:
        out["detail"] = "No `claude` binary on PATH."
        return out

    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=10)
        out["version"] = (r.stdout or "").strip() or None
    except (OSError, subprocess.SubprocessError) as e:
        out["detail"] = f"Could not run {exe}: {e}"
        return out

    if not config.claude_cli_enabled:
        out["detail"] = "Installed but switched off."
        return out

    try:
        done = complete(
            [{"role": "user", "content": "Reply with OK and nothing else."}],
            f"{PREFIX}haiku", config, timeout=min(60.0, config.claude_cli_timeout),
        )
        out["authenticated"] = bool(done.text.strip())
        out["detail"] = "Ready."
    except ClaudeCliAuthError as e:
        out["authenticated"] = False
        out["detail"] = f"Not signed in — run `claude auth`. ({e})"
    except ClaudeCliError as e:
        out["authenticated"] = False
        out["detail"] = str(e)[:200]
    return out
