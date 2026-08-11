"""Inference — the actual model calls.

Ollama and OpenRouter both speak OpenAI-compatible chat completions, so the
only difference is the base URL and the auth header. That symmetry is what
makes "local or cloud, same call" possible without an abstraction layer.

Rate limits are handled in two places, on purpose:
  - here, by retrying the same model with backoff (transient throttling)
  - in picker.pick(exclude=...), by rotating to a different model (exhausted)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator

import httpx

from .config import Config
from .types import Completion, Usage

logger = logging.getLogger("llm_sidecar.client")

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_PREFIX = "ollama/"
RETRYABLE_STATUS = (429, 503)


def _route(model: str, config: Config) -> tuple[str, str, dict]:
    """(url, wire_model, headers) for a model id."""
    if model.startswith(OLLAMA_PREFIX):
        return (
            f"{config.ollama_host}/v1/chat/completions",
            model[len(OLLAMA_PREFIX):],
            {"Content-Type": "application/json"},
        )
    return (
        OPENROUTER_CHAT_URL,
        model,
        {
            "Authorization": f"Bearer {config.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": config.referer,
            "X-Title": config.app_title,
        },
    )


def _messages(prompt: str, system: str | None) -> list[dict]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def _usage_from(raw: dict, is_local: bool) -> Usage:
    # Local inference has no bill, whatever the response claims.
    return Usage(
        prompt_tokens=raw.get("prompt_tokens", 0),
        completion_tokens=raw.get("completion_tokens", 0),
        total_tokens=raw.get("total_tokens", 0),
        cost_usd=0.0 if is_local else (raw.get("cost") or 0.0),
    )


def complete(
    prompt: str,
    model: str,
    config: Config,
    system: str | None = None,
    max_tokens: int = 1000,
    temperature: float = 0.7,
) -> Completion:
    """One blocking completion, retrying transient throttling with backoff."""
    url, wire_model, headers = _route(model, config)
    is_local = model.startswith(OLLAMA_PREFIX)
    payload = {
        "model": wire_model,
        "messages": _messages(prompt, system),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    delays = config.retry_delays
    for attempt in range(len(delays) + 1):
        try:
            r = httpx.post(url, json=payload, headers=headers, timeout=config.request_timeout)
            r.raise_for_status()
            data = r.json()
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return Completion(
                text=text,
                model=model,
                usage=_usage_from(data.get("usage") or {}, is_local),
            )
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in RETRYABLE_STATUS and attempt < len(delays):
                wait = delays[attempt]
                logger.warning(f"{model} → HTTP {status}, retrying in {wait}s")
                time.sleep(wait)
                continue
            raise
    raise AssertionError("unreachable")  # loop always returns or raises


async def stream(
    prompt: str,
    model: str,
    config: Config,
    system: str | None = None,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[dict]:
    """Yield {"type": "token", "text": ...} then a final {"type": "usage", ...}.

    Retries only before the first token — once tokens are out the door the
    consumer has already seen them, and a retry would duplicate the prefix."""
    url, wire_model, headers = _route(model, config)
    is_local = model.startswith(OLLAMA_PREFIX)
    payload = {
        "model": wire_model,
        "messages": _messages(prompt, system),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        # OpenRouter only reports usage on a streamed response if asked.
        "stream_options": {"include_usage": True},
    }

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=config.request_timeout)
    delays = config.retry_delays

    try:
        for attempt in range(len(delays) + 1):
            emitted = False
            try:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        chunk = line[6:].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            data = json.loads(chunk)
                        except json.JSONDecodeError:
                            continue

                        choices = data.get("choices") or []
                        if choices:
                            delta = (choices[0].get("delta") or {}).get("content")
                            if delta:
                                emitted = True
                                yield {"type": "token", "text": delta}
                        if data.get("usage"):
                            yield {
                                "type": "usage",
                                "usage": _usage_from(data["usage"], is_local),
                            }
                return
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else 0
                if status in RETRYABLE_STATUS and attempt < len(delays) and not emitted:
                    wait = delays[attempt]
                    logger.warning(f"{model} → HTTP {status}, retrying stream in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                raise
    finally:
        if owns_client:
            await client.aclose()
