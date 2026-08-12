"""llm-sidecar — a local capability sidecar for AI tooling.

One process that any tool on the machine can lean on for:
  - inference that routes itself (local Ollama, free cloud, paid cloud) and
    only ever hands back a model verified to be answering right now
  - web search that upgrades from keyless DuckDuckGo to self-hosted SearXNG
    without the caller knowing
  - grounded claim verification with citations
  - bounded structured work: summarise, classify, extract

Typical use:

    from llm_sidecar import Sidecar

    sc = Sidecar()
    print(sc.complete("Explain CRDTs in two sentences").text)
    print(sc.verify(["The Eiffel Tower is in Berlin"]))
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Iterable

from . import (
    cache,
    catalogue,
    client,
    config as config_mod,
    hardware,
    ledger,
    ops,
    picker,
    verify as verify_mod,
)
from .config import Config
from .types import (
    ClaimVerdict,
    Completion,
    ModelInfo,
    NoWorkingModel,
    Pick,
    SearchResult,
    SidecarError,
    Usage,
)

logger = logging.getLogger("llm_sidecar")

__version__ = "0.3.0"

__all__ = [
    "Sidecar",
    "Config",
    "Completion",
    "Usage",
    "ModelInfo",
    "Pick",
    "SearchResult",
    "ClaimVerdict",
    "SidecarError",
    "NoWorkingModel",
    "__version__",
]


class Sidecar:
    """The facade. Holds config and a resolved-model cache; everything else
    lives in the modules and stays independently importable."""

    def __init__(self, config: Config | None = None, **overrides):
        self.config = config or config_mod.load(**overrides)
        # Resolved "tier:budget" -> (model id, when it was verified).
        #
        # Budget is part of the key on purpose: a request for the `best`
        # budget must not be served the model a previous `free` request
        # resolved to. Getting a cheap model when you asked for an expensive
        # one is a silent wrong answer, which is worse than a slow one.
        #
        # The timestamp is there because a pretest proves a model worked
        # *then*. Free tiers do not stay working, so entries expire.
        self._resolved: dict[str, tuple[str, float]] = {}
        self._failed: set[str] = set()
        # The daemon shares one Sidecar across FastAPI's threadpool, so this
        # state is touched concurrently. Only the bookkeeping is guarded —
        # the pick itself happens outside the lock, because it makes network
        # calls and holding a lock across those would serialise every request
        # behind the slowest cold start.
        self._lock = threading.Lock()

    # ── model resolution ──────────────────────────────────────────────────

    @property
    def resolved(self) -> dict[str, str]:
        """What each tier:budget combination has resolved to so far."""
        with self._lock:
            return {k: v[0] for k, v in self._resolved.items()}

    def model_for(self, tier: str | None = None, budget: str | None = None) -> str:
        """The model id this tier resolves to — a pin if configured, else a
        freshly verified pick."""
        tier = tier or self.config.default_tier
        budget = budget or self.config.default_budget
        pinned = self.config.tier_model(tier)
        if pinned:
            return pinned

        key = f"{tier}:{budget}"
        with self._lock:
            hit = self._resolved.get(key)
            if hit and (time.time() - hit[1]) < self.config.resolution_ttl_seconds:
                return hit[0]
            excluded = set(self._failed)

        chosen = picker.pick(self.config, budget, exclude=excluded).model_id

        with self._lock:
            # Another thread may have resolved this key while we were probing.
            # Defer to it rather than overwriting: both models were verified,
            # and churning the choice would scatter requests across models for
            # no benefit.
            existing = self._resolved.get(key)
            if existing and (time.time() - existing[1]) < self.config.resolution_ttl_seconds:
                return existing[0]
            self._resolved[key] = (chosen, time.time())
        return chosen

    def pool(self, budget: str | None = None) -> dict[str, Pick]:
        """Three distinct verified models mapped to tiers, for parallel work."""
        budget = budget or self.config.default_budget
        with self._lock:
            excluded = set(self._failed)
        picks = picker.pick_pool(self.config, budget, exclude=excluded)
        now = time.time()
        with self._lock:
            self._resolved.update({f"{t}:{budget}": (p.model_id, now) for t, p in picks.items()})
        return picks

    def _rotate(self, model: str) -> None:
        """Mark a model dead for this process so the next pick skips it."""
        with self._lock:
            self._failed.add(model)
            self._resolved = {k: v for k, v in self._resolved.items() if v[0] != model}

    # ── inference ─────────────────────────────────────────────────────────

    def complete(
        self,
        prompt: str | None = None,
        *,
        messages: Iterable[dict] | None = None,
        tier: str | None = None,
        budget: str | None = None,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        operation: str = "complete",
    ) -> Completion:
        """A completion. Pass `messages` for a real conversation, or
        `prompt`/`system` for the one-shot case. Give a `model` to pin one, or
        a `tier`/`budget` to let it resolve.

        On a hard failure from an auto-picked model we rotate to a different
        verified model and try once more — the client has already exhausted
        its own backoff by then, so the model itself is the problem."""
        msgs = client.build_messages(prompt, system, messages)

        chosen = model or self.model_for(tier, budget)
        hit = cache.get_completion(self.config, msgs, chosen, max_tokens, temperature)
        if hit is not None:
            done = Completion(
                text=hit["text"], model=hit["model"],
                usage=Usage(**hit["usage"]), cached=True,
            )
            self._record(done, operation)
            return done

        try:
            done = client.complete(msgs, chosen, self.config,
                                   max_tokens=max_tokens, temperature=temperature)
        except Exception:
            if model:  # explicitly pinned — the caller meant that model
                raise
            self._rotate(chosen)
            fallback = self.model_for(tier, budget)
            done = client.complete(msgs, fallback, self.config,
                                   max_tokens=max_tokens, temperature=temperature)

        cache.put_completion(self.config, msgs, done.model, max_tokens, temperature, {
            "text": done.text, "model": done.model, "usage": done.usage.__dict__,
        })
        self._record(done, operation)
        return done

    def stream(
        self,
        prompt: str | None = None,
        *,
        messages: Iterable[dict] | None = None,
        tier: str | None = None,
        budget: str | None = None,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> AsyncIterator[dict]:
        """Token stream. Yields {"type": "token"|"usage", ...}."""
        msgs = client.build_messages(prompt, system, messages)
        chosen = model or self.model_for(tier, budget)
        return client.stream(msgs, chosen, self.config,
                             max_tokens=max_tokens, temperature=temperature)

    async def acomplete(self, prompt: str | None = None, **kwargs) -> Completion:
        """`complete` off the event loop.

        The underlying call is blocking httpx, so this hands it to a thread
        rather than pretending to be async. That's still the right primitive:
        it means an async caller can gather() a hundred completions without
        blocking the loop, which is the case that actually matters."""
        return await asyncio.to_thread(self.complete, prompt, **kwargs)

    def complete_many(
        self,
        prompts: list[str],
        *,
        max_workers: int | None = None,
        **kwargs,
    ) -> list[Completion]:
        """Several completions concurrently, results in input order.

        A failed prompt yields a Completion with empty text rather than
        sinking the batch — with twenty prompts in flight you want the
        nineteen that worked."""
        if not prompts:
            return []
        workers = max_workers or min(len(prompts), self.config.max_completion_workers)

        def one(p: str) -> Completion:
            try:
                return self.complete(p, **kwargs)
            except Exception as e:
                logger.warning(f"complete_many: {str(e)[:120]}")
                return Completion(text="", model="", usage=Usage())

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, prompts))

    def _record(self, done: Completion, operation: str) -> None:
        if self.config.ledger_enabled:
            ledger.record(
                done.model, done.usage.prompt_tokens, done.usage.completion_tokens,
                done.usage.cost_usd, operation=operation, cached=done.cached,
                latency_s=done.latency_s,
            )

    # ── tools ─────────────────────────────────────────────────────────────

    def search(self, query: str, max_results: int = 5, news: bool = False) -> list[SearchResult]:
        from . import search as search_mod
        return search_mod.search(query, self.config, max_results=max_results, news=news)

    def read_url(self, url: str, max_chars: int = 20000) -> str:
        from . import search as search_mod
        return search_mod.read_url(url, self.config, max_chars=max_chars)

    def verify(self, claims: list[str], model: str | None = None) -> list[ClaimVerdict]:
        return verify_mod.verify_claims(claims, self.config, model=model, sidecar=self)

    def extract_claims(self, text: str, model: str | None = None) -> list[str]:
        return verify_mod.extract_claims(text, self, model=model)

    def fact_check(self, text: str) -> list[ClaimVerdict]:
        """Extract the claims from a document and verify each one."""
        return self.verify(self.extract_claims(text))

    # ── structured operations ─────────────────────────────────────────────

    def summarise(self, text: str, style: str = "brief", focus: str = "") -> str:
        return ops.summarise(self, text, style=style, focus=focus)

    # Same function, for anyone who spells it the other way.
    summarize = summarise

    def classify(self, items: list[str], labels: list[str], multi: bool = False) -> list[dict]:
        return ops.classify(self, items, labels, multi=multi)

    def extract(self, text: str, fields: dict[str, str]) -> dict:
        return ops.extract(self, text, fields)

    # ── introspection ─────────────────────────────────────────────────────

    def models(self, refresh: bool = False) -> list[ModelInfo]:
        """Every model we could route to — cloud catalogue plus local Ollama."""
        return (
            catalogue.openrouter_models(self.config, force_refresh=refresh)
            + catalogue.ollama_models(self.config)
        )

    def local_models(self, context_tokens: int = 8000) -> list[dict]:
        """Installed Ollama models scored against this machine's memory."""
        return hardware.advise(self.config, context_tokens)

    def usage(self, days: int | None = None) -> dict:
        """Spend and call counts from the ledger."""
        return ledger.summary(days)

    def status(self) -> dict:
        """Everything a caller might want to know about this sidecar."""
        from .search import resolve_provider

        try:
            provider = resolve_provider(self.config).__name__.rsplit(".", 1)[-1]
        except Exception as e:
            provider = f"unavailable ({e})"

        local = self.local_models()
        return {
            "version": __version__,
            "cloud_configured": self.config.has_cloud,
            "default_budget": self.config.default_budget,
            "search_provider": provider,
            "hardware": hardware.probe(),
            "local_models": [
                {"id": m["id"], "verdict": m["verdict"], "size_gib": m["size_gib"]}
                for m in local
            ],
            "resolved_tiers": self.resolved,
            "cache": cache.stats(),
            "usage_30d": ledger.summary(30),
        }
