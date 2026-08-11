"""llm-sidecar — a local capability sidecar for AI tooling.

One process that any tool on the machine can lean on for:
  - inference that routes itself (local Ollama, free cloud, paid cloud) and
    only ever hands back a model verified to be answering right now
  - web search that upgrades from keyless DuckDuckGo to self-hosted SearXNG
    without the caller knowing
  - grounded claim verification with citations

Typical use:

    from llm_sidecar import Sidecar

    sc = Sidecar()
    print(sc.complete("Explain CRDTs in two sentences").text)
    print(sc.verify(["The Eiffel Tower is in Berlin"]))
"""

from __future__ import annotations

from typing import AsyncIterator

from . import catalogue, client, config as config_mod, picker, verify as verify_mod
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

__version__ = "0.1.0"

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
        # Resolved tier -> model id for this process. Picking costs a live
        # pretest call, so we do it once per tier and reuse, rotating only
        # when a model actually fails.
        self._resolved: dict[str, str] = {}
        self._failed: set[str] = set()

    # ── model resolution ──────────────────────────────────────────────────

    def model_for(self, tier: str | None = None, budget: str | None = None) -> str:
        """The model id this tier resolves to — a pin if configured, else a
        freshly verified pick."""
        tier = tier or self.config.default_tier
        pinned = self.config.tier_model(tier)
        if pinned:
            return pinned
        if tier in self._resolved:
            return self._resolved[tier]
        chosen = picker.pick(self.config, budget, exclude=self._failed).model_id
        self._resolved[tier] = chosen
        return chosen

    def pool(self, budget: str | None = None) -> dict[str, Pick]:
        """Three distinct verified models mapped to tiers, for parallel work."""
        picks = picker.pick_pool(self.config, budget, exclude=self._failed)
        self._resolved.update({tier: p.model_id for tier, p in picks.items()})
        return picks

    def _rotate(self, model: str) -> None:
        """Mark a model dead for this process so the next pick skips it."""
        self._failed.add(model)
        self._resolved = {t: m for t, m in self._resolved.items() if m != model}

    # ── inference ─────────────────────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        *,
        tier: str | None = None,
        budget: str | None = None,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> Completion:
        """A completion. Give a `model` to pin one, or a `tier`/`budget` to let
        it resolve.

        On a hard failure from an auto-picked model we rotate to a different
        verified model and try once more — client.complete has already
        exhausted its own backoff by then, so the model itself is the problem."""
        chosen = model or self.model_for(tier, budget)
        try:
            return client.complete(
                prompt, chosen, self.config,
                system=system, max_tokens=max_tokens, temperature=temperature,
            )
        except Exception:
            if model:  # explicitly pinned — the caller meant that model
                raise
            self._rotate(chosen)
            fallback = self.model_for(tier, budget)
            return client.complete(
                prompt, fallback, self.config,
                system=system, max_tokens=max_tokens, temperature=temperature,
            )

    def stream(
        self,
        prompt: str,
        *,
        tier: str | None = None,
        budget: str | None = None,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> AsyncIterator[dict]:
        """Token stream. Yields {"type": "token"|"usage", ...}."""
        chosen = model or self.model_for(tier, budget)
        return client.stream(
            prompt, chosen, self.config,
            system=system, max_tokens=max_tokens, temperature=temperature,
        )

    # ── tools ─────────────────────────────────────────────────────────────

    def search(self, query: str, max_results: int = 5, news: bool = False) -> list[SearchResult]:
        from . import search as search_mod
        return search_mod.search(query, self.config, max_results=max_results, news=news)

    def read_url(self, url: str, max_chars: int = 20000) -> str:
        from . import search as search_mod
        return search_mod.read_url(url, self.config, max_chars=max_chars)

    def verify(self, claims: list[str], model: str | None = None) -> list[ClaimVerdict]:
        return verify_mod.verify_claims(claims, self.config, model=model)

    def extract_claims(self, text: str, model: str | None = None) -> list[str]:
        return verify_mod.extract_claims(text, self.config, model or self.model_for("fast"))

    # ── introspection ─────────────────────────────────────────────────────

    def models(self, refresh: bool = False) -> list[ModelInfo]:
        """Every model we could route to — cloud catalogue plus local Ollama."""
        return (
            catalogue.openrouter_models(self.config, force_refresh=refresh)
            + catalogue.ollama_models(self.config)
        )
