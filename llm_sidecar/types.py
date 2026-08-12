"""Shared value types. Plain dataclasses — no ORM, no framework."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class Completion:
    """One model response, with a receipt attached."""
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    cached: bool = False

    @property
    def is_local(self) -> bool:
        return self.model.startswith("ollama/")


@dataclass
class ModelInfo:
    """A model in the OpenRouter catalogue, or a locally-installed Ollama model."""
    id: str
    name: str
    context_length: int = 0
    prompt_price_per_million: float = 0.0
    completion_price_per_million: float = 0.0

    @property
    def is_free(self) -> bool:
        return self.prompt_price_per_million == 0.0 and self.completion_price_per_million == 0.0

    @property
    def is_local(self) -> bool:
        return self.id.startswith("ollama/")


@dataclass
class Pick:
    """A model that responded to a live pretest just now."""
    model_id: str
    model_name: str
    attempts: list[dict] = field(default_factory=list)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    published: str = ""
    is_news: bool = False


@dataclass
class Answer:
    """A question answered from retrieved sources.

    `grounded` is the field that matters: False means the sources did not
    contain the answer, and `text` explains what was missing rather than
    guessing. Callers should branch on it instead of just printing text."""
    question: str
    text: str
    grounded: bool = False
    sources: list[str] = field(default_factory=list)
    caveat: str = ""
    model: str = ""


@dataclass
class ClaimVerdict:
    claim: str
    verdict: str  # supported | contradicted | unverified | not_a_claim
    note: str = ""
    sources: list[str] = field(default_factory=list)


class SidecarError(Exception):
    """Base for every error this package raises deliberately."""


class NoWorkingModel(SidecarError):
    """Every candidate model was tried and none responded."""

    def __init__(self, message: str, attempts: list[dict] | None = None):
        super().__init__(message)
        self.attempts = attempts or []
