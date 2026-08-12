"""Configuration — a plain dataclass, loaded from a JSON file plus env vars.

Deliberately has no database and no framework dependency: the whole point of
this package is that any tool can import it without inheriting someone else's
persistence layer. Callers that already have their own config (Agora, for
instance) can construct a Config directly and never touch the file at all.

Precedence, lowest to highest:
  built-in defaults  <  ~/.config/llm-sidecar/config.json  <  environment
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

CONFIG_DIR = Path(os.getenv("LLM_SIDECAR_HOME", Path.home() / ".config" / "llm-sidecar"))
CACHE_DIR = Path(os.getenv("LLM_SIDECAR_CACHE", Path.home() / ".cache" / "llm-sidecar"))
CONFIG_FILE = CONFIG_DIR / "config.json"

# How long the OpenRouter model catalogue stays fresh before we refetch.
CATALOGUE_TTL_SECONDS = 24 * 60 * 60


@dataclass
class Config:
    # ── inference ─────────────────────────────────────────────────────────
    openrouter_api_key: str | None = None
    ollama_host: str = "http://localhost:11434"

    # Explicit per-tier model pins. When a tier is unset, the picker resolves
    # it live against `default_budget` instead.
    models: dict[str, str] = field(default_factory=dict)  # {fast,balanced,powerful}
    default_budget: str = "free"  # free | cheap | best
    default_tier: str = "balanced"

    # ── search ────────────────────────────────────────────────────────────
    # "auto" probes for a local SearXNG and falls back to DuckDuckGo, so a
    # fresh install works with no setup but transparently upgrades if the
    # user runs an instance.
    search_provider: str = "auto"  # auto | ddg | searxng
    searxng_url: str = "http://localhost:8888"

    # ── cache ─────────────────────────────────────────────────────────────
    # Only deterministic (temperature 0) completions are cached — see cache.py.
    cache_enabled: bool = True
    cache_ttl_seconds: float = 7 * 24 * 60 * 60
    # Searches expire faster: the point of a live search is that it's live.
    search_cache_ttl_seconds: float = 60 * 60
    # Fetched pages sit between the two: more stable than a search ranking,
    # less stable than a model's answer to a fixed prompt.
    page_cache_ttl_seconds: float = 6 * 60 * 60
    # Size budget for the whole cache directory, trimmed oldest-first.
    cache_max_bytes: int = 256 * 1024 * 1024

    # ── ledger ────────────────────────────────────────────────────────────
    ledger_enabled: bool = True

    # ── concurrency ───────────────────────────────────────────────────────
    # Parallel searches when gathering evidence for many claims at once.
    # Higher finishes sooner but is likelier to trip a provider's rate limit;
    # DuckDuckGo in particular does not enjoy this.
    max_search_workers: int = 4
    # Judge batches run concurrently against one model; keep this modest.
    max_judge_workers: int = 3
    # Concurrency ceiling for Sidecar.complete_many().
    max_completion_workers: int = 5

    # ── resolution ────────────────────────────────────────────────────────
    # How long a tier keeps its resolved model before being re-verified. The
    # pretest proved the model worked *then*; free tiers do not stay working.
    resolution_ttl_seconds: float = 15 * 60

    # ── daemon ────────────────────────────────────────────────────────────
    # Loopback by default and deliberately so: this process holds an API key
    # and will spend real money on request. Binding it to 0.0.0.0 hands that
    # to anything on the network.
    daemon_host: str = "127.0.0.1"
    daemon_port: int = 4001
    # When set, requests must present it as `Authorization: Bearer <token>`.
    # Off by default — on a loopback socket the OS is already the boundary —
    # but worth setting on a shared machine.
    daemon_token: str | None = None

    # ── network ───────────────────────────────────────────────────────────
    request_timeout: float = 300.0
    pretest_timeout: float = 15.0
    # Backoff schedule for 429/503. Length also caps the retry count.
    retry_delays: tuple[float, ...] = (5.0, 20.0)

    # ── identification (OpenRouter attributes calls to these) ─────────────
    referer: str = "https://github.com/awaistechnologist/llm-sidecar"
    app_title: str = "llm-sidecar"

    @property
    def has_cloud(self) -> bool:
        return bool(self.openrouter_api_key)

    def tier_model(self, tier: str) -> str | None:
        """The pinned model for a tier, or None if it should be auto-picked."""
        return self.models.get(tier) or None


def _from_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        # A broken config file should not stop the process from starting —
        # env vars and defaults are enough to run.
        return {}


def _from_env() -> dict:
    out: dict = {}
    # OPENROUTER_API_KEY (rather than an LLM_SIDECAR_ prefix) so an existing
    # environment that already talks to OpenRouter works unchanged.
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        out["openrouter_api_key"] = key
    if os.getenv("OLLAMA_HOST"):
        out["ollama_host"] = os.environ["OLLAMA_HOST"]
    if os.getenv("LLM_SIDECAR_BUDGET"):
        out["default_budget"] = os.environ["LLM_SIDECAR_BUDGET"]
    if os.getenv("LLM_SIDECAR_SEARCH_PROVIDER"):
        out["search_provider"] = os.environ["LLM_SIDECAR_SEARCH_PROVIDER"]
    if os.getenv("SEARXNG_URL"):
        out["searxng_url"] = os.environ["SEARXNG_URL"]
    if os.getenv("LLM_SIDECAR_HOST"):
        out["daemon_host"] = os.environ["LLM_SIDECAR_HOST"]
    if os.getenv("LLM_SIDECAR_PORT"):
        try:
            out["daemon_port"] = int(os.environ["LLM_SIDECAR_PORT"])
        except ValueError:
            pass
    if os.getenv("LLM_SIDECAR_TOKEN"):
        out["daemon_token"] = os.environ["LLM_SIDECAR_TOKEN"]
    if os.getenv("LLM_SIDECAR_NO_CACHE"):
        out["cache_enabled"] = False
    if os.getenv("LLM_SIDECAR_NO_LEDGER"):
        out["ledger_enabled"] = False

    # Per-tier pins: LLM_SIDECAR_MODEL_FAST etc.
    models = {}
    for tier in ("fast", "balanced", "powerful"):
        v = os.getenv(f"LLM_SIDECAR_MODEL_{tier.upper()}")
        if v:
            models[tier] = v
    if models:
        out["models"] = models
    return out


def load(**overrides) -> Config:
    """Build a Config from file + env, with keyword overrides winning outright."""
    data: dict = {}
    data.update(_from_file(CONFIG_FILE))
    data.update(_from_env())
    data.update({k: v for k, v in overrides.items() if v is not None})

    # Merge rather than replace the models dict, so an env pin for one tier
    # doesn't wipe file pins for the others.
    known = {f.name for f in Config.__dataclass_fields__.values()}
    clean = {k: v for k, v in data.items() if k in known}
    if "retry_delays" in clean:
        clean["retry_delays"] = tuple(clean["retry_delays"])
    return Config(**clean)


def save(config: Config, include_api_key: bool = False) -> Path:
    """Persist config to disk. Returns the path written.

    The API key is dropped unless explicitly requested — this file is plain
    JSON in the user's home directory, and a key belongs in the environment
    or a keyring rather than sitting on disk in the clear."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = asdict(config)
    data["retry_delays"] = list(config.retry_delays)
    if not include_api_key:
        data.pop("openrouter_api_key", None)
    tmp = CONFIG_FILE.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(CONFIG_FILE)
    return CONFIG_FILE
