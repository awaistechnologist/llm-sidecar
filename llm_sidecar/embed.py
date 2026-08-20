"""Embeddings — text to vectors, routed the same way completions are.

The README used to rule out "embeddings, vector store, memory" as one item.
That lumped three different things together. A vector store and cross-session
memory really are a different product; a routed embedding call is the same
shape as a routed completion — local model or cloud, probe before use, cost
receipt — and belongs here.

The motivating case, from a consumer: matching a job description to a profile
with `text.includes(keyword)`. Substring matching is not similarity, and no
amount of prompt engineering upstream fixes it.

Embedding models are not chat models. Ollama will happily embed with a chat
model and return something, but a model trained for retrieval does the job far
better, so the picker prefers those and says when it fell back.
"""

from __future__ import annotations

import logging
import math

import httpx

from .config import Config
from .types import Embedding, SidecarError

logger = logging.getLogger("llm_sidecar.embed")

OLLAMA_PREFIX = "ollama/"
OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"

# Purpose-built embedding models, best first. Matched as substrings so a
# version bump does not silently drop an entry — the same lesson the chat
# picker learned the hard way.
LOCAL_EMBED_FAMILIES = ("embeddinggemma", "nomic-embed", "mxbai-embed", "bge-", "all-minilm", "snowflake-arctic-embed")
# OpenRouter proxies chat completions, not embeddings — it lists no embedding
# models at all. So this is a local-only capability today, and saying so beats
# a cloud fallback that cannot work.

MAX_BATCH = 256

# Several embedding models are trained with task prefixes and rank badly
# without them. This is not a detail a consumer should have to know, so it is
# applied automatically per family.
#
# Measured on nomic-embed-text, ranking candidates for "python backend
# developer": without prefixes it put "C++ embedded engineer" (0.460) above
# "5 years Django and FastAPI" (0.419). With them, Django leads at 0.470.
# Same model, same texts — the prefixes are the whole difference.
TASK_PREFIXES = {
    "nomic-embed": {"query": "search_query: ", "document": "search_document: ",
                    "symmetric": "clustering: "},
    "bge-": {"query": "Represent this sentence for searching relevant passages: ",
             "document": "", "symmetric": ""},
    "mxbai-embed": {"query": "Represent this sentence for searching relevant passages: ",
                    "document": "", "symmetric": ""},
    "embeddinggemma": {"query": "task: search result | query: ",
                       "document": "title: none | text: ",
                       "symmetric": "task: clustering | query: "},
}


def prefix_for(model: str, task: str) -> str:
    """The prefix this model wants for this kind of text, or empty."""
    lower = model.lower()
    for family, tasks in TASK_PREFIXES.items():
        if family in lower:
            return tasks.get(task, "")
    return ""


def pick_model(config: Config, allow_chat_model: bool = False) -> tuple[str, bool]:
    """(model id, is_dedicated). Prefers a purpose-built embedding model.

    Reads Ollama directly rather than through catalogue.ollama_models(), which
    filters out anything matching "embed" as specialised — right for chat,
    exactly wrong here."""
    from . import catalogue

    if config.embed_model:
        return config.embed_model, True

    installed = [m.get("name", "") for m in catalogue.ollama_models_raw(config)]
    for family in LOCAL_EMBED_FAMILIES:
        for name in installed:
            if family in name.lower():
                return f"{OLLAMA_PREFIX}{name}", True

    if allow_chat_model and installed:
        return f"{OLLAMA_PREFIX}{installed[0]}", False

    # Refusing beats returning bad vectors. Measured with llama3.2:1b: asked to
    # rank candidates for "python backend developer", it put "C++ embedded
    # engineer" above "5 years Django and FastAPI". Silently wrong rankings are
    # worse than an error, because nothing downstream can detect them.
    raise SidecarError(
        "No embedding model installed. Embeddings need a model trained for "
        "retrieval — a chat model produces vectors, but they rank badly:\n\n"
        "  ollama pull nomic-embed-text\n\n"
        "OpenRouter does not serve embedding models, so this is local-only. "
        "Pass allow_chat_model=True to use a chat model anyway."
    )


def embed(texts: list[str], config: Config, model: str | None = None,
          allow_chat_model: bool = False, task: str = "symmetric") -> Embedding:
    """Vectors for a list of texts, in order.

    `task` is "query", "document" or "symmetric" — it selects the prefix the
    model was trained with. Comparing a query embedding against document
    embeddings is what retrieval means; embedding both the same way is what
    plain similarity means."""
    texts = [t for t in texts if isinstance(t, str)]
    if not texts:
        return Embedding(vectors=[], model="", dedicated=True)
    if len(texts) > MAX_BATCH:
        raise SidecarError(f"{len(texts)} texts; the limit is {MAX_BATCH} per call.")

    dedicated = True
    if model is None:
        model, dedicated = pick_model(config, allow_chat_model=allow_chat_model)
    if not dedicated:
        logger.warning(f"{model} is a chat model, not an embedding model — "
                       "vectors will be weaker. `ollama pull nomic-embed-text` improves this.")

    prefix = prefix_for(model, task)
    prepared = [prefix + t for t in texts] if prefix else texts

    if model.startswith(OLLAMA_PREFIX):
        vectors = _ollama(prepared, model[len(OLLAMA_PREFIX):], config)
    else:
        vectors = _openrouter(prepared, model, config)

    return Embedding(vectors=vectors, model=model, dedicated=dedicated)


def _ollama(texts: list[str], wire_model: str, config: Config) -> list[list[float]]:
    r = httpx.post(f"{config.ollama_host}/api/embed",
                   json={"model": wire_model, "input": texts},
                   timeout=config.request_timeout)
    r.raise_for_status()
    data = r.json()
    if "embeddings" not in data:
        raise SidecarError(f"Ollama returned no embeddings: {str(data)[:160]}")
    return data["embeddings"]


def _openrouter(texts: list[str], model: str, config: Config) -> list[list[float]]:
    if not config.has_cloud:
        raise SidecarError("A cloud embedding model needs an OpenRouter API key.")
    r = httpx.post(
        OPENROUTER_EMBED_URL,
        json={"model": model, "input": texts},
        headers={"Authorization": f"Bearer {config.openrouter_api_key}",
                 "Content-Type": "application/json",
                 "HTTP-Referer": config.referer, "X-Title": config.app_title},
        timeout=config.request_timeout,
    )
    r.raise_for_status()
    data = r.json()
    if "data" not in data:
        raise SidecarError(f"Embedding call returned no data: {str(data)[:160]}")
    # The API may return out of order; index says where each belongs.
    ordered = sorted(data["data"], key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in ordered]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity, -1 to 1. Zero for a zero vector rather than NaN."""
    if len(a) != len(b):
        raise SidecarError(f"Vectors differ in length: {len(a)} vs {len(b)}. "
                           "Embeddings from different models are not comparable.")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def rank(query: str, candidates: list[str], config: Config,
         model: str | None = None, top_k: int | None = None,
         allow_chat_model: bool = False) -> list[dict]:
    """Candidates ordered by similarity to the query, best first.

    The query and the candidates are embedded with different task prefixes,
    which is what the models were trained for and what makes retrieval work.
    Both go through one model, so the scores are comparable."""
    if not candidates:
        return []
    chosen, dedicated = ((model, True) if model
                         else pick_model(config, allow_chat_model=allow_chat_model))

    q = embed([query], config, model=chosen, task="query").vectors[0]
    docs = embed(candidates, config, model=chosen, task="document").vectors

    scored = [{"text": t, "score": cosine(q, v)} for t, v in zip(candidates, docs)]
    scored.sort(key=lambda r: -r["score"])
    return scored[:top_k] if top_k else scored
