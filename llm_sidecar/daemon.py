"""HTTP daemon — the sidecar proper.

Speaks the chat-completions wire format that every provider copied from
OpenAI, which means any tool that lets you override its base URL can route
through this and inherit the verified picking, free-tier rotation and backoff
without a line of integration code:

    llm-sidecar serve
    export OPENAI_BASE_URL=http://localhost:4001/v1

The daemon does not care that the format is named after OpenAI, and neither
does anything it talks to — Ollama and OpenRouter both accept the same shape.

Run:
    python -m llm_sidecar.daemon
    uvicorn llm_sidecar.daemon:app --port 4001
"""

from __future__ import annotations

import json
import logging
import time
import uuid

from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import Sidecar, __version__
from .config import Config
from .types import NoWorkingModel, SidecarError

logger = logging.getLogger("llm_sidecar.daemon")

# Model names that mean "you decide" rather than naming a specific model.
TIER_ALIASES = ("fast", "balanced", "powerful")
BUDGET_ALIASES = ("free", "cheap", "best")


class Message(BaseModel):
    role: str
    content: str = ""


class TierRequest(BaseModel):
    tier: str
    # Empty string clears the pin and returns the tier to auto-resolution.
    model: str = ""


class SummariseRequest(BaseModel):
    text: str
    style: str = "brief"
    focus: str = ""


class ApiKeyRequest(BaseModel):
    # Empty string clears the key and returns the sidecar to local-only.
    key: str = ""
    # Write it to ~/.config/llm-sidecar/config.json in plaintext. Off by
    # default: the caller has to ask for that explicitly.
    persist: bool = False


class BudgetRequest(BaseModel):
    budget: str


class AskRequest(BaseModel):
    question: str
    # Override the search query when the question itself searches badly.
    query: str = ""
    max_sources: int = 4
    read_pages: int = 3
    # "local" searches and reads pages ourselves (free). "openrouter" has
    # OpenRouter retrieve and answer in one call — better sources, but billed.
    via: str = "local"


class ClassifyRequest(BaseModel):
    items: list[str]
    labels: list[str]
    multi: bool = False


class ExtractRequest(BaseModel):
    text: str
    fields: dict[str, str]


class TextRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5
    news: bool = False


class ReadRequest(BaseModel):
    url: str
    max_chars: int = 20000


class VerifyRequest(BaseModel):
    claims: list[str]
    model: str = ""


class ChatRequest(BaseModel):
    # Required by the format, but often meaningless here — see resolve_target.
    model: str = "auto"
    messages: list[Message]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = Field(default=1000, ge=1)


def create_app(config: Config | None = None) -> FastAPI:
    from . import config as config_mod

    cfg = config or config_mod.load()
    sidecar = Sidecar(cfg)
    app = FastAPI(title="llm-sidecar", version=__version__)

    def require_token(authorization: str | None = Header(default=None)) -> None:
        """Optional shared-secret gate.

        Note the deliberate asymmetry: when no token is configured we accept
        anything, including the dummy keys tools send because the format
        demands an api_key field. The loopback bind is the real boundary."""
        if not cfg.daemon_token:
            return
        expected = f"Bearer {cfg.daemon_token}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")

    def resolve_target(name: str) -> dict:
        """Turn the request's `model` field into routing instructions.

        Three cases, in order:
          - a tier or budget alias  -> route by that
          - something that looks like a real model id (has a `/`, as both
            OpenRouter and our ollama/ ids do) -> use it verbatim
          - anything else ("gpt-4o", "auto", whatever the tool hardcoded)
            -> ignore it and route by the configured default

        That last case is the point of the daemon: a tool that only knows how
        to ask for "gpt-4o" gets a working free model instead, and never finds
        out."""
        n = (name or "").strip()
        if n in TIER_ALIASES:
            return {"tier": n}
        if n in BUDGET_ALIASES:
            return {"budget": n}
        if "/" in n:
            return {"model": n}
        return {}

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "version": __version__, "has_cloud": cfg.has_cloud,
                "budget": cfg.default_budget, "resolved": sidecar.resolved}

    @app.get("/", include_in_schema=False)
    def dashboard():
        if not cfg.ui_enabled:
            raise HTTPException(
                status_code=404,
                detail="The dashboard is disabled. Start without --no-ui (or unset "
                       "LLM_SIDECAR_NO_UI) to enable it. The API is unaffected.",
            )
        return _page()

    def _page():
        """The single-page dashboard.

        Served from the package rather than a separate static host: this is a
        loopback process the user already runs, and a UI that needs its own
        deployment step is a UI nobody opens."""
        page = Path(__file__).parent / "ui" / "index.html"
        if not page.exists():
            return HTMLResponse("<h1>llm-sidecar</h1><p>Dashboard asset missing.</p>",
                                status_code=500)
        return HTMLResponse(page.read_text())

    def _op(fn, *args, **kwargs):
        """Run a capability and translate its failures into HTTP.

        Every tool panel in the dashboard funnels through here so the error
        contract is identical no matter which capability misbehaves."""
        try:
            return fn(*args, **kwargs)
        except SidecarError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except NoWorkingModel as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:
            logger.exception("operation failed")
            raise HTTPException(status_code=502, detail=str(e)) from e

    @app.post("/v1/answer")
    def op_answer(req: AskRequest, _: None = Depends(require_token)) -> dict:
        """Answer a question from live sources.

        Sits under /v1 alongside verify rather than in /ops: it is a headline
        capability, not a utility."""
        a = _op(sidecar.answer, req.question, query=req.query or None,
                max_sources=req.max_sources, read_pages=req.read_pages, via=req.via)
        return a.__dict__

    @app.post("/ops/classify")
    def op_classify(req: ClassifyRequest, _: None = Depends(require_token)) -> dict:
        return {"results": _op(sidecar.classify, req.items, req.labels, multi=req.multi)}

    @app.post("/ops/extract")
    def op_extract(req: ExtractRequest, _: None = Depends(require_token)) -> dict:
        return {"fields": _op(sidecar.extract, req.text, req.fields)}

    @app.post("/ops/extract-claims")
    def op_extract_claims(req: TextRequest, _: None = Depends(require_token)) -> dict:
        return {"claims": _op(sidecar.extract_claims, req.text)}

    @app.post("/ops/fact-check")
    def op_fact_check(req: TextRequest, _: None = Depends(require_token)) -> dict:
        verdicts = _op(sidecar.fact_check, req.text)
        return {
            "checked": len(verdicts),
            "contradicted": sum(1 for v in verdicts if v.verdict == "contradicted"),
            "results": [v.__dict__ for v in verdicts],
        }

    @app.post("/ops/search")
    def op_search(req: SearchRequest, _: None = Depends(require_token)) -> dict:
        results = _op(sidecar.search, req.query, max_results=req.max_results, news=req.news)
        return {"results": [r.__dict__ for r in results]}

    @app.post("/ops/read-url")
    def op_read_url(req: ReadRequest, _: None = Depends(require_token)) -> dict:
        return {"url": req.url, "text": _op(sidecar.read_url, req.url, max_chars=req.max_chars)}

    @app.get("/usage/daily")
    def usage_daily(days: int = 30, _: None = Depends(require_token)) -> dict:
        from . import ledger
        return {"days": ledger.daily(days)}

    @app.get("/local-models")
    def local_models(context: int = 8000, _: None = Depends(require_token)) -> dict:
        return {"models": sidecar.local_models(context_tokens=context)}

    def _mask(key: str | None) -> str:
        """Never return the key itself — a loopback port is not a vault."""
        return f"…{key[-4:]}" if key and len(key) >= 4 else ("set" if key else "")

    @app.get("/config")
    def get_config(_: None = Depends(require_token)) -> dict:
        return {
            "cloud_configured": cfg.has_cloud,
            "key_preview": _mask(cfg.openrouter_api_key),
            "default_budget": cfg.default_budget,
            "ollama_host": cfg.ollama_host,
            "search_provider": cfg.search_provider,
            "models": cfg.models,
        }

    @app.post("/config/api-key")
    def set_api_key(req: ApiKeyRequest, _: None = Depends(require_token)) -> dict:
        """Set or clear the OpenRouter key on the running daemon.

        Applies immediately — the picker reads config at call time, so the
        next request can already route to cloud models. The response never
        echoes the key back, only a masked preview."""
        from . import catalogue, config as config_mod

        key = req.key.strip() or None
        cfg.openrouter_api_key = key
        # The catalogue was fetched (or not) under the old auth; drop the memo
        # so the next lookup reflects what this key can actually see.
        catalogue.forget()
        # Resolved tiers were chosen from a different candidate pool.
        sidecar._resolved.clear()

        persisted = False
        if req.persist:
            try:
                config_mod.save(cfg, include_api_key=True)
                persisted = True
            except OSError as e:
                raise HTTPException(status_code=500, detail=f"Could not write config: {e}") from e

        return {
            "ok": True,
            "cloud_configured": cfg.has_cloud,
            "key_preview": _mask(key),
            "persisted": persisted,
        }

    @app.post("/config/budget")
    def set_budget(req: BudgetRequest, _: None = Depends(require_token)) -> dict:
        if req.budget not in BUDGET_ALIASES:
            raise HTTPException(status_code=400,
                                detail=f"Unknown budget {req.budget!r}. Expected one of {BUDGET_ALIASES}.")
        cfg.default_budget = req.budget
        return {"ok": True, "default_budget": cfg.default_budget}

    @app.post("/config/tier")
    def set_tier(req: TierRequest, _: None = Depends(require_token)) -> dict:
        """Pin a tier to a model, or clear the pin with an empty model.

        Runtime only — deliberately not written to disk. The dashboard is for
        trying things; a click that silently rewrites the user's config file
        is a worse surprise than one that doesn't survive a restart."""
        if req.tier not in TIER_ALIASES:
            raise HTTPException(status_code=400,
                                detail=f"Unknown tier {req.tier!r}. Expected one of {TIER_ALIASES}.")
        models = dict(cfg.models)
        if req.model:
            models[req.tier] = req.model
        else:
            models.pop(req.tier, None)
        cfg.models = models
        return {"ok": True, "models": cfg.models, "persisted": False}

    @app.post("/ops/summarise")
    def summarise(req: SummariseRequest, _: None = Depends(require_token)) -> dict:
        try:
            return {"summary": sidecar.summarise(req.text, style=req.style, focus=req.focus)}
        except SidecarError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

    @app.get("/searxng/status")
    def searxng_status(_: None = Depends(require_token)) -> dict:
        from . import services
        try:
            return services.status(cfg)
        except Exception as e:
            return {"error": str(e)}

    @app.get("/status")
    def status(_: None = Depends(require_token)) -> dict:
        """Everything about this sidecar: hardware, local model fit, search
        provider, resolved tiers, cache size, 30-day spend."""
        return sidecar.status()

    @app.get("/usage")
    def usage(days: int | None = None, _: None = Depends(require_token)) -> dict:
        return sidecar.usage(days)

    @app.post("/cache/clear")
    def cache_clear(_: None = Depends(require_token)) -> dict:
        from . import cache as cache_mod
        return {"removed": cache_mod.clear()}

    @app.post("/v1/verify")
    def verify(req: VerifyRequest, _: None = Depends(require_token)) -> dict:
        """Grounded fact-checking. Not part of the chat-completions standard —
        it's the capability the daemon exists to share, so it gets an endpoint
        alongside the compatibility surface."""
        try:
            verdicts = sidecar.verify(req.claims, model=req.model or None)
        except SidecarError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception("verify failed")
            raise HTTPException(status_code=502, detail=str(e)) from e
        return {"results": [v.__dict__ for v in verdicts]}

    @app.get("/v1/models")
    def list_models(_: None = Depends(require_token)) -> dict:
        """Catalogue in the standard listing shape, so tools that populate a
        model dropdown from this endpoint work. The aliases are listed first
        because they're the ones worth choosing."""
        data = [
            {"id": a, "object": "model", "owned_by": "llm-sidecar", "created": 0}
            for a in ("auto",) + TIER_ALIASES + BUDGET_ALIASES
        ]
        for m in sidecar.models():
            data.append({
                "id": m.id,
                "object": "model",
                "owned_by": "ollama" if m.is_local else "openrouter",
                "created": 0,
                "context_length": m.context_length,
            })
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest, _: None = Depends(require_token)):
        if not req.messages:
            raise HTTPException(status_code=400, detail="`messages` must not be empty.")

        target = resolve_target(req.model)
        msgs = [m.model_dump() for m in req.messages]
        created = int(time.time())
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if req.stream:
            return StreamingResponse(
                _stream_body(sidecar, rid, created, msgs, target, req),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
            )

        try:
            completion = sidecar.complete(
                messages=msgs,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                operation="daemon",
                **target,
            )
        except NoWorkingModel as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except SidecarError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception("completion failed")
            raise HTTPException(status_code=502, detail=f"Upstream call failed: {e}") from e

        return {
            "id": rid,
            "object": "chat.completion",
            "created": created,
            "model": completion.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": completion.text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            },
            # Non-standard, additive: the receipt. Clients ignore what they
            # don't recognise, and it's the whole reason to route through here.
            "x_sidecar": {
                "cost_usd": completion.usage.cost_usd,
                "local": completion.is_local,
                "cached": completion.cached,
                "latency_s": completion.latency_s,
                "requested_model": req.model,
            },
        }

    return app


async def _stream_body(sidecar, rid, created, msgs, target, req):
    """SSE chunks in the standard streaming shape, terminated by [DONE]."""
    def chunk(delta: dict, model: str, finish: str | None = None) -> str:
        payload = {
            "id": rid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    model = "unknown"
    try:
        model = target.get("model") or sidecar.model_for(target.get("tier"), target.get("budget"))
        yield chunk({"role": "assistant", "content": ""}, model)

        usage = None
        async for ev in sidecar.stream(
            messages=msgs,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            model=model,
            operation="daemon-stream",
        ):
            if ev["type"] == "token":
                yield chunk({"content": ev["text"]}, model)
            elif ev["type"] == "usage":
                usage = ev["usage"]

        yield chunk({}, model, finish="stop")

        # A final frame carrying the receipt. The streaming format has no slot
        # for cost, and dropping it meant a streamed reply arrived with no way
        # to know what it spent — which is the one thing routing through here
        # is supposed to tell you. Clients that don't expect it see a chunk
        # with no choices and ignore it, which is the documented behaviour for
        # the usage frame OpenAI itself emits.
        if usage:
            yield "data: " + json.dumps({
                "id": rid, "object": "chat.completion.chunk", "created": created,
                "model": model, "choices": [],
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                },
                "x_sidecar": {
                    "cost_usd": usage.cost_usd,
                    "local": model.startswith("ollama/"),
                    "streamed": True,
                },
            }) + "\n\n"
    except Exception as e:
        logger.exception("stream failed")
        # The status line is long gone by now, so an error can only be
        # delivered in-band. Emit it as a final chunk rather than truncating
        # silently, which would look like a successful empty answer.
        yield chunk({"content": f"\n\n[llm-sidecar error: {e}]"}, model, finish="stop")
    yield "data: [DONE]\n\n"


app = create_app()


def main(no_ui: bool = False, port: int | None = None) -> None:
    import uvicorn

    from . import config as config_mod

    cfg = config_mod.load()
    if no_ui:
        cfg.ui_enabled = False
    if port:
        cfg.daemon_port = port

    logging.basicConfig(level=logging.INFO)
    base = f"http://{cfg.daemon_host}:{cfg.daemon_port}"
    logger.info(
        f"llm-sidecar API on {base}/v1 "
        f"(cloud={'yes' if cfg.has_cloud else 'no'}, budget={cfg.default_budget})"
    )
    logger.info(f"dashboard on {base}" if cfg.ui_enabled else "dashboard disabled")
    uvicorn.run(create_app(cfg), host=cfg.daemon_host, port=cfg.daemon_port, log_level="info")


if __name__ == "__main__":
    main()
