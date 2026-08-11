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

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import Sidecar
from .config import Config
from .types import NoWorkingModel, SidecarError

logger = logging.getLogger("llm_sidecar.daemon")

# Model names that mean "you decide" rather than naming a specific model.
TIER_ALIASES = ("fast", "balanced", "powerful")
BUDGET_ALIASES = ("free", "cheap", "best")


class Message(BaseModel):
    role: str
    content: str = ""


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
    app = FastAPI(title="llm-sidecar", version="0.1.0")

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

    def flatten(messages: list[Message]) -> tuple[str, str | None]:
        """Split the message list into (prompt, system).

        The core takes a single prompt plus an optional system string, so a
        multi-turn conversation is rendered back into one prompt. Lossy for
        long chats — a proper messages-through path is the obvious next step."""
        system_parts = [m.content for m in messages if m.role == "system"]
        rest = [m for m in messages if m.role != "system"]

        if len(rest) == 1:
            prompt = rest[0].content
        else:
            prompt = "\n\n".join(f"{m.role}: {m.content}" for m in rest)
        return prompt, ("\n\n".join(system_parts) or None)

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "version": "0.1.0",
            "has_cloud": cfg.has_cloud,
            "budget": cfg.default_budget,
            "resolved": sidecar.resolved,
        }

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

        prompt, system = flatten(req.messages)
        target = resolve_target(req.model)
        created = int(time.time())
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if req.stream:
            return StreamingResponse(
                _stream_body(sidecar, rid, created, prompt, system, target, req),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
            )

        try:
            completion = sidecar.complete(
                prompt,
                system=system,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
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
                "requested_model": req.model,
            },
        }

    return app


async def _stream_body(sidecar, rid, created, prompt, system, target, req):
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

        async for ev in sidecar.stream(
            prompt,
            system=system,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            model=model,
        ):
            if ev["type"] == "token":
                yield chunk({"content": ev["text"]}, model)

        yield chunk({}, model, finish="stop")
    except Exception as e:
        logger.exception("stream failed")
        # The status line is long gone by now, so an error can only be
        # delivered in-band. Emit it as a final chunk rather than truncating
        # silently, which would look like a successful empty answer.
        yield chunk({"content": f"\n\n[llm-sidecar error: {e}]"}, model, finish="stop")
    yield "data: [DONE]\n\n"


app = create_app()


def main() -> None:
    import uvicorn

    from . import config as config_mod

    cfg = config_mod.load()
    logging.basicConfig(level=logging.INFO)
    logger.info(
        f"llm-sidecar on http://{cfg.daemon_host}:{cfg.daemon_port}/v1 "
        f"(cloud={'yes' if cfg.has_cloud else 'no'}, budget={cfg.default_budget})"
    )
    uvicorn.run(create_app(cfg), host=cfg.daemon_host, port=cfg.daemon_port, log_level="info")


if __name__ == "__main__":
    main()
