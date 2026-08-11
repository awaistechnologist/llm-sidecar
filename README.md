# llm-sidecar

A local capability sidecar for AI tooling. One dependency that any program on
the machine can lean on for inference, search, and grounded fact-checking —
without hardcoding a provider, an API key, or a model name.

> **Status: early.** Lives inside the Agora repo while the API settles. It has
> no dependency on Agora and will move to its own repo once it has a second
> real consumer.

## Why

Most "LLM router" projects answer *which provider should this call go to*.
That's the plumbing here, not the point. The point is the layer above it:
capabilities that are hard to build well and that every tool re-implements
badly — grounded search, cited verification — all sharing one routing, budget,
and cost substrate.

The routing does earn its keep in one specific way: **a catalogue entry is not
a working model.** Free tiers throttle, checkpoints get retired, endpoints
start returning empty completions. So before handing back a model, the picker
spends one tiny call proving it answers *right now*.

## Install

Nothing to install yet beyond the parent repo's requirements (`httpx`, `ddgs`).

```bash
pip install -r requirements.txt
```

## Use

```python
from llm_sidecar import Sidecar

sc = Sidecar()

# Inference — routes itself. No model name required.
sc.complete("Explain CRDTs in two sentences").text
sc.complete("Summarise this log", tier="fast", budget="free")
sc.complete("Hard question", model="ollama/qwen2.5:32b")   # or pin one

# Streaming
async for ev in sc.stream("Write a haiku"):
    if ev["type"] == "token":
        print(ev["text"], end="")

# Search — DuckDuckGo by default, SearXNG if you're running one
sc.search("rust async runtime comparison")
sc.read_url("https://example.com/article")     # full text, not a snippet

# Grounded verification — the differentiated bit
for v in sc.verify(["The Eiffel Tower is in Berlin"]):
    print(v.verdict, v.note, v.sources)
    # contradicted  The evidence states the Eiffel Tower is in Paris.  [...]
```

Everything above runs with **no API key** if you have Ollama installed.

## Configuration

Precedence: defaults < `~/.config/llm-sidecar/config.json` < environment <
keyword arguments to `Sidecar(...)`.

| Env var | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Unset means local-only. Not required. |
| `OLLAMA_HOST` | `http://localhost:11434` | Local inference endpoint |
| `LLM_SIDECAR_BUDGET` | `free` | `free` \| `cheap` \| `best` |
| `LLM_SIDECAR_MODEL_{FAST,BALANCED,POWERFUL}` | — | Pin a tier, skipping auto-pick |
| `LLM_SIDECAR_SEARCH_PROVIDER` | `auto` | `auto` \| `ddg` \| `searxng` |
| `SEARXNG_URL` | `http://localhost:8888` | Where to find SearXNG |

`config.save()` deliberately **does not write the API key** to disk. Pass
`include_api_key=True` if you really want it in a plaintext file in `$HOME`.

### Tiers and budgets

A **tier** (`fast`/`balanced`/`powerful`) is what the caller asks for. A
**budget** (`free`/`cheap`/`best`) is what it's allowed to cost. Pin a tier to
a model and it's used verbatim; leave it unset and the picker resolves it live
against the budget.

`Sidecar.pool()` returns three *distinct* verified models mapped onto the
tiers. That's for parallel workloads: spreading concurrent calls across
providers is the difference between a free tier that works and one that 429s
halfway through.

### SearXNG (optional)

DuckDuckGo is the default because it needs no setup. It also rate-limits hard
and returns short snippets. If you run several searches in parallel — which
`verify()` does — a local SearXNG is a real upgrade:

```bash
docker run -d -p 8888:8080 --name searxng searxng/searxng
```

Then enable JSON output in its `settings.yml`:

```yaml
search:
  formats: [html, json]
```

With `search_provider: auto` (the default), it's detected and used
automatically; if it stops answering, searches fall back to DuckDuckGo.

## Layout

| Module | Responsibility |
|---|---|
| `config.py` | `Config` dataclass, file + env loading |
| `catalogue.py` | OpenRouter model list (disk-cached), local Ollama models |
| `picker.py` | Candidate ranking, live pretest, `pick` / `pick_pool` |
| `client.py` | `complete` / `stream`, provider routing, 429 backoff |
| `search/` | Provider dispatch, DDG, SearXNG, `read_url` |
| `verify.py` | Claim extraction, evidence gathering, judging |
| `__init__.py` | The `Sidecar` facade |

No SQLAlchemy, no FastAPI, no framework. `Config` is a plain dataclass, so a
host application with its own settings store constructs one and never touches
the config file.

## Not built yet

- **OpenAI-compatible daemon** (`/v1/chat/completions` on localhost) so any
  tool in any language can point `OPENAI_BASE_URL` at it and inherit the
  routing. This is the main reason the package exists in this shape.
- **MCP server** exposing `search` / `read_url` / `verify_claims` to agents.
  Note the asymmetry: MCP is for agents calling *us*. We call SearXNG over its
  plain JSON API rather than through an MCP server, because this is a Python
  process and spawning a Node subprocess to do an HTTP GET buys nothing.
- Response caching, hardware-aware local model scoring, usage ledger.

## Known rough edges

- `read_url` is a regex-based text extractor, not a readability port. It keeps
  nav chrome and boilerplate. Fine for feeding a model, not for display.
- The `PREFERRED` model lists in `picker.py` go stale as the catalogue churns.
  Missing entries are skipped silently, so staleness degrades ranking rather
  than breaking anything.
- No test suite yet.
