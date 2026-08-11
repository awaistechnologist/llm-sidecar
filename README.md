# llm-sidecar

A local capability sidecar for AI tooling. One dependency that any program on
the machine can lean on for inference, search, and grounded fact-checking —
without hardcoding a provider, an API key, or a model name.

> **Status: early (0.1.0).** The API may still move. Extracted from
> [Agora](https://github.com/awaistechnologist/agora), where the routing and
> verification logic was originally built and proven; Agora will become a
> consumer rather than the owner.

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

## Three ways in

The same core, three doors, because the consumers can't use each other's
interface:

| Door | For | Entry point |
|---|---|---|
| Python library | programs that can import it | `from llm_sidecar import Sidecar` |
| HTTP daemon | any tool in any language | `llm-sidecar serve` |
| MCP server | agents | `llm-sidecar mcp` |

## Install

Not on PyPI yet.

```bash
git clone https://github.com/awaistechnologist/llm-sidecar
cd llm-sidecar
pip install -e ".[all]"
```

The core depends only on `httpx`. Install just what you need instead:
`.[search]` for web search, `.[daemon]` for the HTTP server, `.[mcp]` for the
MCP server.

Nothing else is required — with Ollama installed, every feature below works
with no API key at all.

## Library

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

## Daemon

```bash
llm-sidecar serve                 # http://127.0.0.1:4001/v1
```

It speaks the chat-completions format every provider copied from OpenAI. That
format is a de facto standard, not a vendor tie — Ollama and OpenRouter both
accept the identical request shape, which is why routing between them is a URL
swap. Because it's standard, most tools let you point at it:

```bash
export OPENAI_BASE_URL=http://localhost:4001/v1
export OPENAI_API_KEY=unused          # the format demands the field
```

The variable is named after OpenAI because the `openai` SDK reads it, and that
SDK became the common client for talking to everything. Other tools call the
same setting `--openai-api-base`, `apiBase`, or just "Base URL" — they all mean
"where do I POST". The daemon doesn't care what any of them call it.

**The `model` field is a request, not an instruction.** A tool that hardcodes
`gpt-4o` gets a verified working model instead and never notices:

```console
$ curl -s localhost:4001/v1/chat/completions -d '{"model":"gpt-4o", ...}'
{"model": "ollama/qwen2.5:32b", ..., "x_sidecar": {"cost_usd": 0.0, "local": true,
                                                   "requested_model": "gpt-4o"}}
```

Resolution order for `model`: a tier alias (`fast`/`balanced`/`powerful`) or
budget alias (`free`/`cheap`/`best`) routes by that; anything containing a `/`
is treated as a real model id and used verbatim; everything else is ignored in
favour of the configured default.

`x_sidecar` is a non-standard addition carrying the cost receipt. Clients
ignore fields they don't recognise.

## MCP

```bash
llm-sidecar mcp                   # stdio
```

```json
{"mcpServers": {"llm-sidecar": {
  "command": "/path/to/venv/bin/python", "args": ["-m", "llm_sidecar.mcp_server"]}}}
```

Tools: `search_web`, `read_url`, `verify_claims`, `extract_claims`,
`delegate`, `sidecar_status`.

Note what's *not* here: a general "call an LLM" tool. An MCP client is already
a model, so exposing inference to it is close to a no-op. What it can't do for
itself is fetch live evidence and grade claims against it — so that's what the
MCP surface is for. `delegate` is the exception, and it's there for offloading
bulk work rather than for reasoning.

## CLI

```bash
llm-sidecar status                        # what's configured and reachable
llm-sidecar ask "explain CRDTs" --budget free
llm-sidecar verify "The Eiffel Tower is in Berlin"
```

`verify` exits non-zero if any claim came back contradicted, so it drops into
a pipeline or a pre-commit hook.

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
| `LLM_SIDECAR_HOST` | `127.0.0.1` | Daemon bind address |
| `LLM_SIDECAR_PORT` | `4001` | Daemon port |
| `LLM_SIDECAR_TOKEN` | — | Require `Authorization: Bearer <token>` |

The daemon binds loopback deliberately. It holds an API key and spends real
money on request; on `0.0.0.0` that capability belongs to anything on the
network. `LLM_SIDECAR_TOKEN` adds a shared secret on top, worth setting on a
shared machine.

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
| `daemon.py` | HTTP server, chat-completions format |
| `mcp_server.py` | MCP tools |
| `cli.py` | `serve` / `mcp` / `status` / `ask` / `verify` |

The core imports nothing but `httpx`. FastAPI and `mcp` are optional extras,
pulled in only by the doors that need them — a program that wants
`Sidecar().complete()` shouldn't be made to install a web framework. `Config`
is a plain dataclass, so a host application with its own settings store builds
one directly and never touches the config file.

## Tests

```bash
pytest                                    # 37 passing, fully offline
```

Fully offline — every network path is stubbed. Live-provider behaviour is
verified by hand; see the PR description for what was checked against real
Ollama and OpenRouter endpoints.

## Not built yet

- Response caching keyed on prompt + model
- Hardware-aware scoring of local models (which Ollama models actually fit)
- A persistent usage ledger — costs are per-response today, nothing aggregates
- Multi-turn: the daemon flattens a message list into one prompt (see below)

## Known rough edges

- **The daemon flattens conversations.** The core takes a single prompt plus a
  system string, so multi-turn chats are rendered into one prompt with role
  prefixes. Fine for one-shot tool calls, lossy for a long chat. A proper
  messages-through path is the main thing standing between this and being a
  drop-in for chat UIs.
- `read_url` is a regex text extractor, not a readability port. It keeps nav
  chrome. Fine for feeding a model, not for display.
- The `PREFERRED` model lists in `picker.py` go stale as the catalogue churns.
  Missing entries are skipped silently, so staleness degrades ranking rather
  than breaking anything.
- Tier resolution is cached per process and never expires. A model that starts
  failing gets rotated away on error, but a long-lived daemon won't
  periodically re-check whether something better came back.
