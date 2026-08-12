# llm-sidecar

A local capability sidecar for AI tooling. One dependency that any program on
the machine can lean on for inference, search, and grounded fact-checking —
without hardcoding a provider, an API key, or a model name.

> **Status: early (0.3.0).** The API may still move. Extracted from
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
| Dashboard | you | `llm-sidecar serve`, then open the URL |

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

# Ask a question and get an answer from live sources, with citations
a = sc.answer("What is Iran's population?")
if a.grounded:                     # False means the sources didn't settle it
    print(a.text, a.sources)

# Grounded verification — the differentiated bit
for v in sc.verify(["The Eiffel Tower is in Berlin"]):
    print(v.verdict, v.note, v.sources)
    # contradicted  The evidence states the Eiffel Tower is in Paris.  [...]

sc.fact_check(article_text)        # extract every claim, then verify each

# Structured work, all at temperature 0 so it's repeatable and cached
sc.summarise(long_text, style="bullets", focus="security implications")
sc.classify(tickets, labels=["bug", "feature", "question"])
sc.extract(invoice, {"total": "amount with currency", "due_date": "ISO date"})

# Concurrency — batched, or off the event loop
sc.complete_many([p1, p2, p3])            # results in input order
await sc.acomplete("...")

# Introspection
sc.local_models()   # installed Ollama models scored against this machine
sc.usage(days=30)   # what you have spent, by model
sc.status()         # everything at once
```

### Conversations

`complete` and `stream` take a full message list, not just a prompt:

```python
sc.complete(messages=[
    {"role": "system",    "content": "Answer in one word."},
    {"role": "user",      "content": "Capital of France?"},
    {"role": "assistant", "content": "Paris"},
    {"role": "user",      "content": "And Japan?"},
])   # -> "Tokyo"
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

Tools: `answer_question`, `search_web`, `read_url`, `verify_claims`, `extract_claims`,
`fact_check_document`, `summarise`, `classify`, `extract_fields`, `delegate`,
`usage_report`, `sidecar_status`.

Note what's *not* here: a general "call an LLM" tool. An MCP client is already
a model, so exposing inference to it is close to a no-op. What it can't do for
itself is fetch live evidence and grade claims against it — so that's what the
MCP surface is for. `delegate` is the exception, and it's there for offloading
bulk work rather than for reasoning.

## CLI

```bash
llm-sidecar status                        # what's configured and reachable
llm-sidecar models                        # local models scored against your RAM
llm-sidecar usage --days 30               # what you have spent, by model
llm-sidecar ask "explain CRDTs" --budget free
llm-sidecar answer "What is Iran's population?"
llm-sidecar verify "The Eiffel Tower is in Berlin"
llm-sidecar sum README.md --style bullets
llm-sidecar cache stats                   # or: cache clear
llm-sidecar searxng up                    # better search, one command
```

`verify` exits non-zero if any claim came back contradicted, so it drops into
a pipeline or a pre-commit hook.

## Dashboard

`llm-sidecar serve` also serves a dashboard at **http://localhost:4001**. Three
tabs, because there are three reasons to open it:

**Dashboard** — status, which local models actually fit this machine (with
one-click tier pinning), spend over 30 days, cache size, whether SearXNG is
really being used.

**Chat** — multi-turn, streaming, against whatever the sidecar routes you to.
This is not trying to be a chat client; point LibreChat or Open WebUI at the
`/v1` endpoint if that's what you want. It exists because every reply carries
**the model that answered, what it cost, and whether it came from cache** —
which is the routing made visible, and no general chat client can show it.

**Tools** — every capability, in one place: ask a grounded question, verify, fact-check a document,
summarise, classify, extract fields, extract claims, search, read a URL. The
point is to try something before wiring it into anything.

One HTML file, no build step, no dependencies, no external requests — no CDN,
no fonts, no analytics. A loopback page that phoned out would be both a privacy
problem and broken offline; there's a test asserting it doesn't.

Tier pins made here apply to the running daemon and are **not written to your
config** — a click that silently rewrites a file on disk is a worse surprise
than one that doesn't survive a restart.

### Turning it off

The dashboard is opt-out, not mandatory:

```bash
llm-sidecar serve --no-ui          # or: LLM_SIDECAR_NO_UI=1
```

`/` then returns 404 and the API is untouched. Note this controls *serving*,
not installing — the page is a ~40 KB file inside the package either way.

Swagger for the same API is at `/docs`.

## How a model gets chosen

Two questions, always. **Tier** = how capable. **Budget** = what it may cost.
They are independent, and every capability answers both — most of them by
taking the default.

### 1 · What each capability asks for

```
  verify · fact_check · summarise · classify        ┐
  extract · extract_claims · delegate               ├──► tier "fast"
                                                    ┘    bulk work, cheap

  answer · chat · complete()                        ───► tier "balanced"
                                                         the default tier

  budget is the configured default ("free") unless you pass one.
```

Override per call: `sc.summarise(text)` uses fast; `sc.complete(tier="powerful",
budget="best")` uses neither default.

### 2 · Tier + budget ──► an actual model

Four checks, first match wins. Same path for every capability above.

```
  ① model="ollama/qwen2.5:32b" given?  ──yes──►  use it. never rotated.
                    │ no
  ② is this tier pinned?               ──yes──►  use the pin.
     (config, env, or the F·B·P buttons)
                    │ no
  ③ resolved this tier+budget < 15 min ago? ─yes──►  reuse it.
                    │ no
  ④ resolve ▼
```

### When is a free OpenRouter model used?

Only one combination reaches them. **Budget decides; tier is irrelevant to
free-vs-paid.**

| budget | API key | picks from |
|---|---|---|
| `free` | **yes** | **free OpenRouter models first**, Ollama as backup |
| `free` | no | Ollama only |
| `cheap` | yes | paid under $1/M — never free, never Ollama |
| `cheap` | no | nothing eligible |
| `best` | yes | paid over $5/M |
| `best` | no | nothing eligible |

All three tiers draw from the same pool. With a key and `budget=free`, the
three tiers get three *different* free models — running parallel work through
one free endpoint is how you collect 429s.

Two things skip the table entirely: an explicit `model=` id, and a pinned
tier. **A pin beats the budget**, so a tier pinned to an Ollama model stays
local no matter what budget you ask for. If the dashboard says `fast + free →
ollama/...` when you expected cloud, check for a pin first.

`GET /resolve-preview?tier=fast&budget=free` answers all of this for your
current configuration without probing anything, and the chat panel shows it
inline.

### 3 · Resolving

```
  budget decides who is eligible
  ────────────────────────────────────────────────
   free  + API key   free cloud models, Ollama as backup
   free  + no key    Ollama only
   cheap             paid, under $1 per M tokens
   best              paid, over $5 per M tokens
                    │
                    ▼
        candidates, best first
                    │
         probe 3 at once: "Reply OK"
                    │
        ┌───────────┴───────────┐
   one answers              all 3 fail
        │                        │
   use it, cache 15 min    probe the next 3 …
                                 │
                          nothing left ──► NoWorkingModel
```

**Which free model, specifically?** The order is a curated list first
(`PREFERRED` in `picker.py`), then everything else in the budget sorted by
context length. Curated entries missing from the live catalogue are skipped
silently, which happens constantly — of nine curated free picks, one survived
to today. So in practice the order is "the curated ones still alive, then the
roomiest".

A catalogue entry is not a working model — free tiers throttle, checkpoints
get retired — so nothing is handed back without a live probe. If a model
passes the probe and then fails the real call, it is marked dead for the
process and the next request routes elsewhere.

### The `model` field over HTTP, and why "auto" is a trap

`model` can only carry **one** axis at a time:

| value | sets | leaves alone |
|---|---|---|
| `fast` `balanced` `powerful` | tier | budget |
| `free` `cheap` `best` | budget | tier |
| anything with a `/` | the exact model | overrides both |
| `auto`, `gpt-4o`, any other string | **nothing** | both stay default |

`auto` is not a mode. It is an unrecognised string, and every unrecognised
string means "use both defaults" — which is exactly the behaviour that lets a
tool hardcoding `gpt-4o` get a working model without knowing it.

Because one field can't say "powerful **and** best", the daemon also accepts a
separate `budget` field, and the dashboard's chat panel has two selectors
rather than one dropdown mixing all of the above together.

### Worked examples

| you do | tier | budget | lands on |
|---|---|---|---|
| chat, defaults | balanced | free | best local model, or a free cloud one |
| `sc.verify([...])` | fast | free | the fast slot |
| `{"model":"powerful","budget":"best"}` | powerful | best | a top-tier paid model |
| `{"model":"ollama/llama3.2:1b"}` | — | — | exactly that, budget ignored |
| pinned `fast` → `gemma3:27b` | fast | any | the pin, always |

### One picker, everything

There is no separate chat model. Setting a key, changing the budget, or
pinning a tier changes **every** capability at once.

### What answered, and what it cost

| where | how |
|---|---|
| dashboard chat | `ollama/llama3.2:3b · 0.3s · 37 tok · free` under each reply |
| library | `.model` `.usage.cost_usd` `.cached` `.latency_s` |
| HTTP | `model`, plus an `x_sidecar` block |
| streaming | a final frame with usage and cost |
| CLI | a receipt on stderr |
| all of it | the ledger — `llm-sidecar usage` |

## Configuration

Precedence: defaults < `~/.config/llm-sidecar/config.json` < environment <
keyword arguments to `Sidecar(...)`.

| Env var | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Unset means local-only. Not required. Settable from the dashboard. |
| `OLLAMA_HOST` | `http://localhost:11434` | Local inference endpoint |
| `LLM_SIDECAR_BUDGET` | `free` | `free` \| `cheap` \| `best` |
| `LLM_SIDECAR_MODEL_{FAST,BALANCED,POWERFUL}` | — | Pin a tier, skipping auto-pick |
| `LLM_SIDECAR_SEARCH_PROVIDER` | `auto` | `auto` \| `ddg` \| `searxng` |
| `SEARXNG_URL` | `http://localhost:8888` | Where to find SearXNG |
| `LLM_SIDECAR_HOST` | `127.0.0.1` | Daemon bind address |
| `LLM_SIDECAR_PORT` | `4001` | Daemon port |
| `LLM_SIDECAR_TOKEN` | — | Require `Authorization: Bearer <token>` |
| `LLM_SIDECAR_NO_UI` | — | Set to serve the API without the dashboard |
| `LLM_SIDECAR_NO_CACHE` | — | Set to disable the response cache |
| `LLM_SIDECAR_NO_LEDGER` | — | Set to stop recording usage |

The daemon binds loopback deliberately. It holds an API key and spends real
money on request; on `0.0.0.0` that capability belongs to anything on the
network. `LLM_SIDECAR_TOKEN` adds a shared secret on top, worth setting on a
shared machine.

The cache is trimmed oldest-first to `cache_max_bytes` (256 MiB by default),
and the usage ledger rotates at 8 MiB keeping one previous generation —
neither grows without bound in your home directory.

`config.save()` deliberately **does not write the API key** to disk. Pass
`include_api_key=True` if you really want it in a plaintext file in `$HOME`.

### Retrieval: who does the searching

| mode | cost | when |
|---|---|---|
| DuckDuckGo | free, no setup | the default |
| SearXNG | free, one command to start | more engines, no shared rate limit |
| OpenRouter | **billed per result** | when the free ones are being blocked |

The first two are ordinary web search: results come back, we read the pages,
then a model answers. OpenRouter is different in kind — it is not a search
API. Its web plugin retrieves *inside* a chat completion, so you cannot ask it
for results without also paying for a completion. That is why it appears as
`answer(via="openrouter")` and `complete(web=True)` rather than as a third
search provider.

It costs roughly $4 per 1000 results on top of tokens, so it is never a
default and never reachable implicitly. What you get for that: retrieval that
doesn't care whether DuckDuckGo is serving you CAPTCHAs, and noticeably better
sources — World Bank and WHO where local search was returning forum threads.

```python
sc.answer("What is Iran's population?")                      # free
sc.answer("What is Iran's population?", via="openrouter")    # billed
sc.complete("What shipped in Python 3.14?", web=True)        # billed
```

Web-retrieving calls are never cached: the reason to pay for retrieval is that
the answer might have changed.

### Better search with SearXNG

DuckDuckGo is the default because it needs no setup. It also rate-limits, and
its snippets are short. SearXNG aggregates many engines, is self-hosted so the
only rate limit is your own, and pairs well with `read_url` for full text.

One command:

```bash
llm-sidecar searxng up
```

That writes a compose file and settings to `~/.config/llm-sidecar/searxng/`,
generates a secret key, starts the container, and waits until it actually
answers a JSON query before reporting success. First run pulls the image, so
give it a few minutes.

```bash
llm-sidecar searxng status      # is it up, and is search actually using it?
llm-sidecar searxng logs
llm-sidecar searxng down
llm-sidecar searxng up --port 9999
```

Nothing needs configuring afterwards — with `search_provider: auto` (the
default) it is detected and used, and searches fall back to DuckDuckGo the
moment it stops answering.

**Why this needs its own command.** SearXNG ships with the JSON API disabled.
Start the stock image and every request returns 403, which llm-sidecar treats
as "unavailable" and quietly falls back — so the failure looks like nothing
happening at all. The shipped `settings.yml` enables `formats: [html, json]`
and turns off the bot limiter, which is protection for a public instance and
would otherwise only be throttling you.

The instance is bound to `127.0.0.1` deliberately: it has no authentication
and no limiter, so exposing it would hand anyone a free search proxy on your
address. The config directory is yours once created — `up` never overwrites
files that already exist.

Already running SearXNG some other way? Point `SEARXNG_URL` at it and skip all
of the above; `status` reports any reachable instance, not just ours.

## Layout

| Module | Responsibility |
|---|---|
| `config.py` | `Config` dataclass, file + env loading |
| `catalogue.py` | OpenRouter model list (disk-cached), local Ollama models |
| `picker.py` | Candidate ranking, live pretest, `pick` / `pick_pool` |
| `client.py` | `complete` / `stream`, provider routing, 429 backoff |
| `search/` | Provider dispatch, DDG, SearXNG, `read_url` |
| `verify.py` | Claim extraction, evidence gathering, grading |
| `__init__.py` | The `Sidecar` facade |
| `cache.py` | Response + search cache, deterministic requests only |
| `ledger.py` | Append-only usage and spend record |
| `hardware.py` | Memory probe and local-model fit scoring |
| `ops.py` | `summarise` / `classify` / `extract` with pinned schemas |
| `daemon.py` | HTTP server, chat-completions format |
| `mcp_server.py` | MCP tools |
| `services.py` | Container lifecycle (Docker or Podman) |
| `deploy/searxng/` | Compose file and settings shipped with the package |
| `ui/index.html` | The dashboard — one file, no build |
| `cli.py` | `serve` / `mcp` / `status` / `models` / `usage` / `ask` / `verify` / `sum` / `cache` / `searxng` |

The core imports nothing but `httpx`. FastAPI and `mcp` are optional extras,
pulled in only by the doors that need them — a program that wants
`Sidecar().complete()` shouldn't be made to install a web framework. `Config`
is a plain dataclass, so a host application with its own settings store builds
one directly and never touches the config file.

## Tests

```bash
pytest                                    # 148 passing, fully offline
```

Fully offline — every network path is stubbed. Live-provider behaviour is
verified by hand; see the PR description for what was checked against real
Ollama and OpenRouter endpoints.

## Caching and cost

Deterministic requests (`temperature=0`) are cached to disk, which is why
re-checking a document after an edit does not re-pay for the claims that
didn't change. Creative requests are deliberately **not** cached — returning
a byte-identical answer to someone who asked for randomness is a surprise.
Searches cache for an hour; a live search that isn't live is worthless.

Every call is appended to a usage ledger (`~/.local/share/llm-sidecar/`), so
`llm-sidecar usage` can tell you what you actually spent and on which models.
Both are opt-out via `LLM_SIDECAR_NO_CACHE` / `LLM_SIDECAR_NO_LEDGER`.

## Performance

Measured on this machine, not estimated:

| | before | after |
|---|---|---|
| Evidence gathering, 8 claims | 11.3s sequential | 2.3s parallel |
| Model probing, 4 dead candidates ahead | ~10s | 4.0s |
| Batch of 5 completions | 2.6s | 0.3s |
| Repeat deterministic completion | 8.4s | cached |
| Catalogue reads per `pick_pool()` | 6 disk reads | memoised |

Probing runs in waves rather than one candidate at a time: a cold start used
to pay the full timeout for every stale entry ahead of the live model.
Priority order still decides *which* model wins — concurrency only changes how
fast it is found.

Verification batches also run concurrently, which helps against cloud models. It does
not help much against a single local Ollama model, since those requests queue
server-side anyway.

## Not built yet

- Streaming does not consult the cache (it would have to replay stored tokens)
- No embeddings, vector store, or cross-session memory — different product
- The daemon has no request queue or admission control; concurrency is
  whatever the ASGI worker allows

## Known rough edges

- `read_url` is a regex text extractor, not a readability port. It keeps nav
  chrome. Fine for feeding a model, not for display.
- **Verification quality is bounded by search quality.** Verification can only
  grade what the search returned. Ambiguous evidence — a namesake, a replica,
  a stale page — produces a wrong verdict. SearXNG helps by covering more
  engines, but it is not a fix.
- `llm-sidecar searxng` drives Docker or Podman via their compose plugins. Any
  other runtime: run the compose file yourself and set `SEARXNG_URL`.
- `read_url` extracts the main content region and drops `<nav>`/`<header>`/
  `<footer>`/`<aside>`, which handles most pages. It is still regex-based, not
  a readability port, so unusual markup degrades to the whole page.
- The `PREFERRED` model lists in `picker.py` go stale as the catalogue churns.
  Missing entries are skipped silently, so staleness degrades ranking rather
  than breaking anything.
- Tier resolution is cached per process and never expires. A model that starts
  failing gets rotated away on error, but a long-lived daemon won't
  periodically re-check whether something better came back.
