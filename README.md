# llm-sidecar

**A local sidecar that gives every tool on your machine grounded, cited, routed AI — and never asks you which model to use.**

One process. It picks a working model, searches the web, reads pages, checks
facts, and tells you what everything cost. Works with **no API key at all** if
you have [Ollama](https://ollama.com); works better with one.

```console
$ llm-sidecar answer "What is Iran's population?"
As of mid-2026, Iran's population is estimated at 93,168,497 (Worldometer,
based on UN 2024 Revision). The 2025 estimate from Wikipedia is 92,417,681.

Caveat: Sources differ by year and methodology: Worldometer (2026 mid-year)
gives 93.17 million; Wikipedia (2025 est.) gives 92.42 million.

Sources:
  · https://en.wikipedia.org/wiki/Demographics_of_Iran
  · https://www.worldometers.info/world-population/iran-population/
```

That came from the live web, not training data, and it says where it got it —
including that the sources disagree. Ask something the web can't settle and it
tells you, instead of guessing.

It is not instant: a grounded answer means a search, two or three page fetches
and a model call, so expect tens of seconds — more on free models, which are
slow and sometimes have to be rotated past.

> **Status: early (0.3.0).** The API may still move. Extracted from
> [Agora](https://github.com/awaistechnologist/agora), where the routing and
> verification were originally built and proven.

---

## Sixty seconds

```bash
git clone https://github.com/awaistechnologist/llm-sidecar
cd llm-sidecar && pip install -e ".[all]"

llm-sidecar status            # what it can see: your hardware, models, search
llm-sidecar serve             # API on :4001, dashboard at http://localhost:4001
```

Open the dashboard: a chat window, every capability in a Tools tab, and a live
view of what is being chosen and what it costs. No key required.

---

## What it does

| | |
|---|---|
| **`answer`** | Ask a question → searches, reads the pages, answers **from those pages only**, with citations. Says "not in the sources" rather than guessing. |
| **`verify`** | Grade claims against live evidence: supported / contradicted / unverified, each cited. Unresolved claims are re-checked against full page text. |
| **`fact_check`** | Pull every claim out of a document and verify each one. |
| **`complete` / `stream`** | Routed inference. Local, free cloud, or paid — you don't name a model. |
| **`search` / `read_url`** | Keyless web search, and full page text with the navigation stripped. |
| **`summarise` / `classify` / `extract`** | Bounded structured work at temperature 0, so it's repeatable and cached. |
| **hardware advisor** | Which of your Ollama models actually fit in RAM, before one crawls in swap. |
| **usage ledger** | What you spent, on which model, over time. |

## Why it exists

Most "LLM router" projects answer *which provider should this call go to*.
That's plumbing. The point here is the layer above it: capabilities that are
hard to build well and that every tool re-implements badly — grounded search,
cited verification — sharing one routing, budget and cost substrate.

The routing does earn its keep in one specific way: **a catalogue entry is not
a working model.** Free tiers throttle, checkpoints get retired, endpoints
start returning empty completions. So before handing back a model, the picker
spends one tiny call proving it answers *right now*.

---

## Four ways in

Same core, four doors, because the consumers can't use each other's interface.

### Library

```python
from llm_sidecar import Sidecar
sc = Sidecar()

a = sc.answer("What shipped in Python 3.14?")
if a.grounded:                          # False = the sources didn't settle it
    print(a.text, a.sources)

sc.verify(["The Eiffel Tower is in Berlin"])      # → contradicted, cited
sc.fact_check(article)                             # extract claims, verify each
sc.summarise(text, style="bullets", focus="security")
sc.classify(tickets, labels=["bug", "feature", "question"])
sc.extract(invoice, {"total": "amount with currency", "due_date": "ISO date"})

sc.complete("Explain CRDTs")                       # routes itself
sc.complete(messages=[...])                        # real multi-turn
sc.complete_many([p1, p2, p3])                     # concurrent, ordered
await sc.acomplete("...")

sc.local_models()   # your Ollama models, scored against this machine
sc.usage(days=30)   # what you spent
sc.status()         # everything at once
```

### HTTP daemon — for any tool, in any language

```bash
llm-sidecar serve
export OPENAI_BASE_URL=http://localhost:4001/v1
export OPENAI_API_KEY=unused        # the format demands the field
```

It speaks the chat-completions format every provider copied from OpenAI. That
format is a de facto standard, not a vendor tie — Ollama and OpenRouter accept
the identical request shape, which is why routing between them is a URL swap.
The variable is named after OpenAI because the `openai` SDK reads it; other
tools call the same setting `--openai-api-base`, `apiBase`, or "Base URL".

**The `model` field is a request, not an instruction.** A tool that hardcodes
`gpt-4o` gets a verified working model and never finds out:

```console
$ curl -s localhost:4001/v1/chat/completions -d '{"model":"gpt-4o", ...}'
{"model": "nvidia/nemotron-3-ultra-550b-a55b:free", ...,
 "x_sidecar": {"cost_usd": 0.0, "local": false, "cached": false}}
```

Beyond the standard surface: `/v1/answer`, `/v1/verify`, `/ops/*` for each
capability, `/status`, `/usage`, `/resolve-preview`, `/config/*`. Swagger at
`/docs`.

### MCP — for agents

```json
{"mcpServers": {"llm-sidecar": {
  "command": "/path/to/venv/bin/python", "args": ["-m", "llm_sidecar.mcp_server"]}}}
```

`answer_question` · `search_web` · `read_url` · `verify_claims` ·
`extract_claims` · `fact_check_document` · `summarise` · `classify` ·
`extract_fields` · `delegate` · `usage_report` · `sidecar_status`

Note what's *not* there: a general "call an LLM" tool. An MCP client is
already a model, so exposing inference to it is close to a no-op. What it
can't do for itself is fetch live evidence and grade claims against it.
`delegate` is the exception — it's for offloading bulk work to something cheap.

### CLI

```bash
llm-sidecar answer "who currently runs the ECB?"   # non-zero exit if ungrounded
llm-sidecar verify "the Great Wall is visible from space"
llm-sidecar sum report.md --style bullets
llm-sidecar models                                  # local models vs your RAM
llm-sidecar usage --days 30
llm-sidecar searxng up                              # better search, one command
```

---

## How a model gets chosen

Two questions, always. **Tier** = how capable. **Budget** = what it may cost.
They are independent.

### What each capability asks for

```
  verify · fact_check · summarise · classify        ┐
  extract · extract_claims · delegate               ├──► tier "fast"
                                                    ┘    bulk work, cheap

  answer · chat · complete()                        ───► tier "balanced"

  budget is always your configured default unless you pass one.
```

There is no separate "chat model". One picker serves everything, so changing a
key or a budget changes every capability at once.

### The resolution order

Four checks, first match wins.

```
  ① model="ollama/qwen2.5:32b" given?   ──yes──►  use it. never rotated.
                    │ no
  ② is this tier LOCKED to a model?     ──yes──►  use it. budget ignored.
                    │ no
  ③ resolved this tier+budget <15m ago? ──yes──►  reuse it.
                    │ no
  ④ resolve ▼

     budget picks the candidate pool
     ┌──────────────────────────────────────────────┐
     │ free + key     free cloud models, Ollama next│
     │ free + no key  Ollama only                   │
     │ cheap          paid, under $1 per M tokens   │
     │ best           paid, over $5 per M tokens    │
     └──────────────────────────────────────────────┘
                    │
          probe 3 at once: "Reply OK"
                    │
        ┌───────────┴────────────┐
   one answers               all 3 fail
        │                         │
  use it, cache 15m       probe the next 3 …
                                  │
                        nothing left → NoWorkingModel
```

Probing concurrently changes only how *fast* a model is found — the
highest-priority success still wins, not whichever replied first. If a model
passes the probe and then fails the real call, it's marked dead for the
process and the next request routes elsewhere.

**Which free cloud model, specifically?** A curated list first, then the rest
of the pool by context length. Curated entries missing from the live catalogue
are skipped silently, which happens constantly — of nine curated free picks,
one survived to today. So in practice: the survivors, then the roomiest.

### When free OpenRouter models are used

Exactly one combination reaches them. **Tier is irrelevant to free-vs-paid.**

| budget | API key | picks from |
|---|---|---|
| `free` | **yes** | **free OpenRouter first**, Ollama as backup |
| `free` | no | Ollama only |
| `cheap` | yes | paid under $1/M — never free, never Ollama |
| `best` | yes | paid over $5/M |
| `cheap` / `best` | no | nothing eligible |

With a key and `budget=free`, the three tiers get three *different* free
models — pushing parallel work through one free endpoint is how you collect
429s.

### Locking a tier

Normally a tier chooses its own model and re-checks every 15 minutes.
**Locking** overrides that: always this model, no search, no probe, **no
budget** — a locked tier stays put even if you ask for `best`.

```bash
LLM_SIDECAR_MODEL_FAST=ollama/gemma3:27b          # env
```
```
dashboard → Local models → click "fast" on a row
POST /config/tier {"tier": "fast", "model": "..."}
```

Worth it for reproducibility and no probe latency. The cost: a locked tier
never rotates away, so if that model starts failing, your calls fail with it.

### `auto` is not a mode

`model` can only carry **one** axis:

| value | sets | leaves alone |
|---|---|---|
| `fast` `balanced` `powerful` | tier | budget |
| `free` `cheap` `best` | budget | tier |
| anything with a `/` | the exact model | overrides both |
| `auto`, `gpt-4o`, any other string | **nothing** | both stay default |

`auto` is an unrecognised string, and every unrecognised string means "use
both defaults" — which is exactly what lets a tool hardcoding `gpt-4o` work.
Because one field can't say "powerful **and** best", the daemon also takes a
separate `budget` field, and the dashboard has two selectors.

Not sure what you'd get? `GET /resolve-preview?tier=fast&budget=free` tells
you — current state, the candidate order, and why — without probing anything.

### What answered, and what it cost

| where | how |
|---|---|
| dashboard chat | `ollama/llama3.2:3b · 0.3s · 37 tok · free` under each reply |
| library | `.model` `.usage.cost_usd` `.cached` `.latency_s` |
| HTTP | `model`, plus an `x_sidecar` block |
| streaming | a final frame carrying usage and cost |
| CLI | a receipt on stderr |
| all of it | the ledger — `llm-sidecar usage` |

---

## Retrieval: who does the searching

| mode | cost | when |
|---|---|---|
| DuckDuckGo | free, no setup | the default |
| SearXNG | free, one command | more engines, no shared rate limit |
| OpenRouter | **billed per result** | when the free ones are being blocked |

### SearXNG

```bash
llm-sidecar searxng up
```

Writes a compose file and settings to `~/.config/llm-sidecar/searxng/`,
generates a secret, starts the container, and waits until it really answers a
JSON query before reporting success. Then it's detected automatically.

Why it needs a command: SearXNG ships with its JSON API **disabled**, so the
stock image returns 403 to every request, which reads as "unavailable" and
falls back silently — the failure looks like nothing happening. The shipped
settings enable `formats: [html, json]` and turn off the bot limiter, which
protects public instances and here would only throttle you. Bound to
`127.0.0.1`: no auth, no limiter, so exposing it would hand anyone a free
search proxy on your address.

Already running one? Point `SEARXNG_URL` at it.

### OpenRouter

Not a search API. Its web plugin retrieves **inside** a chat completion — you
can't get results without paying for a completion too — which is why it's a
mode on the answer path rather than a search provider.

```python
sc.answer("...", via="openrouter")     # billed, ~$4 per 1000 results
sc.complete("...", web=True)
```

Worth it when local search is being CAPTCHA'd: retrieval happens on their
side, against better sources. Never a default, never implicit, and never
cached — paying for retrieval and then serving a stored answer defeats the
point.

---

## Dashboard

`llm-sidecar serve` also serves a dashboard at **http://localhost:4001**.

**Dashboard** — API key and budget, status, which local models fit, spend over
30 days, cache size, whether SearXNG is really being used.
**Chat** — multi-turn, streaming, with the model and cost under every reply.
**Tools** — every capability, in one place, to try before you wire it in.

One HTML file. No build step, no dependencies, **no external requests** — no
CDN, no fonts, no analytics; there's a test asserting it. Settings apply to
the running daemon and are not written to disk unless you tick "remember".

Opt out entirely with `llm-sidecar serve --no-ui` (or `LLM_SIDECAR_NO_UI`);
`/` then 404s and the API is untouched. That controls *serving*, not
installing — the page is a ~40 KB file in the package either way.

---

## Configuration

Precedence: defaults < `~/.config/llm-sidecar/config.json` < environment <
keyword arguments to `Sidecar(...)`.

| Env var | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Unset means local-only. Settable from the dashboard. |
| `OLLAMA_HOST` | `http://localhost:11434` | Local inference endpoint |
| `LLM_SIDECAR_BUDGET` | `free` | `free` \| `cheap` \| `best` |
| `LLM_SIDECAR_MODEL_{FAST,BALANCED,POWERFUL}` | — | Lock a tier |
| `LLM_SIDECAR_SEARCH_PROVIDER` | `auto` | `auto` \| `ddg` \| `searxng` |
| `SEARXNG_URL` | `http://localhost:8888` | Where to find SearXNG |
| `LLM_SIDECAR_HOST` / `_PORT` | `127.0.0.1` / `4001` | Daemon bind |
| `LLM_SIDECAR_TOKEN` | — | Require `Authorization: Bearer …` |
| `LLM_SIDECAR_NO_UI` | — | API only, no dashboard |
| `LLM_SIDECAR_NO_CACHE` / `_NO_LEDGER` | — | Turn those off |

The daemon binds loopback deliberately: it holds an API key and spends real
money on request. `config.save()` does **not** write the API key unless asked.

### Caching and cost

Deterministic requests (`temperature=0`) and searches are cached to disk,
which is why re-checking a document doesn't re-pay for the claims that didn't
change. Creative requests are deliberately **not** cached — a byte-identical
"random" answer is a surprise, not an optimisation. The cache is trimmed
oldest-first to 256 MiB; the ledger rotates at 8 MiB.

---

## Performance

Measured on an M3 Max, not estimated.

| | before | after |
|---|---|---|
| Evidence gathering, 8 claims | 11.3s sequential | 2.3s parallel |
| Model probing, 4 dead candidates ahead | ~10s | 4.0s |
| Batch of 5 completions | 2.6s | 0.3s |
| Repeat deterministic completion | 8.4s | cached |

---

## Honest limitations

- **Verification is only as good as retrieval.** The verifier can only grade
  what came back; ambiguous evidence produces a wrong verdict. Unverified
  claims are re-checked against full page text, and SearXNG covers more
  engines — neither is a fix. When retrieval finds nothing useful you get
  `unverified`, and that is the correct answer.
- `read_url` strips `<nav>`/`<header>`/`<footer>` and finds the content
  region, but it's regex-based, not a readability port. Unusual markup
  degrades to the whole page.
- Streaming bypasses the completion cache. Replaying stored tokens is a
  different feature.
- The daemon has no request queue or admission control. Fine for one user,
  wrong for anything shared.
- Parallel verification helps against cloud models, not against a single local
  Ollama model — those queue server-side anyway.
- No embeddings, vector store, or cross-session memory. Different product.
- `searxng` drives Docker or Podman compose. Anything else: run the compose
  file yourself and set `SEARXNG_URL`.

---

## Layout

| Module | Responsibility |
|---|---|
| `config.py` | `Config` dataclass, file + env loading |
| `catalogue.py` | OpenRouter list (disk-cached), local Ollama models |
| `picker.py` | Candidate ranking, live probing, `pick` / `pick_pool` |
| `client.py` | `complete` / `stream`, provider routing, backoff, web plugin |
| `answer.py` | Grounded question answering |
| `verify.py` | Claim extraction, evidence, grading, escalation |
| `ops.py` | `summarise` / `classify` / `extract` with pinned schemas |
| `search/` | Provider dispatch, DDG, SearXNG, `read_url` |
| `cache.py` · `ledger.py` · `hardware.py` | Cache, spend record, memory fit |
| `daemon.py` · `mcp_server.py` · `cli.py` · `ui/` | The four doors |

The core imports nothing but `httpx`. FastAPI and `mcp` are optional extras,
pulled in only by the doors that need them:

```bash
pip install -e ".[search]"    # web search
pip install -e ".[daemon]"    # HTTP server + dashboard
pip install -e ".[mcp]"       # MCP server
pip install -e ".[all]"
```

## Tests

```bash
pytest
```

152 tests, fully offline — every network path is stubbed. Live-provider
behaviour is verified by hand.

## Licence

MIT.
