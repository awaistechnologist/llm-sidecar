# Changelog

Notable changes per release. Dates are release dates.

## [0.6.0] — 2026-08-20

Four gaps reported by a consumer building against it, all confirmed against
real code before being acted on.

### Added
- **Embeddings.** `sc.embed()`, `sc.similarity()`, `sc.rank()`, and
  `POST /v1/embeddings` in OpenAI's response shape plus `POST /v1/rank`. The
  motivating case was a suitability score implemented as
  `text.includes(keyword)` — substring matching is not similarity.
- **Schema-constrained output.** `sc.complete(schema=...)` and a
  `response_format` passthrough on the daemon. Routed to the provider, which
  enforces it at decode time, rather than asking a model nicely and parsing
  whatever comes back.
- **CORS**, configurable via `LLM_SIDECAR_CORS_ORIGINS`, closed by default.

### Fixed
- The embedding picker could never see an embedding model:
  `catalogue.ollama_models()` filters anything matching "embed" as
  specialised, which is right for chat routing and exactly wrong here.

### Notes
- Task prefixes are applied automatically per model family. nomic-embed-text
  ranks *wrongly* without them — measured, same texts, same model: "C++
  embedded engineer" 0.460 above "5 years Django and FastAPI" 0.419 without,
  Django leading at 0.470 with.
- Embeddings refuse rather than falling back to a chat model, for the same
  reason: wrong rankings are undetectable downstream.
- A quality/speed hint already exists and needed no new API — `tier`
  (fast/balanced/powerful) and `budget` (free/cheap/best) are separate axes,
  both settable per call and over HTTP.

## [0.5.2] — 2026-08-12

### Fixed
- **`fast` was picking the largest local model.** The local candidate list is
  sorted largest-first and every tier took the first entry, so tier was
  ignored entirely for Ollama — "fast" selected a 19 GB 32B, the slowest model
  on the machine. Since `verify`, `summarise`, `classify` and `extract` all
  ask for the fast tier, every bulk operation was running on the biggest model
  available. Tier now maps to size locally.

### Added
- `llm-sidecar models --suggest` — what to pull, scored against your memory.
  The advisor could only rank models you already had, which is no help when
  you have none; that path now prints suggestions and an `ollama pull` line.
- `llm-sidecar models` shows which models are loaded in RAM. `ollama list` and
  `ollama ps` answer different questions and the difference confuses people.

## [0.5.1] — 2026-08-12

### Fixed
- README install block could not be copy-pasted. zsh's `interactive_comments`
  is off in many setups, so a trailing `# once, if you don't have pipx` was not
  a comment — the apostrophe in "don't" opened a quote and the terminal hung on
  `cmdand quote>`. Copy-pasteable blocks no longer carry inline comments, and a
  test now parses every one of them in a shell configured that way.

### Added
- A troubleshooting section for `command not found: pip` and
  `externally-managed-environment` — the two things a new user hits first, and
  the reason the install instructions say pipx rather than pip.

## [0.5.0] — 2026-08-12

Install and first-run fixes. Everything in this release came from watching an
actual install fail.

### Changed
- **`pip install llm-sidecar` now installs everything.** The base package was
  httpx alone, with the daemon, search and MCP behind extras — so the default
  install produced a command whose main subcommands died with a raw
  ModuleNotFoundError. The extras still exist as no-ops so old instructions
  keep working.

### Added
- `llm-sidecar service install` — writes a launchd agent or systemd user unit
  so the daemon starts at login and restarts on failure. `uninstall` and
  `status` too.
- `llm-sidecar config key <KEY> --save`, `config budget`, `config show`,
  `config clear-key`. Setting a key previously needed the daemon running and a
  curl command.
- Python 3.14 in CI. pipx picked it by default on a real machine and nothing
  had ever tested it.

### Fixed
- `service install` refuses when the executable sits in `~/Documents`,
  `~/Desktop` or `~/Downloads`. macOS keeps those behind a privacy prompt that
  background services never receive, so the agent died at startup and
  KeepAlive restarted it forever. It now explains and points at pipx.
- The generated launchd plist sets `ThrottleInterval`, so a genuine crash is
  visible in the log rather than drowned by restarts.
- `config key` without `--save` exits non-zero and says why, instead of
  reporting success for something that vanishes when the process exits.

## [0.4.0] — 2026-08-12

First public release.

### Added
- **`answer()`** — grounded question answering. Searches, reads the top pages
  in full, answers from those pages only, with citations. Reports when the
  sources don't settle the question rather than guessing.
- **Dashboard** at `http://localhost:4001` — status, settings (API key,
  budget, tier locks), a chat panel with per-reply cost receipts, and every
  capability in a Tools tab. One HTML file, no build step, no external
  requests. Opt out with `--no-ui`.
- **OpenRouter web retrieval** — `answer(via="openrouter")` and
  `complete(web=True)`. Billed per result, never a default.
- **One-command SearXNG** — `llm-sidecar searxng up`.
- `install.sh` / `run.sh` and Windows equivalents.
- `complete_many()`, `acomplete()`, `complete_json()`.
- CLI: `answer`, `models`, `usage`, `sum`, `cache`, `searxng`.
- Published to PyPI; tagged releases publish via OIDC, no token stored.

### Changed
- Model preference is now expressed as **families** rather than exact ids.
  Exact ids rotted fast — of nine curated free picks, one still existed a few
  months later, because every version bump silently dropped an entry.
- Terminology: "judge" is now "verify", "pinned" is now "locked". Both were
  undefined jargon that had leaked into the UI.
- The daemon accepts `budget` as its own field: `model` can only carry one
  axis, so "powerful and best" was previously unsayable.

### Fixed
- Every file read and write now declares `encoding="utf-8"`. The locale
  default is cp1252 on Windows, which mangled the dashboard HTML.
- Provider errors arriving as HTTP 200 with an `error` body now raise, so
  routing rotates instead of returning an empty completion as a success.
- Streamed calls were invisible to the usage ledger and carried no cost
  receipt.
- Tier resolution is cached per **tier and budget**; a `free` request could
  previously poison a later `best` one.
- `:batch` and `:online` model variants are excluded from routing — both pass
  a probe and then behave wrongly.

## [0.1.0] – [0.3.0]

Development history, before the project was public. Core routing and verified
model picking, conversations, caching, usage ledger, hardware advisor,
structured operations, and an efficiency pass. See the commit log.
