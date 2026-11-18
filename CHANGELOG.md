# Changelog

Notable changes per release. Dates are release dates.

## [0.7.1] — 2026-11-18

Three bugs reported by a consumer producing a real picture book, all of
which they had to work around.

### Fixed
- **The skip-if-unchanged digest ignored `resolution`.** For `upscale` that
  is the only parameter which changes the output — width and height come
  from the profile and mean nothing — so re-upscaling one image at a
  different resolution hashed identically to the first and was skipped as
  already current. A stale upscale silently survived.
- **`filename` was accepted and ignored by the running server.** It was
  added after 0.7.0 was tagged, and the daemon runs the released package,
  so the field was dropped without a word. Shipping it is the fix.

### Changed
- **`resolution` is documented as the short edge, which is what it has
  always been.** mflux scales by `resolution / min(width, height)`; this
  project's docs said "long edge", which sent a consumer's first upscales
  to 3840 px wide. The behaviour matches mflux's own CLI and is unchanged —
  only described correctly, in the CLI help, the MCP tool and USING.md.

## [0.7.0] — 2026-11-17

### Added
- **The `claude/` provider.** A Claude subscription as a third backend,
  alongside Ollama and OpenRouter, by running the Claude Code CLI locally —
  no API key and no per-token bill. `llm-sidecar config claude-cli on`, then
  `claude/haiku`, `claude/sonnet`, `claude/opus`, or any name
  `claude --model` accepts. Works through the library, the daemon
  (`/v1/chat/completions`, streaming included), the MCP server, the CLI and
  the dashboard.

  Off by default even where the CLI is installed, and never auto-picked by
  `budget`. Ollama costs nothing to call; a subscription costs an allowance
  the sidecar cannot see, so it is only used when named or locked to a tier.

  Each call is one `claude -p` subprocess with its agent capabilities
  removed: `--tools ""`, `--safe-mode`, `--strict-mcp-config`,
  `--no-session-persistence`. The prompt goes in on stdin rather than argv,
  so it does not appear in `ps`.
- `GET /claude-cli/status`, `POST /config/claude-cli`,
  `llm-sidecar config claude-cli on|off`, and `Sidecar.claude_cli_check()`.
- `Usage.notional_cost_usd` — what a call would have cost on metered
  pricing, kept separate from what it actually cost.

### Added — images
- **Local image generation**, as the optional `[images]` extra. Profiles
  rather than model names: `draft`, `fast`, `cinematic`,
  `characters-txt2img`, `characters-consistent`, `kontext-edit`, `lineart`,
  `upscale`. Available through the library, `llm-sidecar image gen`, the
  daemon (`/v1/images/generate`, `/v1/images/profiles`, `/v1/images/status`)
  and three MCP tools, so Claude Code can make images from any repo on the
  machine without that repo carrying mflux.

  Replaces five separate integrations that had drifted to three mflux
  versions, two repos importing another repo's virtualenv, a model reload
  per image, per-directory locks that could not stop two repos competing,
  and five provenance formats.

  - One model resident at a time, swapped on demand, dropped after 10 idle
    minutes. `draft` peaks near 31 GiB, so holding two is what makes a 36 GiB
    machine swap.
  - A machine-wide lock taken through a file, because the daemon and a
    library-mode caller in another repo are different processes.
  - The service owns output naming — `<stem>_<seed>.png`, never overwriting.
  - A manifest beside each image with the request, model, mflux version,
    licence and a digest over everything that changes the pixels, including
    the *contents* of reference images rather than their paths.
  - Licences enforced before anything loads: `commercial` defaults to true,
    so the non-commercial profiles are refused unless a caller opts out.
  - A memory guard that refuses rather than swaps, reusing the same
    `hardware` module the model picker uses.

  `[images]` is deliberately not part of `[all]`: mflux is Apple-Silicon-only
  and pulls several GB, and `install.sh` runs `pip install -e ".[all]"`.
  Everything imports mflux lazily, so the package installs, imports and tests
  cleanly on Linux and Windows, where the HTTP routes answer 501 with an
  explanation.

### Fixed — image generation speed
- **MLX's buffer cache was uncapped**, which made 1024x1024 generation
  unpredictable: 22.5s to 77.6s across runs on a machine that started 92%
  free, with the cache growing to 16.2 GiB beside 3.2 GiB of live weights.
  mflux's CLI caps it through `CallbackManager`; driving the Python classes
  directly skips that. The two ways of driving mflux have different costs —
  in-process callers get the uncapped cache, CLI-per-image callers get a
  model rebuild on every prompt — and a resident model behind a capped cache
  avoids both.

  The size turned out to matter more than the fact of capping. Four runs
  each, where a generation peaks at 11.9 GiB:

  | cap | times |
  |---|---|
  | uncapped | 22.5 - 77.6s |
  | 4 GiB | 35.8 - 48.7s |
  | 8 GiB | 31.8 - 40.3s |
  | 12 GiB | 23.1 - 23.9s |

  A cap *below* the working set makes MLX release and re-acquire memory
  mid-generation, and that churn costs more than the paging did — 4 GiB was
  slower than no cap at all on its good runs. The cap is now the profile's
  peak plus 15%, raised automatically when a generation peaks higher than
  estimated. End to end through the engine: 22.6 - 25.9s.

  Two things that looked right and were not: clearing the cache after every
  image (a zero cap, the worst case above) and mflux's own `MemorySaver`,
  which caps the cache but also evicts the text encoder — and since FLUX.2
  keeps no prompt cache, that rebuilds the model per prompt for about 20s of
  tax on every image.
- `image bench` times four runs and prints each. A two-run bench called the
  uncapped build faster and missed the cliff entirely.
- Memory-tight profiles now warn. `draft` needs ~31 GiB against ~33 usable
  on a 36 GiB Mac; the cache cap cannot help when the working set alone
  nearly fills RAM.

### Fixed
Three bugs found by running the thing rather than reasoning about it:
- **2480 prompt tokens per call.** `--system-prompt` *replaces* Claude Code's
  own; omitting it carried the full coding-agent framing into every
  completion. Measured at 2480 tokens for "Say OK." against 2 with a minimal
  default in place — a thousandfold overhead charged to the user's allowance,
  and the wrong framing for a general-purpose call. A default system prompt
  is now always sent.
- **Duplicated streamed output.** The CLI emits incremental deltas *and* the
  finished assistant message carrying the same text; reading from whichever
  shape appeared returned "12345" as "1234512345". Deltas win now, with the
  whole message kept only as a fallback.
- **Phantom spend in the ledger.** The CLI reports a metered-API equivalent
  cost even on a subscription ($0.00057 for a live Haiku call), which would
  have made `llm-sidecar usage` overstate a bill nobody owed. Recorded as $0
  when `claude_cli_use_subscription` is on — which is knowable for certain,
  because that setting strips `ANTHROPIC_API_KEY` from the child process.

### Changed
- The pretest token ceiling is 512, not 10. Reasoning models spend their
  allowance thinking before writing any content, so a 10-token cap failed
  models that work and could leave `free` with nothing.

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
