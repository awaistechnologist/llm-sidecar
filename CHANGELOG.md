# Changelog

Notable changes per release. Dates are release dates.

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
