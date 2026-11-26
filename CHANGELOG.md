# Changelog

Notable changes per release. Dates are release dates.

This file is **served, not just published**: `whats_new(since=...)` over MCP,
`changelog://llm-sidecar`, and `GET /changelog?since=...` all read it, so a
consumer can ask what changed rather than wait to be told.

That means an **Action required** section is read by people and agents you
will never meet. Write each item as a condition they can test against their
own code — "You rename image files after generation" — not as an instruction
aimed at whoever reported it. State what changed, what now breaks, and what
to do instead. Anything that does not demand a change belongs in Added,
Fixed or Changed.

## [0.12.0] — 2026-11-26

### Added
- **`claude-api/` provider: Claude through the Anthropic API, paid from API
  credits.** Max, Ultra and Team plans include a monthly credit grant; this
  spends it and never goes past it. The sidecar keeps a running total of
  each call's cost for the billing period and refuses a call before sending
  it if its worst case could pass 95% of the allowance
  (`claude_api_monthly_credit_usd`, default 100; `claude_api_reset_day`,
  default 1). A "credit balance is too low" reply stops the provider until
  the next period. Off by default and never auto-picked. Needs the
  `[claude-api]` extra and credentials from `ant auth login` or
  `ANTHROPIC_API_KEY`. Ids: `claude-api/haiku|sonnet|opus` or an exact id.
- **Allowances: what each provider has left.** `llm-sidecar allowances`,
  `GET /allowances`, the `allowances` MCP tool and a dashboard panel show
  OpenRouter spend, key limit and free-model requests left today; the Claude
  subscription's 5-hour and weekly usage windows; and API credit used
  against its allowance. Each figure is labelled live, last-seen or
  estimated.
- The phone app lists `claude-api/` models with the credit left.
- `llm-sidecar ask --model <id>`. The README showed this flag; it did not exist.
- **Default models for every provider.** The dashboard's Settings has a
  default-model picker per tier listing local, `claude/`, `claude-api/` and
  OpenRouter models; `llm-sidecar config tier <tier> <model|auto>` does the
  same from the shell. Tier locks used to offer local models only.
- A Claude API credits switch, allowance and renewal day on the dashboard.
- `delegate` over MCP takes an optional `model`.

### Changed
- **Settings changed on the dashboard are saved.** Budget, tier defaults
  and the Claude switches used to apply to the running daemon only and
  revert at the next restart. The OpenRouter key is still saved only when
  "remember" is ticked, and the dashboard now says where the key in use
  lives (saved on this machine, environment, or this session only).
- `llm-sidecar config budget|tier|claude-cli|claude-api…` also applies the
  change to a running daemon, rather than waiting for a restart.
- The dashboard chat's model list includes the Claude subscription and API
  credit models; it listed Ollama and OpenRouter only.
- `claude/` calls now read Claude Code's `stream-json` output for blocking
  calls too, which is how the subscription's usage windows and the
  credential each call used are seen. When the last reading says a window
  is used up, `claude/` fails at once with the reset time.
- A `claude/` call that Claude Code reports as billed to an API key (for
  example through an `ant auth login` sign-in, which stripping environment
  variables does not remove) is now recorded as real spend.
- When OpenRouter reports today's free-model requests used up, the `free`
  budget skips OpenRouter's free models and goes to local ones.

## [0.11.0] — 2026-11-26

### Added
- **SearXNG runs natively, no Docker.** `llm-sidecar searxng install` puts
  SearXNG in the sidecar's data directory (a git checkout, its own venv,
  and the same `granian` server the official container runs) and starts it
  at login through launchd, or a systemd user unit on Linux. Measured on a
  36 GiB Mac: ~150 MiB, against ~5 GiB for Docker Desktop's VM plus ~1.8
  GiB for Docker's own apps, to host a container using 172 MiB. Search
  also stops depending on another app being started: every reboot without
  Docker at login had quietly turned it off.

- `llm-sidecar searxng update` pulls the latest SearXNG and rolls back if
  the new version does not answer. SearXNG scrapes other sites and its
  engines break when they change; Docker hid updates behind `docker pull`.
  `searxng status` runs a real query and lists failing engines and why
  (CAPTCHA, timeout), and says when the checkout is over 30 days old.
- `llm-sidecar searxng uninstall [--purge]`, and `logs` for the native
  instance.
- **The daemon restarts SearXNG when it stops answering.** launchd already
  restarts a crashed process; this covers a wedged one or an unloaded
  service. At most once every five minutes.
- **Search says when it is not working.** `search()` still returns `[]` on
  failure, but now records it, and `search.health()` reports `ok`,
  `degraded` (on the DuckDuckGo fallback, or recent empty results) or
  `down`, with a sentence saying what to do. Shown on the dashboard, in
  `/searxng/status` (`health`, `native`), in `llm-sidecar searxng status`,
  and in the phone app's Web mode where the question is typed. Before
  this, a dead search engine looked like an answer with no sources.

### Removed
- **The Docker route for SearXNG**: `llm-sidecar searxng up` and `down`,
  the compose file, and the Docker/Podman detection. `/searxng/status` no
  longer reports `container` or `docker`.

### Action required
- **You ran SearXNG with `llm-sidecar searxng up`.** Remove the container
  (`docker rm -f llm-sidecar-searxng`), which frees port 8888, then run
  `llm-sidecar searxng install`. Your `settings.yml` is kept and reused.

## [0.10.0] — 2026-11-18

### Changed
- **Image generation now goes to the daemon, so one process owns the model.**
  Every caller — the CLI, the MCP server, the Python library, another repo —
  posts the job to the local service and does not load a model itself.

  This is what every other capability here already did. Chat and embeddings
  go to Ollama, which owns those weights. Images were the exception only
  because mflux is a library rather than a server, so whichever process
  called it paid the ~12 GiB: four MCP servers each answering an image
  request could hold four copies of the same model on one machine.

  With no daemon running it still works in-process, serialised by the
  machine-wide file lock, which is the fallback rather than the design.
  `GET /v1/images/status` and `images.status()` report
  `delegating_to_daemon` so a caller can tell which path it is on.

  A refusal from the daemon — a non-commercial profile, a missing reference
  — is re-raised as the same exception class and is **not** retried locally.
  Falling back on a refusal would mean `commercial=true` is enforced only
  when the service happens to be down.

### Added
- **The sidecar on your phone.** `llm-sidecar phone` puts the daemon on your
  Tailscale network with `tailscale serve` — never `funnel` — on its own
  HTTPS port, 8443, so it cannot replace anything already served on 443.
  Only your own signed-in devices can reach it; it refuses and undoes the
  setup if that port turns out to be on Funnel, which would make it public.

  The daemon serves a phone-first app at `/app/`: chat with any engine,
  grouped by where it runs (on the Mac, the Claude subscription, OpenRouter);
  web answers with their sources; image generation on the Mac's GPU; links
  and text shared from any app to summarise, ask about or fact-check; voice
  in and replies read aloud. It installs from Chrome with a Share entry, and
  like the dashboard it loads nothing from anywhere else.

  `mobile/` is an Android app around the same page. It adds the system
  speech recogniser, saving images to the gallery, links opening in their
  own apps, and an offline screen that retries when you come back.

- **`llm-sidecar image prepare --profile <name>`** saves an already-quantised
  copy of a profile's model, and later loads use it. mflux otherwise loads
  the full-precision weights and quantises them in memory, so both copies
  are resident at once: that is most of zimage-negprompt's 27.8 GiB peak.
  Same weights, same images, without the full-precision copy. Preparing
  costs the old peak once and waits for the memory. It then makes one image
  to measure the new peak, which the memory check uses from then on.
  Also `POST /v1/images/prepare`, and `prepared` in `/v1/images/status`.
- **A memory watchdog runs during every image job.** It reads macOS's
  memory pressure level and swap use once a second, without forking a
  process. If pressure stays critical or swap grows by more than 3 GiB
  during a job, the process holding the model exits on the spot: that
  releases all of its memory at once, launchd restarts the daemon, and
  image work pauses for 60 seconds. One failed image, not a panicked Mac.
  With that in place, `prepare` (and the first run of a freshly prepared
  model) may start with 80% of the peak free instead of all of it plus a
  margin. The full figure is close to unreachable on a 36 GiB Mac: macOS
  and login items hold ~10 GiB even after a restart.
- **Image requests wait for memory** (`wait_for_memory_s`, default 300)
  before refusing, because the callers that hit the limit are batch
  scripts that stop at the first error. The phone app passes 0.
- An idle model gives MLX's buffer cache back after a minute and keeps
  its weights. Idle, `standard` held 12.4 GiB of cache beside 3.2 GiB of
  weights.

### Action required
- **You call the daemon by a name other than `localhost`, `127.0.0.1`, `::1`,
  a `*.ts.net` name, or `host.docker.internal` / `host.containers.internal`**
  — a LAN hostname, a custom `/etc/hosts` entry, a reverse proxy that passes
  its own Host through. The daemon now checks the Host header and answers
  those with a 403 whose message names the fix. Test it with
  `curl -i http://<the-name-you-use>:4001/health`; if it is refused, add the
  name with `LLM_SIDECAR_HOSTS=name1,name2`. A daemon bound to `0.0.0.0` does
  not check, because binding every interface already says it is reachable by
  any name.

  The check exists because Tailscale delivers phone requests to 127.0.0.1,
  making the Host header the only thing that tells them apart — and because
  without it any web page could reach a loopback daemon by DNS rebinding,
  re-pointing its own domain at 127.0.0.1.
- **You call the Python library or the CLI with `out_dir` pointing somewhere
  the daemon cannot write.** The job now runs in the daemon's process, so
  the image is written by the daemon's user with the daemon's file-access
  permissions, not yours. On macOS this bites with privacy-protected
  folders: a service installed under launchd may have no access to
  `~/Documents`, `~/Desktop` or `~/Downloads` even though your shell does.
  Test it by generating one image into your real output directory with the
  service running; if it fails where it used to work, grant the service
  Full Disk Access or write somewhere outside those folders.
- **You import `llm_sidecar` and relied on image generation happening in
  your own process** — to pre-warm the engine, to read MLX memory figures
  after a call, or to measure the engine itself. Those now describe a
  process that never loaded a model. Pass `allow_daemon=False` to
  `images.generate()` to force the local path. (`llm-sidecar image bench`
  already does this, because it measures the engine.)

### Fixed
- **Image jobs could crash the whole Mac.** A profile that fits "tight"
  (zimage-negprompt: ~28 GiB on a 36 GiB machine) was allowed to load
  whatever else was running, with only a log warning. Beside a browser,
  Ollama and Docker that starved macOS of memory until its watchdog
  restarted the machine (twice, 18 and 25 Nov). Now, when a job is next
  in the queue, the engine checks the memory free at that moment. If there
  is not enough it asks Ollama to unload its models, and if there is still
  not enough it refuses with `MemoryBusy`: HTTP 503 with `Retry-After`,
  re-raised as `images.MemoryBusy` for library callers. **A pipeline that
  treated every non-200 as fatal should retry on 503.**
- **After a GPU timeout, the next job went into a broken engine.** A Metal
  command-buffer timeout leaves MLX unusable in that process. The engine
  now unloads the model, refuses image work for 60 seconds
  (`MemoryBusy`), and, when it runs as the launchd service, exits so
  launchd starts a clean process. The job that hit the fault gets
  `EngineFault` (503). Other requests in flight at that moment are cut
  off too.
- **"There is no Stream(cpu, N) in current thread"**: 35 image jobs failed
  with this. MLX streams belong to one thread, and the server ran each
  request on whichever worker thread was free, so a model built on one
  thread was later used from another. All MLX work now runs on a single
  dedicated thread.
- **An idle model was only unloaded when someone polled status**, so a
  daemon nobody polled could keep 27 GiB resident indefinitely. A timer
  now unloads it, and a model that takes more than 60% of usable memory
  goes after 1 idle minute instead of 10.
- The test module had 21 duplicated test definitions from a bad edit, and
  because Python keeps the last definition of a name, the copies that ran
  were the stale ones. One of them had no isolation from the local service,
  so running the suite on a machine with the daemon up posted real
  generation jobs to it and then blocked on a 1800s timeout. Deduplicated,
  and the fixture that drives images now pins `daemon_available` to False.

## [0.9.3] — 2026-11-18

### Fixed
- **The install instructions on PyPI were not usable.** The README is the
  PyPI description, and it told people to run `/path/to/venv/bin/python` — a
  placeholder nobody can paste and no directory can parse. Both documented
  launch paths are now real commands, verified from a clean install:
  `uvx --from llm-sidecar llm-sidecar-mcp` with nothing installed first, or
  `llm-sidecar-mcp` after `pipx install llm-sidecar`.
- The README listed 12 MCP tools when there are 17 — the four image tools
  and `whats_new` were missing, and had been for three releases. A reader
  cannot tell a short list from a complete one, so a test now fails if a
  tool exists that the README does not name, or if the stated count drifts.
- `USING.md`'s MCP registration command had a hardcoded path into one
  machine's home directory.

### Added
- Tests for the surfaces a consumer touches: every MCP tool is driven with
  its real signature while everything underneath fails, asserting it returns
  an error rather than raising at an agent; `changes` degrades honestly when
  the changelog is missing; and `images.generate` is covered without mflux.
- A writeup of the image benchmarking in `docs/`, including the result that
  contradicted the design — a cache cap *below* the working set was the
  slowest setting measured.

### Changed
- The scheduler tests no longer assert a grouping count across racing
  threads. That failed on one CI runner and passed everywhere else, which is
  a flaky test rather than a flaky scheduler — and a red check on the default
  branch shows up on directory listings as "tests failing".

## [0.9.2] — 2026-11-18

### Added
- **Listed in the MCP Registry.** `llm-sidecar-mcp` is a dedicated console
  script for the MCP server: an MCP client installing a PyPI package runs the
  console script with no arguments, and the multi-command CLI exited with an
  argparse error on that, which is a poor first run. `llm-sidecar mcp` still
  works and is unchanged.
- The README carries the `mcp-name` marker the registry uses to verify PyPI
  ownership, and `server.json` describes the server for publication. A test
  keeps the two in agreement, because a broken marker is only noticed at the
  next publish.

## [0.9.1] — 2026-11-18

0.9.0 was tagged but never published: its build failed, so nothing reached
PyPI. 0.9.1 is the first release carrying the change-discovery feature
described under 0.9.0 below.

### Fixed
- **The release build failed on the changelog it was meant to ship.**
  `python -m build` builds the sdist and then builds the wheel *from inside
  it*, so `CHANGELOG.md` — force-included into the wheel but absent from the
  sdist's file list — came back as "Forced include not found". Local
  verification had used `python -m build --wheel`, which goes straight from
  the source tree and cannot see the problem.
- CI now builds sdist and wheel the way the release does, installs the wheel
  and asserts it serves a non-empty changelog. A packaging mistake fails a
  pull request instead of a release.

## [0.9.0] — 2026-11-18

### Added
- **Consumers can ask what changed instead of being told.** A consumer pins
  assumptions — profile names, defaults, what a parameter means, where files
  land — and those go stale. Telling each one by hand does not scale and, in
  practice, gets skipped: this project renamed a profile and left an MCP tool
  advertising the old name for two releases.

  Three ways in, all reading one source of truth (CHANGELOG.md, now shipped
  inside the wheel rather than duplicated):

  - `whats_new(since="0.7.0")` — an MCP tool. `since` is the version *you*
    integrated against, not the one running.
  - `changelog://llm-sidecar` — an MCP resource, for clients that prefer
    resources to tools.
  - `GET /changelog?since=0.7.0` — for consumers that are not MCP clients.

  The field that matters is `action_required`, drawn from a release's
  **Action required**, **Breaking** or **Removed** sections. It is empty when
  upgrading is enough, so "something changed" and "you must do something" are
  distinguishable without reading prose.

  The server's MCP instructions now tell an agent to call `whats_new` when
  the version it connects to differs from the one it integrated against.
  That is the only place the habit can be planted without a human relaying
  it.

### Action required
Three behaviours were corrected in 0.7.1. Code written against the old
behaviour still runs, and now produces the wrong result. Check whether any
of these describe your integration:

- **You rename image files after generation.** Output is
  `<out_dir>/<stem>_<seed>.png`; it always was. If you renamed afterwards to
  get a name of your own, send `filename` instead and the service writes
  exactly that. Renaming on top of `filename` will fight it.
- **You scale `resolution` before sending it to `upscale`.** It sets the
  **short** edge — `scale = resolution / min(width, height)` — and always
  did; the documentation said "long edge" and was wrong. If you halved or
  doubled the value to compensate, stop: the behaviour did not change, only
  its description, so the compensation is now a second error.
- **You force upscales to regenerate.** The skip-if-unchanged digest ignored
  `resolution`, so a second upscale of one image at a different size was
  wrongly skipped. It is covered now. If you pass `skip_existing: false` or
  vary the seed purely to defeat that, you can stop and let the digest work.

None of these affect you if you take the response's `path` as authoritative
and do not post-process names or sizes.

## [0.8.1] — 2026-11-18

### Fixed
- **The `list_image_profiles` MCP tool advertised a profile that no longer
  exists.** Its docstring still read "named for intent (draft, fast,
  cinematic, characters-consistent)" two releases after `draft` became
  `zimage-negprompt`. An agent following that would ask for a profile the
  service refuses — and nothing caught it, because the data the tool
  *returns* was always correct and only the prose was wrong. A test now
  fails if a retired profile name appears in any image tool's docstring.
- The `zimage-negprompt` note still said "~31 GiB" after the peak was
  measured at 27.8 and the field corrected to 28.
- The `upscale` profile note now states that `resolution` sets the short
  edge, where a caller choosing a profile will actually see it.

## [0.8.0] — 2026-11-18

### Added
- **The image queue prefers whatever model is already loaded.** Strict
  first-come-first-served had a pathological case that was measured, not
  theorised: with two consumers using the daemon with different profiles,
  the log showed a construct and an unload on *every single generation* —
  klein-edit, unload, z-image-turbo, unload, klein-edit, unload. Each swap
  is a model reload, and for Z-Image that alone costs about 40s (117s cold
  against 76s warm).

  Both consumers had been told to "order your batch by profile", which
  cannot work: neither can see the other's queue.

  Now the oldest waiter needing the resident model goes first, with
  fairness bounded rather than hoped for — a request overtaken three times
  goes next whatever is loaded. Twelve strictly alternating jobs went from
  11 model swaps to 3.

  Affinity is keyed on the model rather than the profile, so
  `characters-consistent` and `lineart` are affine and swap nothing.

  Two honest limits: this orders work inside the daemon, and a library-mode
  caller in another process is still serialised by the file lock, which has
  no priority; and `GET /v1/images/status` now reports
  `queue: {waiting, busy, served, model_swaps, overtakes}` so thrashing is
  visible instead of being something you have to go and measure.

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
