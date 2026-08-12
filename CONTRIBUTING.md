# Contributing

Thanks for looking. This is an early project — the API still moves — so the
most useful contributions right now are bug reports from machines and setups I
don't have.

## Getting set up

```bash
git clone https://github.com/awaistechnologist/llm-sidecar
cd llm-sidecar
./install.sh          # Windows: install.bat
./.venv/bin/pytest    # 153 tests, ~5 seconds
```

The suite is **fully offline** — every network path is stubbed. It needs no
API key, no Ollama, and no internet. If a test you write needs any of those,
it belongs in the manual-verification pile instead, not the suite.

## What a good change looks like

**Say why in the code.** The comments here explain *why* a thing is the way it
is, not what the line does. `align-items: flex-start` is worth a comment
because the default silently stretches a badge into a coloured blob; a loop
that iterates a list is not.

**Bring a test for a bug fix.** Ideally one that fails before your change. If
the bug could only be caught by running the thing — several of the sharpest
ones here were — say so in the PR rather than inventing a test that doesn't
really cover it.

**Be honest in the commit message.** State what you tried, what you verified,
and what you didn't. "Not tested against a real provider" is useful
information; silence is not.

**Don't add dependencies casually.** The core imports only `httpx`, on purpose
— a program that just wants `Sidecar().complete()` shouldn't be made to
install a web framework. New dependencies belong behind an optional extra, and
need a reason.

## Things that would genuinely help

- Bug reports from **Windows and Linux**. Most of this was written on a Mac,
  and CI has already caught one bug that only appeared on Windows.
- Model families worth adding to `PREFERRED_FAMILIES` in `picker.py`.
- Better `read_url` extraction. It is regex-based and keeps boilerplate on
  unusual markup.
- Anything in the README's "Honest limitations" section.

## Running the parts

```bash
./.venv/bin/llm-sidecar serve      # daemon + dashboard
./.venv/bin/llm-sidecar mcp        # MCP server on stdio
./.venv/bin/llm-sidecar status     # what your machine can reach
```

## Reporting a security issue

Please don't open a public issue. See [SECURITY.md](SECURITY.md).
