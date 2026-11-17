# Using llm-sidecar from another app or agent

A local daemon on this machine at **`http://127.0.0.1:4001`**. It gives you
chat, grounded answers with citations, fact-checking, embeddings, and local
image generation — without your project holding an API key, a model list, or
mflux.

Nothing here needs an API key from you. The sidecar holds the credentials.

---

## 1. Chat — it is OpenAI-compatible

Point any OpenAI client at it. The `api_key` is required by the format and
ignored.

```python
from openai import OpenAI

llm = OpenAI(base_url="http://127.0.0.1:4001/v1", api_key="unused")

r = llm.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Explain CRDTs in two sentences"}],
)
print(r.choices[0].message.content)
```

Streaming, `response_format`, and JSON schemas all work as usual.

### What to put in `model`

You almost never need a real model id.

| `model` | what happens |
|---|---|
| `auto` | the sidecar decides |
| `fast` / `balanced` / `powerful` | pick by speed-vs-quality |
| `free` / `cheap` / `best` | pick by what it may cost |
| `claude/sonnet`, `claude/opus`, `claude/haiku` | the Claude subscription on this machine |
| `ollama/llama3.1:latest` | a specific local model |
| anything else (`gpt-4o`, …) | ignored; routed by the default |

That last row is the point: an app hardcoded to `gpt-4o` gets a working model
instead of an error, and never finds out.

Every response carries an `x_sidecar` block saying what actually answered,
what it cost, and whether it was cached.

---

## 2. Beyond chat

```bash
curl -s localhost:4001/v1/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the population of Iran?"}'
```

| endpoint | for |
|---|---|
| `POST /v1/answer` | answer a question from live web sources, **with citations** |
| `POST /v1/verify` | fact-check a list of claims against live evidence |
| `POST /v1/embeddings` | OpenAI shape; local embedding models |
| `POST /v1/rank` | rank candidates by meaning, not substring |
| `POST /ops/summarise`, `/ops/classify`, `/ops/extract` | bounded structured work |
| `POST /ops/search`, `/ops/read-url` | web search and page reading |
| `GET /v1/models`, `/status`, `/health` | what is available |

**Check `grounded` on `/v1/answer` before trusting the text.** `false` means
the sources did not contain the answer, and the text says what was missing
rather than guessing.

---

## 3. Images (Apple Silicon)

```bash
curl -s localhost:4001/v1/images/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a red apple on a wooden table",
       "profile":"draft",
       "out_dir":"/Users/you/pics",
       "stem":"apple",
       "project":"my-app"}'
```

Ask for a **profile**, never a model. `GET /v1/images/profiles` lists them.

| profile | for | refs | commercial use |
|---|---|---|---|
| `draft` | general work, ungated, real negative prompt | 0 | yes |
| `fast` | FLUX.1-schnell (needs an HF token) | 0 | yes |
| `cinematic` | best-looking | 0 | **no** |
| `characters-txt2img` | character sheets | 0 | yes |
| `characters-consistent` | **same character across frames** | up to 3 | yes |
| `kontext-edit` | continuity edit of an existing frame | 1 | **no** |
| `lineart` | colouring pages | up to 3 | yes |
| `upscale` | SeedVR2 | 1 | yes |

Four things to know:

- **It is slow and synchronous.** 15s to several minutes. Set your client
  timeout to at least 300s.
- **One at a time, machine-wide.** A second request waits rather than
  competing for memory. That is deliberate.
- **`commercial` defaults to `true`**, which *refuses* `cinematic` and
  `kontext-edit`. Send `"commercial": false` only if the output genuinely is
  not commercial.
- **Consistency needs references.** Describing a character in words gives a
  different person every time. Pass `refs` with `characters-consistent`.

Each image gets a `<image>.png.json` manifest beside it recording the
request, model, licence and a digest. Re-requesting an identical image is
skipped rather than regenerated.

**Content policy stays yours.** Imagery gates, modesty rules, anachronism
lists — the sidecar does not own them and never weakens them.

---

## 4. For Claude Code and other MCP agents

Register once and the tools appear in every repo:

```bash
claude mcp add llm-sidecar -- /Users/awaistahir/.local/share/llm-sidecar/venv/bin/python -m llm_sidecar.mcp_server
```

Tools: `search_web`, `answer_question`, `read_url`, `verify_claims`,
`extract_claims`, `fact_check_document`, `delegate`, `summarise`, `classify`,
`extract_fields`, `generate_image`, `edit_with_references`, `upscale_image`,
`list_image_profiles`, `usage_report`, `sidecar_status`.

---

## 5. As a Python library

Same machine, no HTTP:

```python
from llm_sidecar import Sidecar

sc = Sidecar()
print(sc.complete("Explain CRDTs in two sentences").text)
print(sc.answer("What is the population of Iran?").text)
print(sc.verify(["The Eiffel Tower is in Berlin"]))
```

In library mode image generation still takes the same machine-wide lock, so
it will not fight the daemon.

---

## 6. Assume this

- **Loopback only.** Do not expose port 4001. It is unauthenticated by
  design, holds an API key, and can spend money.
- **It may not be running.** Treat a connection error as "fall back", not as
  a crash. `GET /health` is cheap.
- **Models are not stable.** Ask for `auto` or a tier. Anything that pins an
  exact cloud model id will rot — a previous curated list of nine lost eight
  in a few months.
- **Don't cache its answers yourself.** It already caches deterministic
  completions, searches and pages.
