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
       "profile":"standard",
       "out_dir":"/Users/you/pics",
       "stem":"apple",
       "project":"my-app"}'
```

Ask for a **profile**, never a model. `GET /v1/images/profiles` lists them
with current numbers.

### What each profile costs

Measured on an M3 Max (36 GiB) through the daemon. Plan from these, not from
the fastest one.

| profile | typical size | time | peak RAM | refs | commercial |
|---|---|---|---|---|---|
| `standard` **(default)** | 1024x1024 | **23s** | 12 GiB | 0 | yes |
| `characters-consistent` | 1024x640 | **60s** (2 refs) / **93s** (3) | 14 GiB | up to 3 | yes |
| `zimage-negprompt` | 1024x768 | **76s** warm, 117s cold | 28 GiB | 0 | yes |
| `upscale` (SeedVR2) | to 2400px | ~132s | 20 GiB | 1 | yes |
| `fast` | 1024x1024 | est. | 12 GiB | 0 | yes (gated: HF token) |
| `cinematic` | 1024x1024 | est. | 14 GiB | 0 | **no** |
| `kontext-edit` | 1024x1024 | est. | 14 GiB | 1 | **no** |
| `lineart` | 1024x640 | ~93s | 14 GiB | up to 3 | yes |

### Seven things that will bite you otherwise

1. **Budget from the edit figure, not the generation figure.** A picture book
   is mostly `characters-consistent` at 60-93s, not `standard` at 23s. Using
   23s to estimate a batch will be wrong by 4x.
2. **Order a batch by profile.** Switching profiles unloads one model and
   loads another, and for `zimage-negprompt` that swap alone costs ~40s
   (117s cold against 76s warm). Generate everything, *then* upscale the
   keepers — do not interleave.
3. **Only `zimage-negprompt` can keep something OUT of an image.** Every
   other profile runs klein at guidance 1.0, which cannot exclude anything:
   a `negative` sent to them is accepted and silently ignored. If a negative
   prompt is your content gate, name that profile explicitly. It is never
   substituted in, and it is not the default because it needs 28 GiB.
4. **Slow and synchronous.** Set your client timeout to **at least 300s**;
   600s if you use 3 refs or upscale.
5. **One at a time, machine-wide.** A second request waits rather than
   competing for memory. That is the point — two models at once is what made
   a 36 GiB machine swap.
6. **`commercial` defaults to `true`**, which *refuses* `cinematic` and
   `kontext-edit` (non-commercial weights). Send `"commercial": false` only
   if the output genuinely is not commercial.
7. **Consistency needs references.** Describing a character in words gives a
   different person every time. Pass up to 3 `refs` with
   `characters-consistent`.

### Writing prompts for these models

Reported from real use, and surprising enough to be worth stating:

- **Naming a thing pulls it in, even to forbid it.** "no hair visible" added
  hair. A style block containing "friendly faces" added a child to a prompt
  about an apple. These models have no reliable negation: describe only what
  you want present.
- **`guidance` is fixed at 1.0** for the klein profiles and is not a knob
  the API exposes. Raising it is how you would normally strengthen prompt
  adherence elsewhere; here you cannot, so put the weight in the prompt.
- **A negative prompt does nothing** except on `zimage-negprompt`. It is
  accepted and ignored everywhere else — see above. If you are relying on
  one to keep a subject out, that profile is the only one that honours it.
- **A style block applies to every image it is in.** If it describes people
  ("friendly faces", "rosy cheeks"), expect people to appear in images that
  were not supposed to contain any. Keep character description out of the
  shared style string and in the per-image prompt.

### Where your image is written

`<out_dir>/<stem>_<seed>.png`, **not** `<out_dir>/<stem>.png`. The seed is
in the filename on purpose: it makes an image reproducible from its own name
and means two seeds of the same shot cannot silently overwrite each other.
A collision becomes `<stem>_<seed>-2.png` rather than a clobber.

**Use the `path` in the response rather than predicting the name.** It is
returned on every call, including when the image was skipped as already
current, and it is authoritative.

If your pipeline has its own naming scheme — a shot list numbered
`S001.png`, say — send `"filename": "S001.png"` and that is exactly what
you get, rather than renaming afterwards. Never-overwrite still applies: a
collision becomes `S001-2.png`. The name is treated as a basename, so it
cannot write outside `out_dir`.

### Don't regenerate what hasn't changed

Every image gets a `<image>.png.json` manifest recording the request, model,
licence and a **digest** over everything that changes the pixels — including
the *contents* of the reference images, not their paths. Rename a reference
and the image stays current; edit one and it does not.

Re-requesting an identical image returns immediately with
`"skipped": true` and `"seconds": 0.0`. Pass `"skip_existing": false` to
force a regenerate.

### Quality is fixed at q8, deliberately

4-bit is about 20% faster and paints fake signatures into the output, which
was measured, not assumed. Every profile pins q8 or better and the service
refuses anything lower. If you need more speed, reduce the **resolution** —
1024x640 is roughly half the cost of 1024x1024 — and upscale the keepers.

**Content policy stays yours.** Imagery gates, modesty rules, anachronism
lists — the sidecar does not own them and never weakens them. It offers no
way to turn your gate off.

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
