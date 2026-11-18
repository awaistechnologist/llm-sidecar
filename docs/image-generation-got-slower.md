# My image generation got slower the longer it ran, and a smaller cache made it worse

I built a local image service on an M3 Max so five projects on one machine
could stop each carrying their own copy of [mflux](https://github.com/filipstrand/mflux).
One pinned install, one model resident at a time, one queue. The warm model
was the point: reloading weights costs real time, so keeping one loaded
should make everything after the first image faster.

Then I benchmarked it, and almost everything I had assumed turned out to be
wrong.

## The symptom

FLUX.2 klein-4b at q8, 1024×1024, four denoising steps, on a machine that
started 92% free:

```
run 1    23.9s
run 2    22.5s
run 3    77.6s
```

Same prompt shape, same size, same process. The third image took three and a
half times the second.

My first guess was thermal throttling. `pmset -g therm` recorded no warning.
Second guess was memory pressure, but the machine reported 64% free after the
run. Third guess was the text encoder: FLUX.2 klein uses Qwen3 at
`hidden_size 2560`, roughly as large as the 4B transformer it sits beside,
and mflux's own source says keeping it loaded "wastes 8-12 GB throughout the
denoising loop".

That last one was a real finding — and not the cause.

## What the numbers actually said

MLX reports three figures worth watching:

```
active   3.2 GiB     the weights and live arrays
cache   16.1 GiB     freed buffers it keeps for reuse
peak    11.8 GiB     the high-water mark of a generation
```

**Sixteen gigabytes of cache beside three gigabytes of weights**, on a
36 GiB machine. MLX keeps freed buffers rather than returning them to the
OS, and nothing bounded it.

The first version of the fix was obvious and wrong: clear the cache after
every image. That halved the second run and did not fix the trend — the
third was still 85s against 34s cold. So I measured the cap itself.

## The result that changed the design

Four runs at each setting, where a generation peaks at 11.9 GiB:

| MLX cache cap | times | worst / best |
|---|---|---|
| uncapped (grew to 16.2 GiB) | 22.5 – 77.6s | 3.4× |
| 4 GiB | 35.8 – 48.7s | 1.4× |
| 8 GiB | 31.8 – 40.3s | 1.3× |
| **12 GiB** | **23.1 – 23.9s** | **1.03×** |

Smaller is not safer. **A cap below the working set was the slowest setting
tested** — slower, on its good runs, than no cap at all. Capping at 4 GiB
forces MLX to release and re-acquire memory in the middle of a generation,
and that churn costs more than the paging it avoids.

A cap just above the working set holds it without churning and still refuses
to balloon. So the rule is not "keep the cache small". It is **size the cap
to the peak, not to a fraction of RAM** — which is what the service does now,
re-measuring and raising the cap when a generation peaks higher than the
profile estimated.

End to end through the real engine, with that in place: **22.6 – 25.9s**.

## Two other assumptions that did not survive

**A warm model was not faster.** mflux loads weights lazily, construction is
under a second, and warm measured no faster than cold for klein. What holding
one model buys is a bound on how much sits in memory at once — which on a
36 GiB laptop with a 31 GiB profile is reason enough, just not the reason I
had written in the module docstring. For Z-Image, much the biggest model
here, residency does pay: 117s cold against 76s warm.

**mflux's own `MemorySaver` was the wrong tool**, despite the CLI always
installing it. It caps the cache, which is right, but it also evicts the text
encoder after encoding — and FLUX.2 keeps no prompt cache, so the model is
rebuilt for every prompt. Measured, that costs about 20s on *every* image:
43s steady against 32s. It trades a rare cliff for a permanent tax.

## The bug my own instrumentation introduced

While measuring all this I added a `peak` figure to the service's status
endpoint, and three different jobs came back reporting `27.83 GiB` — a number
three different models do not produce.

`mx.get_peak_memory()` is a **process-lifetime** maximum. The code read it as
if it described the generation that had just run. So one Z-Image job peaking
at 27.8 GiB sized a klein *edit's* cache cap: 26.4 GiB for a 12 GiB profile.
That run paged more than any other measured — my own code causing the exact
failure the cap exists to prevent.

Resetting the high-water mark before each generation fixed it, and the
operation that benefited most was reference-guided editing, which is what a
picture book is mostly made of:

```
3-reference edit     before      after
time                 136.6s      93.0s
cache                23.3 GiB    14.9 GiB
pageouts             797         41
```

## What I would tell myself at the start

**Benchmark three runs, not two.** A two-run benchmark called the uncapped
build the fast one and missed the cliff entirely. The slowest run is the
number you live with.

**Instrument before theorising.** Thermal, memory pressure and the text
encoder were all plausible, all cheap to assert, and all wrong. The figure
that mattered — cache size — was one API call away and I had not looked.

**A measurement that does not match the real path is not a measurement.**
Half the mistakes here were verification taking a shortcut: reading peak
memory without resetting it, capping the cache but measuring after rather
than during, and later building a release artefact a different way than CI
builds it.

**Smaller is not automatically safer.** The intuition that a tight cache
bound is the conservative choice cost 50% on every image until the numbers
said otherwise.

---

The service is [llm-sidecar](https://github.com/awaistechnologist/llm-sidecar).
`llm-sidecar image bench` runs the benchmark above on your own machine, times
four images and prints every one, because the average is the number that
hides the problem.
