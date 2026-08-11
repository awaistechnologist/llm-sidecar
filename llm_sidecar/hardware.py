"""Hardware awareness — will this local model actually run well here?

Ollama will happily load a model that doesn't fit, then spill to swap and
crawl at a rate that looks like a hang. Being *installed* is not the same as
being *usable*, and the difference is invisible until you're 90 seconds into
a call that should have taken three.

So we do the arithmetic up front: how much memory the machine really has to
give, how much the model needs at the context you intend to use, and whether
the gap is comfortable. Apple Silicon is treated specially because unified
memory means the GPU draws from the same pool as everything else.

Deliberately dependency-free — no psutil, no GPUtil. `sysctl` and
`/proc/meminfo` are always there, and a missing reading degrades to "unknown"
rather than a crash.
"""

from __future__ import annotations

import logging
import platform
import re
import subprocess

from .config import Config
from .types import ModelInfo

logger = logging.getLogger("llm_sidecar.hardware")

GIB = 1024 ** 3

# Rough KV-cache cost per 1k tokens of context, in GiB. Varies by
# architecture; this is a middle-of-the-road figure good enough to tell "fits
# easily" from "will swap", which is the only question being asked.
KV_GIB_PER_1K_CTX = 0.13

# Leave this much for the OS, the editor, the browser. Loading a model that
# fits only if nothing else is running is not "fitting".
HEADROOM_GIB = 3.0


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def probe() -> dict:
    """What this machine has. Missing values come back as None."""
    system = platform.system()
    info = {
        "system": system,
        "arch": platform.machine(),
        "total_ram_gib": None,
        "available_ram_gib": None,
        "unified_memory": False,
        "chip": None,
    }

    if system == "Darwin":
        raw = _run(["sysctl", "-n", "hw.memsize"])
        if raw and raw.isdigit():
            info["total_ram_gib"] = round(int(raw) / GIB, 1)
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["chip"] = chip
        # Apple Silicon shares one memory pool between CPU and GPU, so the
        # whole figure is available to inference rather than a separate VRAM.
        info["unified_memory"] = platform.machine() == "arm64"

        # `vm_stat` pages are the only free-memory signal without psutil.
        vm = _run(["vm_stat"])
        if vm:
            page = 4096
            m = re.search(r"page size of (\d+) bytes", vm)
            if m:
                page = int(m.group(1))
            free = inactive = 0
            for line in vm.splitlines():
                if line.startswith("Pages free:"):
                    free = int(re.sub(r"\D", "", line) or 0)
                elif line.startswith("Pages inactive:"):
                    inactive = int(re.sub(r"\D", "", line) or 0)
            if free or inactive:
                info["available_ram_gib"] = round((free + inactive) * page / GIB, 1)

    elif system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                mem = {}
                for line in f:
                    k, _, v = line.partition(":")
                    mem[k.strip()] = v.strip()
            if "MemTotal" in mem:
                info["total_ram_gib"] = round(int(mem["MemTotal"].split()[0]) / (1024 ** 2), 1)
            if "MemAvailable" in mem:
                info["available_ram_gib"] = round(int(mem["MemAvailable"].split()[0]) / (1024 ** 2), 1)
        except OSError:
            pass
        # Discrete NVIDIA card, if present — that's the real inference budget.
        smi = _run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        if smi and smi.split("\n")[0].strip().isdigit():
            info["vram_gib"] = round(int(smi.split("\n")[0].strip()) / 1024, 1)

    return info


def usable_gib(hw: dict | None = None) -> float | None:
    """Memory we can realistically give a model, after headroom.

    `hw` is checked against None rather than falsiness: an empty dict is a
    caller saying "assume we know nothing about this machine", and quietly
    probing the real one instead would answer a different question."""
    if hw is None:
        hw = probe()
    budget = hw.get("vram_gib") or hw.get("total_ram_gib")
    if budget is None:
        return None
    return max(0.0, budget - HEADROOM_GIB)


def requirement_gib(size_bytes: int, context_tokens: int = 8000) -> float:
    """Weights plus KV cache at the intended context length."""
    return size_bytes / GIB + (context_tokens / 1000) * KV_GIB_PER_1K_CTX


def assess(size_bytes: int, hw: dict | None = None, context_tokens: int = 8000) -> dict:
    """Verdict for one model: fits / tight / too_big / unknown."""
    need = requirement_gib(size_bytes, context_tokens)
    have = usable_gib(hw)
    if have is None:
        return {"verdict": "unknown", "needs_gib": round(need, 1), "usable_gib": None}

    if need <= have * 0.6:
        verdict = "fits"
    elif need <= have:
        verdict = "tight"
    else:
        verdict = "too_big"
    return {
        "verdict": verdict,
        "needs_gib": round(need, 1),
        "usable_gib": round(have, 1),
        "headroom_gib": round(have - need, 1),
    }


def advise(config: Config, context_tokens: int = 8000) -> list[dict]:
    """Every installed Ollama model, scored against this machine.

    Ordered best-first: usable models by descending size (bigger reasons
    better, given it fits), then the ones that don't fit."""
    import httpx

    try:
        r = httpx.get(f"{config.ollama_host}/api/tags", timeout=2.0)
        raw = r.json().get("models") or [] if r.status_code == 200 else []
    except Exception:
        return []

    hw = probe()
    rows = []
    for m in raw:
        name = m.get("name")
        if not name:
            continue
        size = m.get("size", 0)
        a = assess(size, hw, context_tokens)
        rows.append({
            "id": f"ollama/{name}",
            "name": name,
            "size_gib": round(size / GIB, 1),
            "parameter_size": (m.get("details") or {}).get("parameter_size", ""),
            "quantization": (m.get("details") or {}).get("quantization_level", ""),
            **a,
        })

    rank = {"fits": 0, "tight": 1, "unknown": 2, "too_big": 3}
    rows.sort(key=lambda r: (rank[r["verdict"]], -r["size_gib"]))
    return rows


def best_local(config: Config, context_tokens: int = 8000) -> ModelInfo | None:
    """The largest installed model that comfortably fits, if any."""
    for row in advise(config, context_tokens):
        if row["verdict"] == "fits":
            return ModelInfo(id=row["id"], name=f"{row['name']} (local)")
    return None
