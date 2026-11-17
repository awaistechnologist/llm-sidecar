"""Naming and provenance — one format, so five repos stop inventing five.

Two problems solved here, both of which every project on this machine hit
separately:

**Never overwrite.** mflux refuses to overwrite an existing file and writes
alongside it instead, which three repos each worked around differently. The
service owns naming instead: `<stem>_<seed>.png`, and a collision becomes
`<stem>_<seed>-2.png` rather than a silent clobber or a refusal.

**A digest that answers "is this still current?"** Taken over everything that
changes the pixels — prompt, negative, model, steps, size, seed, and the
*contents* of the reference images rather than their paths. A caller can
compare the digest in the manifest against what it would ask for now, and
skip the work when they match. Renaming a reference file does not invalidate
it; editing one does.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def digest(*, profile: str, model: str, prompt: str, negative: str | None,
           steps: int, width: int, height: int, seed: int,
           ref_hashes: list[str]) -> str:
    """A stable fingerprint of everything that determines the output.

    Hashed from a sorted JSON document rather than a concatenated string, so
    adding a field later cannot silently collide with an old digest."""
    doc = {
        "profile": profile, "model": model, "prompt": prompt,
        "negative": negative or "", "steps": steps,
        "width": width, "height": height, "seed": seed,
        "refs": sorted(ref_hashes),
    }
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def output_path(out_dir: Path, stem: str, seed: int, suffix: str = ".png") -> Path:
    """`<out_dir>/<stem>_<seed>.png`, never overwriting an existing file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = out_dir / f"{stem}_{seed}{suffix}"
    if not candidate.exists():
        return candidate
    n = 2
    while (alt := out_dir / f"{stem}_{seed}-{n}{suffix}").exists():
        n += 1
    return alt


def write(image_path: Path, payload: dict) -> Path:
    """The manifest beside the image: `<image>.json`.

    Beside rather than in a central index on purpose — copy the image
    somewhere and its provenance travels with it, which a database row does
    not."""
    path = Path(image_path).with_suffix(Path(image_path).suffix + ".json")
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def read(image_path: Path) -> dict | None:
    path = Path(image_path).with_suffix(Path(image_path).suffix + ".json")
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def build(*, request: dict, profile, digest_value: str, seed: int,
          ref_hashes: list[str], timing: dict, mflux_version: str | None) -> dict:
    """Everything needed to reproduce or audit one image."""
    return {
        "request": request,
        "digest": digest_value,
        "profile": profile.name,
        "model_id": profile.model,
        "variant": profile.variant,
        "mflux_version": mflux_version,
        "quantize": profile.quantize,
        "licence": profile.licence,
        "commercial_ok": profile.commercial_ok,
        "seed": seed,
        "ref_sha256": ref_hashes,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": timing.get("generate_s"),
        "warm_model": timing.get("warm"),
        "host": socket.gethostname(),
        "machine": f"{platform.system()} {platform.machine()}",
        "backend": "local",
    }


def is_current(image_path: Path, expected_digest: str) -> bool:
    """Whether an existing image was made from exactly this request.

    The point of the digest: a caller regenerating a folder can skip what has
    not changed instead of rebuilding everything to be safe."""
    m = read(image_path)
    return bool(m and m.get("digest") == expected_digest)
