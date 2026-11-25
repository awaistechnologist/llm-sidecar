"""The sidecar from your phone.

Three pieces, the same shape as any other always-on service you reach from a
phone without opening it to the internet:

  1. The daemon stays on loopback. Nothing here binds a wider address.
  2. Tailscale's `serve` puts a private HTTPS address in front of it that
     only devices signed in to your own tailnet can open — at home, on
     mobile data, anywhere. Not `funnel`, which would make it public.
  3. A phone-first web app at /app/, installable from Chrome, which the
     Android app in `android/` wraps.

Two things about (2) shape the code below.

Tailscale proxies to 127.0.0.1, so a request from your phone arrives looking
exactly like one from this machine. That is the intended boundary — your
tailnet is your devices — but it means "came from loopback" can no longer be
read as "came from this machine". Nothing in the daemon does read it that
way; keep it so.

And because the Host header is now the only thing distinguishing those
requests, the daemon checks it. That check was overdue regardless: a
loopback port with no Host check is reachable from any web page you visit
through DNS rebinding — a page on evil.example re-points its own name at
127.0.0.1 and its scripts become same-origin with this daemon, which holds
an API key and spends money on request.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# The HTTPS port on the tailnet name. Not 443: that is the port a Tailscale
# user is most likely to already be serving something else on, and `serve`
# replaces whatever was there. A port of our own cannot clobber anything.
DEFAULT_HTTPS_PORT = 8443

# Hosts that mean "this machine" however the caller spelt it, plus the names
# Docker Desktop and Podman give the host from inside a container — calling
# a host-side sidecar from a container is the commonest non-local use, and
# it already worked before this check existed.
LOCAL_HOSTS = {
    "localhost", "127.0.0.1", "::1", "[::1]",
    "host.docker.internal", "host.containers.internal",
}

# Bind addresses that mean "every interface". Someone who chose one of these
# has decided the daemon is reachable by name from elsewhere, so a Host
# allowlist would only break the clients they set it up for.
WILDCARD_BINDS = {"0.0.0.0", "::", "[::]"}

# Where images generated from the phone land. Under the data directory, not
# ~/Pictures or ~/Documents: a service started by launchd on macOS may have
# no access to those privacy-protected folders, and then every generation
# would fail at the very last step.
def image_dir() -> Path:
    from .ledger import DATA_DIR
    d = DATA_DIR / "phone-images"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Host header check ─────────────────────────────────────────────────────


def _bare(host: str) -> str:
    """'name:8443' -> 'name'; '[::1]:4001' -> '::1'. Lower-cased."""
    h = (host or "").strip().lower()
    if h.startswith("["):
        return h[1:h.index("]")] if "]" in h else h
    if h.count(":") == 1:
        h = h.split(":", 1)[0]
    return h.rstrip(".")


def allowed_host(host: str, config) -> bool:
    """Whether to answer a request carrying this Host header.

    *.ts.net is allowed because those names resolve only inside your own
    tailnet; a rebinding page cannot point one at your loopback, since it
    does not control any ts.net name."""
    if (config.daemon_host or "").strip().lower() in WILDCARD_BINDS:
        return True
    h = _bare(host)
    if not h:
        # HTTP/1.0 with no Host. No browser sends that, so it cannot be a
        # rebinding attack, and refusing it would break crude clients.
        return True
    if h in LOCAL_HOSTS or h == _bare(config.daemon_host or ""):
        return True
    if h.endswith(".ts.net"):
        return True
    return h in {_bare(x) for x in (config.allowed_hosts or [])}


def install_host_guard(app, config) -> None:
    from starlette.responses import JSONResponse

    @app.middleware("http")
    async def host_guard(request, call_next):
        host = request.headers.get("host", "")
        if not allowed_host(host, config):
            logger.warning("refused request with Host %r", host)
            return JSONResponse(status_code=403, content={"detail": (
                f"This sidecar does not answer to the name {_bare(host)!r}. "
                "It answers to localhost, *.ts.net and the Docker host "
                "aliases; add others with LLM_SIDECAR_HOSTS=name1,name2. "
                "(The check stops web pages reaching it by DNS rebinding.)"
            )})
        return await call_next(request)


# ── the phone app's routes ────────────────────────────────────────────────

UI = Path(__file__).parent / "ui"
# Only these files are served from ui/, by exact name. A path parameter into
# a directory is a traversal bug waiting for someone to add a file.
STATIC = {
    "manifest.webmanifest": "application/manifest+json",
    "sw.js": "text/javascript",
    "icon-192.png": "image/png",
    "icon-512.png": "image/png",
    "icon-maskable-512.png": "image/png",
    "apple-touch-icon.png": "image/png",
}
_IMAGE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,120}\.png$")


from pydantic import BaseModel  # noqa: E402


class PhoneImage(BaseModel):
    # At module level, not inside install_routes: this module uses
    # `from __future__ import annotations`, so FastAPI resolves the route's
    # annotation by name from module globals. Defined in the function, it
    # could not find it and read the body as a query parameter — every
    # generation from the phone failed with a 422.
    prompt: str
    profile: str = "standard"
    negative: str | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None


def install_routes(app, cfg, sidecar, require_token, op) -> None:
    from fastapi import Depends, Header, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

    def page():
        if not cfg.ui_enabled:
            raise HTTPException(status_code=404, detail="The UI is disabled.")
        p = UI / "app.html"
        # no-cache so a phone that installed the app picks up a new version
        # on its next open rather than whenever the service worker decides.
        return HTMLResponse(p.read_text(encoding="utf-8"),
                            headers={"Cache-Control": "no-cache"})

    @app.get("/app", include_in_schema=False)
    def app_redirect():
        # The service worker's scope is /app/, and /app is outside it. One
        # canonical address keeps "installed" and "in the browser" the same.
        return RedirectResponse("/app/", status_code=308)

    @app.get("/app/", include_in_schema=False)
    def app_page():
        return page()

    @app.get("/app/{name}", include_in_schema=False)
    def app_static(name: str):
        if name not in STATIC or not cfg.ui_enabled:
            raise HTTPException(status_code=404)
        headers = {"Cache-Control": "no-cache"}
        if name == "sw.js":
            # Lets the worker control /app/ even though it is served from it.
            headers["Service-Worker-Allowed"] = "/app/"
        return FileResponse(UI / name, media_type=STATIC[name], headers=headers)

    @app.get("/app/api/engines")
    def engines(_: None = Depends(require_token)) -> dict:
        """Everything the phone needs to draw its model picker, in one call.

        Deliberately cheap: no model is probed and the Claude CLI is not
        run. /claude-cli/status costs a real model call to check sign-in,
        and a picker that spends money to open is a bad picker."""
        from . import claude_cli

        local, cloud = [], []
        for m in sidecar.models():
            if m.is_local:
                local.append(m.id)
            elif not m.id.startswith("claude/"):
                cloud.append(m.id)
        claude = ([m.id for m in claude_cli.models(cfg)]
                  if cfg.claude_cli_enabled and claude_cli.binary(cfg) else [])

        images: dict = {"available": False}
        try:
            from . import images as images_mod
            from .images import profiles as profiles_mod
            st = images_mod.status()
            images = {
                "available": bool(st.get("available")),
                "busy": (st.get("queue") or {}).get("busy", False),
                "waiting": (st.get("queue") or {}).get("waiting", 0),
                "profiles": [
                    {k: p.get(k) for k in ("name", "note", "measured_s", "size",
                                           "max_refs", "supports_negative",
                                           "commercial_ok", "gated")}
                    for p in profiles_mod.describe()
                    # Reference and upscale profiles need an image to start
                    # from, which the phone app does not send yet.
                    if not p.get("max_refs") and p.get("name") != "upscale"
                ],
            }
        except Exception as e:
            images = {"available": False, "detail": str(e)[:200]}

        return {
            "version": _version(),
            "local": local,
            "claude": claude,
            "cloud": {"configured": cfg.has_cloud, "count": len(cloud)},
            "budget": cfg.default_budget,
            "images": images,
        }

    @app.post("/app/api/image")
    def phone_image(req: PhoneImage, _: None = Depends(require_token)) -> dict:
        """Generate into the phone gallery and return a URL for it.

        A separate route from /v1/images/generate because that one takes an
        out_dir from the caller, and a phone has no business choosing where
        on this machine a file is written."""
        from . import images as images_mod

        stem = re.sub(r"[^a-z0-9]+", "-", req.prompt.lower()).strip("-")[:40] or "image"
        try:
            r = images_mod.generate(images_mod.ImageRequest(
                prompt=req.prompt, profile=req.profile, negative=req.negative,
                width=req.width, height=req.height, seed=req.seed,
                out_dir=str(image_dir()), stem=f"{time.strftime('%Y%m%d-%H%M%S')}-{stem}",
                project="phone", skip_existing=False,
            ), allow_daemon=False)
        except images_mod.ImageUnavailable as e:
            raise HTTPException(status_code=501, detail=str(e)) from e
        except (images_mod.LicenceError, images_mod.ProfileError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except images_mod.OutOfMemory as e:
            raise HTTPException(status_code=507, detail=str(e)) from e
        except (images_mod.MemoryBusy, images_mod.EngineFault) as e:
            raise HTTPException(status_code=503, detail=str(e),
                                headers={"Retry-After": str(e.retry_after_s)}) from e
        except Exception as e:
            logger.exception("phone image failed")
            raise HTTPException(status_code=500, detail=str(e)) from e
        name = Path(r.path).name
        return {"url": f"/app/images/{name}", "name": name, "seconds": r.seconds,
                "model": r.model, "profile": r.profile, "seed": r.seed,
                "warm": r.warm, "width": r.width, "height": r.height}

    @app.get("/app/api/gallery")
    def gallery(limit: int = 60, _: None = Depends(require_token)) -> dict:
        files = sorted(image_dir().glob("*.png"), key=lambda p: p.stat().st_mtime,
                       reverse=True)[:max(1, min(limit, 500))]
        out = []
        for f in files:
            prompt = ""
            try:
                meta = json.loads(Path(str(f) + ".json").read_text(encoding="utf-8"))
                prompt = (meta.get("request") or {}).get("prompt", "") or meta.get("prompt", "")
            except Exception:
                pass
            out.append({"url": f"/app/images/{f.name}", "name": f.name,
                        "prompt": prompt, "at": f.stat().st_mtime})
        return {"images": out}

    @app.get("/app/images/{name}", include_in_schema=False)
    def phone_image_file(name: str, t: str = "", authorization: str | None = Header(default=None)):
        # An <img> tag cannot send an Authorization header, so when a token
        # is configured the page appends it as ?t=. Accepted for image files
        # only, where the alternative is images that never load.
        if cfg.daemon_token and authorization != f"Bearer {cfg.daemon_token}" \
                and t != cfg.daemon_token:
            raise HTTPException(status_code=401, detail="Invalid or missing token.")
        # A strict pattern rather than resolving and checking the parent:
        # no separators, no dot-dot, png only. Nothing to get clever with.
        if not _IMAGE_NAME.match(name) or ".." in name:
            raise HTTPException(status_code=404)
        f = image_dir() / name
        if not f.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(f, media_type="image/png",
                            headers={"Cache-Control": "private, max-age=31536000, immutable"})


def _version() -> str:
    from . import __version__
    return __version__


# ── `llm-sidecar phone` ───────────────────────────────────────────────────

TAILSCALE_PATHS = (
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
    "/usr/bin/tailscale", "/usr/local/bin/tailscale", "/opt/homebrew/bin/tailscale",
)


def tailscale_binary() -> str | None:
    found = shutil.which("tailscale")
    if found:
        return found
    for p in TAILSCALE_PATHS:
        if Path(p).exists():
            return p
    return None


def _ts(ts: str, *args: str, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run([ts, *args], capture_output=True, text=True, timeout=timeout)


def tailnet_name(ts: str) -> str | None:
    r = _ts(ts, "status", "--json")
    if r.returncode != 0:
        return None
    try:
        return (json.loads(r.stdout).get("Self") or {}).get("DNSName", "").rstrip(".") or None
    except ValueError:
        return None


def funnel_exposed(ts: str, port: int) -> bool:
    """Whether this HTTPS port is on Funnel, i.e. public on the internet.

    Checked after setting up, because `serve` would quietly keep a Funnel
    someone had already enabled on the same port — and an unauthenticated
    daemon holding an API key must never be one."""
    r = _ts(ts, "serve", "status", "--json")
    if r.returncode != 0 or not r.stdout.strip():
        return False
    try:
        allow = json.loads(r.stdout).get("AllowFunnel") or {}
    except ValueError:
        return False
    return any(v and k.endswith(f":{port}") for k, v in allow.items())


def _qr(text: str) -> str | None:
    try:
        import qrcode  # optional; not a dependency
    except ImportError:
        return None
    q = qrcode.QRCode(border=1)
    q.add_data(text)
    q.make(fit=True)
    import io
    buf = io.StringIO()
    q.print_ascii(out=buf, invert=True)
    return buf.getvalue()


def run(action: str, config, https_port: int = DEFAULT_HTTPS_PORT, out=print) -> int:
    import httpx

    ts = tailscale_binary()
    if not ts:
        out("Tailscale isn't installed. It is what makes the sidecar reachable from")
        out("your phone without exposing it to your Wi-Fi or the internet.")
        out("")
        out("  macOS : brew install --cask tailscale   (or the Mac App Store)")
        out("  Linux : https://tailscale.com/download")
        out("")
        out("Sign in, install Tailscale on the phone with the same account, then")
        out("run this again.")
        return 1

    target = f"http://127.0.0.1:{config.daemon_port}"

    if action == "off":
        r = _ts(ts, "serve", f"--https={https_port}", "off")
        if r.returncode != 0:
            out((r.stderr or r.stdout).strip() or "tailscale serve off failed")
            return 1
        out(f"Stopped serving the sidecar on port {https_port} of your tailnet.")
        return 0

    name = tailnet_name(ts)
    if not name:
        out("Tailscale is installed but not signed in, or not running.")
        out("Open Tailscale, sign in, and run this again.")
        return 1
    url = f"https://{name}:{https_port}/app/"

    if action == "status":
        r = _ts(ts, "serve", "status")
        serving = f":{https_port}" in (r.stdout or "") and target.split("//")[1] in (r.stdout or "")
        out(f"tailnet name : {name}")
        out(f"phone app    : {url if serving else '(not serving — run: llm-sidecar phone)'}")
        if serving and funnel_exposed(ts, https_port):
            out("WARNING      : this port is on Funnel — PUBLIC. Run `llm-sidecar phone off`.")
        return 0

    try:
        ok = httpx.get(f"{target}/health", timeout=3).status_code == 200
    except Exception:
        ok = False
    if not ok:
        out(f"The sidecar isn't answering on {target}.")
        out("Start it first, ideally as a service so it is up whenever the machine is:")
        out("")
        out("  llm-sidecar service install")
        return 1

    r = _ts(ts, "serve", "--bg", f"--https={https_port}", target)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).strip()
        out("tailscale serve failed:")
        out("  " + msg.replace("\n", "\n  "))
        if "https" in msg.lower() and "enable" in msg.lower():
            out("")
            out("HTTPS certificates are probably off for your tailnet. Enable them in")
            out("the Tailscale admin console (DNS → HTTPS Certificates), then retry.")
        return 1

    if funnel_exposed(ts, https_port):
        _ts(ts, "serve", f"--https={https_port}", "off")
        out(f"Refused: port {https_port} has Tailscale Funnel enabled, which would make")
        out("the sidecar public on the internet. I have switched it back off.")
        out("Turn Funnel off for that port, or pick another with --port.")
        return 1

    out("")
    out("The sidecar is on your tailnet. Only devices signed in to your own")
    out("Tailscale account can open it — not your Wi-Fi, not the internet.")
    out("")
    out(f"  {url}")
    out("")
    qr = _qr(url)
    if qr:
        out(qr)
    out("On the phone, with Tailscale connected:")
    out("  • Chrome: open the address, then ⋮ → Install app. It appears in the")
    out("    Share menu, so text and links from any app can be sent to it.")
    out("  • Android app (mobile/ in the repo): open the address once in Chrome")
    out("    and tap \"Use in the Android app\", or paste it on the app's first screen.")
    if not qr:
        out("")
        out("(pip install qrcode to get a scannable code printed here.)")
    if not config.daemon_token:
        out("")
        out("Anyone in your tailnet can use it. If you share devices in your tailnet")
        out("with other people, set LLM_SIDECAR_TOKEN so the app asks for it.")
    return 0
