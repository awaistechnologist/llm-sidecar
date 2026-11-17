"""Command line entry point.

    llm-sidecar serve            # OpenAI-compatible daemon on :4001
    llm-sidecar mcp              # MCP server on stdio
    llm-sidecar status           # what's configured and reachable
    llm-sidecar ask "question"   # one-shot completion
    llm-sidecar answer "question"  # grounded answer with citations
    llm-sidecar verify "claim"   # fact-check with citations
    llm-sidecar models           # local models scored against this machine
    llm-sidecar usage            # what you have spent
    llm-sidecar sum FILE         # summarise a file
    llm-sidecar searxng up       # start a local SearXNG for better search
    llm-sidecar config key KEY   # save your OpenRouter key
    llm-sidecar service install  # run at login, restart on failure
"""

from __future__ import annotations

import argparse
import json
import sys


def _status() -> int:
    from . import Sidecar

    sc = Sidecar()
    s = sc.status()
    hw = s["hardware"]
    usable = ", ".join(
        f"{m['id'].split('/', 1)[1]}" for m in s["local_models"] if m["verdict"] == "fits"
    )

    print(f"version        : {s['version']}")
    print(f"cloud key      : {'set' if s['cloud_configured'] else 'not set (local only)'}")
    print(f"budget         : {s['default_budget']}")
    hint = "" if s["search_provider"] == "searxng" else "   (`llm-sidecar searxng up` for better search)"
    print(f"search         : {s['search_provider']}{hint}")
    print(f"machine        : {hw.get('chip') or hw['system']} · {hw.get('total_ram_gib', '?')} GiB")
    print(f"local (fits)   : {usable or 'none'}")
    tight = [m['id'] for m in s['local_models'] if m['verdict'] != 'fits']
    if tight:
        print(f"local (won't)  : {', '.join(t.split('/', 1)[1] for t in tight)}")
    cc = s.get("claude_cli") or {}
    if cc.get("enabled"):
        print(f"claude/ (CLI)  : on · {', '.join(cc.get('models') or []) or 'no binary found'}")
    elif cc.get("installed"):
        print("claude/ (CLI)  : installed but off   "
              "(`llm-sidecar config claude-cli on` to use your subscription)")
    c = s["cache"]
    print(f"cache          : {c['completions']} completions, {c['searches']} searches "
          f"({c['bytes'] / 1024:.0f} KiB)")
    u = s["usage_30d"]
    print(f"last 30 days   : {u['calls']} calls, ${u['cost_usd']:.4f}, {u['total_tokens']:,} tokens")
    print(f"daemon would be: http://{sc.config.daemon_host}:{sc.config.daemon_port}/v1")

    if not s["cloud_configured"] and not s["local_models"] and not cc.get("models"):
        print("\nNothing to route to. Any one of these fixes it:", file=sys.stderr)
        print("  install Ollama                      (local, free)", file=sys.stderr)
        print("  set OPENROUTER_API_KEY              (cloud)", file=sys.stderr)
        print("  llm-sidecar config claude-cli on    (your Claude subscription)",
              file=sys.stderr)
        return 1
    return 0


def _models(args) -> int:
    from . import Sidecar

    rows = Sidecar().local_models(context_tokens=args.context)

    if not rows or args.suggest:
        from . import hardware

        if not rows:
            print("No local models installed.\n")
        picks = hardware.suggest(context_tokens=args.context)
        if not picks:
            print("Could not read this machine's memory, so no suggestions.", file=sys.stderr)
            return 1
        hw = hardware.probe()
        print(f"Your machine ({hw.get('chip') or hw['system']}, "
              f"{hw.get('total_ram_gib', '?')} GiB) could run:\n")
        mark = {"fits": "✓", "tight": "~"}
        for r in picks:
            print(f"  {mark.get(r['verdict'], '?')} {r['name']:<20} {r['size_gib']:5.1f}G  {r['why']}")
        print(f"\n  ollama pull {picks[0]['name']}")
        print("\n✓ comfortable   ~ tight, will be slow under memory pressure")
        return 0 if rows else 1
    mark = {"fits": "✓", "tight": "~", "too_big": "✗", "unknown": "?"}
    print(f"{'':2} {'MODEL':<26} {'SIZE':>7} {'NEEDS':>7}  {'PARAMS':>6}  {'QUANT':<8} STATE")
    for r in rows:
        state = "loaded" if r.get("loaded") else ""
        print(f"{mark[r['verdict']]:2} {r['name']:<26} {r['size_gib']:6.1f}G {r['needs_gib']:6.1f}G"
              f"  {r['parameter_size']:>6}  {r['quantization']:<8} {state}")
    print(f"\n✓ fits  ~ tight  ✗ too big   (at {args.context:,} token context)")
    print("All of them are usable. \"loaded\" means it is in memory now and answers")
    print("immediately; the rest cost a load first. Ollama unloads after ~5 idle minutes.")
    return 0


def _usage(args) -> int:
    from . import ledger

    s = ledger.summary(args.days)
    if not s["calls"]:
        print("No usage recorded yet.")
        return 0
    window = f"last {args.days} days" if args.days else "all time"
    print(f"{window}: {s['calls']} calls · ${s['cost_usd']:.4f} · "
          f"{s['total_tokens']:,} tokens · {s['cached']} served from cache\n")
    print(f"{'MODEL':<40} {'CALLS':>6} {'COST':>10} {'TOKENS':>10} {'AVG':>7}")
    for m in s["by_model"]:
        print(f"{m['model'][:40]:<40} {m['calls']:>6} {m['cost_usd']:>10.4f} "
              f"{m['total_tokens']:>10,} {m['avg_latency_s']:>6.1f}s")
    return 0


def _config(args) -> int:
    """Set the key without needing the daemon running, or curl."""
    from . import config as config_mod

    cfg = config_mod.load()

    if args.action == "key":
        key = (args.value or "").strip()
        if not key:
            print("Pass the key: llm-sidecar config key sk-or-…", file=sys.stderr)
            return 1
        cfg.openrouter_api_key = key
        if args.save:
            path = config_mod.save(cfg, include_api_key=True)
            print(f"Saved to {path}")
            print("  Plain text, readable by anything running as you.")
        else:
            print("Key accepted but NOT saved — this process is about to exit, "
                  "so that achieved nothing.")
            print()
            print("  Persist it:   llm-sidecar config key <KEY> --save")
            print("  Or per-shell: export OPENROUTER_API_KEY=<KEY>")
            return 1
        return 0

    if args.action == "budget":
        if args.value not in ("free", "cheap", "best"):
            print("Budget must be free, cheap or best.", file=sys.stderr)
            return 1
        cfg.default_budget = args.value
        print(f"Saved to {config_mod.save(cfg)}")
        return 0

    if args.action == "show":
        key = cfg.openrouter_api_key
        print(f"config file    : {config_mod.CONFIG_FILE}")
        print(f"openrouter key : {'…' + key[-4:] if key else 'not set'}")
        print(f"budget         : {cfg.default_budget}")
        print(f"ollama         : {cfg.ollama_host}")
        print(f"search         : {cfg.search_provider}")
        print(f"locked tiers   : {cfg.models or 'none'}")
        print(f"claude/ (CLI)  : {'on' if cfg.claude_cli_enabled else 'off'}")
        return 0

    if args.action == "claude-cli":
        from . import claude_cli

        want = (args.value or "").strip().lower()
        if want not in ("on", "off", ""):
            print("Use: llm-sidecar config claude-cli on|off", file=sys.stderr)
            return 1

        if want:
            cfg.claude_cli_enabled = (want == "on")
            print(f"Saved to {config_mod.save(cfg)}")

        info = claude_cli.check(cfg)
        print(f"claude binary  : {info['binary'] or 'not found'}")
        print(f"version        : {info['version'] or 'unknown'}")
        print(f"enabled        : {'yes' if info['enabled'] else 'no'}")
        if info["authenticated"] is not None:
            print(f"signed in      : {'yes' if info['authenticated'] else 'no'}")
        print(f"               : {info['detail']}")
        if info["enabled"] and info["authenticated"]:
            print()
            print("Use it by naming a model:")
            print("  llm-sidecar ask --model claude/sonnet \"...\"")
            print("Or make it a tier default, in the dashboard or the environment:")
            print("  export LLM_SIDECAR_MODEL_POWERFUL=claude/opus")
        return 0 if (info["binary"] and (not info["enabled"] or info["authenticated"]))\
            else 1

    if args.action == "clear-key":
        cfg.openrouter_api_key = None
        # Explicit, because save() now preserves an on-disk key by default —
        # removing it has to be asked for, not implied.
        config_mod.save(cfg, clear_api_key=True)
        print("Key removed from the config file. Local models only.")
        print("  An OPENROUTER_API_KEY in your environment still wins over this.")
        return 0
    return 2


def _image(args) -> int:
    from . import images as im
    from .images import engine, profiles as pf

    if args.action == "profiles":
        print(f"{'PROFILE':<24} {'STEPS':>5} {'REFS':>4} {'PEAK':>6}  {'COMMERCIAL':<10} LICENCE")
        for p in pf.describe():
            print(f"{p['name']:<24} {p['steps']:>5} {p['max_refs']:>4} "
                  f"{p['peak_gib']:>5.0f}G  {'yes' if p['commercial_ok'] else 'NO':<10} {p['licence']}")
        print()
        print("Ask for a profile, not a model. Each one fixes the model, steps,")
        print("quantisation and licence together, so upgrading it upgrades every caller.")
        return 0

    if args.action == "status":
        s = engine.status()
        if not s["available"]:
            print("Image generation is not installed.")
            print('  pip install "llm-sidecar[images]"      (Apple Silicon only)')
            return 1
        print(f"mflux          : {s['mflux']}")
        print(f"resident model : {s['loaded'] or 'none'}")
        if s["loaded"]:
            print(f"idle           : {s['idle_s']:.0f}s "
                  f"(unloads after {engine.IDLE_UNLOAD_S // 60:.0f} min)")
        print(f"usable memory  : {s['usable_gib']} GiB")
        return 0

    if args.action == "bench":
        return _image_bench(args)

    if not args.prompt and args.profile != "upscale":
        print("Give a prompt: llm-sidecar image gen \"a red apple\" --out ./pics",
              file=sys.stderr)
        return 1

    try:
        r = im.generate(im.ImageRequest(
            prompt=args.prompt, profile=args.profile, out_dir=args.out,
            stem=args.stem, refs=args.refs, negative=args.negative or None,
            width=args.width, height=args.height, steps=args.steps,
            seed=args.seed, resolution=args.resolution, project=args.project,
            commercial=not args.non_commercial,
        ))
    except im.ImageError as e:
        print(str(e), file=sys.stderr)
        return 1

    if r.skipped:
        print(f"up to date, skipped: {r.path}")
        return 0
    print(f"{r.path}")
    print(f"  {r.profile} · {r.model} · seed {r.seed} · {r.seconds:.1f}s"
          f"{'' if r.warm else '  (cold model — the next one is faster)'}")
    print(f"  manifest: {r.manifest_path}")
    return 0


def _image_bench(args) -> int:
    """Measure what keeping a model resident actually saves, on this machine.

    The whole argument for a shared service over `mflux` per subprocess is
    that reloading costs real time. That is a number, not an opinion, so it
    gets measured rather than asserted."""
    import tempfile

    from . import images as im
    from .images import engine

    try:
        engine.require()
    except im.ImageError as e:
        print(str(e), file=sys.stderr)
        return 1

    profile = args.profile
    prompt = args.prompt or "a red apple on a wooden table, soft daylight"
    with tempfile.TemporaryDirectory() as tmp:
        engine.RESIDENT.unload()
        rows = []
        for i, label in enumerate(("cold", "warm", "warm")):
            r = im.generate(im.ImageRequest(
                prompt=prompt, profile=profile, out_dir=tmp, stem=f"bench{i}",
                width=args.width or 512, height=args.height or 512,
                seed=100 + i, project="bench",
            ))
            rows.append((label, r.seconds))
            print(f"  {label:<5} {r.seconds:6.1f}s")

    cold = rows[0][1]
    warm = sum(s for l, s in rows[1:] if l == "warm") / max(1, len(rows) - 1)
    print()
    print(f"profile {profile}: cold {cold:.1f}s, warm {warm:.1f}s "
          f"— keeping it resident saves {cold - warm:.1f}s per image.")
    return 0


def _service(args) -> int:
    from . import service
    from .types import SidecarError

    try:
        if args.action == "install":
            r = service.install(port=args.port)
            print(f"Installed as a {r['manager']} service on port {r['port']}.")
            print(f"  definition : {r['path']}")
            if r.get("log"):
                print(f"  logs       : {r['log']}")
            if r.get("note"):
                print(f"  note       : {r['note']}")
            print()
            print("It starts at login and restarts if it dies.")
            print(f"  check: curl -s localhost:{r['port']}/health")
        elif args.action == "uninstall":
            r = service.uninstall()
            print("Removed." if r["removed"] else "Nothing was installed.")
        else:
            s = service.status()
            print(f"platform  : {s['platform']}")
            print(f"installed : {'yes' if s['installed'] else 'no'}  ({s['path']})")
            print(f"running   : {'yes' if s['running'] else 'no'}")
            print(f"logs      : {s['log']}")
            if not s["installed"]:
                print()
                print("  Install it: llm-sidecar service install")
    except SidecarError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _searxng(args) -> int:
    from . import config as config_mod, services
    from .types import SidecarError

    cfg = config_mod.load()

    if args.action == "status":
        s = services.status(cfg)
        answering = "yes" if s["answering_json"] else "no"
        print(f"url            : {s['url']}")
        print(f"answering JSON : {answering}")
        print(f"docker         : {s['docker']}")
        print(f"config written : {'yes' if s['installed'] else 'no'} ({s['instance_dir']})")
        if s["container"]:
            print(f"container      : {s['container']['state']} — {s['container']['status']}")
        else:
            print("container      : not created")
        if s["answering_json"]:
            print("\nSearXNG is being used for search automatically.")
        else:
            print("\nSearch is falling back to DuckDuckGo. Run `llm-sidecar searxng up`.")
        return 0

    try:
        if args.action == "up":
            print("Starting SearXNG (first run pulls the image, this can take a minute)...")
            r = services.up(cfg, port=args.port)
            print(f"\nReady at {r['url']}")
            print(f"Config:  {r['dir']}")
            print("\nNothing else to do — search auto-detects it. Verify with "
                  "`llm-sidecar searxng status`.")
            if cfg.searxng_url.rstrip("/") != r["url"]:
                print(f"\nNote: your configured searxng_url is {cfg.searxng_url}, but the "
                      f"instance is on {r['url']}. Set SEARXNG_URL={r['url']} so it's found.")
        elif args.action == "down":
            services.down(remove=args.volumes)
            print("Stopped. Search falls back to DuckDuckGo.")
        elif args.action == "logs":
            print(services.logs(args.lines) or "(no output)")
    except SidecarError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _summarise(args) -> int:
    from . import Sidecar

    try:
        text = (sys.stdin.read() if args.file == "-"
                else open(args.file, encoding="utf-8").read())
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    try:
        print(Sidecar().summarise(text, style=args.style, focus=args.focus))
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def _ask(args) -> int:
    from . import Sidecar

    sc = Sidecar()
    try:
        c = sc.complete(args.prompt, tier=args.tier, budget=args.budget, max_tokens=args.max_tokens)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(c.text)
    if not args.quiet:
        cost = "free" if c.usage.cost_usd == 0 else f"${c.usage.cost_usd:.6f}"
        print(f"\n— {c.model} · {c.usage.total_tokens} tokens · {cost}", file=sys.stderr)
    return 0


def _answer(args) -> int:
    from . import Sidecar

    try:
        a = Sidecar().answer(" ".join(args.question),
                             max_sources=args.sources, via=args.via)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(a.__dict__, indent=2))
        return 0 if a.grounded else 1

    if not a.grounded:
        print("Not answered from the sources.\n", file=sys.stderr)
    print(a.text)
    if a.caveat:
        print(f"\nCaveat: {a.caveat}")
    if a.sources:
        print("\nSources:")
        for s in a.sources:
            print(f"  · {s}")
    # Non-zero when ungrounded, so a script can tell "I don't know" from an
    # answer without parsing prose.
    return 0 if a.grounded else 1


def _verify(args) -> int:
    from . import Sidecar

    sc = Sidecar()
    try:
        results = sc.verify(args.claims)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2))
        return 0

    marks = {"supported": "✓", "contradicted": "✗", "unverified": "?", "not_a_claim": "—"}
    for r in results:
        print(f"{marks.get(r.verdict, '?')} [{r.verdict}] {r.claim}")
        if r.note:
            print(f"    {r.note}")
        for s in r.sources[:3]:
            print(f"    · {s}")
        print()
    # Non-zero when something was actually contradicted, so this is usable in
    # a pipeline or a pre-commit hook.
    return 1 if any(r.verdict == "contradicted" for r in results) else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="llm-sidecar", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    srv = sub.add_parser("serve", help="run the OpenAI-compatible HTTP daemon")
    srv.add_argument("--no-ui", action="store_true",
                     help="serve the API only, without the dashboard")
    srv.add_argument("--port", type=int, default=None)
    sub.add_parser("mcp", help="run the MCP server on stdio")
    sub.add_parser("status", help="show what is configured and reachable")

    mods = sub.add_parser("models", help="local models scored against this machine's memory")
    mods.add_argument("--context", type=int, default=8000, help="context length to budget for")
    mods.add_argument("--suggest", action="store_true",
                      help="what to pull, based on your memory, rather than what you have")

    use = sub.add_parser("usage", help="what you have spent, by model")
    use.add_argument("--days", type=int, default=30, help="0 for all time")

    sm = sub.add_parser("sum", help="summarise a file (- for stdin)")
    sm.add_argument("file")
    sm.add_argument("--style", default="brief", choices=["brief", "bullets", "detailed", "tldr"])
    sm.add_argument("--focus", default="")

    ca = sub.add_parser("cache", help="inspect or clear the response cache")
    ca.add_argument("action", nargs="?", default="stats", choices=["stats", "clear"])

    cf = sub.add_parser("config", help="set your API key, budget, and see what's configured")
    cf.add_argument("action",
                    choices=["key", "budget", "show", "clear-key", "claude-cli"])
    cf.add_argument("value", nargs="?", default="")
    cf.add_argument("--save", action="store_true",
                    help="write it to the config file (plain text) rather than this process only")

    img = sub.add_parser("image", help="generate images locally (Apple Silicon)")
    img.add_argument("action", nargs="?", default="status",
                     choices=["gen", "profiles", "status", "bench"])
    img.add_argument("prompt", nargs="?", default="")
    img.add_argument("--profile", default="draft")
    img.add_argument("--out", default=".", help="output directory")
    img.add_argument("--stem", default="image", help="filename stem")
    img.add_argument("--ref", action="append", dest="refs", default=[],
                     help="reference image (repeatable, for consistency profiles)")
    img.add_argument("--negative", default="")
    img.add_argument("--width", type=int)
    img.add_argument("--height", type=int)
    img.add_argument("--steps", type=int)
    img.add_argument("--seed", type=int)
    img.add_argument("--resolution", type=int, help="upscale: target long edge")
    img.add_argument("--project", default="", help="recorded in the manifest")
    img.add_argument("--non-commercial", action="store_true",
                     help="permit non-commercial profiles (cinematic, kontext-edit)")

    sv = sub.add_parser("service", help="run the daemon at login (launchd / systemd)")
    sv.add_argument("action", nargs="?", default="status",
                    choices=["install", "uninstall", "status"])
    sv.add_argument("--port", type=int, default=4001)

    sx = sub.add_parser("searxng", help="run a local SearXNG for better search")
    sx.add_argument("action", nargs="?", default="status",
                    choices=["up", "down", "status", "logs"])
    sx.add_argument("--port", type=int, default=None, help="host port (default 8888)")
    sx.add_argument("--volumes", action="store_true", help="with down: also remove volumes")
    sx.add_argument("--lines", type=int, default=50, help="with logs: how many lines")

    ask = sub.add_parser("ask", help="one-shot completion")
    ask.add_argument("prompt")
    ask.add_argument("--tier", default=None, choices=["fast", "balanced", "powerful"])
    ask.add_argument("--budget", default=None, choices=["free", "cheap", "best"])
    ask.add_argument("--max-tokens", type=int, default=1000, dest="max_tokens")
    ask.add_argument("-q", "--quiet", action="store_true", help="suppress the cost receipt")

    ans = sub.add_parser("answer", help="answer a question from live sources, with citations")
    ans.add_argument("question", nargs="+")
    ans.add_argument("--sources", type=int, default=4)
    ans.add_argument("--via", default="local", choices=["local", "openrouter"],
                     help="openrouter retrieves on their side — better sources, but billed")
    ans.add_argument("--json", action="store_true")

    ver = sub.add_parser("verify", help="fact-check claims against live web evidence")
    ver.add_argument("claims", nargs="+")
    ver.add_argument("--json", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "serve":
        from .daemon import main as serve
        serve(no_ui=args.no_ui, port=args.port)
        return 0
    if args.cmd == "mcp":
        from .mcp_server import main as mcp_main
        mcp_main()
        return 0
    if args.cmd == "status":
        return _status()
    if args.cmd == "models":
        return _models(args)
    if args.cmd == "usage":
        args.days = args.days or None
        return _usage(args)
    if args.cmd == "sum":
        return _summarise(args)
    if args.cmd == "image":
        return _image(args)
    if args.cmd == "config":
        return _config(args)
    if args.cmd == "service":
        return _service(args)
    if args.cmd == "searxng":
        return _searxng(args)
    if args.cmd == "cache":
        from . import cache
        if args.action == "clear":
            print(f"Removed {cache.clear()} cached entries.")
        else:
            s = cache.stats()
            print(f"{s['completions']} completions, {s['searches']} searches, "
                  f"{s['bytes'] / 1024:.0f} KiB at {s['path']}")
        return 0
    if args.cmd == "ask":
        return _ask(args)
    if args.cmd == "answer":
        return _answer(args)
    if args.cmd == "verify":
        return _verify(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
