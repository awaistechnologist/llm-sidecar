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
    c = s["cache"]
    print(f"cache          : {c['completions']} completions, {c['searches']} searches "
          f"({c['bytes'] / 1024:.0f} KiB)")
    u = s["usage_30d"]
    print(f"last 30 days   : {u['calls']} calls, ${u['cost_usd']:.4f}, {u['total_tokens']:,} tokens")
    print(f"daemon would be: http://{sc.config.daemon_host}:{sc.config.daemon_port}/v1")

    if not s["cloud_configured"] and not s["local_models"]:
        print("\nNothing to route to. Install Ollama or set OPENROUTER_API_KEY.", file=sys.stderr)
        return 1
    return 0


def _models(args) -> int:
    from . import Sidecar

    rows = Sidecar().local_models(context_tokens=args.context)
    if not rows:
        print("No local models found. Is Ollama running?", file=sys.stderr)
        return 1
    mark = {"fits": "✓", "tight": "~", "too_big": "✗", "unknown": "?"}
    print(f"{'':2} {'MODEL':<26} {'SIZE':>7} {'NEEDS':>7}  PARAMS  QUANT")
    for r in rows:
        print(f"{mark[r['verdict']]:2} {r['name']:<26} {r['size_gib']:6.1f}G {r['needs_gib']:6.1f}G"
              f"  {r['parameter_size']:>6}  {r['quantization']}")
    print(f"\n✓ fits  ~ tight  ✗ too big   (at {args.context:,} token context)")
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
        return 0

    if args.action == "clear-key":
        cfg.openrouter_api_key = None
        config_mod.save(cfg)
        print("Key removed from the config file. Local models only.")
        print("  An OPENROUTER_API_KEY in your environment still wins over this.")
        return 0
    return 2


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

    use = sub.add_parser("usage", help="what you have spent, by model")
    use.add_argument("--days", type=int, default=30, help="0 for all time")

    sm = sub.add_parser("sum", help="summarise a file (- for stdin)")
    sm.add_argument("file")
    sm.add_argument("--style", default="brief", choices=["brief", "bullets", "detailed", "tldr"])
    sm.add_argument("--focus", default="")

    ca = sub.add_parser("cache", help="inspect or clear the response cache")
    ca.add_argument("action", nargs="?", default="stats", choices=["stats", "clear"])

    cf = sub.add_parser("config", help="set your API key, budget, and see what's configured")
    cf.add_argument("action", choices=["key", "budget", "show", "clear-key"])
    cf.add_argument("value", nargs="?", default="")
    cf.add_argument("--save", action="store_true",
                    help="write it to the config file (plain text) rather than this process only")

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
