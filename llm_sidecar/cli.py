"""Command line entry point.

    llm-sidecar serve            # OpenAI-compatible daemon on :4001
    llm-sidecar mcp              # MCP server on stdio
    llm-sidecar status           # what's configured and reachable
    llm-sidecar ask "question"   # one-shot completion
    llm-sidecar verify "claim"   # fact-check with citations
    llm-sidecar models           # local models scored against this machine
    llm-sidecar usage            # what you have spent
    llm-sidecar sum FILE         # summarise a file
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
    print(f"search         : {s['search_provider']}")
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


def _summarise(args) -> int:
    from . import Sidecar

    try:
        text = sys.stdin.read() if args.file == "-" else open(args.file).read()
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

    sub.add_parser("serve", help="run the OpenAI-compatible HTTP daemon")
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

    ask = sub.add_parser("ask", help="one-shot completion")
    ask.add_argument("prompt")
    ask.add_argument("--tier", default=None, choices=["fast", "balanced", "powerful"])
    ask.add_argument("--budget", default=None, choices=["free", "cheap", "best"])
    ask.add_argument("--max-tokens", type=int, default=1000, dest="max_tokens")
    ask.add_argument("-q", "--quiet", action="store_true", help="suppress the cost receipt")

    ver = sub.add_parser("verify", help="fact-check claims against live web evidence")
    ver.add_argument("claims", nargs="+")
    ver.add_argument("--json", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "serve":
        from .daemon import main as serve
        serve()
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
    if args.cmd == "verify":
        return _verify(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
