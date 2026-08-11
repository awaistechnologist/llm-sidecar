"""Command line entry point.

    llm-sidecar serve            # OpenAI-compatible daemon on :4001
    llm-sidecar mcp              # MCP server on stdio
    llm-sidecar status           # what's configured and reachable
    llm-sidecar ask "question"   # one-shot completion
    llm-sidecar verify "claim"   # fact-check with citations
"""

from __future__ import annotations

import argparse
import json
import sys


def _status() -> int:
    from . import Sidecar
    from .search import resolve_provider

    sc = Sidecar()
    local = [m.id for m in sc.models() if m.is_local]
    try:
        provider = resolve_provider(sc.config).__name__.rsplit(".", 1)[-1]
    except Exception as e:
        provider = f"unavailable ({e})"

    print(f"cloud key      : {'set' if sc.config.has_cloud else 'not set (local only)'}")
    print(f"budget         : {sc.config.default_budget}")
    print(f"search         : {provider}")
    print(f"ollama         : {sc.config.ollama_host}")
    print(f"local models   : {', '.join(local) if local else 'none found'}")
    print(f"daemon would be: http://{sc.config.daemon_host}:{sc.config.daemon_port}/v1")
    if not sc.config.has_cloud and not local:
        print("\nNothing to route to. Install Ollama or set OPENROUTER_API_KEY.", file=sys.stderr)
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
    if args.cmd == "ask":
        return _ask(args)
    if args.cmd == "verify":
        return _verify(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
