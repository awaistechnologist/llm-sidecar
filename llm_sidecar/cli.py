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
    llm-sidecar searxng install  # run SearXNG natively at login, for better search
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
    hint = "" if s["search_provider"] == "searxng" else "   (`llm-sidecar searxng install` for better search)"
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
    ca = s.get("claude_api") or {}
    if ca.get("enabled"):
        print(f"claude-api/    : on · {ca.get('detail', '')}")
    elif ca.get("credentials"):
        print("claude-api/    : credentials found but off   "
              "(`llm-sidecar config claude-api on` to use API credits)")
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


def _allowances(args) -> int:
    import json as _json
    from . import Sidecar

    r = Sidecar().allowances(refresh=args.refresh)
    if args.json:
        print(_json.dumps(r, indent=2, default=str))
        return 0
    mark = {"ok": "ok  ", "warn": "low ", "exhausted": "OUT ", "off": "off ", "unknown": "?   "}
    for e in r["providers"]:
        print(f"{mark.get(e.get('state'), '?   ')} {e['label']}")
        print(f"     {e.get('summary', '')}")
        if e.get("source") == "estimated":
            print("     (estimated: no API reports this balance)")
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
        print(f"claude-api/    : {'on' if cfg.claude_api_enabled else 'off'} · "
              f"${cfg.claude_api_monthly_credit_usd:g}/period from day {cfg.claude_api_reset_day}")
        return 0

    if args.action in ("claude-api-credit", "claude-api-reset-day"):
        try:
            if args.action == "claude-api-credit":
                cfg.claude_api_monthly_credit_usd = max(0.0, float(args.value))
            else:
                cfg.claude_api_reset_day = min(31, max(1, int(args.value)))
        except ValueError:
            print("Give a number.", file=sys.stderr)
            return 1
        print(f"Saved to {config_mod.save(cfg)}")
        return 0

    if args.action == "claude-api":
        from . import claude_api

        want = (args.value or "").strip().lower()
        if want not in ("on", "off", ""):
            print("Use: llm-sidecar config claude-api on|off", file=sys.stderr)
            return 1
        if want:
            cfg.claude_api_enabled = (want == "on")
            print(f"Saved to {config_mod.save(cfg)}")
        info = claude_api.check(cfg)
        s = claude_api.spend(cfg)
        print(f"enabled        : {'yes' if info['enabled'] else 'no'}")
        print(f"sdk            : {'installed' if info['sdk'] else 'missing'}")
        print(f"credentials    : {info['credentials'] or 'none'}")
        print(f"allowance      : ${s['allowance_usd']:g} per period, renews {s['resets']}")
        print(f"spent          : ${s['spent_usd']:.4f} this period (sidecar's own count)")
        print(f"               : {info['detail']}")
        if info["enabled"] and info["credentials"]:
            print()
            print("Use it by naming a model:")
            print("  llm-sidecar ask --model claude-api/haiku \"...\"")
        return 0 if info["sdk"] and (not info["enabled"] or info["credentials"]) else 1

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

    if args.action == "prepare":
        print(f"Preparing {args.profile}: saving a quantised copy so it loads "
              f"without the full-precision peak.")
        print("This loads the model the slow way one last time. It needs the "
              "same free memory a normal cold load does, and waits for it.")
        try:
            r = im.prepare(args.profile, wait_for_memory_s=args.wait)
        except im.ImageError as e:
            print(str(e), file=sys.stderr)
            return 1
        if r.get("already"):
            print(f"already prepared: {r['path']}")
        else:
            print(f"saved {r.get('size_gib')} GiB to {r['path']}")
        if r.get("peak_gib"):
            print(f"peak memory    : {r['peak_gib']} GiB (was {r['was_peak_gib']} GiB)")
        return 0

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
    runs = max(2, args.runs)

    # Four runs by default, not two, and every one is printed.
    #
    # A two-run bench said this machine was fine. The third run was the one
    # that fell off a cliff — 23.9s, 22.5s, then 77.6s — because MLX's buffer
    # cache had grown to 16 GiB by then and the kernel started paging. An
    # average would have hidden that; the slowest run is the number you
    # actually live with.
    print(f"  {profile} at {args.width or 1024}x{args.height or 1024}, "
          f"{runs} runs\n")
    # The bench runs in-process on purpose (see below), so while a daemon is
    # up there can be two models resident at once — which is the memory
    # pressure this bench exists to detect. Say so rather than let it look
    # like a property of the profile.
    from . import config as config_mod
    if im.daemon_available(config_mod.load()):
        print("  note: the daemon is running and may hold a model resident.")
        print("  These runs load their own, so figures include that. Stop")
        print("  the service for a clean measurement.\n")
    with tempfile.TemporaryDirectory() as tmp:
        engine.RESIDENT.unload()
        times = []
        for i in range(runs):
            # allow_daemon=False because this command measures THIS
            # process's engine: it unloads the resident model above and
            # reads mlx_cache_gib from engine.status() below. Delegating
            # would do the work in the daemon and report the cache of a
            # process that never loaded anything — four runs of 0G.
            r = im.generate(im.ImageRequest(
                prompt=f"{prompt} (take {i})", profile=profile, out_dir=tmp,
                stem=f"bench{i}", width=args.width or 1024,
                height=args.height or 1024, seed=100 + i, project="bench",
            ), allow_daemon=False)
            times.append(r.seconds)
            st = engine.status()
            print(f"  run{i + 1} {r.seconds:6.1f}s  "
                  f"{'cold' if i == 0 else 'warm'}  "
                  f"mlx cache {st.get('mlx_cache_gib', '?')}G")

    steady = times[1:] or times
    best, worst = min(steady), max(steady)
    print()
    print(f"  first {times[0]:.1f}s · then {best:.1f}-{worst:.1f}s")
    if worst > best * 1.5:
        print()
        print("  Unstable: the slowest steady run is more than half again the")
        print("  fastest. That is memory pressure, not the model. Close "
              "something,")
        print("  or use a profile with a smaller peak.")
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
    from . import config as config_mod
    from . import searxng_native as native
    from .search import health
    from .types import SidecarError

    cfg = config_mod.load()
    try:
        if args.action == "install":
            r = native.install(cfg, port=args.port or native.DEFAULT_PORT)
            print(f"\nReady at {r['url']}  (SearXNG {r['commit']}, {r['commit_date'][:10]})")
            print(f"Runs at login, restarts if it dies. Files: {r['home']}")
            if r["hint"]:
                print("\n" + r["hint"])
        elif args.action == "update":
            r = native.update()
            if r["updated"]:
                print(f"Updated {r['from']} → {r['commit']} ({r['commit_date'][:10]}).")
            else:
                print(f"Already current: {r['commit']} ({r['commit_date'][:10]}).")
        elif args.action == "uninstall":
            native.uninstall(purge=args.purge)
            print("Removed the SearXNG service." + (" Deleted its files." if args.purge
                  else f" Files kept in {native.home()} (--purge deletes them)."))
        elif args.action == "logs":
            lines = native.log_path().read_text(encoding="utf-8", errors="replace").splitlines()
            print("\n".join(lines[-args.lines:]) or "(no output)")
        else:
            s = native.status(cfg, with_engines=True)
            h = health(cfg)
            print(f"search         : {h['state']} — {h['message']}")
            if not s["installed"]:
                print("SearXNG        : not installed (`llm-sidecar searxng install`)")
                return 0
            print(f"SearXNG        : {'answering' if s['answering'] else 'NOT answering'}"
                  f" at {s['url']} (service {'running' if s['service_running'] else 'stopped'})")
            age = f", {s['age_days']} days old" if s["age_days"] is not None else ""
            print(f"version        : {s['commit']}{age}"
                  + ("  — run `llm-sidecar searxng update`" if s["stale"] else ""))
            eng = s.get("engines")
            if eng:
                failing = ", ".join(f"{e['engine']} ({e['reason']})" for e in eng["failing"])
                print(f"test query     : {eng['results']} results"
                      + (f"; failing: {failing}" if failing else "; all engines answered"))
            print(f"log            : {s['log']}")
    except (SidecarError, OSError) as e:
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
        c = sc.complete(args.prompt, tier=args.tier, budget=args.budget,
                        model=args.model or None, max_tokens=args.max_tokens)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(c.text)
    if not args.quiet:
        cost = "free" if c.usage.cost_usd == 0 else f"${c.usage.cost_usd:.6f}"
        if c.model.startswith("claude/") and c.usage.cost_usd == 0:
            cost = "subscription"
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

    al = sub.add_parser("allowances",
                        help="what each provider has left: credit, free requests, usage windows")
    al.add_argument("--refresh", action="store_true",
                    help="re-read now (spends one tiny claude/ call for the subscription windows)")
    al.add_argument("--json", action="store_true")

    sm = sub.add_parser("sum", help="summarise a file (- for stdin)")
    sm.add_argument("file")
    sm.add_argument("--style", default="brief", choices=["brief", "bullets", "detailed", "tldr"])
    sm.add_argument("--focus", default="")

    ca = sub.add_parser("cache", help="inspect or clear the response cache")
    ca.add_argument("action", nargs="?", default="stats", choices=["stats", "clear"])

    cf = sub.add_parser("config", help="set your API key, budget, and see what's configured")
    cf.add_argument("action",
                    choices=["key", "budget", "show", "clear-key", "claude-cli",
                             "claude-api", "claude-api-credit", "claude-api-reset-day"])
    cf.add_argument("value", nargs="?", default="")
    cf.add_argument("--save", action="store_true",
                    help="write it to the config file (plain text) rather than this process only")

    img = sub.add_parser("image", help="generate images locally (Apple Silicon)")
    img.add_argument("action", nargs="?", default="status",
                     choices=["gen", "profiles", "status", "bench", "prepare"])
    img.add_argument("prompt", nargs="?", default="")
    img.add_argument("--profile", default="standard")
    img.add_argument("--out", default=".", help="output directory")
    img.add_argument("--stem", default="image", help="filename stem")
    img.add_argument("--ref", action="append", dest="refs", default=[],
                     help="reference image (repeatable, for consistency profiles)")
    img.add_argument("--negative", default="")
    img.add_argument("--width", type=int)
    img.add_argument("--height", type=int)
    img.add_argument("--steps", type=int)
    img.add_argument("--seed", type=int)
    img.add_argument("--resolution", type=int,
                     help="upscale: target SHORT edge in pixels "
                          "(mflux scales by resolution / min(w, h))")
    img.add_argument("--project", default="", help="recorded in the manifest")
    img.add_argument("--runs", type=int, default=4,
                     help="with bench: how many images to time (default 4)")
    img.add_argument("--wait", type=float, default=600,
                     help="with prepare: seconds to wait for free memory (default 600)")
    img.add_argument("--non-commercial", action="store_true",
                     help="permit non-commercial profiles (cinematic, kontext-edit)")

    sv = sub.add_parser("service", help="run the daemon at login (launchd / systemd)")
    sv.add_argument("action", nargs="?", default="status",
                    choices=["install", "uninstall", "status"])
    sv.add_argument("--port", type=int, default=4001)

    ph = sub.add_parser(
        "phone", help="use it from your phone, privately, over Tailscale",
        description="Put the sidecar on your Tailscale network so your phone can use it "
                    "at home or away. Only your own signed-in devices can reach it; it is "
                    "not exposed to your Wi-Fi or the internet. Then open the printed "
                    "address on the phone, or use the Android app in mobile/.")
    ph.add_argument("action", nargs="?", default="on", choices=["on", "off", "status"])
    ph.add_argument("--port", type=int, default=8443,
                    help="HTTPS port on your tailnet name (default 8443, so it cannot "
                         "replace anything you already serve on 443)")

    sx = sub.add_parser("searxng", help="run a local SearXNG for better search")
    sx.add_argument("action", nargs="?", default="status",
                    choices=["install", "update", "uninstall", "status", "logs"])
    sx.add_argument("--port", type=int, default=None, help="host port (default 8888)")
    sx.add_argument("--purge", action="store_true",
                    help="with uninstall: also delete the checkout and its environment")
    sx.add_argument("--lines", type=int, default=50, help="with logs: how many lines")

    ask = sub.add_parser("ask", help="one-shot completion")
    ask.add_argument("prompt")
    ask.add_argument("--tier", default=None, choices=["fast", "balanced", "powerful"])
    ask.add_argument("--budget", default=None, choices=["free", "cheap", "best"])
    ask.add_argument("--model", default=None,
                     help="a model id, e.g. claude-api/haiku or ollama/llama3.2:3b")
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
    if args.cmd == "allowances":
        return _allowances(args)
    if args.cmd == "sum":
        return _summarise(args)
    if args.cmd == "image":
        return _image(args)
    if args.cmd == "config":
        return _config(args)
    if args.cmd == "service":
        return _service(args)
    if args.cmd == "phone":
        from . import config as config_mod
        from . import phone
        return phone.run(args.action, config_mod.load(), https_port=args.port)
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
