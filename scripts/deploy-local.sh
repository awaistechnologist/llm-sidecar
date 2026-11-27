#!/usr/bin/env bash
# Put a released version on this machine, and prove it rather than assume it.
#
# Written because a throwaway version of this printed DEPLOYED over a failed
# install, and because three separate steps have silently lagged a version
# behind at different times: PyPI's index serving stale views to pip, pipx
# reusing a cached wheel, and the daemon running whatever was installed when
# it last started. Every step here is checked, and the script exits non-zero
# rather than reporting success it has not verified.
#
#   usage: scripts/deploy-local.sh 0.9.3
set -euo pipefail

V="${1:?usage: deploy-local.sh <version>}"
SERVICE_VENV="$HOME/.local/share/llm-sidecar/venv"
fail() { echo "FAILED: $*" >&2; exit 1; }

# 1. The release workflow must have succeeded before anything is installed.
for _ in $(seq 1 60); do
  r=$(gh run list --workflow publish.yml --limit 1 --json status,conclusion 2>/dev/null) || true
  [ "$(jq -r '.[0].status' <<<"$r")" = completed ] && {
    cc=$(jq -r '.[0].conclusion' <<<"$r")
    echo "publish workflow: $cc"
    [ "$cc" = success ] || fail "publish workflow $cc"
    break
  }
  sleep 20
done

# 2. pip must genuinely see it. PyPI has served its HTML index and its JSON
#    endpoints out of step for minutes at a time, so asking once is not enough.
ok=0
for i in $(seq 1 40); do
  "$SERVICE_VENV/bin/python" -m pip install -q --no-cache-dir --upgrade \
    "llm-sidecar[images,claude-api]==$V" >/dev/null 2>&1 && { ok=1; echo "pip installed $V after $((i*30))s"; break; }
  sleep 30
done
[ "$ok" = 1 ] || fail "pip never saw $V"

# 3. The CLI on PATH. A plain --force has repeatedly reinstalled a cached
#    wheel, so this uninstalls first and then checks what actually landed.
pipx uninstall llm-sidecar >/dev/null 2>&1 || true
# The [claude-api] extra: without it the CLI cannot use API credits (0.12.0
# was deployed without it, and `ask --model claude-api/...` failed).
pipx install "llm-sidecar[claude-api]==$V" >/dev/null 2>&1 || fail "pipx install failed"

# 4. Verify each one separately, from a neutral directory: running python
#    inside the checkout imports the repo rather than site-packages, which
#    once made a stale deployment look current.
cd /tmp
sv=$("$SERVICE_VENV/bin/python" -c 'import llm_sidecar;print(llm_sidecar.__version__)')
[ "$sv" = "$V" ] || fail "service venv is $sv"
cli=$("$(dirname "$(readlink -f "$(command -v llm-sidecar)")")/python" \
      -c 'import llm_sidecar;print(llm_sidecar.__version__)')
[ "$cli" = "$V" ] || fail "PATH CLI is $cli"

# 5. The daemon serves whatever it loaded at startup, so restart and ask it.
launchctl kickstart -k "gui/$(id -u)/com.llm-sidecar.daemon" >/dev/null 2>&1 || true
h=$(curl -s --retry 12 --retry-delay 4 --retry-connrefused --max-time 90 localhost:4001/health)
grep -q "\"version\":\"$V\"" <<<"$h" || fail "daemon reports: $h"

echo "  service venv : $sv"
echo "  PATH CLI     : $cli"
echo "  daemon       : $V"
echo VERIFIED
