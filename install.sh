#!/usr/bin/env bash
# Install llm-sidecar into a local virtualenv.
#
# Safe to re-run: an existing venv is reused and the install is refreshed.
# Nothing is written outside this directory, and nothing is installed globally.
set -euo pipefail

cd "$(dirname "$0")"

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
RED=$'\033[31m'; RESET=$'\033[0m'
ok()   { echo "  ${GREEN}✓${RESET} $*"; }
warn() { echo "  ${YELLOW}!${RESET} $*"; }
bad()  { echo "  ${RED}✗${RESET} $*"; }

echo
echo "${BOLD}llm-sidecar — install${RESET}"
echo

# ── Python ────────────────────────────────────────────────────────────────────
# 3.11 is the floor: the codebase uses `X | Y` unions at runtime.
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  bad "Python 3.11 or newer is required, and I couldn't find one."
  echo "     macOS:  brew install python@3.13"
  echo "     Ubuntu: sudo apt install python3.13 python3.13-venv"
  exit 1
fi
ok "Python $("$PYTHON" -c 'import platform; print(platform.python_version())') ($PYTHON)"

# ── virtualenv ────────────────────────────────────────────────────────────────
if [ -d .venv ]; then
  ok "Reusing existing .venv"
else
  "$PYTHON" -m venv .venv
  ok "Created .venv"
fi

echo "  … installing dependencies (this takes a minute)"
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -e ".[all]"
ok "Installed llm-sidecar $(./.venv/bin/python -c 'import llm_sidecar; print(llm_sidecar.__version__)')"

# ── what it can actually reach ────────────────────────────────────────────────
echo
echo "${BOLD}Checking what's available${RESET}"

if curl -sf --max-time 2 "${OLLAMA_HOST:-http://localhost:11434}/api/tags" >/dev/null 2>&1; then
  COUNT=$(./.venv/bin/python -c "
from llm_sidecar import catalogue
from llm_sidecar.config import Config
print(len(catalogue.ollama_models(Config())))" 2>/dev/null || echo "?")
  ok "Ollama is running — $COUNT model(s) installed"
elif command -v ollama >/dev/null 2>&1; then
  warn "Ollama is installed but not running. Start it, or run: ollama serve"
else
  warn "No Ollama. Local models unavailable — install from https://ollama.com"
  echo "     ${DIM}or set OPENROUTER_API_KEY to use cloud models instead${RESET}"
fi

if [ -n "${OPENROUTER_API_KEY:-}" ]; then
  ok "OPENROUTER_API_KEY is set — free and paid cloud models available"
elif [ -f .env ] && grep -q '^OPENROUTER_API_KEY=.' .env 2>/dev/null; then
  ok "OPENROUTER_API_KEY found in .env"
else
  warn "No OpenRouter key — local models only"
  echo "     ${DIM}add one later in the dashboard, or put it in a .env file here${RESET}"
fi

if command -v docker >/dev/null 2>&1 || command -v podman >/dev/null 2>&1; then
  ok "Container runtime found — 'llm-sidecar searxng up' will work"
else
  warn "No Docker/Podman. Search falls back to DuckDuckGo (fine, just slower to trust)"
fi

# ── next steps ────────────────────────────────────────────────────────────────
REPO="$(pwd)"
echo
echo "${BOLD}Done. Next:${RESET}"
echo
echo "  ./run.sh                          start the daemon + dashboard"
echo "  ./.venv/bin/llm-sidecar status    what it can see"
echo "  ./.venv/bin/llm-sidecar answer \"what is the population of Iran?\""
echo
echo "${DIM}To use it from an MCP client (Claude Desktop, Cursor, …), add:${RESET}"
echo
cat <<JSON
  {"mcpServers": {"llm-sidecar": {
    "command": "$REPO/.venv/bin/python",
    "args": ["-m", "llm_sidecar.mcp_server"]}}}
JSON
echo
