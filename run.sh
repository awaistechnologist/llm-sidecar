#!/usr/bin/env bash
# Start the llm-sidecar daemon and open the dashboard.
#
# Any arguments are passed through to `llm-sidecar serve`, so:
#   ./run.sh --no-ui
#   ./run.sh --port 4100
set -euo pipefail

cd "$(dirname "$0")"

YELLOW=$'\033[33m'; RED=$'\033[31m'; DIM=$'\033[2m'; RESET=$'\033[0m'

if [ ! -x .venv/bin/llm-sidecar ]; then
  echo "${RED}Not installed yet.${RESET} Run ./install.sh first."
  exit 1
fi

# A .env here is a convenience for keys you don't want in your shell profile.
# Only KEY=value lines are read; anything else is ignored rather than eval'd,
# because sourcing a file blindly runs whatever is in it.
if [ -f .env ]; then
  while IFS='=' read -r key value; do
    case "$key" in
      ''|\#*) continue ;;
      *[!A-Za-z0-9_]*) continue ;;
    esac
    # Don't override something already exported in this shell.
    if [ -z "${!key:-}" ]; then
      export "$key=${value}"
    fi
  done < .env
fi

PORT="${LLM_SIDECAR_PORT:-4001}"
# Honour --port if it was passed through.
prev=""
for arg in "$@"; do
  [ "$prev" = "--port" ] && PORT="$arg"
  prev="$arg"
done

if command -v lsof >/dev/null 2>&1 && lsof -ti:"$PORT" >/dev/null 2>&1; then
  echo "${YELLOW}Port $PORT is already in use.${RESET}"
  echo "  Something may already be running: http://localhost:$PORT"
  echo "  ${DIM}Stop it, or start elsewhere with: ./run.sh --port 4100${RESET}"
  exit 1
fi

URL="http://localhost:$PORT"
echo "Starting llm-sidecar → $URL"

# Open the dashboard once the daemon is actually answering, rather than
# immediately — a browser tab showing a connection error is worse than a
# second of waiting. Backgrounded so it doesn't block the server.
case " $* " in
  *" --no-ui "*) ;;
  *)
    (
      for _ in $(seq 1 40); do
        if curl -sf --max-time 1 "$URL/health" >/dev/null 2>&1; then
          command -v open    >/dev/null 2>&1 && open "$URL"    && exit 0
          command -v xdg-open >/dev/null 2>&1 && xdg-open "$URL" && exit 0
          exit 0
        fi
        sleep 0.5
      done
    ) &
    ;;
esac

# exec so Ctrl-C reaches the server directly rather than this wrapper.
exec ./.venv/bin/llm-sidecar serve "$@"
