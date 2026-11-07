#!/usr/bin/env bash
# Run Job Hunter: ./run.sh [all|api|worker]
#   all   (default) worker in background + API in foreground
#           (APP_HOST/APP_PORT, defaults 127.0.0.1:8088)
#   api             API only
#   worker          scheduler/worker only
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "No virtualenv found — run ./setup.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source ".venv/bin/activate"

# Mirror AppSettings (config/app.yaml + APP_* env): the API binds
# APP_HOST/APP_PORT, so the readiness poll and printed URL must too.
APP_HOST="${APP_HOST:-127.0.0.1}"
APP_PORT="${APP_PORT:-8088}"
export APP_HOST APP_PORT

case "${1:-all}" in
  api)
    exec python main.py api
    ;;
  worker)
    exec python main.py worker
    ;;
  all)
    python main.py worker &
    WORKER_PID=$!
    trap '[ -n "${WORKER_PID:-}" ] && kill "$WORKER_PID" 2>/dev/null || true' EXIT INT TERM
    trap 'echo "worker exited (pid $WORKER_PID) — if discovery stalls, check data/logs/app.log and restart with ./run.sh worker" >&2' EXIT
    echo "worker pid: $WORKER_PID — waiting for API readiness on http://${APP_HOST}:${APP_PORT} ..."
    # Poll /healthz (up to 10s) before printing the UI URL.
    READY=0
    for _ in $(seq 1 20); do
      if python -c "import urllib.request, os; urllib.request.urlopen(f\"http://{os.environ['APP_HOST']}:{os.environ['APP_PORT']}/healthz\", timeout=1)" 2>/dev/null; then
        READY=1
        break
      fi
      sleep 0.5
    done
    if [ "$READY" = "1" ]; then
      echo "worker pid: $WORKER_PID | ui: http://${APP_HOST}:${APP_PORT} (Ctrl-C stops both)"
    else
      echo "warning: API did not become ready within 10s — check data/logs/app.log; URL may still work once uvicorn binds" >&2
      echo "worker pid: $WORKER_PID | ui: http://${APP_HOST}:${APP_PORT} (Ctrl-C stops both)"
    fi
    python main.py api
    ;;
  *)
    echo "usage: $0 [all|api|worker]" >&2
    exit 2
    ;;
esac
