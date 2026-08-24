#!/usr/bin/env bash
# Run Job Hunter: ./run.sh [all|api|worker]
#   all   (default) worker in background + API in foreground on :8088
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
    echo "worker pid: $WORKER_PID | ui: http://127.0.0.1:8088 (Ctrl-C stops both)"
    python main.py api
    ;;
  *)
    echo "usage: $0 [all|api|worker]" >&2
    exit 2
    ;;
esac
