#!/usr/bin/env bash
# One-time setup for Job Hunter: venv, dependencies, env file, database.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=".venv"

echo "==> creating virtualenv ($VENV)"
"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip --quiet

echo "==> installing job_hunter + dev/semantic extras"
pip install --quiet -e '.[dev,semantic]'

echo "==> preparing gateway env file"
GATEWAY_ENV="src/job_hunter/llm_gateway/.env"
if [ ! -f "$GATEWAY_ENV" ]; then
  cp .env.example "$GATEWAY_ENV"
  echo "    created $GATEWAY_ENV — ADD YOUR REAL API KEYS before running discovery."
else
  echo "    $GATEWAY_ENV already present, leaving untouched."
fi

echo "==> bootstrapping database (migrations + taxonomy + defaults)"
python main.py seed-db

echo "==> running unit tests"
PYTHONPATH="src/job_hunter" python -m pytest tests/unit -q

echo
echo "Setup complete. Next:"
echo "  1. edit $GATEWAY_ENV with real provider keys"
echo "  2. ./run.sh            # api + worker together"
