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

echo "==> ingesting company seeds"
python main.py discover-companies

echo "==> verifying ATS boards (populates ats_provider/board_ref)"
echo "    discovery needs these; without them a fresh install fetches zero jobs"
python main.py verify-ats --chunk "${ATS_CHUNK:-30}" --rounds "${ATS_ROUNDS:-4}" || \
  echo "    warning: ATS verification incomplete; rerun 'python main.py verify-ats' later" >&2

if [ "${SKIP_NETWORK_SETUP:-0}" != "1" ]; then
  echo "==> bulk fetching job openings from public aggregators"
  python main.py fetch-jobs --limit "${FETCH_LIMIT:-2000}" || \
    echo "    warning: bulk fetch incomplete; rerun 'python main.py fetch-jobs' later" >&2

  echo "==> scouting new companies from LinkedIn job pages (via search index)"
  echo "    verifies each company website; appends to seeds/companies_scouted.yaml"
  python main.py scout-companies --queries "${SCOUT_QUERIES:-12}" || \
    echo "    warning: scouting incomplete; rerun 'python main.py scout-companies' later" >&2
fi

echo "==> running unit tests"
PYTHONPATH="src/job_hunter" python -m pytest tests/unit -q

echo
echo "Setup complete. Next:"
echo "  1. edit $GATEWAY_ENV with real provider keys"
echo "  2. ./run.sh            # api + worker together"
