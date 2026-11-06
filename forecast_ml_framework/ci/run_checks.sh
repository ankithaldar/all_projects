#!/usr/bin/env bash
# Local equivalent of .github/workflows/forecast-ml-ci.yml.
# Run from anywhere: bash forecast_ml_framework/ci/run_checks.sh
# Environment parity: the uv venv (`uv sync`) is canonical — it
# installs all deps, including the dev group (pytest, pytest-cov).
# The system Anaconda env is not canonical; skips for deps it lacks
# (e.g. mlflow optional extras) are expected there.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "==> lint (pylint, root .pylintrc on forecast_ml_framework/src) <=="
uv run --project "${PROJECT_DIR}" --with pylint \
  pylint --rcfile=.pylintrc --fail-under=9.0 forecast_ml_framework/src

echo "==> test (uv sync + pytest, coverage gate >=85%) <=="
# Mirror of the CI test job: total coverage of
# src/forecast_ml_framework must be >= 85%.
(cd "${PROJECT_DIR}" && uv sync --frozen && uv run pytest -q \
  --cov=src/forecast_ml_framework \
  --cov-report=term-missing \
  --cov-fail-under=85)

echo "==> demo (registry / import smoke) <=="
(cd "${PROJECT_DIR}" && uv run python -c \
  "from forecast_ml_framework.pipeline_registry import register_pipelines; \
pipelines = register_pipelines(); \
assert pipelines, 'no pipelines registered'; \
assert {'data', 'modeling', 'scoring'} <= set(pipelines), sorted(pipelines); \
assert 'scoring' in pipelines, sorted(pipelines); \
print('registered pipelines:', sorted(pipelines))")

echo "==> terraform fmt check <=="
if command -v terraform >/dev/null 2>&1; then
  terraform fmt -check -recursive "${PROJECT_DIR}/infra"
else
  echo "terraform not installed; skipping fmt check (CI runs it via setup-terraform)"
fi

echo "All checks passed."
