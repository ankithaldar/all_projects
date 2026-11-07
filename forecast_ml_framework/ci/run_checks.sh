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

echo "==> lockfile freshness (uv lock --check) <=="
uv lock --check --project "${PROJECT_DIR}"

echo "==> lint (pylint, root .pylintrc on forecast_ml_framework/src) <=="
# pylint is pinned (MLops-17): an unpinned --with would let an
# upstream pylint release flip the --fail-under=9.0 gate with no
# change in this repo. Bump the pin deliberately, never implicitly.
uv run --frozen --project "${PROJECT_DIR}" --with 'pylint==4.1.3' \
  pylint --rcfile=.pylintrc --fail-under=9.0 forecast_ml_framework/src

echo "==> test (uv sync + pytest, coverage gate >=85%) <=="
# Mirror of the CI test job: total coverage of
# src/forecast_ml_framework must be >= 85%.
(cd "${PROJECT_DIR}" && uv sync --frozen && uv run --frozen pytest -q \
  --cov=src/forecast_ml_framework \
  --cov-report=term-missing \
  --cov-fail-under=85)

echo "==> demo (registry / import smoke) <=="
(cd "${PROJECT_DIR}" && uv run --frozen python -c \
  "from forecast_ml_framework.pipeline_registry import register_pipelines; \
pipelines = register_pipelines(); \
assert pipelines, 'no pipelines registered'; \
assert {'data', 'modeling', 'scoring'} <= set(pipelines), sorted(pipelines); \
assert 'scoring' in pipelines, sorted(pipelines); \
print('registered pipelines:', sorted(pipelines))")

echo "==> wheel build (packaged conf in zip namelist) <=="
# O2: a string-only force-include check cannot prove the
# built wheel ships conf. Build the wheel and assert the
# zip namelist (stdlib zipfile, no install/import) contains
# forecast_ml_framework/conf/base/catalog.yml.
WHEEL_CHECK_DIR="${PROJECT_DIR}/.wheel-check"
rm -rf "${WHEEL_CHECK_DIR}"
mkdir -p "${WHEEL_CHECK_DIR}"
uv build --wheel --project "${PROJECT_DIR}" --out-dir "${WHEEL_CHECK_DIR}"
python3 -c 'import sys, zipfile; names=zipfile.ZipFile(sys.argv[1]).namelist(); assert "forecast_ml_framework/conf/base/catalog.yml" in names, [n for n in names if "catalog" in n]; print("wheel namelist ok: forecast_ml_framework/conf/base/catalog.yml")' "${WHEEL_CHECK_DIR}"/*.whl
rm -rf "${WHEEL_CHECK_DIR}"

echo "==> terraform fmt + validate (both stacks) <=="
TF_ABSENT=0
if command -v terraform >/dev/null 2>&1; then
  terraform fmt -check -recursive "${PROJECT_DIR}/infra"
  # init -backend=false needs no credentials and no state: it
  # resolves providers against the committed .terraform.lock.hcl,
  # so an out-of-date lock file fails here, not at apply time.
  for stack in aws gcp; do
    echo "--- terraform init -backend=false + validate: infra/${stack} ---"
    (cd "${PROJECT_DIR}/infra/${stack}" \
      && terraform init -backend=false -input=false \
      && terraform validate)
  done
else
  # Fail-open ONLY when the binary is absent, and LOUDLY
  # (MLops-16): a silent skip previously reported green for
  # unvalidated infrastructure. CI hard-fails on the same checks.
  TF_ABSENT=1
  echo "=================================================================="
  echo "WARNING: terraform binary NOT FOUND -- fmt/validate SKIPPED."
  echo "WARNING: infrastructure was NOT validated locally. Install"
  echo "WARNING: terraform (>= 1.3) to validate infra/aws + infra/gcp,"
  echo "WARNING: or rely on CI (forecast-ml-ci.yml terraform job)."
  echo "=================================================================="
fi

echo "All checks passed."
if [ "${TF_ABSENT}" = "1" ]; then
  echo "WARNING: terraform not validated (binary absent) -- see above."
fi
