#!/usr/bin/env bash
# Stage driver for the RSNA knee MVP inside Kaggle notebooks.
#
# Usage:
#   bash kaggle_run.sh <stage> [extra args passed to main.py]
#
# Stages:
#   setup   Install pinned requirements (offline wheel dir supported).
#   index   Header-only DICOM scan -> index.parquet
#   labels  Rule-based pseudo-labels -> labels_pseudo.csv
#   folds   StratifiedGroupKFold assignment -> folds.csv
#   cache   Decode every indexed series -> sharded HDF5 volume dataset(s)
#   selftest Preflight: artifacts/mount/cache/model + 2 real steps.
#   train   Resume-aware fold training (session-budget + checkpoint push).
#   infer   Fold-ensemble prediction -> submission.csv
#   sweep   Noise-floor study (seeds x folds, BLUEPRINT 11.0-1)
#   all     index -> labels -> folds (train/infer run per-session instead)
#
# Environment overrides:
#   EXPERIMENT      Experiment YAML under configs/experiments/
#                   (default: mvp_efnv2s_384_k24_5f.yaml).
#   WHEELS_DIR      Directory of offline wheels; enables --no-index installs.
#   PIP_EXTRA       Additional pip install arguments.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
# Reduce allocator fragmentation on 16 GB T4s (advice from CUDA OOM dumps).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Local credentials/overrides; harmless when the file is absent.
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

STAGE="${1:-all}"
shift || true
EXPERIMENT="${EXPERIMENT:-configs/experiments/mvp_efnv2s_384_k24_5f.yaml}"
MAIN="python ${ROOT}/main.py"

run_stage() {
  local stage="$1"; shift || true
  echo "=== kaggle_run: stage=${stage} experiment=${EXPERIMENT} ==="
  ${MAIN} "${stage}" --experiment "${EXPERIMENT}" "$@"
}

install_requirements() {
  local args=(-r "${ROOT}/requirements.txt" ${PIP_EXTRA:-})
  if [[ -n "${WHEELS_DIR:-}" && -d "${WHEELS_DIR}" ]]; then
    echo "=== kaggle_run: offline pip from ${WHEELS_DIR} ==="
    python -m pip install --no-index --find-links "${WHEELS_DIR}" "${args[@]}"
  else
    echo "=== kaggle_run: online pip install ==="
    python -m pip install "${args[@]}"
  fi
}

case "${STAGE}" in
  setup)
    install_requirements
    ;;
  index|labels|folds)
    # Short user-facing names map to main.py subcommands.
    run_stage "build-${STAGE}" "$@"
    ;;
  cache)
    run_stage "build-cache" "$@"
    ;;
  selftest)
    run_stage "selftest" "$@"
    ;;
  train|infer)
    run_stage "${STAGE}" "$@"
    ;;
  sweep)
    run_stage "sweep" "$@"
    ;;
  all)
    for step in index labels folds; do
      run_stage "${step}" "$@"
    done
    echo "=== kaggle_run: data stages complete; run 'train'/'infer' per session ==="
    ;;
  *)
    echo "Unknown stage: ${STAGE} (expected setup|index|labels|folds|train|infer|sweep|all)" >&2
    exit 2
    ;;
esac
