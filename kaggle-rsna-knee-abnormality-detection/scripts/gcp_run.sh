#!/usr/bin/env bash
# =============================================================================
# gcp_run.sh -- from-scratch runner for a GCP GPU VM (or any Linux box).
#
# Downloads the RSNA competition data via the Kaggle API, installs deps and
# executes the full 7-kernel pipeline (scripts/run_all.sh) resumably.
#
# 1) Create the VM (Deep Learning image ships CUDA + torch preinstalled):
#
#    gcloud compute instances create knee-gpu \
#      --zone=us-central1-a \
#      --machine-type=g2-standard-8 \
#      --accelerator=type=nvidia-l4,count=1 \
#      --maintenance-policy=TERMINATE \
#      --image-family=pytorch-latest-gpu \
#      --image-project=deeplearning-platform-release \
#      --boot-disk-size=300GB
#
#    Budget alternative: --machine-type=n1-highmem-8 with
#    --accelerator=type=nvidia-tesla-t4,count=1.
#
# 2) SSH in, clone/copy this repo, then:
#
#    export KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx     # or ~/.kaggle/kaggle.json
#    bash scripts/gcp_run.sh
#
#    Prereq: accept the competition rules once on kaggle.com, otherwise the
#    download 403s. Data is ~20 GB zipped; keep >=100 GB free.
#
# Optional env:
#   REPO_DIR    repo location          (default: script's parent dir)
#   DATA_ROOT   data dir                (default: $HOME/data/rsna-knee)
#   WORK        artifacts/scratch dir   (default: $WORK -> ./outputs)
#   EXP         experiment yaml         (configs/experiment/student_2p5d_effnetv2.yaml)
#   WITH_MONAI  1 -> install monai for the 3D encoder experiment
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMPETITION='rsna-knee-abnormality-detection'
DATA_ROOT="${DATA_ROOT:-$HOME/data/$COMPETITION}"
EXP="${EXP:-configs/experiment/student_2p5d_effnetv2.yaml}"

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --- 0. sanity --------------------------------------------------------------
log "repo: $REPO_DIR"
[ -f "$REPO_DIR/requirements.txt" ] || die "$REPO_DIR does not look like the repo root"
command -v nvidia-smi >/dev/null && nvidia-smi -L || log 'WARNING: no GPU visible; training will be CPU-slow'

PY="$(command -v python3 || command -v python)"
"$PY" - <<'EOF' || die 'python >=3.10,<3.13 required'
import sys
assert (3, 10) <= sys.version_info[:2] <= (3, 12), sys.version
EOF

# --- 1. kaggle credentials --------------------------------------------------
mkdir -p "$HOME/.kaggle"
if [ ! -s "$HOME/.kaggle/kaggle.json" ]; then
  : "${KAGGLE_USERNAME:?set KAGGLE_USERNAME or place ~/.kaggle/kaggle.json}"
  : "${KAGGLE_KEY:?set KAGGLE_KEY or place ~/.kaggle/kaggle.json}"
  printf '{"username":"%s","key":"%s"}\n' "$KAGGLE_USERNAME" "$KAGGLE_KEY" \
    > "$HOME/.kaggle/kaggle.json"
fi
chmod 600 "$HOME/.kaggle/kaggle.json"

# --- 2. dependencies --------------------------------------------------------
log 'installing python deps'
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r "$REPO_DIR/requirements.txt"
if [ "${WITH_MONAI:-0}" = '1' ]; then
  "$PY" -m pip install -q 'monai>=1.3'
fi

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# --- 3. competition data ----------------------------------------------------
if [ -f "$DATA_ROOT/train.csv" ] && [ -d "$DATA_ROOT/train_series" ]; then
  log "skip download ($DATA_ROOT already populated)"
else
  log "downloading $COMPETITION -> $DATA_ROOT"
  mkdir -p "$DATA_ROOT"
  "$PY" -m kaggle competitions download \
      -c "$COMPETITION" -p "$DATA_ROOT" --unzip \
    || die "download failed -- accept rules at https://www.kaggle.com/competitions/$COMPETITION/rules then retry"
fi

# --- 4. pipeline ------------------------------------------------------------
log 'running full pipeline (resumable; re-run to continue after a stop)'
export DATA_ROOT WORK="${WORK:-$REPO_DIR/outputs}" EXP PYTHON
bash "$REPO_DIR/scripts/run_all.sh"

log "DONE -> $WORK/submission_blended.csv"
