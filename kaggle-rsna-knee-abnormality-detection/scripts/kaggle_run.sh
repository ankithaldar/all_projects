#!/usr/bin/env bash
# =============================================================================
# kaggle_run.sh -- one-cell bootstrap + stage dispatcher for Kaggle kernels.
#
# Pulls this repo straight from remote git (public or private via token),
# installs deps once per session, then runs any pipeline stage against the
# notebook's mounted inputs (/kaggle/input) and persistent scratch
# (/kaggle/working). Works identically on Colab or any Linux box.
#
# Kernel setup (once): Add-ons -> Internet ON, Accelerator GPU T4 x2 / P100,
# Add data -> Competition "RSNA Knee Abnormality Detection", and optionally
# prior-stage artifact datasets + Add-ons > Secrets (GITHUB_TOKEN,
# DISCORD_WEBHOOK_URL, WANDB_API_KEY, ...).
#
# First cell of every kernel:
#   %%bash
#   git clone --depth 1 https://github.com/<user>/<repo>.git /kaggle/working/repo
#   bash /kaggle/working/repo/scripts/kaggle_run.sh all      # or any stage below
#
# Private repo: add secret GITHUB_TOKEN (Add-ons > Secrets) and clone with
#   git clone --depth 1 "https://x-access-token:$GITHUB_TOKEN@github.com/<user>/<repo>.git" ...
#
# Stages (= BLUEPRINT kernels 1-7):
#   volumes | folds | weak-labels | teacher | student | self-train |
#   infer | blend | all
#
# Multi-kernel handoff: publish stage outputs as versioned datasets and point
# the consumer kernel at them via env overrides before calling this script:
#   export VOLUMES_CACHE=/kaggle/input/knee-volumes-cache-v1
#   export FOLDS=/kaggle/input/knee-folds-v1/train_folds.csv
#   export WEAK_LABELS=/kaggle/input/knee-weak-labels-v1/weak_labels.parquet
#
# Optional env:
#   REPO_DIR   repo location            (default: script's parent dir)
#   REPO_URL   used to clone when the repo is absent (standalone/curl usage)
#   GIT_REF    branch/tag to clone
#   EXP        experiment yaml          (configs/experiment/student_2p5d_effnetv2.yaml)
#   DATA_ROOT  data dir                 (/kaggle/input/rsna-knee-abnormality-detection)
#   WORK       scratch/artifact dir     (/kaggle/working)
#   WITH_MONAI 1 -> install monai (student_3d_resnet experiment)
#   FORCE_PIP  1 -> reinstall deps despite the session marker
# =============================================================================
set -euo pipefail

STAGE="${1:-all}"
PY="${PYTHON:-python3}"

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --- 0. clone from remote git when the repo is not present ------------------
if [ ! -e "$REPO_DIR/src/knee/__init__.py" ]; then
  : "${REPO_URL:?set REPO_URL=https://github.com/<user>/<repo>.git or place the script inside the repo}"
  TOKEN="${GITHUB_TOKEN:-}"
  if [ -z "$TOKEN" ]; then
    TOKEN="$("$PY" - <<'EOF' 2>/dev/null || true
try:
    from kaggle_secrets import UserSecretsClient
    print(UserSecretsClient().get_secret('GITHUB_TOKEN') or '')
except Exception:
    print('')
EOF
)"
  fi
  URL="$REPO_URL"
  [ -n "$TOKEN" ] && URL="https://x-access-token:${TOKEN}@${REPO_URL#https://}"
  log "cloning $REPO_URL -> $REPO_DIR"
  rm -rf "$REPO_DIR"
  clone_args=(--depth 1)
  [ -n "${GIT_REF:-}" ] && clone_args+=(-b "$GIT_REF")
  git clone "${clone_args[@]}" "$URL" "$REPO_DIR"
elif [ "${FORCE_SYNC:-0}" = '1' ] && [ -d "$REPO_DIR/.git" ]; then
  log "syncing $REPO_DIR to origin"
  git -C "$REPO_DIR" fetch --depth 1 origin
  git -C "$REPO_DIR" reset --hard '@{u}'
fi

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# --- 1. environment ---------------------------------------------------------
WORK="${WORK:-/kaggle/working}"
DATA_ROOT="${DATA_ROOT:-/kaggle/input/rsna-knee-abnormality-detection}"
EXP="${EXP:-configs/experiment/student_2p5d_effnetv2.yaml}"
FOLDS="${FOLDS:-$WORK/train_folds.csv}"
VOLUMES_CACHE="${VOLUMES_CACHE:-$WORK/volumes_cache}"
WEAK_LABELS="${WEAK_LABELS:-$WORK/weak_labels.parquet}"
TEACHER_DIR="${TEACHER_DIR:-$WORK/text_teacher}"

if [ ! -e "$WORK/.deps_ok" ] || [ "${FORCE_PIP:-0}" = '1' ]; then
  log 'installing python deps'
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements.txt
  [ "${WITH_MONAI:-0}" = '1' ] && "$PY" -m pip install -q 'monai>=1.3'
  mkdir -p "$WORK" && touch "$WORK/.deps_ok"
fi

# T4/P100/K80 lack bf16 tensor cores -> fall back to fp16-mixed for training.
EXTRA_SETS=''
if nvidia-smi -L 2>/dev/null | grep -Eqi 'T4|P100|P4|K80'; then
  log 'pre-Ampere GPU detected -> precision 16-mixed'
  EXTRA_SETS='train.trainer.params.precision=16-mixed'
fi

# --- 2. stage dispatch ------------------------------------------------------
run() { log "$*"; "$PY" "$@"; }

case "$STAGE" in
  volumes)
    run scripts/prepare_volumes.py --data-root "$DATA_ROOT" \
        --cache-dir "$VOLUMES_CACHE" --series-csv train_series.csv --workers 4 ;;
  folds)
    run scripts/make_folds.py --train-csv "$DATA_ROOT/train.csv" --out-csv "$FOLDS" ;;
  weak-labels)
    teacher_flag=''
    [ -f "$TEACHER_DIR/oof_probs.parquet" ] && teacher_flag="--teacher-dir $TEACHER_DIR"
    # shellcheck disable=SC2086
    run scripts/build_weak_labels.py \
        --config configs/labeling/text_teacher.yaml $teacher_flag ;;
  teacher)
    run scripts/train_text_teacher.py --config configs/labeling/text_teacher.yaml ;;
  student)
    # shellcheck disable=SC2086
    run scripts/train_image_student.py --config "$EXP" --set \
        "paths.volumes_cache=$VOLUMES_CACHE" "paths.folds_csv=$FOLDS" \
        "paths.weak_labels_parquet=$WEAK_LABELS" "paths.output_dir=$WORK" $EXTRA_SETS ;;
  self-train)
    run scripts/self_train.py --config "$EXP" \
        --student-oof "$WORK/predictions/$(basename "$EXP" .yaml)_oof.parquet" --round 2 ;;
  infer)
    run scripts/infer.py --config "$EXP" --test-csv "$DATA_ROOT/test.csv" \
        --ckpt-dir "$WORK/checkpoints" --out "$WORK/submission_${EXP##*/}.csv" ;;
  blend)
    run scripts/blend_submissions.py \
        --oof "$WORK/predictions/*_oof.parquet" --gold "$FOLDS" \
        --subs "$WORK/submission_*.csv" --out "$WORK/submission_blended.csv" ;;
  all)
    log 'full pipeline via scripts/run_all.sh'
    export DATA_ROOT WORK EXP PYTHON="$PY"
    bash scripts/run_all.sh ;;
  *)
    die "unknown stage '$STAGE' (volumes folds weak-labels teacher student self-train infer blend all)" ;;
esac

log "stage '$STAGE' complete"
