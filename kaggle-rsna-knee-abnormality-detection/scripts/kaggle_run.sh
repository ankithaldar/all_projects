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
#   volumes | folds | weak-labels | weak-labels-llm | teacher | student |
#   self-train | infer | blend | publish | all
#
# Kaggle budgets (30 GB disk vs ~570 GB data, 12 h/kernel):
# * Volumes are STREAM-DECODED from /kaggle/input at train/infer time --
#   building a full volumes_cache is impossible and NOT required.
# * The 'volumes' stage is an OPTIONAL accelerator producing one bounded,
#   resumable shard:  export SHARD=i NUM_SHARDS=n (and MIN_FREE_GB).
#   Publish each shard as its own versioned dataset and mount them all;
#   partial coverage is fine (misses fall back to live DICOM decode).
# * TIME_BUDGET_HOURS (default 11) caps training wall-clock per kernel
#   via Lightning max_time so checkpoints are written cleanly before
#   Kaggle's hard kill; finished folds persist as per-fold OOF parquets
#   and are skipped on re-run. Use FOLDS_LIST='0,1' to split folds
#   across kernels within the ~30 GPU-hour total.
#
# Cross-kernel state lives in ONE private Kaggle dataset (DATASET_NAME,
# default ah2022-rsna-knee-abnormality-detection). Add-ons -> Secrets ->
# KAGGLE_USERNAME + KAGGLE_KEY, then either:
#     bash scripts/kaggle_run.sh publish          # push when you choose
# or  export AUTO_PUBLISH=1                       # ...or after every stage
# Next kernel: Add Data -> Your Datasets -> that dataset, then
#     export PREV_OUTPUT=/kaggle/input/<dataset-slug>
# and the bootstrap copies checkpoints/OOF/folds/labels forward before
# the stage runs. Save Version remains a belt-and-braces backup.
#
# Multi-kernel handoff of OTHER artifacts: publish stage outputs as
# versioned datasets and point the consumer kernel at them via env:
#   export FOLDS=/kaggle/input/knee-folds-v1/train_folds.csv
#   export WEAK_LABELS=/kaggle/input/knee-weak-labels-v1/weak_labels.parquet
#
# Optional env:
#   REPO_DIR   repo location            (default: script's parent dir)
#   REPO_URL   used to clone when the repo is absent (standalone/curl usage)
#   GIT_REF    branch/tag to clone
#   EXP        experiment yaml          (configs/experiment/student_2p5d_effnetv2.yaml)
#   DATA_ROOT  data dir                 (/kaggle/input/competitions/rsna-knee-abnormality-detection)
#   WORK       scratch/artifact dir     (/kaggle/working)
#   WITH_MONAI 1 -> install monai (student_3d_resnet experiment)
#   FORCE_PIP  1 -> reinstall deps despite the session marker
#   TIME_BUDGET_HOURS  training wall-clock cap (default 11)
#   FOLDS_LIST  folds for the student stage, e.g. '0,1' (default: config)
#   RESUME      1 (default) -> interrupted folds resume from last.ckpt
#   PREV_OUTPUT prior kernel output mount(s), colon-separated -- copied
#               into $WORK before dispatch (fresh-container handoff)
#   DATASET_NAME private dataset receiving all artifacts
#                (default ah2022-rsna-knee-abnormality-detection)
#   AUTO_PUBLISH 1 -> run 'publish' automatically after each stage
#   PUBLISH_MESSAGE  version note for the dataset push
#   SHARD/NUM_SHARDS/MIN_FREE_GB  volumes-stage sharding + disk guard
# =============================================================================
# Multi-kernel lifecycle (each kernel = fresh container):
#   kernel N:   ... run stage ... then SAVE VERSION (persists /kaggle/working)
#   kernel N+1: Add Data -> Your Work -> select kernel N's output, then
#               export PREV_OUTPUT=/kaggle/input/<kernel-N-output-slug>
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
DATA_ROOT="${DATA_ROOT:-/kaggle/input/competitions/rsna-knee-abnormality-detection}"
EXP="${EXP:-configs/experiment/student_2p5d_effnetv2.yaml}"
FOLDS="${FOLDS:-$WORK/train_folds.csv}"
# No default cache: volumes stream straight from /kaggle/input. Point
# VOLUMES_CACHE at one dir or os.pathsep-joined shard mounts to use them
# as a read-only accelerator.
VOLUMES_CACHE="${VOLUMES_CACHE:-}"
WEAK_LABELS="${WEAK_LABELS:-$WORK/weak_labels.parquet}"
TEACHER_DIR="${TEACHER_DIR:-$WORK/text_teacher}"
TIME_BUDGET_HOURS="${TIME_BUDGET_HOURS:-11}"
DATASET_NAME="${DATASET_NAME:-ah2022-rsna-knee-abnormality-detection}"

# --- 1b. restore prior-kernel artifacts (fresh-container handoff) -----------
# Every Kaggle kernel boots a NEW container: previous outputs exist only
# as read-only /kaggle/input mounts (after Save Version) and
# /kaggle/working starts empty. PREV_OUTPUT (one dir or colon-separated
# several) is copied forward so finished folds skip, interrupted ones
# resume and the training script can keep writing into this session's
# writable $WORK. Mount layouts vary across Kaggle UIs (/kaggle/input/
# <slug> vs /kaggle/input/datasets/<owner>/<slug>): when PREV_OUTPUT is
# unset we auto-discover by slug with a depth-bounded find that never
# descends into the multi-hundred-GB data payload.
if [ -z "${PREV_OUTPUT:-}" ] && [ -d /kaggle/input ] && [ -n "$DATASET_NAME" ]; then
  DISCOVERED=$(find /kaggle/input -maxdepth 3 -type d \
    -name "$DATASET_NAME" 2>/dev/null | head -n1 || true)
  if [ -n "$DISCOVERED" ]; then
    log "discovered dataset mount at $DISCOVERED"
    PREV_OUTPUT="$DISCOVERED"
  fi
fi
if [ -n "${PREV_OUTPUT:-}" ]; then
  log 'restoring prior kernel artifacts -> '"$WORK"
  mkdir -p "$WORK/checkpoints" "$WORK/predictions"
  for src in ${PREV_OUTPUT//:/ }; do
    if [ ! -d "$src" ]; then
      log "warning: PREV_OUTPUT '$src' not mounted; skipping"
      continue
    fi
    if [ -d "$src/checkpoints" ]; then
      cp -a "$src/checkpoints/." "$WORK/checkpoints/"
    fi
    if [ -d "$src/predictions" ]; then
      cp -a "$src/predictions/." "$WORK/predictions/"
    fi
    # Single-file artifacts: only adopt when this session lacks them.
    if [ -f "$src/train_folds.csv" ] && [ ! -f "$FOLDS" ]; then
      cp -a "$src/train_folds.csv" "$FOLDS"
    fi
    if [ -f "$src/weak_labels.parquet" ] && [ ! -f "$WEAK_LABELS" ]; then
      cp -a "$src/weak_labels.parquet" "$WEAK_LABELS"
    fi
    if [ -d "$src/text_teacher" ] && [ ! -e "$TEACHER_DIR" ]; then
      cp -a "$src/text_teacher" "$TEACHER_DIR"
    fi
  done
  log "restored: $(find "$WORK/checkpoints" -name '*.ckpt' | wc -l) ckpts, $(ls "$WORK/predictions" 2>/dev/null | wc -l) oof files"
fi

if [ ! -e "$WORK/.deps_ok" ] || [ "${FORCE_PIP:-0}" = '1' ]; then
  # Fresh containers already ship a CUDA-matched torch/lightning/timm/
  # transformers stack -- blanket-installing requirements.txt would
  # waste ~10 min and risk replacing the GPU torch build. Install ONLY
  # what is actually missing; FORCE_PIP=1 restores the old behavior.
  # Optional trackers (neptune/wandb) stay out unless already present.
  log 'checking python deps'
  MISSING=$("$PY" - <<'EOF'
import importlib.util

REQUIRED = {
  'lightning': 'lightning>=2.2',
  'timm': 'timm>=1.0',
  'transformers': 'transformers>=4.44',
  'sentencepiece': 'sentencepiece',
  'albumentations': 'albumentations>=1.4',
  'cv2': 'opencv-python-headless',
  'pydicom': 'pydicom',
  'pylibjpeg': 'pylibjpeg',
  'gdcm': 'python-gdcm',
  'scipy': 'scipy',
  'sklearn': 'scikit-learn',
  'iterative_stratification': 'iterative-stratification',
  'omegaconf': 'omegaconf>=2.3',
  'pydantic': 'pydantic>=2.6',
  'dotenv': 'python-dotenv',
  'kaggle': 'kaggle',
}
missing = [
  spec for module, spec in REQUIRED.items()
  if importlib.util.find_spec(module) is None
]
try:
  import torch  # noqa: F401  # never replace the CUDA-matched build
except Exception:
  missing.insert(0, 'torch>=2.1')
print('\n'.join(missing))
EOF
)
  if [ -n "$MISSING" ]; then
    log "installing: $(printf '%s ' $MISSING)"
    "$PY" -m pip install -q --upgrade pip >/dev/null
    "$PY" -m pip install -q $MISSING
  else
    log 'all core deps already present in this image'
  fi
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

publish_artifacts() {
  # Push every resumable artifact to ONE private Kaggle dataset
  # (DATASET_NAME) so the next fresh-container kernel can PREV_OUTPUT it.
  run scripts/publish_dataset.py --work "$WORK" \
      --dataset-name "$DATASET_NAME" ${PUBLISH_MESSAGE:+--message "$PUBLISH_MESSAGE"}
}

case "$STAGE" in
  volumes)
    run scripts/prepare_volumes.py --data-root "$DATA_ROOT" \
        --cache-dir "${CACHE_DIR:-$WORK/volumes_cache}" \
        --series-csv train_series.csv --workers 4 \
        --shard "${SHARD:-0}" --num-shards "${NUM_SHARDS:-1}" \
        --min-free-gb "${MIN_FREE_GB:-4}" ;;
  folds)
    run scripts/make_folds.py --train-csv "$DATA_ROOT/train.csv" --out-csv "$FOLDS" ;;
  weak-labels)
    teacher_flag=''
    [ -f "$TEACHER_DIR/oof_probs.parquet" ] && teacher_flag="--teacher-dir $TEACHER_DIR"
    # --out-parquet: write to canonical $WEAK_LABELS (the student stage
    # and the published dataset both expect it at $WORK root), NOT the
    # text_teacher config dir.
    # shellcheck disable=SC2086
    run scripts/build_weak_labels.py \
        --config configs/labeling/text_teacher.yaml \
        --out-parquet "$WEAK_LABELS" $teacher_flag ;;
  weak-labels-llm)
    # LLM tier via OpenRouter (OPENROUTER_API_KEY secret). Writes the
    # fused parquet AND its resume cache to canonical $WORK paths so
    # they land in the SAME published dataset as everything else.
    run scripts/build_weak_labels_llm.py \
        --config configs/labeling/text_teacher.yaml \
        --out "$WEAK_LABELS" --cache "$WORK/llm_label_cache.parquet" \
        --model "${LLM_MODEL:-stealth/ox-alpha}" \
        --concurrency "${LLM_CONCURRENCY:-2}" ;;
  teacher)
    run scripts/train_text_teacher.py --config configs/labeling/text_teacher.yaml ;;
  student)
    cache_flag=''
    [ -n "$VOLUMES_CACHE" ] && \
      cache_flag="paths.volumes_cache=$VOLUMES_CACHE"
    fold_flag=''
    [ -n "${FOLDS_LIST:-}" ] && fold_flag="--folds $FOLDS_LIST"
    # RESUME=1 (default): interrupted folds continue from fold<N>/last.ckpt
    resume_flag="train.resume=${RESUME:-1}"
    # shellcheck disable=SC2086
    run scripts/train_image_student.py --config "$EXP" $fold_flag --set \
        "paths.folds_csv=$FOLDS" \
        "paths.weak_labels_parquet=$WEAK_LABELS" "paths.output_dir=$WORK" \
        "train.time_budget_hours=$TIME_BUDGET_HOURS" "$resume_flag" \
        $cache_flag $EXTRA_SETS ;;
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
  publish)
    publish_artifacts ;;
  all)
    log 'full pipeline via scripts/run_all.sh'
    export DATA_ROOT WORK EXP PYTHON="$PY"
    bash scripts/run_all.sh ;;
  *)
    die "unknown stage '$STAGE' (volumes folds weak-labels weak-labels-llm teacher student self-train infer blend publish all)" ;;
esac

# AUTO_PUBLISH=1: push artifacts after every successful stage so a 12 h
# kill never loses more than one stage of work (publish itself skips).
if [ "${AUTO_PUBLISH:-0}" = '1' ] && [ "$STAGE" != 'publish' ]; then
  log 'AUTO_PUBLISH enabled'
  publish_artifacts
fi

log "stage '$STAGE' complete"
