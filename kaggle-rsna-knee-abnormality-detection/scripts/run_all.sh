#!/usr/bin/env bash
# =============================================================================
# run_all.sh -- end-to-end orchestration of the pipeline stages.
#
# Kaggle reality: ~570 GB of DICOM vs 30 GB of scratch and a 12 h kernel
# wall. A full volumes_cache therefore CANNOT be built in one shot.
# Volumes are stream-decoded at train/infer time by default
# (src/knee/datasets/volume_store). Two OPTIONAL accelerators exist:
#   BUILD_CACHE=1    mint one bounded shard (SHARD=i NUM_SHARDS=n)
#   CACHE_VOLUMES=1  incremental caching: decode within VOL_MINUTES,
#                    auto-publishing <DATASET_NAME>-volNN datasets and
#                    deleting local copies on disk pressure; progress
#                    tracked in train_folds.csv (vol_shard column).
#
# Every stage is resumable: artifacts land under $WORK, finished image
# folds persist as per-fold OOF parquets and are skipped on re-run, and
# TIME_BUDGET_HOURS caps training via Lightning max_time.
#
# Usage:
#   DATA_ROOT=/path/to/rsna bash scripts/run_all.sh
# Required env:
#   DATA_ROOT   directory holding train.csv, train_series/, ...
# Optional env:
#   WORK        scratch/artifact dir        (default ./outputs)
#   FOLDS       folds csv                    (default $WORK/train_folds.csv)
#   EXP         experiment yaml              (configs/experiment/student_2p5d_effnetv2.yaml)
#   BUILD_CACHE 1 -> decode an optional volumes shard first (default 0)
#   CACHE_VOLUMES 1 -> incremental shard caching + dataset publishing
#   DATASET_NAME base name for shards (default ah2022-rsna-knee-abnormality-detection)
#   VOL_MINUTES/VOL_SHARD_SIZE/VOL_PUSH_EVERY_MINUTES cache-volumes knobs
#   SHARD/NUM_SHARDS/MIN_FREE_GB shard selection + free-disk guard
#   TIME_BUDGET_HOURS wall-clock cap per student invocation (default 11)
#   FOLDS_LIST  comma fold ids for kernel 5 (default: config's train_folds)
# =============================================================================
set -euo pipefail

DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the competition data directory}"
WORK="${WORK:-./outputs}"
EXP="${EXP:-configs/experiment/student_2p5d_effnetv2.yaml}"
FOLDS="${FOLDS:-$WORK/train_folds.csv}"
CACHE="${CACHE:-$WORK/volumes_cache}"
LABELS="$WORK/weak_labels.parquet"
DATASET_NAME="${DATASET_NAME:-ah2022-rsna-knee-abnormality-detection}"
PY="${PYTHON:-python}"

log() { printf '\n==> %s\n' "$*"; }
done_if() { [ -e "$1" ] && { log "skip $2 ($1 exists)"; return 0; }; }

mkdir -p "$WORK"

# --- kernel 1 (OPTIONAL): bounded volumes shard -----------------------------
if [ "${BUILD_CACHE:-0}" = '1' ]; then
  done_if "$CACHE/volumes_manifest.parquet" "kernel1:volumes" || \
    "$PY" scripts/prepare_volumes.py --data-root "$DATA_ROOT" \
         --cache-dir "$CACHE" --series-csv train_series.csv --workers 4 \
         --shard "${SHARD:-0}" --num-shards "${NUM_SHARDS:-1}" \
         --min-free-gb "${MIN_FREE_GB:-4}"
else
  log 'kernel1:volumes skipped (streaming decode is the default)'
fi

# --- kernel 2: folds --------------------------------------------------------
done_if "$FOLDS" "kernel2:folds" || \
  "$PY" scripts/make_folds.py --train-csv "$DATA_ROOT/train.csv" --out-csv "$FOLDS"

# --- kernel 2b (OPTIONAL): incremental volume-cache shards ------------------
# CACHE_VOLUMES=1 -> decode within VOL_MINUTES, auto-publishing
# <DATASET_NAME>-volNN datasets and deleting local npz copies on disk
# pressure; train_folds.csv gains the vol_shard tracking column.
if [ "${CACHE_VOLUMES:-0}" = '1' ]; then
  "$PY" scripts/cache_volumes.py --data-root "$DATA_ROOT" \
       --train-folds "$FOLDS" --work "$WORK" \
       --base-name "$DATASET_NAME" --series-csv train_series.csv \
       --minutes "${VOL_MINUTES:-480}" \
       --push-every-minutes "${VOL_PUSH_EVERY_MINUTES:-20}" \
       --shard-size "${VOL_SHARD_SIZE:-1500}" \
       --min-free-gb "${MIN_FREE_GB:-6}"
else
  log 'kernel2b:cache-volumes skipped (enable with CACHE_VOLUMES=1)'
fi

# --- kernel 3: weak labels (rules only; add teacher after kernel 4) ---------
done_if "$LABELS" "kernel3:weak-labels(rules)" || \
  "$PY" scripts/build_weak_labels.py \
      --config configs/labeling/text_teacher.yaml \
      $( [ -f "$WORK/text_teacher/oof_probs.parquet" ] && \
          echo "--teacher-dir $WORK/text_teacher" )

# --- kernel 4: text teacher -------------------------------------------------
done_if "$WORK/text_teacher/oof_probs.parquet" "kernel4:text-teacher" || \
  "$PY" scripts/train_text_teacher.py --config configs/labeling/text_teacher.yaml

# --- kernel 5: image student (fold-sharded, time-budgeted) ------------------
# No done_if here: the script itself skips completed folds and merges all
# persisted per-fold OOF parts, so partial shards resume correctly.
"$PY" scripts/train_image_student.py --config "$EXP" \
    ${FOLDS_LIST:+--folds "$FOLDS_LIST"} --set \
    "paths.folds_csv=$FOLDS" \
    "paths.weak_labels_parquet=$LABELS" "paths.output_dir=$WORK" \
    "train.time_budget_hours=${TIME_BUDGET_HOURS:-11}" \
    "train.resume=${RESUME:-1}"

# --- kernel 6: inference ----------------------------------------------------
"$PY" scripts/infer.py --config "$EXP" --test-csv "$DATA_ROOT/test.csv" \
    --ckpt-dir "$WORK/checkpoints" --out "$WORK/submission_${EXP##*/}.csv"

# --- kernel 7: blend (single member here; add members as they finish) -------
"$PY" scripts/blend_submissions.py \
    --oof "$WORK/predictions/*_oof.parquet" \
    --gold "$FOLDS" \
    --subs "$WORK/submission_*.csv" \
    --out "$WORK/submission_blended.csv"

log "DONE -> $WORK/submission_blended.csv"
