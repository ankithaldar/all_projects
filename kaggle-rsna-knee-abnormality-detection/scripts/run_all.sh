#!/usr/bin/env bash
# =============================================================================
# run_all.sh -- end-to-end orchestration of the 7-kernel pipeline.
#
# On Kaggle each stage runs in its own kernel (see BLUEPRINT section 9);
# this script chains the identical commands locally or on a beefy box.
# Every stage is resumable: artifacts land under $WORK and stages skip
# themselves when their outputs already exist.
#
# Usage:
#   DATA_ROOT=/path/to/rsna bash scripts/run_all.sh
# Required env:
#   DATA_ROOT   directory holding train.csv, train_series/, ...
# Optional env:
#   WORK        scratch/artifact dir        (default ./outputs)
#   FOLDS       folds csv                    (default $WORK/train_folds.csv)
#   EXP         experiment yaml              (configs/experiment/student_2p5d_effnetv2.yaml)
# =============================================================================
set -euo pipefail

DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the competition data directory}"
WORK="${WORK:-./outputs}"
EXP="${EXP:-configs/experiment/student_2p5d_effnetv2.yaml}"
FOLDS="${FOLDS:-$WORK/train_folds.csv}"
CACHE="$WORK/volumes_cache"
LABELS="$WORK/weak_labels.parquet"
PY="${PYTHON:-python}"

log() { printf '\n==> %s\n' "$*"; }
done_if() { [ -e "$1" ] && { log "skip $2 ($1 exists)"; return 0; }; }

mkdir -p "$WORK"

# --- kernel 1: volumes ------------------------------------------------------
done_if "$CACHE/volumes_manifest.parquet" "kernel1:volumes" || \
  "$PY" scripts/prepare_volumes.py --data-root "$DATA_ROOT" \
       --cache-dir "$CACHE" --series-csv train_series.csv --workers 4

# --- kernel 2: folds --------------------------------------------------------
done_if "$FOLDS" "kernel2:folds" || \
  "$PY" scripts/make_folds.py --train-csv "$DATA_ROOT/train.csv" --out-csv "$FOLDS"

# --- kernel 3: weak labels (rules only; add teacher after kernel 4) ---------
done_if "$LABELS" "kernel3:weak-labels(rules)" || \
  "$PY" scripts/build_weak_labels.py \
      --config configs/labeling/text_teacher.yaml \
      $( [ -f "$WORK/text_teacher/oof_probs.parquet" ] && \
          echo "--teacher-dir $WORK/text_teacher" )

# --- kernel 4: text teacher -------------------------------------------------
done_if "$WORK/text_teacher/oof_probs.parquet" "kernel4:text-teacher" || \
  "$PY" scripts/train_text_teacher.py --config configs/labeling/text_teacher.yaml

# --- kernel 5: image student (all folds) ------------------------------------
STUDENT_OOF="$WORK/predictions/$(basename "$EXP" .yaml)_oof.parquet"
done_if "$STUDENT_OOF" "kernel5:image-student" || \
  "$PY" scripts/train_image_student.py --config "$EXP" --set \
      "paths.volumes_cache=$CACHE" "paths.folds_csv=$FOLDS" \
      "paths.weak_labels_parquet=$LABELS" "paths.output_dir=$WORK"

# --- kernel 5b: self-train round 2 (optional; uncomment to enable) ----------
# LABELS_R2="$WORK/weak_labels_round2.parquet"
# done_if "$LABELS_R2" "kernel5b:self-train" || \
#   "$PY" scripts/self_train.py --config "$EXP" --student-oof "$STUDENT_OOF" --round 2
# Re-run kernel 5 with paths.weak_labels_parquet=$LABELS_R2 before blending.

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
