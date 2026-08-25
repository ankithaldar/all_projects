#!/usr/bin/env bash
# ==============================================================================
# World of Supply - Google Colab bootstrap
#
# Usage in a Colab notebook cell:
#   !GIT_URL=https://github.com/<you>/<your-repo>.git bash colab_run.sh
#
# Or upload the project folder to /content/world_of_supply first, then:
#   !bash colab_run.sh
#
# Optional environment variables:
#   GIT_URL           git repo to clone (default: use existing PROJECT_DIR)
#   PROJECT_DIR       project location (default: /content/world_of_supply)
#   OUT_DIR           artifact output (default: /content/wos_outputs)
#   TRAIN_ITERATIONS  PPO iterations to run (default: 3)
#   NUM_EPI           baseline episodes (default: 2)
#   MOUNT_DRIVE       1 = save artifacts to Google Drive (Colab only)
#
# The script also works outside Colab (e.g. any Linux box with python3).
# ==============================================================================
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/content/world_of_supply}"
GIT_URL="${GIT_URL:-}"
OUT_DIR="${OUT_DIR:-/content/wos_outputs}"
TRAIN_ITERATIONS="${TRAIN_ITERATIONS:-3}"
NUM_EPI="${NUM_EPI:-2}"
export PYTHONUNBUFFERED=1

echo '== [1/7] Environment =='
pick_python() {
  local candidate
  for candidate in "${PYTHON_BIN:-}" python3 python3.13 python3.12 python3.11 python3.10; do
    [ -z "$candidate" ] && continue
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)' 2>/dev/null; then
        echo "$candidate"
        return 0
      fi
    fi
  done
  echo 'ERROR: no Python 3.10-3.13 found (project pins ray<3 / torch support)' >&2
  return 1
}
PYTHON_BIN="$(pick_python)" || exit 1
echo "Using interpreter: $PYTHON_BIN ($($PYTHON_BIN --version))"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L | head -1 || true
else
  echo 'No GPU detected - running CPU mode (fine for the demo training)'
fi
free -h | awk 'NR<=2 {print}'

echo '== [2/7] Fetch project =='
if [ -n "$GIT_URL" ]; then
  rm -rf "$PROJECT_DIR"
  git clone --depth 1 "$GIT_URL" "$PROJECT_DIR"
elif [ ! -d "$PROJECT_DIR" ]; then
  echo "ERROR: $PROJECT_DIR not found. Set GIT_URL or upload the project there." >&2
  exit 1
fi
cd "$PROJECT_DIR"
git log --oneline -1 2>/dev/null || true

echo '== [3/7] Install dependencies =='
"$PYTHON_BIN" -m pip install -q --upgrade pip
"$PYTHON_BIN" -m pip install -q -e .
"$PYTHON_BIN" -m pip install -q pytest
"$PYTHON_BIN" - <<'PY'
import torch, ray, gymnasium
print(f"torch {torch.__version__} | ray {ray.__version__} | gymnasium {gymnasium.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
PY

echo '== [4/7] Test suite =='
"$PYTHON_BIN" -m pytest tests -q 2>&1 | tail -1

echo '== [5/7] Prepare outputs =='
if [ "${MOUNT_DRIVE:-0}" = "1" ]; then
  "$PYTHON_BIN" - <<'PY'
from google.colab import drive
drive.mount('/content/drive')
PY
  OUT_DIR='/content/drive/MyDrive/world_of_supply_outputs'
fi
mkdir -p "$OUT_DIR/frames"

echo '== [6/7] Demo: simulate + render + baseline =='
"$PYTHON_BIN" main.py simulate --ticks 60 --seed 42 --render-dir "$OUT_DIR/frames" 2>&1 | grep -E '^--- tick (1|60)/' || true
echo "frames rendered: $(ls "$OUT_DIR/frames" | wc -l) (in $OUT_DIR/frames)"
"$PYTHON_BIN" main.py baseline --episodes "$NUM_EPI" --seed 7

echo '== [7/7] PPO training =='
"$PYTHON_BIN" main.py train --iterations "$TRAIN_ITERATIONS" --toy-only 2>&1 | tee "$OUT_DIR/training.log" | grep -E '^curriculum|^iteration' || true

echo
echo "All done. Artifacts: $OUT_DIR"
[ "${MOUNT_DRIVE:-0}" = "1" ] && echo 'Artifacts are also in your Google Drive at MyDrive/world_of_supply_outputs.'
