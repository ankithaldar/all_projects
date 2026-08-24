#!/usr/bin/env bash
# =============================================================================
# setup_structure.sh -- Production-grade project scaffold for
# RSNA Knee Abnormality Detection (Kaggle, macro-AUC over 12 targets).
#
# Idempotent: safe to re-run. Never overwrites existing files.
# Usage:
#   bash setup_structure.sh [PROJECT_ROOT]     # default: script directory
# =============================================================================
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
echo "==> Scaffolding RSNA knee solution at: ${ROOT}"

# ----------------------------------------------------------------- helpers --
mk()  { mkdir -p "$1"; }
touch_if_absent() { [ -e "$1" ] || touch "$1"; }

# ------------------------------------------------------------- top-level ----
mk "${ROOT}/configs/experiment"
mk "${ROOT}/configs/labeling"
mk "${ROOT}/data"                       # local-only scratch (gitignored)
mk "${ROOT}/docs"
mk "${ROOT}/notebooks"
mk "${ROOT}/scripts"
mk "${ROOT}/src/knee/config"
mk "${ROOT}/src/knee/data"
mk "${ROOT}/src/knee/labeling"
mk "${ROOT}/src/knee/models"
mk "${ROOT}/src/knee/training"
mk "${ROOT}/src/knee/inference"
mk "${ROOT}/src/knee/utils"
mk "${ROOT}/tests"

# --------------------------------------------------- package init files -----
for pkg in \
  "src/knee" \
  "src/knee/config" \
  "src/knee/data" \
  "src/knee/labeling" \
  "src/knee/models" \
  "src/knee/training" \
  "src/knee/inference" \
  "src/knee/utils"; do
  touch_if_absent "${ROOT}/${pkg}/__init__.py"
done

# ------------------------------------------------------- placeholder docs ---
touch_if_absent "${ROOT}/README.md"
touch_if_absent "${ROOT}/BLUEPRINT.md"
touch_if_absent "${ROOT}/.env.example"
touch_if_absent "${ROOT}/requirements.txt"
touch_if_absent "${ROOT}/Makefile"

for d in data outputs; do
  # Anchor to repo root ("/data/") so it does NOT ignore src/knee/data/
  grep -qx "^/${d}/$" "${ROOT}/.gitignore" 2>/dev/null || echo "/${d}/" >> "${ROOT}/.gitignore"
done

echo "==> Structure created."
find "${ROOT}" -type d -not -path '*/.git*' | sort | sed "s|${ROOT}|.|"
