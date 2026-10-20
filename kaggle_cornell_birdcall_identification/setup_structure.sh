#!/usr/bin/env bash
# Provisions the environment for both backends.
#
#   ./setup_structure.sh
#
# Python must be 3.12 or 3.13: TensorFlow has no stable wheel for 3.14.

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV="${VENV:-.venv}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${HERE}"

if [ ! -d "${VENV}" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python "${PYTHON_VERSION}" "${VENV}"
  else
    python"${PYTHON_VERSION}" -m venv "${VENV}"
  fi
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-tf.txt
python -m pip install -e .

# The build-time-only dependency: librosa builds the Slaney mel matrix,
# which is then committed so that neither backend needs it at runtime.
python -m pip install 'librosa==0.10.2'

echo
echo "Ready. Try:"
echo "  source ${VENV}/bin/activate"
echo "  pytest -q"
echo "  birdcall-train-pl --config src/birdcall/config_params/experiments/example_config/example_config.yaml"
