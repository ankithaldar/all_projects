#!/usr/bin/env bash
# Run all tests for the truck-carton project.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== truck-carton test suite ==="
echo ""

PYTEST_ARGS=(-v --tb=short)

# Coverage flag
if [[ "${1:-}" == "--cov" ]]; then
  PYTEST_ARGS+=(--cov=src/truck_carton --cov-report=term-missing)
  shift
fi

# Pass remaining args through
PYTEST_ARGS+=("$@")

echo "Running: pytest ${PYTEST_ARGS[*]}"
echo ""

python -m pytest "${PYTEST_ARGS[@]}"

EXIT_CODE=$?
echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "All tests passed."
else
  echo "Some tests failed (exit code $EXIT_CODE)."
fi

exit $EXIT_CODE
