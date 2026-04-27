#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Tests ==="

# Use uv run to ensure correct venv
uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=90 "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "All tests passed with >= 90% coverage."
else
  echo ""
  echo "Tests failed (exit code $EXIT_CODE)."
fi

exit $EXIT_CODE
