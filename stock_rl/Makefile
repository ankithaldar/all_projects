# Single place where tooling is invoked, so nobody has to remember flags.
# pylint does not search parent directories for a config file, so --rcfile
# is mandatory: without it pylint silently uses its own 4-space defaults and
# flags every correctly indented line in this 2-space codebase.

PYLINT_RC := ../.pylintrc
UV       := uv

.PHONY: help sync test lint check clean

help:
	@echo "make sync   install dependencies (uv sync)"
	@echo "make test   run tests with coverage"
	@echo "make lint   run pylint against $(PYLINT_RC)"
	@echo "make check  lint + test (run this before every push)"
	@echo "make clean  remove caches and build output"

sync:
	$(UV) sync

test:
	$(UV) run pytest --cov

lint:
	$(UV) run pylint --rcfile=$(PYLINT_RC) src/stock_rl tests

check: lint test

clean:
	rm -rf .pytest_cache .coverage htmlcov dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
