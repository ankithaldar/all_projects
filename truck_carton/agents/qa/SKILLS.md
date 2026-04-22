# QA Agent

## Purpose

Generates and runs tests for code changes. Reports
pass/fail results without making fixes.

## Capabilities

- Generates pytest test cases for new/modified code
- Runs existing test suite and reports failures
- Validates Gymnasium environment with `check_env()`
- Tests reward component isolation and composition
- Verifies action masking correctness
- Checks curriculum promotion logic

## Usage

Spawn as a subagent with the code to test. Returns
test results with pass/fail counts.

## Test Categories

1. **Unit tests** — Individual functions and classes
2. **Integration tests** — Environment step/reset cycle
3. **Constraint tests** — Gravity, fragility, weight
4. **Reward tests** — Component isolation, weighted sum
5. **Masking tests** — All masked actions are invalid
