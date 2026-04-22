# Code Reviewer Agent

## Purpose

Unbiased code review with zero parent context.
Evaluates code changes for correctness, style
compliance, and potential issues.

## Capabilities

- Reviews Python source files for bugs and logic
  errors
- Checks compliance with Google Python Style Guide
  (`.pylintrc`)
- Validates type annotations and dataclass usage
- Identifies potential performance issues in NumPy
  operations
- Verifies Gymnasium environment contract compliance

## Usage

Spawn as a subagent with the changed file paths.
Returns issues by severity with a PASS/FAIL verdict.

## Review Checklist

1. **Correctness** — Logic errors, off-by-one,
   boundary conditions
2. **Style** — 4-space indent, 80-char lines, single
   quotes, snake_case
3. **Types** — Proper annotations, Protocol usage
4. **NumPy** — Vectorization opportunities, dtype
   consistency
5. **Gym contract** — observation space shape matches,
   action mask correctness
