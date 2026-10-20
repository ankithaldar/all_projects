#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Replay a run from its manifest.

A manifest you cannot act on is documentation, not reproducibility.  This
command reads a ``manifest.json``, re-computes the SHA-256 of every
recorded input, and compares it with what the manifest recorded.

An input that no longer hashes the same is a **clear error**, not a
warning: the audio changed underneath the run, so the run that the
manifest describes is not the run that would happen now.  The command
reports **every** mismatch by name and refuses to run.

The recorded package versions are compared as well, but a difference
there is **reported, not fatal**: a patch bump is not a different
experiment, and refusing to replay because ``pandas`` moved from 2.2 to
2.3 would be theatre.  The comparison is there so the operator can see
it.

Usage:

    birdcall-reproduce manifest.json --check-only
    birdcall-reproduce manifest.json --run
'''

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from birdcall.provenance import package_versions
from birdcall.provenance import sha256_file


def load_manifest(path: Path) -> dict:
  '''Reads a manifest.

  Args:
    path: The path of the manifest.

  Returns:
    The manifest document.

  Raises:
    FileNotFoundError: If the manifest does not exist.
    ValueError: If the manifest is not a mapping.
  '''
  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(f'Manifest not found: {path}')
  document = json.loads(path.read_text(encoding='utf-8'))
  if not isinstance(document, dict):
    raise ValueError(f'Manifest {path} is not a mapping.')
  return document


def check_inputs(manifest: dict) -> list[str]:
  '''Re-hashes every recorded input and compares it with the manifest.

  Args:
    manifest: The manifest document.

  Returns:
    One message per mismatching input, naming it.
  '''
  problems = []
  for name, entry in (manifest.get('inputs') or {}).items():
    path = Path(entry['path'])
    expected = entry.get('sha256')
    if not path.exists():
      problems.append(f'input {name!r} is missing: {path}')
      continue
    found = sha256_file(path)
    if found != expected:
      problems.append(f'input {name!r} changed: {path} hashes to {found}, '
                      f'the manifest records {expected}')
  return problems


def check_packages(manifest: dict) -> list[str]:
  '''Compares the recorded package versions with the installed ones.

  A difference is reported but is **not** fatal: a patch bump is not a
  different experiment, but it is worth saying out loud.

  Args:
    manifest: The manifest document.

  Returns:
    One message per differing package.
  '''
  recorded = manifest.get('packages') or {}
  installed = package_versions()
  differences = []
  for name, version in recorded.items():
    current = installed.get(name)
    if current is None:
      differences.append(f'package {name!r} is recorded at {version} but is '
                         'not installed')
    elif current != version:
      differences.append(f'package {name!r} is recorded at {version} but '
                         f'{current} is installed')
  return differences


def run_command(command: list[str]) -> int:
  '''Runs the command a manifest records.

  Args:
    command: The command, as a list of arguments.

  Returns:
    The command's exit code.
  '''
  completed = subprocess.run(list(command), check=False)
  return completed.returncode


def build_parser() -> argparse.ArgumentParser:
  '''Builds the command line parser.

  Returns:
    The parser.
  '''
  parser = argparse.ArgumentParser(
      description='Verify, and optionally replay, a run from its manifest.')
  parser.add_argument('manifest', type=Path,
                      help='The path of the manifest.json.')
  group = parser.add_mutually_exclusive_group()
  group.add_argument('--check-only', action='store_true',
                     help='Verify the inputs and exit.')
  group.add_argument('--run', action='store_true',
                     help='Re-run the command the manifest records.')
  return parser


def main(argv: Optional[list[str]] = None) -> int:
  '''Runs the replay.

  Args:
    argv: The command line arguments, or ``None`` for the real ones.

  Returns:
    The exit code.  Zero means every check passed.
  '''
  args = build_parser().parse_args(argv)
  manifest = load_manifest(args.manifest)
  problems = check_inputs(manifest)
  for problem in problems:
    print(f'ERROR: {problem}', file=sys.stderr)
  for difference in check_packages(manifest):
    print(f'WARNING: {difference}', file=sys.stderr)
  if problems:
    print('ERROR: refusing to run, the inputs do not match the manifest.',
          file=sys.stderr)
    return 1
  if not args.run:
    print('Every recorded input matches the manifest.')
    return 0
  command = manifest.get('command') or []
  if not command:
    print('ERROR: the manifest records no command to run.', file=sys.stderr)
    return 1
  print(f'Running: {" ".join(str(part) for part in command)}')
  return run_command(command)


if __name__ == '__main__':
  raise SystemExit(main())
