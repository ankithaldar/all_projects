#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''CLI entry point for the truck-carton application.

Thin dispatcher: forwards to scripts/train.py,
scripts/evaluate.py, and scripts/visualize.py via
subprocess without duplicating their logic.

Example:
  python main.py --help
  python main.py train --timesteps 1000
  python main.py evaluate --episodes 10
  python main.py visualize --mode grid
'''

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

_COMMANDS: dict[str, str] = {
  'train': 'Train the packing agent (scripts/train.py)',
  'evaluate': 'Evaluate an agent (scripts/evaluate.py)',
  'visualize': 'Visualize results (scripts/visualize.py)',
}


def build_parser() -> argparse.ArgumentParser:
  '''Build the top-level argument parser.

  Returns:
    Parser with one subparser per supported command.
  '''
  parser = argparse.ArgumentParser(
    description='Truck-carton RL command-line entry point'
  )
  subs = parser.add_subparsers(
    dest='command', metavar='command', title='commands'
  )
  for name, help_text in _COMMANDS.items():
    subs.add_parser(name, help=help_text)
  return parser


def _script_path(command: str) -> Path:
  '''Resolve the delegate script for a command.

  Args:
    command: Subcommand name, e.g. 'train'.

  Returns:
    Path to the script under scripts/.
  '''
  here = Path(__file__).resolve().parent
  return here / 'scripts' / (command + '.py')


def _dispatch(command: str, extra_args: list[str]) -> int:
  '''Run the delegate script as a subprocess.

  Args:
    command: Subcommand name, e.g. 'train'.
    extra_args: Unparsed trailing args passed through.

  Returns:
    Child exit code, or 1 when the script is missing
    or the child was killed by a signal.
  '''
  script = _script_path(command)
  if not script.is_file():
    print(f'error: no such script: {script}', file=sys.stderr)
    return 1
  proc = subprocess.run([sys.executable, str(script), *extra_args], check=False)
  if proc.returncode < 0:
    return 1
  return proc.returncode


def main(argv: Sequence[str] | None = None) -> int:
  '''Dispatch CLI args to the matching subcommand.

  Args:
    argv: Optional argument list (defaults to sys.argv).

  Returns:
    Process exit code: 0 on success or --help, 2 when
    no subcommand is given, otherwise the child code.
  '''
  parser = build_parser()
  ns, extra = parser.parse_known_args(argv)
  if ns.command is None:
    parser.print_help(sys.stdout)
    return 2
  return _dispatch(ns.command, extra)


if __name__ == '__main__':
  raise SystemExit(main())
