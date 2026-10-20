#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The convenience launcher.

It dispatches to the three entry points of the project:

``train``
    Train one experiment, on the backend the configuration selects.
``vote``
    Run the 13-model vote and write the submission.
``reproduce``
    Replay an experiment from its manifest.

Every subcommand takes the same arguments as the corresponding console
script, and the console scripts remain the supported interface -- this
launcher is a convenience, not an API.
'''

import argparse
import sys

COMMANDS = ('train', 'train-pl', 'train-tf', 'vote', 'reproduce')


def _entry_point(command: str):
  '''Resolves the entry point of a command.

  The imports are inside the function, so that running one backend does
  not import the other one's framework.  The cross-backend parity tests
  import both in one process, and that import order is delicate.

  Args:
    command: The name of the command.

  Returns:
    The callable that runs the command.
  '''
  if command in ('train', 'train-pl'):
    from birdcall import train_pl  # pylint: disable=import-outside-toplevel
    return train_pl.main
  if command == 'train-tf':
    from birdcall import train_tf  # pylint: disable=import-outside-toplevel
    return train_tf.main
  if command == 'vote':
    from birdcall import run_vote  # pylint: disable=import-outside-toplevel
    return run_vote.main
  from birdcall import reproduce  # pylint: disable=import-outside-toplevel
  return reproduce.main


def main(argv: list[str] | None = None) -> int:
  '''Dispatches to the selected entry point.

  Args:
    argv: The arguments, or ``None`` for the ones of the process.

  Returns:
    The exit code of the command.
  '''
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('command', choices=COMMANDS,
                      help='What to run.')
  arguments, rest = parser.parse_known_args(argv)
  return _entry_point(arguments.command)(rest)


if __name__ == '__main__':
  sys.exit(main())
