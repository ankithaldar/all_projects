#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Developer entry point.

Exposes the framework's registration and inspection helpers without a cluster.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
  """Run one of the developer utilities.

  Args:
    argv: The argument vector.

  Returns:
    A process exit code.
  """
  from forecasting_ml_framework.cli import commands  # noqa: PLC0415

  parser = argparse.ArgumentParser(
    prog='forecasting-ml', description='Developer entry point for the forecasting ML framework.'
  )
  parser.add_argument('command', choices=('pipeline', 'routines', 'version', 'run'), help='The utility to run.')
  known, remainder = parser.parse_known_args(argv)

  if known.command == 'pipeline':
    from forecasting_ml_framework.pipelines.registry import describe_pipelines  # noqa: PLC0415

    print(describe_pipelines())  # noqa: T201 - a CLI utility prints
    return 0
  if known.command == 'routines':
    from forecasting_ml_framework.preprocessing.routines import AVAILABLE_ROUTINES  # noqa: PLC0415

    # Every routine's constructor takes `(global_params, params)`. Calling
    # `routine()` with no arguments raised `TypeError` on the first entry, so
    # `main.py routines` -- the documented way to inspect the transform
    # catalogue -- printed a traceback instead of the catalogue. The empty bags
    # are correct here: `get_default_dict` reads only class-level defaults and
    # performs no I/O, so no configuration is needed to read the surface.
    for name, routine in sorted(AVAILABLE_ROUTINES.items()):
      declared = routine(global_params=None, params={}).get_default_dict()
      print(f'{name}: {sorted(declared)}')  # noqa: T201 - a CLI utility prints
    return 0

  if known.command == 'version':
    from forecasting_ml_framework import __version__  # noqa: PLC0415

    print(__version__)  # noqa: T201 - a CLI utility prints
    return 0

  # `run` is dispatched through the click group, which is how the orchestrator
  # invokes it. The command name is restored into the argument vector because the
  # argparse pass above consumed it as a positional; without this the group is
  # entered with no sub-command and prints its help, which is what a bare
  # `main.py version` used to do.
  sys.argv = ['forecasting-ml', known.command, *remainder]
  commands()
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
