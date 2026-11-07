#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Console entry point for the forecast ML framework.

Declared in pyproject.toml as the ``forecast-ml`` and
``forecast-ml-framework`` console scripts. Delegates to
``forecast_ml_framework.cli.main`` (train/predict/serve/run);
``--version`` and ``--help`` are preserved. Importing this
module is side-effect free. On the Kedro fallback path the
caller's ``argv`` is forwarded via ``sys.argv`` (V12), so
Kedro sees the same arguments the CLI would have parsed.
'''

import importlib.metadata
import sys

try:
  _DIST_VERSION = importlib.metadata.version('forecast-ml-framework')
except importlib.metadata.PackageNotFoundError:
  _DIST_VERSION = '0.1.0'

_VERSION = f'forecast-ml {_DIST_VERSION}'


def main(argv=None):
  '''Run the console entry point.

  Args:
    argv: Optional argument list (defaults to sys.argv).

  Returns:
    Exit code as int; 0 on success.
  '''
  args = list(sys.argv[1:] if argv is None else argv)
  if '--version' in args:
    print(_VERSION)
    return 0
  try:
    from forecast_ml_framework.cli import main as cli_main
  except ImportError:
    try:
      from kedro.framework.cli import main as kedro_main

      # Kedro's Click entry point reads sys.argv; forward
      # the caller's argv so it is never discarded (V12).
      if argv is not None:
        saved, sys.argv = sys.argv, [sys.argv[0], *argv]
        try:
          return kedro_main()
        finally:
          sys.argv = saved
      return kedro_main()
    except ImportError:
      print('forecast-ml: train, predict, serve, run')
      return 0
  return cli_main(argv)


if __name__ == '__main__':
  raise SystemExit(main())
