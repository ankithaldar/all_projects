#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Console entry point for the forecast ML framework.

Declared in pyproject.toml as the sole console script,
``forecast-ml``. Delegates to
``forecast_ml_framework.cli.main`` (train/predict/serve/run);
``--version`` and ``--help`` are preserved. Importing this
module is side-effect free.
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

      return kedro_main()
    except ImportError:
      print('forecast-ml: train, predict, serve, run')
      return 0
  return cli_main(argv)


if __name__ == '__main__':
  raise SystemExit(main())
