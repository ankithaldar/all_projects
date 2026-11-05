#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Console entry point for the forecast ML framework.

Declared in pyproject.toml as both ``forecast-ml`` and
``forecastin-ml``. Delegates to
``forecast_ml_framework.cli.main`` (train/predict/serve/run);
falls back to Kedro, then to help text, when the CLI is
unavailable. Importing this module is side-effect free.
'''

import argparse


def _build_parser():
  '''Build the fallback argument parser.

  Returns:
    Configured argparse.ArgumentParser with --help support.
  '''
  parser = argparse.ArgumentParser(
    prog='forecast-ml',
    description='Forecast ML framework console.',
  )
  parser.add_argument(
    '--version',
    action='version',
    version='forecast-ml 0.1.0',
  )
  return parser


def main(argv=None):
  '''Run the console entry point.

  Args:
    argv: Optional argument list (defaults to sys.argv).

  Returns:
    Exit code as int; 0 on success.
  '''
  try:
    from forecast_ml_framework.cli import main as cli_main
  except ImportError:
    parser = _build_parser()
    parser.parse_args(argv)
    try:
      from kedro.framework.cli import main as kedro_main

      return kedro_main()
    except ImportError:
      parser.print_help()
      return 0
  return cli_main(argv)


if __name__ == '__main__':
  raise SystemExit(main())
