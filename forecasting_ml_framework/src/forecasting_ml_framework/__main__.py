#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The bootstrap entry point, invoked as the Spark driver's main file.

The scheduler submits it as::

    python <model_root>/src/<entry_point>.py run --pipeline=<name> --env=<conf_env> \
      --params=k1:v1,k2:v2,...

Its purpose is to make runtime parameters resolvable by the configuration system
*before* the orchestration framework starts, which it achieves by rewriting the
configuration documents on disk. See
:mod:`forecasting_ml_framework.platform.bootstrap` for the rewrite semantics.

Run directly::

    python src/forecasting_ml_framework/__main__.py \\
      run --pipeline=fe_training --env=base \\
      --params=suffix:params,model_key:demo_retention,run_date:2026-09-29
"""

from __future__ import annotations

import sys
from pathlib import Path

from forecasting_ml_framework.platform.bootstrap import bootstrap

#: The repository directory name under the platform mount root. Overridable via
#: the environment so one image can serve several checkouts.
DEFAULT_MODEL_REPO = 'forecasting_ml_framework'


def main(argv: list[str] | None = None) -> None:
  """Rewrite the configuration for this run, then hand over to the orchestrator.

  Args:
    argv: The argument vector. Defaults to ``sys.argv[1:]``.

  Raises:
    SystemExit: Always, because ``bootstrap`` terminates by handing over to
      Kedro's CLI, which owns the process from that point.
  """
  import os  # noqa: PLC0415

  arguments = list(argv if argv is not None else sys.argv[1:])
  conf_env = _read_flag(arguments, '--env') or 'base'
  model_repo = os.getenv('FORECASTING_ML_MODEL_REPO', DEFAULT_MODEL_REPO)
  # When the process is already running from inside the checkout (local
  # development, the test-suite), the checkout is the workspace; on a cluster the
  # repository is mounted at the platform root and the platform resolves it, so
  # the environment override is the only signal that a different root is wanted.
  model_root = os.getenv('FORECASTING_ML_MODEL_ROOT') or str(Path(__file__).resolve().parents[2])
  raw_params = _read_flag(arguments, '--params') or ''

  bootstrap(
    model_repo=model_repo,
    conf_env=conf_env,
    raw_params=raw_params,
    argv=arguments,
    model_root=model_root,
  )
  # `bootstrap` exits via sys.exit; this guard exists so a linter sees the control
  # flow and so an interactive invocation does not fall through.
  raise SystemExit(0)


def _read_flag(arguments: list[str], flag: str) -> str | None:
  """Read a ``--flag=value`` argument from the vector.

  Args:
    arguments: The argument vector.
    flag: The flag name, including its leading dashes.

  Returns:
    The flag's value, or ``None`` when absent.
  """
  prefix = f'{flag}='
  for argument in arguments:
    if argument.startswith(prefix):
      return argument.split('=', 1)[1]
  return None


if __name__ == '__main__':
  main()
