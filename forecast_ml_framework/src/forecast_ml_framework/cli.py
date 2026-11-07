#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''CLI wrapper for forecast-ml.

Usage: forecast-ml train|predict|serve|run --env local

Delegates to ``kedro run`` when kedro is available, otherwise
prints the planned command. Stdlib argparse only.

Optional API-key auth is supported via FORECAST_API_KEY in the
serving app (see README Serving API); no Docker is provided.

Registry aliases (V12): ``train`` runs the ``modeling``
pipeline, ``predict`` runs ``scoring``, ``run`` runs the
default pipeline (no ``--pipeline`` flag), and ``serve`` runs
the uvicorn app (never a Kedro pipeline). The twelve registry
display names are real subparsers mapping to
``kedro run --pipeline <alias>``: ``data``, ``preprocessing``,
``feature_selection``, ``modeling``, ``evaluation``,
``predict`` (scoring alias), ``scoring``, ``eda``,
``monitoring``, ``retraining``, ``cost_tracking`` and
``interpretability``. In the registry, ``data`` /
``preprocessing`` / ``feature_selection`` are served by
``pipelines.data_prep`` and ``evaluation`` / ``modeling`` /
``predict`` by ``pipelines.modeling`` (``data_prep``).
``serve`` calls
``serving.app.validate_bind_host`` before spawning uvicorn
and exits 2 on RuntimeError, so a non-loopback bind without a
valid API key is refused at the CLI entry point.

Name divergence (``predict``, intentional and pinned by
``test_cli.py``): ``forecast-ml predict`` runs the
``scoring`` pipeline, whereas
``kedro run --pipeline predict`` runs the ``modeling``
pipeline (registry ``predict`` maps to
``pipelines.modeling`` in ``pipeline_registry``). The same
name therefore selects different pipelines depending on the
entry point; aligning the two would break the pinned CLI
mapping, so the divergence is documented, not removed.
'''

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _load_kedro_cli():
  '''Lazily load Kedro's top-level Click group, or None.

  Kedro 1.7 loads project commands from
  ``<package>.cli:cli`` and fails badly when ``cli`` is
  missing, so this module must expose it. Guarded so the
  argparse ``forecast-ml`` entry point keeps working when
  Kedro/Click are unavailable.

  Returns:
    Kedro's top-level ``click.Group`` (``kedro``), or None
    when Kedro/Click cannot be imported.
  '''
  try:
    # pylint: disable=import-outside-toplevel
    import click

    # pylint: disable=import-outside-toplevel
    from kedro.framework.cli.cli import cli as kedro_cli_group
    if not isinstance(kedro_cli_group, click.Group):
      return None
    return kedro_cli_group
  except Exception:  # pylint: disable=broad-except
    return None


# Kedro-compatible group: Kedro's top-level group when
# available, else None (argparse entry point unaffected).
cli = _load_kedro_cli()

# CLI command -> Kedro pipeline. ``run`` uses the default
# pipeline (no --pipeline flag); ``serve`` is not a Kedro
# pipeline at all (uvicorn, see _planned_command).
_COMMAND_PIPELINES = {
  'train': 'modeling',
  'predict': 'scoring',
  'run': None,
  'serve': None,
}

_ALLOWED_ENVS = frozenset({'base', 'local', 'gcp', 'aws'})

#: Registry display names exposed as real subparsers
#: (``predict`` is already a subparser, so it is excluded
#: here to avoid a duplicate registration).
_REGISTRY_ALIASES = (
  'data', 'preprocessing', 'feature_selection', 'modeling',
  'evaluation', 'scoring', 'eda', 'monitoring', 'retraining',
  'cost_tracking', 'interpretability')


def _env_allowed(env: str) -> bool:
  '''True when ``env`` is whitelisted (rejects traversal).'''
  return env in _ALLOWED_ENVS


def _planned_command(
    command: str, env: str, host: str = '127.0.0.1',
    port: int = 8000, artifact: str | None = None) -> list[str]:
  '''Build planned command for a CLI command.

  Args:
    command: CLI command (train/predict/serve/run); any other
      value is treated as a Kedro pipeline name directly.
    env: Kedro environment.
    host: Bind host for ``serve``.
    port: Bind port for ``serve``.
    artifact: Model artifact path for ``serve``.

  Returns:
    Planned argv: ``train`` -> modeling pipeline, ``predict``
    -> scoring pipeline, ``run`` -> default pipeline (no
    ``--pipeline``), ``serve`` -> uvicorn serving the FastAPI
    app in ``forecast_ml_framework.serving.app``.
  '''
  if command == 'serve':
    # Artifact is passed via MODEL_ARTIFACT_PATH env in
    # _run_kedro, never as a uvicorn flag.
    return [
      'uvicorn', 'forecast_ml_framework.serving.app:app',
      '--host', host, '--port', str(port)]
  pipeline = _COMMAND_PIPELINES.get(command, command)
  if pipeline is None:
    return ['kedro', 'run', '--env', env]
  return ['kedro', 'run', '--pipeline', pipeline, '--env', env]


def _conf_source_parent() -> Path | None:
  '''Return settings.CONF_SOURCE parent, or None if unavailable.'''
  try:
    # pylint: disable=import-outside-toplevel
    from forecast_ml_framework.settings import CONF_SOURCE
    return Path(CONF_SOURCE).resolve().parent
  except Exception:  # pylint: disable=broad-except
    return None


def _find_project_root() -> Path | None:
  '''Find nearest ancestor that is strictly this project.

  Strict: ``[tool.kedro] package_name`` must equal
  ``forecast_ml_framework`` AND (``conf`` exists OR the
  candidate equals the CONF_SOURCE parent).

  Returns:
    Project root directory, or None when no ancestor qualifies.
  '''
  conf_parent = _conf_source_parent()
  for candidate in (Path.cwd(), *Path.cwd().parents):
    pyproject = candidate / 'pyproject.toml'
    try:
      text = pyproject.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
      continue
    package_name = None
    try:
      import tomllib  # pylint: disable=import-outside-toplevel
      data = tomllib.loads(text)
      package_name = data.get('tool', {}).get(
        'kedro', {}).get('package_name')
    except Exception:  # pylint: disable=broad-except
      package_name = None
    if package_name is None:
      # Fallback scan for the kedro section's package_name.
      for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('package_name'):
          package_name = stripped.split('=', 1)[-1].strip().strip(
            '\'"')
          break
    if package_name != 'forecast_ml_framework':
      continue
    if (candidate / 'conf').exists():
      return candidate
    if conf_parent is not None and candidate == conf_parent:
      return candidate
    if conf_parent is not None and candidate.resolve() == conf_parent:
      return candidate
  return None


def _run_kedro(
    command: str, env: str, host: str = '127.0.0.1',
    port: int = 8000, artifact: str | None = None) -> int:
  '''Run planned command or print it when the binary is absent.

  Args:
    command: CLI command to run.
    env: Kedro environment.
    host: Bind host for ``serve``.
    port: Bind port for ``serve``.
    artifact: Model artifact path for ``serve``.

  Returns:
    Process exit code, or 2 when the Kedro project root
    cannot be found, the env is invalid, or ``serve`` is
    refused by ``validate_bind_host`` (non-loopback bind
    without a valid API key) -- in which case nothing is
    spawned.
  '''
  if not _env_allowed(env):
    print('forecast-ml: error: invalid env: ' + str(env),
          file=sys.stderr)
    return 2
  if command == 'serve':
    try:
      # pylint: disable=import-outside-toplevel
      from forecast_ml_framework.serving.app import validate_bind_host
      validate_bind_host(host)
    except RuntimeError as exc:
      print('forecast-ml: error: ' + str(exc), file=sys.stderr)
      return 2
  root = _find_project_root()
  if root is None:
    print('forecast-ml: error: Kedro project not found '
          '(run from the forecast_ml_framework directory '
          'or install the package)', file=sys.stderr)
    return 2
  planned = _planned_command(
    command, env, host=host, port=port, artifact=artifact)
  if shutil.which(planned[0]) is None:
    print('planned command: ' + ' '.join(planned))
    return 0
  run_env = None
  if command == 'serve' and artifact is not None:
    run_env = dict(os.environ)
    run_env['MODEL_ARTIFACT_PATH'] = str(Path(artifact).resolve())
  return subprocess.run(
    planned, check=False, cwd=str(root), env=run_env).returncode


def _list_pipelines() -> int:
  '''Print one ``<name> <module>`` line per registered pipeline.'''
  try:
    # pylint: disable=import-outside-toplevel
    from forecast_ml_framework import pipeline_registry
    pipelines = pipeline_registry.register_pipelines()
    for name in sorted(pipelines):
      # pylint: disable=protected-access
      module_name = pipeline_registry._PIPELINE_MODULES[name]
      print(f'{name} {module_name}')
    return 0
  except Exception as exc:  # pylint: disable=broad-except
    print(f'forecast-ml: error: {exc}', file=sys.stderr)
    return 2


def _validate_config(env: str) -> int:
  '''Validate config for ``env`` and print ``config ok``.'''
  if not _env_allowed(env):
    print('forecast-ml: error: invalid env: ' + str(env),
          file=sys.stderr)
    return 2
  try:
    # pylint: disable=import-outside-toplevel
    from forecast_ml_framework.core import config_schema
    plain = config_schema.load_config(env)
    # Unreachable-by-design: no model_extra warning branch.
    # ParametersConfig sets extra='forbid', so
    # validate_parameters raises on unknown top-level keys
    # (handled by the except below -> exit 2) instead of
    # returning extras worth warning about.
    config_schema.validate_parameters(plain['parameters'])
    print(f'config ok ({env})')
    return 0
  except Exception as exc:  # pylint: disable=broad-except
    print(f'forecast-ml: error: {exc}', file=sys.stderr)
    return 2


def _dry_run(pipeline: str, env: str) -> int:
  '''Print the planned command and node names for a pipeline.'''
  try:
    # pylint: disable=import-outside-toplevel
    from forecast_ml_framework import pipeline_registry
    pipelines = pipeline_registry.register_pipelines()
    if pipeline not in pipelines:
      print('forecast-ml: error: unknown pipeline: ' + pipeline,
            file=sys.stderr)
      return 2
    planned = ['kedro', 'run', '--pipeline', pipeline, '--env', env]
    print('planned command: ' + ' '.join(planned))
    for node in pipelines[pipeline].nodes:
      print(node.name)
    return 0
  except Exception as exc:  # pylint: disable=broad-except
    print(f'forecast-ml: error: {exc}', file=sys.stderr)
    return 2


def _build_parser() -> argparse.ArgumentParser:
  '''Build CLI argument parser.'''
  parser = argparse.ArgumentParser(
    prog='forecast-ml',
    description='Train, predict, or serve a forecast model.',
  )
  sub = parser.add_subparsers(dest='command', required=True)
  for name in ('train', 'predict', 'serve', 'run'):
    child = sub.add_parser(name, help=f'Run the {name} pipeline.')
    child.add_argument(
      '--env', default='local', help='Kedro environment.')
    child.add_argument(
      '--artifact', default=None, help='Model artifact path.')
    child.add_argument('--host', default='127.0.0.1')
    child.add_argument('--port', type=int, default=8000)
  for alias in _REGISTRY_ALIASES:
    child = sub.add_parser(
        alias, help=f'Run the {alias} pipeline.')
    child.add_argument(
      '--env', default='local', help='Kedro environment.')
    child.add_argument(
      '--artifact', default=None, help='Model artifact path.')
    child.add_argument('--host', default='127.0.0.1')
    child.add_argument('--port', type=int, default=8000)
  child = sub.add_parser(
    'list-pipelines', help='List registered pipelines.')
  child.add_argument(
    '--env', default='local', help='Kedro environment.')
  child = sub.add_parser(
    'validate-config', help='Validate the config for an env.')
  child.add_argument(
    '--env', default='local', help='Kedro environment.')
  child = sub.add_parser('dry-run', help='Print a planned run.')
  child.add_argument('pipeline', help='Pipeline name.')
  child.add_argument(
    '--env', default='local', help='Kedro environment.')
  return parser


def main(argv: list[str] | None = None) -> int:
  '''Run CLI entry point.

  Args:
    argv: Argument list, defaults to sys.argv.

  Returns:
    Process exit code.
  '''
  args = _build_parser().parse_args(argv)
  env = getattr(args, 'env', None)
  if env is not None and not _env_allowed(env):
    print('forecast-ml: error: invalid env: ' + str(env),
          file=sys.stderr)
    return 2
  if args.command == 'list-pipelines':
    return _list_pipelines()
  if args.command == 'validate-config':
    return _validate_config(args.env)
  if args.command == 'dry-run':
    return _dry_run(args.pipeline, args.env)
  artifact_abs = None
  if args.artifact is not None:
    artifact_abs = str(Path(args.artifact).resolve())
    if not Path(artifact_abs).is_file():
      print('forecast-ml: error: artifact not found: ' + args.artifact,
            file=sys.stderr)
      return 2
  return _run_kedro(
    args.command, args.env, host=args.host, port=args.port,
    artifact=artifact_abs)


if __name__ == '__main__':
  sys.exit(main())
