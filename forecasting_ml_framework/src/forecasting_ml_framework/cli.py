#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The custom ``run`` command and the project command.

The ``run`` command differs from the stock one in three ways, and each difference
exists because of something the framework's own machinery needs.

**(a) Spark bootstrap before session creation.** The Spark tuning document is
applied *here* and only here, so this is the single point at which the tuning
takes effect. Every other component delegates session construction to the
platform builder, which returns the already-configured session — which is why the
tuning had to be applied at the earliest possible moment rather than at each
call site.

**(b) Slice-label injection.** Six string tokens are injected into the parameter
dictionary immediately before the session is created. They are not data; they are
consumed as the ``data_type`` argument, and they are the mechanism by which one
function serves both calibrated and uncalibrated evaluation.

**(c) Explicit session context.** The run is executed with a named session and a
chosen runner, so a run's identity and its execution mode are both explicit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import click
from kedro.framework.cli.utils import CONTEXT_SETTINGS, split_string
from kedro.framework.session import KedroSession
from kedro.utils import load_obj

from forecasting_ml_framework import constants
from forecasting_ml_framework.observability.logging import configure_logging, get_logger

LOGGER = get_logger(__name__)

#: The click group, registered as the project's ``run`` entry point. A
#: ``click.Group`` is constructed directly rather than via ``click.group()``,
#: which without a function argument returns a bare decorator rather than a group.
commands = click.Group(name='run', context_settings=CONTEXT_SETTINGS)
project_commands = click.Group(name='project', context_settings=CONTEXT_SETTINGS)

#: Alias for :data:`project_commands`, under the name the ``kedro.project_commands``
#: entry point declares. Kedro resolves that attribute by name from the installed
#: distribution, so the symbol must exist under exactly the declared spelling or the
#: sub-commands are unreachable. Aliasing rather than renaming keeps the two from
#: ever diverging again.
project = project_commands


def _bootstrap_spark(params: dict[str, Any], conf_env: str, conf_dir: str | Path | None = None) -> Any:
  """Construct the process-wide Spark session with the tuning document applied.

  Args:
    params: The run parameters.
    conf_env: The configuration environment.
    conf_dir: The configuration root.

  Returns:
    The live ``SparkSession``.
  """
  from forecasting_ml_framework.platform.spark import get_spark_session  # noqa: PLC0415

  session, _ = get_spark_session(params=params, apply_tuning=True, conf_dir=conf_dir, conf_env=conf_env)
  return session


@commands.command()
@click.option('--from-nodes', 'from_nodes', default='', help='Run only the nodes downstream of these nodes.')
@click.option('--to-nodes', 'to_nodes', default='', help='Run only the nodes upstream of these nodes.')
@click.option('--nodes', 'nodes', default='', help='Run only these nodes.')
@click.option('--runner', '-r', default='SequentialRunner', show_default=True, help='The runner to use.')
@click.option('--tags', 'tags', default='', help='Run only nodes with these tags.')
@click.option('--env', '-e', 'env', default='', help='The Kedro configuration environment.')
@click.option('--conf-source', '-c', 'conf_source', default=None, help='Path to a custom configuration directory.')
@click.option('--namespace', 'namespace', default='', help='The namespace for the run.')
@click.option('--runner-args', 'runner_args', default='{}', help='Arguments passed to the runner.')
@click.option('--pipeline', '-p', 'pipeline', default='', help='The pipeline to run.')
@click.option('--params', '-P', 'params', default='', help='Runtime parameters, as key:value pairs.')
@click.option('--parallel', 'is_parallel', flag_value=True, default=False, help='Run nodes in parallel.')
@click.option('--open-vis', flag_value=True, default=False, help='Serve the visualisation and block.')
@click.option('--service-account', 'is_service_account', flag_value=True, default=False,
              help='Reserved for a service-account execution path.')
@click.argument('package_name', nargs=1, default='.')
def run(
  pipeline: str,
  from_nodes: str,
  to_nodes: str,
  nodes: str,
  runner: str,
  tags: str,
  env: str,
  namespace: str,
  runner_args: str,
  conf_source: str,
  params: str,
  is_parallel: bool,
  open_vis: bool,
  is_service_account: bool,
  package_name: str,
) -> None:
  """Run a pipeline with the framework's Spark bootstrap and slice labels.

  Args:
    pipeline: The pipeline name.
    from_nodes: Run only the nodes downstream of these.
    to_nodes: Run only the nodes upstream of these.
    nodes: Run only these nodes.
    runner: The runner class name.
    tags: Run only nodes carrying these tags.
    env: The Kedro configuration environment.
    namespace: The namespace for the run.
    runner_args: Extra runner arguments, as JSON.
    conf_source: A custom configuration directory.
    params: Runtime parameters, as ``key:value`` pairs.
    is_parallel: Whether to run nodes in parallel.
    open_vis: Whether to serve the visualisation and block.
    is_service_account: Reserved for a service-account execution path.
    package_name: The project package.

  Raises:
    click.BadParameter: If the environment is not resolvable.
  """
  del is_service_account
  configure_logging()

  conf_env = env or 'base'
  runtime_params = _parse_runtime_params(params)
  # The slice tokens are string dispatch, not data. They are injected here so
  # that one evaluation function can serve both the calibrated and the
  # uncalibrated column families, keyed on a `calib_` prefix.
  runtime_params.update(constants.SLICE_TOKEN_MAP)

  _bootstrap_spark(runtime_params, conf_env, conf_source)
  LOGGER.info('Running pipeline', extra={'pipeline': pipeline, 'env': conf_env, 'package': package_name})

  with KedroSession.create(
    env=conf_env,
    conf_source=conf_source,
    package_name=package_name,
    runtime_params=runtime_params,
  ) as session:
    session.run(
      tags=split_string(tags),
      node_names=split_string(nodes),
      from_nodes=split_string(from_nodes),
      to_nodes=split_string(to_nodes),
      pipeline_name=pipeline,
      namespace=namespace or None,
      runner=load_obj(runner),
      runner_args=_parse_runner_args(runner_args),
      asynchronous=is_parallel,
      open_vis=open_vis,
    )


@project_commands.command('pipeline')
@click.argument('pipeline_name', required=False)
def pipeline_description(pipeline_name: str = '') -> None:
  """Print the registered graph, because it is not statically declared.

  The pipeline graph is a function of five environment variables read at
  registration time, which is what lets one codebase serve an unbounded number of
  model instances. The cost is that it cannot be introspected without executing
  the factories. This command closes that gap.

  Args:
    pipeline_name: An optional pipeline to describe. All are described when
      omitted.
  """
  from forecasting_ml_framework.pipelines.registry import describe_pipelines, register_pipelines  # noqa: PLC0415

  text = describe_pipelines(register_pipelines())
  click.echo(text if not pipeline_name else _filter_pipeline(text, pipeline_name))


@project_commands.command('routines')
def routine_catalogue() -> None:
  """Print the available preprocessing routines and their parameter surfaces.

  A configuration registers a curated subset of the available transforms, so the
  ones it does not activate are not discoverable from the parameters file alone.
  This command makes the whole catalogue visible from the CLI.

  """
  from forecasting_ml_framework.preprocessing.routines import AVAILABLE_ROUTINES  # noqa: PLC0415

  rows = []
  for name, routine in sorted(AVAILABLE_ROUTINES.items()):
    # Routines take (global_params, params); the catalogue needs only the
    # declared schema, so both arguments are passed empty.
    instance = routine(global_params=None, params={})
    rows.append((name, ', '.join(sorted(instance.get_default_dict()))))
  width = max(len(row[0]) for row in rows)
  click.echo(f'Available preprocessing routines ({len(rows)}):\n')
  for name, keys in rows:
    click.echo(f'  {name.ljust(width)}  {keys}')
  click.echo('\nActivate one with a single-key block in data_prep_params carrying skip: False.')


@commands.command('version')
def show_version() -> None:
  """Print the framework version and the artefact format version.

  The artefact format version matters operationally: model handles are pickles,
  and loading one under a different library version fails deep inside
  ``pickle`` with an unreadable message. Printing both at run start makes the
  pairing visible in the log.

  """
  from forecasting_ml_framework import __version__  # noqa: PLC0415
  from forecasting_ml_framework.utils.storage import ARTEFACT_FORMAT_VERSION  # noqa: PLC0415

  click.echo(f'forecasting-ml-framework {__version__} (artefact format {ARTEFACT_FORMAT_VERSION})')


def _parse_runner_args(raw: str) -> dict[str, Any]:
  """Parse the runner arguments from JSON.

  A passed this string through ``eval``. Having criticised ``eval`` on
  configuration data elsewhere in this framework, applying it to a command-line
  argument would be inconsistent; JSON is the right format for a flat argument
  map and it cannot execute anything.

  Args:
    raw: The raw JSON string.

  Returns:
    The parsed mapping.

  Raises:
    click.BadParameter: If the string is not a JSON object.
  """
  if not raw or not raw.strip():
    return {}
  try:
    parsed = json.loads(raw)
  except json.JSONDecodeError as error:
    raise click.BadParameter(f'--runner-args must be a JSON object: {error}') from error
  if not isinstance(parsed, dict):
    raise click.BadParameter('--runner-args must be a JSON object')
  return parsed


def _parse_runtime_params(params: str) -> dict[str, Any]:
  """Parse a ``--params`` string into a flat mapping.

  Args:
    params: The raw parameter string.

  Returns:
    A mapping of key to value.
  """
  from forecasting_ml_framework.utils.text import split_params  # noqa: PLC0415

  if not params:
    return {}
  parsed = split_params(params)
  # The CLI surface is flat, so a dotted key is flattened into its leaf name as
  # well; the configuration rewriter addresses keys by their top-level names.
  flattened: dict[str, Any] = {}
  for key, value in parsed.items():
    flattened[key] = value
    if '.' in key:
      flattened[key.split('.')[-1]] = value
  return flattened


def _filter_pipeline(text: str, pipeline_name: str) -> str:
  """Extract one pipeline's section from the graph description.

  Args:
    text: The full description.
    pipeline_name: The pipeline to isolate.

  Returns:
    That pipeline's section, or the full text when the name is unknown.
  """
  blocks = text.split('\n\n')
  matched = [block for block in blocks if block.startswith(pipeline_name)]
  return '\n\n'.join(matched) if matched else text


def main() -> None:
  """The console-script entry point."""
  commands()


if __name__ == '__main__':  # pragma: no cover
  os.environ.setdefault('FORECASTING_ML_LOG_LEVEL', 'INFO')
  main()
