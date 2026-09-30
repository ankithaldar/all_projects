#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bootstrap: make runtime parameters resolvable *before* the orchestrator starts.

This is the least conventional part of the system, so it is worth stating
precisely. The orchestrator resolves ``${namespace:key}`` references against
namespaces it knows: ``params``, ``globals``, ``runtime_params``. The framework
needs a value to be resolvable from the *highest-priority* channel that supplies
it, and it cannot express that in a configuration document. So the bootstrap
rewrites the documents on disk before the orchestrator reads them.

The rewrite establishes a three-tier precedence by text substitution:

.. code-block:: text

  globals document    (per-model,  version-controlled)     lowest
  parameters document (per-project defaults)
  runtime parameters  (per-run,     from the scheduler)    highest

For each key in the globals document, exactly one of two mutually exclusive
branches applies, so a value can never resolve from both channels:

* the key **was** supplied at runtime -> every ``${KEY`` becomes
  ``${runtime_params:KEY``;
* the key was **not** supplied -> every ``${KEY`` becomes ``${globals:KEY``.

The suffix namespace is resolved separately and asymmetrically, and deliberately
so: the parameters document is rewritten to the *default* suffix token, so the
node layer can look configuration up by a stable name, while the catalog is
repointed at the *actual* suffix, so the graph and the catalog agree.
``except Exception`` that only printed. A rewrite failure therefore produced a
run against partially-rewritten configuration and surfaced minutes later as an
unresolvable interpolation error, separated from its cause by the whole startup
sequence. The rewrite now fails fast, before the orchestrator starts.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forecasting_ml_framework.exceptions import ConfigurationRewriteError
from forecasting_ml_framework.observability.logging import configure_logging, get_logger
from forecasting_ml_framework.platform.environment import resolve_environment

LOGGER = get_logger(__name__)

#: Configuration documents that are rewritten.
PARAMETERS_FILENAME = 'parameters.yml'
GLOBALS_FILENAME = 'globals.yml'
CATALOG_PREFIX = 'catalog'


@dataclass(frozen=True)
class RewritePlan:
  """The resolved decisions for one configuration rewrite.

  Attributes:
    conf_dir: The configuration directory being rewritten.
    suffix: The model-instance namespace token.
    runtime_keys: Top-level keys supplied as runtime parameters.
    globals_keys: Top-level keys declared by the globals document.
    touched: The files that were modified, for the run log.
  """

  conf_dir: Path
  suffix: str
  runtime_keys: frozenset[str] = field(default_factory=frozenset)
  globals_keys: frozenset[str] = field(default_factory=frozenset)
  touched: tuple[str, ...] = ()

  def route(self, key: str) -> str | None:
    """Return the namespace a key should be routed to.

    Args:
      key: The bare configuration key.

    Returns:
      The namespace to bind, or ``None`` when the key is not a framework
      channel key and must be left alone.
    """
    if key in self.runtime_keys:
      return 'runtime_params'
    if key in self.globals_keys:
      return 'globals'
    return None


def resolve_workspace_path(
  model_repo: str,
  is_hosted_platform: bool,
  mount_root: str = '/mnt',
  tmp_root: str = '/tmp',
) -> str:
  """Choose the workspace root for the current platform.

  Args:
    model_repo: The repository directory name.
    is_hosted_platform: Whether the run executes on the hosted notebook platform,
      which mounts the repository read-only under ``/mnt``.
    mount_root: The mount root used in hosted-platform mode.
    tmp_root: The root used otherwise.

  Returns:
    The absolute workspace path.
  """
  root = mount_root if is_hosted_platform else tmp_root
  return str(Path(root) / model_repo)


def load_globals_keys(conf_dir: Path) -> dict[str, Any]:
  """Read the globals document and return its top-level keys and values.

  Args:
    conf_dir: The configuration directory.

  Returns:
    The globals document as a mapping. An absent document yields an empty
    mapping, which simply disables the middle precedence tier.
  """
  import yaml  # noqa: PLC0415

  path = conf_dir / GLOBALS_FILENAME
  if not path.is_file():
    LOGGER.warning('No globals document found; the middle precedence tier is disabled', extra={'path': str(path)})
    return {}
  with path.open(encoding='utf-8') as handle:
    loaded = yaml.safe_load(handle) or {}
  if not isinstance(loaded, dict):
    raise ConfigurationRewriteError(
      f'The globals document must be a mapping, found {type(loaded).__name__}',
      path=str(path),
    )
  return loaded


def build_plan(conf_dir: Path, suffix: str, runtime_params: dict[str, Any]) -> RewritePlan:
  """Build the rewrite plan for a run.

  Args:
    conf_dir: The configuration directory.
    suffix: The model-instance namespace token.
    runtime_params: The parsed runtime parameters.

  Returns:
    The plan.
  """
  globals_document = load_globals_keys(conf_dir)
  return RewritePlan(
    conf_dir=conf_dir,
    suffix=suffix,
    runtime_keys=frozenset(runtime_params),
    globals_keys=frozenset(globals_document),
  )


def rewrite_parameters(plan: RewritePlan) -> Path | None:
  """Rewrite the parameters document.

  Two transformations apply. The suffix namespace is resolved to the **default**
  token, so the node layer can address configuration blocks by a stable name
  regardless of which model is running. Bare channel keys are then routed to
  their precedence namespace.

  Args:
    plan: The rewrite plan.

  Returns:
    The path that was written, or ``None`` when no parameters document exists.

  Raises:
    ConfigurationRewriteError: If the document is missing or unreadable.
  """
  from forecasting_ml_framework.constants import DEFAULT_SUFFIX  # noqa: PLC0415

  path = plan.conf_dir / PARAMETERS_FILENAME
  if not path.is_file():
    raise ConfigurationRewriteError(
      f'No parameters document found at {path}. The bootstrap cannot establish configuration '
      'precedence without it.',
      path=str(path),
    )

  text = _read(path)
  text = text.replace('_${suffix}:', f'_{DEFAULT_SUFFIX}:')
  text = _route_keys(text, plan)
  _write(path, text)
  return path


def rewrite_catalogs(plan: RewritePlan) -> list[Path]:
  """Rewrite every catalog document in the configuration directory.

  Three transformations apply: the suffix namespace is resolved to the
  **actual** token so the graph and the catalog agree; bare channel keys not
  supplied at runtime are routed to the globals namespace; and every supplied
  key is routed to the runtime-parameter namespace.

  Args:
    plan: The rewrite plan.

  Returns:
    The paths that were written.
  """
  written: list[Path] = []
  for path in sorted(plan.conf_dir.glob(f'{CATALOG_PREFIX}*.yml')):
    text = _read(path)
    text = text.replace('_${suffix}:', f'_{plan.suffix}:')
    text = _route_keys(text, plan)
    _write(path, text)
    written.append(path)
  return written


def apply_rewrite(plan: RewritePlan) -> tuple[RewritePlan, list[Path]]:
  """Execute the full rewrite and return the plan plus the files touched.

  Args:
    plan: The rewrite plan.

  Returns:
    A two-tuple of the plan with its ``touched`` field populated, and the list
    of written paths.

  Raises:
    ConfigurationRewriteError: If any step fails. The error is raised rather
      than logged so the run never starts against partially-rewritten
      configuration.
  """
  try:
    written = [p for p in (rewrite_parameters(plan), *rewrite_catalogs(plan)) if p is not None]
  except ConfigurationRewriteError:
    raise
  except Exception as error:  # noqa: BLE001 - a issue was here
    raise ConfigurationRewriteError(
      f'Configuration rewrite failed: {error}. Refusing to start the orchestrator against '
      'partially-rewritten configuration, because the resulting failure would surface minutes '
      'later as an unresolvable interpolation error or a missing dataset.',
      conf_dir=str(plan.conf_dir),
      error=str(error),
    ) from error

  completed = RewritePlan(
    conf_dir=plan.conf_dir,
    suffix=plan.suffix,
    runtime_keys=plan.runtime_keys,
    globals_keys=plan.globals_keys,
    touched=tuple(str(path) for path in written),
  )
  LOGGER.info(
    'Configuration rewrite complete',
    extra={'files': [str(p) for p in written], 'suffix': plan.suffix, 'runtime_keys': sorted(plan.runtime_keys)},
  )
  return completed, written


def prepare_workspace(
  model_repo: str,
  conf_env: str,
  is_hosted_platform: bool,
  runtime_params: dict[str, Any],
  model_root: str | Path | None = None,
  model_conf_dir: str = 'conf',
) -> tuple[Path, RewritePlan, list[Path]]:
  """Prepare the workspace and apply the configuration rewrite.

  Args:
    model_repo: The repository directory name under the platform mount root.
    conf_env: The configuration environment.
    is_hosted_platform: Whether the run executes on the hosted notebook
      platform, which mounts the repository read-only under ``/mnt``.
    runtime_params: The parsed runtime parameters.
    model_root: An explicit workspace root, overriding platform resolution. Used
      by local development and by the test-suite.
    model_conf_dir: The configuration directory name.

  Returns:
    A three-tuple of the workspace path, the completed plan, and the files
    written.

  Raises:
    ConfigurationRewriteError: If the configuration directory does not exist or
      the rewrite fails.
  """
  root = Path(model_root) if model_root else Path(resolve_workspace_path(model_repo, is_hosted_platform))
  conf_dir = root / model_conf_dir / conf_env if conf_env else root / model_conf_dir / 'base'
  if not conf_dir.is_dir():
    raise ConfigurationRewriteError(
      f'Configuration directory {conf_dir} does not exist. The environment {conf_env!r} must have '
      'a configuration directory, or fall back to "base".',
      conf_dir=str(conf_dir),
      conf_env=conf_env,
    )

  suffix = str(runtime_params.get('suffix', 'params'))
  plan = build_plan(conf_dir, suffix, runtime_params)
  completed, written = apply_rewrite(plan)
  publish_runtime_parameters(runtime_params)
  return root, completed, written


#: The runtime parameters of the current process, published by
#: :func:`publish_runtime_parameters` during the configuration rewrite.
#:
#: This is the store the ``${runtime:NAME}`` resolver reads. It exists because the
#: resolver's original implementation read a ``FORECASTING_ML_PARAM_<name>``
#: environment variable that nothing in the framework ever wrote, so every
#: reference resolved to the empty string -- and an empty string is a legal value
#: for every configuration type, so the failure was silent until much later.
_RUNTIME_PARAMETERS: dict[str, Any] = {}


def publish_runtime_parameters(runtime_params: dict[str, Any]) -> None:
  """Record the run's runtime parameters for the ``${runtime:NAME}`` resolver.

  Published at the same moment as the configuration rewrite, because that is the
  point at which the run's parameters are known to be final: they have been
  parsed, and they are the same values the rewrite has just bound into the
  documents. Publishing earlier would let a resolver observe a partial set.

  Args:
    runtime_params: The parsed runtime parameters.
  """
  _RUNTIME_PARAMETERS.clear()
  _RUNTIME_PARAMETERS.update(runtime_params or {})


def runtime_parameter(name: str) -> Any:
  """Return one published runtime parameter.

  Args:
    name: The parameter name.

  Returns:
    The value, or ``None`` when the parameter was not supplied for this run.
  """
  return _RUNTIME_PARAMETERS.get(name)


def bootstrap(
  model_repo: str,
  conf_env: str,
  raw_params: str,
  argv: list[str] | None = None,
  model_root: str | Path | None = None,
) -> None:
  """Entry point executed as the Spark driver's main file.

  The scheduler invokes this as::

      python <model_root>/src/<entry_point>.py run --pipeline=<name> --env=<conf_env> \
        --params=k1:v1,k2:v2,...

  Args:
    model_repo: The repository directory name under the platform mount root.
    conf_env: The configuration environment.
    raw_params: The raw ``--params`` string.
    argv: The argument vector. Defaults to ``sys.argv``.
    model_root: The workspace directory, when the process is already running from
      inside the checkout. On a cluster the repository is mounted at the platform
      root and the platform resolves the location itself, so this is ``None``
      there; locally it is the checkout, and passing it is what stops the
      workspace from being resolved a second time against a platform mount that
      does not exist.
  """
  from kedro.framework.cli import main as kedro_main  # noqa: PLC0415

  from forecasting_ml_framework.utils.text import split_params  # noqa: PLC0415

  configure_logging()
  arguments = list(argv if argv is not None else sys.argv[1:])
  runtime_params = _extract_raw_params(arguments) or split_params(raw_params or '')

  is_hosted = resolve_environment(runtime_params).is_hosted
  root = prepare_workspace(model_repo, conf_env, is_hosted, runtime_params, model_root=model_root)[0]

  os.chdir(root)
  for candidate in (root, root / 'src'):
    if str(candidate) not in sys.path:
      sys.path.insert(0, str(candidate))
  # Container images without a CA bundle break every outbound TLS call.
  os.environ.setdefault('CURL_CA_BUNDLE', '/etc/ssl/certs/ca-bundle.crt')
  _write_telemetry_opt_in(root)

  sys.argv[0] = 'kedro'
  sys.argv[1:] = _rebuild_arguments(arguments, root, conf_env, runtime_params)
  LOGGER.info('Handing over to the orchestrator', extra={'pipeline_arguments': sys.argv[1:]})
  sys.exit(kedro_main())


def _write_telemetry_opt_in(root: Path) -> None:
  """Record the telemetry opt-in, where the root allows it.

  A workspace mounts the repository read-only, so an unguarded write here raised
  ``PermissionError`` and killed the run before the orchestrator was even entered
  -- a failure with no connection to the work the run was attempting. The file is
  an opt-in rather than a requirement, so a root that cannot take it is logged and
  skipped rather than treated as fatal.

  Args:
    root: The workspace root.
  """
  try:
    Path(root / '.telemetry').write_text('allow\n', encoding='utf-8')
  except OSError as error:
    LOGGER.info(
      'The telemetry opt-in could not be written; the root is not writable. This is expected on a '
      'read-only workspace mount and does not affect the run.',
      extra={'root': str(root), 'error': str(error)},
    )


def _extract_raw_params(arguments: list[str]) -> dict[str, Any]:
  """Parse a ``--params=`` argument out of the raw argument vector.

  Args:
    arguments: The raw argument vector.

  Returns:
    The parsed parameter mapping, or an empty mapping.
  """
  from forecasting_ml_framework.utils.text import split_params  # noqa: PLC0415

  for argument in arguments:
    if argument.startswith('--params='):
      return split_params(argument.split('=', 1)[1])
    if argument == '--params' or argument.startswith('--params='):
      return {}
  return {}


def _rebuild_arguments(
  arguments: list[str],
  root: Path,
  conf_env: str,
  runtime_params: dict[str, Any],
) -> list[str]:
  """Re-emit the orchestrator arguments with an explicit environment.

  Args:
    arguments: The raw argument vector.
    root: The workspace root.
    conf_env: The configuration environment.
    runtime_params: The parsed runtime parameters.

  Returns:
    The rewritten argument vector.
  """
  rebuilt: list[str] = []
  seen_env = False
  for argument in arguments:
    if argument == '--params' or argument.startswith('--params='):
      continue
    if argument == '--env' or argument.startswith('--env='):
      seen_env = True
      continue
    rebuilt.append(argument)
  if not seen_env:
    rebuilt.extend(['--env', conf_env])
  del root, runtime_params
  return rebuilt


def _route_keys(text: str, plan: RewritePlan) -> str:
  """Rewrite bare channel keys to their precedence namespace.

  Substitution is applied longest-key-first so that a key which is a prefix of
  another (``suffix`` and ``suffix_prefix``) cannot partially match.

  Args:
    text: The document text.
    plan: The rewrite plan.

  Returns:
    The rewritten text.
  """
  keys = sorted(plan.globals_keys | plan.runtime_keys, key=len, reverse=True)
  for key in keys:
    namespace = plan.route(key)
    if namespace is None:
      continue
    text = text.replace('${' + key, '${' + namespace + ':' + key)
  return text


def _read(path: Path) -> str:
  """Read a configuration document.

  Args:
    path: The document path.

  Returns:
    The document text.
  """
  try:
    return path.read_text(encoding='utf-8')
  except OSError as error:
    raise ConfigurationRewriteError(f'Unable to read {path}', path=str(path), error=str(error)) from error


def _write(path: Path, text: str) -> None:
  """Write a configuration document.

  Args:
    path: The document path.
    text: The document text.

  Raises:
    ConfigurationRewriteError: If the write fails.
  """
  try:
    path.write_text(text, encoding='utf-8')
  except OSError as error:
    raise ConfigurationRewriteError(f'Unable to write {path}', path=str(path), error=str(error)) from error
