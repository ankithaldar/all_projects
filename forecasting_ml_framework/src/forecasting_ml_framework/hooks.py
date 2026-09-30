#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Project lifecycle hooks.

The hook manager is the mechanism that makes the pipeline graph
environment-driven: it reads the run's runtime parameters once the Kedro context
exists and writes the five specialisation variables into the process environment
before the pipeline registry reads them.

The per-node provenance hooks are the most successful observability feature in
the framework, and they are preserved here. For every node they report the
resolved object-storage path of every pickle-backed input and output. Given the
framework's reliance on path-suffixed artefacts, a run log that says exactly
which file each model handle was loaded from and saved to is the primary tool for
diagnosing artefact-resolution problems — and the design had a real
artefact-resolution issue that this instrumentation would have made visible
immediately.

a iterated ``catalog._datasets``, a private attribute of the
catalog class, which is a version-coupling risk that could break the hooks
silently on a Kedro upgrade. A public accessor is preferred, and the private
access is retained only as a fallback for older catalog implementations.
"""

from __future__ import annotations

import os
from typing import Any

from kedro.framework.hooks import hook_impl

from forecasting_ml_framework import constants
from forecasting_ml_framework.observability.logging import get_logger
from forecasting_ml_framework.observability.tracking import end_active_runs, register_model, tracking_is_configured
from forecasting_ml_framework.pipelines.registry import PipelineTopology
from forecasting_ml_framework.platform.environment import as_flag, resolve_environment

LOGGER = get_logger(__name__)


class ModelTrackingHooks:
  """The project's single hook manager.

  Kedro executes hooks last-in-first-out within a manager, so the post-pipeline
  hook below runs after the per-node hooks.
  """

  @hook_impl
  def after_context_created(self, context: Any) -> None:
    """Publish the run's specialisation axes into the process environment.

    This must fire before the pipeline registry resolves the graph. Kedro's lazy
    pipeline resolution makes that ordering hold, but it is implicit in Kedro
    rather than stated by it, so the requirement is documented here and in the
    registry module's docstring -- a hook that runs one step late publishes axes
    no resolver has read yet, and the graph silently falls back to defaults.

    Args:
      context: The Kedro context.
    """
    params = dict(getattr(context, 'params', {}) or {})
    # The boolean axes are coerced rather than passed through. A YAML document can
    # legitimately carry `is_regression: "False"` as a quoted string, and a
    # non-empty string is truthy -- so the unquoted read selected the *regression*
    # topology for a classification run, silently. Coercion is what makes the
    # declared value the effective one.
    topology = PipelineTopology(
      suffix=str(params.get(constants.ENV_SUFFIX, constants.DEFAULT_SUFFIX)),
      split=as_flag(params.get(constants.ENV_SPLIT), default=False, key=constants.ENV_SPLIT),
      is_regression=as_flag(
        params.get(constants.ENV_IS_REGRESSION), default=False, key=constants.ENV_IS_REGRESSION
      ),
      run_mode=str(params.get(constants.ENV_RUN_MODE, 'prod')),
      train_mode=str(params.get(constants.ENV_TRAIN_MODE, 'train_only')),
    )
    topology.export_to_environment()
    LOGGER.info('Pipeline topology resolved', extra={'topology': str(topology)})

  # NOTE: there are deliberately no ``before_dataset_loaded`` /
  # ``after_dataset_saved`` implementations here. Both were present and both
  # declared a ``dataset`` parameter, which is not in Kedro's hook
  # specification. pluggy validates a hookimpl's argument names against the
  # specification at *registration* time, so the mismatch raised
  # ``PluginValidationError`` while the hook manager was being built -- before any
  # node ran, for every run, classification and regression alike. It is not a
  # degraded feature; it is a hard failure with a message that points at pluggy
  # rather than at the cause.
  #
  # They were also redundant. The specification supplies the *node* and the
  # *data*, never the dataset object, so neither hook could report the resolved
  # path it was written to report. :meth:`before_node_run` and
  # :meth:`after_node_run` receive the catalog itself and already log the same
  # provenance for every input and output, which is the behaviour the module
  # docstring describes.

  @hook_impl
  def before_node_run(self, node: Any, catalog: Any, inputs: dict[str, Any], is_async: bool, run_id: str) -> None:
    """Log the node and the provenance of its pickle-backed inputs.

    Args:
      node: The node about to run.
      catalog: The data catalog.
      inputs: The resolved inputs.
      is_async: Whether the node runs asynchronously.
      run_id: The run identifier.
    """
    LOGGER.info('Node starting', extra={'node': node.name, 'async': is_async, 'run_id': run_id})
    for name, dataset in _iter_datasets(catalog):
      if name in inputs:
        _log_provenance(name, dataset, 'input for', node.name)

  @hook_impl
  def after_node_run(self, node: Any, catalog: Any, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
    """Log the node and the provenance of its pickle-backed outputs.

    Args:
      node: The node that ran.
      catalog: The data catalog.
      inputs: The resolved inputs.
      outputs: The resolved outputs.
    """
    del inputs
    LOGGER.info('Node complete', extra={'node': node.name, 'outputs': sorted(outputs)})
    for name, dataset in _iter_datasets(catalog):
      if name in outputs:
        _log_provenance(name, dataset, 'output for', node.name)

  @hook_impl
  def after_pipeline_run(self, run_params: dict[str, Any], run_result: Any, pipeline: Any, catalog: Any) -> None:
    """Register the promoted model with the tracking server, then close runs.

    The promotion-aware branch is the point of this hook: in comparison mode the
    winning model is the validation node's output, so that is what is registered.
    Loading the training node's output instead would register a model that was
    rejected.

    This hook also guarantees no tracking run leaks. MLflow permits nested runs,
    so the close loop is defensive.

    Args:
      run_params: The run's parameters.
      run_result: The run's result.
      pipeline: The pipeline that ran.
      catalog: The data catalog.
    """
    del run_result
    LOGGER.info(
      'Pipeline complete',
      extra={'pipeline': run_params.get('pipeline_name'), 'env': run_params.get('env')},
    )
    if not tracking_is_configured():
      return
    if run_params.get('pipeline_name') != constants.PIPELINE_MODEL_TRAINING:
      return

    runtime = run_params.get('runtime_params', {}) or {}
    # The axes and the model key are read from the parameters document, with the
    # `--params` runtime bag as a fallback for a key the scheduler genuinely
    # supplies per run. `train_mode` is a *graph* axis and lives only in the
    # document, so reading it from the runtime bag defaulted it to `train_only`
    # and made a comparison run register its rejected challenger.
    document = _load_parameters_document(pipeline, catalog) or {}
    model, params, metrics = _collect_registration_payload(pipeline, catalog, document)
    register_model(
      model_key=str(document.get(constants.MODEL_KEY, runtime.get(constants.MODEL_KEY, ''))),
      run_name=str(runtime.get('run_name', '')),
      params=params,
      model=model,
      metrics=metrics,
    )
    end_active_runs()

  @hook_impl
  def on_pipeline_error(self, error: Exception, run_params: dict[str, Any], pipeline: Any, catalog: Any) -> None:
    """Close any active tracking run and record the failure.

    Args:
      error: The raised exception.
      run_params: The run's parameters.
      pipeline: The pipeline that failed.
      catalog: The data catalog.
    """
    del run_params, pipeline, catalog
    LOGGER.error('Pipeline failed', extra={'error': str(error), 'type': type(error).__name__}, exc_info=True)
    end_active_runs('FAILED')


def _collect_registration_payload(
  pipeline: Any, catalog: Any, document: dict[str, Any] | None
) -> tuple[Any, dict[str, Any] | None, dict[str, float] | None]:
  """Gather the estimator, its parameters and its metric row for registration.

  The metric frame carries a date and a timestamp, which are not valid metric
  values, so the frame is restricted to numeric columns before it is read. The
  promotion-aware branch selects the validation node's output in comparison mode.

  The specialisation axes are read from the **parameters document**, which is
  where they are written. They were previously read from ``run_params``, which is
  the ``--params`` channel and never carries them, so ``train_mode`` always
  resolved to its default and ``looking_for_validation`` was always false. In a
  comparison run that meant the *training* node's output was registered -- the
  rejected challenger -- which is the single outcome this hook exists to prevent,
  reached by reading the wrong dictionary. ``model_key`` was read the same way and
  was likewise always empty.

  Args:
    pipeline: The pipeline that ran.
    catalog: The data catalog.
    document: The resolved parameters document, or ``None`` when unavailable.

  Returns:
    A three-tuple of the estimator, the parameter block and the metric row.
  """
  import pandas as pd  # noqa: PLC0415

  document = document or {}
  train_mode = str(document.get(constants.ENV_TRAIN_MODE, 'train_only'))
  is_regression = _read_axis(document, 'is_regression', 'False').lower() in ('true', '1', 'yes', 'on')
  looking_for_validation = 'compare' in train_mode.lower()

  model: Any = None
  params: dict[str, Any] | None = None
  metrics: dict[str, float] | None = None

  for pipeline_node in pipeline.nodes:
    name = pipeline_node.name
    # The parentheses are load-bearing: `a and b or c` binds as `(a and b) or c`,
    # so the unguarded form selected the training node whenever a validation node
    # happened to appear anywhere in the graph.
    selects_model = (
      (name.startswith('model_validation') and looking_for_validation)
      or (name.startswith('model_training') and not looking_for_validation)
    )
    if selects_model and pipeline_node.outputs:
      model = catalog.load(pipeline_node.outputs[0])
    if name.startswith('model_training') and params is None and pipeline_node.inputs:
      # The parameters document is the node's FIRST input; its second and third
      # are the training frame and the metadata contract. Reading a later input
      # and calling `.get` on it raised, because the contract has no `.get` -- and
      # it did so in `after_pipeline_run`, which is after the pipeline has already
      # succeeded. The run failed at the last possible moment, on a fully trained
      # model, for a reason that pointed nowhere near the real fault.
      node_document = catalog.load(pipeline_node.inputs[0])
      if isinstance(node_document, dict):
        block = _modelling_block(node_document, is_regression)
        params = block.get('model_params') if isinstance(block, dict) else None
    if name.startswith('train_model_evaluation') and pipeline_node.outputs:
      frame = catalog.load(pipeline_node.outputs[0])
      if isinstance(frame, pd.DataFrame) and not frame.empty:
        metrics = frame.select_dtypes(include=['number']).iloc[0].to_dict()

  return model, params, metrics


def _load_parameters_document(pipeline: Any, catalog: Any) -> dict[str, Any] | None:
  """Return the run's parameters document, as the graph sees it.

  Any node that declares ``parameters`` as an input carries the resolved
  document, so the training node is used as the locator. Reading it from the
  graph rather than from the hook's own arguments matters because the graph is
  the only place that knows which document the run actually resolved.

  Args:
    pipeline: The pipeline that ran.
    catalog: The data catalog.

  Returns:
    The parameters document, or ``None`` when it cannot be read.
  """
  for pipeline_node in getattr(pipeline, 'nodes', []):
    if not pipeline_node.name.startswith('model_training') or not pipeline_node.inputs:
      continue
    try:
      document = catalog.load(pipeline_node.inputs[0])
    except Exception as error:  # noqa: BLE001 - registration must never fail a run
      LOGGER.warning('Could not load the parameters document for tracking', extra={'error': str(error)})
      return None
    return document if isinstance(document, dict) else None
  return None


def _read_axis(document: dict[str, Any], key: str, default: str) -> str:
  """Read one specialisation axis from the parameters document.

  Args:
    document: The parameters document.
    key: The axis name.
    default: The value used when the axis is absent.

  Returns:
    The axis value as a string.
  """
  value = document.get(key, default)
  return str(value)


def _modelling_block(document: dict[str, Any], is_regression: bool) -> dict[str, Any]:
  """Return the modelling block matching the run's task family.

  Both blocks are present in the shipped document -- ``modeling_params`` and
  ``modeling_reg_params`` -- and the previous code took whichever carried a
  ``model_params`` key first, which is always the classification block. A
  regression model was therefore registered against the classifier's
  hyperparameters, so the registry entry did not describe the artefact it was
  attached to. Resolution is delegated to the node layer's own resolver, so the
  hook and the trainer cannot disagree about which block a run is using.

  Args:
    document: The parameters document.
    is_regression: Whether the run uses the regression task family.

  Returns:
    The matching block, or an empty mapping when none is present.
  """
  from forecasting_ml_framework.nodes.validation_nodes import _modelling_block as resolve_block  # noqa: PLC0415

  try:
    return resolve_block(document, regression=is_regression)
  except Exception as error:  # noqa: BLE001 - registration must never fail a run
    LOGGER.warning(
      'Could not resolve the modelling block for registration; hyperparameters will be omitted',
      extra={'is_regression': is_regression, 'error': str(error)},
    )
    return {}


def _iter_datasets(catalog: Any) -> Any:
  """Iterate every dataset the catalog declares, through its public interface.

  ``keys()`` and ``get()`` are used because they are the only accessors that are
  both public and complete. Kedro 1.0 holds datasets in *two* private mappings
  -- ``_datasets`` for initialised instances and ``_lazy_datasets`` for the
  deferred ones built from configuration -- and everything a node actually reads
  or writes arrives through the second. Reading ``_datasets`` alone therefore
  yielded nothing at all: the per-node provenance record, described in this
  module as the primary tool for diagnosing artefact-resolution problems, was
  silent on every node of every run. ``DataCatalog`` has no ``datasets``
  attribute either, so the first branch never fired.

  Args:
    catalog: The data catalog.

  Yields:
    ``(name, dataset)`` pairs for every declared dataset that materialises.
  """
  getter = getattr(catalog, 'get', None)
  for name in list(getattr(catalog, 'keys', list)()):
    try:
      dataset = getter(name) if callable(getter) else None
    except Exception as error:  # noqa: BLE001 - provenance must never fail a node
      # Materialising a dataset can fail (a missing file, an unreachable store).
      # That is precisely the situation the provenance record exists to explain,
      # so it is reported rather than propagated: a logging hook must never be
      # the reason a node fails.
      LOGGER.warning(
        'Could not materialise a dataset for the provenance record',
        extra={'dataset': name, 'error': str(error)},
      )
      continue
    if dataset is not None:
      yield name, dataset


def _log_provenance(name: str, dataset: Any, action: str, node_name: str = '') -> None:
  """Log the resolved path of a serialisation-backed dataset.

  Only pickle-backed datasets are reported: they are the model handles, the
  fitted registries and the metadata objects, whose provenance is what an
  operator needs when an artefact resolves unexpectedly. The bulk dataframes'
  paths are already predictable from the naming convention.

  Args:
    name: The catalog key.
    dataset: The dataset instance.
    action: The action being performed.
    node_name: The node the action belongs to, when known.
  """
  # A hook must never fail a run, so the description is best-effort. ``_describe``
  # is Kedro's dataset contract method and has no public accessor, which is why the
  # access is not a protected-member violation in spirit: there is no alternative.
  describe = getattr(dataset, '_describe', None)
  if not callable(describe):
    return
  module = type(dataset).__module__ or ''
  if 'pickle' not in module.lower() and 'parquet' not in module.lower():
    return
  try:
    described = describe()
  except Exception:  # noqa: BLE001 - the hook must never fail a run
    return
  if not isinstance(described, dict):
    return
  LOGGER.info(
    'Dataset provenance',
    extra={'dataset': name, 'action': action, 'path': described.get('filepath'), 'node': node_name},
  )


def write_run_name(runtime_params: dict[str, Any], env: str = 'local') -> str:
  """Compute a run identifier and export it for correlation.

  A computed a well-formed run name and exported it as an environment
  variable, and no framework code reads it. It is read by the logging filter,
  so every framework log line carries it.

  Args:
    runtime_params: The run's runtime parameters.
    env: The configuration environment.

  Returns:
    The run identifier.
  """
  from datetime import datetime  # noqa: PLC0415

  from forecasting_ml_framework.observability.logging import RUN_NAME_ENV_VAR  # noqa: PLC0415

  stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  run_day = runtime_params.get(constants.RUN_DATE, 'na')
  if resolve_environment(runtime_params).is_hosted:
    run_name = f'hosted_{run_day}_{stamp}'
  else:
    run_name = f'{env}_{run_day}_{stamp}'
  os.environ[RUN_NAME_ENV_VAR] = run_name
  return run_name
