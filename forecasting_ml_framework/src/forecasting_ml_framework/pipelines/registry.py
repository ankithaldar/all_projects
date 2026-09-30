#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The pipeline registry and the four pipeline factories.

The pipeline graph is not statically declared. It is a *function of five
environment variables* read at registration time, and those variables are written
into the process environment by a lifecycle hook that reads the run's runtime
parameters.

What that buys is the framework's reason for existing: one codebase serves an
unbounded number of model instances, and adding a model requires no pipeline code
at all — only a different ``suffix`` and a different column list. The
classification and regression topologies are unified behind one registry entry
point, so the external scheduler does not need to know which kind of model it is
launching.

What it costs, and what is done about it here:

* **Static analysis is defeated.** Mitigated by :func:`describe_pipelines`, which
  renders the built graph as text. A had no such affordance, so the
  graph could not be inspected without executing the factories with the correct
  environment.
* **Environment variables are a global mutable side channel.** Two concurrent
  pipeline runs in one process would clobber each other. The framework is safe
  only because the scheduler runs one pipeline per process — an invariant
  enforced externally. :func:`describe_pipelines` therefore prints the resolved
  topology at registration, so a mismatch is visible in the log immediately
  rather than at the first dataset access.
* **Module-level import side effects.** A pipeline modules printed
  their own absolute source path and a computed project root on import, which
  executed during registration, before any node ran. There are none here.
"""

from __future__ import annotations

import os
from typing import Any

from kedro.pipeline import Pipeline

from forecasting_ml_framework import constants
from forecasting_ml_framework.observability.logging import get_logger
from forecasting_ml_framework.platform.environment import as_flag

LOGGER = get_logger(__name__)


def register_pipelines() -> dict[str, Pipeline]:
  """Build the four pipelines and their composition.

  Returns:
    A mapping of pipeline name to pipeline, with five entries: the four
    independently runnable stages plus ``__default__``, their sum.

  Raises:
    PipelineTopologyError: If a required environment variable holds an
      unsupported value. The check belongs to :meth:`PipelineTopology.from_environment`,
      and this function propagates it rather than substituting a default -- an
      unrecognised run mode must not silently select a topology.
  """
  from forecasting_ml_framework.pipelines.classification import (  # noqa: PLC0415
    create_fe_scoring_pipeline,
    create_fe_training_pipeline,
    create_model_scoring_pipeline,
    create_model_training_pipeline,
  )

  topology = PipelineTopology.from_environment()
  if topology.is_regression:
    from forecasting_ml_framework.pipelines.regression import (  # noqa: PLC0415
      create_fe_scoring_pipeline as create_fe_scoring_regression,
    )
    from forecasting_ml_framework.pipelines.regression import (  # noqa: PLC0415
      create_fe_training_pipeline as create_fe_training_regression,
    )
    from forecasting_ml_framework.pipelines.regression import (  # noqa: PLC0415
      create_model_scoring_pipeline as create_model_scoring_regression,
    )
    from forecasting_ml_framework.pipelines.regression import (  # noqa: PLC0415
      create_model_training_pipeline as create_model_training_regression,
    )
    factories = (
      create_fe_training_regression,
      create_model_training_regression,
      create_fe_scoring_regression,
      create_model_scoring_regression,
    )
  else:
    factories = (
      create_fe_training_pipeline,
      create_model_training_pipeline,
      create_fe_scoring_pipeline,
      create_model_scoring_pipeline,
    )

  fe_training, model_training, fe_scoring, model_scoring = (
    factory(topology) for factory in factories
  )

  pipelines = {
    constants.PIPELINE_FE_TRAINING: fe_training,
    constants.PIPELINE_MODEL_TRAINING: model_training,
    constants.PIPELINE_FE_SCORING: fe_scoring,
    constants.PIPELINE_MODEL_SCORING: model_scoring,
    constants.PIPELINE_DEFAULT: fe_training + model_training + fe_scoring + model_scoring,
  }
  for name, pipeline in pipelines.items():
    LOGGER.info(
      'Registered pipeline',
      extra={'pipeline': name, 'nodes': len(pipeline.nodes)},
    )
  return pipelines


class PipelineTopology:
  """The five specialisation axes that determine the graph's shape.

  Attributes:
    suffix: The per-model-instance namespace token. Every dataset name in the
      graph is suffixed with it, which is what allows many model instances to
      share one codebase, one container image and one output prefix without
      collision.
    split: Whether the Spark-level split node is inserted.
    is_regression: Whether the regression node variants are used.
    run_mode: ``'prod'`` withholds the target column from the scoreset; ``'eval'``
      adds the four test-evaluation nodes.
    train_mode: Which promotion mode is active.
  """

  __slots__ = ('suffix', 'split', 'is_regression', 'run_mode', 'train_mode')

  def __init__(
    self,
    suffix: str = constants.DEFAULT_SUFFIX,
    split: bool = False,
    is_regression: bool = False,
    run_mode: str = 'prod',
    train_mode: str = 'train_only',
  ) -> None:
    """Build the topology.

    Args:
      suffix: The namespace token.
      split: Whether to insert the split node.
      is_regression: Whether to use the regression variants.
      run_mode: ``'prod'`` or ``'eval'``.
      train_mode: The promotion mode.

    Raises:
      PipelineTopologyError: If ``run_mode`` or ``train_mode`` is unsupported.
    """
    from forecasting_ml_framework.exceptions import PipelineTopologyError  # noqa: PLC0415

    valid_modes = {mode.value for mode in constants.RunMode}
    if run_mode not in valid_modes:
      raise PipelineTopologyError(
        f'Unsupported run_mode: {run_mode!r}. Supported: {sorted(valid_modes)}', run_mode=run_mode
      )
    # `train_mode` is validated for the same reason and with the same weight as
    # `run_mode`. Only `run_mode` was checked, so a typo in `train_mode` was
    # accepted and then *tested* against the compare-mode vocabulary: an
    # unrecognised token is not "compare", so `is_compare_mode` was False and the
    # graph selected the non-comparison promotion node. The run then retrained,
    # wrote the production model trio, and applied no gate at all -- which is the
    # exact outcome the gate exists to prevent, reached by a spelling mistake.
    valid_train_modes = {mode.value for mode in constants.TrainMode}
    if train_mode not in valid_train_modes:
      raise PipelineTopologyError(
        f'Unsupported train_mode: {train_mode!r}. Supported: {sorted(valid_train_modes)}. An '
        f'unrecognised value is not treated as a non-comparison mode, because that would '
        f'silently select the ungated promotion node.',
        train_mode=train_mode,
      )
    self.suffix = suffix
    self.split = split
    self.is_regression = is_regression
    self.run_mode = run_mode
    self.train_mode = train_mode

  @classmethod
  def from_environment(cls) -> PipelineTopology:
    """Build the topology from the five environment variables.

    Returns:
      The topology.
    """
    return cls(
      suffix=os.getenv(constants.ENV_SUFFIX, constants.DEFAULT_SUFFIX),
      split=as_bool(os.getenv(constants.ENV_SPLIT, 'False')),
      is_regression=as_bool(os.getenv(constants.ENV_IS_REGRESSION, 'False')),
      run_mode=os.getenv(constants.ENV_RUN_MODE, 'prod'),
      train_mode=os.getenv(constants.ENV_TRAIN_MODE, 'train_only'),
    )

  def export_to_environment(self) -> None:
    """Write the topology into the process environment.

    The lifecycle hook calls this once the Kedro context exists, so the
    registration-time read finds a populated environment.
    """
    for key, value in (
      (constants.ENV_SUFFIX, self.suffix),
      (constants.ENV_SPLIT, str(self.split)),
      (constants.ENV_IS_REGRESSION, str(self.is_regression)),
      (constants.ENV_RUN_MODE, self.run_mode),
      (constants.ENV_TRAIN_MODE, self.train_mode),
    ):
      os.environ[key] = value

  @property
  def is_evaluation(self) -> bool:
    """Report whether the run is in evaluation mode.

    Returns:
      ``True`` in ``eval`` mode.
    """
    return self.run_mode == constants.RunMode.EVAL.value

  @property
  def is_compare_mode(self) -> bool:
    """Report whether the run performs a champion/challenger comparison.

    Returns:
      ``True`` when the training mode selects the promotion gate.
    """
    return 'compare' in self.train_mode.lower()

  def __str__(self) -> str:
    """Render the topology for the run log.

    Returns:
      A comma-joined key/value string.
    """
    return (
      f'suffix={self.suffix}, split={self.split}, is_regression={self.is_regression}, '
      f'run_mode={self.run_mode}, train_mode={self.train_mode}'
    )


def describe_pipelines(pipelines: dict[str, Pipeline] | None = None) -> str:
  """Render the registered graph as text.

  The graph has no static declaration, so a reviewer cannot see the topology
  without running the factories. This renders it.

  Args:
    pipelines: The registered mapping. Resolved from the registry when omitted.

  Returns:
    A multi-line description of every pipeline's nodes and datasets.
  """
  resolved = pipelines if pipelines is not None else register_pipelines()
  lines: list[str] = [f'Pipeline topology: {PipelineTopology.from_environment()}']
  for name, pipeline in resolved.items():
    lines.append(f'\n{name}  ({len(pipeline.nodes)} nodes)')
    for node in pipeline.nodes:
      inputs = ', '.join(node.inputs)
      outputs = ', '.join(node.outputs)
      lines.append(f'  - {node.name}')
      lines.append(f'      in : {inputs}')
      lines.append(f'      out: {outputs}')
  return '\n'.join(lines)


def as_bool(value: Any) -> bool:
  """Interpret a mode flag that arrives as a string or a boolean.

  A YAML document can legitimately spell a flag as a quoted string, and a
  non-empty string is truthy, so an unquoted read inverts the meaning of a flag the
  operator set to false. Every boolean mode axis is routed through this function
  rather than read directly.

  The environment flags have their own resolver, which additionally accepts the
  legacy hosted spelling and applies the back-end rule; see
  :mod:`forecasting_ml_framework.platform.environment`.

  Args:
    value: The raw value.

  Returns:
    The boolean interpretation.

  Raises:
    ConfigurationError: If the value is not recognisable as a boolean. Failing
      loudly is the point: a misread flag changes the graph shape, and a silent
      default would make the change look intentional.
  """
  return as_flag(value, default=False)
