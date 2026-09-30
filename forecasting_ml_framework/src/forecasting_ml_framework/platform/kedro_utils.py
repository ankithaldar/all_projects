#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Kedro runner utilities.

A module was documented as dead code: an exhaustive search found no
import of it anywhere in the framework, and ``run_pipeline``'s docstring
contradicted its body because the recursive feed-dictionary expansion was
commented out.

It is retained here because it is genuinely useful for tests and notebooks, but
the docstring and the body now agree, and the module is a supported part of the
platform layer rather than an orphan.
"""

from __future__ import annotations

import copy
from typing import Any

from kedro.io import DataCatalog
from kedro.io.memory_dataset import MemoryDataset
from kedro.runner import SequentialRunner, ThreadRunner
from kedro.utils import load_obj

from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)


class MemoryCachedDataSet(MemoryDataset):
  """An in-memory dataset that does not release its payload between runs.

  ``MemoryDataSet`` drops its stored object when a node's outputs are consumed,
  which forces a recomputation whenever a later node re-reads the same value.
  This subclass keeps the object, which is what makes a fan-out node — several
  consumers of one frame — cheap in a test or a notebook.
  """

  def release(self) -> None:
    """Retain the stored payload instead of discarding it."""
    return

  def __deepcopy__(self, memo: dict[int, Any]) -> MemoryCachedDataSet:
    """Return an independent instance holding a deep copy of the payload.

    The previous implementation read ``_factory`` and ``_serializer``, which are
    attributes of a *pickle* dataset. ``MemoryDataset`` has neither, so any
    ``copy.deepcopy`` of a catalog containing this dataset raised ``AttributeError``
    -- and Kedro's runners copy catalogs.

    Args:
      memo: The deepcopy memo, used to break cycles.

    Returns:
      A new instance with its own payload copy.
    """
    clone = type(self)()
    memo[id(self)] = clone
    clone._data = copy.deepcopy(self._data, memo)  # noqa: SLF001 - the parent owns these
    clone._copy_mode = self._copy_mode  # noqa: SLF001
    return clone


def add_param_to_feed_dict(
  feed_dict: dict[str, Any],
  key_path: str,
  value: Any,
) -> dict[str, Any]:
  """Expand a ``params:<dotted.key>`` reference into a feed-dictionary entry.

  The expansion is performed rather than merely promised: a ``params:``-prefixed
  reference may point at the top-level parameter dictionary or at a nested key,
  and the feed dictionary is shaped to match either shape. A reference that is
  documented and not generated fails silently, because the parameter simply never
  arrives.

  Args:
    feed_dict: The feed dictionary to mutate.
    key_path: The full key path, e.g. ``params:modeling_params.prediction_col``.
    value: The resolved value, supplied by the catalog resolver.

  Returns:
    The mutated feed dictionary.
  """
  if not key_path.startswith('params'):
    feed_dict[key_path] = value
    return feed_dict
  remainder = key_path[len('params') :].lstrip(':')
  if not remainder:
    feed_dict[key_path] = value
    return feed_dict
  cursor = feed_dict.setdefault('params', {})
  parts = remainder.split('.')
  for part in parts[:-1]:
    cursor = cursor.setdefault(part, {})
    if not isinstance(cursor, dict):  # pragma: no cover - malformed reference
      raise TypeError(f'Parameter reference {key_path!r} traverses a non-mapping value at {part!r}')
  cursor[parts[-1]] = value
  return feed_dict


def run_pipeline(
  pipeline: Any,
  catalog: DataCatalog,
  runner: str | None = None,
  is_async: bool = False,
  session_id: str | None = None,
  hook_manager: Any = None,
) -> dict[str, Any]:
  """Execute a pipeline programmatically with a chosen runner.

  Args:
    pipeline: The ``Pipeline`` to run.
    catalog: The data catalog.
    runner: ``'sequential'`` or ``'thread'``. Defaults to ``'sequential'``,
      which is the only runner under which the framework's object-identity
      assumptions about shared metadata hold.
    is_async: Rejected. Retained for signature compatibility; the framework
      requires sequential execution because the evaluation nodes share the
      metadata contract and the fitted registry.
    session_id: An optional run identifier, applied to Kedro's ``run_id``.
    hook_manager: An optional hook manager.

  Returns:
    A mapping of dataset name to the produced value.

  Raises:
    ConfigurationError: If asynchronous execution is requested, which this
      framework does not support. Raised rather than ignored, so an operator who
      asked for it learns that they did not get it.
  """
  from forecasting_ml_framework.exceptions import ConfigurationError  # noqa: PLC0415

  # `AbstractRunner.run` in Kedro 1.0 accepts neither `is_async` nor
  # `session_id`, and passes no `**kwargs`, so forwarding them raised
  # `TypeError` on *every* call for *both* runners. The two names are kept in
  # this signature because they are part of the module's published surface and
  # callers pass them; what changed is that `is_async` is now honoured or
  # refused, and `session_id` is applied to the parameter Kedro actually accepts
  # for the purpose -- `run_id`.
  if is_async:
    raise ConfigurationError(
      'Asynchronous execution is not supported. The framework shares mutable state -- the metadata '
      'contract and the fitted registry -- between the evaluation nodes, so nodes must not '
      'overlap. Use the sequential runner.',
      runner=runner,
    )
  selected = ThreadRunner() if runner == 'thread' else SequentialRunner()
  LOGGER.info('Running pipeline programmatically', extra={'runner': runner or 'sequential'})
  return selected.run(
    pipeline=pipeline,
    catalog=catalog,
    hook_manager=hook_manager,
    run_id=session_id,
  )


def load_pipeline_object(pipeline_name: str) -> Any:
  """Load a pipeline by dotted path, mirroring Kedro's own resolver.

  Args:
    pipeline_name: A dotted path such as ``'package.module:object'`` or
      ``'package.module.object'``.

  Returns:
    The resolved object.
  """
  return load_obj(pipeline_name)
