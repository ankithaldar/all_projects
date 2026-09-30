#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared bases for the routine family.

a library declared ``__str__`` and ``copy`` identically in
ten of twenty-seven classes and omitted them from the rest, so roughly nine
hundred lines were removable boilerplate. All three shared members now live on
:class:`~forecasting_ml_framework.core.routine.PreprocessRoutine`, and this
module adds two further bases that remove the repetition that remains:

:class:`SparkMlRoutine`
  For transforms whose fitted state is a Spark ML ``PipelineModel``. The state
  is written with Spark's native ``write().overwrite().save(path)`` — a
  ``PipelineModel`` is not reliably picklable and Spark's format preserves the
  transformer graph — and reloaded in ``apply``.

:class:`UdfRoutine`
  For transforms that delegate their arithmetic to a pure helper through a
  ``pandas_udf`` or a scalar UDF. It owns the column-projection boilerplate and
  the metadata protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from forecasting_ml_framework.core.metadata import Metadata
from forecasting_ml_framework.core.routine import PreprocessRoutine
from forecasting_ml_framework.exceptions import DataContractError
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)


class SparkMlRoutine(PreprocessRoutine):
  """A transform whose fitted state is persisted as a Spark ML pipeline model.

  Attributes:
    save_model_path: The object-storage path of the persisted pipeline model.
      It travels with the pickled registry, so ``apply`` can reload it verbatim.
  """

  def __init__(self) -> None:
    """Install the declared defaults and an empty artefact path."""
    super().__init__()
    self.save_model_path: str = ''

  def artefact_path(self, metadata: Metadata) -> str:
    """Return the object-storage path for this routine's fitted pipeline.

    Args:
      metadata: The schema contract, whose ``intermediate_file_path`` roots the
        artefact tree.

    Returns:
      The absolute artefact path.
    """
    if not metadata.intermediate_file_path:
      raise ValueError(
        f'{type(self).__name__} persists a Spark pipeline model and therefore requires '
        'metadata.intermediate_file_path. Set global_params.intermediate_file_path in the '
        'configuration.'
      )
    return f'{metadata.intermediate_file_path.rstrip("/")}/{self.artefact_name()}'

  def artefact_name(self) -> str:
    """Return the artefact directory name for this routine.

    The default is the class name, which is sufficient because the artefact tree
    is already namespaced by ``model_key``. A routine registered twice with
    different parameters should override this so the two artefacts do not
    collide.

    Returns:
      The directory name.
    """
    return type(self).__name__

  def _fit_pipeline(self, df: Any, stages: list[Any]) -> Any:
    """Fit a Spark ML pipeline and persist it.

    Args:
      df: The input frame.
      stages: The pipeline stages, in order.

    Returns:
      The fitted ``PipelineModel``.
    """
    from pyspark.ml import Pipeline  # noqa: PLC0415

    fitted = Pipeline(stages=stages).fit(df)
    fitted.write().overwrite().save(self.save_model_path)
    LOGGER.info(
      'Persisted fitted pipeline',
      extra={'routine': self.name, 'path': self.save_model_path, 'stages': len(stages)},
    )
    return fitted

  def _load_pipeline(self) -> Any:
    """Reload a previously persisted pipeline model.

    Returns:
      The loaded ``PipelineModel``.

    Raises:
      FileNotFoundError: If the artefact is absent, which means the registry was
        applied without having been fitted.
    """
    from pyspark.ml import PipelineModel  # noqa: PLC0415

    LOGGER.info('Reloading fitted pipeline', extra={'routine': self.name, 'path': self.save_model_path})
    return PipelineModel.load(self.save_model_path)


class UdfRoutine(PreprocessRoutine):
  """A transform that delegates its arithmetic to a pure helper through a UDF.

  Subclasses declare the helper, the declared Spark return type, the helper's
  bound keyword arguments, and whether the output is a new suffixed column, a
  new unsuffixed column, or an in-place rewrite of the source column.
  """

  #: The pure helper invoked per row.
  helper: Callable[..., Any] = None  # type: ignore[assignment]
  #: The declared Spark return type of the helper.
  return_type: Any = None
  #: The suffix applied to a derived column name, or ``None`` for in-place.
  output_suffix: str | None = None
  #: Whether a derived column is registered in ``numerical_cols`` as well.
  output_is_numerical: bool = True

  def helper_kwargs(self) -> dict[str, Any]:
    """Return the keyword arguments bound into the helper.

    Returns:
      A mapping of helper keyword argument to value.
    """
    return {}

  def derived_name(self, column: str) -> str:
    """Return the output column name for a source column.

    Args:
      column: The source column name.

    Returns:
      The derived column name, or the source name for an in-place transform.
    """
    if self.output_suffix is None:
      return column
    return f'{column}{self.output_suffix}'

  def _build_udf(self) -> Any:
    """Construct the Spark UDF from the helper and its bound arguments.

    Returns:
      A ``pyspark.sql.functions.udf`` wrapping the helper.
    """
    from functools import partial  # noqa: PLC0415

    from pyspark.sql import functions as sf  # noqa: PLC0415

    return sf.udf(partial(self.helper, **self.helper_kwargs()), returnType=self.return_type)

  def apply(self, df: Any, metadata: Metadata) -> tuple[Any, Metadata]:
    """Apply the helper to each selected column.

    Args:
      df: The input frame.
      metadata: The schema contract. A defensive copy is taken, because a
        summarising routine *writes* new features and the mutation must not
        escape into the caller's frame unintentionally.

    Returns:
      A two-tuple of the frame and the contract.
    """
    from pyspark.sql import functions as sf  # noqa: PLC0415

    metadata = metadata.copy()
    udf = self._build_udf()
    for column in self.use_cols:
      derived = self.derived_name(column)
      df = df.withColumn(derived, udf(sf.col(column), *self.extra_udf_columns(column)))
      if derived != column:
        metadata.register_derived_feature(derived, numerical=self.output_is_numerical)
    return df, metadata

  def extra_udf_columns(self, column: str) -> list[Any]:
    """Return further column expressions passed to the helper after the array.

    The default is empty, which is the overwhelmingly common case: a sequence
    summariser is a function of the array alone. It exists because a minority are
    not -- recency is the age of the newest event *at the run date*, so it needs
    that date as a second argument. Without the hook, such a routine binds the
    sentinel unconditionally and emits a constant column: configured, registered
    as a feature, and modelling-inert, which is the hardest kind of defect to see.

    Args:
      column: The array column being summarised.

    Returns:
      Additional column expressions, or an empty list.
    """
    del column
    return []

  def fit(self, df: Any, metadata: Metadata) -> tuple[UdfRoutine, Any, Metadata]:
    """Resolve the target columns. No state is learned.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A three-tuple of the routine, the frame and the contract.
    """
    self.use_cols = self.get_selected_columns(
      self.cols.get('exclude_cols', []),
      self.cols.get('include_cols', []),
      metadata.feature_cols,
    )
    LOGGER.debug('Resolved %s columns', self.name, extra={'columns': self.use_cols})
    return self, df, metadata


def empty_column_selector(params: Mapping[str, Any] | None) -> tuple[list[str], list[str]]:
  """Extract the include and exclude lists from a routine's ``cols`` block.

  Args:
    params: The routine's own configuration block.

  Returns:
    A two-tuple of the exclude list and the include list.
  """
  cols = (params or {}).get('cols') or {}
  return list(cols.get('exclude_cols') or []), list(cols.get('include_cols') or [])


def require_columns(frame: Any, columns: Sequence[str], routine: str) -> None:
  """Assert that a frame carries the columns a transform is about to read.

  Args:
    frame: The input frame.
    columns: The columns the transform will read.
    routine: The routine name, used in the error message.

  Raises:
    DataContractError: If a column is absent. The routine is named in the
      message, so the failure is attributable to a configuration entry rather
      than to an anonymous column lookup deep inside a transform.
  """
  available = set(frame.columns)
  missing = [column for column in columns if column not in available]
  if missing:
    raise DataContractError(
      f'{routine} was configured to act on {len(missing)} column(s) that the frame does not '
      f'contain: {sorted(missing)[:20]}. Check the routine\'s cols.include_cols / '
      'cols.exclude_cols against the declared column roles.',
      routine=routine,
      missing=sorted(missing),
    )
