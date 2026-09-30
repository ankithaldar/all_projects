#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The ordered, composable, serialisable transform chain.

The registry is the framework's central train/serve-parity mechanism. It is
serialised to object storage during the feature-engineering stage and reloaded
verbatim during scoring. The scoring path therefore **cannot** apply a transform
that was not fitted, because it has nothing else to apply. That is a structural
guarantee rather than a convention.

The one semantic worth stating precisely: ``fit`` is not a pure fit. Each routine
is fitted and immediately applied in the same loop iteration, so a stateful
routine's ``fit`` receives the previous routine's ``apply`` output. That is what
makes chained transforms (impute -> cap -> encode -> bin) well-defined. The cost
is that fitted state cannot be inspected without materialising data through the
entire chain.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from forecasting_ml_framework.core.metadata import Metadata
from forecasting_ml_framework.core.routine import PreprocessRoutine
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)


class Registry:
  """An ordered chain of :class:`PreprocessRoutine` instances.

  The registry object **is** the fitted-preprocessing artefact. Two state
  persistence strategies coexist inside it, and both are required for the round
  trip to work:

  * state carried in the pickled registry, for transforms whose state is small
    and picklable in place (bounds dictionaries, bin maps, target-encoding
    tables);
  * state written to a separate external artefact under
    ``metadata.intermediate_file_path``, for Spark ML pipelines and large
    estimators that are not reliably picklable.
  """

  def __init__(self, routines: Sequence[PreprocessRoutine] | None = None) -> None:
    """Build an empty or pre-populated registry.

    Args:
      routines: An initial ordered chain.
    """
    self.routines: list[PreprocessRoutine] = list(routines or [])

  # ---------------------------------------------------------------------- #
  # Chain construction
  # ---------------------------------------------------------------------- #
  def append(self, routine: PreprocessRoutine) -> None:
    """Append a routine to the end of the chain.

    Args:
      routine: The routine to append.
    """
    self.routines.append(routine)

  def get_routines(self) -> list[PreprocessRoutine]:
    """Return the ordered chain.

    Returns:
      The live list of routines.
    """
    return self.routines

  def __len__(self) -> int:
    """Return the chain length.

    Returns:
      The number of routines.
    """
    return len(self.routines)

  def __iter__(self) -> Any:
    """Iterate over the chain in execution order.

    Returns:
      An iterator over the routines.
    """
    return iter(self.routines)

  def __add__(self, other: Registry) -> Registry:
    """Concatenate two chains into a new registry.

    The routines are **copied**, not shared. Sharing was the defect: the new list
    is new but the routine objects in it are the same instances, so fitting the
    combined chain overwrote the *original* registry's learned state in place.
    That is the most dangerous class of bug this framework has, because it
    invalidates a persisted train-serve contract: a registry that was fitted, and
    possibly already serialised, silently became a different object. ``copy()``
    already existed and was already correct; these two constructors simply did
    not use it.

    Args:
      other: The registry to append.

    Returns:
      A new registry with copies of the concatenated routines. Neither operand
        is mutated, and neither shares routine state with the result.
    """
    return Registry([routine.copy() for routine in self.routines + other.routines])

  @classmethod
  def merge(cls, *registries: Registry) -> Registry:
    """Concatenate any number of registries into a new one.

    As with :meth:`__add__`, the routines are copied rather than shared, for the
    same reason and with the same consequence.

    Args:
      *registries: The registries to merge, in order.

    Returns:
      A new registry holding copies of every routine.
    """
    routines: list[PreprocessRoutine] = []
    for registry in registries:
      routines.extend(registry.routines)
    return cls([routine.copy() for routine in routines])

  # ---------------------------------------------------------------------- #
  # Execution
  # ---------------------------------------------------------------------- #
  def fit(self, df: Any, metadata: Metadata, params: dict[str, Any] | None = None) -> tuple[Registry, Any, Metadata]:
    """Fit and apply the whole chain, interleaved per routine.

    A single Spark session is acquired before the loop rather than once per
    routine. The accessor refreshes a cloud credential, so calling it per routine
    would pay a subprocess round-trip per transform for a session that
    ``getOrCreate`` returns unchanged -- the cost is real, the refresh is not.

    Row counts are emitted at ``DEBUG`` rather than unconditionally. A full
    ``df.count()`` is a Spark job in its own right, and running one per routine
    purely to produce a log line is the single largest avoidable cost in the
    feature tier, paid twice over because the scoring path repeats the same
    chain.

    Args:
      df: The input frame.
      metadata: The schema contract.
      params: The run parameters, forwarded to the session accessor.

    Returns:
      A three-tuple of the registry, the frame and the contract.
    """
    from forecasting_ml_framework.platform.spark import get_spark_session  # noqa: PLC0415 - platform layer

    get_spark_session(params=params)
    self._protect_identifiers(metadata)
    LOGGER.info(
      'Fitting feature registry', extra={'routines': [routine.name for routine in self.routines], 'rows': len(self)}
    )
    for index, routine in enumerate(self.routines, start=1):
      LOGGER.debug('=== routine %d/%d start: %s ===', index, len(self.routines), routine.name)
      _, df, metadata = routine.fit(df, metadata)
      df, metadata = routine.apply(df, metadata)
      LOGGER.debug('=== routine %d/%d complete: %s ===', index, len(self.routines), routine.name)
    return self, df, metadata

  def _protect_identifiers(self, metadata: Metadata) -> None:
    """Hand every routine the identifier columns it must not transform.

    ``PreprocessRoutine.get_selected_columns`` subtracts ``key_cols_list`` from
    whatever it resolves, and that subtraction is the mechanism behind the
    documented guarantee that an identifier column is never transformed. Nothing
    ever populated ``key_cols_list``, so the guarantee held only by accident --
    identifiers happened not to be in ``feature_cols``, and a routine that named
    one in ``include_cols`` transformed it. The target is protected for the same
    reason: it is the label, and a transformed label is not the label the
    evaluation metrics are defined against.

    It is done here rather than in ``build_registry`` because that function
    receives the shared parameter bag and the configuration list, not the schema
    contract. The contract is what carries the roles, and this is the first point
    at which both are available.

    Args:
      metadata: The schema contract, whose identifier and target roles are read.
    """
    protected = [*(metadata.id_cols or []), *(metadata.seq_cols or [])]
    if metadata.target_col:
      protected.append(metadata.target_col)
    for routine in self.routines:
      routine.key_cols_list = list(protected)

  def apply(self, df: Any, metadata: Metadata, data_type: str | None = None) -> tuple[Any, Metadata]:
    """Apply the fitted chain without refitting.

    Args:
      df: The input frame.
      metadata: The schema contract.
      data_type: An optional label describing the slice being scored. It is
        retained for symmetry with the signature and for run-log
        clarity; the applied behaviour is identical for every slice.

    Returns:
      A two-tuple of the frame and the contract.
    """
    LOGGER.info('Applying fitted feature registry', extra={'slice': data_type, 'routines': len(self.routines)})
    for index, routine in enumerate(self.routines, start=1):
      LOGGER.debug('=== apply %d/%d start: %s ===', index, len(self.routines), routine.name)
      df, metadata = routine.apply(df, metadata)
    return df, metadata

  def transform(
    self,
    df: Any,
    metadata: Metadata,
    params: dict[str, Any] | None = None,
    apply_only: bool = False,
  ) -> tuple[Any, Metadata]:
    """Fit-and-apply in a single call, optionally skipping the fit.

    This is a convenience entry point for notebooks and one-shot runs. The
    production pipelines use :meth:`fit` and :meth:`apply` as two distinct stages,
    because scoring must apply the *persisted* registry rather than a freshly
    fitted one.

    Args:
      df: The input frame.
      metadata: The schema contract.
      params: The run parameters.
      apply_only: When ``True``, only :meth:`apply` is executed.

    Returns:
      A two-tuple of the frame and the contract.
    """
    if apply_only:
      return self.apply(df, metadata)
    _, df, metadata = self.fit(df, metadata, params)
    return df, metadata

  # ---------------------------------------------------------------------- #
  # Serialisation
  # ---------------------------------------------------------------------- #
  def copy(self) -> Registry:
    """Return a deep copy of the registry and its fitted state.

    Returns:
      A deep copy.
    """
    return Registry([routine.copy() for routine in self.routines])

  def __str__(self) -> str:
    """Render the chain for the run log.

    Returns:
      A comma-joined list of routine names.
    """
    return ', '.join(routine.name for routine in self.routines)

  def __repr__(self) -> str:
    """Render a short representation.

    Returns:
      The representation string.
    """
    return f'Registry({len(self.routines)} routines: {self})'
