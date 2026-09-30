#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The two-phase feature-transform contract.

Every feature transform implements two methods::

    fit(df, metadata)   ->  (self, df, metadata)   # learn state, optionally transform
    apply(df, metadata) ->  (df, metadata)         # transform using learned state

The base class makes ``fit`` default to ``apply``, so a stateless transform
overrides neither and simply inherits. Four mechanisms live in this small class:

``get_default_dict``
  The configuration schema system. Each subclass declares its complete parameter
  surface and ``__init__`` installs every key as an attribute. There is no
  argument parsing and no type checking beyond the declared defaults, so the
  subclass owns its validation.

``load_params``
  Two-tier configuration precedence. Routine-specific parameters are applied
  first; the shared ``global_params`` bag is then applied **only for keys the
  routine-specific block did not set**.

``get_selected_columns``
  The shared column resolver. Precedence is ``include_cols`` if given else the
  supplied role list, then subtract ``exclude_cols``, then subtract the protected
  key columns.

``fit`` / ``apply``
  The template. The registry interleaves them per routine, so routine *n*'s
  ``fit`` sees routine *n-1*'s output.
"""

from __future__ import annotations

import copy as copy_module
from collections.abc import Mapping, Sequence
from typing import Any

from forecasting_ml_framework.core.metadata import Metadata
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)


class PreprocessRoutine:
  """Base class for every feature transform.

  The class is deliberately small: the leverage comes from how many routines,
  estimators and pipelines attach to it.

  Attributes:
    skip: Whether the routine participates in the chain. Membership is decided
      by configuration, not by this attribute.
    key_cols_list: Never-transform identifier columns, subtracted from every
      resolved target.
    name: The routine's display name, used in the run log.
    transform_cols: The columns resolved by the most recent
      :meth:`get_selected_columns` call, cached as a side effect.
  """

  def __init__(self) -> None:
    """Install the declared defaults onto the instance."""
    self.skip = False
    self.key_cols_list: list[str] = []
    self.name = type(self).__name__
    self.transform_cols: list[str] = []
    for key, value in self.get_default_dict().items():
      setattr(self, key, value)

  # ---------------------------------------------------------------------- #
  # Declarative configuration schema
  # ---------------------------------------------------------------------- #
  @classmethod
  def get_param_handle(cls) -> str:
    """Return the configuration key that registers this routine.

    A class method rather than an instance method: the handle is a property of
    the *type*, since it is what identifies the routine in configuration, and it
    must be callable both on the class and on an instance. A bare method taking
    no parameters raises ``TypeError`` the moment anything calls it on an
    instance, and a latent error of that kind is invisible until the call lands.

    Returns:
      The routine's configuration key.
    """
    return cls.__name__

  def get_default_dict(self) -> dict[str, Any]:
    """Declare the routine's complete parameter surface.

    Returns:
      A mapping of configuration key to default value. Always includes ``skip``.
    """
    return {'skip': False}

  # ---------------------------------------------------------------------- #
  # Two-tier configuration precedence
  # ---------------------------------------------------------------------- #
  def load_params(
    self,
    global_params: Mapping[str, Any] | None,
    params: Mapping[str, Any] | None,
  ) -> PreprocessRoutine:
    """Apply routine-specific then shared parameters onto this instance.

    Both arguments are normalised before either is read, so
    ``load_params(globals, None)`` behaves identically to
    ``load_params(globals, {})``. The shared bag is optional by configuration --
    a routine block may be the only block present -- so a local block that is
    absent must not be a distinct code path from one that is empty.

    Args:
      global_params: The shared parameter bag applied only where the
        routine-specific block did not set a key.
      params: The routine's own configuration block.

    Returns:
      ``self``, so construction can be written as a single expression.
    """
    local = dict(params or {})
    shared = dict(global_params or {})
    for key, value in local.items():
      setattr(self, key, value)
    for key, value in shared.items():
      if key not in local:
        setattr(self, key, value)
    return self

  # ---------------------------------------------------------------------- #
  # The shared column resolver
  # ---------------------------------------------------------------------- #
  def get_selected_columns(
    self,
    exclude_cols: Sequence[str] | None,
    include_cols: Sequence[str] | None,
    cols_all: Sequence[str],
  ) -> list[str]:
    """Resolve which columns this routine should act on.

    The *third* argument is the semantically meaningful choice: it is the role
    list the routine is scoped to.

    Args:
      exclude_cols: Literal column names to exclude.
      include_cols: A literal allow-list that replaces ``cols_all`` entirely.
        This is how a sequence transform names its array columns explicitly.
      cols_all: The role list to target, e.g. ``metadata.numerical_cols``.

    Returns:
      The resolved column list, also cached on ``self.transform_cols``.
    """
    self.transform_cols = list(include_cols) if include_cols else list(cols_all)
    if exclude_cols:
      excluded = set(exclude_cols)
      self.transform_cols = [column for column in self.transform_cols if column not in excluded]
    if self.key_cols_list:
      protected = set(self.key_cols_list)
      self.transform_cols = [column for column in self.transform_cols if column not in protected]
    return self.transform_cols

  # ---------------------------------------------------------------------- #
  # The template
  # ---------------------------------------------------------------------- #
  def fit(self, df: Any, metadata: Metadata) -> tuple[PreprocessRoutine, Any, Metadata]:
    """Learn state, then return the routine for serialisation.

    The default implementation delegates to :meth:`apply`, which is exactly
    right for a stateless transform.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A three-tuple of the routine, the frame and the contract.
    """
    df, metadata = self.apply(df, metadata)
    return self, df, metadata

  def apply(self, df: Any, metadata: Metadata) -> tuple[Any, Metadata]:
    """Transform the frame using learned state.

    The default implementation is a no-op.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A two-tuple of the frame and the contract.
    """
    return df, metadata

  # ---------------------------------------------------------------------- #
  # Serialisation contract
  # ---------------------------------------------------------------------- #
  def __str__(self) -> str:
    """Render the routine's effective configuration for the run log.

    Returns:
      A comma-joined attribute dump.
    """
    return ', '.join(f'{key}: {value}' for key, value in self.__dict__.items() if value not in (None, [], {}, ''))

  def copy(self) -> PreprocessRoutine:
    """Return a deep copy of the routine.

    A routine library re-declared this identically in ten classes and
    omitted it from the rest; declaring it once here removes roughly nine hundred
    lines of boilerplate.

    Returns:
      A deep copy.
    """
    return copy_module.deepcopy(self)

  def __repr__(self) -> str:
    """Render a short representation.

    Returns:
      The representation string.
    """
    return f'{type(self).__name__}(name={self.name!r}, skip={self.skip!r})'
