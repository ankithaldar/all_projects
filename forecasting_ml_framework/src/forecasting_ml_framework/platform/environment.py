#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Resolution of the two environment flags.

The framework distinguishes the four **mode flags** -- task family, training
strategy, execution intent, holdout policy -- from the two **environment flags**
documented here. The mode flags decide *what a run computes*; the environment
flags decide *where it runs*. The separation is deliberate: it is what makes a
local run a valid correctness test of a production one, because the two perform
the same work and differ only in where the work lands.

Two flags, governing unrelated concerns:

``is_hosted_env``
    The run executes inside a managed workspace rather than on the managed
    cluster. It governs **how the run obtains credentials** and nothing else.

``is_local``
    The run executes on a developer's own machine, against no cloud resources.
    It governs **which persistence back end every data transaction uses**.

They are orthogonal rather than positions of one switch, and both directions
matter. A workspace run may legitimately use hosted storage, so the hosted flag
must not imply a local store; and a developer's machine has no workload identity
to inherit, so the local flag must not imply a workspace. Conflating them is what
produces the class of defect where a local run is assumed to behave like a
distributed one.

The rename is a naming change only. The hosted flag was previously named for the
specific workspace product that first required it; a flag named for a product
cannot be reasoned about, cannot be transferred to a different workspace, and
misleads every reader without context for it. The legacy spelling is honoured for
one compatibility window, and its use is logged so the remaining call sites can be
found by measurement rather than by reading every configuration file.

The back end follows ``is_local`` **alone**. An explicit ``db_backend`` overrides
the implicit rule for a deployment that genuinely needs to read from one engine
and write elsewhere, but ordinary configuration needs only the flag.

Absent configuration is a production run. An environment that sets neither flag
behaves exactly as it did before the embedded back end existed, because the
failure that matters most here is a production run silently writing to a
developer's machine, and a default of "hosted" makes that impossible.
'''

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from forecasting_ml_framework import constants
from forecasting_ml_framework.exceptions import ConfigurationError
from forecasting_ml_framework.observability.logging import get_logger

LOGGER = get_logger(__name__)

#: Strings accepted as an affirmative boolean. The comparison is case-insensitive
#: because a YAML document can spell a flag in any case, and a non-empty string is
#: truthy -- so an unquoted read inverts the meaning of a flag set to false.
_TRUE_TOKENS = frozenset({'true', '1', 'yes', 'on'})
_FALSE_TOKENS = frozenset({'false', '0', 'no', 'off', ''})


def as_flag(value: Any, default: bool = False, key: str = '') -> bool:
  '''Interpret an environment or mode flag.

  Args:
    value: The raw value, which may be a boolean, a string, or absent.
    default: The value to use when ``value`` is ``None``.
    key: The flag's name, used in the error message.

  Returns:
    The boolean interpretation. Anything unrecognised raises rather than defaulting,
    because a flag that silently reads as false is indistinguishable from one the
    operator set to false -- and a credential path selected by the wrong reading of a
    flag fails later, somewhere unrelated.

  Raises:
    ConfigurationError: If the value is neither affirmative nor negative.
  '''
  if value is None:
    return default
  if isinstance(value, bool):
    return value
  token = str(value).strip().lower()
  if token in _TRUE_TOKENS:
    return True
  if token in _FALSE_TOKENS:
    return False
  raise ConfigurationError(
    f'Cannot interpret {key or "flag"}={value!r} as a boolean. Use true or false.',
    flag=key or None,
    value=str(value),
  )


@dataclass(frozen=True)
class RunEnvironment:
  '''The resolved environment for one run.

  Attributes:
    is_hosted: Whether the run executes inside a managed workspace. Governs
      credential acquisition only.
    is_local: Whether the run executes against the embedded back end. Governs
      persistence only.
    db_backend: The resolved relational back end.
    used_legacy_flag: Whether the compatibility path was taken. Recorded so the
      rename can be completed from measurement rather than by inspection.
  '''

  is_hosted: bool = False
  is_local: bool = False
  db_backend: str = constants.BACKEND_DISTRIBUTED
  used_legacy_flag: bool = False

  @property
  def is_distributed(self) -> bool:
    '''Whether data transactions target the managed warehouse.

    Returns:
      ``True`` unless the embedded back end was selected.
    '''
    return self.db_backend == constants.BACKEND_DISTRIBUTED

  @property
  def uses_local_artefacts(self) -> bool:
    '''Whether artefacts are held under a local root rather than the managed store.

    The same flag selects both substitutions, which is deliberate: a run must not
    be able to write its tables to one back end and its model weights to another,
    because that configuration exercises neither environment correctly.

    Returns:
      ``True`` when the local root applies.
    '''
    return self.is_local

  def __str__(self) -> str:
    '''Render the resolved environment for the run log.

    Returns:
      A single-line description.
    '''
    hosted = 'hosted_env' if self.is_hosted else 'cluster'
    local = 'local' if self.is_local else 'managed'
    return f'{self.db_backend} backend, {hosted} credentials, {local} artefacts'


def resolve_environment(source: Mapping[str, Any] | None = None) -> RunEnvironment:
  '''Resolve the two environment flags and the back end they select.

  Resolution order for the hosted flag is: the current name, then the legacy
  name, then false. The legacy name is honoured rather than rejected so that the
  rename can be introduced without a coordinated edit of every workflow definition
  that supplies it, which is the lower-risk sequence.

  Args:
    source: The parameter mapping to read. Defaults to the process environment, so
      a caller that has no parameters document still gets a resolved value.

  Returns:
    The resolved environment.

  Raises:
    ConfigurationError: If a flag cannot be interpreted, or ``db_backend`` names an
      unknown back end.
  '''
  values = os.environ if source is None else source

  legacy = values.get(constants.ENV_IS_HOSTED_LEGACY)
  current = values.get(constants.ENV_IS_HOSTED)
  used_legacy = current is None and legacy is not None
  if used_legacy:
    LOGGER.info(
      'The %r flag is deprecated; use %r. The hosted flag names the capability being selected '
      'rather than the product that motivated it.',
      constants.ENV_IS_HOSTED_LEGACY,
      constants.ENV_IS_HOSTED,
    )
  is_hosted = as_flag(
    current if current is not None else legacy,
    default=False,
    key=constants.ENV_IS_HOSTED if not used_legacy else constants.ENV_IS_HOSTED_LEGACY,
  )

  is_local = as_flag(values.get(constants.ENV_IS_LOCAL), default=False, key=constants.ENV_IS_LOCAL)
  backend = _resolve_backend(values.get(constants.ENV_DB_BACKEND), is_local)

  environment = RunEnvironment(
    is_hosted=is_hosted,
    is_local=is_local,
    db_backend=backend,
    used_legacy_flag=used_legacy,
  )
  LOGGER.info('Resolved the run environment', extra={'environment': str(environment)})
  return environment


def _resolve_backend(declared: Any, is_local: bool) -> str:
  '''Resolve the relational back end.

  Args:
    declared: The explicit ``db_backend`` override, if any.
    is_local: Whether the local flag is set.

  Returns:
    The resolved back-end identifier.

  Raises:
    ConfigurationError: If an override names an unknown back end. An unknown name is
      rejected rather than defaulted, because defaulting an unrecognised back end
      would send a run to the wrong store without any indication that the setting
      had been ignored.
  '''
  if declared is None or not str(declared).strip():
    return constants.BACKEND_EMBEDDED if is_local else constants.BACKEND_DISTRIBUTED

  backend = str(declared).strip().lower()
  if backend not in constants.DB_BACKENDS:
    raise ConfigurationError(
      f'Unknown db_backend {declared!r}. Supported back ends: {list(constants.DB_BACKENDS)}.',
      db_backend=str(declared),
      supported=list(constants.DB_BACKENDS),
    )
  if backend == constants.BACKEND_EMBEDDED and not is_local:
    LOGGER.warning(
      'db_backend is set to the embedded store without is_local. The embedded store holds '
      'single-process frames and one writer at a time, and is not a production store; confirm '
      'this is intended.',
      extra={'db_backend': backend},
    )
  return backend


def resolve_artefact_root(source: Mapping[str, Any] | None = None) -> str:
  '''Resolve the root under which artefacts are held.

  Args:
    source: The parameter mapping to read.

  Returns:
    The configured root when one is declared, and otherwise the local temporary root
    when the local flag is set. A managed run that declares no root returns an empty
    string, which the caller must treat as "keep the configured path" rather than as
    a root in its own right.
  '''
  declared = (source or {}).get('artefact_root')
  if declared:
    return str(declared)
  if resolve_environment(source).uses_local_artefacts:
    return constants.LOCAL_ARTEFACT_ROOT
  return ''
