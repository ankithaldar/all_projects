#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for the two environment flags and the back end they select.

The properties under test are the ones the dual-backend model rests on, and they
are all falsifiable: the default must be the managed path, the two flags must be
independent in both directions, and the legacy spelling must keep working during
its compatibility window. Each is a two-line assertion, which is the point --
these are contract tests for a mechanism whose failure mode is a production run
silently redirected at a developer's machine.
'''

import pytest

from forecasting_ml_framework import constants
from forecasting_ml_framework.exceptions import ConfigurationError
from forecasting_ml_framework.platform.environment import (
  as_flag,
  resolve_artefact_root,
  resolve_environment,
)


class TestDefaultIsManaged:
  '''The default must never redirect a run to a local store.'''

  def test_absent_configuration_resolves_to_the_managed_backend(self) -> None:
    environment = resolve_environment({})
    assert environment.db_backend == constants.BACKEND_DISTRIBUTED
    assert environment.is_distributed

  def test_absent_configuration_is_not_local(self) -> None:
    assert resolve_environment({}).is_local is False

  def test_absent_configuration_does_not_use_local_artefacts(self) -> None:
    assert resolve_environment({}).uses_local_artefacts is False


class TestFlagOrthogonality:
  '''The two flags govern unrelated concerns and must not imply each other.'''

  def test_local_alone_selects_the_embedded_backend(self) -> None:
    environment = resolve_environment({'is_local': True})
    assert environment.db_backend == constants.BACKEND_EMBEDDED
    assert environment.is_hosted is False

  def test_hosted_alone_leaves_the_managed_backend(self) -> None:
    environment = resolve_environment({'is_hosted_env': True})
    assert environment.db_backend == constants.BACKEND_DISTRIBUTED
    assert environment.is_local is False

  def test_both_flags_is_the_redundant_but_valid_combination(self) -> None:
    environment = resolve_environment({'is_hosted_env': True, 'is_local': True})
    assert environment.is_hosted is True
    assert environment.is_local is True
    assert environment.db_backend == constants.BACKEND_EMBEDDED

  def test_is_local_selects_local_artefacts_as_well(self) -> None:
    assert resolve_environment({'is_local': True}).uses_local_artefacts is True


class TestLegacySpelling:
  '''The rename carries a compatibility window.'''

  def test_legacy_name_is_honoured(self) -> None:
    environment = resolve_environment({'is_hosted_platform': True})
    assert environment.is_hosted is True
    assert environment.used_legacy_flag is True

  def test_current_name_wins_over_the_legacy_name(self) -> None:
    environment = resolve_environment({'is_hosted_env': False, 'is_hosted_platform': True})
    assert environment.is_hosted is False
    assert environment.used_legacy_flag is False

  def test_legacy_flag_is_reported_for_removal(self) -> None:
    assert resolve_environment({'is_hosted_platform': True}).used_legacy_flag is True


class TestStringSpellings:
  '''A quoted YAML scalar is truthy, so every spelling must be interpreted.'''

  @pytest.mark.parametrize(
    'raw',
    [True, 'true', 'True', 'TRUE', ' yes ', 'on', '1'],
  )
  def test_affirmative_spellings(self, raw: object) -> None:
    assert as_flag(raw) is True

  @pytest.mark.parametrize(
    'raw',
    [False, 'false', 'False', 'FALSE', ' no ', 'off', '0', ''],
  )
  def test_negative_spellings(self, raw: object) -> None:
    assert as_flag(raw) is False

  def test_a_quoted_false_is_false(self) -> None:
    # The defect this guards: 'False' is a non-empty string, so an unquoted read
    # made it truthy and selected the wrong pipeline family.
    assert as_flag('False') is False

  def test_absent_value_uses_the_default(self) -> None:
    assert as_flag(None, default=True) is True
    assert as_flag(None, default=False) is False

  def test_unrecognised_value_raises(self) -> None:
    with pytest.raises(ConfigurationError):
      as_flag('maybe', key='is_local')


class TestBackendOverride:
  '''The explicit override, and its guard rail.'''

  def test_explicit_override_is_honoured(self) -> None:
    environment = resolve_environment({'db_backend': 'embedded'})
    assert environment.db_backend == constants.BACKEND_EMBEDDED

  def test_explicit_override_beats_the_implicit_rule(self) -> None:
    environment = resolve_environment({'is_local': True, 'db_backend': 'distributed'})
    assert environment.db_backend == constants.BACKEND_DISTRIBUTED

  def test_unknown_backend_is_rejected(self) -> None:
    with pytest.raises(ConfigurationError):
      resolve_environment({'db_backend': 'postgres'})


class TestArtefactRoot:
  '''The dual-root rule for artefacts.'''

  def test_local_run_resolves_to_the_local_root(self) -> None:
    assert resolve_artefact_root({'is_local': True}) == constants.LOCAL_ARTEFACT_ROOT

  def test_managed_run_declares_no_local_root(self) -> None:
    assert resolve_artefact_root({}) == ''

  def test_a_declared_root_wins(self) -> None:
    assert resolve_artefact_root({'is_local': True, 'artefact_root': '/mnt/out'}) == '/mnt/out'
