#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the domain layer: Metadata, the routine contract and the registry."""

from __future__ import annotations

import pytest

from forecasting_ml_framework.core.metadata import Metadata
from forecasting_ml_framework.core.registry import Registry
from forecasting_ml_framework.core.routine import PreprocessRoutine
from forecasting_ml_framework.exceptions import (
  ColumnRoleConflictError,
  RowCountAssertionError,
  TargetCardinalityError,
)
from forecasting_ml_framework.observability.quality import (
  QualityPolicy,
  assert_column_roles_disjoint,
  assert_row_count,
  assert_target_cardinality,
)
from forecasting_ml_framework.preprocessing.routines import Imputations


class TestFeatureResolution:
  """``feature_resolution`` is the single feature-resolution funnel."""

  def test_membership_is_double_gated(self, metadata: Metadata) -> None:
    """A column must appear in the requested role *and* in ``feature_cols``.

    Args:
      metadata: The contract fixture.
    """
    # 'plan_tier' is a categorical AND a feature, so it survives.
    resolved = metadata.feature_resolution(args=('categorical_cols',))
    assert resolved == ['plan_tier', 'region_code']

    # A literal list of a column that is not a declared feature is excluded.
    resolved = metadata.feature_resolution(args=(['cust_id'],))
    assert resolved == []

  def test_order_is_preserved_from_the_source_list(self, metadata: Metadata) -> None:
    """The resolved order is the model's feature order for the artefact's life.

    Args:
      metadata: The contract fixture.
    """
    assert metadata.feature_resolution() == metadata.feature_cols

  def test_exclusions_are_prefix_anchored_regular_expressions(self) -> None:
    """``^tenure_.*`` excludes a family; a bare prefix also matches descendants."""
    metadata = Metadata(feature_cols=['tenure_days', 'tenure_01', 'other'])
    assert metadata.feature_resolution(feature_exclusions=['^tenure_.*']) == ['other']
    assert metadata.feature_resolution(feature_exclusions=['tenure']) == ['other']

  def test_inclusions_are_applied_before_exclusions(self) -> None:
    """Exclusions win, so an exclusion cannot be undone by an inclusion.

    Args:
      None.
    """
    metadata = Metadata(feature_cols=['a', 'b', 'c'])
    resolved = metadata.feature_resolution(
      feature_inclusions=['a', 'b'], feature_exclusions=['^b$']
    )
    assert resolved == ['a']

  def test_union_of_roles_and_literals(self) -> None:
    """Roles and literal lists can be unioned, deduplicated, in order.

    Args:
      None.
    """
    metadata = Metadata(feature_cols=['a', 'b', 'c', 'd'], numerical_cols=['a', 'b'])
    assert metadata.feature_resolution(args=('numerical_cols', ['c', 'a'])) == ['a', 'b', 'c']


class TestMutation:
  """The mutation protocol is the part of the family most often got wrong."""

  def test_copy_is_deep(self, metadata: Metadata) -> None:
    """A copy shares nothing with a.

    Args:
      metadata: The contract fixture.
    """
    copied = metadata.copy()
    copied.feature_cols.append('extra')
    assert 'extra' not in metadata.feature_cols
    assert copied.target_col == metadata.target_col

  def test_register_derived_feature_enters_both_numeric_namespaces(self) -> None:
    """A derived column is visible to the trainer and to later numeric routines.

    Args:
      None.
    """
    metadata = Metadata(feature_cols=['usage_01'], numerical_cols=['usage_01'])
    metadata.register_derived_feature('usage_01_avg')
    assert 'usage_01_avg' in metadata.feature_cols
    assert 'usage_01_avg' in metadata.numerical_cols

  def test_register_derived_feature_is_idempotent(self) -> None:
    """Registering the same name twice must not duplicate it.

    Args:
      None.
    """
    metadata = Metadata(feature_cols=[], numerical_cols=[])
    metadata.register_derived_feature('x')
    metadata.register_derived_feature('x')
    assert metadata.feature_cols == ['x']

  def test_replace_feature_retires_the_source(self) -> None:
    """A re-encoded column replaces its source rather than joining it.

    Args:
      None.
    """
    metadata = Metadata(feature_cols=['plan_tier', 'usage_01'], categorical_cols=['plan_tier'])
    metadata.replace_feature('plan_tier', 'plan_tier_le')
    assert 'plan_tier' not in metadata.feature_cols
    assert 'plan_tier_le' in metadata.feature_cols
    assert 'plan_tier_le' in metadata.numerical_cols
    assert 'usage_01' in metadata.feature_cols

  def test_reclassify_moves_between_namespaces(self) -> None:
    """A binned categorical becomes numerical.

    Args:
      None.
    """
    metadata = Metadata(feature_cols=['a', 'b'], categorical_cols=['a'], numerical_cols=['b'])
    metadata.reclassify_as_numerical(['a'])
    assert 'a' not in metadata.categorical_cols
    assert metadata.numerical_cols == ['b', 'a']

  def test_detach_sequence_features_removes_arrays_from_the_model_list(self) -> None:
    """The array column must not reach an estimator.

    Args:
      None.
    """
    metadata = Metadata(feature_cols=['usage_arr', 'usage_01'], seq_cols=['usage_arr'], numerical_cols=['usage_01'])
    removed = metadata.detach_sequence_features()
    assert removed == ['usage_arr']
    assert metadata.feature_cols == ['usage_01']
    assert metadata.seq_cols == ['usage_arr']


class TestAssertions:
  """the framework observed every count and asserted none."""

  def test_row_count_within_tolerance_passes(self) -> None:
    """A small drift is within a configured tolerance.

    Args:
      None.
    """
    policy = QualityPolicy(expected_row_count=1000, row_count_tolerance=0.01)
    assert_row_count(1005, policy, 'stage')

  def test_row_count_beyond_tolerance_raises(self) -> None:
    """A large drift is a hard failure, naming the stage and both counts.

    Args:
      None.
    """
    policy = QualityPolicy(expected_row_count=1000, row_count_tolerance=0.01)
    with pytest.raises(RowCountAssertionError) as excinfo:
      assert_row_count(600, policy, 'fit_registry')
    assert 'fit_registry' in str(excinfo.value)

  def test_unset_expected_count_disables_the_check(self) -> None:
    """No baseline means no assertion.

    Args:
      None.
    """
    assert_row_count(0, QualityPolicy(), 'stage')

  def test_single_class_target_raises(self) -> None:
    """A single-class frame makes every downstream metric undefined.

    Args:
      None.
    """
    with pytest.raises(TargetCardinalityError):
      assert_target_cardinality(1, 'label')

  def test_two_class_target_passes(self) -> None:
    """Two classes is learnable.

    Args:
      None.
    """
    assert_target_cardinality(2, 'label')

  def test_column_role_overlap_raises(self) -> None:
    """A column in two roles is a configuration issue.

    Args:
      None.
    """
    with pytest.raises(ColumnRoleConflictError) as excinfo:
      assert_column_roles_disjoint({'NUM_COL': ['a'], 'SEQ_COL': ['a']})
    assert 'a' in str(excinfo.value)

  def test_disjoint_roles_pass(self) -> None:
    """A disjoint contract is valid.

    Args:
      None.
    """
    assert_column_roles_disjoint({'NUM_COL': ['a'], 'CAT_COL': ['b'], 'SEQ_COL': ['c']})


class TestRoutineContract:
  """The two-phase contract and its two-tier configuration precedence."""

  def test_fit_defaults_to_apply(self) -> None:
    """A stateless transform needs to override neither method.

    The base implementation delegates to ``apply``, which is exactly right for a
    transform with no learned state.
    """
    routine = StatelessRoutine()
    fitted, frame, contract = routine.fit('frame', 'contract')
    assert fitted is routine
    assert (frame, contract) == ('frame', 'contract')
    assert routine.applied is True

  def test_load_params_gives_local_precedence(self) -> None:
    """A routine-specific key wins over the shared bag.

    Args:
      None.
    """
    routine = Imputations(global_params={'strategy': 'median', 'val_count': 5}, params={'strategy': 'mean'})
    assert routine.strategy == 'mean'
    assert routine.val_count == 5

  def test_load_params_tolerates_a_none_local_block(self) -> None:
    """A dereferenced ``params`` inside the shared block.

    ``load_params(globals, None)`` raised ``AttributeError``.

    Args:
      None.
    """
    routine = Imputations(global_params={'val_count': 5}, params=None)
    assert routine.val_count == 5

  def test_get_param_handle_is_callable_on_an_instance(self) -> None:
    """It was declared as a bare instance method with no parameters.

    Calling it on an instance raised ``TypeError`` across the whole family.

    Args:
      None.
    """
    routine = Imputations(global_params=None, params={})
    assert routine.get_param_handle() == 'Imputations'
    assert Imputations.get_param_handle() == 'Imputations'

  def test_column_resolution_precedence(self) -> None:
    """Include wins, then exclude, then the protected key columns.

    Args:
      None.
    """
    routine = PreprocessRoutine()
    routine.key_cols_list = ['service_id']
    assert routine.get_selected_columns(None, None, ['a', 'service_id', 'b']) == ['a', 'b']
    assert routine.get_selected_columns(['b'], None, ['a', 'b', 'c']) == ['a', 'c']
    assert routine.get_selected_columns(None, ['c'], ['a', 'b', 'c']) == ['c']

  def test_declared_defaults_become_attributes(self) -> None:
    """Every key in the schema exists as an attribute.

    Args:
      None.
    """
    routine = Imputations(global_params=None, params={})
    defaults = routine.get_default_dict()
    for key, value in defaults.items():
      assert getattr(routine, key) == value

  def test_str_and_copy_are_inherited_not_redeclared(self) -> None:
    """Ten classes re-declared these identically, seventeen omitted them.

    Args:
      None.
    """
    routine = Imputations(global_params=None, params={'strategy': 'mean'})
    assert 'strategy: mean' in str(routine)
    assert routine.copy() is not routine


class TestRegistry:
  """The registry is the train/serve-parity mechanism."""

  def test_append_and_length(self) -> None:
    """The chain reports its contents.

    Args:
      None.
    """
    registry = Registry()
    routine = Imputations(global_params=None, params={})
    registry.append(routine)
    assert len(registry) == 1
    assert registry.get_routines() == [routine]

  def test_addition_returns_a_new_registry(self) -> None:
    """Concatenation does not mutate either operand.

    Args:
      None.
    """
    left = Registry([Imputations(global_params=None, params={})])
    right = Registry([CapOutliersStub()])
    merged = left + right
    assert len(merged) == 2
    assert len(left) == 1
    assert len(right) == 1

  def test_merge_is_class_level(self) -> None:
    """Class-level merge concatenates any number of registries.

    Args:
      None.
    """
    merged = Registry.merge(
      Registry([Imputations(global_params=None, params={})]),
      Registry([Imputations(global_params=None, params={})]),
    )
    assert len(merged) == 2

  def test_copy_is_deep(self) -> None:
    """A's copy referenced an unimported ``copy``.

    Args:
      None.
    """
    routine = Imputations(global_params=None, params={})
    routine.use_cols = ['a']
    copied = Registry([routine]).copy()
    copied.routines[0].use_cols.append('b')
    assert routine.use_cols == ['a']


class CapOutliersStub(PreprocessRoutine):
  """A minimal second routine, so the registry tests have two to work with."""

  def __init__(self, global_params=None, params=None) -> None:
    """Build the stub.

    Args:
      global_params: Unused.
      params: Unused.
    """
    super().__init__()
    self.name = 'CapOutliersStub'


class StatelessRoutine(PreprocessRoutine):
  """A routine that overrides neither ``fit`` nor ``apply``.

  It exists to pin down the base contract: ``fit`` must default to ``apply``,
  which is what lets a stateless transform declare neither method.
  """

  def __init__(self) -> None:
    """Build the routine."""
    super().__init__()
    self.name = 'Stateless'
    self.applied = False

  def apply(self, df, metadata):
    """Record that the transform ran.

    Args:
      df: The input frame.
      metadata: The schema contract.

    Returns:
      A two-tuple of the frame and the contract, unchanged.
    """
    self.applied = True
    return df, metadata
