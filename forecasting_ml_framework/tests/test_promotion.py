#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests for the two-gate promotion decision.

The promotion gate is where the framework makes a governance decision, so its two
gates are tested as decision logic rather than through the warehouse they write
to. Both gates are independent: acceptance compares two models on the holdout, and
temporal compares two observation windows. A challenger must clear both.

The tests focus on :func:`_gate`, the composition point. The defect these pin is
that the temporal verdict was computed and then discarded by an unconditional
reassignment, which left the gate looking enforced at every call site while being
dead in fact.
'''

from datetime import date

import pytest

from forecasting_ml_framework.exceptions import MetricConfigurationError
from forecasting_ml_framework.modeling.metrics import parse_metric
from forecasting_ml_framework.modeling.promotion import PromotionDecision, decide_promotion
from forecasting_ml_framework.nodes.validation_nodes import _gate

BLOCK = 'modeling_demo'
SPEC = parse_metric('LIFT_DECILE_10', kind='classification')


def _parameters() -> dict:
  """Build a minimal parameters document carrying one modelling block.

  Returns:
    The document.
  """
  return {
    'model_key': 'demo',
    BLOCK: {'model_handle': 'sklearn.dummy.DummyClassifier', 'eval_metric': 'LIFT_DECILE_10'},
  }


class TestTemporalGate:
  """The temporal verdict alone decides when only it is consulted."""

  def test_a_later_window_promotes(self) -> None:
    assert decide_promotion(date(2024, 3, 1), date(2024, 2, 1)).action == 'PROMOTE'

  def test_the_same_window_is_a_no_op(self) -> None:
    """Idempotent replay protection."""
    assert decide_promotion(date(2024, 2, 1), date(2024, 2, 1)).action == 'NO_OP'

  def test_an_older_window_is_recorded_only(self) -> None:
    """A model evaluated against stale data cannot supersede a better-informed one."""
    assert decide_promotion(date(2024, 1, 1), date(2024, 2, 1)).action == 'RECORD_ONLY'


class TestGateComposition:
  """Both gates must pass, and the narrower failure must be reported."""

  def test_both_gates_pass_promotes(self) -> None:
    temporal = PromotionDecision('PROMOTE', 'later window')
    assert _gate(True, 0.5, SPEC, 10, temporal).action == 'PROMOTE'

  def test_acceptance_failure_is_recorded_only(self) -> None:
    """A better-informed but worse model is still a worse model."""
    temporal = PromotionDecision('PROMOTE', 'later window')
    assert _gate(False, 12.0, SPEC, 10, temporal).action == 'RECORD_ONLY'

  def test_temporal_failure_blocks_an_accepted_challenger(self) -> None:
    """The defect: this returned PROMOTE because the temporal verdict was discarded."""
    temporal = PromotionDecision('RECORD_ONLY', 'older window')
    decision = _gate(True, 0.1, SPEC, 10, temporal)
    assert decision.action == 'RECORD_ONLY'
    assert 'temporal' in decision.rationale.lower()

  def test_temporal_failure_blocks_even_a_perfect_score(self) -> None:
    temporal = PromotionDecision('RECORD_ONLY', 'older window')
    assert _gate(True, 0.0, SPEC, 10, temporal).action == 'RECORD_ONLY'

  def test_an_idempotent_replay_is_not_a_promotion(self) -> None:
    temporal = PromotionDecision('NO_OP', 'already recorded')
    assert _gate(True, 0.0, SPEC, 10, temporal).action != 'PROMOTE'

  def test_the_band_used_is_recorded_on_every_path(self) -> None:
    temporal = PromotionDecision('PROMOTE', 'later window')
    assert _gate(True, 0.5, SPEC, 7, temporal).band_used == 7
    assert _gate(False, 12.0, SPEC, 7, temporal).band_used == 7


class TestMetricConfiguration:
  """An unrecognised metric fails the promotion rather than substituting one."""

  def test_a_typo_raises(self) -> None:
    with pytest.raises(MetricConfigurationError):
      _parameters()[BLOCK]['eval_metric'] = 'LFT_DECILE_10'
      parse_metric('LFT_DECILE_10', kind='classification', strict=True)

  def test_absent_metric_uses_the_declared_default(self) -> None:
    assert parse_metric(None, kind='classification').name == 'LIFT_DECILE_10'


class TestRegressionGateDirection:
  """The acceptance sign follows the metric's declared direction."""

  def test_a_loss_metric_is_lower_is_better(self) -> None:
    spec = parse_metric('RMSE', kind='regression')
    assert spec.higher_is_better is False
    assert 'lower-is-better' in _gate(True, 1.0, spec, 0, PromotionDecision('PROMOTE', 'ok')).rationale

  def test_a_score_metric_is_higher_is_better(self) -> None:
    spec = parse_metric('R2', kind='regression')
    assert spec.higher_is_better is True
    assert 'higher-is-better' in _gate(True, 1.0, spec, 0, PromotionDecision('PROMOTE', 'ok')).rationale
