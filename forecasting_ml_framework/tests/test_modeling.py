#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the modelling tier: calibration, thresholds, metrics and the gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting_ml_framework import constants
from forecasting_ml_framework.exceptions import (
  MetricBandEmptyError,
  MetricConfigurationError,
)
from forecasting_ml_framework.modeling.calibration import calibrate_probability, select_threshold
from forecasting_ml_framework.modeling.metrics import (
  DEFAULT_PROMOTION_METRIC,
  METRIC_REGISTRY,
  acceptance_threshold,
  band_metrics,
  evaluate_acceptance,
  parse_metric,
  select_band,
)
from forecasting_ml_framework.modeling.promotion import decide_promotion


@pytest.fixture
def labelled_scores() -> pd.DataFrame:
  """Return a labelled frame with a strong, well-separated score.

  Returns:
    A pandas ``DataFrame`` with a label, a hard prediction and a score.
  """
  rng = np.random.default_rng(7)
  size = 1000
  frame = pd.DataFrame({'label': 0, 'prediction': 0, 'score': rng.uniform(0, 0.3, size)})
  positive = frame.sample(n=40, random_state=7)
  frame.loc[positive.index, 'label'] = 1
  frame.loc[positive.index, 'score'] = rng.uniform(0.7, 1.0, len(positive))
  frame['prediction'] = (frame['score'] >= 0.5).astype(int)
  return frame


class TestCalibration:
  """Exact Bayes prior-shift correction."""

  def test_identity_when_no_reweighting_was_applied(self) -> None:
    """With no reweighting, real_rate equals sample_rate and the correction is identity.

    Args:
      None.
    """
    raw = pd.Series([0.1, 0.5, 0.9])
    calibrated = calibrate_probability(raw, real_rate=0.05, sample_rate=0.05)
    np.testing.assert_allclose(calibrated.to_numpy(), raw.to_numpy(), atol=1e-9)

  def test_upweighting_training_inflation_lowers_the_score(self) -> None:
    """An inflated training rate means the raw score overstates the true probability.

    Args:
      None.
    """
    raw = pd.Series([0.5])
    calibrated = calibrate_probability(raw, real_rate=0.02, sample_rate=0.50)
    assert calibrated.iloc[0] < 0.5
    assert 0.0 <= calibrated.iloc[0] <= 1.0

  def test_matches_the_closed_form(self) -> None:
    """The implementation reproduces the Saerens closed form.

    The ratio direction is asserted explicitly, because inverting it produces a
    score that still looks like a probability while moving the wrong way -- an
    error that is invisible in every downstream check.

    Args:
      None.
    """
    raw, real, sample = 0.7, 0.03, 0.4
    expected = (raw * real / sample) / ((1 - raw) * (1 - real) / (1 - sample) + raw * real / sample)
    assert calibrate_probability(pd.Series([raw]), real, sample).iloc[0] == pytest.approx(expected)

  @pytest.mark.parametrize(('raw', 'real', 'sample'), [(0.7, 0.03, 0.40), (0.3, 0.02, 0.15), (0.9, 0.10, 0.50)])
  def test_enriched_training_sample_lowers_the_score(self, raw: float, real: float, sample: float) -> None:
    """``pi_true < pi_sample`` must reduce the odds, never increase them.

    Args:
      raw: The raw score.
      real: The population positive rate.
      sample: The training positive rate.
    """
    assert calibrate_probability(pd.Series([raw]), real, sample).iloc[0] < raw

  @pytest.mark.parametrize(('raw', 'real', 'sample'), [(0.3, 0.40, 0.03), (0.2, 0.15, 0.02)])
  def test_depleted_training_sample_raises_the_score(self, raw: float, real: float, sample: float) -> None:
    """The correction is symmetric: a depleted sample must raise the score.

    Args:
      raw: The raw score.
      real: The population positive rate.
      sample: The training positive rate.
    """
    assert calibrate_probability(pd.Series([raw]), real, sample).iloc[0] > raw

  @pytest.mark.parametrize(('real', 'sample'), [(0.0, 0.4), (0.03, 0.0)])
  def test_degenerate_rate_returns_the_raw_score(self, real: float, sample: float) -> None:
    """A zero rate must not divide by zero and silently disable calibration.

    Args:
      real: The population rate.
      sample: The training rate.
    """
    raw = pd.Series([0.42])
    assert calibrate_probability(raw, real, sample).iloc[0] == pytest.approx(0.42)


class TestThresholdSelection:
  """Four criteria, one of which zooms because a coarse grid cannot answer."""

  @pytest.mark.parametrize('method', ['pr_curve', 'roc_curve', 'mcc_curve', 'ks_statistics'])
  def test_every_criterion_returns_a_usable_threshold(self, method: str, labelled_scores: pd.DataFrame) -> None:
    """Each criterion returns a probability in ``[0, 1]``.

    Args:
      method: The criterion under test.
      labelled_scores: The labelled fixture.
    """
    threshold = select_threshold(labelled_scores, method, 'label', 'score')
    assert 0.0 <= threshold <= 1.0

  def test_unknown_criterion_degrades_to_half(self, labelled_scores: pd.DataFrame) -> None:
    """An unrecognised method degrades rather than raising.

    Args:
      labelled_scores: The labelled fixture.
    """
    assert select_threshold(labelled_scores, 'nonsense', 'label', 'score') == 0.5

  def test_mcc_zooms_below_the_coarse_grid_resolution(self) -> None:
    """A tiny optimum below 0.01 must still be found.

    The first grid's resolution cannot express an answer below 0.01, so it reports
    its leftmost point; the progressive zoom is what makes an extreme-imbalance
    operating point selectable at all.
    """
    rng = np.random.default_rng(3)
    negatives = rng.uniform(0.0, 0.0008, 4000)
    positives = rng.uniform(0.0004, 0.0012, 40)
    frame = pd.DataFrame(
      {
        'label': np.r_[np.zeros(4000), np.ones(40)],
        'score': np.r_[negatives, positives],
      }
    )
    threshold = select_threshold(frame, 'mcc_curve', 'label', 'score')
    assert threshold < 0.01, 'the progressive zoom did not engage'
    assert threshold > 0.0

  def test_rounding_is_precision_adaptive(self, labelled_scores: pd.DataFrame) -> None:
    """A sub-1e-4 threshold keeps six decimal places.

    Args:
      labelled_scores: The labelled fixture.
    """
    frame = labelled_scores.copy()
    frame['score'] = frame['score'] * 1e-5
    threshold = select_threshold(frame, 'mcc_curve', 'label', 'score')
    assert round(threshold, 6) == threshold

  def test_missing_column_raises(self, labelled_scores: pd.DataFrame) -> None:
    """A missing column is a configuration issue and must be named.

    Args:
      labelled_scores: The labelled fixture.
    """
    with pytest.raises(ValueError, match='missing column'):
      select_threshold(labelled_scores, 'mcc_curve', 'not_a_column', 'score')


class TestMetricParsing:
  """The metric grammar and its one-sourced vocabulary."""

  def test_unbanded_metric(self) -> None:
    """A bare name is evaluated over the whole population.

    Args:
      None.
    """
    spec = parse_metric('AUC')
    assert (spec.metric, spec.band, spec.is_banded) == ('AUC', 0, False)

  def test_decile_band(self) -> None:
    """A banded name carries its band number.

    Args:
      None.
    """
    spec = parse_metric('lift_decile_10')
    assert (spec.metric, spec.band, spec.band_kind) == ('LIFT', 10, 'DECILE')

  def test_centile_band(self) -> None:
    """A centile band is distinguished from a decile band.

    Args:
      None.
    """
    spec = parse_metric('LIFT_CENTILE_100')
    assert (spec.metric, spec.band, spec.band_kind) == ('LIFT', 100, 'CENTILE')

  def test_absent_metric_uses_the_default(self) -> None:
    """No configuration means the default, which is logged.

    Args:
      None.
    """
    assert parse_metric(None).name == DEFAULT_PROMOTION_METRIC

  def test_unknown_metric_fails_the_promotion(self) -> None:
    """A typo must not silently change the promotion metric.

    A raised and caught a ``ValueError`` in the same block, printing
    the message and continuing on the default -- so a production promotion could
    be decided on a metric the operator did not choose.
    """
    with pytest.raises(MetricConfigurationError):
      parse_metric('LIFT_DECIL_10', strict=True)

  def test_unknown_metric_substitutes_when_not_strict(self) -> None:
    """The lenient mode exists for exploration and is explicit.

    Args:
      None.
    """
    assert parse_metric('NONSENSE', strict=False).name == DEFAULT_PROMOTION_METRIC

  def test_adjusted_r2_is_reachable_and_spelled_once(self) -> None:
    """Three spellings made adjusted R-squared unselectable.

    Args:
      None.
    """
    spec = parse_metric('ADJ_R2', kind='regression')
    assert spec.metric == 'ADJ_R2'
    assert spec.higher_is_better is True
    assert 'ADJ_R2' in METRIC_REGISTRY

  def test_malformed_band_raises(self) -> None:
    """A band qualifier with no number is a typo, not a fallback.

    Args:
      None.
    """
    with pytest.raises(MetricConfigurationError):
      parse_metric('LIFT_DECILE_ten')


class TestAcceptanceGate:
  """the sign was inverted for every loss metric."""

  def test_higher_is_better_accepts_a_small_regression(self) -> None:
    """A non-regression gate, not an improvement gate.

    Retrained models on rolling windows are routinely indistinguishable, so an
    improvement gate would reject near-ties on sampling noise.

    Args:
      None.
    """
    spec = parse_metric('LIFT_DECILE_10')
    accepted, deviation = evaluate_acceptance(0.95, 1.0, spec)
    assert accepted is True
    assert deviation == pytest.approx(-5.0)

  def test_higher_is_better_rejects_a_large_regression(self) -> None:
    """A large regression is rejected.

    Args:
      None.
    """
    spec = parse_metric('LIFT_DECILE_10')
    accepted, _ = evaluate_acceptance(0.80, 1.0, spec)
    assert accepted is False

  def test_higher_is_better_accepts_a_large_improvement(self) -> None:
    """The tolerance is asymmetric: there is no upper bound.

    Args:
      None.
    """
    spec = parse_metric('LIFT_DECILE_10')
    assert evaluate_acceptance(1.5, 1.0, spec)[0] is True

  def test_lower_is_better_uses_a_symmetric_band(self) -> None:
    """A loss metric gates on a symmetric +/-5 % non-regression band.

    The direction only determines which way is better; the *band* is the same
    either way. That is the intended design: retrained models on rolling windows
    are routinely statistically indistinguishable, so a one-sided gate would
    reject near-ties on sampling noise and stall the lifecycle.

    Args:
      None.
    """
    spec = parse_metric('RMSE', kind='regression')
    # A 4% worse model is inside the band and is accepted -- the same as for a
    # higher-is-better metric that is 4% worse.
    assert evaluate_acceptance(104.0, 100.0, spec)[0] is True
    # A 4% better model is inside the band and is accepted.
    assert evaluate_acceptance(96.0, 100.0, spec)[0] is True

  def test_lower_is_better_rejects_beyond_the_band(self) -> None:
    """A regression beyond the tolerance is rejected, whichever side it is on.

    Args:
      None.
    """
    spec = parse_metric('RMSE', kind='regression')
    accepted, deviation = evaluate_acceptance(110.0, 100.0, spec)
    assert deviation == pytest.approx(10.0)
    assert accepted is False

  def test_direction_is_what_differs_between_the_two_kinds(self) -> None:
    """The same numeric deviation is judged opposite ways for the two kinds.

    A 6 % rise is a regression for a loss and an improvement for a score; the
    gate must judge each accordingly. This is the property a lacked
    entirely for regression metrics, because every reachable regression metric
    was a loss.
    """
    score = parse_metric('R2', kind='regression')
    loss = parse_metric('RMSE', kind='regression')
    assert score.higher_is_better is True
    assert loss.higher_is_better is False
    # diff = +6.0
    assert evaluate_acceptance(1.06, 1.0, score)[0] is True
    assert evaluate_acceptance(106.0, 100.0, loss)[0] is False

  def test_zero_incumbent_score_accepts(self) -> None:
    """A zero denominator has no meaningful relative deviation.

    Args:
      None.
    """
    spec = parse_metric('LIFT_DECILE_10')
    assert evaluate_acceptance(1.0, 0.0, spec)[0] is True

  def test_boundary_deviation_is_accepted(self) -> None:
    """A challenger exactly at tolerance is accepted, not rejected by float noise.

    ``0.95`` against ``1.0`` yields ``-5.000000000000004`` in binary floating
    point, so a strict comparison would reject a challenger that has precisely
    met the rule.
    """
    spec = parse_metric('LIFT_DECILE_10')
    accepted, deviation = evaluate_acceptance(0.95, 1.0, spec)
    assert deviation == pytest.approx(-5.0)
    assert accepted is True

  def test_tolerance_is_five_percent(self) -> None:
    """The documented tolerance.

    Args:
      None.
    """
    assert acceptance_threshold() == -5.0


class TestBandMetrics:
  """The band table and the band-selection contract."""

  def test_produces_ten_ordered_bands(self, labelled_scores: pd.DataFrame) -> None:
    """Rank-based banding yields equal-volume, 1-indexed bands.

    Args:
      labelled_scores: The labelled fixture.
    """
    spec = parse_metric('LIFT_DECILE_10')
    bands = band_metrics(labelled_scores, 'label', 'prediction', 'score', spec)
    assert len(bands) == 10
    assert sorted(bands['score_ntile']) == list(range(1, 11))
    assert {'LIFT', 'AUC', 'PRECISION', 'RECALL', 'ACCURACY', 'F1', 'BRIER_SCORE'} <= set(bands.columns)

  def test_top_band_has_the_highest_lift(self, labelled_scores: pd.DataFrame) -> None:
    """A well-separated score puts its lift at the top.

    Args:
      labelled_scores: The labelled fixture.
    """
    spec = parse_metric('LIFT_DECILE_10')
    bands = band_metrics(labelled_scores, 'label', 'prediction', 'score', spec)
    top = bands[bands['score_ntile'] == bands['score_ntile'].max()].iloc[0]
    assert top['LIFT'] > bands['LIFT'].mean()

  def test_empty_band_raises_rather_than_substituting(self, labelled_scores: pd.DataFrame) -> None:
    """Reporting one band's value under another band's metric name.

    Asking for decile 12 of a ten-decile banding must not quietly return decile
    10's lift. That changes a *reported number*, not merely whether a task runs,
    so it must never be silent.

    Args:
      labelled_scores: The labelled fixture.
    """
    spec = parse_metric('LIFT_DECILE_12')
    bands = band_metrics(labelled_scores, 'label', 'prediction', 'score', parse_metric('LIFT_DECILE_10'))
    with pytest.raises(MetricBandEmptyError) as excinfo:
      select_band(bands, spec)
    assert 'LIFT_DECILE_12' in str(excinfo.value)

  def test_existing_band_is_returned_verbatim(self, labelled_scores: pd.DataFrame) -> None:
    """A band that exists is returned as-is, with no substitution.

    Args:
      labelled_scores: The labelled fixture.
    """
    bands = band_metrics(labelled_scores, 'label', 'prediction', 'score', parse_metric('LIFT_DECILE_10'))
    selected = select_band(bands, parse_metric('LIFT_DECILE_7'))
    assert len(selected) == 1
    assert selected.iloc[0]['score_ntile'] == 7

  def test_unbanded_spec_returns_everything(self, labelled_scores: pd.DataFrame) -> None:
    """An unbanded metric flows through the same code path as a banded one.

    Args:
      labelled_scores: The labelled fixture.
    """
    bands = band_metrics(labelled_scores, 'label', 'prediction', 'score', parse_metric('AUC'))
    assert len(select_band(bands, parse_metric('AUC'))) == len(bands)

  def test_duplicate_score_edges_do_not_raise(self) -> None:
    """``duplicates='drop'`` is what keeps a low-cardinality score from crashing.

    Args:
      None.
    """
    frame = pd.DataFrame({'label': [0, 1] * 50, 'prediction': [0, 1] * 50, 'score': [0.5] * 100})
    bands = band_metrics(frame, 'label', 'prediction', 'score', parse_metric('LIFT_DECILE_10'))
    assert len(bands) >= 1

  def test_single_class_band_yields_nan_not_an_exception(self) -> None:
    """A degenerate band is expected from a skewed score, not an error.

    Args:
      None.
    """
    frame = pd.DataFrame({'label': [0] * 90 + [1] * 10, 'prediction': [0] * 100, 'score': [0.1] * 90 + [0.9] * 10})
    bands = band_metrics(frame, 'label', 'prediction', 'score', parse_metric('LIFT_DECILE_10'))
    assert not bands.empty


class TestTemporalGate:
  """The promotion decision is gated on evidence time, not on score."""

  def test_later_evidence_promotes(self) -> None:
    """A strictly later observation window means the challenger is better informed."""
    from datetime import date  # noqa: PLC0415

    assert decide_promotion(date(2026, 10, 1), date(2026, 9, 1)).action == 'PROMOTE'

  def test_equal_evidence_is_a_no_op(self) -> None:
    """Equality is the idempotent-replay guard."""
    from datetime import date  # noqa: PLC0415

    decision = decide_promotion(date(2026, 9, 1), date(2026, 9, 1))
    assert decision.action == 'NO_OP'
    assert 'duplicate promotion' in decision.rationale

  def test_earlier_evidence_is_recorded_only(self) -> None:
    """Stale evidence cannot supersede a better-informed incumbent."""
    from datetime import date  # noqa: PLC0415

    assert decide_promotion(date(2026, 8, 1), date(2026, 9, 1)).action == 'RECORD_ONLY'

  def test_every_decision_carries_a_rationale(self) -> None:
    """The reasoning behind an entry is durable, not confined to a container log.

    Args:
      None.
    """
    from datetime import date  # noqa: PLC0415

    for new, old in ((date(2026, 10, 1), date(2026, 9, 1)), (date(2026, 9, 1), date(2026, 9, 1))):
      assert decide_promotion(new, old).rationale


class TestScoreContract:
  """The score-table column contract."""

  def test_both_score_columns_are_mandatory(self) -> None:
    """The dual-score contract is the framework's published shape.

    Args:
      None.
    """
    from forecasting_ml_framework.modeling.scoring import SCORE_COLUMNS  # noqa: PLC0415

    assert constants.ORIG_SCORE_VALUE in SCORE_COLUMNS
    assert constants.SCORE_VALUE in SCORE_COLUMNS

  def test_banding_is_ten_and_one_hundred(self) -> None:
    """The band counts the downstream drift monitor assumes.

    Args:
      None.
    """
    assert constants.DECILE_COUNT == 10
    assert constants.CENTILE_COUNT == 100
