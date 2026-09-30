#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Regression tests for defects found by direct execution against the framework.

Every test here pins a behaviour that was previously wrong, and each is small
enough to read in full. They are grouped by the defect they cover rather than by
the module that contains the fix, because what makes them worth keeping is the
property -- "an uninformative model reports lift ~1.0" -- and not the location.
'''

import sqlite3

import numpy as np
import pandas as pd
import pytest

from forecasting_ml_framework import constants
from forecasting_ml_framework.local.sql_translate import SqlTranslationError, to_sqlite
from forecasting_ml_framework.modeling.calibration import MCC_ZOOM_GRIDS, select_threshold
from forecasting_ml_framework.modeling.metrics import band_metrics, parse_metric


class TestLiftIsLift:
  '''LIFT must be a ratio against the base rate, not a share of positives.'''

  def test_an_uninformative_model_reports_lift_near_one(self) -> None:
    generator = np.random.default_rng(0)
    size = 20000
    score = generator.random(size)
    frame = pd.DataFrame(
      {'y': generator.random(size) < 0.5, 'p': score, 'pred': (score >= 0.5).astype(int)}
    )
    bands = band_metrics(frame, 'y', 'pred', 'p', parse_metric('LIFT_DECILE_10'))
    # The defect multiplied the band's share of positives by the band count, so a
    # coin flip reported 10.0 in every band instead of 1.0.
    assert bands['LIFT'].mean() == pytest.approx(1.0, abs=0.05)
    assert bands['LIFT'].max() < 1.5

  def test_a_separating_model_lifts_in_the_top_band(self) -> None:
    generator = np.random.default_rng(0)
    size = 20000
    score = generator.random(size)
    frame = pd.DataFrame({'y': score > 0.9, 'p': score, 'pred': (score >= 0.5).astype(int)})
    bands = band_metrics(frame, 'y', 'pred', 'p', parse_metric('LIFT_DECILE_10'))
    top = bands.loc[bands['score_ntile'] == bands['score_ntile'].max()].iloc[0]
    bottom = bands.loc[bands['score_ntile'] == bands['score_ntile'].min()].iloc[0]
    assert top['LIFT'] > 5.0
    assert bottom['LIFT'] < 1.0

  def test_overall_rate_is_reported(self) -> None:
    frame = pd.DataFrame(
      {
        'y': [1] * 50 + [0] * 50,
        'p': np.arange(100) / 100,
        'pred': [1] * 50 + [0] * 50,
      }
    )
    bands = band_metrics(frame, 'y', 'pred', 'p', parse_metric('LIFT_DECILE_10'))
    assert bands['OVERALL_RATE'].iloc[0] == pytest.approx(0.5)


class TestMccZoomEngages:
  '''The progressive zoom must actually reach its finer grids.'''

  def test_grid_lengths_differ_only_by_floating_point(self) -> None:
    # The defect terminated the zoom on the first pass because the terminal test
    # compared grid LENGTHS, and the grids are all ~100 points long, so it reported
    # "last grid" immediately and grids two and three were unreachable. The zoom is
    # now driven by position, so the lengths no longer carry the meaning -- but they
    # must still be near-identical, or the escalation is not the uniform one it
    # claims to be.
    assert len(MCC_ZOOM_GRIDS) == 3
    assert max(len(grid) for grid in MCC_ZOOM_GRIDS) - min(len(grid) for grid in MCC_ZOOM_GRIDS) <= 1

  def test_grids_escalate_in_resolution(self) -> None:
    steps = [float(grid[1] - grid[0]) for grid in MCC_ZOOM_GRIDS]
    assert steps == sorted(steps, reverse=True)
    assert steps[0] > steps[-1]

  def test_the_finest_grid_reaches_a_sub_percent_threshold(self) -> None:
    # Grids two and three exist to express an answer below 0.01. If the finest
    # grid cannot represent one, the zoom cannot do the job it was built for.
    assert float(MCC_ZOOM_GRIDS[-1][-1]) < 0.001

  def test_a_sub_percent_positive_rate_yields_a_usable_threshold(self) -> None:
    generator = np.random.default_rng(7)
    size = 20000
    score = generator.random(size)
    positives = (score > 0.995).astype(int)
    frame = pd.DataFrame({'y': positives, 'p': score})
    threshold = select_threshold(frame, 'mcc_curve', 'y', 'p')
    # Without the zoom the coarse grid's resolution cannot express an answer below
    # 0.01 and reports 0.0, which predicts nobody positive.
    assert threshold > 0.0


class TestMetricGrammar:
  '''A multi-word metric must be bandable.'''

  @pytest.mark.parametrize(
    ('name', 'expected'),
    [
      ('ADJ_R2_DECILE_10', 'ADJ_R2'),
      ('ADJ_R2_CENTILE_5', 'ADJ_R2'),
      ('BRIER_SCORE_DECILE_2', 'BRIER_SCORE'),
      ('R2_DECILE_10', 'R2'),
      ('LIFT_DECILE_10', 'LIFT'),
    ],
  )
  def test_base_name_is_decomposed(self, name: str, expected: str) -> None:
    spec = parse_metric(name, kind='regression' if 'R2' in name else 'classification')
    assert spec.metric == expected

  def test_an_unbanded_multi_word_metric_still_parses(self) -> None:
    assert parse_metric('ADJ_R2', kind='regression').metric == 'ADJ_R2'


class TestNestedDateAdd:
  '''Nested DATE_ADD must translate into runnable SQLite.

  The defect matched the outer call first and read the interval out of the inner
  one, leaving a half-translated statement SQLite could not parse -- reached by
  silently mistranslating the query, which is what the translator exists to
  prevent.
  '''

  @staticmethod
  def _evaluate(sql: str) -> list:
    """Execute a translated statement against a small in-memory table.

    Args:
      sql: The translated statement.

    Returns:
      The rows it returned.
    """
    connection = sqlite3.connect(':memory:')
    connection.execute('CREATE TABLE t (rpt_dt TEXT, a INTEGER)')
    connection.execute("INSERT INTO t VALUES ('2024-01-01', 5)")
    try:
      return connection.execute(sql.replace('FROM t', "FROM t WHERE rpt_dt='2024-01-01'")).fetchall()
    finally:
      connection.close()

  def test_a_single_date_add(self) -> None:
    assert self._evaluate(to_sqlite('SELECT DATE_ADD(rpt_dt, INTERVAL 58 DAY) FROM t'))[0][0] == '2024-02-28'

  def test_a_nested_date_add_adds_both_intervals(self) -> None:
    translated = to_sqlite('SELECT DATE_ADD(DATE_ADD(rpt_dt, INTERVAL 30 DAY), INTERVAL 58 DAY) FROM t')
    assert 'DATE_ADD' not in translated
    assert self._evaluate(translated)[0][0] == '2024-03-29'

  def test_three_levels_of_nesting(self) -> None:
    translated = to_sqlite(
      'SELECT DATE_ADD(DATE_ADD(DATE_ADD(rpt_dt, INTERVAL 1 DAY), INTERVAL 2 WEEK), INTERVAL 1 QUARTER) FROM t'
    )
    assert translated.count('DATE_ADD') == 0
    assert self._evaluate(translated)[0][0] == '2024-04-16'

  def test_an_interval_of_zero_collapses_to_a_plain_date(self) -> None:
    assert self._evaluate(to_sqlite('SELECT DATE_ADD(rpt_dt, INTERVAL 0 DAY) FROM t'))[0][0] == '2024-01-01'

  def test_an_unknown_unit_is_refused_rather_than_guessed(self) -> None:
    with pytest.raises(SqlTranslationError):
      to_sqlite('SELECT DATE_ADD(a, INTERVAL 5 FORTNIGHT) FROM t')

  def test_a_malformed_call_is_refused(self) -> None:
    with pytest.raises(SqlTranslationError):
      to_sqlite('SELECT DATE_ADD(a, 5) FROM t')


class TestArtefactAddressingIsSchemeAgnostic:
  '''A local artefact address carries no scheme.'''

  def test_the_local_root_is_a_bare_path(self) -> None:
    assert '://' not in constants.LOCAL_ARTEFACT_ROOT

  def test_the_local_root_is_under_a_temporary_directory(self) -> None:
    assert constants.LOCAL_ARTEFACT_ROOT.startswith('/tmp/')
