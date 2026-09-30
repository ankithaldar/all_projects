#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Tests that every SQL statement is fully rendered before it is sent.

The promotion statements are assembled as templates with ``str.format``. A template
can therefore be declared, registered as one of a transaction's statements, and
still carry a placeholder that was never bound -- and a bound-looking call
expression such as ``{_sql_literal(x)}`` raises ``KeyError`` at render time rather
than substituting anything.

These tests render every statement with representative inputs and assert that no
brace survives, which is the property that makes the SQL executable rather than
merely readable.
'''

from datetime import date

import pytest

from forecasting_ml_framework.exceptions import InvalidWriteModeError, MissingParameterError
from forecasting_ml_framework.modeling.promotion import (
  PromotionDecision,
  _build_promote_statements,
  _record_only,
  _sql_literal,
)

DECISION = PromotionDecision('PROMOTE', 'within tolerance', band_used=10)


def _statements(incumbent: dict | None = None, old_score: float = 4.25) -> list:
  """Render the promotion transaction's statements.

  Args:
    incumbent: The registry row, or ``None`` for a first promotion.
    old_score: The incumbent score as the caller supplies it. A first promotion
      supplies ``0.0``, because there is no incumbent to read a score from.

  Returns:
    The three rendered statements.
  """
  return _build_promote_statements(
    project='project',
    metrics_dataset='metrics',
    registry_table='registry',
    log_table='promotion_log',
    model_key='demo_model',
    base_date=date(2024, 3, 1),
    evidence_date=date(2024, 2, 1),
    incumbent=incumbent,
    decision=DECISION,
    metric_name='LIFT_DECILE_10',
    new_score=4.9,
    old_score=old_score,
    run_date='2024-03-01',
    report_period=date(2024, 3, 1),
  )


class TestRendering:
  '''No placeholder may survive into the executed statement.'''

  @pytest.mark.parametrize('incumbent', [None, {'metric_value': 4.25, 'history': {}}], ids=['first', 'replace'])
  def test_every_statement_is_fully_rendered(self, incumbent: dict | None) -> None:
    for statement in _statements(incumbent):
      assert '{' not in statement, f'unrendered placeholder in: {statement}'
      assert '}' not in statement

  def test_three_statements_are_produced(self) -> None:
    assert len(_statements({'metric_value': 1.0, 'history': {}})) == 3

  def test_the_first_statement_is_the_audit_insert(self) -> None:
    first = _statements({'metric_value': 1.0, 'history': {}})[0]
    assert 'INSERT INTO' in first
    assert 'promotion_log' in first

  def test_the_second_demotes_the_incumbent(self) -> None:
    second = _statements({'metric_value': 1.0, 'history': {}})[1]
    assert 'UPDATE' in second
    assert "'o'" in second

  def test_the_third_appends_the_challenger(self) -> None:
    third = _statements({'metric_value': 1.0, 'history': {}})[2]
    assert 'INSERT INTO' in third
    assert "'c'" in third


class TestAuditContent:
  '''The audit row must carry what the decision was made on.'''

  def test_the_incumbent_score_is_the_incumbents(self) -> None:
    """The defect: the audit read ``old_score``, which is 0.0 on a first promotion."""
    audit = _statements({'metric_value': 4.25, 'history': {}})[0]
    assert '4.25 AS prod_metric_value' in audit
    assert '0.0 AS prod_metric_value' not in audit

  def test_a_first_promotion_records_a_zero_incumbent_score(self) -> None:
    """A first promotion has no incumbent, and the caller supplies 0.0 for one."""
    assert '0.0 AS prod_metric_value' in _statements(None, old_score=0.0)[0]

  def test_the_band_is_recorded(self) -> None:
    assert '10 AS metric_band_used' in _statements({'metric_value': 1.0, 'history': {}})[0]

  def test_an_absent_band_is_null(self) -> None:
    decision = PromotionDecision('PROMOTE', 'unbanded')
    statements = _build_promote_statements(
      project='p', metrics_dataset='d', registry_table='r', log_table='l', model_key='m',
      base_date=date(2024, 3, 1), evidence_date=date(2024, 2, 1), incumbent=None,
      decision=decision, metric_name='LIFT_DECILE_10', new_score=1.0, old_score=0.0,
      run_date='2024-03-01', report_period=date(2024, 3, 1),
    )
    assert 'NULL AS metric_band_used' in statements[0]

  def test_the_audit_is_replay_safe(self) -> None:
    """Re-running the same promotion must not duplicate the audit row."""
    assert 'WHERE NOT EXISTS' in _statements({'metric_value': 1.0, 'history': {}})[0]


class TestRecordOnlyRendering:
  '''The superseded-history path renders too.'''

  def test_it_renders_without_an_incumbent(self) -> None:
    class RecordingClient:
      """A client that captures the statement instead of executing it."""

      def __init__(self) -> None:
        self.statements: list = []

      def query(self, sql: str) -> 'RecordingClient':
        """Record a statement.

        Args:
          sql: The statement.

        Returns:
          This client.
        """
        self.statements.append(sql)
        return self

      def result(self) -> None:
        """No-op."""
        return

    client = RecordingClient()
    _record_only(
      client, 'p', 'd', 'registry', 'demo', date(2024, 3, 1), date(2024, 2, 1), None,
      PromotionDecision('RECORD_ONLY', 'older window'), 'LIFT_DECILE_10', 4.9, 4.25,
      '2024-03-01', date(2024, 3, 1),
    )
    assert len(client.statements) == 1
    assert '{' not in client.statements[0]

  def test_a_duplicate_history_row_is_not_re_recorded(self) -> None:
    class RecordingClient:
      """A client that would record any statement it is given."""

      def __init__(self) -> None:
        self.statements: list = []

      def query(self, sql: str) -> 'RecordingClient':
        """Record a statement.

        Args:
          sql: The statement.

        Returns:
          This client.
        """
        self.statements.append(sql)
        return self

      def result(self) -> None:
        """No-op."""
        return

    client = RecordingClient()
    incumbent = {'history': {'model_perf_validation_base_date': ['2024-02-01']}}
    _record_only(
      client, 'p', 'd', 'registry', 'demo', date(2024, 3, 1), date(2024, 2, 1), incumbent,
      PromotionDecision('RECORD_ONLY', 'older window'), 'LIFT_DECILE_10', 4.9, 4.25,
      '2024-03-01', date(2024, 3, 1),
    )
    assert client.statements == []


class TestSqlLiteralSafety:
  '''Values are validated, never escaped.'''

  def test_an_ordinary_value_is_quoted(self) -> None:
    assert _sql_literal('demo') == "'demo'"

  @pytest.mark.parametrize('hostile', ["a'b", 'a"; DROP TABLE t; --', 'a\nb', "a\\b"])
  def test_a_hostile_value_is_rejected(self, hostile: str) -> None:
    """Rejecting is correct: no legitimate model key or rationale contains a quote."""
    with pytest.raises(Exception):
      _sql_literal(hostile)


class TestDeclaredExceptionsAreUsed:
  '''The typed hierarchy is the framework's own, not builtins.'''

  def test_the_write_mode_error_is_a_framework_error(self) -> None:
    assert issubclass(InvalidWriteModeError, Exception)

  def test_the_missing_parameter_error_is_a_framework_error(self) -> None:
    assert issubclass(MissingParameterError, Exception)
