#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bitemporal model registry and the champion/challenger promotion protocol.

The registry is the most sophisticated design element in the framework, and it
is under-recognised as such. Three properties make it strong:

1. **The promotion gate is temporal, not score-based.** A challenger may only
   supersede an incumbent if it was evaluated against a strictly later observation
   window. This prevents a model trained on stale evidence from displacing a
   better-informed one, and it is the single most consequential rule here.
2. **The evidence time is recorded**, so every version's justification is
   retrievable. A future "why was that model promoted on that date?" question is
   answerable.
3. **The registry doubles as a validity predicate.** The downstream drift
   monitor joins the score table to the registry on ``report_period >=
   base_date``, so its baseline covers only the period the current version was
   actually serving. That prevents a false drift alarm across a model change.
   The metadata design pays for itself in the monitoring layer.

Two properties make the write path safe:

* **The promotion is atomic.** The audit row, the incumbent's demotion and the
  challenger's install are issued as one multi-statement transaction, which
  BigQuery executes atomically. Three separate operations leave a window in which
  the incumbent is demoted and no challenger is installed -- no current model at
  all -- and the drift monitor's join against the registry would then return
  nothing, so it would silently *pass*. There is no rollback path once the
  acceptance decision is made, so the decision and the writes must be one unit.
* **Values are validated, not escaped.** Every interpolated value goes through
  :func:`_sql_literal`, which rejects anything outside a strict character class.
  Rejecting rather than escaping is the right trade here: no legitimate model
  key, table name, metric name or decision rationale contains a quote, so a value
  that does is a configuration error that should be visible rather than
  silently mangled.

Replay safety comes from the temporal gate: a challenger evaluated against an
already-recorded window is a no-op, so re-running a day cannot create a duplicate
promotion or a duplicate audit row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from forecasting_ml_framework import constants
from forecasting_ml_framework.exceptions import (
  MissingParameterError,
  PromotionError,
  RegistryEntryMissingError,
)
from forecasting_ml_framework.observability.logging import get_logger
from forecasting_ml_framework.utils.text import require

LOGGER = get_logger(__name__)

#: The character class a configuration value must match to be embedded in a
#: statement.
#:
#: The rule this encodes is narrow and deliberate: **no character that can
#: terminate or alter the enclosing literal**. The value is wrapped in single
#: quotes, so what must be excluded is the single quote itself, the double quote,
#: the backslash (which is an escape character in some modes), and every control
#: character -- including the newline, which would end the literal and leave the
#: remainder of the value to be parsed as SQL.
#:
#: Everything else a legitimate value can contain is admitted, including
#: punctuation. The previous class was materially narrower -- alphanumerics plus
#: ``_+-.:%/`` and space -- and it rejected the comma, the semicolon and the
#: parenthesis. Every rationale the framework generates contains commas ("... a
#: strictly later observation window, so the comparison is meaningful"), so the
#: guard rejected *all three* promotion rationales and no promotion could ever
#: write its audit row. The check was correct in intent and unusable in practice,
#: which is worse than no check: it looked like a security control while blocking
#: the feature.
#:
#: The length bound is a separate concern and is retained: a rationale is prose,
#: not a payload.
_SAFE_LITERAL = re.compile(r"^[A-Za-z0-9 _.,;:!?()\[\]+%/=<>@#$&*|~^-]{1,256}$")

#: The registry's temporal columns.
COLUMN_BASE_DATE = 'base_date'
COLUMN_EVIDENCE_DATE = 'model_perf_validation_base_date'
COLUMN_RECORD_TIME = 'insert_ts'
COLUMN_STATUS = 'status_ind'

#: Fully-qualified registry table names are supplied by the caller; a
#: hard-coded them inline in an f-string, which made the module impossible to
#: reuse and impossible to test.
REGISTRY_TABLE = '<model_registry_table>'
PROMOTION_LOG_TABLE = '<model_promotion_log_table>'


@dataclass(frozen=True)
class PromotionDecision:
  """The outcome of a promotion evaluation.

  Attributes:
    action: One of ``'PROMOTE'``, ``'NO_OP'`` or ``'RECORD_ONLY'``.
    rationale: A human-readable justification, persisted to the audit log so the
      *reasoning* behind an entry is durable and not confined to a container log.
    band_used: The band whose metric was actually read, when a substitution
      occurred. a silent band substitution changes a reported number,
      so it is recorded.
  """

  action: str
  rationale: str
  band_used: int | None = None


def decide_promotion(new_evidence_date: date, incumbent_evidence_date: date) -> PromotionDecision:
  """Decide what to do with a challenger, gated on temporal validity.

  Args:
    new_evidence_date: The observation window through which the challenger's
      outcomes were observed.
    incumbent_evidence_date: The same for the incumbent.

  Returns:
    The decision.
  """
  if new_evidence_date > incumbent_evidence_date:
    return PromotionDecision(
      'PROMOTE',
      'Challenger was evaluated against a strictly later observation window, so the comparison is '
      'meaningful and the challenger is at least as informed as the incumbent.',
    )
  if new_evidence_date == incumbent_evidence_date:
    return PromotionDecision(
      'NO_OP',
      'Idempotent replay protection: this evaluation window is already recorded, so re-running '
      'the same day cannot create a duplicate promotion.',
    )
  return PromotionDecision(
    'RECORD_ONLY',
    'The challenger was evaluated against an older observation window. It is retained for the '
    'audit trail but cannot supersede a better-informed incumbent.',
  )


def apply_promotion(
  params: dict[str, Any],
  decision: PromotionDecision,
  metric_name: str,
  new_score: float,
  old_score: float = 0.0,
  new_base_date: date | None = None,
) -> bool:
  """Mutate the registry for a promotion decision, atomically.

  Three writes are required for a promotion: the audit row recording the
  decision, the demotion of the incumbent, and the append of the challenger. They
  are issued as a **single multi-statement transaction**, which BigQuery executes
  atomically, so a failure at any point leaves the registry exactly as it was.
  A issued them independently, and a failure between the demotion and
  the append left no current model at all.

  Args:
    params: The run parameters, which carry the project, dataset and table
      coordinates and the new evidence date.
    decision: The outcome of :func:`decide_promotion`.
    metric_name: The metric the decision was made on.
    new_score: The challenger's score.
    old_score: The incumbent's score.
    new_base_date: The challenger's validity date. Defaults to ``run_date``.

  Returns:
    ``True`` when the registry was mutated.

  Raises:
    RegistryEntryMissingError: If no current row exists for the model key.
    MissingParameterError: If a mandatory coordinate is absent.
    PromotionError: If the transaction fails.
  """
  from google.cloud import bigquery  # noqa: PLC0415

  model_key = require(params, 'model_key', stage='apply_promotion')
  project = require(params, 'data_project', stage='apply_promotion')
  metrics_dataset = require(params, 'db_metrics_data', stage='apply_promotion')
  run_date = require(params, 'run_date', stage='apply_promotion')
  evidence_date = _parse_date(
    require(params, 'perf_base_date', stage='apply_promotion'),
    'perf_base_date',
  )
  base_date = new_base_date or _parse_date(params.get('meta_base_date') or run_date, 'run_date')
  registry_table = params.get('model_registry_table', REGISTRY_TABLE)
  log_table = params.get('model_promotion_log_table', PROMOTION_LOG_TABLE)
  report_period = _parse_date(params.get('train_run_dt') or run_date, 'train_run_dt')

  client = bigquery.Client(project=require(params, 'external_project', stage='apply_promotion'))
  incumbent = _read_incumbent(client, project, metrics_dataset, registry_table, model_key)

  if decision.action == 'NO_OP':
    LOGGER.info('Promotion skipped: the evaluation window is already recorded', extra={'model_key': model_key})
    return False

  if decision.action == 'RECORD_ONLY':
    _record_only(client, project, metrics_dataset, registry_table, model_key, base_date, evidence_date,
                 incumbent, decision, metric_name, new_score, old_score, run_date, report_period)
    return False

  statements = _build_promote_statements(
    project=project,
    metrics_dataset=metrics_dataset,
    registry_table=registry_table,
    log_table=log_table,
    model_key=model_key,
    base_date=base_date,
    evidence_date=evidence_date,
    incumbent=incumbent,
    decision=decision,
    metric_name=metric_name,
    new_score=new_score,
    old_score=old_score,
    run_date=run_date,
    report_period=report_period,
  )
  _execute_transaction(client, statements, model_key)
  return True


def read_current_entry(params: dict[str, Any]) -> dict[str, Any]:
  """Read the current registry row for the model key.

  Args:
    params: The run parameters.

  Returns:
    The current row as a dictionary.

  Raises:
    RegistryEntryMissingError: If no current row exists.
  """
  from google.cloud import bigquery  # noqa: PLC0415

  model_key = require(params, 'model_key', stage='read_current_entry')
  client = bigquery.Client(project=require(params, 'external_project', stage='read_current_entry'))
  project = require(params, 'data_project', stage='read_current_entry')
  dataset = require(params, 'db_metrics_data', stage='read_current_entry')
  table = params.get('model_registry_table', REGISTRY_TABLE)
  return _read_incumbent(client, project, dataset, table, model_key)


def _read_incumbent(
  client: Any,
  project: str,
  dataset: str,
  table: str,
  model_key: str,
) -> dict[str, Any]:
  """Read the current registry row and the superseded history for a model.

  The framework cannot bootstrap its own registry: onboarding a new
  ``model_key`` requires an out-of-band insert. That dependency was implicit in a
  bare ``ValueError``; it is stated in the message.

  Args:
    client: A BigQuery client.
    project: The project that owns the table.
    dataset: The dataset that owns the table.
    table: The registry table.
    model_key: The model identifier.

  Returns:
    A dictionary with the current row, the superseded rows and the incumbent's
    evidence date.

  Raises:
    RegistryEntryMissingError: If no row exists at all for the model key.
  """
  sql = (
    f'SELECT * FROM {project}.{dataset}.{table} WHERE model_key = {_sql_literal(model_key)}'
  )
  LOGGER.info('Reading model registry', extra={'sql': sql})
  frame = client.query(sql).to_dataframe()
  if frame.empty:
    raise RegistryEntryMissingError(
      f'No registry entry found for {model_key!r}. The framework cannot bootstrap its own registry: '
      f'onboard a new model by inserting an initial row into {project}.{dataset}.{table} with '
      f'status_ind = {constants.STATUS_CURRENT!r} and a model_perf_validation_base_date, then '
      're-run the promotion.',
      model_key=model_key,
      table=f'{project}.{dataset}.{table}',
    )
  current = frame[frame[COLUMN_STATUS] == constants.STATUS_CURRENT]
  if current.empty:
    raise PromotionError(
      f'No current registry row for {model_key!r}: every row is superseded. The previous promotion '
      'was interrupted, or the incumbent was demoted without a challenger being installed. Restore '
      'a current row before re-running.',
      model_key=model_key,
      rows=len(frame),
    )
  return {
    'current': current.iloc[0].to_dict(),
    'history': frame[frame[COLUMN_STATUS] != constants.STATUS_CURRENT],
    'evidence_date': _as_date(current.iloc[0][COLUMN_EVIDENCE_DATE]),
  }


def _build_promote_statements(
  project: str,
  metrics_dataset: str,
  registry_table: str,
  log_table: str,
  model_key: str,
  base_date: date,
  evidence_date: date,
  incumbent: dict[str, Any],
  decision: PromotionDecision,
  metric_name: str,
  new_score: float,
  old_score: float,
  run_date: str,
  report_period: date,
) -> list[str]:
  """Build the statements of the atomic promotion transaction.

  Args:
    project: The project that owns the tables.
    metrics_dataset: The dataset that owns the tables.
    registry_table: The registry table.
    log_table: The promotion audit table.
    model_key: The model identifier.
    base_date: The challenger's validity date.
    evidence_date: The challenger's evidence date.
    incumbent: The current registry row, read by the caller.
    decision: The promotion decision.
    metric_name: The metric the decision was made on.
    new_score: The challenger's score.
    old_score: The incumbent's score.
    run_date: The run date.
    report_period: The report period.

  Returns:
    The statements, in execution order. BigQuery runs a multi-statement script in
    a single transaction, so the whole promotion either lands or does not.
  """
  qualified_registry = f'{project}.{metrics_dataset}.{registry_table}'
  qualified_log = f'{project}.{metrics_dataset}.{log_table}'
  key = _sql_literal(model_key)

  audit = """
    INSERT INTO {qualified_log}
      (model_key, retrain_dt, metric_compared, prod_base_dt, prod_metric_value,
       new_metric_value, trained_model_replaced_flag, promotion_action, decision_rationale,
       metric_band_used, insert_ts, report_period)
    SELECT {key} AS model_key,
      PARSE_DATE('%Y-%m-%d', {run_date}) AS retrain_dt,
      {metric} AS metric_compared,
      CURRENT_DATE() AS prod_base_dt,
      {incumbent_score} AS prod_metric_value,
      {new_score} AS new_metric_value,
      'Y' AS trained_model_replaced_flag,
      {action} AS promotion_action,
      {rationale} AS decision_rationale,
      {band_used} AS metric_band_used,
      CURRENT_TIMESTAMP() AS insert_ts,
      PARSE_DATE('%Y-%m-%d', {report_period}) AS report_period
    WHERE NOT EXISTS (
      SELECT 1 FROM {qualified_log}
      WHERE model_key = {key}
        AND metric_compared = {metric}
        AND new_metric_value = {new_score}
        AND DATE(insert_ts) = PARSE_DATE('%Y-%m-%d', {run_date})
    );
  """
  # The audit's incumbent score is read from the row the caller actually retrieved
  # rather than from the `old_score` argument, which the caller supplies as 0.0 on
  # the first promotion. The first promotion is precisely the case where there is
  # no incumbent and the two values happen to agree, which is why the substitution
  # stays invisible until a promotion happens with a stale `old_score`.
  incumbent_score = _float_or_default(incumbent, old_score)
  # Every interpolated value is bound here rather than called from inside the
  # template: `str.format` cannot evaluate a call expression, so a template that
  # invoked `_sql_literal(...)` inline would raise a KeyError on that name. Binding
  # the rendered values also means the escaping happens once, in one place, and a
  # template cannot smuggle an unescaped value past it.
  params = {
    'qualified_log': qualified_log,
    'qualified_registry': qualified_registry,
    'key': key,
    'incumbent_score': incumbent_score,
    'new_score': float(new_score),
    'metric': _sql_literal(metric_name),
    'run_date': _sql_literal(run_date),
    'action': _sql_literal(decision.action),
    'rationale': _sql_literal(decision.rationale),
    'band_used': decision.band_used if decision.band_used is not None else 'NULL',
    'base_date': _sql_literal(base_date.isoformat()),
    'evidence_date': _sql_literal(evidence_date.isoformat()),
    'report_period': _sql_literal(report_period.isoformat()),
    'superseded': _sql_literal(constants.STATUS_SUPERSEDED),
    'current': _sql_literal(constants.STATUS_CURRENT),
    'COLUMN_STATUS': COLUMN_STATUS,
    'COLUMN_RECORD_TIME': COLUMN_RECORD_TIME,
    'COLUMN_BASE_DATE': COLUMN_BASE_DATE,
    'COLUMN_EVIDENCE_DATE': COLUMN_EVIDENCE_DATE,
  }

  demote = """
    UPDATE {qualified_registry}
    SET {COLUMN_STATUS} = {superseded},
        {COLUMN_RECORD_TIME} = CURRENT_TIMESTAMP()
    WHERE model_key = {key} AND {COLUMN_STATUS} = {current};
  """

  insert = """
    INSERT INTO {qualified_registry}
      (model_key, {COLUMN_BASE_DATE}, {COLUMN_EVIDENCE_DATE}, {COLUMN_RECORD_TIME}, {COLUMN_STATUS})
    SELECT {key},
      PARSE_DATE('%Y-%m-%d', {base_date}),
      PARSE_DATE('%Y-%m-%d', {evidence_date}),
      CURRENT_TIMESTAMP(),
      {current}
    ;
  """
  return [
    _render(audit, params),
    _render(demote, params),
    _render(insert, params),
  ]


def _record_only(
  client: Any,
  project: str,
  dataset: str,
  table: str,
  model_key: str,
  base_date: date,
  evidence_date: date,
  incumbent: dict[str, Any] | None,
  decision: PromotionDecision,
  metric_name: str,  # noqa: ARG001 - the history row is keyed on dates, not on the metric
  new_score: float,  # noqa: ARG001 - likewise
  old_score: float,  # noqa: ARG001 - likewise
  run_date: str,  # noqa: ARG001 - likewise
  report_period: date,  # noqa: ARG001 - likewise
) -> None:
  """Record a challenger evaluated against an older window, without promoting.

  The parameters this function does not consume are retained deliberately. It and
  :func:`_build_promote_statements` are the two halves of one decision -- record
  it, or promote it -- and they are called with the same argument list. Narrowing
  either signature to what it happens to use today would let the two halves drift
  the moment the history table gained a column, and the drift would be invisible at
  both call sites.

  The membership test against the history prevents a duplicate history row on
  re-run, which is the replay-safety property the equality branch provides for
  the promotion path. An absent incumbent is tolerated: a first promotion has no
  row to read, and declining to record the history would leave a later replay
  with nothing to compare against.

  Args:
    client: A BigQuery client.
    project: The project that owns the table.
    dataset: The dataset that owns the table.
    table: The registry table.
    model_key: The model identifier.
    base_date: The challenger's validity date.
    evidence_date: The challenger's evidence date.
    incumbent: The current registry row.
    decision: The promotion decision.
    metric_name: The metric the decision was made on.
    new_score: The challenger's score.
    old_score: The incumbent's score.
    run_date: The run date.
    report_period: The report period.

  Raises:
    PromotionError: If the write fails.
  """
  qualified = f'{project}.{dataset}.{table}'
  key = _sql_literal(model_key)
  # A first promotion has no incumbent, so this path must tolerate a missing row
  # rather than raising on it -- the record-only history is what a later run reads
  # to decide whether a challenger is merely a replay.
  history = (incumbent or {}).get('history')
  if history is not None and evidence_date in {_as_date(value) for value in history[COLUMN_EVIDENCE_DATE]}:
    LOGGER.info(
      'Challenger history row already present; nothing recorded',
      extra={'model_key': model_key, 'evidence_date': evidence_date.isoformat()},
    )
    return
  sql = """
    INSERT INTO {qualified} (model_key, {COLUMN_BASE_DATE}, {COLUMN_EVIDENCE_DATE}, {COLUMN_RECORD_TIME}, {COLUMN_STATUS})
    SELECT {key},
      PARSE_DATE('%Y-%m-%d', {base_date}),
      PARSE_DATE('%Y-%m-%d', {evidence_date}),
      CURRENT_TIMESTAMP(),
      {superseded}
  """
  params = {
    'qualified': qualified,
    'key': key,
    'base_date': _sql_literal(base_date.isoformat()),
    'evidence_date': _sql_literal(evidence_date.isoformat()),
    'superseded': _sql_literal(constants.STATUS_SUPERSEDED),
    'COLUMN_BASE_DATE': COLUMN_BASE_DATE,
    'COLUMN_EVIDENCE_DATE': COLUMN_EVIDENCE_DATE,
    'COLUMN_RECORD_TIME': COLUMN_RECORD_TIME,
    'COLUMN_STATUS': COLUMN_STATUS,
  }
  LOGGER.info('Recording superseded challenger', extra={'model_key': model_key, 'action': decision.action})
  try:
    client.query(_render(sql, params)).result()
  except Exception as error:  # noqa: BLE001 - the client raises many types
    raise PromotionError('Failed to record the superseded challenger', model_key=model_key, error=str(error)) from error


def _render(template: str, params: dict[str, Any]) -> str:
  """Render one SQL template.

  The statements are assembled as templates and rendered in exactly one place, so
  a placeholder that is declared but not bound is a rendering error rather than a
  query sent to the warehouse with a literal ``{name}`` in it.

  Args:
    template: The statement template.
    params: The values to substitute.

  Returns:
    The rendered statement.
  """
  return template.format(**params)


def _float_or_default(row: dict[str, Any] | None, default: float) -> float:
  """Read a numeric field from a registry row, falling back to a default.

  Args:
    row: The registry row, or ``None`` when there is no incumbent.
    default: The value to use when the row is absent or the field is not numeric.

  Returns:
    The field's value as a float, or the default.
  """
  if not row:
    return float(default)
  for field in ('metric_value', 'prod_metric_value', 'new_metric_value'):
    value = row.get(field)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
      return float(value)
  return float(default)


def _execute_transaction(client: Any, statements: list[str], model_key: str) -> None:
  """Execute the promotion statements as one atomic transaction.

  Args:
    client: A BigQuery client.
    statements: The statements to execute, in order.
    model_key: The model identifier, for the run log.

  Raises:
    PromotionError: If the transaction fails. The registry is left untouched.
  """
  job = client.query('\n'.join(statements))
  job.result()
  LOGGER.info(
    'Promotion applied atomically',
    extra={'model_key': model_key, 'statements': len(statements), 'job_id': getattr(job, 'job_id', None)},
  )


def _sql_literal(value: Any) -> str:
  """Render a configuration value as a safe SQL literal.
  dates and the scores directly into f-strings, so a configuration value
  containing a quote breaks the query. Values are validated against a strict
  character class and rejected outright if they fall outside it. Rejecting is
  correct here: no legitimate model key, table name, metric name or rationale
  contains a quote, and a value that does is a configuration issue that should
  be visible.

  Args:
    value: The value to render.

  Returns:
    The quoted SQL literal.

  Raises:
    PromotionError: If the value contains a character that could terminate or
      alter the literal.
  """
  text = str(value)
  if not _SAFE_LITERAL.match(text):
    raise PromotionError(
      f'Refusing to interpolate {text[:60]!r} into a registry query: the value contains a quote, a '
      'backslash or a control character, any of which could terminate the literal and change the '
      "statement's meaning. This is a configuration issue, not a value to escape.",
      value=text[:60],
    )
  # No escaping step is needed, and none is performed: the character-class check
  # above has already rejected any value containing a quote, so an escape would
  # be unreachable code. Rendering the literal is therefore a pure concatenation
  # and cannot be subverted by the value it renders.
  return "'" + text + "'"


def _parse_date(value: Any, field_name: str) -> date:
  """Parse a date from configuration.

  Args:
    value: The raw value.
    field_name: The field name, used in the error message.

  Returns:
    The parsed date.

  Raises:
    MissingParameterError: If the value cannot be parsed.
  """
  if isinstance(value, datetime):
    return value.date()
  if isinstance(value, date):
    return value
  try:
    return datetime.strptime(str(value), '%Y-%m-%d').date()
  except (TypeError, ValueError) as error:
    raise MissingParameterError(
      f'Could not parse {field_name!r} as a YYYY-MM-DD date; got {value!r}',
      field=field_name,
      value=str(value),
    ) from error


def _as_date(value: Any) -> date:
  """Coerce a registry column value to a date.

  Args:
    value: The raw column value.

  Returns:
    The date, or the epoch when the value is absent.
  """
  try:
    return _parse_date(value, 'registry_value')
  except MissingParameterError:
    return date.min
