#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''How a request becomes a validated :class:`BacktestRequest`.

It rejects rather than coerces, and it rejects unknown keys rather than
ignoring them. A silently dropped ``max_weight`` is the specific hazard
this exists to prevent: the run would succeed, report a Sharpe, and
describe a portfolio nobody asked for. A body that says ``"5"`` where a
whole number belongs is a type error, while a query that says
``history=5`` is the ordinary spelling, so the two paths have separate
converters and share everything after the conversion.

Both paths then hand the value to the *same* validators, which is why
``GET /api/baselines?max_weight=0.3`` cannot smuggle in a configuration
the POST body would have refused: the two endpoints cannot disagree about
what a legal configuration is.

Split out of the former single-module ``stock_rl.api`` without change,
including every message. The messages name the field, the offending value
and the accepted range, because a refusal a caller cannot act on is a
refusal they will work around.
'''

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import parse_qs

from stock_rl.api.errors import BadRequest
from stock_rl.api.models import BacktestRequest, _number
from stock_rl.api.runner import strategies


def _parse_request(body: bytes, known: Iterable[str]) -> BacktestRequest:
  '''Validate a JSON body into a :class:`BacktestRequest`.

  Rejects rather than coerces, and rejects unknown keys rather than
  ignoring them. A silently dropped ``max_weight`` is the specific
  hazard: the run would succeed, report a Sharpe, and describe a
  portfolio nobody asked for.

  Args:
    body: Raw request body.
    known: Symbols the service has loaded.

  Returns:
    The validated request.

  Raises:
    BadRequest: If the body is not a JSON object, carries unknown keys,
      or carries a value outside its permitted range.
  '''
  text = body.decode('utf-8', errors='replace').strip()
  if not text:
    raise BadRequest('request body is empty; POST a JSON object')
  try:
    payload = json.loads(text)
  except json.JSONDecodeError as exc:
    raise BadRequest(f'body is not valid JSON: {exc}') from exc
  if not isinstance(payload, dict):
    raise BadRequest(
      f'body must be a JSON object, got {type(payload).__name__}')
  allowed = {
    'symbols', 'capital', 'strategy', 'rebalance_days', 'max_weight',
    'history',
  }
  unknown = sorted(set(payload) - allowed)
  if unknown:
    raise BadRequest(
      f'unrecognised field(s) {unknown}; accepted: {sorted(allowed)}')
  defaults = BacktestRequest()
  table = strategies()
  strategy = _text(payload.get('strategy', defaults.strategy),
                   'strategy')
  if strategy not in table:
    raise BadRequest(
      f'unknown strategy {strategy!r}; known: {sorted(table)}')
  symbols = _symbol_list(payload.get('symbols'), known)
  return BacktestRequest(
    symbols=symbols,
    capital=_positive_number(payload.get('capital', defaults.capital),
                            'capital'),
    strategy=strategy,
    rebalance_days=_count(payload.get('rebalance_days',
                                      defaults.rebalance_days),
                          'rebalance_days'),
    max_weight=_capped_number(payload.get('max_weight',
                                          defaults.max_weight),
                              'max_weight'),
    history=_count(payload.get('history', defaults.history), 'history'),
  )


def _symbol_list(value: Any, known: Iterable[str]) -> tuple[str, ...]:
  '''Validate a ``symbols`` field.

  Args:
    value: Candidate list of symbols.
    known: Symbols the service has loaded.

  Returns:
    Symbols in ascending order, deduplicated.

  Raises:
    BadRequest: If the field is not a list of non-empty strings, or
      names a symbol that is not loaded. An unknown symbol is a 400 and
      not a silent omission, because a backtest over fewer names than
      were requested is not the backtest that was asked for.
  '''
  if value is None:
    return ()
  if not isinstance(value, list):
    raise BadRequest(f'symbols must be a list, got {type(value).__name__}')
  loaded = set(known)
  chosen: set[str] = set()
  for item in value:
    if not isinstance(item, str) or not item.strip():
      raise BadRequest(f'symbols entries must be non-empty strings, '
                       f'got {item!r}')
    symbol = item.strip()
    if symbol not in loaded:
      raise BadRequest(
        f'symbol {symbol!r} is not loaded; loaded: {sorted(loaded)}')
    chosen.add(symbol)
  return tuple(sorted(chosen))


def _text(value: Any, name: str) -> str:
  '''Return ``value`` as a non-empty string.

  Args:
    value: Candidate string.
    name: Field name, for the error message.

  Returns:
    The stripped string.

  Raises:
    BadRequest: If the value is not a non-empty string.
  '''
  if not isinstance(value, str) or not value.strip():
    raise BadRequest(f'{name} must be a non-empty string, got {value!r}')
  return value.strip()


def _positive_number(value: Any, name: str) -> float:
  '''Return ``value`` as a finite positive float.

  Args:
    value: Candidate number.
    name: Field name, for the error message.

  Returns:
    The float.

  Raises:
    BadRequest: If the value is not a finite number greater than zero.
      Booleans are rejected explicitly: ``True`` is an int in Python and
      would otherwise be accepted as capital of 1 rupee.
  '''
  number = _number(value, float('nan'))
  if isinstance(value, bool) or not math.isfinite(number) or number <= 0.0:
    raise BadRequest(f'{name} must be a positive finite number, '
                     f'got {value!r}')
  return number


def _capped_number(value: Any, name: str) -> float:
  '''Return ``value`` as a finite float inside ``(0, 1]``.

  Args:
    value: Candidate number.
    name: Field name, for the error message.

  Returns:
    The float.

  Raises:
    BadRequest: If the value is outside ``(0, 1]``.
  '''
  number = _number(value, float('nan'))
  if isinstance(value, bool) or not math.isfinite(number) \
      or not 0.0 < number <= 1.0:
    raise BadRequest(f'{name} must be in (0, 1], got {value!r}')
  return number


def _count(value: Any, name: str) -> int:
  '''Return ``value`` as an integer of at least one.

  Args:
    value: Candidate integer.
    name: Field name, for the error message.

  Returns:
    The integer.

  Raises:
    BadRequest: If the value is not a whole number of at least one.
  '''
  if isinstance(value, bool) or not isinstance(value, int):
    raise BadRequest(f'{name} must be a whole number, got {value!r}')
  if value < 1:
    raise BadRequest(f'{name} must be >= 1, got {value}')
  return value


#: Query parameters ``GET /api/baselines`` accepts. The same four the
#: trigger accepts, and validated by the same functions, so the two
#: endpoints cannot disagree about what a legal configuration is.
_control_fields = frozenset(
  {'capital', 'rebalance_days', 'max_weight', 'history'})


def _control_of(path: str) -> BacktestRequest | None:
  '''Return the control-arm configuration a query string asks for.

  The same validators as :func:`_parse_request` are used, so a query
  cannot smuggle in a value the POST body would have refused, and the two
  endpoints cannot disagree about what a legal configuration is.

  Args:
    path: Raw request path, possibly carrying a query string.

  Returns:
    The requested configuration, or None when the query carries nothing
    and the declared :data:`control_arm` applies.

  Raises:
    BadRequest: If a parameter is unknown or outside its permitted range.
  '''
  query = path.split('?', 1)[1].split('#', 1)[0] if '?' in path else ''
  fields = parse_qs(query, keep_blank_values=True)
  unknown = sorted(set(fields) - _control_fields)
  if unknown:
    raise BadRequest(
      f'unrecognised query parameter(s) {unknown}; accepted: '
      f'{sorted(_control_fields)}')
  if not fields:
    return None
  defaults = BacktestRequest()
  return BacktestRequest(
    symbols=defaults.symbols,
    capital=_positive_number(
      _decimal(_one(fields, 'capital')) or defaults.capital, 'capital'),
    strategy=defaults.strategy,
    rebalance_days=_count(
      _whole(_one(fields, 'rebalance_days'), 'rebalance_days')
      or defaults.rebalance_days,
      'rebalance_days'),
    max_weight=_capped_number(
      _decimal(_one(fields, 'max_weight')) or defaults.max_weight,
      'max_weight'),
    history=_count(_whole(_one(fields, 'history'), 'history')
                   or defaults.history, 'history'),
  )


def _whole(value: str | None, name: str) -> int | None:
  '''Return a query parameter as an int, or None when absent.

  A query string carries text, so a count arrives as ``'5'``. It is
  converted here rather than by :func:`_count`, which is written for the
  JSON body where a whole number must already *be* a number: a body that
  says ``"5"`` is a type error, while a query that says ``history=5`` is
  the ordinary spelling.

  Args:
    value: The raw parameter value.
    name: Parameter name, for the error message.

  Returns:
    The integer, or None when the parameter was absent.

  Raises:
    BadRequest: If the value is not a plain run of ASCII digits.
  '''
  if value is None:
    return None
  text = value.strip()
  if not text.isascii() or not text.isdigit():
    raise BadRequest(
      f'{name} must be a whole number in the query, got {value!r}')
  return int(text)


def _decimal(value: str | None) -> float | None:
  '''Return a query parameter as a float, or None when absent.

  Args:
    value: The raw parameter value.

  Returns:
    The float, or None when the parameter was absent.

  Raises:
    BadRequest: If the value will not parse as a number at all. The
      range is not checked here; the validator that receives the value
      owns that, so there is one range rule rather than two.
  '''
  if value is None:
    return None
  try:
    return float(value.strip())
  except ValueError as exc:
    raise BadRequest(f'not a number: {value!r}') from exc


def _one(
  fields: Mapping[str, list[str]],
  name: str,
) -> str | None:
  '''Return one query value, refusing a repeated one.

  Args:
    fields: Parsed query fields.
    name: Parameter to read.

  Returns:
    The single value, or None when absent.

  Raises:
    BadRequest: If the parameter was given more than once. Silently
      picking one of two is how a proxy and a server end up disagreeing
      about what was asked for.
  '''
  values = fields.get(name) or []
  if len(values) > 1:
    raise BadRequest(f'{name} was given {len(values)} times; give it once')
  return values[0] if values else None
