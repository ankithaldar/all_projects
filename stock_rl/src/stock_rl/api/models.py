#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The shapes the endpoints speak, and the coercions that fill them.

:class:`Signal`, :class:`BacktestRequest` and :class:`PortfolioState` are
the three things the API is asked about -- what to do, what to measure,
what is held -- and :data:`control_arm` is the configuration every
backtest measurement is made under by default. Each default is an
absence rather than a zero: an empty portfolio reports no equity curve
and an unscorable symbol reports ``None``, because a degenerate value
here is indistinguishable from a real one, and the whole point of
reporting the book is to say what is *not* in it.

The private helpers at the bottom are the other half of that promise. A
backtest runner is a callable this API does not own, so its result
arrives as an untrusted mapping and is coerced here into the typed fields
:class:`PortfolioState` holds; anything unusable becomes no state rather
than a partially-populated series. The two datetime helpers are the same
idea applied to timestamps: one place reads the wall clock, one place
prints a bar's date, and a test can replace the former.

Split out of the former single-module ``stock_rl.api`` without change.
:class:`PortfolioState` still computes its own drawdown through
:func:`stock_rl.metrics.max_drawdown` rather than tracking it alongside
the curve, so the imported name below is deliberate.
'''

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from stock_rl.api.constants import signal_max_weight
from stock_rl.metrics import max_drawdown


def _utcnow() -> datetime:
  '''Return the current time as an aware UTC datetime.

  Isolated so the module's only wall-clock read is a single function and
  a test can replace it.

  Returns:
    Timezone-aware datetime in UTC.
  '''
  return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Signal:
  '''One per-symbol recommendation, with the evidence behind it.

  Attributes:
    symbol: Venue symbol.
    action: One of ``'BUY'``, ``'HOLD'``, ``'SELL'``.
    confidence: Midrank percentile of the momentum score across the
      panel, in ``[0, 1]``. An ordering, not a probability; see the
      module docstring.
    reasons: Human-readable evidence, one clause per observable fact.
      Never empty, because a recommendation with no stated reason is the
      artefact this project's audit research is against.
    target_weight: Weight the signal asks for.
    held_weight: Weight currently held.
    score: Trailing 12-1 momentum, or None when history is short.
    rank: One-based rank by score, or None when unscorable.
  '''

  symbol: str
  action: str
  confidence: float
  reasons: tuple[str, ...]
  target_weight: float
  held_weight: float
  score: float | None = None
  rank: int | None = None

  def to_json(self) -> dict[str, Any]:
    '''Return the signal as JSON-safe primitives.

    Returns:
      Mapping with string keys and JSON-safe values.
    '''
    return {
      'symbol': self.symbol,
      'action': self.action,
      'confidence': round(self.confidence, 4),
      'reasons': list(self.reasons),
      'target_weight': round(self.target_weight, 6),
      'held_weight': round(self.held_weight, 6),
      'score': None if self.score is None else round(self.score, 6),
      'rank': self.rank,
    }


@dataclass(frozen=True, slots=True)
class BacktestRequest:
  '''A validated backtest trigger.

  Attributes:
    symbols: Symbols to run. Empty means every loaded symbol.
    capital: Starting cash in rupees.
    strategy: Name of a provider in :mod:`stock_rl.baselines`.
    rebalance_days: Bars between rebalances.
    max_weight: Cap on any single symbol's weight.
    history: Bars required before the first rebalance.
  '''

  symbols: tuple[str, ...] = ()
  capital: float = 10_000_000.0
  strategy: str = 'momentum_ranked'
  rebalance_days: int = 21
  max_weight: float = signal_max_weight
  history: int = 60

  def to_json(self) -> dict[str, Any]:
    '''Return the request as JSON-safe primitives.

    Returns:
      Mapping describing what was asked for, echoed in the response so a
      result can be attributed to the parameters that produced it.
    '''
    return {
      'symbols': list(self.symbols),
      'capital': self.capital,
      'strategy': self.strategy,
      'rebalance_days': self.rebalance_days,
      'max_weight': self.max_weight,
      'history': self.history,
    }


#: Configuration ``GET /api/baselines`` measures the control arm under
#: when the caller does not ask for another one.
#:
#: It is a :class:`BacktestRequest` rather than four loose constants so
#: that the object ``run_baseline`` is handed is the very object the
#: endpoint measured with, which is what stops one named strategy
#: carrying two Sharpes on one dashboard. The values are
#: :class:`BacktestRequest`'s own defaults, which are the same numbers
#: :mod:`stock_rl.portfolio`, :mod:`stock_rl.baselines` and
#: :mod:`stock_rl.experiment.harness` already use, so an
#: unparameterised control arm and an unparameterised
#: ``POST /api/backtest`` agree by construction rather than by
#: coincidence.
#:
#: A caller that means to trigger a run at other settings can ask for the
#: arm at those settings too: ``GET /api/baselines?rebalance_days=5``
#: measures under exactly the request the caller will POST, and the
#: payload says which one it used.
#:
#: PONYTAIL: one configuration for every provider, no per-strategy
#: override. Ceiling: a provider whose own paper uses different settings
#: is measured at these settings, so the row is comparable across
#: providers but not identical to that paper. Upgrade path: accept a
#: per-name override mapping validated by :func:`_parse_request`, and
#: echo the override in the row so the disclosure stays complete.
control_arm = BacktestRequest()


@dataclass(slots=True)
class PortfolioState:
  '''The book the read endpoints report.

  Defaults describe an empty portfolio rather than a zero-valued one: a
  fresh server has no positions and no history, and reporting an equity
  curve of ``[1.0]`` for that would be an assertion that performance
  was exactly zero rather than an absence of any.

  Attributes:
    capital: Starting or reference capital in rupees.
    value: Marked value of the book in rupees.
    cash: Uninvested cash in rupees. Only a live book or a backtest's
      implied undeployed capital populates this; the backtester does not
      report a cash balance, and inventing one would be a fabrication.
    weights: Held weight per symbol.
    targets: Target weight per symbol from the last signal or rebalance.
    equity: Equity curve normalised to 1.0.
    returns: Per-bar returns aligned with ``equity``.
    turnover: Total absolute weight traded, as recorded by the backtester.
    total_cost: Total transaction cost in rupees.
  '''

  capital: float = 0.0
  value: float = 0.0
  cash: float = 0.0
  weights: dict[str, float] = field(default_factory=dict)
  targets: dict[str, float] = field(default_factory=dict)
  equity: list[float] = field(default_factory=list)
  returns: list[float] = field(default_factory=list)
  turnover: float = 0.0
  total_cost: float = 0.0

  @property
  def invested(self) -> float:
    '''Return the sum of held weights, i.e. gross exposure.'''
    return sum(abs(weight) for weight in self.weights.values())

  @property
  def drawdown(self) -> float:
    '''Return the deepest drawdown on the equity curve.

    Computed by :func:`stock_rl.metrics.max_drawdown` rather than
    tracked alongside the curve, so the reported number cannot drift
    away from the series it describes.
    '''
    return max_drawdown(self.equity).depth


def _stamp(moment: datetime) -> str:
  '''Return an ISO 8601 UTC timestamp.

  A :class:`datetime.date` is accepted because :class:`stock_rl.bars.Bar`
  carries whatever the vendor CSV gave it, and daily exports routinely
  hold a bare date. Such a bar has no time of day to report and is
  stamped at midnight UTC rather than refused, so a daily panel's last bar
  can be named in the payload.

  Args:
    moment: Timezone-aware or naive datetime, or a date.

  Returns:
    ISO 8601 string, assumed UTC when the value is naive.
  '''
  if not isinstance(moment, datetime):
    moment = datetime(moment.year, moment.month, moment.day)
  if moment.tzinfo is None:
    moment = moment.replace(tzinfo=timezone.utc)
  return moment.isoformat()


def _floats(value: Any) -> list[float]:
  '''Return ``value`` as a list of finite floats.

  Args:
    value: Candidate sequence.

  Returns:
    The finite floats in it, or an empty list when it is not a sequence
    of numbers. Anything unusable becomes no state rather than a
    partially-populated series.
  '''
  if not isinstance(value, (list, tuple)):
    return []
  found = [float(item) for item in value
           if isinstance(item, (int, float))
           and not isinstance(item, bool)]
  return [item for item in found if math.isfinite(item)]


def _weights(value: Any) -> dict[str, float]:
  '''Return ``value`` as a symbol-to-weight mapping.

  Args:
    value: Candidate mapping.

  Returns:
    Finite weights keyed by symbol, empty when ``value`` is unusable.
  '''
  if not isinstance(value, dict):
    return {}
  found: dict[str, float] = {}
  for symbol, weight in value.items():
    if not isinstance(weight, (int, float)) or isinstance(weight, bool):
      continue
    number = float(weight)
    if math.isfinite(number):
      found[str(symbol)] = number
  return found


def _number(value: Any, fallback: float) -> float:
  '''Return ``value`` as a finite float.

  Args:
    value: Candidate number.
    fallback: Returned when ``value`` is not a finite number.

  Returns:
    The float, or ``fallback``.
  '''
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return fallback
  number = float(value)
  return number if math.isfinite(number) else fallback
