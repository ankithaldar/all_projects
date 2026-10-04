#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Backtest engine: turns a signal function into a cost-aware equity curve.

The whole purpose of this module is to make look-ahead bias structurally
impossible rather than merely discouraged.

Look-ahead happens when information from bar ``t+1`` influences a decision
taken at bar ``t``. The engine closes the door in three ways:

  1. A signal sees only the bars up to and including the bar whose close
     it just observed, enforced by slicing rather than by convention.
  2. A signal produced at the close of bar ``t`` is filled at the *open*
     of bar ``t+1``. You cannot know the open of a bar you have not
     reached, so the fill price is never an input to the decision.
  3. Cost is charged on the notional actually traded, so turnover cannot
     be quietly omitted and then wondered about.

Filling at the next open is deliberately pessimistic. It ignores the fact
that a real order placed at the close might fill near the close, and it
never lets a favourable overnight gap work in our favour. Optimistic
fills are the most common way a backtest invents performance that does
not exist.
'''

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from stock_rl.bars import Bar
from stock_rl.costs import DELIVERY, CostModel, Side

__all__ = ['BacktestResult', 'Fill', 'run_backtest']

#: A signal maps the visible history to a target weight in ``[0, 1]``,
#: where 0 is flat and 1 is fully invested. Negative values are floored to
#: flat rather than interpreted as a short.
#:
#: PONYTAIL: this returns a bare float, which carries no record of *why*
#: the decision was made. That is acceptable for measuring a strategy but
#: not for running one: an unauditable decision cannot be defended to a
#: broker, an exchange or a regulator, and cannot be reviewed after a loss.
#: Ceiling: a signal can explain its own weight but not the reasoning
#: behind it. Upgrade path: widen this to a record carrying an action,
#: a timeframe and a reason; ``run_backtest`` needs only ``.weight``, so
#: the change is additive and does not disturb the backtest loop.
Signal = Callable[[list[Bar]], float]


@dataclass(frozen=True, slots=True)
class Fill:
  '''One executed trade.

  Attributes:
    bar_index: Index of the bar whose open the trade filled at.
    timestamp: Timestamp of that bar.
    quantity: Signed share count. Positive is a buy, negative a sell.
    price: Fill price, taken from the bar open.
    cost: Total charges in rupees, including slippage.
  '''

  bar_index: int
  timestamp: datetime
  quantity: int
  price: float
  cost: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
  '''Everything a backtest produced, with no hidden intermediate state.

  Attributes:
    equity: Equity curve normalised to start at 1.0, one entry per bar.
    returns: Per-bar equity returns, aligned with ``equity``. The first
      entry is 0.0 because there is no prior bar to compare against.
    target_weights: Weight targeted on each bar, after clamping.
    fills: Executed trades in chronological order.
    total_cost: Total transaction cost in rupees.
    bars_processed: Number of bars in the equity curve.
  '''

  equity: list[float] = field(default_factory=list)
  returns: list[float] = field(default_factory=list)
  target_weights: list[float] = field(default_factory=list)
  fills: list[Fill] = field(default_factory=list)
  total_cost: float = 0.0
  bars_processed: int = 0


def _clamp(value: float, low: float, high: float) -> float:
  '''Return ``value`` restricted to ``[low, high]``.

  Args:
    value: Candidate value.
    low: Lower bound.
    high: Upper bound.

  Returns:
    The clamped value.
  '''
  return max(low, min(high, value))


def run_backtest(
  bars: list[Bar],
  signal: Signal,
  capital: float = 1_000_000.0,
  costs: CostModel = DELIVERY,
  max_weight: float = 1.0,
  min_trade_value: float = 0.0,
) -> BacktestResult:
  '''Run a long-only backtest that is fully invested or flat.

  Shares are rounded to whole units, because fractional shares do not
  exist in NSE cash equity and pretending otherwise understates cost for
  large notionals.

  The first bar is never traded because no signal precedes it, and the
  last bar's signal is discarded because there is no following open to
  fill against. Both are deliberate: fabricating either would put fills
  in the curve that could not have happened.

  Args:
    bars: Price bars in ascending timestamp order. At least two required.
    signal: Maps visible history to a target weight.
    capital: Starting cash in rupees.
    costs: Transaction cost model.
    max_weight: Maximum target weight. The book is long-only by
      construction: weights are clamped to ``[0, max_weight]``, so a short
      position is not expressible by accident.
    min_trade_value: Trades below this absolute notional are skipped,
      modelling a minimum ticket size and stopping a large account from
      churning for a few rupees of drift.

  Returns:
    A ``BacktestResult``.

  Raises:
    ValueError: If fewer than two bars are supplied, capital is not
      positive, or ``max_weight`` falls outside ``(0, 1]``.
  '''
  if len(bars) < 2:
    raise ValueError(f'need at least 2 bars, got {len(bars)}')
  if capital <= 0.0:
    raise ValueError(f'capital must be positive, got {capital}')
  if not 0.0 < max_weight <= 1.0:
    raise ValueError(f'max_weight must be in (0, 1], got {max_weight}')

  equity: list[float] = []
  returns: list[float] = []
  weights: list[float] = []
  fills: list[Fill] = []
  cash = capital
  quantity = 0
  total_cost = 0.0
  pending: float | None = None
  # Portfolio value as of the previous bar's close. Order sizing must use
  # this, never the current bar's close: the fill happens at the current
  # bar's open, which has already happened by the time the close is known.
  # Sizing on the close would quietly buy more shares as price rises.
  known_value = capital

  for index, bar in enumerate(bars):
    # 1. Execute any order decided on the previous bar's close, at this
    #    bar's open, sized on the value known at that decision point.
    if pending is not None:
      fill_price = bar.open
      if fill_price > 0.0:
        target_value = pending * known_value
        target_quantity = int(target_value / fill_price)
        delta = target_quantity - quantity
        delta_value = delta * fill_price
        if delta != 0 and abs(delta_value) >= max(min_trade_value, 1e-9):
          side = Side.BUY if delta > 0 else Side.SELL
          cost = costs.one_way(side, abs(delta_value))
          cash -= delta_value + cost
          quantity = target_quantity
          total_cost += cost
          fills.append(Fill(
            bar_index=index,
            timestamp=bar.timestamp,
            quantity=delta,
            price=fill_price,
            cost=cost,
          ))

    # 2. Mark to market at this bar's close, which is the next decision
    #    point's only permitted source of portfolio value.
    known_value = cash + quantity * bar.close
    equity.append(known_value / capital)
    returns.append(
      0.0 if index == 0 else equity[index] / equity[index - 1] - 1.0)
    weights.append(pending if pending is not None else 0.0)
    pending = None

    # 3. Decide for the next bar using only what has happened so far.
    if index + 1 < len(bars):
      pending = _clamp(
        float(signal(bars[:index + 1])), 0.0, max_weight)

  return BacktestResult(
    equity=equity,
    returns=returns,
    target_weights=weights,
    fills=fills,
    total_cost=total_cost,
    bars_processed=len(equity),
  )
