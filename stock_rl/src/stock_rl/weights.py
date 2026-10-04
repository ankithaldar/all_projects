#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Guards for target weights that must never reach a position.

Two failure modes are handled here, and both are silent in plain
arithmetic:

**NaN becomes the maximum weight.** ``max(0.0, min(0.1, nan))`` returns
``0.1``, not 0.0, because ``nan < 0.1`` is False so ``min`` keeps its
incumbent. A policy that divides by zero, or an actor whose loss
numerically blew up, would therefore be handed the largest permitted
position in that symbol -- the worst possible response to an undefined
value, and completely invisible in the equity curve.

**Infinite weight saturates rather than failing.** ``inf`` clamps cleanly
to the cap, which at least cannot exceed the limit, but a non-finite
input is a bug that should be reported at the boundary rather than
quietly turned into a full position.

Both guards raise instead. A weight provider that emits a non-finite
value has a bug, and the caller needs to know which symbol and what
value, not a plausible-looking portfolio.
'''

from __future__ import annotations

import math

from stock_rl.costs import CostModel, Side

__all__ = ['affordable_scale', 'clamp_weight']


def clamp_weight(value: float, max_weight: float) -> float:
  '''Return a weight clamped into ``[0, max_weight]``, rejecting NaN.

  Args:
    value: Candidate weight. Must be a finite number.
    max_weight: Upper bound, already validated by the caller.

  Returns:
    The weight clamped into range.

  Raises:
    ValueError: If ``value`` is NaN or infinite. NaN in particular is
      never coerced, because the arithmetic silently produces a
      maximum-weight position.
  '''
  if math.isnan(value):
    raise ValueError('weight is NaN, refusing to size a position')
  if math.isinf(value):
    raise ValueError('weight is infinite, refusing to size a position')
  return max(0.0, min(max_weight, value))


def affordable_scale(
  cash: float,
  prices: dict[str, float],
  deltas: dict[str, int],
  costs: CostModel,
) -> float:
  '''Return the fraction of intended deltas the available cash can fund.

  Sizing an order as ``int(target * value / price)`` bounds the notional
  by the portfolio value but ignores the charges, so a fully invested
  book necessarily overspends and its cash goes negative. That is
  unpriced borrowing -- invisible on a flat series and free leverage on
  a rising one. This scales the deltas down to what can actually be paid
  for.

  Args:
    cash: Cash available before trading. A negative value means nothing
      is affordable.
    prices: Fill price per symbol.
    deltas: Intended share change per symbol.
    costs: Transaction cost model.

  Returns:
    A scale factor in ``[0, 1]``. ``1.0`` when the trades are fully
    affordable, which is the common case.
  '''
  spend = sum(deltas[symbol] * prices[symbol] for symbol in deltas)
  if spend <= 0.0:
    return 1.0
  charges = _charges(prices, deltas, costs, 1.0)
  if spend + charges <= cash:
    return 1.0
  # Binary search. Costs do not scale linearly with size because the
  # per-order DP charge is flat, so no closed form exists; twenty
  # iterations resolve to well under a single share.
  low = 0.0
  high = 1.0
  for _ in range(20):
    mid = (low + high) / 2.0
    if spend * mid + _charges(prices, deltas, costs, mid) <= cash:
      low = mid
    else:
      high = mid
  return low


def _charges(
  prices: dict[str, float],
  deltas: dict[str, int],
  costs: CostModel,
  scale: float,
) -> float:
  '''Return total transaction cost for scaled deltas.

  Args:
    prices: Fill price per symbol.
    deltas: Intended share change per symbol.
    costs: Transaction cost model.
    scale: Multiplier applied to every delta.

  Returns:
    Total cost in rupees, excluding any delta that scales to zero.
  '''
  total = 0.0
  for symbol, delta in deltas.items():
    if not delta:
      continue
    quantity = abs(delta * scale) * prices[symbol]
    if quantity <= 0.0:
      continue
    total += costs.one_way(
      Side.BUY if delta > 0 else Side.SELL, quantity)
  return total

