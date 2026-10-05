#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Guards for target weights that must never reach a position.

The first two of the three failure modes handled here are silent in
plain arithmetic:

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

**A cap that never reaches the provider is decoration.** The third
guard here is about the call rather than the number. Every provider in
:mod:`stock_rl.baselines` carries its own ``max_weight`` default of
0.10, so a caller who asked for 0.30 and passed the provider to the
engine unbound got a 0.10 book and a payload claiming 0.30 -- three
different requested caps returned one byte-identical Sharpe.
:func:`bind_weight_cap` closes that, and :func:`applied_weight_cap`
reads the answer back off the book so the number a payload reports can
be checked against the number that was applied. Both live here because
:mod:`stock_rl.api`, :mod:`stock_rl.pipeline` and :mod:`stock_rl.cli`
each report a cap, and one implementation of "what cap was this book
built at" is worth more than three that agree today.
'''

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence

from stock_rl.bars import Bar
from stock_rl.costs import CostModel, Side

__all__ = [
  'CAP_KEYWORD',
  'WeightProvider',
  'accepts_weight_cap',
  'affordable_scale',
  'applied_weight_cap',
  'bind_weight_cap',
  'clamp_weight',
]

#: The keyword a weight provider takes its per-symbol cap on.
#:
#: It is bound by **keyword**, never positionally, and that is not a
#: style preference. The five providers in :mod:`stock_rl.baselines`
#: disagree about where the cap sits in their own signatures: second for
#: ``equal_weight`` and ``buy_and_hold``, third for ``low_volatility``,
#: fifth for ``momentum_ranked`` and sixth for
#: ``trend_filtered_momentum``. One positional call over that table hands
#: the weight cap to a moving-average window and computes
#: ``sma(closes, 0.1)`` instead of raising anything a caller can read.
CAP_KEYWORD = 'max_weight'

#: A weight provider maps visible price history to target weights. Only
#: bars strictly before the fill bar are passed in, so a provider cannot
#: see the price it will be filled at.
WeightProvider = Callable[[dict[str, list[Bar]]], dict[str, float]]


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


def accepts_weight_cap(provider: WeightProvider) -> bool:
  '''Return whether ``provider`` can be handed the cap by keyword.

  Every provider in :mod:`stock_rl.baselines` takes ``max_weight``, but
  the table is discovered from that module's ``__all__`` and is therefore
  open to anything registered there later, including a callable with no
  introspectable signature at all. The question is asked rather than
  assumed, because assuming it is what produced the original defect: an
  unbound cap is indistinguishable from a bound one until someone
  compares a requested number against the book that came back.

  A ``**kwargs`` parameter counts as accepting the cap, since a keyword
  cannot collide with anything it declares.

  PONYTAIL: introspection rather than a try/except around the call.
  Ceiling: a callable object whose ``__call__`` hides the keyword behind
  a decorator is reported as not accepting it, and its row then says so.
  Upgrade path: let a provider declare :data:`CAP_KEYWORD` as an
  attribute.

  Args:
    provider: The weight provider to inspect.

  Returns:
    True when the cap can be passed as a named argument.
  '''
  try:
    parameters = inspect.signature(provider).parameters
  except (TypeError, ValueError):
    # A C builtin or a deliberately opaque callable. Its weight may not
    # be settable, and the row reports that rather than guessing.
    return False
  named = (
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
  )
  return any(
    parameter.kind is inspect.Parameter.VAR_KEYWORD
    or (parameter.name == CAP_KEYWORD and parameter.kind in named)
    for parameter in parameters.values()
  )


def bind_weight_cap(
  provider: WeightProvider,
  max_weight: float,
) -> WeightProvider:
  '''Return ``provider`` with the caller's cap bound into it.

  The engine's cap is a ceiling, not a target.
  :func:`stock_rl.portfolio.run_portfolio` clamps whatever it is handed to
  ``max_weight``, so a provider that applies a *tighter* cap of its own
  produces a book strictly inside the ceiling and the requested number is
  never the binding constraint. That is not a harmless difference: each
  of the five baselines in :mod:`stock_rl.baselines` defaults its own
  ``max_weight`` to 0.10, so a caller asking for 0.30 used to be handed a
  0.10 book and a payload claiming 0.30.

  Binding the cap here is done for both sides of the seam: the provider
  stops imposing its own default and the engine keeps enforcing the
  ceiling.

  A provider that cannot take the keyword is called unchanged rather than
  refused. Refusing would file a provider this module cannot introspect
  as a defect when it may simply derive weights from the panels and have
  no opinion about the cap at all. What is not left unspecified is the
  disclosure: the row reports ``cap_bound`` False alongside
  :func:`applied_weight_cap`, so an unbound provider cannot leave a
  number in a payload that no book backs.

  Args:
    provider: The weight provider to bind.
    max_weight: Cap on any single symbol's weight.

  Returns:
    A one-argument callable, the shape ``run_portfolio`` calls.
  '''
  if accepts_weight_cap(provider):
    def capped(visible: dict[str, list[Bar]]) -> dict[str, float]:
      '''Return the provider's weights under the requested cap.

      Args:
        visible: Price panels visible at the decision bar.

      Returns:
        Target weight per symbol.
      '''
      return provider(visible, **{CAP_KEYWORD: max_weight})
    return capped

  def loose(visible: dict[str, list[Bar]]) -> dict[str, float]:
    '''Return the provider's weights, which could not be given a cap.

    Args:
      visible: Price panels visible at the decision bar.

    Returns:
      Target weight per symbol.
    '''
    return provider(visible)
  return loose


def applied_weight_cap(
  snapshots: Sequence[Mapping[str, float]],
) -> float | None:
  '''Return the largest weight the book actually held, or None.

  Measured from the snapshots the backtester recorded rather than from
  the request, so it cannot disagree with the book it describes. This is
  the number that makes ``applied == reported`` checkable: the engine
  guarantees ``applied <= requested``, and a provider whose internal cap
  was tighter is exactly the case where ``applied < requested``.

  A cap is a ceiling and not a target, so equality is the expected
  outcome only when the cap is the binding constraint -- 1/N over three
  symbols tops out at 0.333 no matter how large the requested cap is.
  Callers comparing the two need that in mind.

  Args:
    snapshots: One weight mapping per rebalance, in order, as recorded by
      :func:`stock_rl.portfolio.run_portfolio`.

  Returns:
    The heaviest single-symbol weight across every rebalance, or None if
      the run never rebalanced.
  '''
  held = [
    weight
    for snapshot in snapshots
    for weight in snapshot.values()
  ]
  return max(held) if held else None


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

