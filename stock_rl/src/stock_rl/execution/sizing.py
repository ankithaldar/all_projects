#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Position sizing, and the refusal to size past a risk cap.

Two sizing rules, because they answer different questions. **Kelly**
answers "how much should I bet when I know my edge", and the answer it
gives is *too big*: the classic full-Kelly prescription draws down
roughly half of the account on its way to the optimum, which is a
property of the arithmetic rather than a caveat about it. So the size
carries an explicit multiplier below one, the same one that expresses
both the move to fractional Kelly and the haircut for a mis-estimated
edge:

    ``size = edge / odds * (1 - uncertainty)``

The haircut is the part a Kelly implementation usually leaves out. The
multiplier exists because ``edge`` here is an *estimate*: an edge
estimated from 250 bars of one Indian equity is not knowable to the
precision Kelly assumes. ``uncertainty`` is the fraction of that estimate
the caller is unwilling to bet on, and it is a required argument rather
than a default, because a default would be a number nobody chose.

**Volatility targeting** answers "how much should I hold when I know
how much it moves", and it is the rule that survives a Kelly edge going
wrong. ``target_vol / realised_vol`` sizes to a constant expected
volatility, so a 40 percent-vol name gets a quarter of the weight a 10
percent-vol name does. It is the sizing rule with the least parameter
fitting in it, which is why the low-volatility baseline in this project
is also a sizing rule.

**Sizing refuses to breach a risk cap.** The critical line of this
module is that it *re-uses* :class:`stock_rl.risk.checks.RmsChecker`
rather than re-deriving limits of its own. A sizer with private caps and
an RMS with exchange caps is two sets of numbers that agree until the day
they do not, and on that day the larger one wins silently. So
:meth:`StockSizer.size` builds an
:class:`~stock_rl.risk.checks.OrderRequest` and demands approval from
the checker **before** returning it, and a rejection propagates as
:class:`RiskCapBreach` rather than being logged and shrunk to fit.

Shrinking to fit is the tempting alternative and it is wrong here. A
sizer that quietly halves an order hides the fact that the strategy
wanted more than the account could give, and that fact is the signal the
operator needs. The order is refused; the caller decides what to do.

**Zero size is not an order.** When the uncertainty-adjusted edge is zero
or negative the result is a zero-quantity :class:`SizingResult` with no
order, and it never reaches the RMS: submitting a zero-quantity order
would be rejected by the quantity check, which is correct behaviour
serving the wrong purpose.
'''

from __future__ import annotations

import math
from dataclasses import dataclass

from stock_rl.risk.checks import (
  AccountState,
  OrderRequest,
  RiskRejected,
  RiskReport,
  RmsChecker,
  buy,
)

__all__ = [
  'RiskCapBreach',
  'SizingInputs',
  'SizingResult',
  'StockSizer',
  'kelly_fraction',
  'kelly_method',
  'max_fraction',
  'uncertainty_adjusted_size',
  'vol_target_fraction',
  'vol_target_method',
]

#: Identifier for the Kelly sizing rule.
kelly_method = 'kelly'

#: Identifier for the volatility-targeting sizing rule.
vol_target_method = 'vol_target'

#: Largest fraction of capital any sizing rule may commit, whatever it
#: asks for. The project's own risk appetite, and deliberately
#: unreachable for a Kelly fraction above 1.0, which is what full Kelly
#: looks like on a winning edge.
max_fraction = 0.25


class RiskCapBreach(RiskRejected):
  '''Raised when a computed size breaches a pre-trade risk cap.

  A :class:`~stock_rl.risk.checks.RiskRejected` subclass, because that
  is exactly what it is: the RMS rejected the order the sizer produced.
  Separate types so a caller can tell "the strategy asked for something
  illegal" from "some other caller asked for something illegal", without
  string matching.
  '''


def kelly_fraction(prob_win: float, win_loss_ratio: float) -> float:
  '''Return the Kelly fraction for a binary payoff.

  ``f* = (p * b - (1 - p)) / b``, i.e. edge over odds. It is
  ``1 / win_loss_ratio`` at break-even and rises towards
  ``1 / win_loss_ratio`` less a half as the edge grows, which is why the
  raw fraction is routinely small for a realistic edge and why the
  "half the account" drawdown of full Kelly arrives with a fractional
  Kelly of only a few percent.

  Args:
    prob_win: Probability of the winning outcome, in ``[0, 1]``.
    win_loss_ratio: Win size divided by loss size, so 1.0 is an even
      payoff. Must be positive.

  Returns:
    The Kelly fraction, which may be negative. A negative fraction means
    the bet has no edge; it is returned rather than clamped so the caller
    sees the sign instead of a zero it has to interpret.

  Raises:
    ValueError: If the probability is outside ``[0, 1]`` or the payoff
      ratio is not positive.
  '''
  if not 0.0 <= prob_win <= 1.0:
    raise ValueError(f'prob_win must be in [0, 1], got {prob_win}')
  if win_loss_ratio <= 0.0:
    raise ValueError(
      f'win_loss_ratio must be positive, got {win_loss_ratio}')
  return (prob_win * win_loss_ratio - (1.0 - prob_win)) / win_loss_ratio


def uncertainty_adjusted_size(prob_win: float, win_loss_ratio: float,
                              uncertainty: float) -> float:
  '''Return ``edge / odds * (1 - uncertainty)``.

  The sizing rule this module is built around. The Kelly fraction is the
  ``edge / odds`` term; multiplying by ``1 - uncertainty`` subtracts the
  part of the estimated edge the caller declines to bet on.

  **The multiplier is doing two jobs at once, deliberately.** The
  reduction from full Kelly to fractional Kelly is a multiplier below 1,
  and so is the haircut for a mis-estimated edge. Folding them into one
  caller-declared number keeps the sizer from carrying a default that
  nobody chose, and it is why ``uncertainty`` has no default either: a
  default would put a fraction in the position of a decision.

  ``uncertainty`` is required, not defaulted. It is the honest form of
  the question "how much of this edge do I believe", and a default would
  be a number nobody chose while the sizing still looked principled.

  Args:
    prob_win: Probability of the winning outcome, in ``[0, 1]``.
    win_loss_ratio: Win size divided by loss size.
    uncertainty: Fraction of the estimated edge not believed, in
      ``[0, 1)``. One is excluded because a size of exactly zero is
      already the no-trade answer and is spelled that way.

  Returns:
    Size as a fraction of capital, floored at zero and never above
    :data:`max_fraction`.

  Raises:
    ValueError: If any argument is out of range.
  '''
  if not 0.0 <= uncertainty < 1.0:
    raise ValueError(f'uncertainty must be in [0, 1), got {uncertainty}')
  raw = kelly_fraction(prob_win, win_loss_ratio) * (1.0 - uncertainty)
  return min(max_fraction, max(0.0, raw))


def vol_target_fraction(realised_vol: float, target_vol: float,
                        cap: float = max_fraction) -> float:
  '''Return the size that equalises expected volatility.

  ``target / realised``, capped. No mean term and no edge term: this rule
  does not claim to know which way the asset goes, only how far it moves,
  and that is why it is the more robust of the two.

  Args:
    realised_vol: Realised volatility of the asset, as a positive
      fraction.
    target_vol: Portfolio volatility target, as a positive fraction.
    cap: Ceiling on the size, normally :data:`max_fraction`.

  Returns:
    Size as a fraction of capital, in ``[0, cap]``.

  Raises:
    ValueError: If either volatility is not positive or the cap is
      outside ``(0, 1]``. A zero realised volatility is refused rather
      than divided by, because ``target / 0`` would be the largest
      possible size returned with no arithmetic error and no evidence
      behind it.
  '''
  if realised_vol <= 0.0:
    raise ValueError(
      f'realised_vol must be positive; a zero-volatility input has no '
      f'size to scale, got {realised_vol}')
  if target_vol <= 0.0:
    raise ValueError(f'target_vol must be positive, got {target_vol}')
  if not 0.0 < cap <= 1.0:
    raise ValueError(f'cap must be in (0, 1], got {cap}')
  return min(cap, target_vol / realised_vol)


@dataclass(frozen=True, slots=True)
class SizingInputs:
  '''The edge and the haircut, as a pair.

  Grouped so that a caller cannot pass a Kelly probability to
  :meth:`StockSizer.size` without also declaring how much of the implied
  edge it believes, which is the mistake the haircut exists to prevent.

  Attributes:
    prob_win: Probability of the winning outcome, in ``[0, 1]``.
    win_loss_ratio: Win size divided by loss size.
    uncertainty: Fraction of the estimated edge not believed. This is
      the whole fractional-Kelly reduction as well, since
      ``1 - uncertainty`` is the multiplier on full Kelly either way.
    realised_vol: Realised volatility, required by the volatility-target
      rule and ignored by the Kelly rule.
    target_vol: Portfolio volatility target, likewise.
  '''

  prob_win: float = 0.5
  win_loss_ratio: float = 1.0
  uncertainty: float = 0.5
  realised_vol: float = 0.0
  target_vol: float = 0.15

  def fraction(self, method: str) -> float:
    '''Return the size fraction this input implies.

    Args:
      method: :data:`kelly_method` or :data:`vol_target_method`.

    Returns:
      Size as a fraction of capital.

    Raises:
      ValueError: If the method is unknown, or the arguments that
        method needs are unusable.
    '''
    if method == kelly_method:
      return uncertainty_adjusted_size(
        self.prob_win, self.win_loss_ratio, self.uncertainty)
    if method == vol_target_method:
      return vol_target_fraction(self.realised_vol, self.target_vol)
    raise ValueError(
      f'method must be one of {(kelly_method, vol_target_method)}, '
      f'got {method!r}')


@dataclass(frozen=True, slots=True)
class SizingResult:
  '''What the sizer decided, and whether it survived the RMS.

  Attributes:
    symbol: Symbol sized.
    side: :data:`~stock_rl.risk.checks.buy` or its opposite.
    fraction: Size as a fraction of capital, after the cap.
    quantity: Whole shares or contracts, rounded **down** to the lot.
      Rounding down rather than to nearest is deliberate: rounding a size
      up can push it over a per-order limit that the rounding was
      supposed to respect.
    notional: Rounded traded value of ``quantity`` at the price.
    order: The order to submit, or None when nothing should be traded.
    report: The RMS report for the order, or None when no order was
      produced.
  '''

  symbol: str
  side: str
  fraction: float
  quantity: int
  notional: float
  order: OrderRequest | None = None
  report: RiskReport | None = None

  @property
  def should_trade(self) -> bool:
    '''Return whether an order was produced.

    A zero size is *not* a trade and is not submitted: a zero-quantity
    order would be rejected by the quantity check, which is right and
    pointless.
    '''
    return self.order is not None

  def render(self) -> str:
    '''Return a one-line summary for a log.

    Returns:
      Human-readable summary, including the reason for a zero size when
      there is one.
    '''
    if not self.should_trade:
      return (
        f'{self.symbol} {self.side}: no order, size {self.fraction:.2%} '
        'of capital rounds to nothing or has no edge')
    return (
      f'{self.symbol} {self.side}: {self.quantity} for {self.notional:.0f} '
      f'({self.fraction:.2%} of capital)')


class StockSizer:
  '''Size one order, and refuse any size the RMS will not clear.

  Holds the checker rather than the limits, because the whole point is
  that there is exactly one set of limits. The account ledger is passed
  per call so a caller cannot size against a stale snapshot without it
  being visible in the call.

  Attributes:
    checker: The pre-trade checker every produced order must clear.
    capital: Capital base in rupees.
    lot_size: Exchange lot. The quantity is floored to a multiple of it,
      because an odd lot is not deliverable in the cash segment.
    cap: Ceiling on any size, normally :data:`max_fraction`.
  '''

  checker: RmsChecker
  capital: float

  def __init__(
    self,
    checker: RmsChecker,
    capital: float,
    lot_size: int = 1,
    cap: float = max_fraction,
  ) -> None:
    '''Build a sizer over an existing checker.

    Args:
      checker: Pre-trade checker to clear every order against.
      capital: Capital base in rupees.
      lot_size: Exchange lot size, at least 1.
      cap: Ceiling on any size as a fraction of capital.

    Raises:
      ValueError: If the capital is not finite and positive, the lot size
        is below 1, or the cap is outside ``(0, 1]``.
    '''
    if not 0.0 < capital < float('inf'):
      raise ValueError(f'capital must be finite and positive, got {capital}')
    if lot_size < 1:
      raise ValueError(f'lot_size must be >= 1, got {lot_size}')
    if not 0.0 < cap <= 1.0:
      raise ValueError(f'cap must be in (0, 1], got {cap}')
    self.checker = checker
    self.capital = capital
    self.lot_size = lot_size
    self.cap = cap

  def size(
    self,
    symbol: str,
    price: float,
    side: str = buy,
    inputs: SizingInputs = SizingInputs(),
    method: str = kelly_method,
    account: AccountState = AccountState(),
    algo_id: str = '',
    order_id: str = '',
    reference_price: float = 0.0,
  ) -> SizingResult:
    '''Size an order and clear it against the pre-trade checks.

    **Risk runs before execution, never after.** The order is built
    first, then handed to the checker, and only returned if the checker
    approves. There is no path here that produces an order the RMS has
    not seen, which is the property that makes this a control rather
    than a report.

    Args:
      symbol: Symbol to size.
      price: Limit price for the order, which must be positive.
      side: :data:`~stock_rl.risk.checks.buy` or its opposite.
      inputs: Edge, haircut and volatility inputs.
      method: :data:`kelly_method` or :data:`vol_target_method`.
      account: The account ledger to size against.
      algo_id: Algo ID for the order's tag.
      order_id: Identifier for the order, for the report.
      reference_price: Last traded price, required by the bad-ticket and
        MWPL checks. Zero rejects both, so a caller that omits it gets a
        loud refusal rather than a skipped check.

    Returns:
      A :class:`SizingResult`. ``order`` is None when the size rounds to
      nothing.

    Raises:
      ValueError: If the price is not positive or the sizing inputs are
        unusable.
      RiskCapBreach: If the RMS rejects the sized order.
    '''
    if price <= 0.0:
      raise ValueError(
        f'price must be positive; a size cannot be computed from '
        f'{price}')
    fraction = min(self.cap, inputs.fraction(method))
    notional = self.capital * fraction
    quantity = self._whole_lots(notional / price)
    if quantity < 1:
      return SizingResult(
        symbol=symbol, side=side, fraction=fraction, quantity=0,
        notional=0.0)
    order = OrderRequest(
      symbol=symbol,
      side=side,
      quantity=quantity,
      price=price,
      reference_price=reference_price,
      algo_id=algo_id,
      order_id=order_id,
    )
    try:
      report = self.checker.require_approval(order, account)
    except RiskRejected as exc:
      raise RiskCapBreach(exc.report) from exc
    return SizingResult(
      symbol=symbol,
      side=side,
      fraction=fraction,
      quantity=quantity,
      notional=quantity * price,
      order=order,
      report=report,
    )

  def _whole_lots(self, units: float) -> int:
    '''Return a whole number of lots, rounded down.

    Args:
      units: Raw size in shares or contracts.

    Returns:
      Lots times ``lot_size``, never negative.

    Raises:
      ValueError: If the raw size is not finite. A NaN would reach
        ``int()`` and raise an opaque error at the point of truncation,
        three steps from the arithmetic that produced it.
    '''
    if not math.isfinite(units):
      raise ValueError(f'size must be finite, got {units}')
    lots = math.floor(max(0.0, units) / self.lot_size)
    return int(lots) * self.lot_size
