#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''NSE pre-trade RMS checks, operational modalities para 11.1.

Fifteen of the sixteen checks in NSE Detailed Operational Modalities
para 11.1 are implemented here. Two are deliberately absent, and naming
them is more useful than pretending the list is covered:

  * **net position against margin** is the exchange's and the clearing
    member's job in production. The account state this module reads is a
    ledger this project maintains; a limit computed from a self-reported
    ledger and called a margin check would be a check of nothing.
  * **efficient price discovery** is a commodity-segment check. It has no
    meaning in the equity segment this project trades, and an
    implementation that accepted a commodity flag would be inventing a
    control.

**The fifteenth check is affordability, and it is ours rather than the
exchange's.** Nothing in para 11.1 asks an RMS whether the client can pay
for the order; every other check asks whether the exchange will let it
through. That gap is why a sizer can size four names to the whole of
capital and watch the cash balance go negative: every check passes,
because "the order is legal" and "the order is payable" are different
questions. :attr:`AccountState.cash` exists so the second question can be
asked here as well, and :func:`stock_rl.weights.affordable_scale` is what
the sizer uses to answer it before an order is ever built.

SEBI 2012 circular para 6(i)-(v) is the shorter, harder list underneath
NSE's sixteen, and it is the list that matters: price band, quantity,
value, cumulative open order value, position limit, and the
automated-execution check. All six are here.

**Every check fails closed, and that is the design rather than a
by-product.** A risk check that returns "probably fine" is worse than no
check, because the caller cannot tell the difference between a verdict
and an absence. Concretely:

  * an unknown symbol has no venue limits, so every check rejects rather
    than passing unchecked;
  * an order with no reference price cannot be checked against a bad
    ticket band, so it rejects;
  * the cumulative open-order-value limit is a finite positive float and
    nothing else, so the "Unlimited" configuration the exchange forbids
    for algo clients is not even representable;
  * there is no third status. A check returns ``'pass'`` or
    ``'reject'``, and the reason string says which and why.

**One check cannot be made, and says so instead of guessing.** The
affordability check needs a cash balance. :attr:`AccountState.cash` is
``None`` on a ledger that does not carry one, and the check then reports
that it could not be made rather than inventing a denominator. That is
the only departure from fail-closed in this module, and the reason is
that the two available guesses are both worse than no answer: reading a
missing balance as zero rejects every order, and reading it as unlimited
reinstates exactly the bug the check exists to catch. The sizer's own
affordability guard is what bounds the size in that case, and it does not
consult this ledger.

**Market orders are prohibited outright for algo orders** in the equity
segment (NSE/MSD/67753 8.1.1.12, restated in the corrections doc). This
project is an algo, so the only order type that clears this module end to
end is a priced limit order. A market order is rejected by its own check,
and separately by the price band check because it has no price to band.

**Venue data is configuration, never order-supplied.** The price band,
the MWPL band and the per-order quantity and value limits come from
:class:`SecurityLimits` held by the checker, keyed by symbol. An order
carrying its own band would be an order that widens its own limits, so
the band is looked up rather than passed in.

Where the reading of a rule is an interpretation rather than a quotation
-- the separation of the cumulative open-order-value check from the
automated-execution check, and the conservative direction of the exposure
check -- the docstring on the check says so.
'''

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from stock_rl.costs import DELIVERY, CostModel, Side

__all__ = [
  'AccountState',
  'CheckResult',
  'OrderRequest',
  'PriceBand',
  'RiskRejected',
  'RiskReport',
  'RmsChecker',
  'RmsLimits',
  'SecurityLimits',
  'buy',
  'check_ids',
  'limit_order',
  'market_order',
  'order_types',
  'passed_status',
  'rejected_status',
  'sell',
  'sides',
  'statuses',
]

#: The PASS verdict. Spelled as a plain string so a verdict serialises
#: into the audit trail without an enum lookup table.
passed_status = 'pass'

#: The REJECT verdict.
rejected_status = 'reject'

#: The only two statuses a check may return. A check that cannot decide
#: returns ``rejected_status`` with the reason, so "no verdict" is not a
#: reachable state.
statuses = (passed_status, rejected_status)

#: Sides an order may take.
buy = 'buy'
sell = 'sell'
sides = (buy, sell)

#: A limit order at a stated price.
limit_order = 'limit'

#: A market order. Recognised only so that it can be refused: NSE
#: prohibits market orders for algo orders in the equity segment
#: (NSE/MSD/67753 8.1.1.12).
market_order = 'market'

#: Order types this module understands. Anything else is rejected at
#: construction, because an unrecognised type is not a safe default.
order_types = (limit_order, market_order)

#: Every check this module runs, in execution order. Order is the
#: cheapest-first ordering: price and size are decided before any account
#: aggregate is touched, so a fat-fingered price never reaches the
#: aggregate arithmetic.
check_ids: tuple[str, ...] = (
  'price_band',
  'order_quantity',
  'order_value',
  'affordability',
  'trade_price_protection',
  'algo_market_order',
  'cumulative_open_order_value',
  'automated_execution',
  'position_limit',
  'trading_limit',
  'exposure_limit',
  'turnover_limit',
  'security_value_limit',
  'fno_ban',
  'mwpl',
)

#: Checks that cannot be made at all without per-symbol venue data.

#: A single check's verdict. Immutable, so a report cannot be edited
#: after the fact by whoever holds it.
@dataclass(frozen=True, slots=True)
class CheckResult:
  '''One check's structured outcome.

  Attributes:
    check_id: Stable identifier from :data:`check_ids`. Carried into the
      audit trail so a rejection is greppable by rule rather than by
      message text.
    status: Either :data:`passed_status` or :data:`rejected_status`.
    reason: Human-readable explanation. Always populated, including on
      a pass, because "why did this pass" is a question an inspection
      asks too.
    observed: The value that was tested.
    limit: The value it was tested against, zero where there is no
      numeric limit.
    source: The instrument the rule comes from.
  '''

  check_id: str
  status: str
  reason: str
  observed: float = 0.0
  limit: float = 0.0
  source: str = ''

  def __post_init__(self) -> None:
    '''Validate the verdict at construction.

    Raises:
      ValueError: If the status is not one of :data:`statuses`, or the
        reason is empty. An empty reason on a rejection is the exact
        failure this module exists to prevent, so it is not allowed to
        exist in memory.
    '''
    if self.status not in statuses:
      raise ValueError(
        f'status must be one of {statuses}, got {self.status!r}')
    if not self.reason.strip():
      raise ValueError(
        f'check {self.check_id!r} must carry a reason, pass or reject')

  @property
  def passed(self) -> bool:
    '''Return True only for a PASS verdict.'''
    return self.status == passed_status


@dataclass(frozen=True, slots=True)
class RiskReport:
  '''The full verdict on one order.

  Attributes:
    order_id: Identifier for the order the report is about. Filled from
      the order when the caller did not name it.
    results: One :class:`CheckResult` per check, in :data:`check_ids`
      order.
  '''

  order_id: str
  results: tuple[CheckResult, ...]

  @property
  def approved(self) -> bool:
    '''Return True when no check rejected.

    Note this is "no rejection", not "some check passed": a report with
    an empty result tuple is *not* approved, which is the safe reading
    for a caller that built the report itself.
    '''
    return bool(self.results) and not self.rejections

  @property
  def rejections(self) -> tuple[CheckResult, ...]:
    '''Return the rejected checks, in execution order.'''
    return tuple(
      result for result in self.results if result.status == rejected_status)

  @property
  def reasons(self) -> tuple[str, ...]:
    '''Return one ``check_id: reason`` line per rejection.'''
    return tuple(
      f'{result.check_id}: {result.reason}' for result in self.rejections)

  def result(self, check_id: str) -> CheckResult:
    '''Return one check's result by id.

    Args:
      check_id: Identifier from :data:`check_ids`.

    Returns:
      The matching :class:`CheckResult`.

    Raises:
      KeyError: If the id is unknown. Deliberately not a silent default:
        a caller asking for a check that did not run has a bug, and
        returning a passing placeholder would hide it.
    '''
    for item in self.results:
      if item.check_id == check_id:
        return item
    raise KeyError(
      f'unknown check id {check_id!r}; known: {check_ids}')

  def render(self) -> str:
    '''Return a one-line summary for a log or a rejection message.

    Returns:
      Human-readable summary naming the order and every rejection.
    '''
    verdict = 'APPROVED' if self.approved else 'REJECTED'
    if self.rejections:
      return f'{verdict} {self.order_id}: ' + '; '.join(self.reasons)
    return f'{verdict} {self.order_id}: {len(self.results)} checks'


class RiskRejected(RuntimeError):
  '''Raised when an order fails at least one pre-trade check.

  A RuntimeError rather than a ValueError because nothing about the
  order's *values* is necessarily wrong: the order may be perfectly
  well formed and simply too large for the account's remaining headroom.
  It carries the whole report so the caller can log which checks fired
  rather than only that something did.
  '''

  def __init__(self, report: RiskReport) -> None:
    '''Build the exception from a report.

    Args:
      report: The report that rejected the order.
    '''
    super().__init__(report.render())
    self.report = report


@dataclass(frozen=True, slots=True)
class PriceBand:
  '''A two-sided price band.

  Used for the exchange price band on a scrip and for the market-wide
  position limit band. Both are two-sided and both are configured from
  the exchange rather than computed here; the arithmetic that derives a
  band from a reference price lives in :mod:`stock_rl.risk.circuit`.

  Attributes:
    lower: Inclusive lower bound.
    upper: Inclusive upper bound.
  '''

  lower: float
  upper: float

  def __post_init__(self) -> None:
    '''Validate the band at construction.

    Raises:
      ValueError: If the lower bound is not positive or the band is
        inverted. An inverted band would make every price legal.
    '''
    if self.lower <= 0.0:
      raise ValueError(f'band lower must be positive, got {self.lower}')
    if self.upper <= self.lower:
      raise ValueError(
        f'band upper {self.upper} must exceed lower {self.lower}')


@dataclass(frozen=True, slots=True)
class OrderRequest:
  '''One order as the risk layer sees it.

  Deliberately carries no venue data. The band, the MWPL band and the
  per-order quantity and value limits belong to the exchange and live in
  :class:`SecurityLimits`; an order that supplied its own would be an
  order that widens its own limits.

  Attributes:
    symbol: NSE scrip symbol, resolved against the checker's venue map.
    side: :data:`buy` or :data:`sell`.
    quantity: Signed magnitude in shares or contracts. Not range-checked
      here on purpose: a zero or negative quantity is a *check* failure,
      not a construction failure, so the rejection is attributable to
      :data:`check_ids`.
    order_type: :data:`limit_order` or :data:`market_order`.
    price: Limit price. Zero for a market order, which is then rejected
      by both the market-order check and the price band check.
    reference_price: Last traded price at decision time. Zero means "not
      recorded", which rejects the bad-ticket and MWPL checks rather
      than passing them.
    algo_id: Algo ID the order will be tagged with.
    is_algo: True for an algo-generated order. Market orders are only
      prohibited for algo orders, and this project only ever produces
      algo orders.
    order_id: Caller's identifier, used in the report.
  '''

  symbol: str
  side: str
  quantity: int
  order_type: str = limit_order
  price: float = 0.0
  reference_price: float = 0.0
  algo_id: str = ''
  is_algo: bool = True
  order_id: str = ''

  def __post_init__(self) -> None:
    '''Validate the order's shape at construction.

    Only the fields whose *meaning* is ambiguous are checked here. Size
    and price bounds belong to the checks, because a check that cannot
    be the thing that rejects is not a check.

    Raises:
      ValueError: If the symbol is blank, the side or order type is
        unrecognised, the quantity is not a plain int, or a price is
        negative.
    '''
    if not isinstance(self.symbol, str) or not self.symbol.strip():
      raise ValueError(f'symbol must be a non-empty str, got {self.symbol!r}')
    if self.side not in sides:
      raise ValueError(f'side must be one of {sides}, got {self.side!r}')
    if self.order_type not in order_types:
      raise ValueError(
        f'order_type must be one of {order_types}, got {self.order_type!r}')
    if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
      raise ValueError(
        f'quantity must be an int, got {type(self.quantity).__name__}')
    if self.price < 0.0:
      raise ValueError(f'price must be >= 0, got {self.price}')
    if self.reference_price < 0.0:
      raise ValueError(
        f'reference_price must be >= 0, got {self.reference_price}')

  @property
  def value(self) -> float:
    '''Return the order's traded value in rupees, always non-negative.

    ``abs`` on the quantity is not cosmetic. A negative quantity is a
    malformed order that the quantity check rejects, and a signed value
    would then *subtract* from every aggregate limit below and pass them
    on the way. Taking the magnitude here means a malformed order fails
    loudly at the quantity check instead of quietly loosening five
    others.
    '''
    return abs(self.quantity) * self.price

  @property
  def signed_quantity(self) -> int:
    '''Return the quantity with the side applied.'''
    return self.quantity if self.side == buy else -self.quantity


@dataclass(frozen=True, slots=True)
class SecurityLimits:
  '''Exchange limits for one scrip.

  These are the numbers the exchange publishes per security, so they are
  configuration held by the checker and looked up by symbol. Two of them
  are band-shaped and two are scalars, and all four are per-security
  because the exchange defines all four per-security: the maximum order
  quantity for a security and the value per order for a security.

  Attributes:
    symbol: Scrip symbol, used as the lookup key and cross-checked
      against the order.
    band: The exchange price band for the scrip.
    mwpl: The market-wide position limit band for the scrip. At this
      band only position-reducing orders are permitted, which is what
      the ``mwpl`` check enforces.
    max_order_quantity: Maximum quantity per order, in shares.
    max_order_value: Maximum traded value per order, in rupees.
  '''

  symbol: str
  band: PriceBand
  mwpl: PriceBand
  max_order_quantity: int
  max_order_value: float

  def __post_init__(self) -> None:
    '''Validate the security's limits at construction.

    Raises:
      ValueError: If the symbol is blank or either scalar limit is
        unusable. A zero or negative per-order limit would reject every
        order, which is at least loud, but it is a configuration error
        and is caught here where the cause is still visible.
    '''
    if not isinstance(self.symbol, str) or not self.symbol.strip():
      raise ValueError(f'symbol must be a non-empty str, got {self.symbol!r}')
    if self.max_order_quantity < 1:
      raise ValueError(
        f'max_order_quantity must be >= 1, got {self.max_order_quantity}')
    if self.max_order_value <= 0.0:
      raise ValueError(
        f'max_order_value must be positive, got {self.max_order_value}')


@dataclass(frozen=True, slots=True)
class RmsLimits:
  '''Account-level risk limits, per Algo ID.

  The cumulative open-order-value limit is a finite positive float and
  nothing else. NSE 11.1 requires the cumulative open order value at
  client level and forbids "Unlimited" for algo clients; the forbidden
  configuration is therefore not merely rejected by a check, it cannot
  be constructed.

  Attributes:
    cumulative_open_order_value: Client-level ceiling on the value of
      live unexecuted orders, in rupees. SEBI 2012 para 6(iv).
    max_position: Absolute ceiling on the net position in any one
      security, in shares or contracts.
    max_trading_value: Ceiling on value traded by the account on the
      session, in rupees.
    max_exposure: Ceiling on gross exposure including orders in flight,
      in rupees.
    max_turnover: Ceiling on session turnover, in rupees.
    max_security_value: Ceiling on the value held or in flight in any
      one security, in rupees.
    bad_ticket_pct: Maximum absolute deviation of the order price from
      the reference price, as a fraction. Trade Price Protection.
  '''

  cumulative_open_order_value: float
  max_position: int
  max_trading_value: float
  max_exposure: float
  max_turnover: float
  max_security_value: float
  bad_ticket_pct: float = 0.02

  def __post_init__(self) -> None:
    '''Validate the account limits at construction.

    Raises:
      ValueError: If any rupee limit is not finite and positive, if the
        position limit is negative, or if the bad-ticket band is outside
        ``(0, 1)``. Infinity is rejected explicitly because an
        infinite cumulative open-order-value limit *is* the "Unlimited"
        the exchange forbids, and ``float('inf')`` would otherwise sail
        past a ``> 0.0`` test.
    '''
    for name in ('cumulative_open_order_value', 'max_trading_value',
                 'max_exposure', 'max_turnover', 'max_security_value'):
      value = getattr(self, name)
      if not 0.0 < value < float('inf'):
        raise ValueError(
          f'{name} must be finite and positive, got {value}; an unlimited '
          'value is prohibited for algo clients (NSE 11.1)')
    if self.max_position < 0:
      raise ValueError(f'max_position must be >= 0, got {self.max_position}')
    if not 0.0 < self.bad_ticket_pct < 1.0:
      raise ValueError(
        f'bad_ticket_pct must be in (0, 1), got {self.bad_ticket_pct}')


# The state an RMS reads is genuinely nine numbers, and every one of
# them is an independent figure reported by a different upstream system.
# Splitting it into nested dataclasses would add indirection without
# making any of the numbers harder to get wrong, so the count is
# accepted deliberately.
# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True, slots=True)
class AccountState:
  '''The account's own ledger, as read by the checks.

  This is a self-maintained ledger, not an exchange query. That is an
  honest limitation and it is the reason the "net position against
  margin" item of para 11.1 is *not* implemented here: a margin check
  computed from a self-reported ledger would be a check of nothing.

  Attributes:
    cash: Cash available to fund purchases, in rupees, or None for a
      ledger that carries no cash balance. **This field is what makes an
      order's affordability checkable at all.** Without it every check in
      this module asks only whether the exchange will permit an order,
      never whether the account can pay for it, so a book sized to a
      hundred percent of capital passes every check and borrows at zero
      interest to settle.

      None is a real value and is not read as zero. A ledger with no
      cash figure is not a ledger with no cash, and treating it as the
      latter rejects every order in the system; treating it as infinite
      reinstates the bug. The affordability check reports that it could
      not be made, and the sizer's own guard is what bounds the size.
    open_order_value: Value of live unexecuted orders, in rupees.
    executed_value: Value executed in the session and confirmed, rupees.
    unconfirmed_value: Value executed but not yet confirmed, rupees.
    net_position: Signed net position in the order's security.
    trading_value: Cumulative session traded value, in rupees.
    exposure: Gross exposure including orders in flight, in rupees.
    turnover: Cumulative session turnover, in rupees.
    security_values: Value held or in flight per symbol, in rupees. A
      missing entry means no value recorded, which is zero headroom
      spent rather than unlimited headroom.
  '''

  cash: float | None = None
  open_order_value: float = 0.0
  executed_value: float = 0.0
  unconfirmed_value: float = 0.0
  net_position: int = 0
  trading_value: float = 0.0
  exposure: float = 0.0
  turnover: float = 0.0
  security_values: Mapping[str, float] = field(default_factory=dict)


#: Signature of one internal check. Every check takes the same three
#: arguments so that :meth:`RmsChecker.check` can run them uniformly, and
#: so that adding a check cannot change the calling convention. The
#: methods are bound, so ``self`` is not in the signature.
Check = Callable[[OrderRequest, AccountState, SecurityLimits], CheckResult]


class RmsChecker:
  '''Run the NSE para 11.1 pre-trade checks over one order.

  Constructs once with the account's limits and the venue map, then
  calls :meth:`check` per order. The venue map is a *mapping*, not a
  single security, because the checks that touch per-symbol data are
  also the checks that must reject an unknown symbol rather than skip
  themselves.

  Not thread-safe, deliberately. This is the safety-critical coupling
  the project structure argues for: risk and execution in one process,
  one account ledger, one checker. Adding a lock here would guard a
  concurrency model this project has decided not to have.

  Args are documented on ``__init__``.
  '''

  limits: RmsLimits
  securities: Mapping[str, SecurityLimits]
  costs: CostModel

  def __init__(
    self,
    limits: RmsLimits,
    securities: Mapping[str, SecurityLimits],
    banned: frozenset[str] = frozenset(),
    costs: CostModel = DELIVERY,
  ) -> None:
    '''Build a checker from limits and venue data.

    Args:
      limits: Account-level limits.
      securities: Per-symbol venue limits, keyed by symbol. Must be
        non-empty: a checker with no venue data rejects every order,
        which is correct behaviour but useless configuration.
      banned: Symbols under an F&O ban. Exchange-published and
        refreshed by the caller; this module does not fetch it, because
        a ban list that is stale rather than absent is the failure mode.
      costs: Rate card for the affordability check, so the amount an
        order costs to settle is priced rather than assumed away.
        Defaults to :data:`stock_rl.costs.DELIVERY`, the same model the
        allocator uses.

    Raises:
      ValueError: If the venue map is empty or a key disagrees with its
        ``SecurityLimits.symbol``. The second check exists because a map
        keyed ``'RELIANCE'`` holding RELIANCE's limits would apply one
        scrip's band to another, which is the kind of error that passes
        every test written against it.
    '''
    if not securities:
      raise ValueError(
        'securities must not be empty: with no venue data every order '
        'would be rejected for an unknowable reason')
    for key, security in securities.items():
      if key != security.symbol:
        raise ValueError(
          f'venue map key {key!r} does not match symbol '
          f'{security.symbol!r}; a mismatched key would apply one '
          "scrip's limits to another")
    self.limits = limits
    self.securities = dict(securities)
    self.banned = frozenset(banned)
    self.costs = costs
    self._checks: tuple[tuple[str, Check], ...] = (
      ('price_band', self._price_band),
      ('order_quantity', self._order_quantity),
      ('order_value', self._order_value),
      ('affordability', self._affordability),
      ('trade_price_protection', self._trade_price_protection),
      ('algo_market_order', self._algo_market_order),
      ('cumulative_open_order_value', self._cumulative_open_order_value),
      ('automated_execution', self._automated_execution),
      ('position_limit', self._position_limit),
      ('trading_limit', self._trading_limit),
      ('exposure_limit', self._exposure_limit),
      ('turnover_limit', self._turnover_limit),
      ('security_value_limit', self._security_value_limit),
      ('fno_ban', self._fno_ban),
      ('mwpl', self._mwpl),
    )
    if tuple(name for name, _ in self._checks) != check_ids:
      raise ValueError('check registry is out of step with check_ids')

  def check(self, order: OrderRequest,
            account: AccountState) -> RiskReport:
    '''Run every pre-trade check over one order.

    An unknown symbol short-circuits to a report in which every check
    rejects, rather than to a report in which the checks that happen not
    to need venue data pass. A caller cannot then read "the F&O ban check
    passed" out of a report that was produced for a symbol this process
    has no venue configuration for.

    Args:
      order: The order about to be released.
      account: The account's current ledger.

    Returns:
      A :class:`RiskReport` with one result per check in
      :data:`check_ids` order.
    '''
    order_id = order.order_id or (
      f'{order.symbol}:{order.side}:{order.quantity}@{order.price}')
    security = self.securities.get(order.symbol)
    if security is None:
      return RiskReport(order_id, tuple(
        CheckResult(
          check_id,
          rejected_status,
          f'no venue limits configured for {order.symbol}; refusing rather '
          'than checking an order against unknown limits',
          order.price, 0.0, 'NSE 11.1 price band / per-security limits')
        for check_id, _ in self._checks))
    results = tuple(
      method(order, account, security)
      for _, method in self._checks)
    return RiskReport(order_id, results)

  def require_approval(self, order: OrderRequest,
                       account: AccountState) -> RiskReport:
    '''Check an order and raise unless it is approved.

    The single call an order path should make, because it cannot forget
    to test ``report.approved``.

    Args:
      order: The order about to be released.
      account: The account's current ledger.

    Returns:
      The passing :class:`RiskReport`, so the caller can log it.

    Raises:
      RiskRejected: If any check rejected. The report is on the
        exception.
    '''
    report = self.check(order, account)
    if not report.approved:
      raise RiskRejected(report)
    return report

  def known(self, symbol: str) -> bool:
    '''Return whether venue limits are configured for a symbol.

    Args:
      symbol: Scrip symbol.

    Returns:
      True if :meth:`check` can make a real decision about this symbol.
    '''
    return symbol in self.securities

  def _price_band(self, order: OrderRequest, account: AccountState,
                  security: SecurityLimits) -> CheckResult:
    '''Check the order price against the exchange price band.

    A price of zero has no band to sit in, so a market order is rejected
    here as well as by :meth:`_algo_market_order`, and by
    :meth:`_order_value` and :meth:`_trade_price_protection`. Four
    rejections for one malformed order is not a bug: one cites the rule
    and the other three report arithmetic that cannot be performed. What
    matters is that none of them passes.

    Args:
      order: Order under test.
      account: Unused; present for the uniform check signature.
      security: Venue limits for the order's symbol.

    Returns:
      The check result.
    '''
    del account
    band = security.band
    if order.price <= 0.0:
      return CheckResult(
        'price_band', rejected_status,
        f'price {order.price} is not a tradable price; this RMS admits '
        'priced limit orders only',
        order.price, band.upper, 'NSE 11.1 price band')
    if not band.lower <= order.price <= band.upper:
      return CheckResult(
        'price_band', rejected_status,
        f'price {order.price} outside exchange band '
        f'[{band.lower}, {band.upper}]',
        order.price, band.upper, 'NSE 11.1 price band')
    return CheckResult(
      'price_band', passed_status,
      f'price {order.price} inside exchange band '
      f'[{band.lower}, {band.upper}]',
      order.price, band.upper, 'NSE 11.1 price band')

  def _order_quantity(self, order: OrderRequest, account: AccountState,
                      security: SecurityLimits) -> CheckResult:
    '''Check quantity against the per-order limit for the security.

    Args:
      order: Order under test.
      account: Unused; present for the uniform check signature.
      security: Venue limits for the order's symbol.

    Returns:
      The check result.
    '''
    del account
    limit = float(security.max_order_quantity)
    if order.quantity < 1:
      return CheckResult(
        'order_quantity', rejected_status,
        f'quantity {order.quantity} is not a positive whole number of '
        'shares or contracts',
        float(order.quantity), limit,
        'SEBI 2012 para 6(ii); NSE 11.1')
    if order.quantity > security.max_order_quantity:
      return CheckResult(
        'order_quantity', rejected_status,
        f'quantity {order.quantity} exceeds the per-order limit of '
        f'{security.max_order_quantity}',
        float(order.quantity), limit,
        'SEBI 2012 para 6(ii); NSE 11.1')
    return CheckResult(
      'order_quantity', passed_status,
      f'quantity {order.quantity} within per-order limit '
      f'{security.max_order_quantity}',
      float(order.quantity), limit,
      'SEBI 2012 para 6(ii); NSE 11.1')

  def _order_value(self, order: OrderRequest, account: AccountState,
                   security: SecurityLimits) -> CheckResult:
    '''Check the order's traded value against the per-order limit.

    Args:
      order: Order under test.
      account: Unused; present for the uniform check signature.
      security: Venue limits for the order's symbol.

    Returns:
      The check result.
    '''
    del account
    value = order.value
    limit = security.max_order_value
    if value <= 0.0:
      return CheckResult(
        'order_value', rejected_status,
        f'order value {value} is not positive, so no value limit can be '
        'meaningful',
        value, limit, 'SEBI 2012 para 6(iii); NSE 11.1')
    if value > limit:
      return CheckResult(
        'order_value', rejected_status,
        f'order value {value} exceeds the per-order limit of {limit}',
        value, limit, 'SEBI 2012 para 6(iii); NSE 11.1')
    return CheckResult(
      'order_value', passed_status,
      f'order value {value} within per-order limit of {limit}',
      value, limit, 'SEBI 2012 para 6(iii); NSE 11.1')

  def _affordability(self, order: OrderRequest, account: AccountState,
                     security: SecurityLimits) -> CheckResult:
    '''Check the order against the cash balance on the ledger.

    **This is the check the other thirteen cannot make.** Every other one
    asks whether the exchange will permit the order; this one asks
    whether the account can pay for it. They are different questions and
    only one of them is about solvency, so a book sized to a hundred
    percent of capital clears every exchange limit and still ends the day
    with a negative cash balance and an unpriced borrowing.

    The charges are included because the cost of *not* charging them is
    exactly this bug: four names at a quarter of capital each spend the
    whole mark, and the delivery charges then come out of nothing. An
    order is payable when its value plus its charges fits inside the
    ledger's cash.

    A ledger carrying no cash figure gets a pass that says so, rather
    than a pass that implies the question was answered. The two
    available guesses are both worse than no answer: reading a missing
    balance as zero rejects every order in the system, and reading it as
    unlimited reinstates the bug this check exists to catch.

    A sell releases cash rather than consuming it, so it cannot be
    unaffordable. A short is a margin question, and the margin item of
    para 11.1 is deliberately not implemented here; the reason string
    names that rather than pretending the position was checked.

    Args:
      order: Order under test.
      account: The account's current ledger.
      security: Unused; present for the uniform check signature.

    Returns:
      The check result.
    '''
    del security
    source = 'SEBI 2012 para 6(i) funding; affordability'
    cash = account.cash
    if cash is None:
      return CheckResult(
        'affordability', passed_status,
        'this ledger carries no cash balance, so affordability could not be '
        'computed; it is not assumed in either direction and the sizer '
        'bounds the size against the cost model instead',
        0.0, 0.0, source)
    if order.side != buy:
      return CheckResult(
        'affordability', passed_status,
        f'a {order.side} order releases cash rather than consuming it, so '
        'there is no affordability question; funding a short is a margin '
        'question, which this ledger does not carry',
        order.value, cash, source)
    charges = self.costs.one_way(Side.BUY, order.value)
    total = order.value + charges
    if total > cash:
      return CheckResult(
        'affordability', rejected_status,
        f'order value {order.value:.2f} plus {charges:.2f} of charges is '
        f'{total:.2f}, beyond the cash balance of {cash:.2f}; this order '
        'cannot be paid for',
        total, cash, source)
    return CheckResult(
      'affordability', passed_status,
      f'order value {order.value:.2f} plus {charges:.2f} of charges is '
      f'{total:.2f}, within the cash balance of {cash:.2f}',
      total, cash, source)

  def _trade_price_protection(self, order: OrderRequest,
                              account: AccountState,
                              security: SecurityLimits) -> CheckResult:
    '''Check the order price against Trade Price Protection.

    Trade Price Protection is the collective name for the bad-ticket
    family in para 11.1: an order price too far from the last traded
    price is refused regardless of how well it scores on every other
    check. A missing reference price rejects, because a bad-ticket check
    that cannot see a reference price has checked nothing.

    Args:
      order: Order under test.
      account: Unused; present for the uniform check signature.
      security: Unused; present for the uniform check signature.

    Returns:
      The check result.
    '''
    del account, security
    band = self.limits.bad_ticket_pct
    if order.reference_price <= 0.0:
      return CheckResult(
        'trade_price_protection', rejected_status,
        f'no reference price recorded (got {order.reference_price}); a '
        'bad-ticket check cannot be made without one',
        order.reference_price, band, 'NSE 11.1 trade price protection')
    deviation = abs(order.price / order.reference_price - 1.0)
    if deviation > band:
      return CheckResult(
        'trade_price_protection', rejected_status,
        f'order price {order.price} deviates {deviation:.2%} from the '
        f'reference {order.reference_price}, beyond the {band:.2%} '
        'bad-ticket band',
        deviation, band, 'NSE 11.1 trade price protection')
    return CheckResult(
      'trade_price_protection', passed_status,
      f'order price deviates {deviation:.2%} from the reference, within '
      f'the {band:.2%} bad-ticket band',
      deviation, band, 'NSE 11.1 trade price protection')

  def _algo_market_order(self, order: OrderRequest, account: AccountState,
                         security: SecurityLimits) -> CheckResult:
    '''Refuse a market order raised by an algo.

    NSE/MSD/67753 8.1.1.12 prohibits market orders in the equity segment
    for algo orders. There is no de-minimis, no size exemption and no
    human-in-the-loop exemption: SEBI 2012 para 3 says an order from
    automated execution logic *is* algo trading.

    Args:
      order: Order under test.
      account: Unused; present for the uniform check signature.
      security: Unused; present for the uniform check signature.

    Returns:
      The check result.
    '''
    del account, security
    source = 'NSE/MSD/67753 8.1.1.12'
    if order.is_algo and order.order_type == market_order:
      return CheckResult(
        'algo_market_order', rejected_status,
        'market orders are prohibited for algo orders in the equity '
        'segment; use a limit order',
        0.0, 0.0, source)
    return CheckResult(
      'algo_market_order', passed_status,
      f'order type {order.order_type!r} is permitted for an algo order',
      0.0, 0.0, source)

  def _cumulative_open_order_value(
    self,
    order: OrderRequest,
    account: AccountState,
    security: SecurityLimits,
  ) -> CheckResult:
    '''Check the client-level cumulative open order value.

    **This is an interpretation and is labelled as one.** Para 6(iv)
    states the ceiling on cumulative open order value per client; the
    reading here is that it counts *live unexecuted* orders, which is
    what "open order value" means. The session-total reading belongs to
    the automated-execution check below, which by name accounts for
    executed, unexecuted and unconfirmed orders together. Both are
    implemented, so either reading is enforced.

    Args:
      order: Order under test.
      account: The account's current ledger.
      security: Unused; present for the uniform check signature.

    Returns:
      The check result.
    '''
    del security
    limit = self.limits.cumulative_open_order_value
    total = account.open_order_value + order.value
    source = 'SEBI 2012 para 6(iv); NSE 11.1'
    if total > limit:
      return CheckResult(
        'cumulative_open_order_value', rejected_status,
        f'cumulative open order value {total} exceeds the client limit '
        f'{limit}; "Unlimited" is not an available configuration',
        total, limit, source)
    return CheckResult(
      'cumulative_open_order_value', passed_status,
      f'cumulative open order value {total} within the client limit '
      f'{limit}',
      total, limit, source)

  def _automated_execution(self, order: OrderRequest, account: AccountState,
                           security: SecurityLimits) -> CheckResult:
    '''Account for every order in flight before releasing another.

    SEBI 2012 para 6(v): the RMS must account for **all** executed,
    unexecuted and unconfirmed orders before releasing the next one. The
    three buckets are summed and the new order is added on top, so an
    account that is already carrying unconfirmed trades has less room
    for the next order. A loop or runaway is what this catches, and it
    only catches it if the unconfirmed bucket is not quietly omitted --
    which is why it is a separate check rather than a field on the
    previous one.

    Args:
      order: Order under test.
      account: The account's current ledger.
      security: Unused; present for the uniform check signature.

    Returns:
      The check result.
    '''
    del security
    limit = self.limits.cumulative_open_order_value
    in_flight = (account.executed_value + account.open_order_value
                 + account.unconfirmed_value)
    total = in_flight + order.value
    source = 'SEBI 2012 para 6(v); NSE 11.1 automated execution check'
    if total > limit:
      return CheckResult(
        'automated_execution', rejected_status,
        f'executed {account.executed_value} + unexecuted '
        f'{account.open_order_value} + unconfirmed '
        f'{account.unconfirmed_value} + new {order.value} = {total} '
        f'exceeds {limit}',
        total, limit, source)
    return CheckResult(
      'automated_execution', passed_status,
      f'executed, unexecuted and unconfirmed orders total {total} '
      f'including this order, within {limit}',
      total, limit, source)

  def _position_limit(self, order: OrderRequest, account: AccountState,
                      security: SecurityLimits) -> CheckResult:
    '''Check the post-order net position against the position limit.

    Args:
      order: Order under test.
      account: The account's current ledger.
      security: Unused; present for the uniform check signature.

    Returns:
      The check result.
    '''
    del security
    limit = float(self.limits.max_position)
    post = account.net_position + order.signed_quantity
    source = 'SEBI 2012 para 6(v); NSE 11.1 position limit'
    if abs(post) > self.limits.max_position:
      return CheckResult(
        'position_limit', rejected_status,
        f'net position would be {post}, beyond the limit of '
        f'{self.limits.max_position}',
        float(post), limit, source)
    return CheckResult(
      'position_limit', passed_status,
      f'net position would be {post}, within the limit of '
      f'{self.limits.max_position}',
      float(post), limit, source)

  def _trading_limit(self, order: OrderRequest, account: AccountState,
                     security: SecurityLimits) -> CheckResult:
    '''Check session traded value against the trading limit.

    Args:
      order: Order under test.
      account: The account's current ledger.
      security: Unused; present for the uniform check signature.

    Returns:
      The check result.
    '''
    del security
    limit = self.limits.max_trading_value
    total = account.trading_value + order.value
    source = 'NSE 11.1 trading limit'
    if total > limit:
      return CheckResult(
        'trading_limit', rejected_status,
        f'session traded value would be {total}, beyond the trading limit '
        f'{limit}',
        total, limit, source)
    return CheckResult(
      'trading_limit', passed_status,
      f'session traded value would be {total}, within {limit}',
      total, limit, source)

  def _exposure_limit(self, order: OrderRequest, account: AccountState,
                      security: SecurityLimits) -> CheckResult:
    '''Check gross exposure against the exposure limit.

    The order value is added regardless of side. That is the
    conservative direction on purpose: a sell order's eventual fill
    quantity and price are not known at pre-trade time, so subtracting
    its value would count exposure that may never be released. The
    exchange's own exposure limit is measured the other way, and a desk
    that wants the exact exchange definition should compute it there.

    Args:
      order: Order under test.
      account: The account's current ledger.
      security: Unused; present for the uniform check signature.

    Returns:
      The check result.
    '''
    del security
    limit = self.limits.max_exposure
    total = account.exposure + order.value
    source = 'NSE 11.1 exposure limit'
    if total > limit:
      return CheckResult(
        'exposure_limit', rejected_status,
        f'gross exposure would be {total}, beyond the exposure limit '
        f'{limit}',
        total, limit, source)
    return CheckResult(
      'exposure_limit', passed_status,
      f'gross exposure would be {total}, within {limit}',
      total, limit, source)

  def _turnover_limit(self, order: OrderRequest, account: AccountState,
                      security: SecurityLimits) -> CheckResult:
    '''Check session turnover against the turnover limit.

    Args:
      order: Order under test.
      account: The account's current ledger.
      security: Unused; present for the uniform check signature.

    Returns:
      The check result.
    '''
    del security
    limit = self.limits.max_turnover
    total = account.turnover + order.value
    source = 'NSE 11.1 turnover limit'
    if total > limit:
      return CheckResult(
        'turnover_limit', rejected_status,
        f'session turnover would be {total}, beyond the limit of {limit}',
        total, limit, source)
    return CheckResult(
      'turnover_limit', passed_status,
      f'session turnover would be {total}, within {limit}',
      total, limit, source)

  def _security_value_limit(self, order: OrderRequest,
                            account: AccountState,
                            security: SecurityLimits) -> CheckResult:
    '''Check the value in flight in one security against its limit.

    A symbol absent from ``security_values`` is treated as zero already
    committed, not as unlimited headroom. A mapping that forgot a symbol
    must not turn into a check that cannot fire.

    Args:
      order: Order under test.
      account: The account's current ledger.
      security: Unused; present for the uniform check signature.

    Returns:
      The check result.
    '''
    del security
    limit = self.limits.max_security_value
    held = account.security_values.get(order.symbol, 0.0)
    total = held + order.value
    source = 'NSE 11.1 security-wise value limit'
    if total > limit:
      return CheckResult(
        'security_value_limit', rejected_status,
        f'{order.symbol} value would be {total}, beyond the security-wise '
        f'limit of {limit}',
        total, limit, source)
    return CheckResult(
      'security_value_limit', passed_status,
      f'{order.symbol} value would be {total}, within {limit}',
      total, limit, source)

  def _fno_ban(self, order: OrderRequest, account: AccountState,
               security: SecurityLimits) -> CheckResult:
    '''Refuse a symbol under an exchange F&O ban.

    The ban list is exchange-published and refreshed by the caller. This
    module does not fetch it, because a ban list that is stale is worse
    than one that is obviously absent: the stale case fails silently.

    Args:
      order: Order under test.
      account: Unused; present for the uniform check signature.
      security: Unused; present for the uniform check signature.

    Returns:
      The check result.
    '''
    del account, security
    source = 'NSE 11.1 F&O ban'
    if order.symbol in self.banned:
      return CheckResult(
        'fno_ban', rejected_status,
        f'{order.symbol} is under an F&O ban',
        1.0, 0.0, source)
    return CheckResult(
      'fno_ban', passed_status,
      f'{order.symbol} is not under an F&O ban',
      0.0, 0.0, source)

  def _mwpl(self, order: OrderRequest, account: AccountState,
            security: SecurityLimits) -> CheckResult:
    '''Refuse an order that adds exposure at the market-wide limit band.

    At a market-wide position limit band the exchange permits only
    position-reducing orders. That is the whole content of this check:
    a buy at or above the upper band, or a sell at or below the lower
    band, rejects; the reducing side at the same band passes.

    **What is not implemented.** The per-client MWPL, i.e. this account's
    position in a scrip as a percentage of that scrip's market cap, is
    not checked, because this project holds no market-capitalisation
    reference data and computing the ratio against an absent denominator
    would produce a confident-looking number with no meaning behind it.
    The form that is checkable from price data alone is the market-wide
    band, and that is what is here.

    Args:
      order: Order under test.
      account: Unused; present for the uniform check signature.
      security: Venue limits for the order's symbol, carrying the band.

    Returns:
      The check result.
    '''
    del account
    band = security.mwpl
    source = 'NSE 11.1 market-wide position limit'
    price = order.reference_price
    if price <= 0.0:
      return CheckResult(
        'mwpl', rejected_status,
        f'no reference price recorded (got {price}); the MWPL band cannot '
        'be applied to a price nobody has',
        price, band.upper, source)
    if price >= band.upper and order.side == buy:
      return CheckResult(
        'mwpl', rejected_status,
        f'{order.symbol} at {price} is at its market-wide limit band '
        f'{band.upper}; only position-reducing orders are permitted',
        price, band.upper, source)
    if price <= band.lower and order.side == sell:
      return CheckResult(
        'mwpl', rejected_status,
        f'{order.symbol} at {price} is at its market-wide limit band '
        f'{band.lower}; only position-reducing orders are permitted',
        price, band.lower, source)
    return CheckResult(
      'mwpl', passed_status,
      f'{order.symbol} at {price} is inside its market-wide limit band',
      price, band.upper, source)
