#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Broker abstraction with no broker behind it, on purpose.

**There is no real broker integration in this module, no API key, and no
network call of any kind.** That is a decision, not an omission, and the
reason is the compliance record rather than the engineering one:

  * SEBI 2012 para 5 and the Feb 2025 circular para 5(II)(a) both say
    the broker provides algo trading only after the *exchange's* prior
    permission for **each algo**. Running through a retail API such as
    Zerodha Kite on our own account is Client Direct API, and above the
    exchange's current order-rate threshold the algo must be registered
    through the broker first. This project has not been registered.
  * NSE para 9.1 and 9.9: no modification is allowed to a registered
    black-box algo, and fresh registration is required for **any** change
    to the logic governing it. A model whose logic is retrained nightly
    is a new algo every night as far as the exchange is concerned. Until
    that is resolved, an unauthenticated connector is a live trading
    system for a strategy nobody approved.

So the seam is declared, and the implementations are the two honest ones:
:class:`PaperBroker`, which records what would have been sent and never
leaves the process, and :class:`NullBroker`, which refuses everything and
exists so that "no broker configured" is a *loud* state rather than an
``AttributeError`` three layers up.

Orders arriving here have already cleared :mod:`stock_rl.risk.checks`.
That ordering is the point of the project structure: risk and execution
in one process, so a partition between them cannot fail open.

:class:`FailoverRouter` models multi-broker failover entirely in process.
A real failover scheme is not clever: try the next venue, mark the failed
one unhealthy, and refuse everything when all are down. The subtle part
is that a venue must not be retried forever inside one order, and that an
all-down answer must be a *rejection carrying the reason* rather than an
exception, because the caller's next action is to stop, and a stack trace
three frames up is not a stop.

Every failure a broker can raise is a :class:`BrokerError`. A real
adapter must translate vendor exceptions into it at the boundary; letting
a vendor's exception type escape would put the failover logic at the
mercy of somebody else's API version history.
'''

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from stock_rl.risk.checks import OrderRequest, buy, market_order

__all__ = [
  'Broker',
  'BrokerAck',
  'BrokerError',
  'BrokerHealth',
  'FailoverRouter',
  'NullBroker',
  'PaperBroker',
  'PaperOrder',
  'cancelled',
  'filled',
  'open_order',
  'order_statuses',
  'rejected',
]

#: The broker took the order and it is live.
open_order = 'open'

#: The broker matched the order in full.
filled = 'filled'

#: The broker refused the order.
rejected = 'rejected'

#: The broker cancelled a previously live order.
cancelled = 'cancelled'

#: Every status an acknowledgement may carry.
order_statuses = (open_order, filled, rejected, cancelled)


class BrokerError(RuntimeError):
  '''Raised when a broker call fails.

  One type for every failure, so the failover logic can catch it and so
  a real adapter has a single thing to translate into. A vendor exception
  escaping this boundary would put the retry policy at the mercy of
  somebody else's API version history.
  '''


@dataclass(frozen=True, slots=True)
class BrokerAck:
  '''The broker's answer to one order.

  Attributes:
    broker: Name of the broker that answered.
    order_id: Broker's identifier for the order. Empty on a rejection
      that never reached a matching engine, which is the normal case
      here.
    status: One of :data:`order_statuses`.
    reason: Why, for a rejection or a cancel. Populated on success too,
      so a log line is meaningful either way.
    quantity: Quantity requested.
    filled_quantity: Quantity matched.
    average_price: Average fill price, zero when nothing filled.
  '''

  broker: str
  status: str
  order_id: str = ''
  reason: str = ''
  quantity: int = 0
  filled_quantity: int = 0
  average_price: float = 0.0

  def __post_init__(self) -> None:
    '''Validate the acknowledgement at construction.

    **A fill is checked against the order it claims to fill.** Three
    properties, all of which a paper broker can produce by accident and a
    real adapter can produce by mis-mapping a field:

      * a fill carries a **positive quantity** -- an ack claiming a fill
        with zero quantity reads as a fill downstream and books nothing;
      * that quantity is **within the order**, because a broker cannot
        fill 999 shares of an order for 10 and an ack saying otherwise
        is either the wrong field or a venue that has lost track;
      * the fill carries a **positive price**, because a fill at 0.0
        books a position at zero value -- a position that exists, is
        counted in every exposure limit, and prices the book at nothing.

    The last two are the ones that were missing. Validating only the
    quantity is what let both through.

    The second and third need the *ordered* quantity to compare against,
    so they run only when ``quantity`` is positive. Zero means the venue
    did not state the order size, and there is nothing to check a fill
    against; refusing an acknowledgement for a field the venue never
    filled in would be refusing it for a missing reason.

    PONYTAIL: an acknowledgement is checked for internal consistency, not
    against the :class:`~stock_rl.risk.checks.OrderRequest` that was
    actually sent, because an ack is a value that crosses a process
    boundary and the dataclass carries no reference to the order. Ceiling:
    a venue that fills the right count at the wrong symbol is accepted.
    Upgrade path: a real adapter should re-check the ack against the
    order it holds in its own outbox, where the correspondence is known
    rather than inferred.

    Args are the fields; there are none.

    Raises:
      ValueError: If the status is unrecognised, the broker name is
        blank, or a filled acknowledgement is not a plausible fill of the
        order it names.
    '''
    if self.status not in order_statuses:
      raise ValueError(
        f'status must be one of {order_statuses}, got {self.status!r}')
    if not isinstance(self.broker, str) or not self.broker.strip():
      raise ValueError(f'broker name must not be blank, got {self.broker!r}')
    if self.status == filled and self.filled_quantity < 1:
      raise ValueError(
        'a filled acknowledgement must carry a positive filled quantity')
    # Both of the remaining checks are against *the order*, so they run
    # only when the ack declares one. ``quantity`` of zero means the venue
    # did not state the order size, and there is then nothing to check
    # the fill against -- inventing a bound from an undeclared field
    # would refuse acknowledgements for a missing reason rather than a
    # real one.
    if self.status == filled and self.quantity > 0:
      if self.filled_quantity > self.quantity:
        raise ValueError(
          f'filled_quantity {self.filled_quantity} exceeds the ordered '
          f'quantity {self.quantity}: a broker cannot fill more shares '
          'than it was asked for, and an acknowledgement that says so is '
          'either a mis-mapped field or a venue that has lost track of '
          'the order')
      if self.average_price <= 0.0:
        raise ValueError(
          'a filled acknowledgement must carry a positive average price, '
          f'got {self.average_price}; a fill at zero books a position at '
          'zero value, which counts in every exposure limit and prices '
          'the book at nothing')

  @property
  def ok(self) -> bool:
    '''Return whether the broker accepted the order.'''
    return self.status in (open_order, filled)

  def render(self) -> str:
    '''Return a one-line summary for a log.

    Returns:
      Human-readable summary.
    '''
    if self.ok:
      fill = (f' at {self.average_price}' if self.filled_quantity else '')
      return (
        f'{self.broker} {self.status} {self.order_id}: '
        f'{self.filled_quantity}/{self.quantity}{fill}')
    return f'{self.broker} {self.status}: {self.reason}'


@runtime_checkable
class Broker(Protocol):
  '''The order interface this project needs from a venue.

  Deliberately small. A vendor SDK exposes two hundred methods and a
  subscription model; declaring four here means the failover logic, the
  tests and the audit trail depend on this file and not on a vendor.

  Attributes:
    name: Broker identifier, carried into every acknowledgement.
  '''

  name: str

  def place(self, order: OrderRequest,
            market_price: float) -> BrokerAck:
    '''Send one order.

    Args:
      order: An order that has already cleared the pre-trade checks.
      market_price: Reference market price used to decide whether a
        limit order is marketable. Injected rather than fetched, because
        a paper broker that fetched it would need the network this
        project does not have.

    Returns:
      The broker's acknowledgement.
    '''

  def cancel(self, order_id: str) -> BrokerAck:
    '''Cancel a live order.

    Args:
      order_id: Identifier previously returned by :meth:`place`.

    Returns:
      The broker's acknowledgement.
    '''

  def positions(self) -> Mapping[str, int]:
    '''Return current net positions per symbol.

    Returns:
      Mapping of symbol to signed quantity.
    '''

  def is_healthy(self) -> bool:
    '''Return whether the broker is reachable and accepting orders.

    Returns:
      True when the broker would accept an order now.
    '''


@dataclass(frozen=True, slots=True)
class PaperOrder:
  '''One order as the paper broker recorded it.

  Attributes:
    order: The request as submitted.
    market_price: Market price the fill decision was made against.
    ack: What the broker answered.
  '''

  order: OrderRequest
  market_price: float
  ack: BrokerAck


class PaperBroker:
  '''An in-memory broker that records orders and never leaves the process.

  Fill model, chosen for determinism rather than realism: a buy limit at
  or above the market price fills immediately at the *market* price, a
  sell limit at or below it fills at the market price, and anything
  resting away from the market stays open. There is no queue, no partial
  fill and no randomness, so a test asserts a number instead of a
  distribution.

  Market orders are refused here as well as by the RMS. That is defence
  in depth on a rule that has no exceptions (NSE/MSD/67753
  8.1.1.12): a caller that bypasses the risk layer still cannot send one.

  Args are documented on ``__init__``.
  '''

  name: str
  orders: list[PaperOrder]

  def __init__(self, name: str = 'paper', healthy: bool = True) -> None:
    '''Build a paper broker.

    Args:
      name: Broker identifier.
      healthy: Initial health. A paper broker can be made unhealthy to
        exercise the failover path without inventing a second class.

    Raises:
      ValueError: If the name is blank. An unnamed broker produces
        acknowledgements nobody can attribute in a multi-broker log.
    '''
    if not isinstance(name, str) or not name.strip():
      raise ValueError(f'name must not be blank, got {name!r}')
    self.name = name
    self.healthy = healthy
    self.orders = []
    self._positions: dict[str, int] = {}
    self._live: dict[str, tuple[OrderRequest, float]] = {}
    self._cancelled: set[str] = set()
    self._sequence = 0
    self._faults = 0
    self._fault = 'injected fault'

  def inject_fault(self, count: int = 1, message: str = 'injected fault'
                   ) -> None:
    '''Make the next ``count`` calls fail.

    The deterministic way to exercise failover. Injecting the failure is
    the whole reason the project's own architecture note replaces chaos
    engineering with ``monkeypatch``: the thing being tested is the
    retry policy, and the retry policy does not care whether the outage
    came from a socket or from here.

    Args:
      count: Number of calls that should fail. Zero clears the fault.
      message: Message carried on the raised :class:`BrokerError`.

    Raises:
      ValueError: If the count is negative.
    '''
    if count < 0:
      raise ValueError(f'count must be >= 0, got {count}')
    self._faults = count
    self._fault = message

  def place(self, order: OrderRequest, market_price: float) -> BrokerAck:
    '''Record an order and decide a deterministic fill.

    Args:
      order: An order that has already cleared the pre-trade checks.
      market_price: Reference market price. Must be positive.

    Returns:
      The acknowledgement, always recorded in :attr:`orders` including
      for a refusal. A log of what was *tried* is worth more than a log
      of what succeeded.

    Raises:
      BrokerError: If a fault is injected.
      ValueError: If the market price is not positive.
    '''
    self._maybe_fail()
    if market_price <= 0.0:
      raise ValueError(f'market_price must be positive, got {market_price}')
    if order.order_type == market_order:
      ack = BrokerAck(
        broker=self.name, status=rejected, order_id='',
        reason=('market orders are prohibited for algo orders in the '
                'equity segment (NSE/MSD/67753 8.1.1.12)'),
        quantity=order.quantity)
      self.orders.append(PaperOrder(order, market_price, ack))
      return ack
    ack = self._fill(order, market_price)
    self.orders.append(PaperOrder(order, market_price, ack))
    if ack.status == open_order:
      self._live[ack.order_id] = (order, market_price)
    return ack

  def _fill(self, order: OrderRequest, market_price: float) -> BrokerAck:
    '''Return the acknowledgement the fill model implies.

    Args:
      order: The order to match.
      market_price: Reference market price.

    Returns:
      A filled or open acknowledgement.
    '''
    self._sequence += 1
    order_id = f'{self.name}-{self._sequence:06d}'
    marketable = (order.side == buy and order.price >= market_price) or \
                 (order.side != buy and order.price <= market_price)
    if not marketable:
      return BrokerAck(
        broker=self.name, status=open_order, order_id=order_id,
        reason=(f'limit {order.price} is away from the market '
                f'{market_price}'),
        quantity=order.quantity)
    signed = order.quantity if order.side == buy else -order.quantity
    self._positions[order.symbol] = (
      self._positions.get(order.symbol, 0) + signed)
    return BrokerAck(
      broker=self.name, status=filled, order_id=order_id,
      reason='marketable limit filled at the market price',
      quantity=order.quantity,
      filled_quantity=order.quantity,
      average_price=market_price)

  def cancel(self, order_id: str) -> BrokerAck:
    '''Cancel a recorded live order.

    A cancel appends its own record rather than editing the original
    entry. An order log that is edited in place cannot answer "what did
    this order actually look like when it was sent", and that question is
    the first one an inspection asks about it.

    Args:
      order_id: Identifier returned by :meth:`place`.

    Returns:
      A cancelled acknowledgement, or a rejection when the identifier is
      unknown or already cancelled.

    Raises:
      BrokerError: If a fault is injected.
    '''
    self._maybe_fail()
    live = self._live.pop(order_id, None)
    if live is None:
      if order_id in self._cancelled:
        return BrokerAck(
          broker=self.name, status=rejected, order_id=order_id,
          reason='order is already cancelled')
      return BrokerAck(
        broker=self.name, status=rejected, order_id=order_id,
        reason=f'no live order with id {order_id!r}')
    order, market_price = live
    self._cancelled.add(order_id)
    ack = BrokerAck(
      broker=self.name, status=cancelled, order_id=order_id,
      reason='cancelled by client', quantity=order.quantity)
    self.orders.append(PaperOrder(order, market_price, ack))
    return ack

  def positions(self) -> Mapping[str, int]:
    '''Return current net positions per symbol.

    Returns:
      A copy of the position mapping.
    '''
    return dict(self._positions)

  def is_healthy(self) -> bool:
    '''Return whether this broker would accept an order.

    Returns:
      True when healthy and no fault is pending.
    '''
    return self.healthy and self._faults == 0

  def _maybe_fail(self) -> None:
    '''Raise :class:`BrokerError` while an injected fault is pending.

    Raises:
      BrokerError: If a fault is pending. The counter is decremented
        first, so an injected count of N fails exactly N calls.
    '''
    if self._faults <= 0:
      return
    self._faults -= 1
    raise BrokerError(f'{self.name}: {self._fault}')


class NullBroker:
  '''A broker that refuses everything, so "unconfigured" is loud.

  Exists because the alternative is an unconfigured order path raising
  ``AttributeError`` somewhere unrelated, which is the state in which a
  paper backtest and a live run diverge. Here the divergence is a
  rejection with a reason, on the first order.

  Attributes:
    name: Broker identifier.
  '''

  name: str

  def __init__(self, name: str = 'null') -> None:
    '''Build the refusing broker.

    Args:
      name: Broker identifier.

    Raises:
      ValueError: If the name is blank.
    '''
    if not isinstance(name, str) or not name.strip():
      raise ValueError(f'name must not be blank, got {name!r}')
    self.name = name

  def place(self, order: OrderRequest,
            market_price: float) -> BrokerAck:
    '''Refuse an order.

    Args:
      order: The order that was attempted.
      market_price: Unused; there is no market to price against.

    Returns:
      A rejection naming the reason.
    '''
    del market_price
    return BrokerAck(
      broker=self.name, status=rejected,
      reason=('no broker is configured; this build has no broker '
              'integration, no credentials and no network path by design'),
      quantity=order.quantity)

  def cancel(self, order_id: str) -> BrokerAck:
    '''Refuse a cancellation.

    Args:
      order_id: Identifier that was never issued.

    Returns:
      A rejection naming the reason.
    '''
    return BrokerAck(
      broker=self.name, status=rejected, order_id=order_id,
      reason='no broker is configured, so no order exists to cancel')

  def positions(self) -> Mapping[str, int]:
    '''Return no positions.

    Returns:
      An empty mapping. Not a zero-position book: the broker holds
      nothing because it holds nothing at all.
    '''
    return {}

  def is_healthy(self) -> bool:
    '''Return True, because this broker is working exactly as designed.

    Returns:
      True. Marking it unhealthy would make a failover router skip it,
      which hides the real problem: it is not broken, it is refusing,
      and a refusal is the correct answer to every order.
    '''
    return True


@dataclass(frozen=True, slots=True)
class BrokerHealth:
  '''One broker's health as the router sees it.

  Attributes:
    name: Broker identifier.
    healthy: Whether the router will try this broker.
    failures: Consecutive failed calls.
    last_error: The most recent failure message, empty when healthy.
  '''

  name: str
  healthy: bool = True
  failures: int = 0
  last_error: str = ''

  def render(self) -> str:
    '''Return a one-line status for a log.

    Returns:
      Human-readable status.
    '''
    state = 'healthy' if self.healthy else 'UNHEALTHY'
    detail = f': {self.last_error}' if self.last_error else ''
    return f'{self.name}: {state} ({self.failures} failures){detail}'


class FailoverRouter:
  '''Try brokers in order, tracking health, entirely in process.

  Two properties worth stating, because they are the parts of failover
  that are usually wrong. First, a venue is marked unhealthy after a
  configurable number of **consecutive** failures and is not retried
  within the same order, so one dead venue costs a bounded amount of
  latency. Second, when every venue is down the answer is a rejection
  carrying each venue's reason, not an exception: the caller's correct
  response is to stop, and a stack trace is not a stop.

  Attributes:
    brokers: The venues in preference order.
    failure_threshold: Consecutive failures before a venue is skipped.
  '''

  brokers: tuple[Broker, ...]
  failure_threshold: int

  def __init__(self, brokers: Sequence[Broker],
               failure_threshold: int = 2) -> None:
    '''Build a router over an ordered list of brokers.

    Args:
      brokers: Venues in preference order. Names must be unique, since
        health is tracked by name and two venues sharing one would make
        a failure mark the wrong one.
      failure_threshold: Consecutive failures before a venue is marked
        unhealthy. One means "try it once more on the next order".

    Raises:
      ValueError: If there are no brokers, the threshold is below 1, or
        a name repeats.
    '''
    if not brokers:
      raise ValueError(
        'at least one broker is required; an empty router would reject '
        'every order for a reason nobody could act on')
    if failure_threshold < 1:
      raise ValueError(
        f'failure_threshold must be >= 1, got {failure_threshold}')
    names = [broker.name for broker in brokers]
    if len(set(names)) != len(names):
      raise ValueError(f'broker names must be unique, got {names}')
    self.brokers = tuple(brokers)
    self.failure_threshold = failure_threshold
    self._failures: dict[str, int] = dict.fromkeys(names, 0)
    self._errors: dict[str, str] = dict.fromkeys(names, '')
    self._unhealthy: set[str] = set()

  def place(self, order: OrderRequest, market_price: float) -> BrokerAck:
    '''Send an order through the first willing broker.

    Args:
      order: An order that has already cleared the pre-trade checks.
      market_price: Reference market price passed through to the broker.

    Returns:
      The winning acknowledgement, or a rejection naming every venue
      tried and why each refused.
    '''
    attempted: list[str] = []
    for broker in self.brokers:
      if broker.name in self._unhealthy:
        continue
      attempted.append(broker.name)
      try:
        ack = broker.place(order, market_price)
      except BrokerError as exc:
        self.note_failure(broker.name, str(exc))
        continue
      self.note_success(broker.name)
      return ack
    skipped = [broker.name for broker in self.brokers
               if broker.name in self._unhealthy
               and broker.name not in attempted]
    reasons = []
    for name in attempted + skipped:
      cause = self._errors.get(name) or 'marked unhealthy'
      reasons.append(f'{name}: {cause}')
    return BrokerAck(
      broker='failover',
      status=rejected,
      reason='no broker accepted the order -- ' + '; '.join(reasons),
      quantity=order.quantity)

  def cancel(self, order_id: str) -> BrokerAck:
    '''Cancel through the first broker that knows the order.

    Args:
      order_id: Identifier returned by a previous :meth:`place`.

    Returns:
      The broker's acknowledgement, or a rejection listing the venues
      tried.
    '''
    attempted: list[str] = []
    for broker in self.brokers:
      if broker.name in self._unhealthy:
        continue
      attempted.append(broker.name)
      try:
        ack = broker.cancel(order_id)
      except BrokerError as exc:
        self.note_failure(broker.name, str(exc))
        continue
      self.note_success(broker.name)
      return ack
    return BrokerAck(
      broker='failover', status=rejected, order_id=order_id,
      reason=f'no broker cancelled {order_id!r}; tried {attempted}')

  def positions(self) -> Mapping[str, int]:
    '''Return positions from every healthy broker, summed per symbol.

    **Summed, not merged.** Merging with ``update`` lets a later venue
    overwrite an earlier one for the same scrip, which erases a position
    outright and nets risk away: two venues holding +500 and -500 of the
    same scrip -- entirely normal immediately after a failover, since the
    primary could not cancel what it already held -- reported a book of
    zero while the full gross exposure was live at both. The reported
    book was not a function of the real book at all, and a book the risk
    layer cannot see is a book the risk layer cannot cap.

    **Unhealthy venues are skipped, and that is not a small caveat.**
    This docstring has always said "healthy brokers", and the loop read
    every broker, healthy or not, so a venue that had dropped its session
    kept reporting positions *over the top of* the venue that is actually
    working. Skipping is the right direction rather than the cautious one:
    a failed session's book is stale, and presenting a stale position as
    current is how an operator reconciles against a position that was
    closed an hour ago. The consequence is that a position held *only*
    at an unhealthy venue is absent from this view, and
    :meth:`unreachable_positions` exists so an operator can see exactly
    that blind spot rather than infer it.

    Returns:
      The sum over the healthy venues' signed quantities, per symbol.
      A copy, so a caller cannot rewrite the router's view.
    '''
    merged: dict[str, int] = {}
    for broker in self.brokers:
      if not self.is_healthy(broker.name):
        continue
      try:
        found = dict(broker.positions())
      except BrokerError:
        continue
      for symbol, quantity in found.items():
        merged[symbol] = merged.get(symbol, 0) + quantity
    return merged

  def unreachable_positions(self) -> Mapping[str, Mapping[str, int]]:
    '''Return the positions held only at venues this router has given up.

    The other half of :meth:`positions`, and it exists because that method
    is deliberately incomplete. A venue marked unhealthy is skipped, so
    anything it still holds is invisible there -- and an operator who
    sees a flat book has no way to tell "flat" from "the venue that held
    it stopped answering". This names the difference.

    **Not summed into anything and not netted.** These are positions the
    router cannot manage, and the right response to an unmanageable
    position is to look at it, not to fold it silently into a number that
    then gets compared against a limit.

    Returns:
      One mapping of symbol to signed quantity per unhealthy venue whose
      query succeeded, in rotation order. Empty when every venue is
      healthy, which is the ordinary case.

    PONYTAIL: health is per venue and process-local, so a venue marked
    unhealthy stays skipped for the life of the process. Ceiling: a
    recovered venue is never re-probed, so it remains invisible until the
    next restart. Upgrade path: a half-open probe that re-checks a skipped
    venue after a backoff, which needs a scheduler and a clock this
    project deliberately does not carry.
    '''
    stranded: dict[str, Mapping[str, int]] = {}
    for broker in self.brokers:
      if self.is_healthy(broker.name):
        continue
      try:
        found = dict(broker.positions())
      except BrokerError:
        continue
      stranded[broker.name] = found
    return stranded

  def note_failure(self, name: str, error: str) -> None:
    '''Record a failed call and possibly mark the venue unhealthy.

    Args:
      name: Broker that failed.
      error: Failure message.

    Raises:
      KeyError: If the name is not one of the router's brokers. Health is
        tracked by name, so a record for a venue outside the rotation
        would either be lost or, worse, be applied to whichever venue
        later takes that name.
    '''
    if name not in self._failures:
      raise KeyError(f'unknown broker {name!r}')
    self._failures[name] += 1
    self._errors[name] = error
    if self._failures[name] >= self.failure_threshold:
      self._unhealthy.add(name)

  def note_success(self, name: str) -> None:
    '''Clear a venue's failure streak after a call that worked.

    Args:
      name: Broker that succeeded.

    Raises:
      KeyError: If the name is not one of the router's brokers.
    '''
    if name not in self._failures:
      raise KeyError(f'unknown broker {name!r}')
    self._failures[name] = 0
    self._errors[name] = ''

  def is_healthy(self, name: str) -> bool:
    '''Return whether a venue is currently being tried.

    Args:
      name: Broker name.

    Returns:
      False when the venue has been marked unhealthy.
    '''
    return name not in self._unhealthy

  def health(self) -> tuple[BrokerHealth, ...]:
    '''Return every venue's health, in rotation order.

    Returns:
      One :class:`BrokerHealth` per venue.
    '''
    return tuple(
      BrokerHealth(
        name=broker.name,
        healthy=broker.name not in self._unhealthy,
        failures=self._failures.get(broker.name, 0),
        last_error=self._errors.get(broker.name, ''),
      )
      for broker in self.brokers
    )

  def render(self) -> str:
    '''Return a one-line health summary for a log.

    Returns:
      Human-readable summary of every venue.
    '''
    return ' | '.join(entry.render() for entry in self.health())
