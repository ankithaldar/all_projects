#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Licensed market data vendors, and the entitlement shape they need.

This project buys its data. It does not scrape it, and the reason is in
``docs/research/compliance/scraping-and-data-licensing.md``: NSE's Terms
of Use clause 9 bans systematic and automated collection outright, clause
8 bans storing the content in an electronic retrieval system without
written permission, and the IT Act s.43 exposure is civil compensation
rather than the criminal liability that design documents assume. Paying a
vendor removes the question entirely, for roughly Rs 2,000 to 5,000 a
month, which is the cheapest risk reduction in the project.

The vendor facts this module records, and the traps around them:

  * **TrueData Financial Information Pvt Ltd** and **Global Financial
    Datafeeds LLP** are both on NSE's authorised real-time vendor list,
    and both are the names most Indian retail feeds arrive under. Both
    are correct. ``bars.py`` already refers to them.
  * **"InvestingNote" is fabricated.** It is not an authorised NSE
    vendor, it appears in no list, and it must never appear as a vendor
    name here or anywhere else. The same applies to "Indian Express v.
    Zeal" in the research corpus, which is an invented case.
  * **Entitlements are per segment, not a boolean.** NSE sells CM, F&O,
    CD, Debt and Index separately. A feed that carries Nifty futures
    prints may have no cash-market or no debt entitlement at all, so a
    single ``has_data=True`` is not a usable abstraction. Hence
    :class:`Entitlement` and :class:`Segment`. A per-segment grant is
    still not a decision at the point of a request, so the entitlement
    carries the symbol-to-segment map it is checked against as well:
    without that map a feed cannot name the segment of the symbol being
    asked for, and a check against nothing is not a check.
  * **TradingView Inc. is on the authorised list.** Treating TradingView
    as free or public is wrong at the exchange level; the same applies to
    Bloomberg, Refinitiv, FactSet, ICE and the rest.
  * The list changes. The current published version dropped Accelpix,
    Citi, Deutsche Bank, Two Sigma, Proseon and Investment Technology
    Group, and added Virtu ITG and Vtrender. **Do not hardcode vendor
    names as the source of truth** -- read the list, which lives at
    nseindia.com/market-data/real-time-data-subscription as "List of
    Authorized Realtime Data Vendors (.pdf)".

Because of that last point, this module hardcodes no vendor names. It
declares the *shape* a feed must have, and the one implementation ships
with it is a replay of bars already in memory: which makes every
downstream test offline, deterministic and free, and means a strategy
result can be reproduced from a file without a vendor account.
'''

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Protocol, runtime_checkable

from stock_rl.bars import Bar

__all__ = [
  'Entitlement',
  'EntitlementError',
  'MarketDataFeed',
  'ReplayVendor',
  'Segment',
  'default_stale_after',
]


class Segment(Enum):
  '''An exchange segment an entitlement is granted for.

  NSE sells real-time data per segment rather than as one product, so a
  feed is authorised for some of these and not others. The values are the
  exchange's own abbreviations, which is what appears on an invoice and
  in a contract, so an entitlement set can be reconciled against the
  paperwork without a translation step.

  Members:
    CASH: Cash equity, the segment this project trades.
    FUTURES_AND_OPTIONS: Equity derivatives.
    CURRENCY_DERIVATIVES: Currency derivatives.
    DEBT: Debt securities.
    INDEX: Index data, including the Nifty family. Distinct from cash
      even though Nifty constituents are cash-market securities, because
      an index feed is licensed separately.
  '''

  CASH = 'CM'
  FUTURES_AND_OPTIONS = 'F&O'
  CURRENCY_DERIVATIVES = 'CD'
  DEBT = 'Debt'
  INDEX = 'Index'


class EntitlementError(ValueError):
  '''Raised when a feed is used outside its licensed segments.

  A ValueError subclass, with its own type so that a data-layer caller
  can distinguish an unlicensed request from a malformed argument. The
  distinction matters commercially: it is the difference between a bug
  and a licence breach, and only the second one is a conversation with
  the vendor.
  '''


@dataclass(frozen=True, slots=True)
class Entitlement:
  '''What a feed is licensed to carry, per segment.

  Two facts, and both are needed before a request can be shown to be
  licensed: *which segments* the feed covers, and *which segment each
  symbol trades in*. The second was missing, which made the first
  unusable at the point of a request -- a protocol can promise to raise
  on an unlicensed symbol only if it can name the segment, and a symbol
  with no recorded segment cannot be shown to be licensed at all.

  Attributes:
    vendor: Vendor or feed name, free text. Deliberately not validated
      against a list, because the authoritative list is a PDF on the
      exchange's site that changes without notice and this module must
      not become a stale copy of it.
    segments: Segments the feed may deliver. Empty is a meaningful and
      licensed-for-nothing state, not an absence of information.
    symbol_segments: Venue symbol to the segment it trades in. Absent
      symbol means unknown segment, and :meth:`require_symbol` refuses
      it rather than assuming cash: assuming a licence nobody showed is
      the failure that costs money.
  '''

  vendor: str
  segments: frozenset[Segment] = frozenset()
  symbol_segments: Mapping[str, Segment] = field(
    default_factory=lambda: {})

  @classmethod
  def granted(cls, vendor: str, *segments: Segment,
              symbol_segments: Mapping[str, Segment] | None = None,
              ) -> Entitlement:
    '''Return an entitlement covering the named segments.

    Args:
      vendor: Vendor or feed name.
      segments: Segments the feed is licensed for.
      symbol_segments: Venue symbol to the segment it trades in. Passed
        through so the same value answers both halves of the question,
        "is this segment licensed" and "is this symbol in it".

    Returns:
      The entitlement, with the segments frozen so it cannot be widened
      after the fact. Entitlement is a licence fact, and mutating one
      in place would make the record of what was licensed disagree with
      what the code does.
    '''
    return cls(vendor=vendor, segments=frozenset(segments),
               symbol_segments=dict(symbol_segments or {}))

  @classmethod
  def none(cls, vendor: str) -> Entitlement:
    '''Return an entitlement covering no segments at all.

    Args:
      vendor: Vendor or feed name.

    Returns:
      An empty entitlement. The honest default for a feed whose licence
      has not been read yet, and safer than guessing CM because a wrong
      guess produces licensed-looking data from an unlicensed source.
      It also classifies no symbol, so every request through it is
      refused until a caller says which segment a symbol is in.
    '''
    return cls(vendor=vendor, segments=frozenset())

  def allows(self, segment: Segment) -> bool:
    '''Return True if one segment is licensed.

    Args:
      segment: Segment to test.

    Returns:
      True if the segment is in the granted set.
    '''
    return segment in self.segments

  def segment_of(self, symbol: str) -> Segment | None:
    '''Return the segment a symbol trades in, or None if unrecorded.

    Args:
      symbol: Venue symbol.

    Returns:
      The recorded segment, or None when this entitlement carries no
      classification for that symbol. None means unknown, never "cash".
    '''
    return self.symbol_segments.get(symbol)

  def missing(self, needed: Iterable[Segment]) -> frozenset[Segment]:
    '''Return the needed segments this entitlement does not cover.

    Args:
      needed: Segments a caller intends to use.

    Returns:
      The subset of ``needed`` that is not licensed. Computed rather
      than asserted so a caller can log precisely what is absent.
    '''
    return frozenset(needed) - self.segments

  def require(self, *needed: Segment) -> None:
    '''Raise unless every named segment is licensed.

    Args:
      needed: Segments the caller is about to use.

    Raises:
      EntitlementError: If any of them is unlicensed.
    '''
    absent = sorted(segment.value for segment in self.missing(needed))
    if absent:
      joined = ', '.join(absent)
      raise EntitlementError(
        f'{self.vendor} is not licensed for segment(s) {joined}; '
        'entitlements are per segment, not a single boolean')

  def require_symbol(self, symbol: str) -> None:
    '''Raise unless the segment carrying ``symbol`` is licensed.

    This is the check that makes :attr:`segments` usable at the point of
    a request rather than only on paper, and it refuses in two distinct
    ways because they are two distinct problems: a symbol with no
    recorded segment cannot be shown to be licensed at all, and a symbol
    in an unlicensed segment is a licence breach.

    Args:
      symbol: Venue symbol being requested.

    Raises:
      EntitlementError: If the symbol's segment is unrecorded, or if it
        is recorded and not licensed.
    '''
    segment = self.segment_of(symbol)
    if segment is None:
      raise EntitlementError(
        f'{self.vendor} records no segment for {symbol}, so no licence '
        'can be shown for it; classify the symbol rather than assuming '
        'a segment')
    self.require(segment)


@runtime_checkable
class MarketDataFeed(Protocol):
  '''The surface the backtest core may assume of any market data feed.

  Declared as a Protocol rather than an abstract base class so a licensed
  vendor client can be added without inheriting anything, and so a test
  can assert conformance with ``isinstance`` instead of trusting that the
  adapter was written correctly.

  Attributes:
    vendor_name: Vendor or feed name, for the audit trail. The audit
      trail records which feed produced a price, so this is a compliance
      field rather than a label.
    entitlements: Per-segment licence state, including the symbol to
      segment mapping it is checked against. Any implementation must
      expose it and enforce it on ``bars``: a licence state nothing
      consults is a comment, and an unchecked one is the licence breach
      nobody was told about.
  '''

  vendor_name: str
  entitlements: Entitlement

  def bars(self, symbol: str, start: datetime,
           end: datetime) -> list[Bar]:
    '''Return bars for a symbol in an inclusive time range.

    Args:
      symbol: Venue symbol, e.g. ``RELIANCE``.
      start: First bar timestamp, inclusive.
      end: Last bar timestamp, inclusive.

    Returns:
      Bars in ascending timestamp order.

    Raises:
      EntitlementError: If the symbol's segment is not licensed, or the
        feed records no segment for it.
      KeyError: If the symbol is not carried by this feed.
    '''

  def symbols(self) -> list[str]:
    '''Return every symbol the feed carries.

    Returns:
      Symbols in ascending order.
    '''

  def last_bar_at(self, symbol: str) -> datetime | None:
    '''Return the timestamp of the most recent bar for a symbol.

    Args:
      symbol: Venue symbol.

    Returns:
      The timestamp, or None when the feed carries no bar for it. None
      distinguishes an empty series from a missing symbol only once
      combined with ``symbols()``.
    '''


#: Age at which a replayed bar counts as stale. Thirty seconds against a
#: documented round trip of 150 to 400 ms is generous by two orders of
#: magnitude, which is the point: a bound loose enough never fires in
#: normal operation still catches a feed that has stopped updating.
default_stale_after = timedelta(seconds=30)


class ReplayVendor:
  '''A vendor feed served entirely from bars already in memory.

  This is the implementation the test suite runs against, and it is not
  a mock: it enforces the same contract a real feed would, so a strategy
  that breaks against it breaks against the vendor too. It performs no
  I/O. It does enforce entitlements, on the explicit record the caller
  supplies: every :meth:`bars` call is gated on the symbol's segment
  being licensed, and a symbol with no recorded segment is refused.

  Staleness is the reason it exists rather than a plain list of bars. A
  real-time feed is only useful while it is current, and this project's
  own latency budget puts a full request round trip at 150 to 400
  milliseconds. A quote that has not moved for tens of seconds is not a
  slow quote, it is a dead feed, and a backtest that keeps filling on one
  is producing numbers no live system could have produced.
  '''

  def __init__(
    self,
    vendor_name: str,
    bars_by_symbol: Mapping[str, Sequence[Bar]],
    entitlements: Entitlement | None = None,
    stale_after: timedelta = default_stale_after,
    clock: Callable[[], datetime] | None = None,
    symbol_segments: Mapping[str, Segment] | None = None,
  ) -> None:
    '''Build a replay feed from in-memory bars.

    Args:
      vendor_name: Name recorded in the audit trail. Free text: this
        class deliberately does not check it against NSE's list, because
        a hardcoded list goes stale and a stale list is worse than no
        list.
      bars_by_symbol: Mapping of symbol to its bars, each in strictly
        ascending timestamp order.
      entitlements: Per-segment licence state, including the symbol to
        segment classification the licence is checked against. Defaults
        to no entitlement at all rather than to cash, since assuming a
        licence the caller has not shown is the failure that actually
        costs money -- and an undeclared licence refuses every request
        rather than serving it.
      stale_after: Age at which the newest bar for a symbol counts as
        stale.
      clock: Callable returning the current time, injected so staleness
        can be tested without a wall clock.
      symbol_segments: Venue symbol to segment, merged into the
        entitlement's own classification. Pass it here to keep the
        instrument master next to the bars; pass it to
        :meth:`Entitlement.granted` instead to keep it with the licence.
        Later entries win, so a feed-specific mapping can correct a
        licence-wide one.

    Raises:
      ValueError: If the vendor name is empty, if ``bars_by_symbol`` is
        empty, if ``stale_after`` is not positive, or if any symbol's
        bars are not strictly ascending. Ascending order is checked
        rather than sorted, exactly as ``bars.load_csv`` does: a
        scrambled series must fail loudly instead of being silently
        reordered into a plausible-looking history.
    '''
    if not vendor_name.strip():
      raise ValueError('vendor_name must not be empty')
    if not bars_by_symbol:
      raise ValueError(
        'a replay vendor with no bars serves nothing; pass at least one '
        'symbol')
    if stale_after <= timedelta(0):
      raise ValueError(f'stale_after must be positive, got {stale_after}')
    self.vendor_name = vendor_name
    base = entitlements or Entitlement.none(vendor_name)
    self.entitlements = replace(
      base,
      symbol_segments={**base.symbol_segments, **dict(symbol_segments or {})},
    )
    self.stale_after = stale_after
    self._clock = clock or datetime.now
    self._bars: dict[str, tuple[Bar, ...]] = {
      symbol: tuple(_checked(symbol, series))
      for symbol, series in bars_by_symbol.items()
    }

  def symbols(self) -> list[str]:
    '''Return every symbol carried, in ascending order.

    Returns:
      Sorted symbol list.
    '''
    return sorted(self._bars)

  def has_symbol(self, symbol: str) -> bool:
    '''Return True if the feed carries a symbol.

    Args:
      symbol: Venue symbol.

    Returns:
      True when at least one bar exists for it. Distinct from
      ``last_bar_at``, which returns None both for a missing symbol and
      for an empty one.
    '''
    return symbol in self._bars

  def bars(self, symbol: str, start: datetime,
           end: datetime) -> list[Bar]:
    '''Return bars for a symbol in an inclusive time range.

    Both bounds are inclusive, so a caller asking for a single bar by
    naming its timestamp twice gets exactly that bar. The result is a new
    list each call, so a caller cannot corrupt the replay source by
    mutating what it was handed.

    Args:
      symbol: Venue symbol.
      start: First timestamp to include, inclusive.
      end: Last timestamp to include, inclusive.

    Returns:
      Matching bars in ascending order, possibly empty.

    Raises:
      KeyError: If the symbol is not carried. Raising rather than
        returning an empty list, because an empty list is exactly what a
        backtest silently completes on, and a missing symbol is the
        single most likely ingestion mistake there is.
      EntitlementError: If the symbol's segment is not licensed, or is
        not recorded at all. Checked before the range, because a licence
        that does not cover the request does not become valid by asking
        for less of it.
      ValueError: If ``start`` is after ``end``.
    '''
    series = self._bars.get(symbol)
    if series is None:
      raise KeyError(
        f'{self.vendor_name} does not carry {symbol}; known symbols: '
        f'{self.symbols()}')
    self.entitlements.require_symbol(symbol)
    if start > end:
      raise ValueError(f'start {start} is after end {end}')
    return [bar for bar in series if start <= bar.timestamp <= end]

  def last_bar_at(self, symbol: str) -> datetime | None:
    '''Return the timestamp of the newest bar for a symbol.

    Args:
      symbol: Venue symbol.

    Returns:
      The newest timestamp, or None when the feed does not carry the
      symbol. Replayed symbols always have at least one bar, since an
      empty series is rejected at construction.
    '''
    series = self._bars.get(symbol)
    if not series:
      return None
    return series[-1].timestamp

  def age(self, symbol: str,
          now: datetime | None = None) -> timedelta | None:
    '''Return how old the newest bar for a symbol is.

    Args:
      symbol: Venue symbol.
      now: Instant to measure against, or None to use the injected
        clock.

    Returns:
      The age of the newest bar, never negative, or None when the feed
      does not carry the symbol.

    Raises:
      ValueError: If ``now`` and the bar timestamps disagree about
        awareness. Naive timestamps here mean exchange local time, so an
        aware ``now`` is stripped; the reverse cannot be resolved without
        guessing a time zone and is refused.
    '''
    newest = self.last_bar_at(symbol)
    if newest is None:
      return None
    moment = now if now is not None else self._clock()
    moment = _align(moment, newest)
    age = moment - newest
    # A bar stamped in the future is a clock problem, not a negative
    # age. Reporting zero keeps a downstream staleness check meaningful
    # without inventing a duration.
    return age if age > timedelta(0) else timedelta(0)

  def is_stale(self, symbol: str, now: datetime | None = None,
               stale_after: timedelta | None = None) -> bool:
    '''Return True if a symbol's newest bar is too old to act on.

    Args:
      symbol: Venue symbol.
      now: Instant to measure against, or None to use the injected
        clock.
      stale_after: Override for the feed's default staleness bound.

    Returns:
      True when the newest bar is at least ``stale_after`` old. False for
      a symbol the feed does not carry, on the reasoning that an absent
      bar is a caller error to be raised elsewhere rather than a stale
      quote.

    Raises:
      ValueError: If ``now`` and the bar timestamps disagree about
        awareness.
    '''
    age = self.age(symbol, now)
    if age is None:
      return False
    return age >= (stale_after or self.stale_after)

  def stale_symbols(self, now: datetime | None = None) -> list[str]:
    '''Return every symbol whose newest bar is stale.

    Args:
      now: Instant to measure against, or None to use the injected
        clock.

    Returns:
      Stale symbols in ascending order. Computed in one pass so a caller
      can gate a whole panel on one decision rather than polling symbol
      by symbol and acting between the polls.
    '''
    return sorted(
      symbol for symbol in self._bars
      if self.is_stale(symbol, now)
    )


def _checked(symbol: str, series: Iterable[Bar]) -> list[Bar]:
  '''Validate one symbol's bars and return them as a list.

  Args:
    symbol: Venue symbol, used only for the error message.
    series: Bars in the order supplied.

  Returns:
    The bars, in the order supplied.

  Raises:
    ValueError: If two consecutive bars share or invert a timestamp.
  '''
  bars = list(series)
  for previous, current in zip(bars, bars[1:]):
    if current.timestamp <= previous.timestamp:
      raise ValueError(
        f'{symbol}: bars must be strictly ascending, got '
        f'{current.timestamp} after {previous.timestamp}')
  return bars


def _align(moment: datetime, reference: datetime) -> datetime:
  '''Make an instant comparable with a bar timestamp.

  Args:
    moment: The instant supplied by the caller.
    reference: A bar timestamp to be compared against.

  Returns:
    ``moment`` with its awareness matched to ``reference``.

  Raises:
    ValueError: If the reference is timezone-aware and ``moment`` is
      not, since which zone to use cannot be inferred.
  '''
  if reference.tzinfo is None and moment.tzinfo is not None:
    # Naive means exchange local time throughout this project, so an
    # aware instant is read in its own local wall time.
    return moment.replace(tzinfo=None)
  if reference.tzinfo is not None and moment.tzinfo is None:
    raise ValueError(
      'bar timestamps are timezone-aware but now is naive; supply an '
      'aware instant rather than guessing the exchange time zone')
  return moment
