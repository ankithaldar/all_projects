#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''NSE trading calendar, with holidays injected rather than invented.

The only genuinely load-bearing fact encoded here is that the NSE equity
cash session runs 09:15 to 15:30 IST. Everything else the module knows
about a calendar it was told.

**No holiday list is bundled, on purpose.** An embedded list is a snapshot
that expires silently: it looks authoritative, it is out of date within
months, and nobody diffs it against the exchange circular. Worse, a
wrong holiday is not a warning, it is a day of wrong returns. So the
constructor takes the holidays, and a caller with no list gets a calendar
that knows about weekends only -- which is wrong in an obvious, visible
way rather than wrong in a quiet one. Feed it the exchange's own list.

What this module refuses to do:

  * **Guess Muhurat trading times.** Muhurat trading is a real, separate
    session with its own hours, announced per year per exchange, and it
    is often shorter than the normal session. Those hours are therefore
    ``None`` and the session is a flag: a :class:`Session` with ``opens``
    and ``closes`` unset and ``is_muhurat`` set. A caller that needs to
    know whether the market was open at 16:45 on a Muhurat evening has
    to be told the times. Guessing them, or worse inheriting the normal
    session's hours, would fabricate trading that did not happen -- and a
    fabricated bar is a look-ahead bug, because a strategy that believes
    it traded at 16:45 will fill there.

  * **Second-guess the weekend.** Saturday and Sunday are not traded in
    equity cash. NSE does run occasional Saturday sessions for specific
    products and index rebalancing; those arrive as an injected closure
    or an explicit exception list rather than as a weekend override.

  * **Model trading holidays as a rule.** Several NSE closures recur on
    a lunar or religious calendar and are announced annually. No
    algorithm reproduces them correctly, and one that looked convincing
    would be worse than none.

A ``TradingDay`` with no sessions is a non-trading day, which is what
makes the weekend and the holiday list the same code path. That is not a
simplification for its own sake: it means no code path can treat a
closed day as open.
'''

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

__all__ = [
  'Session',
  'TradingCalendar',
  'TradingDay',
  'coerce_date',
  'max_range_days',
  'muhurat_session',
  'nse_normal_session',
  'search_horizon_days',
  'weekend_days',
]


@dataclass(frozen=True, slots=True)
class Session:
  '''One continuous window in which an exchange accepts orders.

  Attributes:
    name: Identifier for the session, e.g. ``normal`` or ``muhurat``.
    opens: Session open in exchange local time, or None when the hours
      are not known. None means unknown, never "opens at midnight".
    closes: Session close in exchange local time, or None when unknown.
    is_muhurat: True for a Muhurat session. A flag rather than a
      schedule, because the point of Muhurat is that it is a different
      session at a different time, and the only honest encoding of an
      unknown time is to say so.
  '''

  name: str
  opens: time | None = None
  closes: time | None = None
  is_muhurat: bool = False

  @property
  def has_times(self) -> bool:
    '''True when both open and close are known.'''
    return self.opens is not None and self.closes is not None

  @property
  def duration_minutes(self) -> float | None:
    '''Session length in minutes, or None when the hours are unknown.

    A session whose close precedes its open returns None rather than a
    negative length or a wrapped 24-hour one. No Indian exchange runs
    an overnight equity session, so such an input is a mistake, and a
    caller is better served by the nonsense than by a silent correction.
    '''
    if not self.has_times:
      return None
    span = _minutes(self.closes) - _minutes(self.opens)
    return None if span < 0.0 else span

  def contains(self, moment: time) -> bool:
    '''Return True if a wall-clock time falls inside this session.

    Args:
      moment: Time of day in exchange local time.

    Returns:
      True when the hours are known and ``moment`` lies in the inclusive
      range ``[opens, closes]``. False when the hours are unknown, so an
      untimed session never claims to be open.
    '''
    if not self.has_times:
      return False
    return _minutes(self.opens) <= _minutes(moment) <= _minutes(self.closes)

  def describe(self) -> str:
    '''Return a one-line description for logs.

    Returns:
      The session name and hours, or an explicit note that the hours were
      not announced. The wording is chosen so a log line cannot be read
      as asserting a trading window that was never established.
    '''
    if not self.has_times:
      return f'{self.name}: hours not announced'
    return f'{self.name}: {self.opens:%H:%M}-{self.closes:%H:%M} IST'


#: Weekday numbers that are never traded in equity cash, matching
#: ``date.weekday`` where Monday is 0 and Sunday is 6.
weekend_days = (5, 6)

#: NSE equity cash normal session, 09:15 to 15:30 IST: 375 minutes, the
#: denominator for any intraday annualisation. The Closing Auction
#: Session, live under NSE/FAOP/74467 of 29 May 2026, prints at the 15:30
#: close with different microstructure; it is a post-close phase rather
#: than a longer session, so it does not move these times.
nse_normal_session = Session(name='normal', opens=time(9, 15),
                             closes=time(15, 30))

#: The Muhurat session, times deliberately unset. See the module
#: docstring: the hours are announced per year and are not derivable.
muhurat_session = Session(name='muhurat', is_muhurat=True)


@dataclass(frozen=True, slots=True)
class TradingDay:
  '''One calendar date and the sessions it offers.

  A day with no sessions is a non-trading day. Encoding non-trading as
  absence rather than as a flag means a weekend, a trading holiday and a
  date outside the calendar's horizon are the same thing to a caller, so
  there is no path that treats a closed day as open.

  Attributes:
    day: The calendar date.
    sessions: Sessions offered, in chronological order. Empty for a
      non-trading day.
    note: Free-text reason, e.g. the holiday name from the exchange
      circular. Carried so an audit trail can say why a day is missing,
      rather than leaving a gap to be interpreted.
  '''

  day: date
  sessions: tuple[Session, ...] = ()
  note: str = ''

  @property
  def is_trading_day(self) -> bool:
    '''True when at least one session is offered.'''
    return bool(self.sessions)

  @property
  def is_muhurat(self) -> bool:
    '''True when a Muhurat session is offered on this day.'''
    return any(session.is_muhurat for session in self.sessions)

  @property
  def timed_sessions(self) -> tuple[Session, ...]:
    '''Sessions whose hours are known, in chronological order.'''
    return tuple(session for session in self.sessions if session.has_times)

  def session_named(self, name: str) -> Session | None:
    '''Return the session with a given name, if offered.

    Args:
      name: Session name to look for.

    Returns:
      The matching session, or None when the day does not offer it.
    '''
    for session in self.sessions:
      if session.name == name:
        return session
    return None

  def primary(self) -> Session | None:
    '''Return the session that defines the day's trading window.

    Returns:
      The first session with known hours, or None when every session is
      untimed. The first *timed* session rather than simply the first,
      because a Muhurat session with unannounced hours must not become
      the day's trading window.
    '''
    timed = self.timed_sessions
    return timed[0] if timed else None


#: Guard on a single range query. Ten years of calendar is more than any
#: backtest needs in one call, and a longer request is almost always a
#: caller that has the arguments the wrong way round.
max_range_days = 3650

#: How far ``next_trading_day`` and ``previous_trading_day`` search
#: before giving up. Two years of consecutive closures would be an
#: unusable holiday list, so reaching this means the input is wrong.
search_horizon_days = 730


class TradingCalendar:
  '''Which dates trade, given an injected holiday list.

  A plain mutable class rather than a frozen dataclass because holiday
  lists are genuinely amended -- an exchange declares a one-off closure
  mid-year -- and rebuilding a whole calendar to add one date is the
  wrong shape for that.
  '''

  def __init__(
    self,
    holidays: Iterable[date | datetime | str] = (),
    sessions: Sequence[Session] = (nse_normal_session,),
  ) -> None:
    '''Build a calendar.

    Args:
      holidays: Non-trading dates other than weekends. Accepts date,
        datetime and ISO-8601 strings, because holiday lists arrive from
        exchange circulars as text and from spreadsheets as strings.
      sessions: Sessions offered on an ordinary trading day. Defaults to
        the NSE equity cash session only. Muhurat days are supplied
        explicitly, never inferred.

    Raises:
      ValueError: If ``sessions`` is empty. Such a calendar would treat
        every date as closed and fail silently in the one direction that
        loses money quietly.
    '''
    if not sessions:
      raise ValueError(
        'a calendar with no sessions treats every day as closed; pass '
        'at least the normal session')
    self.sessions: tuple[Session, ...] = tuple(sessions)
    self._holidays: set[date] = {coerce_date(day) for day in holidays}

  @property
  def holidays(self) -> frozenset[date]:
    '''Injected closure dates, excluding weekends.'''
    return frozenset(self._holidays)

  def add_holidays(self, days: Iterable[date | datetime | str]) -> None:
    '''Add closure dates to the calendar.

    Args:
      days: Dates to close. A date already closed is ignored rather
        than being an error, because the common caller appends a full
        year's list to a calendar that already excludes weekends.
    '''
    self._holidays.update(coerce_date(day) for day in days)

  def is_weekend(self, day: date | datetime) -> bool:
    '''Return True if a date falls on a Saturday or Sunday.

    Args:
      day: Date to test.

    Returns:
      True for Saturday or Sunday. Both are closed for equity cash.
    '''
    return _as_date(day).weekday() in weekend_days

  def is_holiday(self, day: date | datetime) -> bool:
    '''Return True if a date is an injected closure.

    Args:
      day: Date to test.

    Returns:
      True when the date is in the injected list. A weekend that is not
      in the list returns False here and is still closed, because the
      two questions are separate on purpose: an exchange announcing a
      working Saturday needs to be able to move only the second one.
    '''
    return _as_date(day) in self._holidays

  def is_trading_day(self, day: date | datetime) -> bool:
    '''Return True if orders may be placed on a date.

    Args:
      day: Date to test.

    Returns:
      True for a weekday that is not an injected holiday. An unlisted
      weekday is treated as traded, and that is deliberate: the
      alternative -- treating any date not positively confirmed as
      closed -- silently truncates a price history at its first gap,
      which is a far worse failure than an unobserved closure.
    '''
    moment = _as_date(day)
    if moment.weekday() in weekend_days:
      return False
    return moment not in self._holidays

  def session_for(self, day: date | datetime) -> TradingDay | None:
    '''Return the sessions offered on a date.

    Args:
      day: Date to describe.

    Returns:
      A ``TradingDay`` carrying the configured sessions when the date
      trades, and None when it does not. None rather than an empty
      ``TradingDay``, so a caller who forgets to test the trading day
      fails at the point of use instead of quietly iterating no sessions.
    '''
    moment = _as_date(day)
    if not self.is_trading_day(moment):
      return None
    return TradingDay(day=moment, sessions=self.sessions)

  def session_at(self, day: date | datetime, moment: time) -> Session | None:
    '''Return the session covering a wall-clock time on a date.

    Args:
      day: Date to inspect.
      moment: Time of day in exchange local time.

    Returns:
      The session containing ``moment``, or None when the date is closed
      or the time is outside every known window. A date whose only
      session has unannounced hours also returns None, and the caller is
      expected to read that as unknown rather than as closed.
    '''
    trading_day = self.session_for(day)
    if trading_day is None:
      return None
    for session in trading_day.sessions:
      if session.contains(moment):
        return session
    return None

  def is_market_open(self, day: date | datetime, moment: time) -> bool:
    '''Return True if a time falls inside a known trading session.

    Args:
      day: Date to inspect.
      moment: Time of day in exchange local time.

    Returns:
      True only when a session with known hours contains the time. Both
      bounds are inclusive, so 09:15 and 15:30 are open, which matches
      the exchange: the first order and the closing auction both print
      at those times.
    '''
    return self.session_at(day, moment) is not None

  def trading_days(self, start: date | datetime,
                   end: date | datetime) -> list[date]:
    '''Return every trading date in an inclusive range.

    Args:
      start: First date of the range, inclusive.
      end: Last date of the range, inclusive.

    Returns:
      Trading dates in ascending order. Empty when ``start`` is after
      ``end``, since an inverted range is a legitimate query that simply
      has no answer.

    Raises:
      ValueError: If the range exceeds ``max_range_days``.
    '''
    first = _as_date(start)
    last = _as_date(end)
    if first > last:
      return []
    if (last - first).days > max_range_days:
      raise ValueError(
        f'range of {(last - first).days} days exceeds the {max_range_days}'
        '-day guard; pass holidays explicitly rather than enumerating '
        'a decade')
    return list(self.iter_trading_days(first, last))

  def iter_trading_days(self, start: date | datetime,
                        end: date | datetime) -> Iterator[date]:
    '''Yield trading dates in an inclusive range, without a full list.

    Args:
      start: First date of the range, inclusive.
      end: Last date of the range, inclusive.

    Yields:
      Trading dates in ascending order. Yields nothing when the range is
      inverted.
    '''
    first = _as_date(start)
    last = _as_date(end)
    day = first
    while day <= last:
      if self.is_trading_day(day):
        yield day
      day += timedelta(days=1)

  def next_trading_day(self, day: date | datetime,
                       inclusive: bool = False) -> date | None:
    '''Return the next trading date at or after a date.

    Args:
      day: Date to search from.
      inclusive: When True, a date that is itself a trading day is
        returned rather than stepped over.

    Returns:
      The next trading date, or None if the search runs past the
      ``search_horizon_days`` limit. The limit exists because an
      under-specified holiday list can make the future permanently
      un-trading, and a caller waiting on that result would hang.
    '''
    moment = _as_date(day)
    if not inclusive:
      moment += timedelta(days=1)
    for offset in range(search_horizon_days):
      candidate = moment + timedelta(days=offset)
      if self.is_trading_day(candidate):
        return candidate
    return None

  def previous_trading_day(self, day: date | datetime,
                            inclusive: bool = False) -> date | None:
    '''Return the most recent trading date at or before a date.

    Args:
      day: Date to search backwards from.
      inclusive: When True, a date that is itself a trading day is
        returned rather than stepped over.

    Returns:
      The most recent trading date, or None past the search horizon.
    '''
    moment = _as_date(day)
    if not inclusive:
      moment -= timedelta(days=1)
    for offset in range(search_horizon_days):
      candidate = moment - timedelta(days=offset)
      if self.is_trading_day(candidate):
        return candidate
    return None


def _minutes(moment: time) -> float:
  '''Return a time of day as minutes after midnight.

  Args:
    moment: Time of day.

  Returns:
    Minutes since midnight, with seconds carried as a fraction.
  '''
  return moment.hour * 60.0 + moment.minute + moment.second / 60.0


def _as_date(value: date | datetime) -> date:
  '''Return the date part of a date or datetime.

  Args:
    value: A date or datetime.

  Returns:
    The date, unchanged for a date input.
  '''
  return value.date() if isinstance(value, datetime) else value


def coerce_date(value: str | date | datetime) -> date:
  '''Parse an ISO-8601 date string into a date.

  Holiday lists and record keys arrive as text from exchange circulars
  and from spreadsheets, and ``date.fromisoformat`` accepts the compact
  ``YYYYMMDD`` form Indian exchange files favour as well as the
  hyphenated one.

  Args:
    value: A date, a datetime, or an ISO-8601 date string.

  Returns:
    The parsed date, or the date part of a datetime.

  Raises:
    ValueError: If a string value is not a valid ISO-8601 date.
    TypeError: If the value is of an unsupported type.
  '''
  if isinstance(value, datetime):
    return value.date()
  if isinstance(value, date):
    return value
  if isinstance(value, str):
    return date.fromisoformat(value.strip())
  raise TypeError(f'cannot read a date from {type(value).__name__}')
