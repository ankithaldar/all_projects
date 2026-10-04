#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the trading calendar and the licensed vendor abstraction.

Two failure modes are being pinned down here, because both are silent:

  * a calendar that quietly trades on a holiday, or quietly does not
    trade on a real session, produces returns that are wrong in a way
    nothing downstream can detect;
  * a feed that keeps serving a stale bar lets a backtest fill on a
    price no live system could have obtained.

No network access. Every bar is constructed in memory and every clock is
injected, so the whole file is deterministic and offline.
'''

import ast
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest

import stock_rl
from stock_rl.bars import Bar
from stock_rl.data import vendors
from stock_rl.data.calendar import (
  Session,
  TradingCalendar,
  TradingDay,
  coerce_date,
  max_range_days,
  muhurat_session,
  nse_normal_session,
  search_horizon_days,
  weekend_days,
)
from stock_rl.data.vendors import (
  Entitlement,
  EntitlementError,
  MarketDataFeed,
  ReplayVendor,
  Segment,
)

#: A Wednesday, an ordinary trading day in any year.
WEDNESDAY = date(2026, 1, 28)


def bars(count, start=time(9, 15), price=100.0, symbol='RELIANCE'):
  '''Return a strictly ascending run of one-minute bars.

  Args:
    count: Number of bars to produce.
    start: Timestamp of the first bar, in exchange local time.
    price: Close price of every bar, drifting by a rupee each.
    symbol: Unused, retained so a call site reads as being about one
      symbol.

  Returns:
    A list of ``Bar`` objects one minute apart.
  '''
  del symbol
  base = datetime.combine(WEDNESDAY, start)
  return [
    Bar(base + timedelta(minutes=index), price + index, price + index + 1.0,
        price + index - 1.0, price + index, 1000.0)
    for index in range(count)
  ]


def vendor(**kwargs):
  '''Build a one-symbol ReplayVendor for tests.

  Args:
    kwargs: Overrides passed through to ``ReplayVendor``.

  Returns:
    A vendor serving 10 bars on ``RELIANCE`` and nothing else.
  '''
  defaults = {
    'vendor_name': 'test-replay',
    'bars_by_symbol': {'RELIANCE': bars(10)},
  }
  defaults.update(kwargs)
  return ReplayVendor(**defaults)


class TestWeekends:
  '''Saturday and Sunday are never equity cash sessions.'''

  @pytest.mark.parametrize('day', [
      date(2026, 1, 31),   # Saturday
      date(2026, 2, 1),    # Sunday
  ])
  def test_weekend_is_not_a_trading_day(self, day):
    assert TradingCalendar().is_trading_day(day) is False

  def test_weekend_has_no_sessions(self):
    # None rather than an empty TradingDay, so a caller who forgets to
    # test the trading day fails at the point of use.
    assert TradingCalendar().session_for(date(2026, 1, 31)) is None

  def test_weekend_is_reported_separately_from_a_holiday(self):
    # Two separate questions, because an exchange announcing a working
    # Saturday has to be able to move only the second one.
    calendar = TradingCalendar()
    assert calendar.is_weekend(date(2026, 1, 31)) is True
    assert calendar.is_holiday(date(2026, 1, 31)) is False

  def test_weekday_is_a_trading_day_without_any_holiday_list(self):
    assert TradingCalendar().is_trading_day(WEDNESDAY) is True

  def test_weekend_days_are_saturday_and_sunday(self):
    # date.weekday() numbering, Monday 0.
    assert weekend_days == (5, 6)
    assert WEDNESDAY.weekday() not in weekend_days


class TestInjectedHolidays:
  '''Holidays are supplied, never bundled or derived.'''

  def test_holiday_from_a_string_is_honoured(self):
    calendar = TradingCalendar(holidays=['2026-01-26'])
    assert calendar.is_trading_day(date(2026, 1, 26)) is False
    assert calendar.is_holiday(date(2026, 1, 26)) is True

  def test_holiday_from_a_date_is_honoured(self):
    calendar = TradingCalendar(holidays=[date(2026, 1, 26)])
    assert calendar.is_trading_day(date(2026, 1, 26)) is False

  def test_holiday_from_a_datetime_is_honoured(self):
    calendar = TradingCalendar(
      holidays=[datetime(2026, 1, 26, 9, 15)])
    assert calendar.is_trading_day(date(2026, 1, 26)) is False

  def test_no_holiday_list_means_no_holidays(self):
    # The deliberate consequence of refusing to bundle a list: the
    # calendar knows only about weekends, and says so by omission.
    calendar = TradingCalendar()
    assert calendar.holidays == frozenset()

  def test_holidays_can_be_added_afterwards(self):
    # Exchanges declare one-off closures mid-year, so the list is mutable.
    calendar = TradingCalendar()
    assert calendar.is_trading_day(WEDNESDAY) is True
    calendar.add_holidays([WEDNESDAY])
    assert calendar.is_trading_day(WEDNESDAY) is False

  def test_adding_a_weekend_again_is_not_an_error(self):
    calendar = TradingCalendar(holidays=[date(2026, 1, 31)])
    calendar.add_holidays([date(2026, 1, 31)])
    assert calendar.is_holiday(date(2026, 1, 31)) is True

  def test_a_calendar_with_no_sessions_is_refused(self):
    # Otherwise every date is closed and the calendar fails silently in
    # the one direction that loses money quietly.
    with pytest.raises(ValueError, match='no sessions'):
      TradingCalendar(sessions=[])


class TestSessions:
  '''The NSE session hours, and what is deliberately left blank.'''

  def test_normal_session_is_nine_fifteen_to_three_thirty(self):
    assert nse_normal_session.opens == time(9, 15)
    assert nse_normal_session.closes == time(15, 30)

  def test_normal_session_is_375_minutes(self):
    # The denominator for any intraday annualisation.
    assert nse_normal_session.duration_minutes == 375.0

  @pytest.mark.parametrize('moment,expected', [
      (time(9, 14), False),
      (time(9, 15), True),    # the open itself is inside the session
      (time(12, 0), True),
      (time(15, 30), True),   # the close is inside it too
      (time(15, 31), False),
  ])
  def test_market_open_bounds_are_inclusive(self, moment, expected):
    assert nse_normal_session.contains(moment) is expected

  def test_inverted_session_has_no_duration(self):
    # No Indian equity exchange runs an overnight session, so this input
    # is a mistake and the caller is better served by nonsense than by a
    # silent correction.
    session = Session(name='broken', opens=time(15, 0), closes=time(9, 0))
    assert session.duration_minutes is None

  def test_describe_names_the_hours(self):
    assert '09:15-15:30' in nse_normal_session.describe()


class TestMuhurat:
  '''Muhurat is a flag, never a guessed time.'''

  def test_muhurat_session_has_no_invented_hours(self):
    # Guessing these, or inheriting the normal session's, would
    # fabricate trading that did not happen -- and a fabricated bar is a
    # look-ahead bug, because a strategy believing it traded at 16:45
    # will fill there.
    assert muhurat_session.opens is None
    assert muhurat_session.closes is None
    assert muhurat_session.has_times is False
    assert muhurat_session.duration_minutes is None

  def test_untimed_session_never_claims_to_contain_a_time(self):
    assert muhurat_session.contains(time(16, 45)) is False

  def test_untimed_session_says_its_hours_are_unknown(self):
    # Chosen so a log line cannot be read as asserting a window that was
    # never established.
    assert 'not announced' in muhurat_session.describe()

  def test_muhurat_session_is_flagged(self):
    assert muhurat_session.is_muhurat is True
    assert nse_normal_session.is_muhurat is False

  def test_muhurat_day_is_recognised_through_its_flag(self):
    day = TradingDay(day=WEDNESDAY,
                     sessions=(nse_normal_session, muhurat_session))
    assert day.is_muhurat is True
    assert day.is_trading_day is True

  def test_an_ordinary_day_is_not_muhurat(self):
    calendar = TradingCalendar()
    assert calendar.session_for(WEDNESDAY).is_muhurat is False

  def test_muhurat_only_day_reports_no_timed_window(self):
    # primary() takes the first *timed* session, so an untimed Muhurat
    # session cannot become the day's trading window by being listed
    # first.
    day = TradingDay(day=WEDNESDAY, sessions=(muhurat_session,))
    assert day.primary() is None
    assert not day.timed_sessions

  def test_muhurat_never_becomes_the_primary_session(self):
    day = TradingDay(day=WEDNESDAY,
                     sessions=(muhurat_session, nse_normal_session))
    assert day.primary().name == 'normal'


class TestTradingDay:
  '''A day with no sessions is a non-trading day.'''

  def test_empty_day_is_not_a_trading_day(self):
    assert TradingDay(day=WEDNESDAY).is_trading_day is False

  def test_session_named_returns_none_when_absent(self):
    calendar = TradingCalendar()
    assert calendar.session_for(WEDNESDAY).session_named('muhurat') is None

  def test_session_named_finds_an_offered_session(self):
    calendar = TradingCalendar()
    assert calendar.session_for(WEDNESDAY).session_named('normal') == (
      nse_normal_session)

  def test_note_is_carried_for_the_audit_trail(self):
    day = TradingDay(day=WEDNESDAY, note='Republic Day')
    assert day.note == 'Republic Day'


class TestMarketOpen:
  '''Intraday open/closed queries through the calendar.'''

  def test_open_on_a_trading_day(self):
    assert TradingCalendar().is_market_open(WEDNESDAY, time(10, 0)) is True

  def test_closed_before_the_open(self):
    assert TradingCalendar().is_market_open(WEDNESDAY, time(9, 0)) is False

  def test_closed_on_a_weekend_regardless_of_the_time(self):
    assert TradingCalendar().is_market_open(date(2026, 1, 31),
                                           time(10, 0)) is False

  def test_session_at_returns_the_matching_session(self):
    session = TradingCalendar().session_at(WEDNESDAY, time(10, 0))
    assert session is nse_normal_session

  def test_session_at_is_none_on_a_holiday(self):
    calendar = TradingCalendar(holidays=[WEDNESDAY])
    assert calendar.session_at(WEDNESDAY, time(10, 0)) is None


class TestTradingDayRanges:
  '''Ranges, and the guards that stop a wrong argument order.'''

  def test_range_excludes_weekends(self):
    days = TradingCalendar().trading_days(date(2026, 1, 26),
                                          date(2026, 2, 1))
    assert days == [date(2026, 1, 26), date(2026, 1, 27),
                    date(2026, 1, 28), date(2026, 1, 29), date(2026, 1, 30)]

  def test_range_excludes_injected_holidays(self):
    calendar = TradingCalendar(holidays=[date(2026, 1, 27)])
    assert date(2026, 1, 27) not in calendar.trading_days(
      date(2026, 1, 26), date(2026, 1, 30))

  def test_inverted_range_is_empty_not_an_error(self):
    assert not TradingCalendar().trading_days(date(2026, 2, 1),
                                              date(2026, 1, 1))

  def test_range_longer_than_the_guard_raises(self):
    with pytest.raises(ValueError, match='guard'):
      TradingCalendar().trading_days(date(2000, 1, 1), date(2026, 1, 1))

  def test_guard_is_about_ten_years(self):
    assert max_range_days == 3650

  def test_iterator_agrees_with_the_list(self):
    calendar = TradingCalendar(holidays=[date(2026, 1, 27)])
    span = (date(2026, 1, 26), date(2026, 2, 6))
    assert list(calendar.iter_trading_days(*span)) == (
      calendar.trading_days(*span))


class TestNextAndPrevious:
  '''Stepping to the neighbouring trading day.'''

  def test_next_day_skips_the_weekend(self):
    assert TradingCalendar().next_trading_day(
      date(2026, 1, 30)) == date(2026, 2, 2)

  def test_inclusive_next_returns_the_day_itself(self):
    assert TradingCalendar().next_trading_day(
      WEDNESDAY, inclusive=True) == WEDNESDAY

  def test_previous_day_skips_the_weekend(self):
    assert TradingCalendar().previous_trading_day(
      date(2026, 2, 2)) == date(2026, 1, 30)

  def test_inclusive_previous_returns_the_day_itself(self):
    assert TradingCalendar().previous_trading_day(
      WEDNESDAY, inclusive=True) == WEDNESDAY

  def test_stepping_steps_over_an_injected_holiday(self):
    calendar = TradingCalendar(holidays=[date(2026, 1, 29)])
    assert calendar.next_trading_day(date(2026, 1, 28)) == date(
      2026, 1, 30)

  def test_search_horizon_is_bounded(self):
    # An under-specified holiday list can make the future permanently
    # un-trading; a caller waiting on that result would hang.
    assert search_horizon_days == 730

  def test_search_hands_up_rather_than_hanging_on_a_closed_window(self):
    # An under-specified holiday list can close an unbounded stretch of
    # the calendar; the search has to give up rather than spin.
    window = [date(2026, 1, 1) + timedelta(days=offset)
              for offset in range(search_horizon_days + 10)]
    calendar = TradingCalendar(holidays=window)
    assert calendar.next_trading_day(date(2026, 1, 1)) is None
    # Searching backwards from inside the window finds the day before it,
    # which is a real trading day -- the bound is a bound, not a lie.
    assert calendar.previous_trading_day(date(2026, 6, 1)) == date(
      2025, 12, 31)

  def test_closed_window_gives_up_searching_backwards_too(self):
    window = [date(2026, 1, 1) - timedelta(days=offset)
              for offset in range(search_horizon_days + 10)]
    calendar = TradingCalendar(holidays=window)
    assert calendar.previous_trading_day(date(2026, 1, 1)) is None


class TestCoerceDate:
  '''Dates arrive as text from circulars and spreadsheets.'''

  @pytest.mark.parametrize('value,expected', [
      ('2026-01-26', date(2026, 1, 26)),
      ('  2026-01-26  ', date(2026, 1, 26)),
      ('20260126', date(2026, 1, 26)),
      (date(2026, 1, 26), date(2026, 1, 26)),
      (datetime(2026, 1, 26, 9, 15), date(2026, 1, 26)),
  ])
  def test_supported_inputs(self, value, expected):
    assert coerce_date(value) == expected

  def test_unparseable_string_raises(self):
    with pytest.raises(ValueError):
      coerce_date('Republic Day')

  def test_unsupported_type_raises(self):
    with pytest.raises(TypeError, match='cannot read a date'):
      coerce_date(20260126)


class TestEntitlements:
  '''Entitlements are per segment, which is not the same as a boolean.'''

  def test_granted_covers_only_the_named_segments(self):
    entitlement = Entitlement.granted('a-feed', Segment.CASH)
    assert entitlement.allows(Segment.CASH) is True
    assert entitlement.allows(Segment.INDEX) is False

  def test_index_is_licensed_separately_from_cash(self):
    # Nifty constituents are cash-market securities, yet an index feed
    # is licensed separately, so a cash grant must not imply index.
    entitlement = Entitlement.granted('a-feed', Segment.CASH)
    assert Segment.CASH in entitlement.segments
    assert Segment.INDEX not in entitlement.segments

  def test_all_five_segments_are_named(self):
    # The exchange's own abbreviations, which is what appears on an
    # invoice and can be reconciled against the paperwork.
    assert {segment.value for segment in Segment} == {
      'CM', 'F&O', 'CD', 'Debt', 'Index'}

  def test_none_is_the_honest_default(self):
    # Safer than guessing CM: a wrong guess produces licensed-looking
    # data from an unlicensed source.
    assert Entitlement.none('unread').segments == frozenset()
    assert Entitlement.none('unread').allows(Segment.CASH) is False

  def test_missing_computes_the_gap(self):
    entitlement = Entitlement.granted('a-feed', Segment.CASH)
    assert entitlement.missing([Segment.CASH, Segment.DEBT]) == frozenset(
      {Segment.DEBT})

  def test_require_passes_when_everything_is_licensed(self):
    Entitlement.granted('a-feed', Segment.CASH).require(Segment.CASH)

  def test_require_raises_and_names_the_absent_segments(self):
    entitlement = Entitlement.granted('a-feed', Segment.CASH)
    with pytest.raises(EntitlementError, match='Index'):
      entitlement.require(Segment.CASH, Segment.INDEX)

  def test_entitlement_is_frozen(self):
    # A licence fact must not be mutable in place, or the record of what
    # was licensed would disagree with what the code does.
    entitlement = Entitlement.granted('a-feed', Segment.CASH)
    with pytest.raises(AttributeError):
      entitlement.segments = frozenset({Segment.DEBT})


class TestReplayVendor:
  '''Bars served from memory, with the vendor contract enforced.'''

  def test_conforms_to_the_feed_protocol(self):
    # Asserted rather than assumed: the point of a Protocol is that an
    # adapter can be checked against it.
    assert isinstance(vendor(), MarketDataFeed)

  def test_returns_the_bars_it_was_given(self):
    replay = vendor()
    served = replay.bars('RELIANCE', datetime(2026, 1, 28, 9, 15),
                         datetime(2026, 1, 28, 9, 24))
    assert [bar.close for bar in served] == [
      100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]

  def test_range_bounds_are_inclusive(self):
    replay = vendor()
    served = replay.bars('RELIANCE', datetime(2026, 1, 28, 9, 15),
                         datetime(2026, 1, 28, 9, 15))
    assert len(served) == 1

  def test_range_excludes_bars_outside_the_window(self):
    replay = vendor()
    served = replay.bars('RELIANCE', datetime(2026, 1, 28, 9, 17),
                         datetime(2026, 1, 28, 9, 18))
    assert len(served) == 2

  def test_empty_range_is_empty_not_an_error(self):
    replay = vendor()
    served = replay.bars('RELIANCE', datetime(2026, 1, 28, 20, 0),
                         datetime(2026, 1, 28, 21, 0))
    assert served == []

  def test_unknown_symbol_raises_rather_than_returning_nothing(self):
    # An empty list is exactly what a backtest silently completes on, and
    # a missing symbol is the likeliest ingestion mistake there is.
    with pytest.raises(KeyError, match='does not carry'):
      vendor().bars('NOPE', datetime(2026, 1, 28), datetime(2026, 1, 29))

  def test_inverted_range_raises(self):
    replay = vendor()
    with pytest.raises(ValueError, match='is after end'):
      replay.bars('RELIANCE', datetime(2026, 1, 29), datetime(2026, 1, 28))

  def test_returned_list_is_a_copy(self):
    # So a caller cannot corrupt the replay source it was handed.
    replay = vendor()
    served = replay.bars('RELIANCE', datetime(2026, 1, 28),
                         datetime(2026, 1, 28, 23, 59))
    served.clear()
    assert len(replay.bars('RELIANCE', datetime(2026, 1, 28),
                           datetime(2026, 1, 28, 23, 59))) == 10

  def test_symbols_are_sorted(self):
    replay = vendor(bars_by_symbol={
      'ZZZ': bars(2),
      'AAA': bars(2),
      'MMM': bars(2),
    })
    assert replay.symbols() == ['AAA', 'MMM', 'ZZZ']

  def test_has_symbol_distinguishes_absence_from_emptiness(self):
    replay = vendor()
    assert replay.has_symbol('RELIANCE') is True
    assert replay.has_symbol('NOPE') is False
    assert replay.last_bar_at('NOPE') is None

  def test_last_bar_at_is_the_newest(self):
    assert vendor().last_bar_at('RELIANCE') == datetime(2026, 1, 28, 9, 24)

  def test_non_ascending_bars_are_refused(self):
    # Checked, not sorted: a scrambled series must fail loudly rather
    # than being silently reordered into a plausible history.
    scrambled = list(reversed(bars(4)))
    with pytest.raises(ValueError, match='strictly ascending'):
      vendor(bars_by_symbol={'SCRAMBLED': scrambled})

  def test_duplicate_timestamps_are_refused(self):
    duplicated = bars(3) + [bars(3)[-1]]
    with pytest.raises(ValueError, match='strictly ascending'):
      vendor(bars_by_symbol={'DUP': duplicated})


class TestReplayVendorConstruction:
  '''Constructor guards, all of which fail loudly.'''

  def test_empty_vendor_name_raises(self):
    with pytest.raises(ValueError, match='vendor_name'):
      vendor(vendor_name='   ')

  def test_no_bars_raises(self):
    with pytest.raises(ValueError, match='at least one'):
      vendor(bars_by_symbol={})

  def test_non_positive_stale_bound_raises(self):
    with pytest.raises(ValueError, match='stale_after'):
      vendor(stale_after=timedelta(0))

  def test_default_entitlement_is_none(self):
    # Not cash: assuming a licence the caller has not shown is the
    # failure that costs money.
    assert vendor().entitlements.allows(Segment.CASH) is False

  def test_entitlement_is_carried_through(self):
    replay = vendor(
      entitlements=Entitlement.granted('test-replay', Segment.CASH))
    assert replay.entitlements.allows(Segment.CASH) is True
    assert replay.vendor_name == 'test-replay'


class TestStaleness:
  '''A dead feed must be distinguishable from a slow one.'''

  def test_fresh_bar_is_not_stale(self):
    replay = vendor(clock=lambda: datetime(2026, 1, 28, 9, 24, 5))
    assert replay.is_stale('RELIANCE') is False

  def test_bar_older_than_the_bound_is_stale(self):
    replay = vendor(clock=lambda: datetime(2026, 1, 28, 9, 25))
    assert replay.is_stale('RELIANCE') is True

  def test_bound_is_inclusive(self):
    replay = vendor(clock=lambda: datetime(2026, 1, 28, 9, 24, 30))
    assert replay.is_stale('RELIANCE') is True

  def test_default_bound_is_thirty_seconds(self):
    # Generous by two orders of magnitude against a 150 to 400 ms round
    # trip, which is the point: loose enough never to fire in normal
    # operation, tight enough to catch a feed that stopped updating.
    assert vendor().stale_after == timedelta(seconds=30)

  def test_age_is_measured_from_the_newest_bar(self):
    replay = vendor()
    assert replay.age('RELIANCE', datetime(2026, 1, 28, 9, 30)) == (
      timedelta(minutes=6))

  def test_per_call_bound_overrides_the_default(self):
    replay = vendor(clock=lambda: datetime(2026, 1, 28, 9, 24, 5))
    assert replay.is_stale('RELIANCE', stale_after=timedelta(seconds=1))

  def test_future_bar_reports_zero_age(self):
    # A bar stamped in the future is a clock problem, not a negative age.
    replay = vendor()
    assert replay.age('RELIANCE', datetime(2026, 1, 28, 9, 0)) == (
      timedelta(0))

  def test_unknown_symbol_is_not_stale(self):
    # Absence is a caller error to be raised elsewhere, not a stale
    # quote: conflating them would report a live feed as dead.
    replay = vendor()
    assert replay.age('NOPE') is None
    assert replay.is_stale('NOPE') is False

  def test_stale_symbols_lists_only_the_stale_ones(self):
    # Polling symbol by symbol and acting between polls is how a
    # partially-stale panel produces an unfillable backtest, so the
    # decision has to be available for the whole panel at once.
    now = datetime(2026, 1, 28, 9, 40)
    replay = vendor(bars_by_symbol={
      'LIVE': bars(3, start=time(9, 38)),
      'DEAD': bars(3, start=time(9, 15)),
      'ALSO_LIVE': bars(2, start=time(9, 39)),
    }, clock=lambda: now)
    assert replay.stale_symbols() == ['DEAD']

  def test_stale_symbols_is_empty_when_the_panel_is_current(self):
    now = datetime(2026, 1, 28, 9, 17, 5)
    replay = vendor(bars_by_symbol={'A': bars(3), 'B': bars(3)},
                    clock=lambda: now)
    assert replay.stale_symbols() == []


class TestAwareness:
  '''Naive and aware timestamps must not be mixed silently.'''

  def test_aware_now_is_read_as_exchange_local_time(self):
    # Naive means exchange local time throughout this project, so an
    # aware instant is read in its own wall time rather than refused.
    aware = datetime(2026, 1, 28, 9, 30, tzinfo=utc())
    replay = vendor()
    assert replay.age('RELIANCE', aware) == timedelta(minutes=6)

  def test_naive_now_against_aware_bars_raises(self):
    # Which zone to use cannot be inferred, so it is refused rather than
    # guessed.
    aware_bars = [Bar(datetime(2026, 1, 28, 9, 15, tzinfo=utc()),
                      1.0, 1.0, 1.0, 1.0)]
    replay = vendor(bars_by_symbol={'RELIANCE': aware_bars})
    with pytest.raises(ValueError, match='timezone-aware'):
      replay.age('RELIANCE', datetime(2026, 1, 28, 9, 30))


def utc():
  '''Return UTC, aliased so the awareness tests read clearly.

  Returns:
    The UTC timezone.
  '''
  return timezone.utc


def strings_outside_docstrings(path):
  '''Return every string literal in a module that is not a docstring.

  Args:
    path: Path to a Python source file.

  Returns:
    The text of all non-docstring string literals, lowercased and joined.
    Docstrings are excluded because they are where this project's prose
    names a thing in order to rule it out, and executable code is where a
    thing gets used.
  '''
  tree = ast.parse(path.read_text(encoding='utf-8'))
  docstrings = set()
  for node in ast.walk(tree):
    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                         ast.AsyncFunctionDef)):
      text = ast.get_docstring(node, clean=False)
      if text is not None:
        docstrings.add(text)
  literals = [
    node.value for node in ast.walk(tree)
    if isinstance(node, ast.Constant) and isinstance(node.value, str)
    and node.value not in docstrings
  ]
  return ' '.join(literals).lower()


class TestVendorNamesAreNotHardcoded:
  '''The authorised vendor list changes, so no name is a source of truth.

  Two real vendor names are recorded in the research and neither may be
  validated against a hardcoded list here. The fabricated vendor and the
  invented case must not reach executable code at all, which is worth
  asserting mechanically rather than trusting a reviewer's eye.
  '''

  def test_invented_vendor_name_is_used_as_a_vendor_nowhere(self):
    # Naming it in prose in order to exclude it is fine. Passing it to
    # anything executable is not, and that is what this asserts: the
    # source is parsed and only non-docstring literals are searched.
    root = Path(stock_rl.__file__).parent
    offenders = [
      path.name for path in sorted(root.rglob('*.py'))
      if 'investingnote' in strings_outside_docstrings(path)
    ]
    assert offenders == []

  def test_invented_case_is_used_as_authority_nowhere(self):
    # "Indian Express v. Zeal" does not exist. Same rule: prose may
    # exclude it, code may not cite it.
    root = Path(stock_rl.__file__).parent
    offenders = [
      path.name for path in sorted(root.rglob('*.py'))
      if 'zeal' in strings_outside_docstrings(path)
    ]
    assert offenders == []

  def test_real_vendor_names_are_recorded_in_the_module_docstring(self):
    # Whitespace-normalised, so rewrapping the prose cannot break a test
    # about which names it records.
    text = ' '.join(vendors.__doc__.split())
    assert 'TrueData Financial Information Pvt Ltd' in text
    assert 'Global Financial Datafeeds LLP' in text
    assert 'TradingView Inc.' in text
    assert 'nseindia.com/market-data/real-time-data-subscription' in text

  def test_entitlements_are_documented_as_per_segment(self):
    text = ' '.join(vendors.__doc__.split())
    assert 'per segment, not a boolean' in text
