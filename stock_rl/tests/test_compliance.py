#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for the data acquisition compliance controls.

Two properties are asserted here that a reviewer cannot otherwise check
by reading the code, and which are the whole reason these modules exist:

  * a disallowed path is actually refused, and the robots.txt fetch is
    cached rather than repeated per request;
  * the rate limiter's interval invariant holds **without any real
    sleeping**, because the clock is injected. If it ever regressed to
    ``time.sleep`` these tests would take minutes instead of
    microseconds, which is itself the signal.

No test in this file performs network I/O. Every robots.txt body is
injected, so the suite is offline and deterministic.
'''

import urllib.request
from datetime import date, datetime
from urllib.error import HTTPError, URLError

import pytest

from stock_rl.compliance.algo_tag import (
  AlgoTagError,
  algo_flag_digit,
  allows_algo,
  build_nnf_id,
  client_direct_api_platform,
  documented_flag_digits,
  is_algo_flag,
  known_platform_algo_allowed,
  nnf_length,
  platform_of,
  unverified_flag_digits,
  validate_algo_tag,
  verified_flag_digits,
)
from stock_rl.compliance.ratelimit import (
  BackoffPolicy,
  RateLimiter,
  parse_retry_after,
)
from stock_rl.compliance.retention import (
  AlgoIdError,
  add_years,
  describe,
  expiry_date,
  is_expired,
  purge_candidates,
  registered_algo_ids,
  require_registration,
  retention_classes,
  retention_table,
  retention_years,
  source_of,
  unregistered_algo_id,
  validate_algo_id,
)
from stock_rl.compliance.robots import RobotsChecker, RobotsUnavailable

#: A robots.txt body exercising all three directives this module reads.
ROBOTS = '''User-agent: *
Disallow: /private
Crawl-delay: 2
'''


def fetcher_for(body=None, error=None):
  '''Return a robots.txt fetcher for a canned body or a canned failure.

  Args:
    body: Body to return, or None for a fetcher reporting no robots.txt.
    error: Exception to raise instead of returning, for the RFC 9309
      statuses that mean "assume complete disallow".

  Returns:
    A fetcher callable matching the module's ``Fetcher`` signature.
  '''
  def fetcher(url, timeout, user_agent):
    del url, timeout, user_agent
    if error is not None:
      raise error
    return body

  return fetcher


class FakeClock:
  '''A monotonic clock the test advances by hand.

  Args are documented on ``__init__``.
  '''

  def __init__(self, start=0.0):
    '''Build a clock.

    Args:
      start: Initial reading in seconds.
    '''
    self.now = start
    self.slept = []

  def __call__(self):
    '''Return the current reading.

    Returns:
      Seconds since the clock's own origin.
    '''
    return self.now

  def sleep(self, seconds):
    '''Record a sleep and advance the clock.

    Args:
      seconds: Duration requested by the limiter.
    '''
    self.slept.append(seconds)
    self.now += seconds

  def advance(self, seconds):
    '''Move the clock forward without recording a sleep.

    Args:
      seconds: Duration to advance.
    '''
    self.now += seconds


class TestRobotsDisallow:
  '''The one robots.txt behaviour that matters: refusing a path.'''

  def test_disallowed_path_is_refused(self):
    checker = RobotsChecker('stock-rl', fetcher=fetcher_for(ROBOTS))
    assert checker.must_fetch('https://example.com/public/quote') is True

  def test_blocked_path_returns_false(self):
    # The single most important assertion in this file. A robots.txt
    # checker that returns True for a disallowed path is worse than no
    # checker, because it records having checked.
    checker = RobotsChecker('stock-rl', fetcher=fetcher_for(ROBOTS))
    assert checker.must_fetch('https://example.com/private/quote') is False

  def test_disallow_matches_a_path_prefix_not_a_substring(self):
    # RFC 9309 matching is prefix-based, so "Disallow: /private" does not
    # reach "/a/b/private/c". Asserting the real semantics matters: a
    # checker that claimed to be stricter than the standard would reject
    # paths the origin never disallowed.
    checker = RobotsChecker('stock-rl', fetcher=fetcher_for(ROBOTS))
    assert checker.must_fetch('https://example.com/private/c') is False
    assert checker.must_fetch('https://example.com/a/b/private/c') is True

  def test_explicit_allow_overrides_a_broader_disallow(self):
    body = ('User-agent: *\nDisallow: /data\n'
            'Allow: /data/public\n')
    checker = RobotsChecker('stock-rl', fetcher=fetcher_for(body))
    assert checker.must_fetch('https://example.com/data/secret') is False
    assert checker.must_fetch('https://example.com/data/public/x') is True

  def test_robots_txt_itself_is_always_allowed(self):
    # RFC 9309: /robots.txt is implicitly allowed, or a checker could
    # lock itself out of the file that governs it.
    checker = RobotsChecker('stock-rl', fetcher=fetcher_for(ROBOTS))
    assert checker.must_fetch('https://example.com/robots.txt') is True

  def test_different_origin_gets_its_own_file(self):
    first = 'User-agent: *\nDisallow: /private\n'
    bodies = {'https://example.com': first, 'https://other.com': ''}
    seen = []

    def fetcher(url, timeout, user_agent):
      del timeout, user_agent
      seen.append(url)
      for origin, body in bodies.items():
        if url.startswith(origin):
          return body
      return None

    checker = RobotsChecker('stock-rl', fetcher=fetcher)
    assert checker.must_fetch('https://example.com/private/x') is False
    assert checker.must_fetch('https://other.com/private/x') is True
    assert len(seen) == 2


class TestRobotsCaching:
  '''The cache is what keeps a courtesy check from becoming the load.'''

  def test_one_fetch_per_origin_however_many_lookups(self):
    calls = []
    checker = RobotsChecker('stock-rl', fetcher=fetcher_of(calls))
    for path in ('/a', '/b', '/c', '/d'):
      checker.must_fetch(f'https://example.com{path}')
      checker.crawl_delay(f'https://example.com{path}')
    assert checker.fetch_count == 1
    assert len(calls) == 1

  def test_clear_cache_refetches(self):
    calls = []
    checker = RobotsChecker('stock-rl', fetcher=fetcher_of(calls))
    checker.must_fetch('https://example.com/a')
    assert checker.cached_origins == ('https://example.com',)
    checker.clear_cache()
    checker.must_fetch('https://example.com/a')
    assert checker.fetch_count == 2
    assert checker.cached_origins == ('https://example.com',)

  def test_user_agent_and_timeout_reach_the_fetcher(self):
    calls = []
    checker = RobotsChecker('stock-rl/0.1', fetcher=fetcher_of(calls),
                            timeout=2.5)
    checker.must_fetch('https://example.com/a')
    # A default urllib User-Agent identifies as a bot and several origins
    # refuse it outright, so the real token must go on the wire.
    assert calls == [('https://example.com/robots.txt', 2.5, 'stock-rl/0.1')]

  def test_cached_origin_is_lowercased_for_case_insensitive_hosts(self):
    calls = []
    checker = RobotsChecker('stock-rl', fetcher=fetcher_of(calls))
    checker.must_fetch('https://Example.COM/a')
    checker.must_fetch('https://example.com/b')
    assert len(calls) == 1
    assert checker.cached_origins == ('https://example.com',)


def fetcher_of(calls, body=ROBOTS):
  '''Return a fetcher recording full call tuples and returning ``body``.

  Recording all three arguments, rather than just the URL, is what lets a
  test assert the timeout and user agent actually reach the fetcher
  rather than being silently dropped on the way down.

  Args:
    calls: List to record ``(url, timeout, user_agent)`` tuples in.
    body: Body to return.

  Returns:
    A fetcher callable matching the module's ``Fetcher`` signature.
  '''
  def fetcher(url, timeout, user_agent):
    calls.append((url, timeout, user_agent))
    return body

  return fetcher


class TestRobotsCrawlDelay:
  '''Crawl-delay is honoured where published, and absence is explicit.'''

  def test_declared_delay_is_returned(self):
    checker = RobotsChecker('stock-rl', fetcher=fetcher_for(ROBOTS))
    assert checker.crawl_delay('https://example.com/x') == 2.0

  def test_absent_delay_is_none_not_zero(self):
    # None means "no instruction given". Zero would assert that the
    # origin asked for no delay, which is a different claim and would
    # silently licence hammering.
    checker = RobotsChecker('stock-rl', fetcher=fetcher_for(''))
    assert checker.crawl_delay('https://example.com/x') is None

  def test_agent_specific_group_is_matched(self):
    body = ('User-agent: stock-rl\nCrawl-delay: 7\n\n'
            'User-agent: *\nCrawl-delay: 1\n')
    checker = RobotsChecker('stock-rl', fetcher=fetcher_for(body))
    assert checker.crawl_delay('https://example.com/x') == 7.0


class TestRobotsUnavailable:
  '''RFC 9309's asymmetric treatment of a missing versus failing file.'''

  def test_missing_file_permits_everything(self):
    # 4xx other than 401/403 means the resource does not exist, so the
    # crawler may fetch.
    checker = RobotsChecker('stock-rl', fetcher=fetcher_for(None))
    assert checker.must_fetch('https://example.com/private/x') is True

  def test_server_error_disallows_everything(self):
    checker = RobotsChecker(
      'stock-rl', fetcher=fetcher_for(error=RobotsUnavailable('503')))
    assert checker.must_fetch('https://example.com/public/x') is False

  def test_negative_result_is_cached_too(self):
    # A dead origin must be asked once, not once per request: a client
    # that retries robots.txt as often as it retries pages is the load
    # robots.txt exists to avoid.
    calls = []
    checker = RobotsChecker('stock-rl', fetcher=raiser(calls))
    assert checker.must_fetch('https://example.com/a') is False
    assert checker.must_fetch('https://example.com/b') is False
    assert checker.fetch_count == 1
    assert len(calls) == 1
    assert calls == ['https://example.com/robots.txt']


def raiser(calls):
  '''Return a fetcher that records the call and always raises.

  Args:
    calls: List to record calls in.

  Returns:
    A fetcher callable raising RobotsUnavailable.
  '''
  def fetcher(url, timeout, user_agent):
    del timeout, user_agent
    calls.append(url)
    raise RobotsUnavailable('unavailable')

  return fetcher


class TestDefaultFetcher:
  '''The real HTTP fetcher, exercised against a stubbed urlopen.

  The stub is what keeps this offline: ``urlopen`` is the only thing that
  would touch a socket, and it is replaced rather than mocked around.
  '''

  def stub_urlopen(self, monkeypatch, outcome):
    '''Replace ``urlopen`` with a canned outcome.

    Args:
      monkeypatch: pytest monkeypatch fixture.
      outcome: Either bytes to return, or an exception instance to raise.

    Returns:
      The list of calls the stub recorded.
    '''
    calls = []

    class Response:
      '''Minimal context-manager response stand-in.'''

      def __init__(self, body):
        self._body = body

      def read(self):
        '''Return the canned body.

        Returns:
          The bytes the stub was configured with.
        '''
        return self._body

      def __enter__(self):
        '''Enter the response context.

        Returns:
          This response.
        '''
        return self

      def __exit__(self, *exc_info):
        '''Leave the response context.

        Args:
          exc_info: Unused exception triple.

        Returns:
          False, so an exception is never suppressed.
        '''
        return False

    def urlopen(request, timeout=None):
      calls.append((request.full_url, timeout, request.get_header(
        'User-agent')))
      if isinstance(outcome, BaseException):
        raise outcome
      return Response(outcome)

    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    return calls

  def test_body_is_returned(self, monkeypatch):
    calls = self.stub_urlopen(monkeypatch, b'User-agent: *\nDisallow: /x\n')
    checker = RobotsChecker('stock-rl')
    assert checker.must_fetch('https://example.com/x') is False
    assert calls == [('https://example.com/robots.txt', 10.0, 'stock-rl')]

  def test_non_200_404_is_treated_as_no_file(self, monkeypatch):
    # RFC 9309 s.2.3.1.3: a 4xx other than 401/403 means the resource is
    # not there, so the crawler may fetch.
    error = HTTPError('https://example.com/robots.txt', 404, 'Not Found',
                      {}, None)
    self.stub_urlopen(monkeypatch, error)
    checker = RobotsChecker('stock-rl')
    assert checker.must_fetch('https://example.com/private/x') is True

  @pytest.mark.parametrize('status', [401, 403, 500, 503])
  def test_401_403_and_5xx_disallow_everything(self, monkeypatch, status):
    self.stub_urlopen(monkeypatch, HTTPError(
      'https://example.com/robots.txt', status, 'nope', {}, None))
    checker = RobotsChecker('stock-rl')
    assert checker.must_fetch('https://example.com/public/x') is False

  def test_transport_failure_is_treated_as_no_file(self, monkeypatch):
    # An origin that is down, or a name that does not resolve, must not
    # take the pipeline down over a courtesy file.
    self.stub_urlopen(monkeypatch, URLError('name resolution failed'))
    checker = RobotsChecker('stock-rl')
    assert checker.must_fetch('https://example.com/private/x') is True

  def test_timeout_is_passed_through(self, monkeypatch):
    calls = self.stub_urlopen(monkeypatch, b'')
    RobotsChecker('stock-rl', timeout=3.5).must_fetch('https://example.com/a')
    assert calls[0][1] == 3.5


class TestRobotsValidation:
  '''Bad input fails at the door rather than mid-fetch.'''

  def test_relative_url_raises(self):
    checker = RobotsChecker('stock-rl', fetcher=fetcher_for(ROBOTS))
    with pytest.raises(ValueError, match='not an absolute URL'):
      checker.must_fetch('/private/x')

  def test_empty_user_agent_raises(self):
    with pytest.raises(ValueError, match='user_agent'):
      RobotsChecker('   ', fetcher=fetcher_for(ROBOTS))

  def test_non_positive_timeout_raises(self):
    with pytest.raises(ValueError, match='timeout'):
      RobotsChecker('stock-rl', timeout=0.0, fetcher=fetcher_for(ROBOTS))


class TestRateLimiterInterval:
  '''The min-interval invariant, with no real sleeping anywhere.'''

  def test_first_call_never_waits(self):
    clock = FakeClock()
    limiter = RateLimiter(1.0, clock=clock, sleep=clock.sleep,
                          jitter=lambda: 0.0)
    assert limiter.acquire('a.example') == 0.0
    assert not clock.slept

  def test_second_call_waits_the_remaining_interval(self):
    clock = FakeClock()
    limiter = RateLimiter(1.0, clock=clock, sleep=clock.sleep,
                          jitter=lambda: 0.0)
    limiter.acquire('a.example')
    clock.advance(0.25)
    assert limiter.acquire('a.example') == pytest.approx(0.75)
    assert clock.slept == [pytest.approx(0.75)]

  def test_interval_elapsed_needs_no_wait(self):
    clock = FakeClock()
    limiter = RateLimiter(1.0, clock=clock, sleep=clock.sleep,
                          jitter=lambda: 0.0)
    limiter.acquire('a.example')
    clock.advance(2.0)
    assert limiter.acquire('a.example') == 0.0

  def test_domains_are_paced_independently(self):
    # A global limiter would serialise every host behind the slowest
    # one, which is how a polite crawler becomes a slow crawler.
    clock = FakeClock()
    limiter = RateLimiter(1.0, clock=clock, sleep=clock.sleep,
                          jitter=lambda: 0.0)
    limiter.acquire('a.example')
    assert limiter.acquire('b.example') == 0.0

  def test_release_time_never_falls_below_previous_plus_interval(self):
    # The projected-release-time bookkeeping, asserted directly. This is
    # what makes an instantaneous injected sleeper safe rather than only
    # fast.
    clock = FakeClock()
    limiter = RateLimiter(2.5, clock=clock, sleep=lambda seconds: None,
                          jitter=lambda: 0.0)
    previous = None
    for _ in range(5):
      limiter.acquire('a.example')
      current = limiter.next_allowed('a.example')
      if previous is not None:
        assert current >= previous
      previous = current
    assert previous == 12.5

  def test_zero_interval_disables_pacing_but_not_429(self):
    clock = FakeClock()
    limiter = RateLimiter(0.0, clock=clock, sleep=clock.sleep,
                          jitter=lambda: 0.0)
    for _ in range(4):
      assert limiter.acquire('a.example') == 0.0
    assert limiter.acquire('a.example') == 0.0

  def test_empty_domain_raises(self):
    limiter = RateLimiter(clock=FakeClock(), sleep=lambda s: None,
                          jitter=lambda: 0.0)
    with pytest.raises(ValueError, match='domain'):
      limiter.acquire('   ')

  def test_next_allowed_is_none_for_an_unused_domain(self):
    limiter = RateLimiter(clock=FakeClock(), sleep=lambda s: None,
                          jitter=lambda: 0.0)
    assert limiter.next_allowed('never.example') is None
    assert limiter.strikes('never.example') == 0

  def test_suspension_alone_is_not_a_release_time(self):
    # A 429 with no request behind it creates bookkeeping but no release
    # time, because nothing has been let through yet.
    limiter = RateLimiter(clock=FakeClock(), sleep=lambda s: None,
                          jitter=lambda: 0.0)
    limiter.on_429('a.example')
    assert limiter.next_allowed('a.example') is None
    limiter.acquire('a.example')
    assert limiter.next_allowed('a.example') is not None


class TestBackoff:
  '''Backoff must grow, must be bounded, and must survive a server.'''

  def test_delay_doubles_per_strike(self):
    limiter = RateLimiter(jitter=lambda: 0.0)
    delays = [limiter.backoff_delay(attempt) for attempt in range(4)]
    assert delays == [1.0, 2.0, 4.0, 8.0]

  def test_backoff_is_bounded_by_the_ceiling(self):
    # An unbounded curve converts a rate limit into an outage: the client
    # gives up long before the origin would have let it back in.
    policy = BackoffPolicy(base_seconds=1.0, max_seconds=10.0,
                           jitter_ratio=0.0)
    limiter = RateLimiter(backoff=policy, jitter=lambda: 0.0)
    for attempt in range(50):
      assert limiter.backoff_delay(attempt) <= 10.0
    assert limiter.backoff_delay(50) == 10.0

  def test_huge_attempt_does_not_overflow(self):
    limiter = RateLimiter(backoff=BackoffPolicy(max_seconds=60.0),
                          jitter=lambda: 0.0)
    assert limiter.backoff_delay(10_000) == 60.0

  def test_jitter_only_ever_adds_and_stays_within_its_bound(self):
    policy = BackoffPolicy(base_seconds=1.0, max_seconds=10.0,
                           jitter_ratio=0.5)
    floor = RateLimiter(backoff=policy, jitter=lambda: 0.0)
    ceiling = RateLimiter(backoff=policy, jitter=lambda: 1.0)
    for attempt in range(8):
      base = floor.backoff_delay(attempt)
      assert ceiling.backoff_delay(attempt) == pytest.approx(base * 1.5)
      assert ceiling.backoff_delay(attempt) <= 10.0 * 1.5

  def test_jitter_actually_varies_between_calls(self):
    # Two clients must not retry in lockstep, which is the whole reason
    # jitter exists. A fixed jitter factor would reintroduce the thundering
    # herd it was added to prevent.
    values = iter((0.0, 1.0, 0.25, 0.75))
    limiter = RateLimiter(jitter=lambda: next(values))
    delays = {limiter.backoff_delay(2) for _ in range(4)}
    assert len(delays) == 4

  def test_negative_attempt_raises(self):
    limiter = RateLimiter(jitter=lambda: 0.0)
    with pytest.raises(ValueError, match='attempt'):
      limiter.backoff_delay(-1)

  def test_retry_after_longer_than_backoff_wins(self):
    limiter = RateLimiter(jitter=lambda: 0.0)
    assert limiter.backoff_delay(0, retry_after=30.0) == 30.0

  def test_retry_after_shorter_than_backoff_is_ignored(self):
    limiter = RateLimiter(jitter=lambda: 0.0)
    assert limiter.backoff_delay(5, retry_after=1.0) == 32.0

  def test_invalid_policy_is_refused_at_construction(self):
    with pytest.raises(ValueError, match='base'):
      RateLimiter(backoff=BackoffPolicy(base_seconds=0.0))
    with pytest.raises(ValueError, match='ceiling'):
      RateLimiter(backoff=BackoffPolicy(base_seconds=10.0, max_seconds=1.0))
    with pytest.raises(ValueError, match='jitter_ratio'):
      RateLimiter(backoff=BackoffPolicy(jitter_ratio=-0.1))
    with pytest.raises(ValueError, match='min_interval'):
      RateLimiter(min_interval=-1.0)


class TestOn429:
  '''A 429 must suspend the domain, and success must clear the strike.'''

  def test_429_suspends_the_domain(self):
    clock = FakeClock()
    limiter = RateLimiter(1.0, clock=clock, sleep=clock.sleep,
                          jitter=lambda: 0.0)
    limiter.acquire('a.example')
    limiter.on_429('a.example')
    # The suspension dominates the min interval from here on.
    assert limiter.acquire('a.example') == pytest.approx(1.0)
    assert clock.slept

  def test_strikes_accumulate_and_grow_the_delay(self):
    clock = FakeClock()
    limiter = RateLimiter(1.0, clock=clock, sleep=clock.sleep,
                          jitter=lambda: 0.0)
    first = limiter.on_429('a.example')
    second = limiter.on_429('a.example')
    assert second == pytest.approx(first * 2)
    assert limiter.strikes('a.example') == 2

  def test_success_clears_the_strike_count(self):
    clock = FakeClock()
    limiter = RateLimiter(1.0, clock=clock, sleep=clock.sleep,
                          jitter=lambda: 0.0)
    limiter.on_429('a.example')
    limiter.note_success('a.example')
    assert limiter.strikes('a.example') == 0
    assert limiter.backoff_delay(0) == 1.0

  def test_note_success_on_an_unknown_domain_is_a_no_op(self):
    limiter = RateLimiter(jitter=lambda: 0.0)
    limiter.note_success('never.example')
    assert limiter.strikes('never.example') == 0

  def test_suspension_does_not_move_backwards_on_a_short_retry_after(self):
    clock = FakeClock()
    limiter = RateLimiter(1.0, clock=clock, sleep=clock.sleep,
                          jitter=lambda: 0.0)
    limiter.on_429('a.example', retry_after=10.0)
    clock.advance(1.0)
    # A shorter instruction on a second strike must not shorten the
    # existing suspension, or the client could be released by an origin
    # that is still refusing it.
    limiter.on_429('a.example', retry_after=0.5)
    assert limiter.acquire('a.example') == pytest.approx(9.0)

  def test_empty_domain_and_negative_retry_after_raise(self):
    limiter = RateLimiter(jitter=lambda: 0.0)
    with pytest.raises(ValueError, match='domain'):
      limiter.on_429('')
    with pytest.raises(ValueError, match='retry_after'):
      limiter.on_429('a.example', retry_after=-1.0)
    with pytest.raises(ValueError, match='retry_after'):
      limiter.backoff_delay(0, retry_after=-1.0)


class TestParseRetryAfter:
  '''Both RFC 9110 forms, and a refusal to guess the rest.'''

  def test_delta_seconds(self):
    assert parse_retry_after('120') == 120.0

  def test_whitespace_is_tolerated(self):
    assert parse_retry_after('  30  ') == 30.0

  def test_http_date_in_the_future(self):
    # A date is honoured when it is in the future and refused when it is
    # not, so a stale header cannot hold a client for a negative duration.
    value = parse_retry_after('Wed, 21 Oct 2099 07:28:00 GMT')
    assert value is not None and value > 0.0

  def test_http_date_in_the_past_is_none(self):
    assert parse_retry_after('Wed, 21 Oct 2000 07:28:00 GMT') is None

  def test_date_without_a_zone_is_read_as_gmt(self):
    # RFC 9110 says a recipient should read a zone-less HTTP-date as GMT.
    # Refusing it would lose a legitimate server instruction.
    value = parse_retry_after('Wed, 21 Oct 2099 07:28:00')
    assert value is not None and value > 0.0

  def test_date_with_an_offset_is_honoured(self):
    # An origin may answer in its own zone; the instruction is absolute,
    # not wall-clock, so the offset only shifts the instant.
    zoned = parse_retry_after('Wed, 21 Oct 2099 02:28:00 -0500')
    gmt = parse_retry_after('Wed, 21 Oct 2099 07:28:00 GMT')
    assert zoned == pytest.approx(gmt)

  def test_garbage_is_none_not_an_exception(self):
    for value in ('', '   ', 'soon', '-5'):
      assert parse_retry_after(value) is None

  def test_zero_seconds_is_a_valid_instruction(self):
    assert parse_retry_after('0') == 0.0


class TestRetentionPeriods:
  '''Every figure must be traced, and the unsourced one must stay gone.'''

  def test_five_years_is_the_nse_exchange_floor(self):
    assert retention_years('exchange_audit_floor') == 5
    assert '10.3' in source_of('exchange_audit_floor')

  def test_eight_years_is_the_sebi_brokers_regulation(self):
    assert retention_years('books_of_account') == 8
    assert 'Reg. 16' in source_of('books_of_account')

  def test_algo_trail_is_held_at_eight_not_five(self):
    # Holding the trail at the 5-year exchange floor would satisfy NSE
    # but sit inside the 8-year books-of-account obligation, which
    # captures the trail too. Eight satisfies both.
    assert retention_years('algo_audit_trail') == 8
    assert retention_years('algo_audit_trail') >= retention_years(
      'exchange_audit_floor')

  def test_no_seven_year_period_exists_anywhere(self):
    # Regression guard on the single most-copied number in this area.
    # "7 years as per SEBI" has no source: not Reg. 16, not the 2012 algo
    # circular, not the NSE modalities.
    assert 7 not in {entry.years for entry in retention_table}
    for name in retention_classes:
      assert retention_years(name) != 7

  def test_every_finite_class_is_five_or_eight_years(self):
    assert sorted({retention_years(name) for name in retention_classes}) == [
      5, 8]

  def test_every_entry_names_its_source(self):
    for entry in retention_table:
      assert entry.source.strip()
      assert entry.note.strip()

  def test_unknown_class_raises_rather_than_defaulting(self):
    # A default retention period is how records get deleted too early.
    with pytest.raises(KeyError, match='unknown record class'):
      retention_years('made_up_records')
    with pytest.raises(KeyError):
      source_of('made_up_records')
    with pytest.raises(KeyError):
      expiry_date('made_up_records', date(2020, 1, 1))
    with pytest.raises(KeyError):
      purge_candidates('made_up_records', {}, date(2020, 1, 1))

  def test_describe_names_the_period_and_its_source(self):
    line = describe('exchange_audit_floor')
    assert '5 years' in line and '10.3' in line
    assert 'indefinite' in describe('enforcement_original')
    with pytest.raises(KeyError):
      describe('made_up_records')

  def test_indefinite_class_is_not_in_the_finite_list(self):
    assert 'enforcement_original' not in retention_classes
    assert retention_years('enforcement_original') is None


class TestExpiryMath:
  '''Expiry is calendar arithmetic, inclusive, and leap-safe.'''

  def test_expiry_is_the_anniversary(self):
    assert expiry_date('algo_audit_trail', day(2020, 3, 1)) == day(
      2028, 3, 1)

  def test_expiry_accepts_a_datetime(self):
    created = datetime(2020, 3, 1, 15, 29)
    assert expiry_date('algo_audit_trail', created) == day(2028, 3, 1)

  def test_five_year_floor_expires_five_years_out(self):
    assert expiry_date('exchange_audit_floor', day(2021, 1, 4)) == day(
      2026, 1, 4)

  def test_leap_day_record_clamps_rather_than_raising(self):
    # 2024-02-29 plus eight years is 2032, also a leap year, so the
    # interesting case is a period landing on a non-leap year.
    assert expiry_date('exchange_audit_floor', day(2024, 2, 29)) == day(
      2029, 2, 28)

  def test_expiry_is_inclusive_of_the_end_date(self):
    assert is_expired('algo_audit_trail', day(2020, 3, 1), day(2028, 3, 1))
    assert not is_expired('algo_audit_trail', day(2020, 3, 1),
                          day(2028, 2, 29))

  def test_indefinite_records_never_expire_and_have_no_expiry_date(self):
    # None here is a real answer, not an error: enforcement originals are
    # preserved indefinitely, and a caller must be able to flow that
    # through a purge loop without a special case.
    assert expiry_date('enforcement_original', day(2000, 1, 1)) is None
    assert not is_expired('enforcement_original', day(2000, 1, 1),
                          day(2099, 1, 1))

  def test_purge_candidates_lists_only_expired_keys(self):
    records = {
      'old': day(2010, 1, 1),
      'recent': day(2024, 1, 1),
      'boundary': day(2018, 3, 1),
    }
    assert purge_candidates('algo_audit_trail', records, day(2026, 3, 1)) == [
      'boundary', 'old']

  def test_purge_candidates_is_empty_for_indefinite_records(self):
    records = {'ancient': day(1900, 1, 1)}
    assert purge_candidates('enforcement_original', records,
                            day(2099, 1, 1)) == []

  def test_negative_years_are_refused(self):
    with pytest.raises(ValueError, match='years'):
      add_years(day(2020, 1, 1), -1)

  def test_add_years_is_the_shared_calendar_helper(self):
    # Public because a caller with a custom retention period needs the
    # same arithmetic, and reimplementing it as a 365-day duration is
    # exactly the drift this module exists to avoid.
    assert add_years(day(2024, 2, 29), 1) == day(2025, 2, 28)
    assert add_years(day(2024, 1, 1), 0) == day(2024, 1, 1)


def day(year, month, day_number):
  '''Return a date, kept short because the tests use many.

  Args:
    year: Calendar year.
    month: Calendar month.
    day_number: Day of month.

  Returns:
    The constructed date.
  '''
  return date(year, month, day_number)


class TestAlgoIdValidation:
  '''An Algo ID is an opaque string, and a blank one is a hard stop.'''

  def test_empty_algo_id_is_rejected(self):
    registered = registered_algo_ids({'4242'})
    with pytest.raises(AlgoIdError, match='empty'):
      validate_algo_id('', registered)

  def test_whitespace_algo_id_is_rejected(self):
    # Regression guard: a whitespace-only field must never be coerced to
    # the non-algo sentinel, which would turn an unset Algo ID into a
    # claim that no algo ran.
    registered = registered_algo_ids({'4242'})
    with pytest.raises(AlgoIdError, match='whitespace'):
      validate_algo_id('   ', registered)

  def test_tab_only_algo_id_is_rejected(self):
    registered = registered_algo_ids({'4242'})
    with pytest.raises(AlgoIdError, match='whitespace'):
      validate_algo_id('\t\n ', registered)

  def test_padded_algo_id_is_rejected(self):
    registered = registered_algo_ids({'4242'})
    with pytest.raises(AlgoIdError, match='surrounding whitespace'):
      validate_algo_id(' 4242 ', registered)

  def test_unregistered_client_sentinel_is_accepted(self):
    # NSE permits an unregistered client algo below the exchange OPS
    # threshold to send exactly "99999". Validating it away would make
    # the checker refuse a legitimate order.
    registered = registered_algo_ids(set())
    assert validate_algo_id('99999', registered) == '99999'
    assert not require_registration('99999')

  def test_registered_algo_id_is_accepted_and_marked_registered(self):
    registered = registered_algo_ids({'4242'})
    assert validate_algo_id('4242', registered) == '4242'
    assert require_registration('4242')

  def test_non_algo_sentinel_validates_but_is_not_a_registration(self):
    registered = registered_algo_ids(set())
    assert validate_algo_id('0', registered) == '0'
    assert not require_registration('0')

  def test_unregistered_algo_id_is_rejected(self):
    registered = registered_algo_ids({'4242'})
    with pytest.raises(AlgoIdError, match='not registered'):
      validate_algo_id('7777', registered)

  def test_non_string_algo_id_is_a_type_error(self):
    # The sentinel is the numeric-looking string "99999". An int-typed
    # field would coerce it and lose the value the exchange documented.
    registered = registered_algo_ids(set())
    with pytest.raises(TypeError, match='opaque string'):
      validate_algo_id(99999, registered)

  def test_algo_id_is_not_normalised_to_an_integer(self):
    # A leading zero must survive as a string.
    registered = registered_algo_ids({'0042'})
    assert validate_algo_id('0042', registered) == '0042'

  def test_registered_set_is_frozen_and_includes_sentinels(self):
    registered = registered_algo_ids({'4242'})
    assert unregistered_algo_id in registered
    assert '0' in registered
    assert isinstance(registered, frozenset)


class TestNnfStructure:
  '''The 15-digit NNF ID and the documented platform sentinels.'''

  def test_platform_is_the_first_twelve_digits(self):
    nnf = build_nnf_id(client_direct_api_platform, 0)
    assert platform_of(nnf) == '444444444444'

  def test_client_direct_api_is_the_platform_this_project_uses(self):
    assert client_direct_api_platform == '444444444444'
    assert allows_algo(client_direct_api_platform) is True

  def test_algo_flag_is_the_thirteenth_digit(self):
    nnf = build_nnf_id(client_direct_api_platform, 3)
    assert algo_flag_digit(nnf) == 3
    assert len(nnf) == nnf_length

  def test_platforms_without_algo_support_are_reported(self):
    assert allows_algo('111111111111') is False
    assert allows_algo('333333333333') is False

  def test_dma_platform_permits_algo(self):
    assert allows_algo('222222222222') is True

  def test_unknown_platform_is_none_not_false(self):
    # CTCL has no sentinel: its first six digits are a client PIN.
    # Returning False there would refuse algo orders on a platform the
    # exchange permits them on.
    assert allows_algo('123456789012') is None
    assert len(known_platform_algo_allowed) == 4

  @pytest.mark.parametrize('bad', [
    '4444444444440',       # 13 digits
    '4444444444440000',    # 16 digits
    '44444444444400x',     # non-numeric
    '',                    # empty
  ])
  def test_malformed_nnf_is_rejected(self, bad):
    with pytest.raises(AlgoTagError):
      platform_of(bad)
    with pytest.raises(AlgoTagError):
      algo_flag_digit(bad)

  def test_non_string_nnf_is_rejected(self):
    with pytest.raises(AlgoTagError, match='must be a string'):
      platform_of(444444444444000)

  def test_bad_platform_prefix_length_is_rejected(self):
    with pytest.raises(AlgoTagError, match='12 digits'):
      build_nnf_id('444', 0)

  def test_bad_flag_is_rejected(self):
    with pytest.raises(AlgoTagError, match='0-9'):
      build_nnf_id(client_direct_api_platform, 12)

  def test_bad_tail_is_rejected(self):
    with pytest.raises(AlgoTagError, match='two digits'):
      build_nnf_id(client_direct_api_platform, 0, tail='1')

  def test_short_platform_prefix_is_rejected_by_allows_algo(self):
    with pytest.raises(AlgoTagError, match='12 digits'):
      allows_algo('444')


class TestFlagDocumentedSet:
  '''Only the documented digits are accepted, and 9 is not one of them.'''

  def test_verified_digits_are_a_subset_of_documented(self):
    assert verified_flag_digits <= documented_flag_digits

  def test_nine_is_the_only_undocumented_digit(self):
    assert unverified_flag_digits == frozenset({9})

  def test_no_meaning_table_is_hardcoded(self):
    # The research says not to hardcode a full digit-to-meaning map
    # without the current NNF protocol CD. Only the algo/non-algo
    # partition is encoded, because that is what the validation rule
    # needs.
    assert is_algo_flag(0) and is_algo_flag(2) and is_algo_flag(4)
    assert not any(is_algo_flag(digit) for digit in (1, 3, 5, 6, 7, 8, 9))

  def test_documented_digits_are_the_nine_known_ones(self):
    assert documented_flag_digits == frozenset(range(9))


class TestAlgoFlagPairing:
  '''NSE/FAOP/69296: both halves of the rule are rejections.'''

  def test_algo_flag_with_a_real_algo_id_is_accepted(self):
    nnf = build_nnf_id(client_direct_api_platform, 0)
    assert validate_algo_tag(nnf, '4242') == nnf

  def test_algo_flag_with_the_unregistered_sentinel_is_accepted(self):
    nnf = build_nnf_id(client_direct_api_platform, 0)
    assert validate_algo_tag(nnf, '99999') == nnf

  def test_algo_flag_with_the_non_algo_sentinel_is_rejected(self):
    # An algo order with Algo ID "0" produces a record the audit trail
    # cannot attribute, which is the exact failure the tagging rule
    # exists to prevent.
    nnf = build_nnf_id(client_direct_api_platform, 0)
    with pytest.raises(AlgoTagError, match='non-algo sentinel'):
      validate_algo_tag(nnf, '0')

  @pytest.mark.parametrize('flag', [0, 2, 4])
  def test_every_algo_flag_rejects_the_non_algo_sentinel(self, flag):
    nnf = build_nnf_id(client_direct_api_platform, flag)
    with pytest.raises(AlgoTagError):
      validate_algo_tag(nnf, '0')

  @pytest.mark.parametrize('flag', [1, 3, 5, 6, 7, 8])
  def test_every_non_algo_flag_rejects_a_real_algo_id(self, flag):
    # Sending a real Algo ID with a non-algo flag claims an algo ran when
    # none did, which is worse than the reverse: it is a false record.
    nnf = build_nnf_id(client_direct_api_platform, flag)
    with pytest.raises(AlgoTagError, match='records an algo run'):
      validate_algo_tag(nnf, '99999')

  @pytest.mark.parametrize('flag', [1, 3, 5, 6, 7, 8])
  def test_every_non_algo_flag_accepts_the_non_algo_sentinel(self, flag):
    nnf = build_nnf_id(client_direct_api_platform, flag)
    assert validate_algo_tag(nnf, '0') == nnf

  def test_undocumented_flag_is_refused_outright(self):
    # There is no third option: the choice is between refusing the order
    # and mis-describing it.
    nnf = build_nnf_id(client_direct_api_platform, 9)
    with pytest.raises(AlgoTagError, match='no documented meaning'):
      validate_algo_tag(nnf, '99999')

  def test_malformed_nnf_is_refused_before_the_pairing_rule(self):
    with pytest.raises(AlgoTagError, match='15 digits'):
      validate_algo_tag('444444444444', '99999')

  def test_non_numeric_nnf_is_refused(self):
    with pytest.raises(AlgoTagError, match='all digits'):
      validate_algo_tag('x44444444444400', '99999')
