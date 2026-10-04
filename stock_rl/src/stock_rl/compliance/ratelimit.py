#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Per-domain request pacing, on an injectable clock.

The number in this module has no legal source. "Maximum one request per
second" appears in design documents as though it were an Indian
compliance requirement; it is not, and no such requirement was found in
any statute, rule, circular or exchange document. Writing it down as a
citation would be a fabricated one, which is precisely the failure this
project's research corpus exists to prevent.

What the number *is*: engineering risk management against a real
statutory exposure. The Information Technology Act 2000 s.43(e) and
(f) make a person liable to pay compensation for causing interruption
or obstruction of the operation of a computer system, and s.66 adds
criminal liability only where the act is done dishonestly or
fraudulently, which by the Explanation to that section carries the IPC
s.24 and s.25 meanings of deception to cause loss. A script that
degrades somebody else's server is the only realistic way a data
acquisition tool picks up statutory exposure, and self-pacing is the
cheapest available defence: a client that provably never exceeds its own
interval is hard to describe as the cause of an outage. That is the
entire justification, and it is an engineering one.

Design consequences, all of them deliberate:

  * The interval is a constructor argument, not a constant. It is a risk
    appetite, not a rule, and the honest way to carry one is as
    configuration.
  * The clock, the sleep function and the jitter source are all
    injectable. The invariant this module exists to enforce -- never two
    requests to one domain closer together than the interval -- is then
    testable in microseconds, with no real sleeping anywhere in the test
    suite.
  * Min-interval, not a token bucket. A token bucket permits a burst, and
    a burst is exactly what looks like abuse in a server log. One request
    per interval, always, is the version that survives being audited.
  * HTTP 429 suspends the domain for at least the server's Retry-After
    and past that for a bounded exponential delay, so a rate-limited
    origin cannot be hammered by a client that was told to stop.
'''

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

__all__ = ['BackoffPolicy', 'Clock', 'Jitter', 'RateLimiter', 'Sleeper',
           'parse_retry_after']

#: Source of monotonic seconds. ``time.monotonic`` by default because a
#: wall-clock adjustment would otherwise let a limiter believe it had
#: waited when it had not.
Clock = Callable[[], float]

#: Sleeper taking a number of seconds.
Sleeper = Callable[[float], None]

#: Source of a uniform variate in [0, 1) used for proportional jitter.
Jitter = Callable[[], float]

#: Cap on the exponent, so a pathological strike count cannot overflow a
#: float. 2**32 seconds of backoff is already unreachable in practice.
exponent_cap = 32


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
  '''Bounded exponential backoff with proportional jitter.

  Attributes:
    base_seconds: Delay after the first 429. Doubles per subsequent
      strike, so strike ``n`` costs ``base_seconds * 2 ** (n - 1)``.
    max_seconds: Ceiling on the un-jittered delay. It is a ceiling
      rather than a slope because an uncapped exponential eventually
      waits longer than a client session survives, which converts a rate
      limit into a silent stall.
    jitter_ratio: Extra delay as a fraction of the un-jittered delay,
      sampled uniformly from ``[0, jitter_ratio]``. Jitter exists so
      that several clients rate-limited by the same origin do not retry
      in lockstep. The absolute worst case is therefore
      ``max_seconds * (1 + jitter_ratio)``.
  '''

  base_seconds: float = 1.0
  max_seconds: float = 60.0
  jitter_ratio: float = 0.5


@dataclass(slots=True)
class _DomainState:
  '''Per-domain bookkeeping. Internal, and deliberately mutable.

  Attributes:
    last_allowed: Clock value at which the most recent request was
      released, or None if this domain has never been used.
    suspended_until: Clock value before which no request may be issued,
      set by ``RateLimiter.on_429``.
    strikes: Consecutive 429s, reset by a successful request.
  '''

  last_allowed: float | None = None
  suspended_until: float = 0.0
  strikes: int = 0


def parse_retry_after(value: str) -> float | None:
  '''Parse an HTTP Retry-After header into seconds.

  RFC 9110 s.10.2.3 allows two forms: a non-negative delta in seconds,
  and an HTTP-date. Both are handled because vendor and CDN origins in
  front of an exchange feed have been seen using each; guessing between
  them would otherwise mean either ignoring the server or sleeping for
  the wrong length of time.

  Args:
    value: The raw header value.

  Returns:
    Seconds to wait, or None if the value is unparseable or negative.
    None means "carry on with your own backoff", not "retry now".
  '''
  text = value.strip()
  if not text:
    return None
  if text.isdecimal():
    seconds = float(text)
    return seconds if seconds >= 0.0 else None
  try:
    moment = parsedate_to_datetime(text)
  except (TypeError, ValueError):
    return None
  # A header with a zone token parses aware. One without any -- which
  # RFC 9110's HTTP-date does not require -- parses naive, and RFC 9110
  # says the recipient should read it as GMT.
  if moment.tzinfo is None:
    moment = moment.replace(tzinfo=timezone.utc)
  delta = (moment - utcnow()).total_seconds()
  return delta if delta > 0.0 else None


def utcnow() -> datetime:
  '''Return the current UTC time as an aware datetime.

  Isolated as a function so that the one wall-clock call in this module
  is easy to find, and so that a test needing a fixed clock can
  monkeypatch a single name. It is deliberately the only one: everything
  else in this module runs off the injected clock.

  Returns:
    The current time, timezone-aware, in UTC.
  '''
  return datetime.now(timezone.utc)


class RateLimiter:
  '''Enforce a minimum interval between requests to each domain.

  Not thread-safe and not intended to be: the compliant caller runs one
  process, and SEBI's own requirement is that the order and risk paths
  share a process anyway, which is the argument in
  ``docs/research/architecture/project-structure.md``. Guarding a
  limiter with a lock would be guarding against a concurrency model this
  project has decided not to have.

  Args are documented on ``__init__``.
  '''

  def __init__(
    self,
    min_interval: float = 1.0,
    clock: Clock = time.monotonic,
    sleep: Sleeper = time.sleep,
    jitter: Jitter = random.random,
    backoff: BackoffPolicy = BackoffPolicy(),
  ) -> None:
    '''Build a limiter.

    Args:
      min_interval: Minimum seconds between two requests to the same
        domain. Zero disables pacing without disabling 429 handling.
      clock: Monotonic time source, in seconds. Injectable so the
        interval invariant is testable without waiting.
      sleep: Sleep implementation. An injected sleeper is responsible
        for advancing whatever the injected clock reports; the limiter
        records the *projected* release time rather than re-reading the
        clock, so an instantaneous sleeper is still safe rather than
        merely fast.
      jitter: Uniform variate source for backoff jitter.
      backoff: Backoff shape used after an HTTP 429.

    Raises:
      ValueError: If ``min_interval`` is negative, or if the backoff
        policy has a non-positive base or ceiling, a negative jitter
        ratio, or a ceiling below the base.
    '''
    if min_interval < 0.0:
      raise ValueError(f'min_interval must be >= 0, got {min_interval}')
    if backoff.base_seconds <= 0.0:
      raise ValueError(
        f'backoff base must be positive, got {backoff.base_seconds}')
    if backoff.max_seconds < backoff.base_seconds:
      raise ValueError(
        f'backoff ceiling {backoff.max_seconds} is below base '
        f'{backoff.base_seconds}')
    if backoff.jitter_ratio < 0.0:
      raise ValueError(
        f'jitter_ratio must be >= 0, got {backoff.jitter_ratio}')
    self.min_interval = min_interval
    self.clock = clock
    self.sleep = sleep
    self.jitter = jitter
    self.backoff = backoff
    self._state: dict[str, _DomainState] = {}

  def acquire(self, domain: str) -> float:
    '''Wait until ``domain`` may be requested, then record the request.

    Call this immediately before every outbound request to a domain. It
    returns the number of seconds it waited, which is zero on the first
    use of a domain, and is useful enough to log that a caller is being
    throttled by its own limiter rather than by the origin.

    Args:
      domain: Host or origin to pace. Treated as an opaque string, so a
        caller must be consistent about including the port.

    Returns:
      Seconds spent waiting, zero if the call was released immediately.

    Raises:
      ValueError: If ``domain`` is empty or whitespace, since an empty
        domain would collapse every request onto one bucket.
    '''
    if not domain.strip():
      raise ValueError('domain must not be empty')
    now = self.clock()
    state = self._state.setdefault(domain, _DomainState())
    earliest = now
    if state.last_allowed is not None:
      earliest = max(earliest, state.last_allowed + self.min_interval)
    earliest = max(earliest, state.suspended_until)
    delay = earliest - now
    if delay > 0.0:
      self.sleep(delay)
    # The projected release time is recorded rather than a re-read of
    # the clock, so a caller whose injected sleeper does not advance
    # time still cannot compress the interval on paper.
    state.last_allowed = earliest
    return delay

  def note_success(self, domain: str) -> None:
    '''Clear a domain's 429 strikes after a request that was accepted.

    Args:
      domain: Domain whose penalty should be lifted.
    '''
    state = self._state.get(domain)
    if state is not None:
      state.strikes = 0

  def on_429(self, domain: str, retry_after: float | None = None) -> float:
    '''Penalise a domain that answered HTTP 429 Too Many Requests.

    The server's own instruction wins when it asks for longer than our
    backoff, and is ignored when it asks for less. Ignoring a longer
    Retry-After is how a client gets itself blocked by a CDN rather than
    merely slowed; treating a shorter one as binding would let an origin
    understate its own limit and never reach the exponential part that
    breaks a retry storm.

    Args:
      domain: Domain that returned 429.
      retry_after: Server-requested delay in seconds, usually from the
        Retry-After header via ``parse_retry_after``. None means the
        header was absent.

    Returns:
      Seconds the domain is now suspended for.

    Raises:
      ValueError: If ``domain`` is empty or whitespace, or if
        ``retry_after`` is negative.
    '''
    if not domain.strip():
      raise ValueError('domain must not be empty')
    if retry_after is not None and retry_after < 0.0:
      raise ValueError(f'retry_after must be >= 0, got {retry_after}')
    state = self._state.setdefault(domain, _DomainState())
    delay = self.backoff_delay(state.strikes, retry_after)
    state.strikes += 1
    state.suspended_until = max(state.suspended_until, self.clock() + delay)
    return delay

  def backoff_delay(self, attempt: int,
                    retry_after: float | None = None) -> float:
    '''Return the delay for a backoff attempt, without recording it.

    Kept separate from ``on_429`` so the shape of the curve can be
    asserted directly, which is the invariant that matters: it must
    increase, and it must be bounded. An unbounded curve turns a rate
    limit into an outage, because the client stops trying long before
    the origin would have let it back in.

    Args:
      attempt: Zero-based strike count, so 0 is the first 429.
      retry_after: Server-requested minimum delay in seconds, or None.

    Returns:
      Seconds to wait, jitter included.

    Raises:
      ValueError: If ``attempt`` is negative or ``retry_after`` is
        negative.
    '''
    if attempt < 0:
      raise ValueError(f'attempt must be >= 0, got {attempt}')
    if retry_after is not None and retry_after < 0.0:
      raise ValueError(f'retry_after must be >= 0, got {retry_after}')
    grown = self.backoff.base_seconds * (2.0 ** min(attempt, exponent_cap))
    delay = min(grown, self.backoff.max_seconds)
    if retry_after is not None:
      delay = max(delay, retry_after)
    return delay + self.jitter() * delay * self.backoff.jitter_ratio

  def strikes(self, domain: str) -> int:
    '''Return how many consecutive 429s a domain has served.

    Args:
      domain: Domain to inspect.

    Returns:
      The strike count, zero for a domain never penalised.
    '''
    state = self._state.get(domain)
    return 0 if state is None else state.strikes

  def next_allowed(self, domain: str) -> float | None:
    '''Return the clock value before which a domain must not be called.

    Args:
      domain: Domain to inspect.

    Returns:
      The earliest permissible clock value, or None if the domain has
      never been requested. Exposed so a caller can assert the
      invariant -- that this never falls below the previous release plus
      the interval -- without reaching into private state.
    '''
    state = self._state.get(domain)
    if state is None:
      return None
    earliest = state.last_allowed
    if earliest is None:
      return None
    return max(earliest + self.min_interval, state.suspended_until)
