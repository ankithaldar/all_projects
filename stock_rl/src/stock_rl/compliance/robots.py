#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''robots.txt courtesy, and an honest account of what it is worth.

robots.txt is not a legal requirement in India. That is the single most
important thing this module has to say, because the alternative reading
propagates: an engineer who believes a rule is statutory cites it in a
design document, a reviewer who cannot find the statute loses trust in
the whole register, and a publisher's counsel reads the code first.

What the sources actually support:

  * There is **no reported Indian case** on robots.txt either way. The
    European Parliament's 2025 study states the position plainly: "There
    is no law stating that /robots.txt must be obeyed, nor does it
    constitute a binding contract between site owner and user."
  * The strongest pro-obligation material is academic argument, not a
    holding -- Chang and He, "The Liabilities of Robots.txt", Computer
    Law and Security Review (2025), arXiv:2503.06035 -- which argues it
    can serve as notice sufficient for tortious liability in common-law
    jurisdictions. Persuasive nowhere near a common-law jurisdiction.
  * robots.txt becomes *meaningful* only where a site's Terms of Use
    incorporates it by reference. A Terms of Use **is** an enforceable
    contract in India (IT Act ss.4 and 10A). The obligation, where one
    exists, comes from the ToU; robots.txt supplies only the parameter.
  * robots.txt will not rescue anyone from NSE Terms of Use clause 9,
    which bans systematic or automated collection outright, or clause 8,
    which bans storage in an electronic retrieval system without written
    permission. A False here is not a licence to fetch.

So this module is risk management and courtesy, implemented to RFC 9309.
It is cheap, it keeps the ingestion scripts well behaved, and it is worth
having. It is not compliance with a rule, and the docstring above is
deliberately not dressed up as one.

Crawl-delay is honoured where a site publishes it, because it is the one
machine-readable politeness signal available to a crawler. RFC 9309
treats the directive as an extension rather than part of the core
grammar, so it is optional; where it is absent, the caller's own
:mod:`stock_rl.compliance.ratelimit` interval governs. Where both exist,
the caller must take the larger, and this module does not silently
override either one.

Fetching robots.txt is injectable because a test must never touch the
network. Inject a fetcher returning a canned body.
'''

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Callable
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

__all__ = ['Fetcher', 'RobotsChecker', 'RobotsUnavailable', 'origin_of']

#: Signature of a robots.txt fetcher. Returns the body, or None when the
#: origin serves no robots.txt. Raises ``RobotsUnavailable`` for the
#: statuses RFC 9309 s.2.3.1.4 says mean "assume complete disallow".
Fetcher = Callable[[str, float, str], str | None]


class RobotsUnavailable(Exception):
  '''Raised when robots.txt is unreadable and must be assumed disallowing.

  RFC 9309 s.2.3.1.3 says a 4xx other than 401 and 403 means the
  resource is not available and *may* be treated as allowing everything.
  s.2.3.1.4 says a 5xx means the crawler **must** assume complete
  disallow, and 401 or 403 applies to the whole site. Those three cases
  are the ones this exception carries.
  '''


def _url_fetcher(url: str, timeout: float, user_agent: str) -> str | None:
  '''Fetch a robots.txt body over HTTP.

  Args:
    url: Absolute URL of the robots.txt resource.
    timeout: Socket timeout in seconds.
    user_agent: Agent token sent in the ``User-Agent`` header. Sending a
      real token matters: the default urllib string identifies as a bot
      and several origins refuse it outright.

  Returns:
    The response body as text, or None if the origin serves no
    robots.txt.

  Raises:
    RobotsUnavailable: If the origin answered 401, 403 or a 5xx status,
      which RFC 9309 treats as an instruction to disallow everything.
  '''
  request = urllib.request.Request(url, headers={'User-Agent': user_agent})
  try:
    with urllib.request.urlopen(request, timeout=timeout) as response:
      raw = response.read()
  except urllib.error.HTTPError as error:
    # An HTTPError is itself a response, so it has to be closed even
    # when it is about to be re-raised.
    status = error.code
    error.close()
    if status in (401, 403) or 500 <= status < 600:
      raise RobotsUnavailable(f'{url} returned HTTP {status}') from error
    return None
  except (urllib.error.URLError, OSError, ValueError):
    # No answer at all: the origin is down, the name does not resolve, or
    # the URL is not a http(s) URL. Treated as "no robots.txt", which is
    # the permissive branch. It is permissive because this module has no
    # legal force to enforce, and a hard failure here would take down a
    # pipeline over a courtesy file.
    return None
  return raw.decode('utf-8', 'surrogateescape')


def origin_of(url: str) -> str:
  '''Return the cache key for a URL: its scheme and host.

  The key is lowercase because the scheme and the host are
  case-insensitive, and keeping the port means two services on one host
  get independent policies. The path is deliberately excluded: robots.txt
  is per origin, not per page.

  Args:
    url: Absolute URL.

  Returns:
    The origin in the form ``scheme://host[:port]``.

  Raises:
    ValueError: If the URL carries no scheme or no host, which means it
      is not an absolute URL and has no origin to key on.
  '''
  parts = urlsplit(url)
  if not parts.scheme or not parts.netloc:
    raise ValueError(f'not an absolute URL: {url!r}')
  return f'{parts.scheme}://{parts.netloc}'.lower()


class RobotsChecker:
  '''A robots.txt gate with one cached parse per origin.

  The cache is the point. robots.txt is small, but re-fetching it per
  request turns a courtesy check into the load it was meant to avoid, and
  a client that fetches robots.txt as often as it fetches pages is not
  being a good citizen whatever the file says. Parsers are cached per
  instance, never in module state, so a test can build a fresh checker
  without having to clear a global.

  Args are documented on ``__init__``.
  '''

  def __init__(self, user_agent: str = 'stock-rl',
               fetcher: Fetcher | None = None,
               timeout: float = 10.0) -> None:
    '''Build a checker.

    Args:
      user_agent: Product token sent when fetching robots.txt and matched
        against the file's ``User-agent`` groups. ``Entry.applies_to``
        matches on a case-insensitive substring of the token, so a
        token with a version suffix such as ``stock-rl/0.1`` still
        matches a ``User-agent: stock-rl`` group.
      fetcher: Callable returning a robots.txt body, or None when the
        origin has none. Injecting one is how tests avoid the network.
        Defaults to an HTTP GET.
      timeout: Socket timeout in seconds for the default fetcher.

    Raises:
      ValueError: If ``user_agent`` is empty or whitespace, or if
        ``timeout`` is not positive.
    '''
    if not user_agent.strip():
      raise ValueError('user_agent must not be empty')
    if timeout <= 0.0:
      raise ValueError(f'timeout must be positive, got {timeout}')
    self.user_agent = user_agent
    self.timeout = timeout
    self._fetcher: Fetcher = fetcher or _url_fetcher
    self._parsers: dict[str, RobotFileParser] = {}
    self._fetches = 0

  @property
  def fetch_count(self) -> int:
    '''Number of robots.txt fetches performed, i.e. cache misses.

    Exposed because the caching is an invariant worth asserting in a
    test and invisible from the outside otherwise.
    '''
    return self._fetches

  @property
  def cached_origins(self) -> tuple[str, ...]:
    '''Origins whose robots.txt has been parsed, in first-seen order.'''
    return tuple(self._parsers)

  def must_fetch(self, url: str) -> bool:
    '''Return True if this user agent may fetch ``url`` per robots.txt.

    Args:
      url: Absolute URL of the resource to be fetched.

    Returns:
      True if robots.txt does not forbid it. False if it does, if the
      origin answered 401, 403 or 5xx, or if the URL cannot be resolved
      to an origin.

    Raises:
      ValueError: If ``url`` is not absolute.
    '''
    return self._parser_for(url).can_fetch(self.user_agent, url)

  def crawl_delay(self, url: str) -> float | None:
    '''Return the Crawl-delay declared for this agent, in seconds.

    Args:
      url: Any absolute URL on the origin whose file is being asked
        about. The path is ignored; only the origin matters.

    Returns:
      The delay in seconds, or None when the file declares none for this
      agent. None means *no instruction given*, not *zero*: the caller
      still has to apply its own interval.

    Raises:
      ValueError: If ``url`` is not absolute.
    '''
    delay = self._parser_for(url).crawl_delay(self.user_agent)
    return None if delay is None else float(delay)

  def clear_cache(self) -> None:
    '''Drop every cached parser.

    Needed when a long-running process should re-read the file: robots
    directives change, and a cache with no expiry is a cache that
    eventually lies. A caller serving many origins should call this on a
    schedule, not per request.
    '''
    self._parsers.clear()

  def _parser_for(self, url: str) -> RobotFileParser:
    '''Return the cached parser for a URL's origin, loading it if absent.

    Args:
      url: Absolute URL.

    Returns:
      A parsed RobotFileParser. A negative result -- no file, or an
      unreadable one -- is cached too, so a dead origin is asked once.

    Raises:
      ValueError: If ``url`` is not absolute.
    '''
    origin = origin_of(url)
    parser = self._parsers.get(origin)
    if parser is None:
      parser = self._load(origin)
      self._parsers[origin] = parser
    return parser

  def _load(self, origin: str) -> RobotFileParser:
    '''Fetch and parse one origin's robots.txt.

    Args:
      origin: Scheme and host in the form ``scheme://host[:port]``.

    Returns:
      A parser ready to answer ``can_fetch``. When the file is absent the
      parser is explicitly permissive; when the origin is unreadable in
      the RFC's sense it is explicitly restrictive. Both are set as flags
      rather than left to the parsed-entry path, because an empty parse
      and a refused fetch are different outcomes.
    '''
    parser = RobotFileParser()
    parser.set_url(f'{origin}/robots.txt')
    self._fetches += 1
    try:
      body = self._fetcher(f'{origin}/robots.txt', self.timeout,
                           self.user_agent)
    except RobotsUnavailable:
      parser.disallow_all = True
      parser.modified()
      return parser
    # An empty or missing body parses to no groups, which
    # ``can_fetch`` treats as allow-everything: the correct reading of
    # "this origin publishes no instructions".
    parser.parse((body or '').splitlines())
    return parser
