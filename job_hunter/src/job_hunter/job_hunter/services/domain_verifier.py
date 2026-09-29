#!/usr/bin/env python
# -- coding: utf-8 --

'''Company website resolution and verification.

Given a company name (and optionally a job-board slug), this module produces
ranked candidate domains and proves that a domain really belongs to that
company by matching name tokens against live page content.

The module never trusts a candidate domain on liveness alone: a domain is
only reported as verified when a significant company-name token is observed
in the resolved URL, the document title, or the opening page text.
'''


from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Sequence, Set, Tuple

from bs4 import BeautifulSoup
from job_hunter.adapters.http_client import HttpClient

_DOMAIN_RE = re.compile(r'^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$')
_HOST_RE = re.compile(r'https?://([a-z0-9.-]+)', re.IGNORECASE)
_TAG_RE = re.compile(r'<[^>]+>')
_WS_RE = re.compile(r'\s+')

'''Tokens too generic to prove a domain belongs to a named company.'''
STOP_TOKENS: frozenset = frozenset({
  'and', 'the', 'for', 'private', 'pvt', 'ltd', 'limited', 'llc', 'inc',
  'llp', 'plc', 'gmbh', 'india', 'global', 'technologies', 'technology',
  'tech', 'services', 'service', 'solutions', 'systems', 'group', 'labs',
  'company', 'corporation', 'corp', 'holdings', 'enterprises',
})

'''Corporate suffixes stripped before building candidate domains.'''
NAME_SUFFIXES: Tuple[str, ...] = (
  ' pvt ltd', ' pvt. ltd.', ' pvt ltd.', ' private limited', ' private ltd',
  ' pvt.', ' ltd.', ' ltd', ' limited', ' llc', ' inc.', ' inc', ' llp',
  ' plc', ' gmbh', ' s.a.', ' sa', ' ag', ' bv', ' pte ltd', ' co.',
)

'''Two-label public suffixes common in Indian company domains.'''
MULTI_PART_SUFFIXES: Tuple[str, ...] = (
  'co.in', 'net.in', 'org.in', 'firm.in', 'gen.in', 'ind.in', 'ac.in',
  'edu.in', 'gov.in', 'co.uk', 'org.uk', 'com.au', 'co.jp', 'com.br',
)

'''Tokens that are real words but too common to prove domain ownership.'''
GENERIC_TOKENS: frozenset = frozenset({
  'engineer', 'engineering', 'analyst', 'analytics', 'scientist', 'science',
  'manager', 'digital', 'software', 'systems', 'network', 'international',
  'global', 'worldwide', 'consulting', 'consultants', 'enterprises', 'labs',
  'india', 'bureau', 'staffing', 'partners', 'associates',
})

'''Default TLDs tried when guessing a company website.'''
DEFAULT_TLDS: Tuple[str, ...] = ('com', 'in', 'io', 'ai', 'co', 'net', 'tech')


@dataclass(frozen=True)
class DomainCandidate:
  '''A single proposed website for a company.

  Attributes:
    domain: Registrable domain, lowercase and without scheme.
    origin: Provenance tag ('slug', 'name', or 'search') used for ranking.
    score: Prior confidence before any network probing, in [0.0, 1.0].
  '''

  domain: str
  origin: str = 'name'
  score: float = 0.5

  def __lt__(self, other: 'DomainCandidate') -> bool:
    '''Order candidates by descending prior score.

    Args:
      other: Candidate to compare against.

    Returns:
      True when this candidate has the higher score and so sorts first.
    '''
    return self.score > other.score


@dataclass
class SiteProbe:
  '''Outcome of fetching one candidate domain.

  Attributes:
    domain: Domain that was probed.
    url: Final URL after redirects.
    ok: Whether the page returned usable content.
    haystack: Lowercased text used for name matching.
    error: Failure reason when ``ok`` is False.
  '''

  domain: str
  url: str = ''
  ok: bool = False
  haystack: str = ''
  error: str = ''


@dataclass
class VerifiedSite:
  '''A company website proven to match the company name.

  Attributes:
    domain: Verified registrable domain.
    url: URL that produced the match.
    confidence: Confidence in [0.0, 1.0].
    matched_tokens: Company tokens observed on the page.
    via: Provenance of the accepted candidate.
  '''

  domain: str
  url: str
  confidence: float
  matched_tokens: List[str] = field(default_factory=list)
  via: str = 'name'


def tokenize(name: str, unique: bool = True) -> List[str]:
  '''Split a company name into significant lowercase tokens.

  Args:
    name: Raw company name.
    unique: When True, de-duplicate while preserving first-seen order.

  Returns:
    Tokens of three or more characters with stop words removed, in the
    order they appear in the name.
  '''
  cleaned = ' '.join(name.lower().split())
  for suffix in NAME_SUFFIXES:
    if cleaned.endswith(suffix):
      cleaned = cleaned[: -len(suffix)].strip()
  raw = re.split(r'[^a-z0-9]+', cleaned)
  kept = [token for token in raw if len(token) >= 3 and token not in STOP_TOKENS]
  return list(dict.fromkeys(kept)) if unique else kept


def slug_candidates(name: str, slug: str = '', tlds: Sequence[str] = DEFAULT_TLDS) -> List[str]:
  '''Build domain candidates from a company name and job-board slug.

  Args:
    name: Display company name.
    slug: Slug observed on a job board, for example 'exl-service'.
    tlds: TLDs to try, in preference order.

  Returns:
    Ordered unique domains, best guesses first.
  '''
  stems: List[str] = []
  if slug:
    normalized = slug.lower().strip()
    stems.append(re.sub(r'[^a-z0-9]+', '', normalized))
    if '-' in normalized or '_' in normalized:
      for part in re.split(r'[-_\s]+', normalized):
        part = re.sub(r'[^a-z0-9]+', '', part)
        if len(part) >= 3:
          stems.append(part)
  tokens = tokenize(name)
  if tokens:
    stems.append(''.join(tokens)[:40])
    if len(tokens) > 1:
      stems.append(tokens[0][:40])
      stems.append(''.join(tokens[:2])[:40])
  ordered: List[str] = []
  for stem in stems:
    if not stem or len(stem) < 3 or not _DOMAIN_RE.match(stem):
      continue
    for tld in tlds:
      domain = f'{stem}.{tld}'
      if domain not in ordered:
        ordered.append(domain)
  return ordered


def registrable_domain(url: str) -> str:
  '''Extract the registrable domain from an arbitrary URL.

  Subdomains are collapsed so that ``careers.burohappold.com`` becomes
  ``burohappold.com``. Multi-label public suffixes such as ``co.in`` are
  preserved so the registrable pair stays correct.

  Args:
    url: URL or bare host.

  Returns:
    Registrable domain, or empty string when none could be parsed.
  '''
  candidate = url.strip()
  if '//' not in candidate:
    candidate = f'//{candidate}'
  match = _HOST_RE.search(candidate if candidate.startswith('http') else f'http:{candidate}')
  if not match:
    return ''
  host = match.group(1).lower().split(':')[0]
  if host.startswith('www.'):
    host = host[4:]
  labels = [label for label in host.split('.') if label]
  if len(labels) <= 2:
    return host
  tail_two = '.'.join(labels[-2:])
  if tail_two in MULTI_PART_SUFFIXES and len(labels) >= 3:
    return '.'.join(labels[-3:])
  return tail_two


def sld_of(domain: str) -> str:
  '''Return the second-level label of a domain.

  Args:
    domain: Registrable domain.

  Returns:
    The second-level label, or the host itself when too short.
  '''
  parts = [label for label in domain.lower().split('.') if label]
  if len(parts) >= 3 and '.'.join(parts[-2:]) in MULTI_PART_SUFFIXES:
    return parts[-3]
  return parts[-2] if len(parts) >= 2 else parts[0]


def is_blocked_domain(domain: str) -> bool:
  '''Report whether a domain belongs to a job board, social, or CDN.

  Args:
    domain: Host to test.

  Returns:
    True when the domain must never be recorded as a company website.
  '''
  blocked = (
    'linkedin.com', 'facebook.com', 'twitter.com', 'x.com', 'instagram.com',
    'youtube.com', 'medium.com', 'wikipedia.org', 'github.com', 'gitlab.com',
    'glassdoor.com', 'indeed.com', 'naukri.com', 'monster.com', 'ziprecruiter.com',
    'ambitionbox.com', 'shine.com', 'jobberman.com', 'foundit.in', 'cutshort.io',
    'google.com', 'duckduckgo.com', 'bing.com', 'yahoo.com', 'tracxn.com',
    'thecompanycheck.com', 'indianretailer.com', 'apollo.io', 'zoominfo.com',
    'crunchbase.com', 'owler.com', 'g2.com', 'capterra.com', 'clutch.co',
    'indeedmail.com', 'workable.com', 'greenhouse.io', 'lever.co', 'ashbyhq.com',
    'smartrecruiters.com', 'myworkdayjobs.com', 'oraclecloud.com', 'fasthire.io',
    'hibob.co', 'northstarsearch.com', 'jobvite.com', 'icims.com', 'taleo.net',
    'citadel.com', 'avature.net', 'eightfold.ai', 'jobs.smartrecruiters.com',
  )
  host = domain.lower()
  return any(host == item or host.endswith(f'.{item}') for item in blocked)


class WebsiteVerifier:
  '''Resolve and verify company websites with polite HTTP probing.

  The verifier owns three responsibilities, each exposed as a small public
  method so callers can compose them: candidate generation, single-domain
  probing, and full verification of a named company.
  '''

  def __init__(
    self,
    http: HttpClient,
    search: Optional[Callable[[str], Awaitable[List[str]]]] = None,
    tlds: Sequence[str] = DEFAULT_TLDS,
    min_confidence: float = 0.6,
    probe_budget: int = 6,
  ) -> None:
    '''Initialize the verifier.

    Args:
      http: Shared polite HTTP client used for all probes.
      search: Optional async callable mapping a query to candidate URLs. Used
        to discover a company domain through a search index when direct
        guessing fails.
      tlds: TLDs tried during domain guessing.
      min_confidence: Minimum confidence accepted by :meth:`verify`.
      probe_budget: Total number of domains probed per company. Half of the
        budget is reserved for search-derived candidates so that a weak
        guess never starves the search phase.
    '''
    self._http = http
    self._search = search
    self._tlds = tuple(tlds)
    self._min_confidence = min_confidence
    self._probe_budget = probe_budget

  def candidates_for(
    self,
    name: str,
    slug: str = '',
    search_urls: Sequence[str] = (),
  ) -> List[DomainCandidate]:
    '''Rank candidate domains for one company.

    Args:
      name: Display company name.
      slug: Optional job-board slug.
      search_urls: Optional search result URLs to mine for domains.

    Returns:
      Candidates ordered from most to least likely, de-duplicated.
    '''
    ordered: List[DomainCandidate] = []
    for domain in self._searched_domains(name, search_urls):
      ordered.append(domain)
    searched = {item.domain for item in ordered}
    name_stem = ''.join(tokenize(name))[:40]
    for domain in slug_candidates(name, slug, self._tlds):
      if domain in searched or is_blocked_domain(domain):
        continue
      stem = sld_of(domain)
      if name_stem and stem == name_stem:
        score = 0.9
      elif stem in tokenize(name) or (slug and stem in slug.lower()):
        score = 0.7
      else:
        score = 0.5
      ordered.append(DomainCandidate(domain=domain, origin='slug' if slug else 'name', score=score))
    return sorted(ordered, key=lambda item: (-item.score, item.domain))

  def _searched_domains(self, name: str, search_urls: Sequence[str]) -> List[DomainCandidate]:
    '''Turn search result URLs into ranked domain candidates.

    A search hit is trusted far more than a TLD guess, so these candidates
    always sort ahead of guessed domains.

    Args:
      name: Display company name.
      search_urls: Search result URLs to mine.

    Returns:
      Candidates in descending prior confidence.
    '''
    tokens = set(tokenize(name))
    scored: Dict[str, float] = {}
    for url in search_urls:
      host = registrable_domain(str(url))
      if not host or is_blocked_domain(host):
        continue
      stem = sld_of(host)
      if stem in tokens or any(tok in stem for tok in tokens):
        scored[host] = 0.95
      elif tokens & set(re.split(r'[^a-z0-9]+', sld_of(host))):
        scored[host] = 0.8
      else:
        scored.setdefault(host, 0.25)
    return [
      DomainCandidate(domain=host, origin='search', score=score)
      for host, score in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

  async def probe(self, domain: str) -> SiteProbe:
    '''Fetch a domain and prepare its text for name matching.

    Args:
      domain: Registrable domain to probe.

    Returns:
      A :class:`SiteProbe` describing the outcome; never raises.
    '''
    url = f'https://www.{domain}/'
    try:
      html = await self._http.get_text(url, rpm=20)
    except Exception as exc:  # noqa: BLE001 - probing must never propagate
      return SiteProbe(domain=domain, url=url, ok=False, error=f'{type(exc).__name__}: {exc}')
    if not html:
      html = ''
      try:
        html = await self._http.get_text(f'https://{domain}/', rpm=20)
      except Exception:  # noqa: BLE001 - fall through to the empty-body check
        html = ''
    if len(html) < 200:
      return SiteProbe(domain=domain, url=url, ok=False, error='empty or blocked body')
    haystack = self._page_haystack(html, url)
    if not haystack:
      return SiteProbe(domain=domain, url=url, ok=False, error='no readable text')
    return SiteProbe(domain=domain, url=url, ok=True, haystack=haystack)

  def _page_haystack(self, html: str, url: str) -> str:
    '''Build a normalized lowercase text blob for token matching.

    Args:
      html: Raw page source.
      url: Final page URL, included so domain matches also count.

    Returns:
      Space-collapsed lowercase text, truncated for cheap containment checks.
    '''
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.title.get_text(' ', strip=True) if soup.title else ''
    meta = ''
    for tag in soup.find_all('meta'):
      if str(tag.get('name') or '').lower() in ('description', 'og:title', 'og:description'):
        meta += ' ' + str(tag.get('content') or '')
    for tag in soup(['script', 'style', 'noscript']):
      tag.decompose()
    body = ' '.join(soup.get_text(' ').split())
    blob = f'{url} {title} {meta} {body}'.lower()
    return _WS_RE.sub(' ', _TAG_RE.sub(' ', blob))[:60000]

  async def _search_candidates(self, name: str, slug: str) -> List[DomainCandidate]:
    '''Derive candidate domains from a search index for one company.

    Args:
      name: Display company name.
      slug: Optional job-board slug.

    Returns:
      Candidates mined from search results, highest prior first.
    '''
    if self._search is None:
      return []
    queries = [f'{name} official website', f'{name} careers']
    if slug:
      queries.insert(0, f'{slug} company website')
    urls: List[str] = []
    for query in queries:
      try:
        found = await self._search(query)
      except Exception:  # noqa: BLE001 - search is best effort
        continue
      urls.extend(str(item) for item in (found or []))
    return self._searched_domains(name, urls)

  def score_probe(self, probe: SiteProbe, tokens: Sequence[str], prior: float) -> float:
    '''Score how well a probed page matches a company name.

    A match on generic words such as ``engineer`` or ``india`` proves
    nothing on its own, so at least one distinctive token must be present.

    Args:
      probe: Successful probe.
      tokens: Significant company-name tokens.
      prior: Prior score of the candidate that produced the probe.

    Returns:
      Confidence in [0.0, 1.0]; zero when no distinctive token matched.
    '''
    if not probe.ok or not tokens:
      return 0.0
    haystack = f' {probe.haystack} '
    stem = sld_of(probe.domain)
    matched = [token for token in tokens if f' {token} ' in haystack or token in stem]
    if not matched:
      return 0.0
    distinctive = [token for token in matched if token not in GENERIC_TOKENS]
    if not distinctive:
      return 0.0
    coverage = len(matched) / max(1, len(tokens))
    confidence = 0.45 + 0.30 * coverage
    if any(token == stem or stem.startswith(token) for token in distinctive):
      confidence += 0.20
    elif any(token in stem for token in distinctive):
      confidence += 0.10
    return round(min(0.99, max(0.0, confidence * (0.55 + 0.45 * prior))), 3)

  async def verify(
    self,
    name: str,
    slug: str = '',
    search_urls: Sequence[str] = (),
  ) -> Optional[VerifiedSite]:
    '''Resolve a company website and prove it belongs to that company.

    Verification runs in two ordered phases so that a long tail of weak
    domain guesses can never consume the whole probe budget:

    1. High-prior candidates from the name/slug plus any caller-supplied
       search URLs.
    2. If the search backend is configured and the budget still allows it,
       candidates mined from live search queries.

    Args:
      name: Display company name.
      slug: Optional job-board slug for the company.
      search_urls: Optional search result URLs to mine for domains.

    Returns:
      The winning :class:`VerifiedSite`, or None when nothing verified.
    '''
    tokens = tokenize(name)
    if not tokens:
      return None
    plan = self.candidates_for(name, slug, list(search_urls))
    if self._search is not None:
      plan = await self._search_candidates(name, slug) + plan
    tried: Set[str] = set()
    for candidate in plan:
      if candidate.domain in tried or len(tried) >= self._probe_budget:
        continue
      tried.add(candidate.domain)
      probe = await self.probe(candidate.domain)
      if not probe.ok:
        continue
      confidence = self.score_probe(probe, tokens, candidate.score)
      if confidence < self._min_confidence:
        continue
      stem = sld_of(candidate.domain)
      matched = [tok for tok in tokens if f' {tok} ' in f' {probe.haystack} ' or tok in stem]
      return VerifiedSite(
        domain=candidate.domain,
        url=probe.url,
        confidence=confidence,
        matched_tokens=sorted(matched),
        via=candidate.origin,
      )
    return None


def verified_site_row(name: str, slug: str, site: Optional[VerifiedSite]) -> Dict[str, object]:
  '''Flatten a verification result into a seed-file friendly mapping.

  Args:
    name: Company display name.
    slug: Job-board slug the name was derived from.
    site: Verification result, or None when nothing was proven.

  Returns:
    Mapping with name, domain, slug, confidence, and via keys; domain is
    empty when verification failed.
  '''
  if site is None:
    return {'name': name, 'domain': '', 'slug': slug, 'confidence': 0.0, 'via': 'unverified'}
  return {
    'name': name,
    'domain': site.domain,
    'slug': slug,
    'confidence': site.confidence,
    'via': site.via,
  }

def company_domain_map(entries: Sequence[Dict[str, str]]) -> Dict[str, str]:
  '''Build a lowercase company-name to domain index.

  Args:
    entries: Mappings with at least a 'name' key and an optional 'domain'.

  Returns:
    Mapping of normalized company name to registrable domain.
  '''
  index: Dict[str, str] = {}
  for entry in entries:
    name = str(entry.get('name') or '').strip()
    domain = registrable_domain(str(entry.get('domain') or ''))
    if not name or not domain or is_blocked_domain(domain):
      continue
    index[name.lower()] = domain
  return index
