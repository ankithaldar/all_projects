#!/usr/bin/env python
# -- coding: utf-8 --

'''Company discovery from LinkedIn job pages, located via a search index.

LinkedIn's robots.txt disallows ``/jobs-guest/``, so this module never
requests LinkedIn directly. Instead it queries a search engine whose robots
policy permits crawling, keeps only results that point at LinkedIn job
pages, and recovers the hiring company from each result's URL slug or
title.

For every recovered company the module asks a
:class:`~job_hunter.services.domain_verifier.WebsiteVerifier` to prove the
company's website, classifies the vertical from rule keywords, and detects
the company's ATS board so it can be crawled like any other target.
'''


from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from bs4 import BeautifulSoup
from job_hunter.adapters.career_page import CareerPageDetector
from job_hunter.adapters.http_client import HttpClient
from job_hunter.db.repositories.companies import CompaniesRepository, normalize_name
from job_hunter.services.domain_verifier import (
  VerifiedSite,
  WebsiteVerifier,
  is_blocked_domain,
  registrable_domain,
  tokenize,
)
from job_hunter.services.web_search import (
  WebSearchIndex,
  clean_result_title,
  company_from_linkedin_url,
  is_search_noise,
)
from job_hunter.services.vertical_classifier import VerticalClassifier

logger = logging.getLogger(__name__)

'''Search operators that restrict results to LinkedIn job pages.'''
LINKEDIN_JOBS_OPERATOR = 'site:linkedin.com/jobs/view'

'''Roles used to build the default search query matrix.'''
DEFAULT_ROLES: Tuple[str, ...] = (
  'data scientist', 'senior data scientist', 'machine learning engineer',
  'data engineer', 'analytics engineer', 'applied scientist', 'ml engineer',
  'data analyst', 'business intelligence analyst', 'head of data science',
)

'''Locations used to build the default search query matrix.'''
DEFAULT_LOCATIONS: Tuple[str, ...] = (
  'India', 'Bengaluru', 'Hyderabad', 'Pune', 'Mumbai', 'Delhi NCR', 'Chennai',
  'Gurugram', 'Noida', 'Remote India',
)

'''Company names that are staffing agencies, not direct employers.'''
STAFFING_MARKERS: Tuple[str, ...] = (
  'staffing', 'staffing agency', 'recruitment', 'recruiting', 'talent solutions',
  'human resource', 'hr consultancy', 'outsourcing', 'flex staffing',
  'contract staffing', 'job placement', 'career services',
)

'''Free-text hints that a search result describes an agency, not an employer.'''
AGENCY_TITLE_HINTS: Tuple[str, ...] = (
  'staffing', 'recruitment', 'talent solutions', 'consultancy', 'consultants',
  'outsourcing', 'infotech', 'technologies pvt', 'hr services',
)

# Roles or locations that are too generic to build a useful query.
_GENERIC_QUERY_TOKENS = {'india', 'remote', 'jobs', 'job', 'hiring', 'role'}

_LINKEDIN_JOB_URL_RE = re.compile(
  r'^https?://[a-z]{2,3}\.?linkedin\.com/jobs/view/[^/]+', re.IGNORECASE,
)


@dataclass
class CompanyCandidate:
  '''A company recovered from a LinkedIn job result.

  Attributes:
    name: Display name as published on the job page.
    slug: Slug observed on LinkedIn, used to seed domain guessing.
    job_urls: LinkedIn job URLs the company was seen on.
    sightings: Number of results naming this company.
  '''

  name: str
  slug: str = ''
  job_urls: List[str] = field(default_factory=list)
  sightings: int = 1

  def absorb(self, other: 'CompanyCandidate') -> None:
    '''Merge another sighting of the same company into this one.

    Args:
      other: Sighting to merge; its slug is adopted when this one lacks it.
    '''
    if not self.slug and other.slug:
      self.slug = other.slug
    self.sightings += other.sightings
    for url in other.job_urls:
      if url not in self.job_urls and len(self.job_urls) < 5:
        self.job_urls.append(url)


@dataclass
class ScoutedCompany:
  '''A company whose website has been verified, ready to persist.

  Attributes:
    name: Display name.
    domain: Verified registrable domain; empty when verification failed.
    site: Full verification result when one was produced.
    vertical: Vertical label, or None when rules did not classify it.
    vertical_confidence: Confidence attached to the vertical.
    ats_provider: Detected ATS provider key, if any.
    board_ref: ATS board token, if any.
    careers_url: Careers page used for detection.
    slug: LinkedIn slug the company was found through.
    sightings: Number of job results that named this company.
    status: Initial lifecycle status for the companies table.
  '''

  name: str
  domain: str = ''
  site: Optional[VerifiedSite] = None
  vertical: Optional[str] = None
  vertical_confidence: float = 0.0
  ats_provider: Optional[str] = None
  board_ref: Optional[str] = None
  careers_url: Optional[str] = None
  slug: str = ''
  sightings: int = 1
  status: str = 'needs_review'

  def to_seed_entry(self) -> Dict[str, object]:
    '''Render the company as a seeds YAML mapping.

    Returns:
      Mapping matching the schema consumed by ``ingest_seeds``.
    '''
    entry: Dict[str, object] = {'name': self.name}
    if self.domain:
      entry['domain'] = self.domain
    if self.vertical:
      entry['vertical_hint'] = self.vertical
    priority = 3 if self.sightings >= 3 else (2 if self.sightings >= 2 else 1)
    entry['priority'] = priority
    return entry


@dataclass
class ScoutReport:
  '''Counters describing one scouting run.

  Attributes:
    queries_run: Search queries issued.
    linkedin_hits: Results that pointed at LinkedIn job pages.
    candidates: Distinct companies recovered from those results.
    skipped_agency: Candidates dropped as staffing agencies.
    already_known: Candidates matched to an existing company row.
    verified: Companies whose website was proven.
    unverified: Candidates whose website could not be proven.
    ingested: Rows written to the companies table.
    ats_verified: Companies whose ATS board was detected.
    companies: The scouted companies, verified ones first.
  '''

  queries_run: int = 0
  linkedin_hits: int = 0
  candidates: int = 0
  skipped_agency: int = 0
  already_known: int = 0
  verified: int = 0
  unverified: int = 0
  ingested: int = 0
  ats_verified: int = 0
  companies: List[ScoutedCompany] = field(default_factory=list)

  def summary(self) -> str:
    '''Render a one-line human summary of the run.

    Returns:
      Space-separated counters.
    '''
    return (
      f'queries={self.queries_run} linkedin_hits={self.linkedin_hits} '
      f'candidates={self.candidates} agency_skipped={self.skipped_agency} '
      f'known={self.already_known} verified={self.verified} '
      f'unverified={self.unverified} ingested={self.ingested} '
      f'ats={self.ats_verified}'
    )


def build_queries(
  roles: Sequence[str] = DEFAULT_ROLES,
  locations: Sequence[str] = DEFAULT_LOCATIONS,
  limit: int = 12,
) -> List[str]:
  '''Build a role-by-location matrix of LinkedIn job search queries.

  The matrix is deterministic so repeated runs re-use cached search results
  instead of hammering the search engine with fresh queries.

  Args:
    roles: Job titles to search for.
    locations: Locations to combine with each role.
    limit: Maximum number of queries to return.

  Returns:
    De-duplicated query strings, roles varying fastest within a location.
  '''
  queries: List[str] = []
  seen: Set[str] = set()
  for location in locations:
    for role in roles:
      query = f'{LINKEDIN_JOBS_OPERATOR} {role} {location}'.strip()
      if query in seen:
        continue
      seen.add(query)
      queries.append(query)
      if len(queries) >= max(1, limit):
        return queries
  return queries


def looks_like_agency(name: str, title: str = '') -> bool:
  '''Report whether a name or job title suggests a staffing agency.

  Args:
    name: Company name.
    title: Related search result title or job title.

  Returns:
    True when the text reads like an agency rather than an employer.
  '''
  haystack = f'{name} {title}'.lower()
  if any(separator in name for separator in ('|', '\u2014', ' - ', '--')):
    return True
  return any(marker in haystack for marker in STAFFING_MARKERS) or any(
    hint in haystack for hint in AGENCY_TITLE_HINTS
  )


def is_linkedin_job_url(url: str) -> bool:
  '''Report whether a URL points at a LinkedIn job posting.

  Args:
    url: Candidate URL.

  Returns:
    True for a LinkedIn ``/jobs/view/`` URL.
  '''
  return bool(_LINKEDIN_JOB_URL_RE.match(str(url or '').strip()))


'''Leading words that indicate a job headline was captured instead.'''
STOP_WORDS_SHAPE: Tuple[str, ...] = (
  'women', 'men', 'in', 'at', 'the', 'a', 'an', 'we', 'you', 'our', 'is',
  'are', 'and', 'for', 'with', 'to', 'of', 'on', 'by', 'from',
)


def candidate_from_result(title: str, href: str) -> Optional[CompanyCandidate]:
  '''Recover a hiring company from one search result.

  The company is read from the LinkedIn URL slug first because it is
    structured, then from the ``<Role> at <Company>`` result title.

  Args:
    title: Search result title.
    href: Search result URL.

  Returns:
    A :class:`CompanyCandidate`, or None when no company could be read.
  '''
  if not is_linkedin_job_url(href) or is_search_noise(href):
    return None
  slug = ''
  raw_name = ''
  slug_name = company_from_linkedin_url(href)
  if slug_name:
    slug = re.sub(r'-\d{5,}$', '', href.rstrip('/').rsplit('/', 1)[-1])
    raw_name = slug_name
  if not raw_name:
    raw_name = _company_from_title(title)
  if not raw_name or len(raw_name) < 2 or not _plausible_company_name(raw_name):
    return None
  if looks_like_agency(raw_name, title):
    return None
  return CompanyCandidate(name=raw_name, slug=slug, job_urls=[href])


'''Phrases that show a result title is a headline, not a company name.'''
MARKETING_PHRASES: Tuple[str, ...] = (
  'your solution', 'working for every', 'from pipeline to', 'india mobile',
  'the future of', 'we are hiring', 'join us', 'about us', 'our story',
  'careers at', 'life at', 'shop online', 'buy ', 'order ', 'download',
  'blog', 'news', 'sign up', 'log in', 'dashboard', 'platform for',
  'redefined', 'reimagined', 'revolutionizing', 'transforming', 'unlock',
)

'''A candidate name longer than this is a headline, not a company.'''
MAX_COMPANY_NAME_LEN = 48


def _plausible_company_name(name: str) -> bool:
  '''Report whether a recovered string can be a real company name.

  LinkedIn slugs sometimes capture the job headline rather than the
  employer, producing values such as "Online Shopping India Mobile" or
  "Meet Visa. A network working for every". Such values are rejected so a
  marketing phrase is never recorded as a company.

  Args:
    name: Recovered company name.

  Returns:
    True when the name is short enough and free of marketing phrasing.
  '''
  cleaned = ' '.join(name.split())
  if len(cleaned) > MAX_COMPANY_NAME_LEN or len(cleaned) < 2:
    return False
  lowered = cleaned.lower()
  if any(phrase in lowered for phrase in MARKETING_PHRASES):
    return False
  if any(token in STOP_WORDS_SHAPE for token in lowered.split()):
    return False
  if cleaned.count(' ') > 5:
    return False
  return True


def _company_from_title(title: str) -> str:
  '''Extract a company name from a ``<Role> at <Company>`` result title.

  Args:
    title: Raw search result title.

  Returns:
    The company name, or an empty string when the pattern does not match.
  '''
  cleaned = clean_result_title(title)
  match = re.search(r'\s+at\s+(?P<company>.+)$', cleaned, re.IGNORECASE)
  if not match:
    return ''
  company = match.group('company').strip(' -\u2013\u2014|,')
  return '' if len(company) < 2 or company.lower() in _GENERIC_QUERY_TOKENS else company


class LinkedinCompanyScout:
  '''Discover hiring companies from LinkedIn jobs and verify their sites.

  The scout is composed of four collaborators -- a search index, a website
  verifier, a vertical classifier, and a careers-page detector -- and each
  stage of the pipeline is a separate public method so it can be run and
  tested independently.
  '''

  def __init__(
    self,
    search: WebSearchIndex,
    verifier: WebsiteVerifier,
    classifier: VerticalClassifier,
    detector: CareerPageDetector,
    http: HttpClient,
    repos: CompaniesRepository,
  ) -> None:
    '''Initialize the scout.

    Args:
      search: Search index used to locate LinkedIn job pages.
      verifier: Website verifier that proves company domains.
      classifier: Vertical classifier for industry labelling.
      detector: Careers-page detector used to find ATS boards.
      http: Shared HTTP client owned by the caller.
      repos: Companies repository used for known-company lookups.
    '''
    self._search = search
    self._verifier = verifier
    self._classifier = classifier
    self._detector = detector
    self._http = http
    self._repos = repos

  async def collect_candidates(self, queries: Sequence[str], limit: int = 25) -> Tuple[Dict[str, CompanyCandidate], int, int]:
    '''Run every query and merge the companies they reveal.

    Args:
      queries: Search queries to run.
      limit: Results requested per query.

    Returns:
      Tuple of (candidates keyed by normalized name, linkedin hit count,
      agency candidates dropped).
    '''
    merged: Dict[str, CompanyCandidate] = {}
    hits = 0
    dropped = 0
    for query in queries:
      results = await self._search.search(query, limit=limit)
      for item in results:
        href = str(item.get('href') or '')
        title = str(item.get('title') or '')
        if not is_linkedin_job_url(href):
          continue
        hits += 1
        candidate = candidate_from_result(title, href)
        if candidate is None:
          dropped += 1
          continue
        key = normalize_name(candidate.name)
        if key in merged:
          merged[key].absorb(candidate)
        else:
          merged[key] = candidate
    return merged, hits, dropped

  async def verify_candidate(self, candidate: CompanyCandidate) -> Optional[ScoutedCompany]:
    '''Verify one candidate's website and gather its enrichment.

    Args:
      candidate: Company recovered from a LinkedIn job result.

    Returns:
      A :class:`ScoutedCompany`, or None when the website could not be
      proven.
    '''
    site = await self._verifier.verify(candidate.name, candidate.slug)
    if site is None or is_blocked_domain(site.domain):
      return None
    name = await self._resolve_name(candidate, site)
    vertical, confidence = await self._classify(site.domain)
    return ScoutedCompany(
      name=name,
      domain=site.domain,
      site=site,
      vertical=vertical,
      vertical_confidence=confidence,
      slug=candidate.slug,
      sightings=candidate.sightings,
    )

  async def _resolve_name(self, candidate: CompanyCandidate, site: VerifiedSite) -> str:
    '''Prefer a real display name over a slug-derived approximation.

    LinkedIn slugs are lowercased and lose brand casing, so a company found
    as ``kipi-ai`` is stored as ``Kipi Ai``. When the verified homepage
    carries a proper title, that spelling wins.

    Args:
      candidate: Candidate the site was verified for.
      site: Successful verification result.

    Returns:
      The display name to store.
    '''
    fallback = candidate.name
    if not candidate.slug:
      return fallback
    try:
      html = await self._http.get_text(site.url, rpm=20)
    except Exception:  # noqa: BLE001 - naming is best effort
      return fallback
    if not html:
      return fallback
    title = BeautifulSoup(html, 'html.parser').title
    if title is None:
      return fallback
    text = ' '.join(title.get_text(' ', strip=True).split())
    tokens = set(tokenize(text))
    if text and tokens and set(tokenize(fallback)) <= tokens and _plausible_company_name(text):
      return text
    return fallback

  async def _classify(self, domain: str) -> Tuple[Optional[str], float]:
    '''Classify a verified company from its own website copy.

    Only rule-based classification is used here: it needs no LLM budget and
    the marketing copy on the homepage carries the industry keywords.

    Args:
      domain: Verified company domain.

    Returns:
      (vertical, confidence); (None, 0.0) when rules do not fire.
    '''
    try:
      html = await self._http.get_text(f'https://www.{domain}/', rpm=20)
    except Exception:  # noqa: BLE001 - classification is best effort
      return None, 0.0
    if not html:
      return None, 0.0
    return self._classifier.classify_rules(_strip_html(html)[:6000])

  async def detect_ats(self, company: ScoutedCompany) -> None:
    '''Detect and record a company's ATS board in place.

    Args:
      company: Company whose ATS details should be filled in.
    '''
    if not company.domain:
      return
    try:
      found = await self._detector.detect(company.domain, name=company.name)
    except Exception as exc:  # noqa: BLE001 - detection is best effort
      logger.debug('ats detection failed for %s: %s', company.domain, exc)
      return
    if found is None:
      return
    company.ats_provider, company.board_ref, company.careers_url = found

  async def scout(
    self,
    queries: Sequence[str],
    result_limit: int = 25,
    verify_ats: bool = True,
    ingest: bool = True,
  ) -> ScoutReport:
    '''Run the full discovery pipeline and return a report.

    Args:
      queries: Search queries to run.
      result_limit: Results requested per query.
      verify_ats: Whether to probe careers pages for ATS boards.
      ingest: Whether to write verified companies to the companies table.

    Returns:
      A :class:`ScoutReport` describing the run.
    '''
    report = ScoutReport(queries_run=len(queries))
    candidates, hits, dropped = await self.collect_candidates(queries, limit=result_limit)
    report.linkedin_hits = hits
    report.skipped_agency = dropped
    report.candidates = len(candidates)
    for key, candidate in sorted(candidates.items()):
      if self._repos.resolve_alias(candidate.name) is not None:
        report.already_known += 1
        continue
      company = await self.verify_candidate(candidate)
      if company is None:
        report.unverified += 1
        continue
      if verify_ats:
        await self.detect_ats(company)
        if company.ats_provider:
          report.ats_verified += 1
      report.verified += 1
      if ingest:
        if self._persist(company):
          report.ingested += 1
      report.companies.append(company)
    report.companies.sort(key=lambda item: (-item.sightings, item.name.lower()))
    return report

  def _persist(self, company: ScoutedCompany) -> bool:
    '''Insert or update a scouted company.

    Args:
      company: Verified company to write.

    Returns:
      True when a row was written.
    '''
    try:
      company_id = self._repos.upsert(
        name=company.name,
        domain=company.domain,
        vertical=company.vertical,
        confidence=company.vertical_confidence or None,
        ats_provider=company.ats_provider,
        board_ref=company.board_ref,
        careers_url=company.careers_url,
        priority=3 if company.sightings >= 3 else 2,
        status=company.status,
        discovered_via=f'linkedin_scout:{company.slug or "search"}',
      )
    except Exception as exc:  # noqa: BLE001 - one bad row must not stop a run
      logger.warning('could not persist %s: %s', company.name, exc)
      return False
    if not company_id:
      return False
    self._repos.add_alias(company.name, company_id)
    if company.slug:
      self._repos.add_alias(company.slug, company_id)
    return True


def write_seed_file(path, companies: Sequence[ScoutedCompany], header: str = '') -> int:
  '''Write scouted companies to a seeds YAML file.

  Args:
    path: Destination file path.
    companies: Companies to write, as produced by a scout run.
    header: Optional comment header.

  Returns:
    Number of companies written.
  '''
  import yaml
  from pathlib import Path
  target = Path(path)
  existing: List[Dict[str, object]] = []
  if target.exists():
    try:
      loaded = yaml.safe_load(target.read_text(encoding='utf-8')) or {}
      existing = list(loaded.get('companies') or [])
    except Exception:  # noqa: BLE001 - a bad file is rewritten from scratch
      existing = []
  known = {normalize_name(str(item.get('name') or '')) for item in existing}
  added = 0
  for company in companies:
    key = normalize_name(company.name)
    if not key or key in known:
      continue
    known.add(key)
    existing.append(company.to_seed_entry())
    added += 1
  lines = [f'# {header}'.rstrip(), 'companies:'] if header else ['companies:']
  for item in existing:
    lines.append('  - ' + _inline_mapping(item))
  target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
  return added


def _inline_mapping(entry: Dict[str, object]) -> str:
  '''Render a mapping as a single-line YAML flow mapping.

  Args:
    entry: Mapping to render.

  Returns:
    Flow-style YAML text without the leading list dash.
  '''
  parts: List[str] = []
  for key, value in entry.items():
    if isinstance(value, str):
      rendered = f'"{value}"' if (',' in value or ':' in value or value.strip() != value) else value
    else:
      rendered = str(value).lower() if isinstance(value, bool) else str(value)
    parts.append(f'{key}: {rendered}')
  return '{' + ', '.join(parts) + '}'


def make_scout(settings, seeds_dir=None) -> Tuple[LinkedinCompanyScout, WebSearchIndex, HttpClient]:
  '''Build a wired scout from application settings.

  Args:
    settings: Application settings.
    seeds_dir: Optional seeds directory override for the taxonomy file.

  Returns:
    Tuple of (scout, search index, HTTP client). The caller owns the HTTP
    client and must close it.
  '''
  from pathlib import Path
  config = settings.scout
  tlds = tuple(config.get('domain_tlds') or ('com', 'in', 'io', 'ai', 'co', 'net', 'tech'))
  http = HttpClient()
  search = WebSearchIndex(
    cache_path=settings.data_dir / 'search_cache.json',
    min_interval=float(config.get('search_min_interval_seconds', 3.0)),
  )

  async def _search_bridge(query: str) -> List[str]:
    '''Return search result URLs for a company-domain query.

    Args:
      query: Search query text.

    Returns:
      Candidate URLs with search engines and noise hosts removed.
    '''
    return [url for url in await search.urls(query, 12) if not is_search_noise(url)]

  verifier = WebsiteVerifier(
    http,
    search=_search_bridge,
    tlds=tlds,
    min_confidence=float(config.get('min_domain_confidence', 0.6)),
    probe_budget=int(config.get('probe_budget', 6)),
  )
  taxonomy_dir = Path(seeds_dir) if seeds_dir else settings.seeds_dir
  classifier = VerticalClassifier(taxonomy_dir / 'verticals.yaml')
  detector = CareerPageDetector(http)
  repos = CompaniesRepository(settings.db_path)
  return LinkedinCompanyScout(search, verifier, classifier, detector, http, repos), search, http


def _strip_html(html: str) -> str:
  '''Convert HTML to plain text for keyword classification.

  Args:
    html: Raw page source.

  Returns:
    Whitespace-collapsed plain text.
  '''
  soup = BeautifulSoup(html or '', 'html.parser')
  for tag in soup(['script', 'style', 'noscript']):
    tag.decompose()
  return ' '.join(soup.get_text(' ').split())


_ = registrable_domain
