#!/usr/bin/env python
# -- coding: utf-8 --

'''Tests for company scouting, domain verification, and bulk fetch helpers.'''


from __future__ import annotations

import pytest
from job_hunter.services.bulk_fetch import resolve_sources
from job_hunter.services.domain_verifier import (
  DomainCandidate,
  SiteProbe,
  WebsiteVerifier,
  is_blocked_domain,
  registrable_domain,
  sld_of,
  slug_candidates,
  tokenize,
)
from job_hunter.services.linkedin_scout import (
  _plausible_company_name,
  build_queries,
  candidate_from_result,
  is_linkedin_job_url,
  looks_like_agency,
  write_seed_file,
)


def test_tokenize_drops_legal_suffixes_and_stops() -> None:
  '''Legal suffixes and stop words never become company tokens.

  Args:
    None: Unused.
  '''
  assert tokenize('Acme Technologies India Private Limited') == ['acme']
  assert tokenize('Zeta') == ['zeta']


def test_registrable_domain_collapses_subdomains() -> None:
  '''Subdomains collapse and multi-part suffixes are preserved.

  Args:
    None: Unused.
  '''
  assert registrable_domain('https://careers.burohappold.com/x') == 'burohappold.com'
  assert registrable_domain('https://www.exlservice.com/') == 'exlservice.com'
  assert registrable_domain('https://acme.co.in/') == 'acme.co.in'


def test_sld_of_uses_second_level_label() -> None:
  '''The second-level label is extracted from a registrable domain.

  Args:
    None: Unused.
  '''
  assert sld_of('burohappold.com') == 'burohappold'
  assert sld_of('acme.co.in') == 'acme'


def test_blocked_domains_exclude_boards_and_aggregators() -> None:
  '''Job boards and job aggregators are never accepted as websites.

  Args:
    None: Unused.
  '''
  assert is_blocked_domain('linkedin.com')
  assert is_blocked_domain('boards.greenhouse.io')
  assert is_blocked_domain('www.indeed.com')
  assert not is_blocked_domain('zomato.com')


def test_slug_candidates_prefer_slug_stem() -> None:
  '''The board slug drives the first domain guesses.

  Args:
    None: Unused.
  '''
  domains = slug_candidates('EXL', 'exl-service', ('com', 'in'))
  assert domains[0] == 'exlservice.com'
  assert 'exl.com' in domains


def test_linkedin_job_url_detection() -> None:
  '''Only LinkedIn job view URLs are treated as job links.

  Args:
    None: Unused.
  '''
  assert is_linkedin_job_url('https://in.linkedin.com/jobs/view/ds-at-acme-1234567890')
  assert not is_linkedin_job_url('https://in.linkedin.com/company/acme')
  assert not is_linkedin_job_url('https://example.com/jobs/view/1')


def test_candidate_from_url_slug_and_title() -> None:
  '''Company names come from the URL slug, then the result title.

  Args:
    None: Unused.
  '''
  from_slug = candidate_from_result(
    'Data Scientist at Acme | LinkedIn',
    'https://in.linkedin.com/jobs/view/data-scientist-at-acme-ai-4451195844',
  )
  assert from_slug is not None
  assert from_slug.name == 'Acme Ai'
  assert from_slug.slug == 'data-scientist-at-acme-ai'

  from_title = candidate_from_result(
    'Senior Analyst at Zenith Labs | LinkedIn',
    'https://www.linkedin.com/jobs/view/4455637314/',
  )
  assert from_title is not None
  assert from_title.name == 'Zenith Labs'
  assert from_title.slug == ''


def test_headline_like_names_are_rejected() -> None:
  '''Job headlines captured as names never become companies.

  Args:
    None: Unused.
  '''
  for headline in (
    'Meet Visa. A network working for every',
    'Online Shopping India Mobile, Cameras,',
    'Women In Machine Learning And Data Science',
    'GALA GROUP - Your solution provider',
  ):
    assert not _plausible_company_name(headline)
  for real in ('HackerRank', 'Zomato', 'Buro Happold Engineer India Private Limited'):
    assert _plausible_company_name(real)


def test_agency_candidates_are_rejected() -> None:
  '''Staffing agencies and slogan-piped names are not treated as employers.

  Args:
    None: Unused.
  '''
  assert looks_like_agency('Acme Staffing Services')
  assert looks_like_agency('AEROSPACE REDEFINED | Collins Aerospace')
  assert not looks_like_agency('Acme Analytics')
  assert candidate_from_result(
    'Data Scientist at Acme Staffing | LinkedIn',
    'https://in.linkedin.com/jobs/view/ds-at-acme-staffing-1234567890',
  ) is None


def test_build_queries_is_bounded_and_deterministic() -> None:
  '''The query matrix respects the limit and repeats identically.

  Args:
    None: Unused.
  '''
  first = build_queries(limit=5)
  assert first == build_queries(limit=5)
  assert len(first) == 5
  assert all(query.startswith('site:linkedin.com/jobs/view') for query in first)


def test_score_probe_requires_a_distinctive_token() -> None:
  '''A match on generic words alone never proves a domain.

  Args:
    None: Unused.
  '''
  verifier = WebsiteVerifier.__new__(WebsiteVerifier)
  generic = SiteProbe(domain='engineer.ai', url='https://engineer.ai/', ok=True, haystack='we are an engineer company in india')
  assert verifier.score_probe(generic, ['engineer'], 1.0) == 0.0
  distinctive = SiteProbe(domain='kupi.ai', url='https://kupi.ai/', ok=True, haystack='kupi is an ai platform')
  assert verifier.score_probe(distinctive, ['kupi'], 0.9) > 0.6


def test_write_seed_file_appends_without_duplicates(tmp_path) -> None:
  '''Seed writing merges new companies and keeps existing ones.

  Args:
    tmp_path: Pytest temporary directory.
  '''
  from job_hunter.services.linkedin_scout import ScoutedCompany

  path = tmp_path / 'companies_scouted.yaml'
  first = ScoutedCompany(name='Acme', domain='acme.com', vertical='saas', sightings=4)
  assert write_seed_file(path, [first], header='scouted') == 1
  assert write_seed_file(path, [ScoutedCompany(name='acme', domain='acme.com')]) == 0
  assert write_seed_file(path, [ScoutedCompany(name='Zenith', domain='zenith.io')]) == 1
  text = path.read_text(encoding='utf-8')
  assert 'acme.com' in text and 'zenith.io' in text
  assert text.count('name: Acme') == 1


def test_resolve_sources_defaults_and_dedupes() -> None:
  '''Source resolution falls back to defaults and removes duplicates.

  Args:
    None: Unused.
  '''
  assert 'arbeitnow' in resolve_sources([])
  assert resolve_sources(['jobicy', 'jobicy', 'arbeitnow']) == ['jobicy', 'arbeitnow']


def test_domain_candidate_orders_by_score() -> None:
  '''Candidates sort by descending prior confidence.

  Args:
    None: Unused.
  '''
  low = DomainCandidate(domain='a.com', score=0.3)
  high = DomainCandidate(domain='b.com', score=0.9)
  assert sorted([low, high])[0] is high


def test_scout_rejects_result_without_company() -> None:
  '''A result with no recoverable company yields no candidate.

  Args:
    None: Unused.
  '''
  assert candidate_from_result('Data Scientist | LinkedIn', 'https://www.linkedin.com/jobs/view/4466849157') is None


if __name__ == '__main__':
  pytest.main([__file__])
