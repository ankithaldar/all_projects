#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''RED tests for R2B YAGNI deletions and dedup (TDD fix task).'''

from __future__ import annotations


def test_registry_deleted_symbols_gone() -> None:
  '''Dead registry helpers raise AttributeError on access.

  Args:
    None: Unused.
  '''
  from job_hunter.adapters import registry

  assert not hasattr(registry, 'fetch_for_target')
  assert not hasattr(registry, 'detect_provider_from_urls')
  assert hasattr(registry, 'build_adapter')
  assert hasattr(registry, 'REGISTRY')


def test_health_removed_from_owned_adapters() -> None:
  '''Abstract and owned health probes are deleted.

  Args:
    None: Unused.
  '''
  from job_hunter.adapters.aggregators import AggregatorAdapter
  from job_hunter.adapters.base import SourceAdapter
  from job_hunter.adapters.cutshort import CutshortAdapter

  assert not hasattr(SourceAdapter, 'health')
  assert not hasattr(AggregatorAdapter, 'health')
  assert not hasattr(CutshortAdapter, 'health')
  assert hasattr(SourceAdapter, 'fetch')


def test_strip_html_deduped_to_normalizer() -> None:
  '''Local HTML strippers are gone; canonical converter works.

  Args:
    None: Unused.
  '''
  from job_hunter.adapters import aggregators, cutshort
  from job_hunter.services import linkedin_scout
  from job_hunter.services.normalizer import html_to_text

  assert not hasattr(aggregators, '_strip_html')
  assert not hasattr(cutshort, '_strip_html')
  assert not hasattr(linkedin_scout, '_strip_html')
  assert html_to_text('<p>Hello <b>World</b></p>') == 'Hello World'
  assert 'secret' not in html_to_text('<p>a</p><script>secret</script>')
  assert html_to_text('plain   text') == 'plain text'


def test_aggregator_parsers_use_canonical_text() -> None:
  '''Parsers still strip markup through the canonical helper.

  Args:
    None: Unused.
  '''
  from job_hunter.adapters.aggregators import parse_remotive

  records = parse_remotive(
    {'jobs': [{
      'id': '1',
      'url': 'https://j.example/1',
      'company_name': 'Acme',
      'title': 'Data Scientist',
      'description': '<p>Build <b>models</b></p>',
    }]}
  )
  assert len(records) == 1
  assert records[0].description_text == 'Build models'


def test_canonical_registrable_domain_handles_suffixes() -> None:
  '''Canonical domain helper preserves multi-part suffixes.

  Args:
    None: Unused.
  '''
  from job_hunter.services.domain_verifier import registrable_domain

  assert registrable_domain('https://careers.acme.co.in/x') == 'acme.co.in'
  assert registrable_domain('https://www.exlservice.com/') == 'exlservice.com'
  url = 'https://careers.burohappold.com/x'
  assert registrable_domain(url) == 'burohappold.com'


def test_registrable_wrapper_deleted_canonical_kept() -> None:
  '''Dead wrapper is deleted; canonical helper still extracts hosts.

  Args:
    None: Unused.
  '''
  from job_hunter.services import company_discovery
  from job_hunter.services.domain_verifier import registrable_domain

  assert not hasattr(company_discovery, '_registrable')
  assert registrable_domain('https://jobs.example.com/careers') == (
    'example.com'
  )
  assert registrable_domain('https://example.com/') == 'example.com'


def test_host_of_deleted_search_noise_works() -> None:
  '''Host helper is deleted; noise check still delegates.

  Args:
    None: Unused.
  '''
  from job_hunter.services import web_search

  assert not hasattr(web_search, 'host_of')
  assert web_search.is_search_noise('https://www.google.com/search')
  assert web_search.is_search_noise('https://duckduckgo.com/html/')
  assert not web_search.is_search_noise('https://example.com/jobs')


def test_company_repo_deleted_symbols_gone() -> None:
  '''Unused company repository helpers are deleted.

  Args:
    None: Unused.
  '''
  from job_hunter.db.repositories import companies as companies_mod
  from job_hunter.db.repositories.companies import CompaniesRepository

  assert not hasattr(CompaniesRepository, 'due_for_refresh')
  assert not hasattr(companies_mod, 'stats_summary')
  assert hasattr(CompaniesRepository, 'upsert')


def test_jobs_repo_deleted_symbols_gone() -> None:
  '''Unused job repository helpers are deleted.

  Args:
    None: Unused.
  '''
  from job_hunter.db.repositories.jobs import JobsRepository

  assert not hasattr(JobsRepository, 'raw_json_of')
  assert not hasattr(JobsRepository, 'unseen_hashes')
  assert hasattr(JobsRepository, 'get')
  assert hasattr(JobsRepository, 'exists_hash')


def test_core_deleted_symbols_gone() -> None:
  '''Unused core helpers and errors are deleted.

  Args:
    None: Unused.
  '''
  from job_hunter.core import errors, models
  from job_hunter.core.config import AppSettings

  assert not hasattr(models, 'utcnow')
  assert not hasattr(errors, 'NotFoundError')
  assert not hasattr(AppSettings, 'section')
  assert hasattr(errors, 'ConflictError')
  assert hasattr(errors, 'DatabaseError')
  assert hasattr(errors, 'AdapterError')


def test_llm_deleted_symbols_gone_kept_work() -> None:
  '''Deleted LLM modules are gone; kept client works.

  Args:
    None: Unused.
  '''
  import importlib.util

  from job_hunter.llm import client as client_mod
  from job_hunter.llm.client import GatewayClient

  assert not hasattr(GatewayClient, 'stream_text')
  assert hasattr(client_mod, 'reset_client')
  assert hasattr(client_mod, 'get_client')
  assert importlib.util.find_spec('job_hunter.llm.mcp_tools') is None
  assert (
    importlib.util.find_spec('job_hunter.llm.langchain_bridge') is None
  )


def test_discovery_deleted_paths_gone_seeds_kept() -> None:
  '''Dead expand and harvest paths are deleted; seeds stay.

  Args:
    None: Unused.
  '''
  from job_hunter.services import company_discovery

  assert not hasattr(company_discovery, 'expand_via_search')
  assert not hasattr(company_discovery, 'harvest_from_jobs')
  assert hasattr(company_discovery, 'ingest_seeds')
  assert not hasattr(company_discovery, '_json')
  assert not hasattr(company_discovery, 'asyncio')


def test_verifier_haystack_uses_canonical_text() -> None:
  '''Haystack builder normalizes markup via canonical path.

  Args:
    None: Unused.
  '''
  from job_hunter.adapters.http_client import HttpClient
  from job_hunter.services.domain_verifier import WebsiteVerifier

  verifier = WebsiteVerifier(HttpClient())
  haystack = verifier._page_haystack(
    '<title>Acme Widgets</title><body><p>Acme <b>rockets</b></p></body>',
    'https://www.acme.com/',
  )
  assert 'acme rockets' in haystack
  assert '<b>' not in haystack


def test_scout_still_classifies_page_text() -> None:
  '''Scout module still classifies through canonical text.

  Args:
    None: Unused.
  '''
  from job_hunter.services import linkedin_scout
  from job_hunter.services.normalizer import html_to_text

  assert hasattr(linkedin_scout, 'LinkedinCompanyScout')
  assert html_to_text('<p>Data science platform</p>') == 'Data science platform'
  job_url = (
    'https://www.linkedin.com/jobs/view/'
    'staff-data-scientist-at-acme-ai-1234567'
  )
  candidate = linkedin_scout.candidate_from_result(
    'Staff Data Scientist at Acme AI',
    job_url,
  )
  assert candidate is not None
  assert candidate.name == 'Acme Ai'
