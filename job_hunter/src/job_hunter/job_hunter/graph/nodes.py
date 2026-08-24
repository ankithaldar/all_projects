#!/usr/bin/env python
# -- coding: utf-8 --

'''Graph node implementations for the discovery pipeline (v1 stages).'''


from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from job_hunter.adapters.http_client import HttpClient
from job_hunter.adapters.registry import build_adapter
from job_hunter.core.config import AppSettings
from job_hunter.core.models import (
  CandidateProfile,
  CompanyTarget,
  NormalizedJob,
  RawJobRecord,
  RunPlan,
  WorkMode,
)
from job_hunter.db.repositories.crawl_state import CrawlStateRepository
from job_hunter.db.repositories.companies import CompaniesRepository
from job_hunter.db.repositories.jobs import JobsRepository
from job_hunter.db.repositories.profile import ProfileRepository
from job_hunter.db.repositories.settings import SettingsRepository
from job_hunter.services.dedupe import DedupeEngine
from job_hunter.services.normalizer import (
  canonicalize_url,
  content_hash,
  detect_city,
  html_to_text,
  infer_work_mode,
  normalize_employment,
  parse_posted_at,
  parse_salary_lpa,
)

logger = logging.getLogger(__name__)


def _settings_from_config(config: Dict[str, Any]) -> AppSettings:
  '''Extract settings injected via langgraph config.

  Args:
    config: Runnable config mapping.

  Returns:
    Application settings.
  '''
  return config['configurable']['settings']


def load_profile(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
  '''Load the latest candidate profile into state.

  Args:
    state: Current state.
    config: Runnable config.

  Returns:
    State update.
  '''
  settings = _settings_from_config(config)
  data = ProfileRepository(settings.db_path).get_profile(1) or {}
  floor = SettingsRepository(settings.db_path).get('salary_hard_floor_lpa', 45.0)
  candidate = CandidateProfile(
    target_roles=data.get('target_roles') or ['Data Scientist'],
    seniority_keywords=data.get('seniority_keywords') or ['staff', 'senior', 'principal', 'lead'],
    target_verticals=data.get('target_verticals') or [],
    blocked_verticals=data.get('blocked_verticals') or [],
    cities=data.get('cities') or [],
    relocate_ok=bool(data.get('relocate_ok', 0)),
    remote_pref=data.get('remote_pref') or 'any',
    salary_floor_lpa=float(floor),
    experience_years=float(data.get('experience_years') or 0),
    employment_types=data.get('employment_types') or ['full_time'],
    summary=data.get('summary') or '',
  )
  return {'candidate': candidate}


async def build_plan(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
  '''Select fetch targets from active companies.

  Args:
    state: Current state.
    config: Runnable config.

  Returns:
    State update containing the RunPlan.
  '''
  settings = _settings_from_config(config)
  companies = CompaniesRepository(settings.db_path)
  quick = bool(state['plan']['quick_poll']) if isinstance(state.get('plan'), dict) else False
  if quick:
    rows = [row for row in companies.list_companies(status='active') if row.get('priority', 0) >= 5]
  else:
    rows = [
      row for row in companies.list_companies(status='active')
      if row.get('ats_provider') and row.get('board_ref')
    ]
  targets = [
    CompanyTarget(
      company_id=int(row['id']),
      name=row['name'],
      domain=row.get('domain') or '',
      source_key=row['ats_provider'],
      board_ref=row.get('board_ref') or '',
    )
    for row in rows
  ]
  plan = RunPlan(run_id=int(state['run_id']), targets=targets, quick_poll=quick)
  return {'plan': plan.model_dump()}


async def fetch_pair(target_data: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
  '''Fetch postings for one (source, board_ref) pair.

  Args:
    target_data: Serialized CompanyTarget.
    config: Runnable config.

  Returns:
    Raw jobs, stats, and any error captured.
  '''
  settings = _settings_from_config(config)
  crawl = CrawlStateRepository(settings.db_path)
  scope = f"company:{target_data['company_id']}"
  stats: Dict[str, int] = {}
  errors: List[Any] = []
  raw_jobs: List[RawJobRecord] = []
  if crawl.is_cooled_down(scope):
    stats['cooled_skips'] = 1
    return {'raw_jobs': raw_jobs, 'stats': stats, 'errors': errors}
  http = HttpClient()
  try:
    adapter = build_adapter(target_data['source_key'], http)
    target = CompanyTarget(**target_data)
    records = await adapter.fetch(target)
    since = crawl.get_cursor(f'{scope}:posted')
    if since:
      records = [
        record for record in records
        if not record.posted_at or (record.posted_at >= str(since))
      ]
    raw_jobs = records
    stats[f"fetched:{target_data['source_key']}"] = len(records)
    crawl.set_success(scope, cursor=None)
  except Exception as exc:
    failures = crawl.set_failure(scope)
    errors.append({
      'node': 'fetch_pair',
      'message': f"{target_data['source_key']}/{target_data['board_ref']}: {exc}",
      'recoverable': True,
    })
    stats['fetch_failures'] = 1
    logger.warning('fetch failed (%s consecutive): %s', failures, exc)
  finally:
    await http.close()
  return {'raw_jobs': raw_jobs, 'stats': stats, 'errors': errors}


async def normalize_dedupe(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
  '''Normalize fetched records and persist unseen postings.

  Args:
    state: Current state.
    config: Runnable config.

  Returns:
    Inserted rows plus counters.
  '''
  settings = _settings_from_config(config)
  jobs_repo = JobsRepository(settings.db_path)
  companies_repo = CompaniesRepository(settings.db_path)
  engine = DedupeEngine(jobs_repo)
  inserted: List[NormalizedJob] = []
  dupes = 0
  for record in state.get('raw_jobs') or []:
    text_desc = record.description_text or html_to_text(record.description_html)
    digest = content_hash(record, text_desc)
    city = detect_city(record.location_text, text_desc)
    mode = infer_work_mode(record.work_mode_hint, record.location_text, f"{record.title} {text_desc[:300]}")
    sal_min, sal_max = parse_salary_lpa(record.salary_raw, text_desc)
    company_id = companies_repo.resolve_alias(record.company_name) if record.company_name else None
    normalized = NormalizedJob(
      raw=record,
      canonical_url=canonicalize_url(record.url),
      content_hash=digest,
      company_id=company_id,
      city=city,
      region='',
      work_mode=mode,
      employment_type=normalize_employment(record.employment_type_raw),
      salary_min_lpa=sal_min,
      salary_max_lpa=sal_max,
    )
    verdict, existing_id = engine.classify(normalized)
    if verdict != 'new':
      dupes += 1
      continue
    payload = {
      'source_key': record.source_key,
      'external_id': record.external_id,
      'url': record.url,
      'canonical_url': normalized.canonical_url,
      'company_id': company_id,
      'company_raw_name': record.company_name,
      'title': record.title,
      'location_text': record.location_text,
      'city': normalized.city,
      'work_mode': normalized.work_mode,
      'employment_type': normalized.employment_type,
      'salary_min_lpa': normalized.salary_min_lpa,
      'salary_max_lpa': normalized.salary_max_lpa,
      'salary_raw': record.salary_raw,
      'posted_at': parse_posted_at(record.posted_at),
      'description_text': text_desc[:20000],
      'raw_json': record.model_dump_json(),
      'content_hash': digest,
    }
    try:
      new_id = jobs_repo.insert(payload)
      inserted.append(normalized.model_copy(update={'is_new': True}))
      _ = new_id
    except Exception as exc:
      logger.warning('insert failed for %s: %s', normalized.canonical_url, exc)
  stats = {'new_jobs': len(inserted), 'duplicates': dupes}
  return {
    'inserted': inserted,
    'stats': stats,
  }


