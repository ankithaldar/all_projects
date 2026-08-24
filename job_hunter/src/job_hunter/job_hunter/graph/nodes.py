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
  profiles_repo = ProfileRepository(settings.db_path)
  data = profiles_repo.get_profile(1) or {}
  floor = SettingsRepository(settings.db_path).get('salary_hard_floor_lpa', 45.0)
  skills = profiles_repo.candidate_skill_names(1)
  candidate = CandidateProfile(
    skills=skills,
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
    watermarks = [
      str(record.posted_at) for record in records if record.posted_at
    ]
    raw_jobs = records
    stats[f"fetched:{target_data['source_key']}"] = len(records)
    crawl.set_success(scope, cursor=max(watermarks) if watermarks else None)
    if watermarks:
      crawl.set_success(f'{scope}:posted', cursor=max(watermarks))
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
      inserted.append(normalized.model_copy(update={'is_new': True, 'job_id': new_id}))
    except Exception as exc:
      logger.warning('insert failed for %s: %s', normalized.canonical_url, exc)
  stats = {'new_jobs': len(inserted), 'duplicates': dupes}
  return {
    'inserted': inserted,
    'stats': stats,
  }


async def enrich_jds(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
  '''Extract structured fields for newly inserted postings.

  Args:
    state: Current state.
    config: Runnable config.

  Returns:
    Enrichment map, updated budget, and counters.
  '''
  from job_hunter.core.models import TokenBudget
  from job_hunter.llm.client import get_client
  from job_hunter.services.jd_analyst import JDAnalyst, apply_salary_floor
  from job_hunter.services.skills_taxonomy import SkillsTaxonomy

  settings = _settings_from_config(config)
  jobs_repo = JobsRepository(settings.db_path)
  taxonomy = SkillsTaxonomy(settings.db_path)
  try:
    analyst = JDAnalyst(get_client(settings))
  except Exception as exc:
    logger.warning('gateway unavailable for enrichment: %s', exc)
    return {
      'enriched': dict(state.get('enriched') or {}),
      'budget': state.get('budget') or {},
      'stats': {'enrich_failures': len(state.get('inserted') or [])},
    }
  run_id = int(state['run_id'])

  budget_data = state.get('budget') or {'max_calls': int(
    settings.discovery.get('max_llm_calls_per_run', 300),
  )}
  budget = TokenBudget(**budget_data)
  floor = float(SettingsRepository(settings.db_path).get('salary_hard_floor_lpa', 45.0))
  enriched_map: Dict[str, Any] = dict(state.get('enriched') or {})
  stats: Dict[str, int] = {}

  for job in state.get('inserted') or []:
    if not budget.allow():
      stats['enrich_skipped_budget'] = stats.get('enrich_skipped_budget', 0) + 1
      continue
    row = jobs_repo.get(int(job.job_id))
    description = (row or {}).get('description_text') or ''
    if len(description) < 200:
      stats['enrich_skipped_short'] = stats.get('enrich_skipped_short', 0) + 1
      continue
    budget.spend()
    try:
      extraction = await analyst.extract(
        title=job.raw.title,
        description_text=description,
        session_id=f'{run_id}:jd_analyst',
      )
    except Exception as exc:
      logger.warning('extraction failed job %s: %s', job.job_id, exc)
      stats['enrich_failures'] = stats.get('enrich_failures', 0) + 1
      continue

    sal_min = extraction.salary_min_lpa if extraction.salary_min_lpa is not None else job.salary_min_lpa
    sal_max = extraction.salary_max_lpa if extraction.salary_max_lpa is not None else job.salary_max_lpa
    mode = extraction.work_mode if extraction.work_mode != WorkMode.UNKNOWN else job.work_mode
    jobs_repo.update_fields(int(job.job_id), {
      'salary_min_lpa': sal_min,
      'salary_max_lpa': sal_max,
      'experience_min_yrs': extraction.experience_min_years,
      'experience_max_yrs': extraction.experience_max_years,
      'work_mode': mode,
      'employment_type': extraction.employment_type,
    })
    skill_rows: List[Dict[str, Any]] = []
    for kind, names in (
      ('must_have', extraction.must_have_skills),
      ('nice_to_have', extraction.nice_to_have_skills),
    ):
      for name in names:
        skill_id = taxonomy.resolve(name)
        if skill_id is None:
          taxonomy.load_seed({'skills': {name: None}})
          skill_id = taxonomy.resolve(name)
        if skill_id is not None:
          skill_rows.append({
            'skill_id': skill_id,
            'kind': kind,
            'confidence': extraction.confidence,
          })
    jobs_repo.attach_skill_rows(int(job.job_id), skill_rows)

    passes, reason = apply_salary_floor(extraction, floor)
    if not passes:
      stats['below_salary_floor'] = stats.get('below_salary_floor', 0) + 1
      logger.info('job %s filtered by floor: %s', job.job_id, reason)

    enriched_map[job.content_hash] = {
      'must_have_skills': extraction.must_have_skills,
      'nice_to_have_skills': extraction.nice_to_have_skills,
      'experience_min_years': extraction.experience_min_years,
      'experience_max_years': extraction.experience_max_years,
      'salary_min_lpa': sal_min,
      'salary_max_lpa': sal_max,
      'work_mode': mode,
      'employment_type': extraction.employment_type,
      'confidence': extraction.confidence,
      'needs_review': extraction.confidence < 0.5,
      'passes_floor': passes,
    }
    stats['enriched'] = stats.get('enriched', 0) + 1

  return {
    'enriched': enriched_map,
    'budget': budget.model_dump(),
    'stats': stats,
  }


async def compute_embeddings(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
  '''Embed newly inserted postings when a provider is available.

  Args:
    state: Current state.
    config: Runnable config.

  Returns:
    Counters for embedded and skipped postings.
  '''
  from job_hunter.services.embedder import get_embedder

  settings = _settings_from_config(config)
  jobs_repo = JobsRepository(settings.db_path)
  embedder = get_embedder(settings)
  stats: Dict[str, int] = {'embeddings': 0, 'embedding_skipped': 0}
  if embedder.model_id == NullProvider_model_id():
    stats['embedding_skipped'] = len(state.get('inserted') or [])
    return {'stats': stats}
  pending = [job for job in state.get('inserted') or [] if job.job_id]
  if not pending:
    return {'stats': stats}
  texts = []
  for job in pending:
    row = jobs_repo.get(int(job.job_id)) or {}
    texts.append(f"{job.raw.title}\n{(row.get('description_text') or '')[:1800]}")
  try:
    vectors = embedder.embed(texts)
  except Exception as exc:
    logger.warning('embedding failed: %s', exc)
    stats['embedding_skipped'] = len(pending)
    return {'stats': stats}
  if vectors is None:
    stats['embedding_skipped'] = len(pending)
    return {'stats': stats}
  for job, vector in zip(pending, vectors):
    jobs_repo.save_embedding(int(job.job_id), embedder.model_id, vector)
    stats['embeddings'] += 1
  return {'stats': stats}


def NullProvider_model_id() -> str:
  '''Return the fallback provider marker id without importing heavy deps.

  Returns:
    Marker string.
  '''
  from job_hunter.services.embedder import NullProvider
  return NullProvider.model_id




def _candidate_skill_ids(settings: AppSettings, candidate: CandidateProfile) -> List[int]:
  '''Resolve the candidate's stored skill ids.

  Args:
    settings: Application settings.
    candidate: Profile.

  Returns:
    Skill id list.
  '''
  from job_hunter.services.skills_taxonomy import SkillsTaxonomy
  taxonomy = SkillsTaxonomy(settings.db_path)
  return [
    taxonomy.resolve(name) or -1
    for name in candidate.skills
    if taxonomy.resolve(name) is not None
  ]


async def score_rank_persist(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
  '''Gate, score, rank, and persist recommendations for inserted jobs.

  Args:
    state: Current state.
    config: Runnable config.

  Returns:
    Scored jobs snapshot and counters.
  '''
  import json as _json
  from job_hunter.core.models import ScoredJob
  from job_hunter.db.repositories.recommendations import RecommendationsRepository
  from job_hunter.db.repositories.settings import SettingsRepository as SettingsRepo
  from job_hunter.services.embedder import cosine
  from job_hunter.services.matcher import (
    company_fit,
    gate_failures,
    recency_score,
    salary_fit,
    seniority_fit,
    semantic_fallback,
    skill_coverage,
    title_fit,
  )
  from job_hunter.services.scorer import aggregate, normalize_weights, rationale

  settings = _settings_from_config(config)
  jobs_repo = JobsRepository(settings.db_path)
  recs_repo = RecommendationsRepository(settings.db_path)
  weights = normalize_weights(SettingsRepo(settings.db_path).get('scoring_weights', {}))
  candidate = state['candidate']
  cand_skill_ids = _candidate_skill_ids(settings, candidate)
  embeddings = jobs_repo.load_embeddings(str(settings.embeddings.get('model', 'BAAI/bge-small-en-v1.5')))
  candidate_doc = f"{candidate.summary} {' '.join(candidate.skills)} {' '.join(candidate.target_roles)}"
  candidate_vector = None
  if embeddings:
    from job_hunter.services.embedder import get_embedder
    embedder = get_embedder(settings)
    vectors = embedder.embed([candidate_doc[:2000]])
    candidate_vector = vectors[0] if vectors else None

  scored: List[ScoredJob] = []
  for job in state.get('inserted') or []:
    row = jobs_repo.get_with_company(int(job.job_id)) or {}
    enriched = (state.get('enriched') or {}).get(job.content_hash)
    failures = gate_failures(candidate, row, enriched)
    job_skills = _job_skill_rows(jobs_repo, int(job.job_id))
    must_cov, nice_cov = skill_coverage(cand_skill_ids, job_skills)
    description = (row.get('description_text') or '')[:2500]
    vector = embeddings.get(int(job.job_id))
    if candidate_vector is not None and vector is not None:
      semantic = max(0.0, min(1.0, (cosine(candidate_vector, vector) + 1.0) / 2.0))
    else:
      semantic = semantic_fallback(candidate_doc, f"{row.get('title')} {description}")
    components = {
      'skills_must': must_cov,
      'skills_nice': nice_cov,
      'semantic': round(semantic, 4),
      'seniority': seniority_fit(
        float(candidate.experience_years or 0),
        row.get('experience_min_yrs'),
        row.get('experience_max_yrs'),
      ),
      'title_fit': title_fit(candidate.target_roles, row.get('title') or ''),
      'salary_fit': salary_fit(row.get('salary_min_lpa'), row.get('salary_max_lpa'), candidate.salary_floor_lpa),
      'recency': recency_score(row.get('posted_at')),
      'company_fit': company_fit(
        row.get('vertical'),
        candidate.target_verticals,
        int(row.get('company_priority') or 3),
      ),
    }
    total, breakdown = aggregate(components, weights)
    scored.append(ScoredJob(
      job_id=int(job.job_id),
      title=row.get('title') or '',
      url=row.get('url') or '',
      company_id=job.company_id,
      company_name=row.get('company_name') or job.raw.company_name,
      vertical=row.get('vertical') or 'unknown',
      total_score=total if not failures else 0.0,
      gate_pass=not failures,
      gate_failures=failures,
      breakdown=breakdown,
      rationale=rationale(components),
      posted_at=row.get('posted_at'),
    ))

  passing = sorted(
    [s for s in scored if s.gate_pass],
    key=lambda s: (-s.total_score, s.posted_at or ''),
  )
  rows = [
    {
      'job_id': item.job_id,
      'total_score': item.total_score,
      'breakdown': item.breakdown,
      'rationale': item.rationale,
    }
    for item in passing
  ]
  written = _persist_recs(recs_repo, int(state['run_id']), rows) if rows else 0

  stats = {
    'scored': len(scored),
    'gate_passed': len(passing),
    'recommendations': written,
  }
  return {'scored': scored, 'stats': stats}


def _job_skill_rows(jobs_repo, job_id: int) -> List[Dict[str, Any]]:
  '''Fetch skill link rows for one job.

  Args:
    jobs_repo: Jobs repository.
    job_id: Job id.

  Returns:
    Rows with skill_id/kind.
  '''
  from job_hunter.core.db import connect
  rows = connect(jobs_repo._db_path, readonly=True).execute(
    'SELECT skill_id, kind FROM job_skills WHERE job_id = ?', (job_id,),
  ).fetchall()
  return [dict(row) for row in rows]


def _persist_recs(recs_repo, run_id: int, rows: List[Dict[str, Any]]) -> int:
  '''Persist ranked recommendation rows.

  Args:
    recs_repo: Recommendations repository.
    run_id: Run id.
    rows: Prepared payload rows.

  Returns:
    Count written.
  '''
  normalized = []
  for position, row in enumerate(rows, start=1):
    normalized.append({
      'job_id': row['job_id'],
      'total_score': row['total_score'],
      'gate_pass': True,
      'gate_failures': [],
      'breakdown': row['breakdown'],
      'rationale': row['rationale'],
      'rank': position,
    })
  return recs_repo.upsert_many(run_id, 1, normalized)


async def summarize_run(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """Append a Watchdog quality summary to the run's event stream.

    Args:
      state: Current state.
      config: Runnable config.

    Returns:
      Watchdog issue counter.
    """
    from job_hunter.services.watchdog import build_report, summarize_issues

    settings = _settings_from_config(config)
    report = build_report(settings)
    issues = summarize_issues(report)
    for issue in issues:
        logger.warning('watchdog: %s', issue)
    return {'stats': {'watchdog_issues': len(issues)}}
