#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Graph node implementations for the discovery pipeline (v1 stages).'''

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from job_hunter.adapters.http_client import get_shared_http_client
from job_hunter.adapters.registry import build_adapter
from job_hunter.core.config import AppSettings, source_enabled
from job_hunter.core.logging import node_var
from job_hunter.core.models import (
  CandidateProfile,
  CompanyTarget,
  NodeError,
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
from job_hunter.graph.state import _normalize_inserted, _normalize_raw_jobs
from job_hunter.services.dedupe import DedupeEngine
from job_hunter.services.normalizer import (
  canonicalize_url,
  content_hash,
  detect_city,
  html_to_text,
  infer_work_mode,
  normalize_employment,
  parse_salary_lpa,
)
from job_hunter.core.constants import (
  DEFAULT_EMBED_MODEL,
  DEFAULT_SALARY_FLOOR_LPA,
)

logger = logging.getLogger(__name__)


def safe_node(fn):
  '''Wrap a graph node so exceptions become NodeError entries.

  Supports both async and sync node functions, preserving the
  wrapped function's name/signature via functools.wraps.

  Args:
    fn: Node function (async or sync).

  Returns:
    Wrapped node function returning a state update on success or
    {'errors': [NodeError(...)]}, truncated to 500 chars, on
    exception instead of raising.
  '''
  if inspect.iscoroutinefunction(fn):

    @functools.wraps(fn)
    async def async_wrapper(*args, **kwargs):
      '''Call the async node, converting errors to NodeError.

      Args:
        *args: Positional node arguments.
        **kwargs: Keyword node arguments.

      Returns:
        State update or errors entry.
      '''
      token = node_var.set(fn.__name__)
      try:
        return await fn(*args, **kwargs)
      except asyncio.CancelledError:
        raise
      except KeyboardInterrupt:
        raise
      except Exception as exc:
        logger.warning('node %s failed: %s', fn.__name__, exc)
        err = NodeError(
          node=fn.__name__,
          message=str(exc)[:500],
          recoverable=True,
        )
        return {'errors': [err]}
      finally:
        node_var.reset(token)

    return async_wrapper

  @functools.wraps(fn)
  def sync_wrapper(*args, **kwargs):
    '''Call the sync node, converting errors to NodeError.

    Args:
      *args: Positional node arguments.
      **kwargs: Keyword node arguments.

    Returns:
      State update or errors entry.
    '''
    token = node_var.set(fn.__name__)
    try:
      return fn(*args, **kwargs)
    except asyncio.CancelledError:
      raise
    except KeyboardInterrupt:
      raise
    except Exception as exc:
      logger.warning('node %s failed: %s', fn.__name__, exc)
      err = NodeError(
        node=fn.__name__,
        message=str(exc)[:500],
        recoverable=True,
      )
      return {'errors': [err]}
    finally:
      node_var.reset(token)

  return sync_wrapper


def _settings_from_config(config: Dict[str, Any]) -> AppSettings:
  '''Extract settings injected via langgraph config.

  Args:
    config: Runnable config mapping.

  Returns:
    Application settings.
  '''
  return config['configurable']['settings']


def _normalize_candidate(value: Any) -> CandidateProfile:
  '''Coerce a checkpoint-deserialized candidate back to CandidateProfile.

  Args:
    value: CandidateProfile instance or its dict form.

  Returns:
    CandidateProfile instance.
  '''
  if isinstance(value, dict):
    return CandidateProfile(**value)
  return value


def _load_target_embeddings(
  jobs_repo, model: str, job_ids: List[int],
) -> Dict[int, List[float]]:
  '''Load embeddings for the scoring targets only (F2 job_ids filter).

  Args:
    jobs_repo: Jobs repository.
    model: Embedding model identifier.
    job_ids: Scoring-target job ids.

  Returns:
    Mapping of job id to vector.
  '''
  # F2 adds the job_ids parameter; until it lands, the repository
  # scans every row for the model. Prefer the scoped call.
  params = inspect.signature(jobs_repo.load_embeddings).parameters
  if 'job_ids' in params:
    return jobs_repo.load_embeddings(model, job_ids=job_ids)
  return jobs_repo.load_embeddings(model)


@safe_node
def load_profile(
  state: Dict[str, Any], config: RunnableConfig
) -> Dict[str, Any]:
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
  floor_repo = SettingsRepository(settings.db_path)
  floor = floor_repo.get(
    'salary_hard_floor_lpa', DEFAULT_SALARY_FLOOR_LPA,
  )
  skills = profiles_repo.candidate_skill_names(1)
  candidate = CandidateProfile(
    skills=skills,
    target_roles=data.get('target_roles') or ['Data Scientist'],
    seniority_keywords=data.get('seniority_keywords')
    or ['staff', 'senior', 'principal', 'lead'],
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


def _filtered_due_count(
  settings: AppSettings, sources: dict[str, Any], quick: bool,
) -> int:
  '''Count due rows passing the build_plan Python filters.

  Args:
    settings: Application settings.
    sources: Per-source config mapping.
    quick: Whether the quick-poll priority filter applies.

  Returns:
    Untruncated count of filtered-eligible due rows.
  '''
  import sqlite3
  from contextlib import closing

  sql = (
    'SELECT companies.ats_provider, companies.priority '
    + CompaniesRepository._DUE_FROM
  )
  count = 0
  with closing(sqlite3.connect(str(settings.db_path))) as conn:
    conn.row_factory = sqlite3.Row
    for raw in conn.execute(sql).fetchall():
      if quick and int(raw['priority'] or 0) < 5:
        continue
      if not source_enabled(sources, raw['ats_provider']):
        continue
      count += 1
  return count


@safe_node
async def build_plan(
  state: Dict[str, Any], config: RunnableConfig
) -> Dict[str, Any]:
  '''Select fetch targets from active companies, oldest-checked first.

  Targets come from CompaniesRepository.list_due (last_checked_at
  ASC, never-checked first) so the 50-wide fan-out in
  route_targets rotates through the backlog across runs instead
  of re-fetching the same head. The returned ``deferred_targets``
  stat counts filtered-eligible targets beyond that fan-out cap
  (after priority/source filters, before truncation).

  Args:
    state: Current state.
    config: Runnable config.

  Returns:
    State update containing the RunPlan and deferred counter.
  '''
  from job_hunter.graph.discovery_graph import MAX_FANOUT

  settings = _settings_from_config(config)
  companies = CompaniesRepository(settings.db_path)
  plan = state.get('plan')
  quick = bool(plan['quick_poll']) if isinstance(plan, dict) else False
  due = companies.list_due()
  if quick:
    rows = [
      row
      for row in due
      if row.get('priority', 0) >= 5
      and row.get('ats_provider')
      and row.get('board_ref')
    ]
  else:
    rows = [
      row
      for row in due
      if row.get('ats_provider') and row.get('board_ref')
    ]
  sources = settings.sources or {}
  rows = [
    row for row in rows
    if source_enabled(sources, row.get('ats_provider'))
  ]
  # list_due already excluded cooled scopes in SQL before LIMIT,
  # so rows are fresh here; count and rotate cooled companies
  # separately to preserve the cooled_skips stat.
  cooled_skips = 0
  for row in companies.list_cooled_due():
    cooled_skips += 1
    try:
      companies.mark_checked(int(row['id']))
    except Exception as exc:  # noqa: BLE001 - rotation is best-effort
      logger.warning(
        'mark_checked failed for cooled company %s: %s', row['id'], exc,
      )
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
  # Deferred counts filtered-eligible due rows beyond the
  # fan-out cap, before truncation. list_due caps at 500
  # and count_due ignores the Python-side priority/source
  # filters, so count the filtered due set directly.
  eligible_count = _filtered_due_count(settings, sources, quick)
  deferred = max(0, eligible_count - MAX_FANOUT)
  plan = RunPlan(run_id=int(state['run_id']), targets=targets, quick_poll=quick)
  return {
    'plan': plan.model_dump(),
    'stats': {
      'deferred_targets': deferred, 'cooled_skips': cooled_skips,
    },
  }


@safe_node
async def fetch_pair(
  target_data: Dict[str, Any], config: RunnableConfig
) -> Dict[str, Any]:
  '''Fetch postings for one (source, board_ref) pair.

  The shared process-wide HttpClient (F4) is used so per-host throttle
  and robots caches survive across fetch_pair calls; it is never closed
  here, unlike the previous per-call client. Per-source RPM from
  settings.sources is not threaded through: adapters pass explicit
  rpm= to HttpClient.get_json/get_text, so honoring it would require
  adapter changes (out of scope). Source enablement is enforced
  upstream in build_plan.

  Args:
    target_data: Serialized CompanyTarget.
    config: Runnable config.

  Returns:
    Raw jobs, stats, and any error captured.
  '''
  settings = _settings_from_config(config)
  crawl = CrawlStateRepository(settings.db_path)
  scope = f'company:{target_data["company_id"]}'
  stats: Dict[str, int] = {}
  errors: List[Any] = []
  raw_jobs: List[RawJobRecord] = []
  if crawl.is_cooled_down(scope):
    stats['cooled_skips'] = 1
    company_id = target_data.get('company_id')
    if company_id is not None:
      try:
        CompaniesRepository(settings.db_path).mark_checked(int(company_id))
      except Exception as exc:  # noqa: BLE001 - rotation is best-effort
        logger.warning(
          'mark_checked failed for cooled company %s: %s', company_id, exc,
        )
    return {'raw_jobs': raw_jobs, 'stats': stats, 'errors': errors}
  http = get_shared_http_client()
  try:
    adapter = build_adapter(target_data['source_key'], http)
    target = CompanyTarget(**target_data)
    records = await adapter.fetch(target)
    for record in records:
      if not record.company_name:
        record.company_name = target.name
    since = crawl.get_cursor(f'{scope}:posted')
    if since:
      since_str = str(since)
      records = [
        record
        for record in records
        if not record.posted_at or (record.posted_at >= since_str)
      ]
    watermarks = [
      str(record.posted_at) for record in records if record.posted_at
    ]
    raw_jobs = records
    stats[f'fetched:{target_data["source_key"]}'] = len(records)
    crawl.set_success(scope, cursor=max(watermarks) if watermarks else None)
    if watermarks:
      crawl.set_success(f'{scope}:posted', cursor=max(watermarks))
    # Stamp last_checked_at so the next build_plan rotates past
    # this company; a missing row is a no-op (UPDATE affects 0 rows).
    company_id = target_data.get('company_id')
    if company_id is not None:
      try:
        CompaniesRepository(settings.db_path).mark_checked(
          int(company_id),
        )
      except Exception as exc:  # noqa: BLE001 - rotation is best-effort
        logger.warning(
          'mark_checked failed for company %s: %s', company_id, exc,
        )
  except Exception as exc:
    failures = crawl.set_failure(scope)
    msg = (
      f'{target_data['source_key']}/'
      f'{target_data['board_ref']}: {exc}'
    )
    errors.append(
      NodeError(
        node='fetch_pair',
        message=msg[:500],
        recoverable=True,
      )
    )
    stats['fetch_failures'] = 1
    logger.warning('fetch failed (%s consecutive): %s', failures, exc)
  return {'raw_jobs': raw_jobs, 'stats': stats, 'errors': errors}


@safe_node
async def normalize_dedupe(
  state: Dict[str, Any], config: RunnableConfig
) -> Dict[str, Any]:
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
  from job_hunter.services.persist import persist_record

  _normalize_raw_jobs(state)
  inserted: List[NormalizedJob] = []
  dupes = 0
  for record in state.get('raw_jobs') or []:
    text_desc = record.description_text or html_to_text(record.description_html)
    digest = content_hash(record, text_desc)
    city = detect_city(record.location_text, text_desc)
    mode = infer_work_mode(
      record.work_mode_hint,
      record.location_text,
      f'{record.title} {text_desc[:300]}',
    )
    sal_min, sal_max = parse_salary_lpa(record.salary_raw, text_desc)
    company_id = (
      companies_repo.resolve_alias(record.company_name)
      if record.company_name
      else None
    )
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
    try:
      outcome = await persist_record(
        jobs_repo=jobs_repo,
        record=record,
        company_repo=companies_repo,
      )
    except Exception as exc:
      logger.warning('insert failed for %s: %s', normalized.canonical_url, exc)
      continue
    if outcome == 'duplicate':
      dupes += 1
      continue
    existing = jobs_repo.find_same_url(normalized.canonical_url)
    existing = existing or {}
    new_id = int(existing['id']) if existing.get('id') else None
    update = {'is_new': True, 'job_id': new_id}
    inserted.append(normalized.model_copy(update=update))
  stats = {'new_jobs': len(inserted), 'duplicates': dupes}
  return {
    'inserted': inserted,
    'stats': stats,
  }


@safe_node
async def enrich_jds(
  state: Dict[str, Any], config: RunnableConfig
) -> Dict[str, Any]:
  '''Extract structured fields for newly inserted postings.

  Enriched values are stored as plain dicts keyed by content_hash,
  consistently across graph state and checkpoint serialization.

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

  _normalize_inserted(state)
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

  db_settings = SettingsRepository(settings.db_path)
  disc_overrides = db_settings.get('discovery', {}) or {}
  max_llm_calls = int(
    disc_overrides.get(
      'max_llm_calls_per_run',
      int(settings.discovery.get('max_llm_calls_per_run', 300)),
    )
  )
  raw_budget = state.get('budget')
  if isinstance(raw_budget, TokenBudget):
    budget = raw_budget.model_copy()
  elif isinstance(raw_budget, dict):
    budget = TokenBudget(**raw_budget)
  else:
    budget = TokenBudget(
      max_calls=int(
        settings.discovery.get('max_llm_calls_per_run', 300),
      )
    )
  budget.max_calls = max_llm_calls
  floor_repo = SettingsRepository(settings.db_path)
  floor = float(floor_repo.get(
    'salary_hard_floor_lpa', DEFAULT_SALARY_FLOOR_LPA,
  ))
  enriched_map: Dict[str, Any] = dict(state.get('enriched') or {})
  stats: Dict[str, int] = {}

  cap = int(
    disc_overrides.get(
      'enrich_per_run_cap',
      int(settings.discovery.get('enrich_per_run_cap', 40)),
    )
  )
  attempted = 0
  targets = [
    {
      'job_id': int(j.job_id),
      'content_hash': j.content_hash,
      'title': j.raw.title,
    }
    for j in state.get('inserted') or []
    if j.job_id
  ]
  shortfall = cap - len(targets)
  if shortfall > 0:
    for back in jobs_repo.jobs_needing_enrichment(limit=shortfall):
      targets.append(
        {
          'job_id': int(back['job_id']),
          'content_hash': back['content_hash'],
          'title': back['title'],
        }
      )
  inserted_count = len(state.get('inserted') or [])
  stats['enrich_backlog'] = max(0, len(targets) - inserted_count)

  for job in targets:
    if attempted >= cap:
      stats['enrich_capped'] = stats.get('enrich_capped', 0) + 1
      continue
    attempted += 1
    if not budget.allow():
      stats['enrich_skipped_budget'] = stats.get('enrich_skipped_budget', 0) + 1
      continue
    row = jobs_repo.get(int(job['job_id']))
    description = (row or {}).get('description_text') or ''
    if len(description) < 200:
      stats['enrich_skipped_short'] = stats.get('enrich_skipped_short', 0) + 1
      continue
    budget.spend()
    try:
      extraction = await analyst.extract(
        title=job['title'],
        description_text=description,
        session_id=f'{run_id}:jd_analyst',
      )
    except Exception as exc:
      logger.warning('extraction failed job %s: %s', job['job_id'], exc)
      stats['enrich_failures'] = stats.get('enrich_failures', 0) + 1
      continue

    sal_min = (
      extraction.salary_min_lpa
      if extraction.salary_min_lpa is not None
      else row.get('salary_min_lpa')
    )
    sal_max = (
      extraction.salary_max_lpa
      if extraction.salary_max_lpa is not None
      else row.get('salary_max_lpa')
    )
    mode = (
      extraction.work_mode
      if extraction.work_mode != WorkMode.UNKNOWN
      else (row.get('work_mode') or WorkMode.UNKNOWN)
    )
    jobs_repo.update_fields(
      int(job['job_id']),
      {
        'salary_min_lpa': sal_min,
        'salary_max_lpa': sal_max,
        'experience_min_yrs': extraction.experience_min_years,
        'experience_max_yrs': extraction.experience_max_years,
        'work_mode': mode,
        'employment_type': extraction.employment_type,
      },
    )
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
          skill_rows.append(
            {
              'skill_id': skill_id,
              'kind': kind,
              'confidence': extraction.confidence,
            }
          )
    jobs_repo.attach_skill_rows(int(job['job_id']), skill_rows)

    passes, reason = apply_salary_floor(extraction, floor)
    if not passes:
      stats['below_salary_floor'] = stats.get('below_salary_floor', 0) + 1
      logger.info('job %s filtered by floor: %s', job['job_id'], reason)

    enriched_map[job['content_hash']] = {
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


@safe_node
async def compute_embeddings(
  state: Dict[str, Any], config: RunnableConfig
) -> Dict[str, Any]:
  '''Embed newly inserted postings when a provider is available.

  Args:
    state: Current state.
    config: Runnable config.

  Returns:
    Counters for embedded and skipped postings.
  '''
  from job_hunter.services.embedder import get_embedder

  _normalize_inserted(state)
  settings = _settings_from_config(config)
  jobs_repo = JobsRepository(settings.db_path)
  embedder = get_embedder(settings)
  stats: Dict[str, int] = {'embeddings': 0, 'embedding_skipped': 0}
  if embedder.model_id == _null_provider_model_id():
    stats['embedding_skipped'] = len(state.get('inserted') or [])
    return {'stats': stats}
  pending = [job for job in state.get('inserted') or [] if job.job_id]
  if not pending:
    return {'stats': stats}
  texts = []
  for job in pending:
    row = jobs_repo.get(int(job.job_id)) or {}
    desc = (row.get('description_text') or '')[:1800]
    texts.append(f'{job.raw.title}\n{desc}')
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


def _null_provider_model_id() -> str:
  '''Return the fallback provider marker id without importing heavy deps.

  Returns:
    Marker string.
  '''
  from job_hunter.services.embedder import NullProvider

  return NullProvider.model_id


def _candidate_skill_ids(
  settings: AppSettings, candidate: CandidateProfile
) -> List[int]:
  '''Resolve the candidate's stored skill ids.

  Args:
    settings: Application settings.
    candidate: Profile.

  Returns:
    Skill id list.
  '''
  from job_hunter.services.skills_taxonomy import SkillsTaxonomy

  taxonomy = SkillsTaxonomy(settings.db_path)
  return taxonomy.resolve_many(candidate.skills)


@safe_node
async def score_rank_persist(
  state: Dict[str, Any], config: RunnableConfig
) -> Dict[str, Any]:
  '''Gate, score, rank, and persist recommendations for inserted jobs.

  Args:
    state: Current state.
    config: Runnable config.

  Returns:
    Scored jobs snapshot and counters.
  '''
  from job_hunter.core.models import ScoredJob
  from job_hunter.db.repositories.recommendations import (
    RecommendationsRepository,
  )
  from job_hunter.db.repositories.settings import (
    SettingsRepository as SettingsRepo,
  )
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

  _normalize_inserted(state)
  settings = _settings_from_config(config)
  jobs_repo = JobsRepository(settings.db_path)
  recs_repo = RecommendationsRepository(settings.db_path)
  weights_raw = SettingsRepo(settings.db_path).get('scoring_weights', {})
  weights = normalize_weights(weights_raw)
  candidate = _normalize_candidate(state['candidate'])
  cand_skill_ids = _candidate_skill_ids(settings, candidate)

  scoring_targets = list(state.get('inserted') or [])
  if not scoring_targets:
    for back in jobs_repo.jobs_missing_recommendations(limit=100):
      scoring_targets.append(
        NormalizedJob(
          job_id=int(back['job_id']),
          raw=RawJobRecord(
            source_key='backlog',
            url='',
            title=back['title'],
            company_name=back.get('company_raw_name') or '',
          ),
          canonical_url='',
          content_hash=back['content_hash'],
          company_id=back.get('company_id'),
        )
      )
    if scoring_targets:
      logger.info('scoring %s backlog jobs', len(scoring_targets))

  # Guard against phantom jobs whose job_id is not a real int
  # (e.g. a deserialized or model-constructed 'x'): int() on such
  # a value would crash the whole scoring pass. Skip them with
  # a warning instead so the remaining jobs still score.
  guarded_targets: List[NormalizedJob] = []
  for job in scoring_targets:
    raw_job_id = getattr(job, 'job_id', None)
    if isinstance(raw_job_id, bool) or not isinstance(raw_job_id, int):
      logger.warning('skipping job with non-int job_id %r', raw_job_id)
      continue
    guarded_targets.append(job)
  scoring_targets = guarded_targets

  target_ids = [
    int(job.job_id) for job in scoring_targets if job.job_id
  ]
  embeddings = _load_target_embeddings(
    jobs_repo,
    str(settings.embeddings.get('model', DEFAULT_EMBED_MODEL)),
    target_ids,
  )
  skills_text = ' '.join(candidate.skills)
  roles_text = ' '.join(candidate.target_roles)
  candidate_doc = f'{candidate.summary} {skills_text} {roles_text}'
  candidate_vector = None
  if embeddings:
    from job_hunter.services.embedder import get_embedder

    embedder = get_embedder(settings)
    vectors = embedder.embed([candidate_doc[:2000]])
    candidate_vector = vectors[0] if vectors else None

  skill_map: Dict[int, List[Dict[str, Any]]] = {}
  for skill_row in _job_skill_rows(jobs_repo, target_ids):
    skill_map.setdefault(int(skill_row['job_id']), []).append(skill_row)

  scored: List[ScoredJob] = []
  for job in scoring_targets:
    raw_job_id = getattr(job, 'job_id', None)
    if isinstance(raw_job_id, bool) or not isinstance(raw_job_id, int):
      logger.warning('skipping job with non-int job_id %r', raw_job_id)
      continue
    row = jobs_repo.get_with_company(int(job.job_id)) or {}
    enriched = (state.get('enriched') or {}).get(job.content_hash)
    failures = gate_failures(candidate, row, enriched)
    job_skills = skill_map.get(int(job.job_id), [])
    must_cov, nice_cov = skill_coverage(cand_skill_ids, job_skills)
    description = (row.get('description_text') or '')[:2500]
    vector = embeddings.get(int(job.job_id))
    if candidate_vector is not None and vector is not None:
      norm = (cosine(candidate_vector, vector) + 1.0) / 2.0
      semantic = max(0.0, min(1.0, norm))
    else:
      semantic = semantic_fallback(
        candidate_doc, f'{row.get("title")} {description}'
      )
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
      'salary_fit': salary_fit(
        row.get('salary_min_lpa'),
        row.get('salary_max_lpa'),
        candidate.salary_floor_lpa,
      ),
      'recency': recency_score(row.get('posted_at')),
      'company_fit': company_fit(
        row.get('vertical'),
        candidate.target_verticals,
        int(row.get('company_priority') or 3),
      ),
    }
    total, breakdown = aggregate(components, weights)
    scored.append(
      ScoredJob(
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
      )
    )

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


def _job_skill_rows(
  jobs_repo, job_ids: Any,
) -> List[Dict[str, Any]]:
  '''Fetch skill link rows via the jobs repository (batched).

  Args:
    jobs_repo: Jobs repository.
    job_ids: Single job id or list of job ids.

  Returns:
    Rows with skill_id/kind.
  '''
  if isinstance(job_ids, int):
    ids = [job_ids]
  else:
    ids = [int(job_id) for job_id in job_ids]
  return jobs_repo.job_skill_rows(ids)


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
    normalized.append(
      {
        'job_id': row['job_id'],
        'total_score': row['total_score'],
        'gate_pass': True,
        'gate_failures': [],
        'breakdown': row['breakdown'],
        'rationale': row['rationale'],
        'rank': position,
      }
    )
  return recs_repo.upsert_many(run_id, 1, normalized)


@safe_node
async def summarize_run(
  state: Dict[str, Any], config: RunnableConfig
) -> Dict[str, Any]:
  '''Append a Watchdog quality summary to the run's event stream.

  Args:
    state: Current state.
    config: Runnable config.

  Returns:
    Watchdog issue counter.
  '''
  from job_hunter.services.watchdog import build_report, summarize_issues

  settings = _settings_from_config(config)
  report = build_report(settings)
  issues = summarize_issues(report)
  for issue in issues:
    logger.warning('watchdog: %s', issue)
  return {'stats': {'watchdog_issues': len(issues)}}
