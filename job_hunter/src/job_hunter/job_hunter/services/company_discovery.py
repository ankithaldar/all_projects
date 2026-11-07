#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Company Scout: seed ingestion, posting harvest, and ATS verification.'''


from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from job_hunter.adapters.career_page import CareerPageDetector
from job_hunter.adapters.http_client import HttpClient
from job_hunter.core.config import AppSettings
from job_hunter.db.repositories.companies import CompaniesRepository

logger = logging.getLogger(__name__)


def _clamp_seed_priority(priority: Any) -> int:
  '''Clamp a seed priority into the supported 1..5 range.

  Mirrors ``_clamp_priority`` in ``api/routes/companies``: rejects
  non-finite values (inf/nan) and non-numeric input.

  Args:
    priority: Raw priority value from a seed entry.

  Returns:
    Priority clamped to 1..5.

  Raises:
    ValueError: On non-finite or non-numeric priority values.
  '''
  if isinstance(priority, bool):
    raise ValueError(f'invalid priority: {priority!r}')
  if isinstance(priority, int):
    return max(1, min(priority, 5))
  try:
    value = float(priority)
  except OverflowError as exc:
    raise ValueError(f'invalid priority: {priority!r}') from exc
  if not math.isfinite(value):
    raise ValueError(f'invalid priority: {priority!r}')
  return max(1, min(int(value), 5))

async def ingest_seeds(
  settings: AppSettings, seeds_dir: Path | None = None
) -> int:
  '''Upsert every company listed in seeds/*.yaml files.

  Args:
    settings: Application settings.
    seeds_dir: Optional directory override.

  Returns:
    Number of seed rows processed.
  '''
  repos = CompaniesRepository(settings.db_path)
  count = 0
  base = Path(seeds_dir) if seeds_dir else Path(settings.seeds_dir)
  for path in sorted(base.glob('companies_*.yaml')):
    payload = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    for entry in payload.get('companies') or []:
      if not isinstance(entry, dict):
        logger.warning(
          'skipping non-dict seed entry in %s: %r', path.name, entry,
        )
        continue
      raw_priority = entry.get('priority', 3)
      if raw_priority is None:
        raw_priority = 3
      try:
        priority = _clamp_seed_priority(raw_priority)
      except (TypeError, ValueError, OverflowError):
        logger.warning(
          'skipping seed entry with bad priority in %s: %r',
          path.name, entry,
        )
        continue
      repos.upsert(
        name=str(entry.get('name') or '').strip(),
        domain=str(entry.get('domain') or '').strip(),
        vertical=entry.get('vertical_hint'),
        priority=priority,
        discovered_via=f'seed:{path.name}',
      )
      count += 1
  return count


async def verify_pending(
  settings: AppSettings, chunk: int = 50
) -> Dict[str, int]:
  '''Fingerprint careers pages for companies lacking ATS details.

  Args:
    settings: Application settings.
    chunk: Max companies per pass.

  Returns:
    Counters {verified, failed}.
  '''
  repos = CompaniesRepository(settings.db_path)
  candidates = [
    row for row in repos.list_companies(limit=chunk * 3)
    if row.get('domain') and not row.get('ats_provider')
  ][:chunk]
  http = HttpClient()
  detector = CareerPageDetector(http)
  verified = failed = 0
  try:
    for row in candidates:
      result: Optional[Tuple[str, str, str]] = None
      try:
        result = await detector.detect(str(row['domain']))
      except Exception:
        result = None
      if result is None:
        repos.patch(int(row['id']), {'notes': 'ats detection failed'})
        repos.set_status(int(row['id']), 'needs_review')
        failed += 1
        continue
      provider, ref, careers_url = result
      repos.patch(int(row['id']), {
        'ats_provider': provider,
        'board_ref': ref,
        'careers_url': careers_url,
        'notes': '',
      })
      verified += 1
  finally:
    await http.close()
  return {'verified': verified, 'failed': failed}


async def run_seed_ingestion(config_path: str | Path, seeds_dir: Path) -> int:
  '''CLI helper: bootstrap then ingest seeds from a directory.

  Args:
    config_path: app.yaml path.
    seeds_dir: Directory containing seeds.

  Returns:
    Rows ingested.
  '''
  from job_hunter.core.bootstrap import bootstrap
  bootstrap(config_path, seeds_dir=seeds_dir)
  settings = AppSettings(config_path)
  return await ingest_seeds(settings, seeds_dir=seeds_dir)
