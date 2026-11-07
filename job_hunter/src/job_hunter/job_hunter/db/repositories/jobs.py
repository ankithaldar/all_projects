#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Job persistence, dedupe lookups, and FTS search.'''

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

from job_hunter.core.db import (
  fetch_all,
  fetch_one,
  filter_to_columns,
  quote_identifier,
  session,
)


def sanitize_fts(query: str) -> Optional[str]:
  '''Escape user input into a safe AND-of-quoted-terms MATCH string.

  Args:
    query: Raw user search text.

  Returns:
    Sanitized FTS5 MATCH string, or None when the query has no
    terms (callers must then skip the FTS clause).
  '''
  terms = [t.strip() for t in (query or '').split() if t.strip()]
  if not terms:
    return None
  return ' AND '.join('"' + t.replace('"', '""') + '"' for t in terms)


class JobsRepository:
  '''CRUD for jobs plus dedupe and search helpers.'''

  def __init__(self, db_path: str | Path) -> None:
    '''Initialize the repository.

    Args:
      db_path: Application database path.
    '''
    self._db_path = str(db_path)

  def exists_hash(self, content_hash: str) -> bool:
    '''Return whether a posting hash is already stored.

    Args:
      content_hash: Canonical content hash.

    Returns:
      True when present.
    '''
    with fetch_one(
      self._db_path,
      'SELECT 1 FROM jobs WHERE content_hash = ?',
      (content_hash,),
    ) as row:
      return row is not None

  def find_same_url(self, canonical_url: str) -> Optional[Dict[str, Any]]:
    '''Find an existing job by canonical URL.

    Args:
      canonical_url: Canonical posting URL.

    Returns:
      Row mapping or None.
    '''
    with fetch_one(
      self._db_path,
      'SELECT * FROM jobs WHERE canonical_url = ?',
      (canonical_url,),
    ) as row:
      return dict(row) if row else None

  def insert(self, payload: Dict[str, Any]) -> int:
    '''Insert one normalized job.

    Args:
      payload: Column mapping matching the jobs schema. Unknown
        keys are dropped with a warning, never interpolated.

    Returns:
      New job id.

    Raises:
      ValueError: If no payload key is a real jobs column.
    '''
    with session(self._db_path) as conn:
      kept = filter_to_columns(conn, 'jobs', payload)
      if not kept:
        raise ValueError('no valid columns for jobs insert')
      columns = sorted(kept.keys())
      marks = ', '.join('?' * len(columns))
      quoted = ', '.join(quote_identifier(col) for col in columns)
      cur = conn.execute(
        f'INSERT INTO jobs ({quoted}) VALUES ({marks})',
        tuple(kept[col] for col in columns),
      )
      return int(cur.lastrowid)

  def update_fields(self, job_id: int, fields: Dict[str, Any]) -> None:
    '''Update allow-listed columns on a job.

    Args:
      job_id: Job id.
      fields: Columns to set.
    '''
    allowed = {
      'description_text',
      'salary_min_lpa',
      'salary_max_lpa',
      'salary_raw',
      'experience_min_yrs',
      'experience_max_yrs',
      'work_mode',
      'employment_type',
      'status',
      'quality_score',
      'company_id',
      'posted_at',
      'city',
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
      return
    marks = ', '.join(f'{quote_identifier(key)} = ?' for key in updates)
    with session(self._db_path) as conn:
      conn.execute(
        f"UPDATE jobs SET {marks}, updated_at = datetime('now') WHERE id = ?",
        (*updates.values(), job_id),
      )

  def get(self, job_id: int) -> Optional[Dict[str, Any]]:
    '''Fetch one job.

    Args:
      job_id: Id.

    Returns:
      Row mapping or None.
    '''
    with fetch_one(
      self._db_path,
      'SELECT * FROM jobs WHERE id = ?',
      (job_id,),
    ) as row:
      return dict(row) if row else None

  def get_with_company(self, job_id: int) -> Optional[Dict[str, Any]]:
    '''Fetch one job joined with its company fields.

    Args:
      job_id: Job id.

    Returns:
      Row mapping with company_name, vertical, company_priority.
    '''
    with fetch_one(
      self._db_path,
      'SELECT j.*, c.name AS company_name, c.vertical AS vertical, '
      'c.priority AS company_priority FROM jobs j '
      'LEFT JOIN companies c ON c.id = j.company_id WHERE j.id = ?',
      (job_id,),
    ) as row:
      return dict(row) if row else None

  def list_jobs(
    self,
    q: str = '',
    city: str = '',
    work_mode: str = '',
    vertical: str = '',
    status: str = 'active',
    page: int = 1,
    per_page: int = 25,
  ) -> List[Dict[str, Any]]:
    '''List jobs with optional FTS query and filters.

    Args:
      q: Free-text query against the FTS index. A query that
        sanitizes to nothing (blank/whitespace) skips the FTS
        clause instead of issuing an empty MATCH.
      city: City filter (exact).
      work_mode: Work-mode filter.
      vertical: Vertical via company join.
      status: Job status filter.
      page: 1-based page.
      per_page: Page size.

    Returns:
      Rows enriched with company name/vertical.
    '''
    per_page = max(1, min(per_page, 100))
    base = (
      'SELECT j.*, c.name AS company_name, c.vertical AS vertical '
      'FROM jobs j LEFT JOIN companies c ON c.id = j.company_id '
      'WHERE 1=1'
    )
    params: List[Any] = []
    if status:
      base += ' AND j.status = ?'
      params.append(status)
    if city:
      base += ' AND j.city = ?'
      params.append(city)
    if work_mode:
      base += ' AND j.work_mode = ?'
      params.append(work_mode)
    if vertical:
      base += ' AND c.vertical = ?'
      params.append(vertical)
    fts = sanitize_fts(q) if q else None
    if fts:
      base += ' AND j.id IN (SELECT rowid FROM jobs_fts WHERE jobs_fts MATCH ?)'
      params.append(fts)
    base += (
      ' ORDER BY COALESCE(j.posted_at, j.first_seen_at) DESC LIMIT ? OFFSET ?'
    )
    params.extend([per_page, (max(1, page) - 1) * per_page])
    with fetch_all(self._db_path, base, params) as rows:
      return [dict(row) for row in rows]

  def jobs_missing_recommendations(
    self,
    limit: int = 100,
  ) -> List[Dict[str, Any]]:
    '''Active postings with no recommendation row yet.

    Args:
      limit: Max rows.

    Returns:
      Job rows newest first.
    '''
    limit = max(1, min(limit, 100))
    with fetch_all(
      self._db_path,
      'SELECT j.id AS job_id, j.title, j.content_hash, j.company_id, '
      'j.company_raw_name FROM jobs j LEFT JOIN recommendations r '
      "ON r.job_id = j.id WHERE j.status = 'active' AND r.job_id IS NULL "
      'ORDER BY j.first_seen_at DESC LIMIT ?',
      (limit,),
    ) as rows:
      return [dict(row) for row in rows]

  def jobs_needing_enrichment(
    self,
    limit: int = 50,
  ) -> List[Dict[str, Any]]:
    '''Active postings without any extracted skill rows.

    Args:
      limit: Max rows.

    Returns:
      Job id/title/content_hash rows, newest first.
    '''
    limit = max(1, min(limit, 100))
    with fetch_all(
      self._db_path,
      'SELECT j.id AS job_id, j.title, j.content_hash FROM jobs j '
      'LEFT JOIN job_skills js ON js.job_id = j.id '
      "WHERE j.status = 'active' AND js.job_id IS NULL "
      'ORDER BY j.first_seen_at DESC LIMIT ?',
      (limit,),
    ) as rows:
      return [dict(row) for row in rows]

  def stale_sweep(self, days: int) -> int:
    '''Mark unseen postings stale and expire their recommendations.

    Args:
      days: Age threshold.

    Returns:
      Number of jobs marked stale.
    '''
    with session(self._db_path) as conn:
      cur = conn.execute(
        "UPDATE jobs SET status = 'stale', updated_at = datetime('now') "
        "WHERE status = 'active' AND "
        "COALESCE(posted_at, first_seen_at) < datetime('now', ?)",
        (f'-{days} days',),
      )
      conn.execute(
        "UPDATE recommendations SET status = 'expired' WHERE status = 'new' "
        "AND job_id IN (SELECT id FROM jobs WHERE status = 'stale')",
      )
      return int(cur.rowcount)

  def attach_skill_rows(self, job_id: int, rows: List[Dict[str, Any]]) -> None:
    '''Replace extracted skill links for one job.

    Args:
      job_id: Job id.
      rows: Each item carries skill_id, kind, confidence.
    '''
    with session(self._db_path) as conn:
      conn.execute('DELETE FROM job_skills WHERE job_id = ?', (job_id,))
      for row in rows:
        conn.execute(
          'INSERT OR IGNORE INTO job_skills (job_id, skill_id, kind, '
          'confidence) VALUES (?, ?, ?, ?)',
          (job_id, row['skill_id'], row['kind'], row.get('confidence', 0.8)),
        )

  def job_skill_rows(self, job_ids: List[int]) -> List[Dict[str, Any]]:
    '''Return extracted skill links for many jobs in one query.

    Args:
      job_ids: Job ids to look up.

    Returns:
      Rows with job_id, skill_id, skill_name, kind, confidence.
      Empty when ``job_ids`` is empty.
    '''
    if not job_ids:
      return []
    rows_out: List[Dict[str, Any]] = []
    for start in range(0, len(job_ids), 500):
      chunk = job_ids[start : start + 500]
      marks = ','.join('?' * len(chunk))
      with fetch_all(
        self._db_path,
        'SELECT js.job_id AS job_id, js.skill_id AS skill_id, '
        's.canonical_name AS skill_name, js.kind AS kind, '
        'js.confidence AS confidence FROM job_skills js '
        'JOIN skills s ON s.id = js.skill_id '
        f'WHERE js.job_id IN ({marks}) '
        'ORDER BY js.job_id, s.canonical_name',
        chunk,
      ) as rows:
        rows_out.extend(dict(row) for row in rows)
    return rows_out

  def save_embedding(
    self,
    job_id: int,
    model: str,
    vector: List[float],
  ) -> None:
    '''Store or replace a job embedding blob.

    Args:
      job_id: Job id.
      model: Model identifier.
      vector: Float vector.
    '''
    packed = struct.pack(f'{len(vector)}f', *vector)
    with session(self._db_path) as conn:
      conn.execute(
        'INSERT INTO job_embeddings (job_id, model, dim, vector) '
        'VALUES (?, ?, ?, ?) '
        'ON CONFLICT(job_id, model) DO UPDATE SET vector = excluded.vector, '
        "dim = excluded.dim, created_at = datetime('now')",
        (job_id, model, len(vector), packed),
      )

  def load_embeddings(
    self, model: str, job_ids: list | None = None
  ) -> Dict[int, List[float]]:
    '''Load embeddings for one model keyed by job id.

    Args:
      model: Model identifier.
      job_ids: When given, only load embeddings for these job ids
        (chunked at 500 per query); when None, load all.

    Returns:
      Mapping of job id to vector.
    '''
    if job_ids is None:
      with fetch_all(
        self._db_path,
        'SELECT job_id, dim, vector FROM job_embeddings WHERE model = ?',
        (model,),
      ) as rows:
        return {
          int(row['job_id']): list(
            struct.unpack(str(row['dim']) + 'f', row['vector'])
          )
          for row in rows
        }
    result: Dict[int, List[float]] = {}
    for start in range(0, len(job_ids), 500):
      chunk = list(job_ids[start : start + 500])
      if not chunk:
        continue
      marks = ','.join('?' * len(chunk))
      with fetch_all(
        self._db_path,
        f'SELECT job_id, dim, vector FROM job_embeddings '
        f'WHERE model = ? AND job_id IN ({marks})',
        (model, *chunk),
      ) as rows:
        for row in rows:
          result[int(row['job_id'])] = list(
            struct.unpack(str(row['dim']) + 'f', row['vector'])
          )
    return result
