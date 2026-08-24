#!/usr/bin/env python
# -- coding: utf-8 --

'''Job persistence, dedupe lookups, and FTS search.'''


from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from job_hunter.core.db import connect, session


def sanitize_fts(query: str) -> str:
  '''Escape user input into a safe AND-of-quoted-terms MATCH expression.

  Args:
    query: Raw user search text.

  Returns:
    Sanitized FTS5 MATCH string.
  '''
  terms = [t.strip() for t in (query or '').split() if t.strip()]
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
    row = connect(self._db_path, readonly=True).execute(
      'SELECT 1 FROM jobs WHERE content_hash = ?', (content_hash,),
    ).fetchone()
    return row is not None

  def find_same_url(self, canonical_url: str) -> Optional[Dict[str, Any]]:
    '''Find an existing job by canonical URL.

    Args:
      canonical_url: Canonical posting URL.

    Returns:
      Row mapping or None.
    '''
    row = connect(self._db_path, readonly=True).execute(
      'SELECT * FROM jobs WHERE canonical_url = ?', (canonical_url,),
    ).fetchone()
    return dict(row) if row else None

  def insert(self, payload: Dict[str, Any]) -> int:
    '''Insert one normalized job.

    Args:
      payload: Column mapping matching the jobs schema.

    Returns:
      New job id.
    '''
    columns = sorted(payload.keys())
    marks = ', '.join('?' * len(columns))
    with session(self._db_path) as conn:
      cur = conn.execute(
        f'INSERT INTO jobs ({", ".join(columns)}) VALUES ({marks})',
        tuple(payload[c] for c in columns),
      )
      return int(cur.lastrowid)

  def update_fields(self, job_id: int, fields: Dict[str, Any]) -> None:
    '''Update allow-listed columns on a job.

    Args:
      job_id: Job id.
      fields: Columns to set.
    '''
    allowed = {
      'description_text', 'salary_min_lpa', 'salary_max_lpa', 'salary_raw',
      'experience_min_yrs', 'experience_max_yrs', 'work_mode',
      'employment_type', 'status', 'quality_score', 'company_id',
      'posted_at', 'city',
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
      return
    marks = ', '.join(f'{key} = ?' for key in updates)
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
    row = connect(self._db_path, readonly=True).execute(
      'SELECT * FROM jobs WHERE id = ?', (job_id,),
    ).fetchone()
    return dict(row) if row else None

  def get_with_company(self, job_id: int) -> Optional[Dict[str, Any]]:
    '''Fetch one job joined with its company fields.

    Args:
      job_id: Job id.

    Returns:
      Row mapping with company_name, vertical, company_priority.
    '''
    row = connect(self._db_path, readonly=True).execute(
      'SELECT j.*, c.name AS company_name, c.vertical AS vertical, '
      'c.priority AS company_priority FROM jobs j '
      'LEFT JOIN companies c ON c.id = j.company_id WHERE j.id = ?',
      (job_id,),
    ).fetchone()
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
      q: Free-text query against the FTS index.
      city: City filter (exact).
      work_mode: Work-mode filter.
      vertical: Vertical via company join.
      status: Job status filter.
      page: 1-based page.
      per_page: Page size.

    Returns:
      Rows enriched with company name/vertical.
    '''
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
    if q:
      base += ' AND j.id IN (SELECT rowid FROM jobs_fts WHERE jobs_fts MATCH ?)'
      params.append(sanitize_fts(q))
    base += ' ORDER BY COALESCE(j.posted_at, j.first_seen_at) DESC LIMIT ? OFFSET ?'
    params.extend([per_page, (max(1, page) - 1) * per_page])
    rows = connect(self._db_path, readonly=True).execute(base, params).fetchall()
    return [dict(row) for row in rows]

  def unseen_hashes(self, hashes: List[str]) -> List[str]:
    '''Filter a hash list down to those not yet stored.

    Args:
      hashes: Candidate hashes.

    Returns:
      Missing hashes only.
    '''
    if not hashes:
      return []
    found = set()
    conn = connect(self._db_path, readonly=True)
    for start in range(0, len(hashes), 500):
      chunk = hashes[start:start + 500]
      marks = ','.join('?' * len(chunk))
      rows = conn.execute(
        f'SELECT content_hash FROM jobs WHERE content_hash IN ({marks})',
        chunk,
      ).fetchall()
      found.update(row['content_hash'] for row in rows)
    conn.close()
    return [h for h in hashes if h not in found]

  def jobs_missing_recommendations(self, limit: int = 100) -> List[Dict[str, Any]]:
    '''Active postings with no recommendation row yet.

    Args:
      limit: Max rows.

    Returns:
      Job rows newest first.
    '''
    rows = connect(self._db_path, readonly=True).execute(
      'SELECT j.id AS job_id, j.title, j.content_hash, j.company_id, '
      'j.company_raw_name FROM jobs j LEFT JOIN recommendations r '
      "ON r.job_id = j.id WHERE j.status = 'active' AND r.job_id IS NULL "
      'ORDER BY j.first_seen_at DESC LIMIT ?',
      (limit,),
    ).fetchall()
    return [dict(row) for row in rows]

  def jobs_needing_enrichment(self, limit: int = 50) -> List[Dict[str, Any]]:
    '''Active postings without any extracted skill rows.

    Args:
      limit: Max rows.

    Returns:
      Job id/title/content_hash rows, newest first.
    '''
    rows = connect(self._db_path, readonly=True).execute(
      'SELECT j.id AS job_id, j.title, j.content_hash FROM jobs j '
      'LEFT JOIN job_skills js ON js.job_id = j.id '
      "WHERE j.status = 'active' AND js.job_id IS NULL "
      'ORDER BY j.first_seen_at DESC LIMIT ?',
      (limit,),
    ).fetchall()
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
        'AND job_id IN (SELECT id FROM jobs WHERE status = \'stale\')',
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
          'INSERT OR IGNORE INTO job_skills (job_id, skill_id, kind, confidence) '
          'VALUES (?, ?, ?, ?)',
          (job_id, row['skill_id'], row['kind'], row.get('confidence', 0.8)),
        )

  def save_embedding(self, job_id: int, model: str, vector: List[float]) -> None:
    '''Store or replace a job embedding blob.

    Args:
      job_id: Job id.
      model: Model identifier.
      vector: Float vector.
    '''
    import struct
    packed = struct.pack(f'{len(vector)}f', *vector)
    with session(self._db_path) as conn:
      conn.execute(
        'INSERT INTO job_embeddings (job_id, model, dim, vector) VALUES (?, ?, ?, ?) '
        'ON CONFLICT(job_id, model) DO UPDATE SET vector = excluded.vector, '
        "dim = excluded.dim, created_at = datetime('now')",
        (job_id, model, len(vector), packed),
      )

  def load_embeddings(self, model: str) -> Dict[int, List[float]]:
    '''Load all embeddings for one model keyed by job id.

    Args:
      model: Model identifier.

    Returns:
      Mapping of job id to vector.
    '''
    import struct
    rows = connect(self._db_path, readonly=True).execute(
      'SELECT job_id, dim, vector FROM job_embeddings WHERE model = ?',
      (model,),
    ).fetchall()
    return {
      int(row['job_id']): list(struct.unpack(f'{row["dim"]}f', row['vector']))
      for row in rows
    }

  def raw_json_of(self, job_id: int) -> Optional[str]:
    '''Return stored raw_json for one job.

    Args:
      job_id: Job id.

    Returns:
      Raw JSON string or None.
    '''
    row = connect(self._db_path, readonly=True).execute(
      'SELECT raw_json FROM jobs WHERE id = ?', (job_id,),
    ).fetchone()
    return row['raw_json'] if row else None
