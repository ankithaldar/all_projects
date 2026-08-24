#!/usr/bin/env python
# -- coding: utf-8 --

'''Recommendation persistence and review actions.'''


from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from job_hunter.core.db import connect, session


class RecommendationsRepository:
  '''Store ranked matches and their review lifecycle.'''

  def __init__(self, db_path: str | Path) -> None:
    '''Initialize the repository.

    Args:
      db_path: Application database path.
    '''
    self._db_path = str(db_path)

  def upsert_many(
    self,
    run_id: int,
    candidate_id: int,
    scored_rows: List[Dict[str, Any]],
  ) -> int:
    '''Persist scored jobs as recommendations keyed by job id.

    Args:
      run_id: Producing run.
      candidate_id: Candidate id.
      scored_rows: ScoredJob-shaped mappings with job_id, total_score, etc.

    Returns:
      Number of rows written.
    '''
    count = 0
    with session(self._db_path) as conn:
      for position, row in enumerate(scored_rows, start=1):
        cur = conn.execute(
          'INSERT INTO recommendations (job_id, candidate_id, run_id, '
          'total_score, rank, gate_pass, gate_failures, score_breakdown_json, '
          'rationale, status, rank, reviewed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '
          "'new', ?, NULL) "
          'ON CONFLICT(job_id, candidate_id) DO UPDATE SET '
          'total_score = excluded.total_score, rank = excluded.rank, '
          'gate_pass = excluded.gate_pass, '
          'gate_failures = excluded.gate_failures, '
          'score_breakdown_json = excluded.score_breakdown_json, '
          'rationale = excluded.rationale, run_id = excluded.run_id',
          (
            row['job_id'], candidate_id, run_id, row['total_score'], position,
            int(row['gate_pass']),
            json.dumps(row.get('gate_failures', []), ensure_ascii=False),
            json.dumps(row.get('breakdown', {}), ensure_ascii=False),
            row.get('rationale', ''),
            row.get('rank'),
          ),
        )
        count += max(cur.rowcount, 0)
    return count

  def list_for_candidate(
    self,
    candidate_id: int = 1,
    min_score: float = 0.0,
    status: str = '',
    vertical: str = '',
    page: int = 1,
    per_page: int = 30,
    gate_only: bool = True,
  ) -> List[Dict[str, Any]]:
    '''List recommendations joined with job/company data.

    Args:
      candidate_id: Candidate id.
      min_score: Minimum total score.
      status: Status filter (blank = all).
      vertical: Vertical filter via company join (blank = all).
      page: 1-based page.
      per_page: Page size.
      gate_only: Only gate-passing rows.

    Returns:
      Enriched row mappings.
    '''
    sql = (
      'SELECT r.id AS recommendation_id, r.total_score, r.rank, r.status, '
      'r.gate_failures, r.score_breakdown_json, r.rationale, r.reviewed_at, '
      'r.created_at, j.*, c.name AS company_name, c.vertical AS vertical, '
      'c.priority AS company_priority '
      'FROM recommendations r JOIN jobs j ON j.id = r.job_id '
      'LEFT JOIN companies c ON c.id = j.company_id '
      'WHERE r.candidate_id = ?'
    )
    params: List[Any] = [candidate_id]
    if min_score > 0:
      sql += ' AND r.total_score >= ?'
      params.append(min_score)
    if status:
      sql += ' AND r.status = ?'
      params.append(status)
    if gate_only:
      sql += ' AND r.gate_pass = 1'
    if vertical:
      sql += ' AND c.vertical = ?'
      params.append(vertical)
    sql += ' ORDER BY r.total_score DESC LIMIT ? OFFSET ?'
    params.extend([per_page, (max(1, page) - 1) * per_page])
    rows = connect(self._db_path, readonly=True).execute(sql, params).fetchall()
    result = []
    for row in rows:
      item = dict(row)
      item['score_breakdown'] = json.loads(item.pop('score_breakdown_json') or '{}')
      item['gate_failures'] = json.loads(item.pop('gate_failures') or '[]')
      result.append(item)
    return result

  def set_status(
    self,
    recommendation_id: int,
    status: str,
    candidate_id: int = 1,
  ) -> bool:
    '''Transition a recommendation's review status.

    Args:
      recommendation_id: Row id.
      status: saved|dismissed|applied|new.
      candidate_id: Owner.

    Returns:
      True when a row changed.
    '''
    with session(self._db_path) as conn:
      cur = conn.execute(
        'UPDATE recommendations SET status = ?, '
        "reviewed_at = CASE WHEN ? = 'new' THEN NULL ELSE datetime('now') END "
        'WHERE id = ? AND candidate_id = ?',
        (status, status, recommendation_id, candidate_id),
      )
      return cur.rowcount > 0

  def counts(self, candidate_id: int = 1) -> Dict[str, Any]:
    '''Return dashboard counters.

    Args:
      candidate_id: Candidate id.

    Returns:
      Mapping of counter names to values.
    '''
    conn = connect(self._db_path, readonly=True)
    new_today = conn.execute(
      "SELECT COUNT(*) AS n FROM recommendations WHERE candidate_id = ? "
      "AND status = 'new' AND date(created_at) = date('now')",
      (candidate_id,),
    ).fetchone()['n']
    saved = conn.execute(
      "SELECT COUNT(*) AS n FROM recommendations WHERE candidate_id = ? "
      "AND status = 'saved'",
      (candidate_id,),
    ).fetchone()['n']
    active_jobs = conn.execute(
      "SELECT COUNT(*) AS n FROM jobs WHERE status = 'active'",
    ).fetchone()['n']
    companies = conn.execute(
      "SELECT COUNT(*) AS n FROM companies WHERE status IN ('active','needs_review')",
    ).fetchone()['n']
    top_verticals = conn.execute(
      'SELECT c.vertical AS v, COUNT(*) AS n FROM recommendations r '
      'JOIN jobs j ON j.id = r.job_id JOIN companies c ON c.id = j.company_id '
      'WHERE r.candidate_id = ? AND r.gate_pass = 1 AND c.vertical IS NOT NULL '
      'GROUP BY c.vertical ORDER BY n DESC LIMIT 5',
      (candidate_id,),
    ).fetchall()
    conn.close()
    return {
      'new_today': new_today,
      'saved': saved,
      'active_jobs': active_jobs,
      'companies': companies,
      'top_verticals': [
        {'vertical': row['v'], 'count': row['n']} for row in top_verticals
      ],
    }
