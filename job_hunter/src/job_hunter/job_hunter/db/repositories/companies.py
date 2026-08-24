#!/usr/bin/env python
# -- coding: utf-8 --

'''Company persistence and lookup helpers.'''


from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz
from job_hunter.core.db import connect, session


def normalize_name(name: str) -> str:
  '''Lowercase, strip suffixes, and collapse whitespace.

  Args:
    name: Raw company name.

  Returns:
    Normalized name key.
  '''
  lowered = ' '.join(name.lower().split())
  for suffix in (
    ' private limited', ' pvt ltd', ' pvt. ltd.', ' ltd', ' limited',
    ' inc', ' llc', ' technologies', ' technology',
  ):
    if lowered.endswith(suffix):
      lowered = lowered[: -len(suffix)]
  return lowered.strip(' -')


class CompaniesRepository:
  '''CRUD and matching helpers for the companies table.'''

  def __init__(self, db_path: str | Path) -> None:
    '''Initialize the repository.

    Args:
      db_path: Application database path.
    '''
    self._db_path = str(db_path)

  def upsert(
    self,
    name: str,
    domain: str = '',
    vertical: Optional[str] = None,
    ats_provider: Optional[str] = None,
    board_ref: Optional[str] = None,
    careers_url: Optional[str] = None,
    priority: int = 3,
    status: str = 'active',
    discovered_via: str = 'seed',
    confidence: Optional[float] = None,
  ) -> int:
    '''Insert or update a company keyed by normalized name or domain.

    Args:
      name: Display name.
      domain: Registrable domain when known.
      vertical: Vertical label.
      ats_provider: Detected ATS provider key.
      board_ref: Provider-specific token/org/account.
      careers_url: Careers page URL.
      priority: 1..5 (5 pinned).
      status: Lifecycle status.
      discovered_via: Origin description.
      confidence: Vertical classification confidence.

    Returns:
      Company id.
    '''
    normalized = normalize_name(name)
    with session(self._db_path) as conn:
      row = conn.execute(
        'SELECT id FROM companies WHERE normalized_name = ? '
        'OR (? != "" AND domain = ?)',
        (normalized, domain, domain),
      ).fetchone()
      if row is None:
        cur = conn.execute(
          'INSERT INTO companies (name, normalized_name, domain, vertical, '
          'vertical_confidence, ats_provider, board_ref, careers_url, '
          "priority, status, discovered_via, last_checked_at) "
          'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime(\'now\'))',
          (
            name, normalized, domain, vertical, confidence, ats_provider,
            board_ref, careers_url, priority, status, discovered_via,
          ),
        )
        return int(cur.lastrowid)
      company_id = int(row['id'])
      conn.execute(
        'UPDATE companies SET '
        "name = CASE WHEN ? != '' THEN ? ELSE name END, "
        "domain = CASE WHEN ? != '' THEN ? ELSE domain END, "
        'vertical = COALESCE(?, vertical), '
        'vertical_confidence = COALESCE(?, vertical_confidence), '
        'ats_provider = COALESCE(?, ats_provider), '
        'board_ref = COALESCE(?, board_ref), '
        "careers_url = CASE WHEN ? != '' THEN ? ELSE careers_url END, "
        "status = CASE WHEN ? != 'active' THEN ? ELSE status END, "
        "last_checked_at = datetime('now') "
        'WHERE id = ?',
        (
          domain, domain, domain, domain, vertical, confidence, ats_provider,
          board_ref, careers_url, careers_url, status, status, company_id,
        ),
      )
      return company_id

  def find_by_name(self, name: str, threshold: float = 92.0) -> Optional[Dict[str, Any]]:
    '''Find a company by fuzzy-normalized name.

    Args:
      name: Raw name to match.
      threshold: Minimum token-sort-ratio score.

    Returns:
      Row mapping or None.
    '''
    rows = connect(self._db_path, readonly=True).execute(
      'SELECT id, name, normalized_name FROM companies',
    ).fetchall()
    target = normalize_name(name)
    best_id: Optional[int] = None
    best_score = 0.0
    for row in rows:
      score = float(fuzz.token_sort_ratio(target, row['normalized_name']))
      if score > best_score:
        best_score = score
        best_id = int(row['id'])
    if best_id is not None and best_score >= threshold:
      return self.get(best_id)
    return None

  def get(self, company_id: int) -> Optional[Dict[str, Any]]:
    '''Fetch one company.

    Args:
      company_id: Id.

    Returns:
      Row mapping or None.
    '''
    row = connect(self._db_path, readonly=True).execute(
      'SELECT * FROM companies WHERE id = ?', (company_id,),
    ).fetchone()
    return dict(row) if row else None

  def list_companies(
    self,
    status: Optional[str] = None,
    vertical: Optional[str] = None,
    limit: int = 500,
  ) -> List[Dict[str, Any]]:
    '''List companies with optional filters.

    Args:
      status: Status filter.
      vertical: Vertical filter.
      limit: Page size.

    Returns:
      Rows as mappings.
    '''
    sql = 'SELECT * FROM companies WHERE 1=1'
    params: List[Any] = []
    if status:
      sql += ' AND status = ?'
      params.append(status)
    if vertical:
      sql += ' AND vertical = ?'
      params.append(vertical)
    sql += ' ORDER BY priority DESC, name LIMIT ?'
    params.append(limit)
    rows = connect(self._db_path, readonly=True).execute(sql, params).fetchall()
    return [dict(row) for row in rows]

  def due_for_refresh(self, days: int, chunk: int) -> List[Dict[str, Any]]:
    '''Return active companies not verified recently.

    Args:
      days: Staleness threshold.
      chunk: Max rows.

    Returns:
      Rows with an ATS provider configured.
    '''
    rows = connect(self._db_path, readonly=True).execute(
      "SELECT * FROM companies WHERE status IN ('active','needs_review') "
      'AND ats_provider IS NOT NULL AND (last_checked_at IS NULL OR '
      "last_checked_at < datetime('now', ?)) ORDER BY last_checked_at LIMIT ?",
      (f'-{days} days', chunk),
    ).fetchall()
    return [dict(row) for row in rows]

  def set_status(self, company_id: int, status: str, notes: Optional[str] = None) -> None:
    '''Update lifecycle status.

    Args:
      company_id: Id.
      status: New status value.
      notes: Optional note.
    '''
    with session(self._db_path) as conn:
      conn.execute(
        'UPDATE companies SET status = ?, notes = COALESCE(?, notes) WHERE id = ?',
        (status, notes, company_id),
      )

  def patch(self, company_id: int, fields: Dict[str, Any]) -> None:
    '''Update allow-listed columns.

    Args:
      company_id: Id.
      fields: Column to value mapping.
    '''
    allowed = {
      'priority', 'vertical', 'sub_vertical', 'careers_url', 'notes',
      'ats_provider', 'board_ref', 'domain', 'name',
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
      return
    marks = ', '.join(f'{key} = ?' for key in updates)
    with session(self._db_path) as conn:
      conn.execute(
        f'UPDATE companies SET {marks} WHERE id = ?',
        (*updates.values(), company_id),
      )

  def add_alias(self, alias: str, company_id: int) -> None:
    '''Register an alias pointing at a company.

    Args:
      alias: Alias text.
      company_id: Target company id.
    '''
    with session(self._db_path) as conn:
      conn.execute(
        'INSERT OR IGNORE INTO company_aliases (alias, company_id) VALUES (?, ?)',
        (normalize_name(alias), company_id),
      )

  def resolve_alias(self, name: str) -> Optional[int]:
    '''Resolve a raw name via aliases then fuzzy match.

    Args:
      name: Raw company name.

    Returns:
      Company id or None.
    '''
    row = connect(self._db_path, readonly=True).execute(
      'SELECT company_id FROM company_aliases WHERE alias = ?',
      (normalize_name(name),),
    ).fetchone()
    if row:
      return int(row['company_id'])
    found = self.find_by_name(name)
    return int(found['id']) if found else None


def stats_summary(row: Dict[str, Any]) -> Dict[str, Any]:
  '''Prepare a JSON-safe company payload for API responses.

  Args:
    row: Company row.

  Returns:
    Cleaned mapping.
  '''
  clean = dict(row)
  return {k: v for k, v in clean.items() if k != 'config_json'}
