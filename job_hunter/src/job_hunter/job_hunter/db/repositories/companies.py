#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Company persistence and lookup helpers.'''

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz

from job_hunter.core.db import fetch_all, fetch_one, quote_identifier, session


def normalize_name(name: str) -> str:
  '''Lowercase, strip suffixes, and collapse whitespace.

  Args:
    name: Raw company name.

  Returns:
    Normalized name key.
  '''
  lowered = ' '.join(name.lower().split())
  for suffix in (
    ' private limited',
    ' pvt ltd',
    ' pvt. ltd.',
    ' ltd',
    ' limited',
    ' inc',
    ' llc',
    ' technologies',
    ' technology',
  ):
    if lowered.endswith(suffix):
      lowered = lowered.removesuffix(suffix)
  return lowered.strip(' -')


class CompaniesRepository:
  '''CRUD and matching helpers for the companies table.'''

  # Shared FROM/WHERE base for the due queries: active companies
  # with truthy ATS/board, joined to crawl_state on the
  # 'company:<id>' scope. Cooled scopes are excluded by the
  # _DUE_FROM predicate in SQL before LIMIT so a cooled oldest
  # batch cannot starve fresh companies.
  _DUE_BASE_FROM = (
    'FROM companies LEFT JOIN crawl_state cs '
    "ON cs.scope = 'company:' || companies.id "
    "WHERE companies.status = 'active' "
    "AND NULLIF(companies.ats_provider, '') IS NOT NULL "
    "AND NULLIF(companies.board_ref, '') IS NOT NULL"
  )
  _DUE_FROM = (
    _DUE_BASE_FROM
    + ' AND (cs.cooldown_until IS NULL '
    + "OR cs.cooldown_until <= datetime('now'))"
  )
  _COOLED_FROM = (
    _DUE_BASE_FROM
    + ' AND cs.cooldown_until IS NOT NULL '
    + "AND cs.cooldown_until > datetime('now')"
  )

  def __init__(self, db_path: str | Path) -> None:
    '''Initialize the repository.

    Args:
      db_path: Application database path.
    '''
    self._db_path = str(db_path)

  def _update_sql(self) -> str:
    '''Build the shared ON CONFLICT / domain update clause.

    Returns:
      Update assignments mirroring the historical UPDATE binding
      order (see :meth:`upsert`).
    '''
    return (
      'UPDATE SET '
      "name = CASE WHEN ? != '' THEN ? ELSE name END, "
      'domain = CASE WHEN ? IS NOT NULL THEN ? ELSE domain END, '
      'vertical = COALESCE(?, vertical), '
      'vertical_confidence = COALESCE(?, vertical_confidence), '
      'ats_provider = COALESCE(?, ats_provider), '
      'board_ref = COALESCE(?, board_ref), '
      "careers_url = CASE WHEN ? != '' THEN ? ELSE careers_url END, "
      "status = CASE WHEN ? != 'active' THEN ? ELSE status END"
    )

  def _update_params(
    self,
    domain: str,
    clean_domain: Optional[str],
    vertical: Optional[str],
    confidence: Optional[float],
    ats_provider: Optional[str],
    board_ref: Optional[str],
    careers_url: Optional[str],
    status: str,
  ) -> tuple[Any, ...]:
    '''Assemble the shared update binding values.

    Args:
      domain: Raw domain argument (legacy binding order).
      clean_domain: Stripped domain or None.
      vertical: Vertical label.
      confidence: Vertical classification confidence.
      ats_provider: Detected ATS provider key.
      board_ref: Provider-specific board reference.
      careers_url: Careers page URL.
      status: Lifecycle status.

    Returns:
      Binding values in SQL placeholder order.
    '''
    return (
      domain,
      domain,
      clean_domain,
      clean_domain,
      vertical,
      confidence,
      ats_provider,
      board_ref,
      careers_url,
      careers_url,
      status,
      status,
    )

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
    clean_domain = (domain or '').strip() or None
    careers = careers_url or ''
    params = self._update_params(
      domain,
      clean_domain,
      vertical,
      confidence,
      ats_provider,
      board_ref,
      careers,
      status,
    )
    with session(self._db_path) as conn:
      try:
        conn.execute(
          'INSERT INTO companies (name, normalized_name, domain, vertical, '
          'vertical_confidence, ats_provider, board_ref, careers_url, '
          'priority, status, discovered_via) '
          'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) '
          'ON CONFLICT(normalized_name) DO ' + self._update_sql(),
          (
            name,
            normalized,
            clean_domain,
            vertical,
            confidence,
            ats_provider,
            board_ref,
            careers_url,
            priority,
            status,
            discovered_via,
            *params,
          ),
        )
      except sqlite3.IntegrityError:
        # A domain UNIQUE conflict means the row exists under a
        # different normalized name; fall back to updating it.
        if clean_domain is None:
          raise
        row = conn.execute(
          'SELECT id FROM companies WHERE domain = ?',
          (clean_domain,),
        ).fetchone()
        if row is None:
          raise
        conn.execute(
          'UPDATE companies SET '
          "name = CASE WHEN ? != '' THEN ? ELSE name END, "
          'domain = CASE WHEN ? IS NOT NULL THEN ? ELSE domain END, '
          'vertical = COALESCE(?, vertical), '
          'vertical_confidence = COALESCE(?, vertical_confidence), '
          'ats_provider = COALESCE(?, ats_provider), '
          'board_ref = COALESCE(?, board_ref), '
          "careers_url = CASE WHEN ? != '' THEN ? ELSE careers_url END, "
          "status = CASE WHEN ? != 'active' THEN ? ELSE status END "
          "WHERE id = ?",
          (*params, int(row['id'])),
        )
        return int(row['id'])
      row = conn.execute(
        'SELECT id FROM companies WHERE normalized_name = ?',
        (normalized,),
      ).fetchone()
      return int(row['id'])

  def find_by_name(
    self,
    name: str,
    threshold: float = 92.0,
  ) -> Optional[Dict[str, Any]]:
    '''Find a company by fuzzy-normalized name.

    Args:
      name: Raw name to match.
      threshold: Minimum token-sort-ratio score.

    Returns:
      Row mapping or None.
    '''
    with fetch_all(
      self._db_path,
      'SELECT id, name, normalized_name FROM companies',
    ) as rows:
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
    with fetch_one(
      self._db_path,
      'SELECT * FROM companies WHERE id = ?',
      (company_id,),
    ) as row:
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
    limit = max(1, min(limit, 500))
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
    with fetch_all(self._db_path, sql, params) as rows:
      return [dict(row) for row in rows]

  def list_due(self, limit: int = 500) -> List[Dict[str, Any]]:
    '''List active companies oldest-first for fan-out rotation.

    Ordering is by last_checked_at ASC with NULLs (never checked)
    first, so repeated runs rotate through the whole backlog
    instead of re-fetching the same head every time. Cooled scopes
    and empty-string ATS/board values are excluded in SQL before
    LIMIT (see _DUE_FROM), so fresh companies are never starved
    behind a cooled or unverifiable oldest batch.

    Args:
      limit: Page size.

    Returns:
      Rows as mappings, oldest first.
    '''
    limit = max(1, min(limit, 500))
    with fetch_all(
      self._db_path,
      'SELECT companies.* ' + self._DUE_FROM +
      ' ORDER BY (companies.last_checked_at IS NOT NULL), '
      'companies.last_checked_at ASC, companies.priority DESC, '
      'companies.name LIMIT ?',
      (limit,),
    ) as rows:
      return [dict(row) for row in rows]

  def count_due(self) -> int:
    '''Count due companies sharing the list_due filters.

    Returns:
      Number of active, ATS-bearing, non-cooled companies,
      without the list_due LIMIT truncation.
    '''
    with fetch_one(
      self._db_path,
      'SELECT COUNT(*) AS n ' + self._DUE_FROM,
    ) as row:
      return int(row['n']) if row else 0

  def list_cooled_due(self, limit: int = 500) -> List[Dict[str, Any]]:
    '''List active, ATS-bearing companies inside their cooldown.

    Mirrors the list_due filters except the cooldown predicate,
    so callers can count and rotate cooled companies that
    list_due already excluded in SQL.

    Args:
      limit: Page size.

    Returns:
      Rows as mappings, oldest first.
    '''
    limit = max(1, min(limit, 500))
    with fetch_all(
      self._db_path,
      'SELECT companies.* ' + self._COOLED_FROM +
      ' ORDER BY (companies.last_checked_at IS NOT NULL), '
      'companies.last_checked_at ASC, companies.priority DESC, '
      'companies.name LIMIT ?',
      (limit,),
    ) as rows:
      return [dict(row) for row in rows]

  def mark_checked(self, company_id: int) -> None:
    '''Stamp companies.last_checked_at as a rotation stamp.

    This is a rotation stamp (attempted), not last successful fetch:
    callers stamp after any attempt (success, failure, or cooled
    skip) so the oldest-first rotation advances even when fetching
    does not succeed.

    Args:
      company_id: Id of the company just attempted.
    '''
    with session(self._db_path) as conn:
      conn.execute(
        "UPDATE companies SET last_checked_at = datetime('now') "
        'WHERE id = ?',
        (company_id,),
      )

  def set_status(
    self,
    company_id: int,
    status: str,
    notes: Optional[str] = None,
  ) -> None:
    '''Update lifecycle status.

    Args:
      company_id: Id.
      status: New status value.
      notes: Optional note.
    '''
    with session(self._db_path) as conn:
      conn.execute(
        'UPDATE companies SET status = ?, notes = COALESCE(?, notes) '
        'WHERE id = ?',
        (status, notes, company_id),
      )

  def patch(self, company_id: int, fields: Dict[str, Any]) -> None:
    '''Update allow-listed columns.

    Args:
      company_id: Id.
      fields: Column to value mapping.
    '''
    allowed = {
      'priority',
      'vertical',
      'sub_vertical',
      'careers_url',
      'notes',
      'ats_provider',
      'board_ref',
      'domain',
      'name',
    }
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
      return
    marks = ', '.join(f'{quote_identifier(key)} = ?' for key in updates)
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
        'INSERT OR IGNORE INTO company_aliases (alias, company_id) '
        'VALUES (?, ?)',
        (normalize_name(alias), company_id),
      )

  def resolve_alias(self, name: str) -> Optional[int]:
    '''Resolve a raw name via aliases then fuzzy match.

    Args:
      name: Raw company name.

    Returns:
      Company id or None.
    '''
    with fetch_one(
      self._db_path,
      'SELECT company_id FROM company_aliases WHERE alias = ?',
      (normalize_name(name),),
    ) as row:
      if row:
        return int(row['company_id'])
    found = self.find_by_name(name)
    return int(found['id']) if found else None
