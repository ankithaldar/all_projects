#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Candidate profile and skills persistence.'''

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from job_hunter.core.db import (
  fetch_all,
  fetch_one,
  filter_to_columns,
  quote_identifier,
  session,
)


class ProfileRepository:
  '''Read/write candidate profiles, resumes, and skill links.'''

  def __init__(self, db_path: str | Path) -> None:
    '''Initialize the repository.

    Args:
      db_path: Application database path.
    '''
    self._db_path = str(db_path)

  def get_profile(self, candidate_id: int = 1) -> Optional[Dict[str, Any]]:
    '''Fetch the latest profile for a candidate.

    Args:
      candidate_id: Candidate id.

    Returns:
      Row mapping with decoded JSON columns or None.
    '''
    with fetch_one(
      self._db_path,
      'SELECT * FROM candidate_profiles WHERE candidate_id = ? '
      'ORDER BY version DESC LIMIT 1',
      (candidate_id,),
    ) as row:
      if row is None:
        return None
      data = dict(row)
    for key in (
      'target_roles',
      'seniority_keywords',
      'target_verticals',
      'blocked_verticals',
      'cities',
      'employment_types',
    ):
      data[key] = json.loads(data.get(key) or '[]')
    return data

  def save_profile(
    self,
    fields: Dict[str, Any],
    candidate_id: int = 1,
  ) -> int:
    '''Persist a new profile version.

    Args:
      fields: Column values (JSON lists encoded here). Unknown
        keys such as ``skills_resolved`` are dropped with a
        warning, never interpolated into SQL.
      candidate_id: Candidate id.

    Returns:
      New profile version id.
    '''
    existing = self.get_profile(candidate_id)
    version = (existing['version'] + 1) if existing else 1
    list_keys = (
      'target_roles',
      'seniority_keywords',
      'target_verticals',
      'blocked_verticals',
      'cities',
      'employment_types',
    )
    payload: Dict[str, Any] = {}
    for key, value in fields.items():
      payload[key] = (
        json.dumps(value, ensure_ascii=False) if key in list_keys else value
      )
    with session(self._db_path) as conn:
      kept = filter_to_columns(conn, 'candidate_profiles', payload)
      kept['candidate_id'] = candidate_id
      kept['version'] = version
      columns = sorted(kept.keys())
      quoted = ', '.join(quote_identifier(c) for c in columns)
      marks = ', '.join('?' * len(columns))
      cur = conn.execute(
        f'INSERT INTO candidate_profiles ({quoted}) VALUES ({marks})',
        tuple(kept[c] for c in columns),
      )
      return int(cur.lastrowid)

  def save_resume_ex(
    self,
    file_path: str,
    sha256: str,
    mime: str,
    parsed_ok: bool,
    candidate_id: int = 1,
  ) -> Tuple[int, bool]:
    '''Register an uploaded resume, reporting ownership.

    Args:
      file_path: Stored file location.
      sha256: File digest.
      mime: Detected media type.
      parsed_ok: Whether text extraction succeeded.
      candidate_id: Candidate id.

    Returns:
      Tuple of (resume row id, created). ``created`` is True when
      this call inserted a new row, False when a duplicate sha256
      returned the pre-existing row.

    Raises:
      ValueError: With ``candidate not found`` when the insert
        fails integrity checks and no existing resume matches
        (e.g. an unknown candidate_id foreign key).
    '''
    with session(self._db_path) as conn:
      try:
        cur = conn.execute(
          'INSERT INTO resumes (candidate_id, file_path, sha256, mime, '
          'parsed_ok) VALUES (?, ?, ?, ?, ?)',
          (candidate_id, file_path, sha256, mime, int(parsed_ok)),
        )
        return (int(cur.lastrowid), True)
      except sqlite3.IntegrityError as exc:
        row = conn.execute(
          'SELECT id FROM resumes WHERE sha256 = ?',
          (sha256,),
        ).fetchone()
        if row is None:
          raise ValueError('candidate not found') from exc
        return (int(row['id']), False)

  def save_resume(
    self,
    file_path: str,
    sha256: str,
    mime: str,
    parsed_ok: bool,
    candidate_id: int = 1,
  ) -> int:
    '''Register an uploaded resume.

    Args:
      file_path: Stored file location.
      sha256: File digest.
      mime: Detected media type.
      parsed_ok: Whether text extraction succeeded.
      candidate_id: Candidate id.

    Returns:
      Resume row id. A duplicate sha256 returns the existing id.

    Raises:
      ValueError: With ``candidate not found`` when the insert
        fails integrity checks and no existing resume matches
        (e.g. an unknown candidate_id foreign key).
    '''
    resume_id, _created = self.save_resume_ex(
      file_path, sha256, mime, parsed_ok, candidate_id,
    )
    return resume_id

  def get_resume(self, resume_id: int) -> Optional[Dict[str, Any]]:
    '''Fetch one resume row by id.

    Args:
      resume_id: Resume row id.

    Returns:
      Row mapping or None when no such resume exists.
    '''
    with fetch_one(
      self._db_path,
      'SELECT id, candidate_id, file_path, sha256, mime, '
      'uploaded_at, parsed_ok FROM resumes WHERE id = ?',
      (resume_id,),
    ) as row:
      if row is None:
        return None
      return dict(row)

  def delete_resume(self, resume_id: int) -> None:
    '''Delete one resume row by id.

    Args:
      resume_id: Resume row id to remove.
    '''
    with session(self._db_path) as conn:
      conn.execute(
        'DELETE FROM resumes WHERE id = ?',
        (resume_id,),
      )

  def set_candidate_skills(
    self,
    skill_ids: List[int],
    candidate_id: int = 1,
  ) -> None:
    '''Replace the candidate's skill links.

    Args:
      skill_ids: Canonical skill ids.
      candidate_id: Candidate id.
    '''
    with session(self._db_path) as conn:
      conn.execute(
        'DELETE FROM candidate_skills WHERE candidate_id = ?',
        (candidate_id,),
      )
      for skill_id in skill_ids:
        conn.execute(
          'INSERT OR IGNORE INTO candidate_skills (candidate_id, skill_id) '
          'VALUES (?, ?)',
          (candidate_id, skill_id),
        )

  def candidate_skill_names(self, candidate_id: int = 1) -> List[str]:
    '''Return canonical names of the candidate's skills.

    Args:
      candidate_id: Candidate id.

    Returns:
      Sorted name list.
    '''
    with fetch_all(
      self._db_path,
      'SELECT s.canonical_name AS name FROM candidate_skills cs '
      'JOIN skills s ON s.id = cs.skill_id WHERE cs.candidate_id = ? '
      'ORDER BY s.canonical_name',
      (candidate_id,),
    ) as rows:
      return [row['name'] for row in rows]
