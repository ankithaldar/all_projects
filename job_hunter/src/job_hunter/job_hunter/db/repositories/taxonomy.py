#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Skill taxonomy read repository backing the skills tables.'''

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from job_hunter.core.db import fetch_all, fetch_one


class TaxonomyRepository:
  '''Read canonical skills and resolve aliases to skill ids.'''

  def __init__(self, db_path: str | Path) -> None:
    '''Initialize the repository.

    Args:
      db_path: Application database path.
    '''
    self._db_path = str(db_path)

  def skill_rows(self) -> List[Dict[str, Any]]:
    '''Return every canonical skill.

    Returns:
      Rows with id, canonical_name, category, ordered by name.
    '''
    with fetch_all(
      self._db_path,
      'SELECT id, canonical_name, category FROM skills ORDER BY canonical_name',
    ) as rows:
      return [dict(row) for row in rows]

  def resolve_alias(self, name: str) -> Optional[int]:
    '''Resolve a skill alias to its canonical skill id.

    Args:
      name: Raw skill text in any casing.

    Returns:
      Skill id or None when the alias is unknown.
    '''
    key = ' '.join(name.split()).lower()
    with fetch_one(
      self._db_path,
      'SELECT skill_id FROM skill_aliases WHERE alias = ?',
      (key,),
    ) as row:
      return int(row['skill_id']) if row else None
