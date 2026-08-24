#!/usr/bin/env python
# -- coding: utf-8 --

'''Skill taxonomy: canonical skills, aliases, and candidate linking.'''


from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from job_hunter.core.db import connect, session


class SkillsTaxonomy:
  '''Load and query the canonical skill/alias tables.'''

  def __init__(self, db_path: str | Path) -> None:
    '''Initialize the taxonomy.

    Args:
      db_path: Application database path.
    '''
    self._db_path = str(db_path)
    self._cache: Dict[str, int] = {}

  def load_seed(self, payload: Dict[str, Any]) -> int:
    '''Upsert canonical skills and aliases from seed YAML content.

    Args:
      payload: Mapping like {skills: {Python: {category, aliases: [...]}}}.

    Returns:
      Number of canonical skills processed.
    '''
    count = 0
    with session(self._db_path) as conn:
      for name, spec in (payload.get('skills') or {}).items():
        category = (spec or {}).get('category') if isinstance(spec, dict) else None
        conn.execute(
          'INSERT INTO skills (canonical_name, category) VALUES (?, ?) '
          'ON CONFLICT(canonical_name) DO UPDATE SET category = excluded.category',
          (name.strip(), category),
        )
        row = conn.execute(
          'SELECT id FROM skills WHERE canonical_name = ?', (name.strip(),),
        ).fetchone()
        skill_id = int(row['id'])
        count += 1
        aliases = list((spec or {}).get('aliases') or []) if isinstance(spec, dict) else []
        for alias in [name] + aliases:
          conn.execute(
            'INSERT OR IGNORE INTO skill_aliases (alias, skill_id) VALUES (?, ?)',
            (alias.strip().lower(), skill_id),
          )
    self._cache.clear()
    return count

  def load_seed_file(self, path: str | Path) -> int:
    '''Load a skills_aliases.yaml file.

    Args:
      path: Seed file path.

    Returns:
      Number of skills processed.
    '''
    payload = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    return self.load_seed(payload)

  def _resolve_uncached(self, alias: str) -> Optional[int]:
    '''Look up one alias in the database.

    Args:
      alias: Lowercased alias text.

    Returns:
      Skill id or None.
    '''
    row = connect(self._db_path, readonly=True).execute(
      'SELECT skill_id FROM skill_aliases WHERE alias = ?', (alias,),
    ).fetchone()
    return int(row['skill_id']) if row else None

  def resolve(self, alias: str) -> Optional[int]:
    '''Resolve any casing variant to its canonical skill id.

    Args:
      alias: Raw skill text.

    Returns:
      Skill id or None.
    '''
    key = ' '.join(alias.split()).lower()
    if not self._cache:
      rows = connect(self._db_path, readonly=True).execute(
        'SELECT alias, skill_id FROM skill_aliases',
      ).fetchall()
      self._cache = {row['alias']: int(row['skill_id']) for row in rows}
    return self._cache.get(key)

  def resolve_many(self, names: List[str]) -> List[int]:
    '''Resolve a list of raw skill names to unique ids.

    Args:
      names: Raw skill names.

    Returns:
      Unique resolved skill ids.
    '''
    ids: List[int] = []
    for name in names:
      found = self.resolve(name)
      if found is not None and found not in ids:
        ids.append(found)
    return ids

  def all_skills(self) -> List[Dict[str, Any]]:
    '''Return every canonical skill.

    Returns:
      Rows with id, canonical_name, category.
    '''
    rows = connect(self._db_path, readonly=True).execute(
      'SELECT id, canonical_name, category FROM skills ORDER BY canonical_name',
    ).fetchall()
    return [dict(row) for row in rows]
