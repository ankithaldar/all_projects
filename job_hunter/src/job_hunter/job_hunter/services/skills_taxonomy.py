#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Skill taxonomy: canonical skills, aliases, and candidate linking.'''


from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from job_hunter.core.db import session
from job_hunter.db.repositories.taxonomy import TaxonomyRepository

logger = logging.getLogger(__name__)


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
    skills_payload = payload.get('skills') or {}
    if not isinstance(skills_payload, dict):
      logger.warning(
        'skipping non-dict skills payload: %r', skills_payload,
      )
      return 0
    # Writes stay on session(): TaxonomyRepository is read-only
    # (skill_rows/resolve_alias) and has no seed/upsert method.
    with session(self._db_path) as conn:
      for name, spec in skills_payload.items():
        if not isinstance(name, str):
          logger.warning('skipping non-str skill name: %r', name)
          continue
        is_dict = isinstance(spec, dict)
        if not is_dict and spec is not None:
          logger.warning(
            'skipping non-dict skill spec for %r: %r', name, spec,
          )
          continue
        category = spec.get('category') if is_dict else None
        conn.execute(
          'INSERT INTO skills (canonical_name, category) VALUES (?, ?) '
          'ON CONFLICT(canonical_name) '
          'DO UPDATE SET category = excluded.category',
          (name.strip(), category),
        )
        row = conn.execute(
          'SELECT id FROM skills WHERE canonical_name = ?', (name.strip(),),
        ).fetchone()
        skill_id = int(row['id'])
        count += 1
        raw_aliases = spec.get('aliases') if is_dict else []
        if not is_dict or raw_aliases is None:
          aliases = []
        elif isinstance(raw_aliases, (list, tuple)):
          aliases = []
          for alias in raw_aliases:
            if not isinstance(alias, str):
              logger.warning(
                'skipping non-str alias for %r: %r', name, alias,
              )
              continue
            aliases.append(alias)
        else:
          logger.warning(
            'skipping non-list aliases for %r: %r', name, raw_aliases,
          )
          aliases = []
        for alias in [name] + aliases:
          conn.execute(
            'INSERT OR IGNORE INTO skill_aliases '
            '(alias, skill_id) VALUES (?, ?)',
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
    return TaxonomyRepository(self._db_path).resolve_alias(alias)

  def resolve(self, alias: str) -> Optional[int]:
    '''Resolve any casing variant to its canonical skill id.

    Args:
      alias: Raw skill text.

    Returns:
      Skill id or None.
    '''
    if not isinstance(alias, str):
      return None
    key = ' '.join(alias.split()).lower()
    if key in self._cache:
      return self._cache[key]
    found = TaxonomyRepository(self._db_path).resolve_alias(key)
    if found is not None:
      self._cache[key] = int(found)
    return found

  def resolve_many(self, names: List[str]) -> List[int]:
    '''Resolve a list of raw skill names to unique ids.

    Args:
      names: Raw skill names.

    Returns:
      Unique resolved skill ids.

    Raises:
      TypeError: When names is a bare string/bytes or not a list/tuple.
    '''
    if isinstance(names, (str, bytes)) or not isinstance(
      names, (list, tuple),
    ):
      raise TypeError('names must be a list of str')
    ids: List[int] = []
    for name in names:
      if not isinstance(name, str):
        logger.warning('skipping non-str skill name: %r', name)
        continue
      found = self.resolve(name)
      if found is not None and found not in ids:
        ids.append(found)
    return ids

  def all_skills(self) -> List[Dict[str, Any]]:
    '''Return every canonical skill.

    Returns:
      Rows with id, canonical_name, category.
    '''
    return TaxonomyRepository(self._db_path).skill_rows()
