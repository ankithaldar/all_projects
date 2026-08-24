#!/usr/bin/env python
# -- coding: utf-8 --

'''First-time bootstrap: logging, migrations, taxonomy, default settings.'''


from __future__ import annotations

from pathlib import Path

from job_hunter.core.config import AppSettings
from job_hunter.core.db import run_migrations
from job_hunter.core.logging import setup_logging


def bootstrap(config_path: str | Path, seeds_dir: Path | None = None) -> AppSettings:
  '''Prepare the data directory, schema, skills taxonomy, and defaults.

  Args:
    config_path: Path to app.yaml.
    seeds_dir: Optional seeds override.

  Returns:
    Loaded settings.
  '''
  seeds_dir = Path(seeds_dir) if seeds_dir else None
  settings = AppSettings(config_path)
  setup_logging(settings.log_level, settings.data_dir / 'logs')
  settings.data_dir.mkdir(parents=True, exist_ok=True)
  (settings.data_dir / 'inbox' / 'linkedin').mkdir(parents=True, exist_ok=True)
  (settings.data_dir / 'inbox' / 'naukri').mkdir(parents=True, exist_ok=True)
  (settings.data_dir / 'resumes').mkdir(parents=True, exist_ok=True)
  run_migrations(settings.db_path)
  from job_hunter.core.db import session
  with session(settings.db_path) as conn:
    conn.execute(
      "INSERT OR IGNORE INTO sources (key, kind) VALUES "
      "('workday', 'ats'), ('himalayas', 'aggregator')",
    )

  import yaml
  from job_hunter.db.repositories.settings import SettingsRepository
  repo = SettingsRepository(settings.db_path)
  repo.ensure_defaults(settings)
  taxonomy_path = (seeds_dir or settings.seeds_dir) / 'skills_aliases.yaml'
  if taxonomy_path.exists():
    payload = yaml.safe_load(taxonomy_path.read_text(encoding='utf-8')) or {}
    from job_hunter.services.skills_taxonomy import SkillsTaxonomy
    taxonomy = SkillsTaxonomy(settings.db_path)
    taxonomy.load_seed(payload)
  return settings
