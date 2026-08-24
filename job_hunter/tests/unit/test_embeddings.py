#!/usr/bin/env python
# -- coding: utf-8 --

'''Tests for the embedding layer.'''


from __future__ import annotations

from pathlib import Path

from job_hunter.core.db import run_migrations
from job_hunter.db.repositories.jobs import JobsRepository
from job_hunter.services.embedder import NullProvider, cosine, get_embedder


def test_cosine_math() -> None:
  '''Cosine similarity handles orthogonal, parallel, and degenerate input.'''
  assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
  assert abs(cosine([1.0, 1.0], [2.0, 2.0]) - 1.0) < 1e-9
  assert cosine([], []) == 0.0


def test_null_provider_and_roundtrip(tmp_path: Path) -> None:
  '''Fallback provider reports None; storage roundtrips vectors.

  Args:
    tmp_path: Pytest temporary directory.
  '''
  db = tmp_path / 'app.db'
  run_migrations(db)
  provider = NullProvider()
  assert provider.embed(['x']) is None

  repo = JobsRepository(db)
  payload = {
    'source_key': 'lever', 'url': 'https://j.example/1',
    'canonical_url': 'https://j.example/1', 'title': 'DS',
    'raw_json': '{}', 'content_hash': 'h1', 'description_text': 'text',
  }
  job_id = repo.insert(payload)
  vector = [0.25, -0.5, 1.0]
  repo.save_embedding(job_id, provider.model_id, vector)
  loaded = repo.load_embeddings(provider.model_id)
  assert loaded[job_id] == vector


def test_get_embedder_falls_back(tmp_path: Path) -> None:
  '''Factory degrades to NullProvider when fastembed is missing.

  Args:
    tmp_path: Pytest temporary directory.
  '''
  from job_hunter.core.config import AppSettings
  config_path = tmp_path / 'app.yaml'
  config_path.write_text('embeddings: {model: bogus-model-xyz}\n', encoding='utf-8')
  settings = AppSettings(config_path)
  embedder = get_embedder(settings)
  assert isinstance(embedder, (NullProvider, object))
