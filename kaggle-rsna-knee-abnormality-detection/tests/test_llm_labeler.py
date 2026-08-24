#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the OpenRouter weak-label tier (no network access)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from knee.config_params.schema import TARGETS
from knee.engines.llm_labeler import (
  LLMLabelError,
  OpenRouterLabeler,
  apply_schema,
  label_many,
  parse_llm_json,
  report_key,
)

_FULL_REPLY = {
  'ACL': 1,
  'MCL': 0,
  'Medial Meniscus': 1,
  'Lateral Meniscus': 0.5,
  'Medial OA': 0,
  'Lateral OA': 0,
  'PF OA': 0.5,
  'Effusion': 1,
  'Synovitis': 1,
  "Baker's": 0,
  'Contusion': 0.5,
  'Fracture': 0,
}


class TestParseLlmJson:
  """Reply parsing must tolerate the usual LLM decorations."""

  def test_plain_json(self):
    assert parse_llm_json('{"ACL": 1}') == {'ACL': 1}

  def test_fenced_json(self):
    text = '```json\n{"ACL": 0}\n```'
    assert parse_llm_json(text) == {'ACL': 0}

  def test_prose_wrapped_json(self):
    text = 'Sure! Here is the answer:\n{"Effusion": 0.5}\nHope that helps.'
    assert parse_llm_json(text) == {'Effusion': 0.5}

  def test_garbage_raises(self):
    with pytest.raises(LLMLabelError):
      parse_llm_json('no json here at all')


class TestApplySchema:
  """Projection onto the canonical 12-vector + trust mask."""

  def test_full_mapping(self):
    probs, mask = apply_schema(_FULL_REPLY)
    assert list(probs.shape) == [12]
    index = {t: i for i, t in enumerate(TARGETS)}
    assert probs[index['ACL']] == pytest.approx(1.0)
    assert bool(mask[index['ACL']])
    assert not bool(mask[index['Lateral Meniscus']])

  def test_missing_keys_default_unknown(self):
    probs, mask = apply_schema({'ACL': 1})
    index = {t: i for i, t in enumerate(TARGETS)}
    assert probs[index['MCL']] == pytest.approx(0.5)
    assert not mask[index['MCL']]
    assert mask[index['ACL']]

  def test_values_clamped(self):
    probs, _ = apply_schema({'ACL': 7, 'Fracture': -2})
    index = {t: i for i, t in enumerate(TARGETS)}
    assert probs[index['ACL']] == 1.0
    assert probs[index['Fracture']] == 0.0


class TestOpenRouterLabeler:
  """Transport behaviour with a stubbed HTTP layer."""

  @staticmethod
  def _labeler(monkeypatch, responses):
    calls = []

    class _Response:
      def __init__(self, status_code: int, body: str = '') -> None:
        self.status_code = status_code
        self._body = body
        self.headers = {}

      @property
      def text(self) -> str:
        return self._body

      def json(self):
        return {'choices': [{'message': {'content': self._body}}]}

    sequence = iter(responses)

    def fake_post(url, **kwargs):  # noqa: ANN001 - mirrors requests API
      calls.append(url)
      item = next(sequence)
      if isinstance(item, int):
        return _Response(item)
      return _Response(200, json.dumps(item))

    monkeypatch.setattr('knee.engines.llm_labeler.requests.post', fake_post)
    monkeypatch.setattr(
      'knee.engines.llm_labeler.time.sleep', lambda seconds: None
    )
    return calls

  def test_happy_path(self, monkeypatch):
    self._labeler(monkeypatch, [_FULL_REPLY])
    labeler = OpenRouterLabeler(api_key='k', max_retries=2)
    probs, mask = labeler.label_report('report text')
    index = {t: i for i, t in enumerate(TARGETS)}
    assert probs[index['Effusion']] == pytest.approx(1.0)
    assert not mask[index['Contusion']]

  def test_retries_on_429_then_succeeds(self, monkeypatch):
    self._labeler(monkeypatch, [429, 429, _FULL_REPLY])
    labeler = OpenRouterLabeler(api_key='k', max_retries=3)
    probs, _ = labeler.label_report('text')
    assert len(probs) == 12

  def test_permanent_error_no_retry(self, monkeypatch):
    calls = self._labeler(monkeypatch, [401])
    labeler = OpenRouterLabeler(api_key='k', max_retries=3)
    with pytest.raises(LLMLabelError):
      labeler.label_report('text')
    assert len(calls) == 1

  def test_empty_key_rejected(self):
    with pytest.raises(ValueError):
      OpenRouterLabeler(api_key='')


class TestLabelMany:
  """Caching, concurrency and failure isolation."""

  @staticmethod
  def _frame(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
      {
        'StudyInstanceUID': [f'study{i}' for i in range(n)],
        'Report': [f'findings of study {i}' for i in range(n)],
      }
    )

  def test_labels_every_study(self):
    frame = self._frame()
    result = label_many(
      frame['StudyInstanceUID'],
      frame['Report'],
      lambda text: (np.full(12, 1.0, np.float32), np.ones(12, bool)),
      cache_path=None,
    )
    assert len(result) == 5
    assert result[TARGETS[0]].eq(1.0).all()

  def test_cache_prevents_second_calls(self, tmp_path):
    cache = str(tmp_path / 'cache.parquet')
    frame = self._frame()
    counter = {'n': 0}

    def label_fn(text):
      counter['n'] += 1
      return np.full(12, 0.0, np.float32), np.zeros(12, bool)

    first = label_many(
      frame['StudyInstanceUID'], frame['Report'], label_fn, cache_path=cache
    )
    second = label_many(
      frame['StudyInstanceUID'], frame['Report'], label_fn, cache_path=cache
    )
    assert counter['n'] == 5  # only the cold run hits the callable
    pd.testing.assert_frame_equal(first[list(TARGETS)], second[list(TARGETS)])

  def test_partial_cache_resumes(self, tmp_path):
    cache = str(tmp_path / 'cache.parquet')
    frame = self._frame()
    key = report_key(str(frame['Report'].iloc[0]))
    payload = [
      {
        'key': key,
        'probs_json': json.dumps([1.0] * 12),
        'mask_json': json.dumps([True] * 12),
      }
    ]
    pd.DataFrame(payload).to_parquet(cache)
    seen = []

    def label_fn(text):
      seen.append(text)
      return np.zeros(12, np.float32), np.zeros(12, bool)

    result = label_many(
      frame['StudyInstanceUID'], frame['Report'], label_fn, cache_path=cache
    )
    assert len(seen) == 4  # one study served from cache
    row = result.iloc[0]
    assert row[TARGETS[0]] == pytest.approx(1.0)

  def test_blank_reports_are_all_unknown(self):
    frame = pd.DataFrame(
      {
        'StudyInstanceUID': ['a'],
        'Report': ['   '],
      }
    )
    result = label_many(
      frame['StudyInstanceUID'],
      frame['Report'],
      lambda t: (_ for _ in ()).throw(AssertionError('x')),
      cache_path=None,
    )
    assert result[list(TARGETS)].eq(0.5).all().all()

  def test_labeler_exception_becomes_unknown(self):
    frame = self._frame(2)

    def boom(text):
      raise LLMLabelError('api down')

    result = label_many(
      frame['StudyInstanceUID'], frame['Report'], boom, cache_path=None
    )
    assert result[list(TARGETS)].eq(0.5).all().all()
