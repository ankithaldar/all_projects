#!/usr/bin/env python
# -- coding: utf-8 --

'''Unit tests: episodic, semantic, and procedural memory.'''


from __future__ import annotations

from typing import Any

from chapter03_cognition.memory import MemoryLayer, MemoryStore
from chapter03_cognition.perception import PerceptionLayer
from chapter03_cognition.schemas import Episode


def make_store(tmp_path: Any) -> MemoryStore:
  '''Create a memory store in a temp directory.

  Args:
    tmp_path: Pytest temp path.

  Returns:
    MemoryStore.
  '''
  return MemoryStore(tmp_path / 'memory.db')


def make_episode(**overrides: Any) -> Episode:
  '''Build an episode with sensible defaults.

  Args:
    **overrides: Field overrides.

  Returns:
    Episode.
  '''
  payload = {
    'episode_id': 'e1',
    'session_id': 's1',
    'task': 'Check low stock for store S02 and sales trend',
    'domain': 'retail',
    'strategy': 'conservative',
    'status': 'completed',
    'success': True,
    'answer_preview': 'ok',
    'tool_sequence': ['retail-ops.retail_low_stock_report'],
    'tags': ['retail', 'S02', 'R-401'],
  }
  payload.update(overrides)
  return Episode(**payload)


class TestEpisodes:
  '''Episode persistence and recall.'''

  def test_save_and_list(self, tmp_path: Any) -> None:
    store = make_store(tmp_path)
    try:
      store.save_episode(make_episode())
      episodes = store.list_episodes()
      assert len(episodes) == 1
      assert episodes[0].episode_id == 'e1'
      assert episodes[0].tool_sequence
    finally:
      store.close()

  def test_recall_by_domain_and_entity(self, tmp_path: Any) -> None:
    store = make_store(tmp_path)
    try:
      store.save_episode(make_episode(episode_id='retail-s02'))
      store.save_episode(make_episode(
        episode_id='telecom-cs77',
        task='Dispatch technician to CS-77',
        domain='telecom',
        tags=['telecom', 'CS-77'],
      ))
      recalled = store.recall_episodes(
        domain='retail',
        keywords=['stock', 'sales'],
        entities={'store_ids': ['S02']},
      )
      assert recalled
      assert recalled[0].episode_id == 'retail-s02'
    finally:
      store.close()

  def test_recall_returns_empty_for_unrelated(self, tmp_path: Any) -> None:
    store = make_store(tmp_path)
    try:
      store.save_episode(make_episode())
      recalled = store.recall_episodes(
        domain='telecom', keywords=['latency'], entities={},
      )
      assert recalled == []
    finally:
      store.close()


class TestFacts:
  '''Semantic memory (durable facts).'''

  def test_set_and_get(self, tmp_path: Any) -> None:
    store = make_store(tmp_path)
    try:
      store.set_fact('promo.S02', {'sku': 'R-401'}, domain='retail')
      assert store.get_fact('promo.S02') == {'sku': 'R-401'}
      assert store.get_fact('missing') is None
    finally:
      store.close()

  def test_facts_for_domain_includes_global(self, tmp_path: Any) -> None:
    store = make_store(tmp_path)
    try:
      store.set_fact('retail.only', 1, domain='retail')
      store.set_fact('global', 2, domain='')
      facts = store.facts_for('retail')
      assert facts == {'retail.only': 1, 'global': 2}
    finally:
      store.close()


class TestStrategyStats:
  '''Procedural memory (strategy performance).'''

  def test_record_and_read(self, tmp_path: Any) -> None:
    store = make_store(tmp_path)
    try:
      store.record_strategy_result('conservative', True, 100.0, 10)
      store.record_strategy_result('conservative', False, 200.0, 20)
      stats = store.strategy_stats()
      assert len(stats) == 1
      assert stats[0].runs == 2
      assert stats[0].successes == 1
      assert stats[0].success_rate == 0.5
      assert stats[0].avg_duration_ms == 150.0
    finally:
      store.close()


class TestMemoryLayer:
  '''Context building and strategy recommendation.'''

  def test_context_notes_and_recall(self, tmp_path: Any) -> None:
    store = make_store(tmp_path)
    try:
      store.save_episode(make_episode())
      store.set_fact('last_promo', 'R-401', domain='retail')
      percept = PerceptionLayer().perceive(
        'Check low stock for store S02', 'it-memory',
      )
      context = MemoryLayer(store).build_context(percept)
      assert context.relevant_episodes
      assert context.facts
      assert any('past run' in note for note in context.recall_notes)
    finally:
      store.close()

  def test_recommends_conservative_for_writes(self, tmp_path: Any) -> None:
    store = make_store(tmp_path)
    try:
      percept = PerceptionLayer().perceive(
        'Restock store S01 with 10 units of R-101.', 'it-memory',
      )
      context = MemoryLayer(store).build_context(percept)
      assert context.recommended_strategy == 'conservative'
    finally:
      store.close()

  def test_recommends_exploratory_for_ambiguity(self, tmp_path: Any) -> None:
    store = make_store(tmp_path)
    try:
      percept = PerceptionLayer().perceive(
        'Investigate why latency is high and diagnose the root cause',
        'it-memory',
      )
      context = MemoryLayer(store).build_context(percept)
      assert context.recommended_strategy == 'exploratory'
    finally:
      store.close()

  def test_recommends_best_remembered_strategy(self, tmp_path: Any) -> None:
    store = make_store(tmp_path)
    try:
      for index in range(3):
        store.save_episode(make_episode(
          episode_id=f'exp-{index}', strategy='exploratory', success=True,
        ))
      percept = PerceptionLayer().perceive(
        'Check low stock for store S02', 'it-memory',
      )
      context = MemoryLayer(store).build_context(percept)
      assert context.recommended_strategy == 'exploratory'
    finally:
      store.close()

  def test_recommends_fallback_when_primaries_struggle(
    self, tmp_path: Any,
  ) -> None:
    store = make_store(tmp_path)
    try:
      for index in range(3):
        store.record_strategy_result('conservative', False)
        store.record_strategy_result('exploratory', False)
      store.record_strategy_result('fallback', True)
      percept = PerceptionLayer().perceive(
        'Check low stock for store S02', 'it-memory',
      )
      context = MemoryLayer(store).build_context(percept)
      assert context.recommended_strategy == 'fallback'
    finally:
      store.close()
