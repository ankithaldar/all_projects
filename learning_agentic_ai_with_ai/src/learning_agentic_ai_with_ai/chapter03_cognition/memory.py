#!/usr/bin/env python
# -- coding: utf-8 --

'''Layer 2 - Memory: episodic + semantic + procedural memory.

The memory layer gives the planner three kinds of recall:

  - **episodic** - past runs ("we tried a conservative plan on this store
    last week and the sales tool timed out"), retrieved by domain, keyword,
    and entity overlap with simple token scoring (no embeddings needed for
    a course, and deterministic for tests),
  - **semantic** - durable facts ("store S02 runs promotions on R-401"),
  - **procedural** - per-strategy success statistics, which feed strategy
    selection on the next run.

Storage is a dedicated SQLite file (`data/chapter03_memory.db`) so the
chapter owns its schema while execution history continues to live in the
shared `agentic_common` store. Memory is written during reflection, after
the action layer finishes, so a crashed run never pollutes it with a
half-truth.
'''


from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_common.logging import get_logger, log_event
from chapter03_cognition.schemas import (
  Episode,
  MemoryContext,
  Percept,
  StrategyName,
  StrategyStats,
)

logger = get_logger(__name__)

_SCHEMA = '''
CREATE TABLE IF NOT EXISTS episodes (
  episode_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  task TEXT NOT NULL,
  domain TEXT NOT NULL DEFAULT 'unknown',
  strategy TEXT NOT NULL DEFAULT 'conservative',
  status TEXT NOT NULL DEFAULT 'completed',
  success INTEGER NOT NULL DEFAULT 1,
  answer_preview TEXT NOT NULL DEFAULT '',
  tool_sequence_json TEXT NOT NULL DEFAULT '[]',
  error_classes_json TEXT NOT NULL DEFAULT '[]',
  adaptations_json TEXT NOT NULL DEFAULT '[]',
  duration_ms REAL NOT NULL DEFAULT 0,
  total_tokens INTEGER NOT NULL DEFAULT 0,
  tags_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_domain ON episodes(domain);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);

CREATE TABLE IF NOT EXISTS facts (
  key TEXT PRIMARY KEY,
  value_json TEXT NOT NULL,
  domain TEXT NOT NULL DEFAULT '',
  tags_json TEXT NOT NULL DEFAULT '[]',
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_stats (
  strategy TEXT PRIMARY KEY,
  runs INTEGER NOT NULL DEFAULT 0,
  successes INTEGER NOT NULL DEFAULT 0,
  failures INTEGER NOT NULL DEFAULT 0,
  total_duration_ms REAL NOT NULL DEFAULT 0,
  total_tokens INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL
);
'''


def _utc_now() -> str:
  '''Return the current UTC time as an ISO string.

  Returns:
    ISO 8601 timestamp.
  '''
  return datetime.now(timezone.utc).isoformat()


def _tokens(text: str) -> set:
  '''Lowercase word tokens of a text.

  Args:
    text: Source text.

  Returns:
    Token set.
  '''
  cleaned = ''.join(
    char.lower() if char.isalnum() else ' ' for char in (text or '')
  )
  return {token for token in cleaned.split() if len(token) > 1}


class MemoryStore:
  '''Thread-safe SQLite store for episodes, facts, and strategy stats.'''

  def __init__(self, db_path: Path | str) -> None:
    '''Open (and initialize) the memory store.

    Args:
      db_path: Path to the SQLite database file.
    '''
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    self._lock = threading.Lock()
    self._conn = sqlite3.connect(str(path), check_same_thread=False)
    self._conn.row_factory = sqlite3.Row
    self._conn.executescript(_SCHEMA)
    self._conn.commit()

  def close(self) -> None:
    '''Close the database connection.'''
    with self._lock:
      self._conn.close()

  # ------------------------------------------------------------------
  # Episodes
  # ------------------------------------------------------------------

  def save_episode(self, episode: Episode) -> Episode:
    '''Persist one episode.

    Args:
      episode: Episode to store (created_at filled when empty).

    Returns:
      The stored episode.
    '''
    record = episode.model_copy()
    if not record.created_at:
      record.created_at = _utc_now()
    with self._lock:
      self._conn.execute(
        'INSERT OR REPLACE INTO episodes (episode_id, session_id, task, '
        'domain, strategy, status, success, answer_preview, '
        'tool_sequence_json, error_classes_json, adaptations_json, '
        'duration_ms, total_tokens, tags_json, created_at) '
        'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
        (
          record.episode_id,
          record.session_id,
          record.task,
          record.domain,
          record.strategy,
          record.status,
          int(record.success),
          record.answer_preview,
          json.dumps(record.tool_sequence),
          json.dumps(record.error_classes),
          json.dumps(record.adaptations),
          float(record.duration_ms),
          int(record.total_tokens),
          json.dumps(record.tags),
          record.created_at,
        ),
      )
      self._conn.commit()
    return record

  def list_episodes(self, limit: int = 50) -> List[Episode]:
    '''List episodes, newest first.

    Args:
      limit: Maximum rows returned.

    Returns:
      Episode list.
    '''
    with self._lock:
      rows = self._conn.execute(
        'SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?',
        (int(limit),),
      ).fetchall()
    return [self._row_to_episode(row) for row in rows]

  def recall_episodes(
    self,
    domain: str,
    keywords: List[str],
    entities: Optional[Dict[str, List[str]]] = None,
    limit: int = 5,
  ) -> List[Episode]:
    '''Recall episodes most similar to the current task.

    Scoring: +3 same domain, +1 per shared keyword/tag, +2 per shared
    entity value. Recency breaks ties (newest first).

    Args:
      domain: Current task domain.
      keywords: Current task keywords.
      entities: Current task entities.
      limit: Maximum episodes to return.

    Returns:
      Ranked episodes with a positive score.
    '''
    keyword_tokens = {token for phrase in keywords for token in _tokens(phrase)}
    entity_values = {
      value.lower()
      for values in (entities or {}).values() for value in values
    }

    scored = []
    for episode in self.list_episodes(limit=200):
      score = 0
      if domain != 'unknown' and episode.domain == domain:
        score += 3
      episode_tokens = _tokens(episode.task) | {
        token for tag in episode.tags for token in _tokens(tag)
      }
      score += len(keyword_tokens & episode_tokens)
      episode_values = {value.lower() for value in _entities_of(episode)}
      score += 2 * len(entity_values & episode_values)
      if score > 0:
        scored.append((score, episode.created_at, episode))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [episode for _, _, episode in scored[:limit]]

  @staticmethod
  def _row_to_episode(row: sqlite3.Row) -> Episode:
    '''Convert a database row into an Episode.

    Args:
      row: SQLite row.

    Returns:
      Episode model.
    '''
    return Episode(
      episode_id=row['episode_id'],
      session_id=row['session_id'],
      task=row['task'],
      domain=row['domain'],
      strategy=row['strategy'],
      status=row['status'],
      success=bool(row['success']),
      answer_preview=row['answer_preview'],
      tool_sequence=json.loads(row['tool_sequence_json'] or '[]'),
      error_classes=json.loads(row['error_classes_json'] or '[]'),
      adaptations=json.loads(row['adaptations_json'] or '[]'),
      duration_ms=row['duration_ms'],
      total_tokens=row['total_tokens'],
      tags=json.loads(row['tags_json'] or '[]'),
      created_at=row['created_at'],
    )

  # ------------------------------------------------------------------
  # Facts
  # ------------------------------------------------------------------

  def set_fact(
    self,
    key: str,
    value: Any,
    domain: str = '',
    tags: Optional[List[str]] = None,
  ) -> None:
    '''Upsert a durable fact.

    Args:
      key: Fact key.
      value: JSON-serializable value.
      domain: Optional domain scope.
      tags: Optional retrieval tags.
    '''
    with self._lock:
      self._conn.execute(
        'INSERT INTO facts (key, value_json, domain, tags_json, updated_at) '
        'VALUES (?,?,?,?,?) '
        'ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, '
        'domain=excluded.domain, tags_json=excluded.tags_json, '
        'updated_at=excluded.updated_at',
        (
          key,
          json.dumps(value, default=str),
          domain,
          json.dumps(tags or []),
          _utc_now(),
        ),
      )
      self._conn.commit()

  def get_fact(self, key: str) -> Optional[Any]:
    '''Fetch one fact value.

    Args:
      key: Fact key.

    Returns:
      Decoded value or None.
    '''
    with self._lock:
      row = self._conn.execute(
        'SELECT value_json FROM facts WHERE key = ?', (key,),
      ).fetchone()
    if row is None:
      return None
    return json.loads(row['value_json'])

  def facts_for(self, domain: str) -> Dict[str, Any]:
    '''Fetch facts for a domain plus global facts.

    Args:
      domain: Domain filter (empty string selects global only).

    Returns:
      Mapping of key to value.
    '''
    with self._lock:
      rows = self._conn.execute(
        'SELECT key, value_json FROM facts '
        'WHERE domain = ? OR domain = ? ORDER BY key',
        (domain, ''),
      ).fetchall()
    return {row['key']: json.loads(row['value_json']) for row in rows}

  # ------------------------------------------------------------------
  # Procedural memory (strategy stats)
  # ------------------------------------------------------------------

  def record_strategy_result(
    self,
    strategy: StrategyName,
    success: bool,
    duration_ms: float = 0.0,
    tokens: int = 0,
  ) -> None:
    '''Update aggregate statistics for a strategy.

    Args:
      strategy: Strategy that ran.
      success: Whether the run completed cleanly.
      duration_ms: Run duration.
      tokens: Tokens spent.
    '''
    with self._lock:
      self._conn.execute(
        'INSERT INTO strategy_stats '
        '(strategy, runs, successes, failures, total_duration_ms, '
        'total_tokens, updated_at) VALUES (?,1,?,?,?,?,?) '
        'ON CONFLICT(strategy) DO UPDATE SET '
        'runs = runs + 1, '
        'successes = successes + excluded.successes, '
        'failures = failures + excluded.failures, '
        'total_duration_ms = total_duration_ms + excluded.total_duration_ms, '
        'total_tokens = total_tokens + excluded.total_tokens, '
        'updated_at = excluded.updated_at',
        (
          strategy,
          int(success),
          int(not success),
          float(duration_ms),
          int(tokens),
          _utc_now(),
        ),
      )
      self._conn.commit()

  def strategy_stats(self) -> List[StrategyStats]:
    '''List per-strategy aggregates.

    Returns:
      Strategy stats sorted by strategy name.
    '''
    with self._lock:
      rows = self._conn.execute(
        'SELECT * FROM strategy_stats ORDER BY strategy',
      ).fetchall()
    stats: List[StrategyStats] = []
    for row in rows:
      runs = int(row['runs'])
      stats.append(
        StrategyStats(
          strategy=row['strategy'],
          runs=runs,
          successes=int(row['successes']),
          failures=int(row['failures']),
          avg_duration_ms=(
            row['total_duration_ms'] / runs if runs else 0.0
          ),
          updated_at=row['updated_at'],
        )
      )
    return stats


def _entities_of(episode: Episode) -> List[str]:
  '''Extract identifier-like tags from an episode's tags.

  Args:
    episode: Episode to inspect.

  Returns:
    Tag strings that look like identifiers (contain a digit).
  '''
  return [tag for tag in episode.tags if any(char.isdigit() for char in tag)]


class MemoryLayer:
  '''Builds a MemoryContext for a percept from the memory store.'''

  def __init__(self, store: MemoryStore, recall_limit: int = 5) -> None:
    '''Initialize the layer.

    Args:
      store: Memory store to read from.
      recall_limit: Maximum episodes recalled per task.
    '''
    self._store = store
    self._recall_limit = max(1, recall_limit)

  def build_context(self, percept: Percept) -> MemoryContext:
    '''Assemble relevant memory for a task.

    Args:
      percept: Perception-layer output.

    Returns:
      MemoryContext with episodes, facts, stats, notes, and a strategy
      recommendation.
    '''
    episodes = self._store.recall_episodes(
      domain=percept.signals.domain,
      keywords=percept.signals.keywords,
      entities=percept.entities,
      limit=self._recall_limit,
    )
    facts = self._store.facts_for(percept.signals.domain)
    stats = self._store.strategy_stats()

    context = MemoryContext(
      relevant_episodes=episodes,
      facts=facts,
      strategy_stats=stats,
      recall_notes=self._notes(episodes, facts, stats),
      recommended_strategy=self._recommend(percept, episodes, stats),
    )
    log_event(
      logger, 20, 'memory_recall',
      session_id=percept.session_id,
      episodes=len(episodes),
      facts=len(facts),
      recommended=context.recommended_strategy,
    )
    return context

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  @staticmethod
  def _notes(
    episodes: List[Episode],
    facts: Dict[str, Any],
    stats: List[StrategyStats],
  ) -> List[str]:
    '''Render compact recall notes for the planner prompt.

    Args:
      episodes: Recalled episodes.
      facts: Relevant facts.
      stats: Strategy stats.

    Returns:
      Note lines.
    '''
    notes: List[str] = []
    for episode in episodes[:3]:
      notes.append(
        f'past run [{episode.strategy}] {episode.status}: '
        f'{episode.task[:80]}'
      )
    for key, value in list(facts.items())[:5]:
      notes.append(f'fact {key}: {str(value)[:80]}')
    for stat in stats:
      if stat.runs:
        notes.append(
          f'strategy {stat.strategy}: {stat.success_rate:.0%} success '
          f'over {stat.runs} run(s)'
        )
    if not notes:
      notes.append('no relevant memory yet')
    return notes

  @staticmethod
  def _recommend(
    percept: Percept,
    episodes: List[Episode],
    stats: List[StrategyStats],
  ) -> Optional[StrategyName]:
    '''Recommend a strategy from memory and current signals.

    Risk always wins: state-changing tasks get the conservative profile.
    Otherwise, if remembered episodes for this domain show a strategy with
    a clearly better success rate, prefer it; high ambiguity prefers
    exploratory.

    Args:
      percept: Current percept.
      episodes: Recalled episodes.
      stats: Strategy stats.

    Returns:
      Recommended strategy name, or None to let the selector decide.
    '''
    if percept.signals.write_intent or percept.signals.risk >= 0.5:
      return 'conservative'

    domain_episodes = [
      episode for episode in episodes
      if percept.signals.domain == 'unknown'
      or episode.domain == percept.signals.domain
    ]
    if len(domain_episodes) >= 2:
      by_strategy: Dict[str, List[bool]] = {}
      for episode in domain_episodes:
        by_strategy.setdefault(episode.strategy, []).append(episode.success)
      ranked = sorted(
        by_strategy.items(),
        key=lambda item: (sum(item[1]) / len(item[1]), len(item[1])),
        reverse=True,
      )
      best, results = ranked[0]
      if sum(results) / len(results) >= 0.75 and len(results) >= 2:
        return best  # type: ignore[return-value]

    if percept.signals.ambiguity >= 0.5:
      return 'exploratory'

    by_name = {stat.strategy: stat for stat in stats}
    conservative = by_name.get('conservative')
    exploratory = by_name.get('exploratory')
    fallback = by_name.get('fallback')
    struggling = (
      conservative is not None and conservative.runs >= 2
      and conservative.success_rate < 0.5
      and exploratory is not None and exploratory.runs >= 2
      and exploratory.success_rate < 0.5
    )
    if struggling and fallback is not None and fallback.runs >= 1:
      best_primary = max(
        conservative.success_rate, exploratory.success_rate,
      )
      if fallback.success_rate > best_primary:
        return 'fallback'
    return None
