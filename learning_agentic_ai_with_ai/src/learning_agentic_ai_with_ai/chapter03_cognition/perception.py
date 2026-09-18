#!/usr/bin/env python
# -- coding: utf-8 --

'''Layer 1 - Perception: turn raw text into typed signals.

Perception is the cheapest layer and the most underrated one. Before any
LLM call, we can already answer questions that drive strategy selection:

  - Which business domain is this? (retail / telecom / unknown)
  - Does it ask for a state change? (write intent, risk)
  - How much does it depend on live data? (data_need)
  - How many objectives does it bundle? (multi_step)
  - How much diagnosis is implied? (ambiguity)
  - Which identifiers and capabilities does it reference?

The implementation is deliberately rule-based and deterministic: identical
tasks always yield identical percepts, which makes routing testable and
cheap. An LLM classifier can be added later for `unknown` domains, but the
majority of operational traffic is keyword-shaped enough that this layer
pays for itself without a model call.
'''


from __future__ import annotations

import re
from typing import Dict, List, Tuple

from agentic_common.logging import get_logger, log_event
from agentic_common.security import sanitize_untrusted
from chapter03_cognition.schemas import Percept, PerceptionSignals

logger = get_logger(__name__)

_RETAIL_HINTS = (
  'retail', 'store', 'inventory', 'stock', 'sku', 'sales', 'reorder',
  'restock', 'replenish', 'product', 'shelf', 'warehouse', 'units sold',
)
_TELECOM_HINTS = (
  'telecom', 'cell site', 'site', 'network', 'latency', 'packet loss',
  'degraded', 'outage', 'technician', 'dispatch', 'battery', 'coverage',
  'base station',
)
_RISK_HINTS = (
  'restock', 'place an order', 'create order', 'dispatch', 'cancel',
  'refund', 'delete', 'update', 'approve', 'escalate', 'immediately',
  'urgent', 'override', 'force',
)
_AMBIGUITY_HINTS = (
  'figure out', 'investigate', 'diagnose', 'root cause', 'why ',
  'unclear', 'might', 'possibly', 'recommend', 'decide', 'best ',
  'optimize', 'compare', 'trade-off',
)
_MULTI_STEP_HINTS = (
  ' and then ', ' then ', ' after that', ' finally', ' first ',
  ' next ', ' followed by', ' as well as', ' audit', ' campaign',
  ' across all', ' every store', ' all stores', ' all sites', ' plan',
  ' multiple', ' end-to-end',
)
_DATA_HINTS = (
  'check', 'find', 'look up', 'lookup', 'report', 'status', 'inventory',
  'stock', 'sales', 'trend', 'site', 'network', 'degraded', 'current',
  'today', 'right now', 'latest',
)

_ENTITY_PATTERNS: Dict[str, str] = {
  'store_ids': r'\bS\d{2}\b',
  'skus': r'\bR-\d{2,}\b',
  'site_ids': r'\bCS-\d{2,}\b',
  'technician_ids': r'\bT-\d{2}\b',
  'quantities': r'(?<![\w.-])(\d{1,6})(?![\w])',
}

_CAPABILITY_RULES: Tuple[Tuple[str, str, str], ...] = (
  ('inventory_read', 'retail', 'low stock'),
  ('inventory_read', 'retail', 'stock'),
  ('inventory_read', 'retail', 'inventory'),
  ('inventory_read', 'retail', 'reorder'),
  ('sales_read', 'retail', 'sales'),
  ('sales_read', 'retail', 'trend'),
  ('restock_write', 'retail', 'restock'),
  ('restock_write', 'retail', 'replenish'),
  ('restock_write', 'retail', 'place an order'),
  ('site_read', 'telecom', 'site status'),
  ('site_read', 'telecom', 'degraded'),
  ('site_read', 'telecom', 'network'),
  ('site_read', 'telecom', 'latency'),
  ('dispatch_write', 'telecom', 'dispatch'),
  ('dispatch_write', 'telecom', 'technician'),
)

_SCORE_CAPS = {
  'risk': 6.0,
  'ambiguity': 4.0,
  'multi_step': 8.0,
  'data_need': 8.0,
}


def _match(text: str, phrases: Tuple[str, ...]) -> List[str]:
  '''List phrases present in text.

  Args:
    text: Lowercased text.
    phrases: Candidate phrases.

  Returns:
    Matched phrases in input order.
  '''
  return [phrase for phrase in phrases if phrase in text]


def _normalized_score(hits: int, cap: float) -> float:
  '''Scale a hit count into 0..1.

  Args:
    hits: Number of matched phrases.
    cap: Hit count considered maximal.

  Returns:
    Clamped score.
  '''
  return max(0.0, min(1.0, hits / cap))


class PerceptionLayer:
  '''Deterministic task -> Percept transformation.'''

  def perceive(self, task: str, session_id: str) -> Percept:
    '''Build a percept for one task.

    Args:
      task: Raw task text (may contain untrusted fragments).
      session_id: Owning session id.

    Returns:
      A validated Percept with signals, entities, and capabilities.
    '''
    normalized = re.sub(
      r'\s+', ' ', sanitize_untrusted(task or '', max_chars=4000),
    ).strip()
    text = f' {normalized.lower()} '

    retail_hits = _match(text, _RETAIL_HINTS)
    telecom_hits = _match(text, _TELECOM_HINTS)
    domain = self._domain(retail_hits, telecom_hits)

    risk_hits = _match(text, _RISK_HINTS)
    ambiguity_hits = _match(text, _AMBIGUITY_HINTS)
    multi_hits = _match(text, _MULTI_STEP_HINTS)
    data_hits = _match(text, _DATA_HINTS)

    if ' and ' in text:
      multi_hits = multi_hits + [' and ']
    if text.count('?') > 1:
      multi_hits = multi_hits + ['?']

    risk = _normalized_score(len(risk_hits), _SCORE_CAPS['risk'])
    ambiguity = _normalized_score(
      len(ambiguity_hits), _SCORE_CAPS['ambiguity'],
    )
    multi_step = _normalized_score(
      len(multi_hits), _SCORE_CAPS['multi_step'],
    )
    data_need = _normalized_score(
      len(data_hits), _SCORE_CAPS['data_need'],
    )
    complexity = max(0.0, min(1.0, (
      0.35 * risk
      + 0.25 * multi_step
      + 0.20 * data_need
      + 0.20 * ambiguity
    )))

    entities = self._entities(normalized)
    capabilities = self._capabilities(text, domain)
    keywords = list(dict.fromkeys(
      risk_hits + ambiguity_hits + multi_hits + data_hits,
    ))[:12]

    signals = PerceptionSignals(
      domain=domain,
      risk=round(risk, 3),
      ambiguity=round(ambiguity, 3),
      multi_step=round(multi_step, 3),
      data_need=round(data_need, 3),
      complexity=round(complexity, 3),
      write_intent=bool(risk_hits),
      keywords=keywords,
    )
    percept = Percept(
      task=normalized,
      session_id=session_id,
      signals=signals,
      entities=entities,
      required_capabilities=capabilities,
    )
    log_event(
      logger, 20, 'percept_ready',
      session_id=session_id,
      domain=domain,
      risk=signals.risk,
      data_need=signals.data_need,
      multi_step=signals.multi_step,
      capabilities=capabilities,
    )
    return percept

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  @staticmethod
  def _domain(
    retail_hits: List[str],
    telecom_hits: List[str],
  ) -> str:
    '''Choose a domain from hint matches.

    Args:
      retail_hits: Retail hint matches.
      telecom_hits: Telecom hint matches.

    Returns:
      'retail', 'telecom', or 'unknown'.
    '''
    if len(retail_hits) > len(telecom_hits):
      return 'retail'
    if len(telecom_hits) > len(retail_hits):
      return 'telecom'
    return 'unknown'

  @staticmethod
  def _entities(text: str) -> Dict[str, List[str]]:
    '''Extract recognized identifiers from the task.

    Args:
      text: Normalized task text.

    Returns:
      Mapping of entity kind to unique values in appearance order.
    '''
    entities: Dict[str, List[str]] = {}
    for kind, pattern in _ENTITY_PATTERNS.items():
      values = re.findall(pattern, text.upper())
      unique = list(dict.fromkeys(values))
      if unique:
        entities[kind] = unique[:20]
    return entities

  @staticmethod
  def _capabilities(text: str, domain: str) -> List[str]:
    '''Map domain + phrases to required capability names.

    Args:
      text: Lowercased, space-padded task text.
      domain: Detected domain.

    Returns:
      Ordered unique capability names.
    '''
    capabilities: List[str] = []
    for capability, rule_domain, phrase in _CAPABILITY_RULES:
      if domain not in (rule_domain, 'unknown'):
        continue
      if phrase in text and capability not in capabilities:
        capabilities.append(capability)
    return capabilities
