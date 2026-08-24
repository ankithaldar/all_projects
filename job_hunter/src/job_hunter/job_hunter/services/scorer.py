#!/usr/bin/env python
# -- coding: utf-8 --

'''Weighted aggregation, ranking, and bounded feedback reweighting.'''


from __future__ import annotations

from typing import Any, Dict, List, Tuple

DEFAULT_WEIGHTS: Dict[str, float] = {
  'skills_must': 0.30,
  'skills_nice': 0.10,
  'semantic': 0.25,
  'seniority': 0.10,
  'title_fit': 0.05,
  'salary_fit': 0.10,
  'recency': 0.05,
  'company_fit': 0.05,
}

_MAX_ADJUST = 0.05


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
  '''Fill missing components and renormalize to sum to 1.

  Args:
    weights: Raw weights mapping.

  Returns:
    Complete normalized mapping.
  '''
  merged = {**DEFAULT_WEIGHTS}
  for key, value in (weights or {}).items():
    if key in merged and isinstance(value, (int, float)) and value >= 0:
      merged[key] = float(value)
  total = sum(merged.values()) or 1.0
  return {key: value / total for key, value in merged.items()}


def feedback_adjust(
  weights: Dict[str, float],
  dismiss_ratio: float,
) -> Dict[str, float]:
  '''Shift a sliver of weight toward explicit skills on dismissal signal.

  Args:
    weights: Current weights.
    dismiss_ratio: Fraction of high-score recommendations dismissed (0..1).

  Returns:
    Adjusted weights, bounded to ±_MAX_ADJUST per component.
  '''
  adjusted = dict(weights)
  ratio = max(0.0, min(1.0, dismiss_ratio))
  delta = _MAX_ADJUST * ratio
  adjusted['skills_must'] = adjusted.get('skills_must', DEFAULT_WEIGHTS['skills_must']) + delta
  adjusted['semantic'] = adjusted.get('semantic', DEFAULT_WEIGHTS['semantic']) - delta * 0.6
  adjusted['title_fit'] = adjusted.get('title_fit', DEFAULT_WEIGHTS['title_fit']) - delta * 0.4
  for key in ('semantic', 'title_fit'):
    if adjusted[key] < 0.01:
      adjusted[key] = 0.01
  return adjusted


def aggregate(
  components: Dict[str, float],
  weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
  '''Weighted total with persisted breakdown.

  Args:
    components: Component scores in [0,1].
    weights: Normalized weights.

  Returns:
    (total 0..100, breakdown mapping).
  '''
  breakdown: Dict[str, float] = {}
  total = 0.0
  for key, weight in weights.items():
    value = float(components.get(key, 0.5))
    breakdown[key] = round(value, 4)
    total += weight * value
  return round(100.0 * total, 1), breakdown


def rationale(components: Dict[str, float]) -> str:
  '''Build the top-strengths rationale sentence.

  Args:
    components: Component scores.

  Returns:
    Human-readable one-liner.
  '''
  labels = {
    'skills_must': 'core skill coverage',
    'skills_nice': 'bonus skills',
    'semantic': 'overall relevance',
    'seniority': 'seniority fit',
    'title_fit': 'role match',
    'salary_fit': 'compensation fit',
    'recency': 'fresh posting',
    'company_fit': 'vertical alignment',
  }
  top = sorted(components.items(), key=lambda kv: kv[1], reverse=True)[:3]
  parts = [labels.get(key, key) for key, value in top if value >= 0.5]
  return 'Strong ' + ', '.join(parts) + '.' if parts else 'Mixed match profile.'


