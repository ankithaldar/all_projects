#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tier-1 weak supervision: rule-based report mining.

Precision-first lexicon matching over the 12 findings in the
competition's ~10 languages (BLUEPRINT section 3, Tier 1). Per finding,
a report yields one of three seed states:

- affirmative mention outside negation/hedge scope -> positive seed,
- mentions exist but every one is negated        -> negative seed,
- no mention                                     -> unknown (masked out).

Negation cues are searched in a character window before each match;
hedged mentions ('possible', 'cannot exclude', ...) are ignored rather
than trusted. Expected operating point on gold subset: precision >=0.9,
recall 40-60% -- measured upstream by ``build_weak_labels.py``.
"""

from __future__ import annotations

import re

import numpy as np

_NEGATION_CUES = (
  ' no ', 'without', 'intact', 'unremarkable', 'negative for', 'free of',
  'ruled out', 'excluded', 'normal', 'sin ', 'sem ', 'ohne', 'kein',
  'sans ', 'senza', 'geen ', 'niet ', 'bez ', 'nie ', 'нет', 'без',
  'не ', 'yok', 'なし', '無 ', '未见', '无 ', '未见',
)
_HEDGE_CUES = (
  'possible', 'possibly', 'questionable', '?', 'cannot exclude',
  'can not exclude', 'cannot be excluded', 'probable', 'suspicious',
  'likely', 'suggests', 'mogelijk', 'eventual', 'possível', 'posible',
  'möglich', 'possibile', 'возможно', 'возможен', 'не исключено',
  'вероятно', 'olabilir', '疑い', '可能',
)

_LEXICONS: dict[str, tuple[str, ...]] = {
  'ACL': (
    r'\bacl\b', 'anterior cruciate', r'\blca\b', 'cruzado anterior',
    'croisé antérieur', 'kreuzband', 'vorderes querband',
    r'\bпкс\b', 'öçb', '前交叉', '前十字',
  ),
  'MCL': (
    r'\bmcl\b', 'medial collateral', 'colateral medial', 'collatéral médial',
    'innenband', 'seitenband', r'\bмкс\b', '内側側副', '内侧副',
  ),
  'Medial Meniscus': (
    'medial meniscus', 'menisco medial', 'ménisque médial',
    'medialer meniskus', 'innenmeniskus', r'медиальн\w* мениск',
    '内側半月板', '内侧半月板',
  ),
  'Lateral Meniscus': (
    'lateral meniscus', 'menisco lateral', 'menisco laterale',
    'ménisque latéral', 'außenmeniskus', r'латеральн\w* мениск',
    '外側半月板', '外侧半月板',
  ),
  'Medial OA': (
    'medial compartment', r'osteoartrit\w* medial', 'gonartrose medial',
    'mediale arthrose', r'médi\w+ arthrose', r'артроз\w* медиальн\w+',
  ),
  'Lateral OA': (
    'lateral compartment', r'osteoartrit\w* lateral', 'gonartrose lateral',
    'laterale arthrose', r'латеральн\w+ артроз',
  ),
  'PF OA': (
    'patellofemoral', 'patello-femoral', 'femoropatellar', 'retropatellar',
    'patelofemoral', 'fémoro-patellaire', 'пателлофеморальн', '膝関節症',
  ),
  'Effusion': (
    'effusion', 'joint fluid', 'derrame', 'gelenkerguss', 'épanchement',
    'versoaming', 'выпот', 'siviti', '関節液貯留', '関節水腫', '积液',
  ),
  'Synovitis': (
    'synovitis', 'synovial thickening', 'sinovitis', 'synovite',
    'синовит', '滑膜炎',
  ),
  "Baker's": (
    r'\bbaker', 'popliteal cyst', 'quiste poplíteo', 'baker-zyste',
    'kyste de baker', r'кист\w+ бейкера', 'ベーカー嚢腫', '膝窩嚢腫',
    '腘窝囊肿', "baker's",
  ),
  'Contusion': (
    'contusion', 'bone bruise', 'bone bruising', 'contusión', 'kontusion',
    'contusion osseuse', 'ушиб', '打撲', '骨挫伤', '挫伤',
  ),
  'Fracture': (
    'fracture', 'fractura', 'fraktur', 'перелом', 'kırık', '骨折',
  ),
}

_WINDOW_BEFORE = 40
_WINDOW_AFTER = 12


class RuleBasedLabeler:
  """Deterministic regex labeler producing per-finding seed states."""

  _compiled = {
    finding: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for finding, patterns in _LEXICONS.items()
  }

  def _is_negated(self, text: str, start: int, end: int) -> bool:
    """Check negation cues around a match position.

    Args:
        text: Lowercased full report text.
        start: Match start offset.
        end: Match end offset.

    Returns:
        True when a negation cue falls inside the scope window.
    """
    window = text[max(0, start - _WINDOW_BEFORE):end + _WINDOW_AFTER]
    return any(cue in window for cue in _NEGATION_CUES)

  def _is_hedged(self, text: str, start: int, end: int) -> bool:
    """Check uncertainty cues around a match position.

    Args:
        text: Lowercased full report text.
        start: Match start offset.
        end: Match end offset.

    Returns:
        True when a hedge cue falls inside the scope window.
    """
    window = text[max(0, start - _WINDOW_BEFORE):end + _WINDOW_AFTER]
    return any(cue in window for cue in _HEDGE_CUES)

  def seed_labels(self, report: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract seed probabilities and validity masks for one report.

    Args:
        report: Raw report text in any supported language.

    Returns:
        Tuple ``(probs, mask)`` of float32 arrays shaped ``(12,)``:
        probs are 1.0 / 0.0 / 0.5 and mask is 1 only where a seed or
        negative evidence exists (0.5/0 elsewhere).
    """
    text = f' {str(report).lower()} '
    probs = np.full(len(self._compiled), 0.5, dtype=np.float32)
    masks = np.zeros(len(self._compiled), dtype=np.float32)
    for index, finding in enumerate(_LEXICONS):
      affirmative = False
      mentioned = False
      for pattern in self._compiled[finding]:
        for match in pattern.finditer(text):
          if self._is_hedged(text, match.start(), match.end()):
            continue  # hedged mentions are ignored, never evidence
          mentioned = True
          if self._is_negated(text, match.start(), match.end()):
            continue
          affirmative = True
          break
        if affirmative:
          break
      if affirmative:
        probs[index] = 1.0
        masks[index] = 1.0
      elif mentioned:
        probs[index] = 0.0
        masks[index] = 1.0
    return probs, masks
