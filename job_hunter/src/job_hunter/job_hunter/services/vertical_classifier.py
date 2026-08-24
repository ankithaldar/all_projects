#!/usr/bin/env python
# -- coding: utf-8 --

'''Industry vertical classification: rules first, gateway fallback.'''


from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from job_hunter.llm.client import GatewayClient
from pydantic import BaseModel, Field


class VerticalGuess(BaseModel):
  '''LLM classification result schema.'''

  vertical: str = 'unknown'
  confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class VerticalClassifier:
  '''Classify companies into the configured vertical taxonomy.'''

  def __init__(self, taxonomy_path: str | Path) -> None:
    '''Load the taxonomy file.

    Args:
      taxonomy_path: Path to verticals.yaml.
    '''
    payload = yaml.safe_load(Path(taxonomy_path).read_text(encoding='utf-8')) or {}
    self._verticals: Dict[str, Dict[str, Any]] = payload.get('verticals') or {}
    self._conf: Dict[str, float] = payload.get('default_confidence', {})
    self._keywords: Dict[str, list] = {
      name: [k.lower() for k in spec.get('keywords', [])]
      for name, spec in self._verticals.items()
    }

  @property
  def known_verticals(self) -> list:
    '''Return configured vertical names.

    Returns:
      List of names.
    '''
    return sorted(self._keywords.keys())

  def classify_rules(self, text: str) -> Tuple[Optional[str], float]:
    '''Keyword-rule classification over free text.

    Args:
      text: Company blurb/careers text.

    Returns:
      (vertical, confidence) or (None, 0.0).
    '''
    lowered = f' {text.lower()} '
    best_name: Optional[str] = None
    best_hits = 0
    for name, keywords in self._keywords.items():
      hits = sum(1 for keyword in keywords if f' {keyword} ' in lowered or f'{keyword}' in lowered)
      if hits > best_hits:
        best_hits = hits
        best_name = name
    if best_name is None or best_hits == 0:
      return None, 0.0
    confidence = float(self._conf.get('keyword_hit', 0.7))
    return best_name, confidence

  async def classify(
    self,
    text: str,
    client: GatewayClient,
    session_id: str,
  ) -> Tuple[str, float]:
    '''Classify with rules, escalating to the LLM when rules fail.

    Args:
      text: Company description text.
      client: Gateway client.
      session_id: Correlation id.

    Returns:
      (vertical, confidence); 'unknown' at 0.0 when everything fails.
    '''
    vertical, confidence = self.classify_rules(text)
    if vertical is not None:
      return vertical, confidence
    from job_hunter.llm.structured import complete_structured
    allowed = ', '.join(self.known_verticals + ['unknown'])
    instruction = (
      f'Classify this company into exactly one vertical from: {allowed}. '
      'Respond per the schema.'
    )
    try:
      guess = await complete_structured(
        client, VerticalGuess, instruction, text[:4000], session_id,
      )
      if guess.vertical in allowed.split(', ') and guess.vertical != '':
        return guess.vertical, float(guess.confidence)
    except Exception:
      pass
    return 'unknown', 0.0
