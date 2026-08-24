#!/usr/bin/env python
# -- coding: utf-8 --

'''Structured JSON extraction over the gateway with a repair loop.'''


from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError
from job_hunter.core.errors import StructuredOutputError
from job_hunter.llm.client import GatewayClient

T = TypeVar('T', bound=BaseModel)

_FENCE_RE = re.compile(r'^```(?:json)?\s*|\s*```$', re.MULTILINE)
_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def extract_json(text: str) -> Dict[str, Any]:
  '''Extract the first JSON object from model text.

  Args:
    text: Raw model output.

  Returns:
    Decoded mapping.

  Raises:
    StructuredOutputError: When no JSON object can be decoded.
  '''
  cleaned = _FENCE_RE.sub('', text or '').strip()
  candidates = [cleaned]
  match = _JSON_RE.search(cleaned)
  if match:
    candidates.insert(0, match.group(0))
  for candidate in candidates:
    try:
      value = json.loads(candidate)
      if isinstance(value, dict):
        return value
    except json.JSONDecodeError:
      continue
  raise StructuredOutputError('no JSON object found in model output')


async def complete_structured(
  client: GatewayClient,
  schema: Type[T],
  instruction: str,
  payload: str,
  session_id: str,
  max_repairs: int = 2,
  temperature: float = 0.0,
) -> T:
  '''Request schema-valid output with validator-feedback repairs.

  Args:
    client: Gateway client.
    schema: Pydantic model class describing the desired object.
    instruction: Task instruction including the JSON contract.
    payload: Data to analyze (e.g., job description).
    session_id: Correlation id for cost logs.
    max_repairs: Extra attempts after the first failure.
    temperature: Sampling temperature (low by default).

  Returns:
    Validated model instance.

  Raises:
    StructuredOutputError: After exhausting repair attempts.
  '''
  contract = json.dumps(schema.model_json_schema(), ensure_ascii=False)
  base_prompt = (
    f'{instruction}\n\n'
    'Return ONLY one JSON object conforming to this JSON schema. '
    'No prose, no markdown fences.\n\n'
    f'Schema:\n{contract}\n\nInput:\n{payload[:12000]}'
  )
  messages = [{'role': 'user', 'content': base_prompt}]
  last_error = ''
  for attempt in range(max_repairs + 1):
    if attempt > 0:
      messages.append({
        'role': 'user',
        'content': (
          f'Your previous answer failed validation: {last_error}. '
          'Return corrected JSON only, matching the schema exactly.'
        ),
      })
    response = await client.acomplete_text(
      session_id=session_id,
      messages=messages,
      temperature=temperature,
    )
    content = response.content or ''
    messages.append({'role': 'assistant', 'content': content})
    try:
      return schema.model_validate(extract_json(content))
    except (StructuredOutputError, ValidationError) as exc:
      last_error = str(exc)[:800]
  raise StructuredOutputError(f'schema validation failed after repairs: {last_error}')
