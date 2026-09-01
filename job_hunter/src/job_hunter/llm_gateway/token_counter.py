#!/usr/bin/env python
# -- coding: utf-8 --

'''Token counting utility based on tiktoken with safe fallback.'''

from __future__ import annotations

import json
from typing import Any, Dict, List

from llm_gateway.schemas import ChatMessage
from tiktoken import encoding_for_model, get_encoding


class TokenCounter:
  '''Counts tokens for prompts and completions.'''

  def __init__(self) -> None:
    '''Initialize encoding cache.'''
    self._encodings: Dict[str, Any] = {}

  def _encoding(self, model: str) -> Any:
    '''Get or create a tiktoken encoding.

    Args:
      model: Model name used to select the encoding.

    Returns:
      A tiktoken encoding object.
    '''
    if model in self._encodings:
      return self._encodings[model]

    try:
      encoding = encoding_for_model(model)
    except Exception:
      encoding = get_encoding('cl100k_base')

    self._encodings[model] = encoding
    return encoding

  def count_text(self, text: str, model: str = '') -> int:
    '''Count tokens in a text blob.

    Args:
      text: Input text.
      model: Optional model name used for encoding selection.

    Returns:
      Approximate token count.
    '''
    if not text:
      return 0

    try:
      return len(self._encoding(model).encode(text))
    except Exception:
      return max(1, len(text) // 4)

  def count_messages(
    self,
    messages: List[ChatMessage],
    model: str = '',
  ) -> int:
    '''Count tokens for a list of chat messages.

    Args:
      messages: Normalized chat messages.
      model: Optional model name used for encoding selection.

    Returns:
      Approximate token count.
    '''
    tokens = 0

    for message in messages:
      tokens += 3
      tokens += self.count_text(message.role, model)

      if message.content:
        tokens += self.count_text(message.content, model)

      if message.name:
        tokens += self.count_text(message.name, model)

      if message.tool_calls:
        serialized = json.dumps(
          [call.model_dump() for call in message.tool_calls],
          default=str,
        )
        tokens += self.count_text(serialized, model)

    return tokens
