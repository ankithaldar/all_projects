#!/usr/bin/env python
# -- coding: utf-8 --

'''LangChain chat-model adapter backed by the llm_gateway.'''


from __future__ import annotations

import json
import uuid
from typing import Any, List, Optional

from langchain_core.callbacks import (
  AsyncCallbackManagerForLLMRun,
  CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from job_hunter.llm.client import GatewayClient


def _lc_to_gateway(messages: List[BaseMessage]) -> List[dict]:
  '''Convert LangChain messages to gateway ChatMessage dicts.

  Args:
    messages: LangChain message list.

  Returns:
    OpenAI-style message dicts.
  '''
  out: List[dict] = []
  for message in messages:
    entry: dict = {'role': message.type, 'content': message.content or ''}
    if message.type == 'ai' and getattr(message, 'tool_calls', None):
      entry['tool_calls'] = [
        {
          'id': call.get('id') or str(uuid.uuid4()),
          'type': 'function',
          'function': {
            'name': call['name'],
            'arguments': json.dumps(call.get('args', {})),
          },
        }
        for call in message.tool_calls
      ]
    out.append(entry)
  return out


class GatewayChatModel(BaseChatModel):
  '''Minimal LangChain chat model routing every call through the gateway.'''

  client: Any
  session_id: str = 'langchain'
  temperature: float = 0.2

  @property
  def _llm_type(self) -> str:
    '''Return model family label.

    Returns:
      Static identifier.
    '''
    return 'llm_gateway'

  def bind_session(self, session_id: str) -> 'GatewayChatModel':
    '''Return a copy bound to a run-specific session id.

    Args:
      session_id: Correlation id.

    Returns:
      Configured clone.
    '''
    return self.model_copy(update={'session_id': session_id})

  def _generate(
    self,
    messages: List[BaseMessage],
    stop: Optional[List[str]] = None,
    run_manager: Optional[CallbackManagerForLLMRun] = None,
    **kwargs: Any,
  ) -> ChatResult:
    '''Synchronous generation.

    Args:
      messages: Conversation so far.
      stop: Unused stop sequences.
      run_manager: Callback manager.
      **kwargs: Forwarded options.

    Returns:
      Chat result with one AIMessage.

    Raises:
      RuntimeError: When the gateway fails all providers.
    '''
    import asyncio
    coroutine = self.client.acomplete_text(
      session_id=self.session_id,
      messages=_lc_to_gateway(messages),
      temperature=self.temperature,
    )
    try:
      running = asyncio.get_running_loop()
    except RuntimeError:
      running = None
    response = running.run_until_complete(coroutine) if running else asyncio.run(coroutine)
    message = AIMessage(content=response.content or '')
    return ChatResult(generations=[ChatGeneration(message=message)])

  async def _agenerate(
    self,
    messages: List[BaseMessage],
    stop: Optional[List[str]] = None,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
  ) -> ChatResult:
    '''Asynchronous generation.

    Args:
      messages: Conversation so far.
      stop: Unused stop sequences.
      run_manager: Callback manager.
      **kwargs: Forwarded options.

    Returns:
      Chat result with one AIMessage.
    '''
    response = await self.client.acomplete_text(
      session_id=self.session_id,
      messages=_lc_to_gateway(messages),
      temperature=self.temperature,
    )
    message = AIMessage(content=response.content or '')
    return ChatResult(generations=[ChatGeneration(message=message)])
